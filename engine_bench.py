#!/usr/bin/env python3
"""Bench the lean engine against the vendor decoder on the same seeded LibriSpeech sample.

  python engine_bench.py --n 300 --batch 8 --graph              # WER + ms/step
  python engine_bench.py --n 20 --batch 1 --compare-vendor 20    # token-exact check vs vendor
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from common import DEFAULT_CHECKPOINT, resolve_checkpoint
from transformers import AutoFeatureExtractor, AutoTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

from audio8_asr_infinite.examples.torch_streaming_decode import StreamingDecodeConfig
from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForConditionalGeneration,
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)
from audio8_asr_infinite.simulated_streaming_audio import decode_visible_text, iter_realtime_audio_windows
from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch
from engine import LOOK_AHEAD, LOOK_BACK, SAMPLES_PER_TOKEN, LeanAudio8

HERE = Path(__file__).resolve().parent
LEFT_PAD, DELAY, RIGHT_PAD_TEXT = 18, 6, 10
PREFILL = LEFT_PAD + DELAY + 1


def load_split(split, max_sec, corpus="LibriSpeech"):
    root = HERE / "data" / corpus / split
    items = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            dur = sf.info(str(path)).duration
            if dur <= max_sec:
                items.append({"id": uid, "path": str(path), "ref": text, "dur": dur})
    return items


def n_windows(wav, cfg):
    it, _ = iter_realtime_audio_windows(
        wav, audio_config=cfg, num_delay_tokens=DELAY, right_pad_text_tokens=RIGHT_PAD_TEXT,
        streaming_look_ahead_ms=2.5, streaming_look_back_ms=52.5, initial_prefill_tokens=PREFILL)
    return sum(1 for _ in it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--corpus", default="LibriSpeech")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-sec", type=float, default=29.0)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--compile", default="", help="torch.compile mode for the step, e.g. max-autotune-no-cudagraphs")
    ap.add_argument("--quant", default="none", choices=["none", "fp8dyn", "int8wo", "fp8wo", "int4wo", "int4hqq", "int4mlp", "mix"])
    ap.add_argument("--group", type=int, default=128, help="int4 group size")
    ap.add_argument("--compare-vendor", type=int, default=0, help="token-compare the first K clips vs vendor")
    ap.add_argument("--mem-gib", type=float, default=0.0)
    ap.add_argument("--tag", default="engine")
    a = ap.parse_args()
    a.checkpoint = resolve_checkpoint(a.checkpoint)

    if a.mem_gib:

        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    dev, dt = torch.device("cuda"), torch.bfloat16
    items = load_split(a.split, a.max_sec, a.corpus)
    pool = len(items)
    random.Random(a.seed).shuffle(items)
    items = items[: a.n] if a.n else items
    items.sort(key=lambda x: x["dur"])

    cfg_m = Audio8ASRInfiniteConfig.from_pretrained(a.checkpoint)
    cfg_m.semantic_vad_horizons_seconds = None  # upstream loader bug: VAD heads in a 2nd file
    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained(a.checkpoint, trust_remote_code=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        a.checkpoint, config=cfg_m, trust_remote_code=True, torch_dtype=dt).to(dev).eval()
    if a.quant == "mix":
        # FP8 (accurate) for the decoder MLPs + lm_head, int4 HQQ (fast at M=1) for everything else.
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, Int4WeightOnlyConfig,
                                          PerRow, quantize_)
        is_lin = lambda m: isinstance(m, torch.nn.Linear)
        dec_mlp = lambda f: "language_model.model.layers." in f and ".mlp." in f
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                  filter_fn=lambda m, f: is_lin(m) and (dec_mlp(f) or f.endswith("lm_head")))
        quantize_(model, Int4WeightOnlyConfig(group_size=a.group, int4_packing_format="tile_packed_to_4d",
                                              int4_choose_qparams_algorithm="hqq"),
                  filter_fn=lambda m, f: is_lin(m) and not dec_mlp(f) and not f.endswith("lm_head")
                  and not any(k in f for k in ("multi_modal_projector", "ada_rms_norm")))
        torch.cuda.empty_cache()
    elif a.quant != "none":
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig,
                                          Int4WeightOnlyConfig, Int8WeightOnlyConfig, PerRow, quantize_)
        qcfg = {"fp8dyn": lambda: Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                "fp8wo": Float8WeightOnlyConfig, "int8wo": Int8WeightOnlyConfig,
                "int4wo": lambda: Int4WeightOnlyConfig(group_size=a.group, int4_packing_format="tile_packed_to_4d"),
                "int4hqq": lambda: Int4WeightOnlyConfig(group_size=a.group, int4_packing_format="tile_packed_to_4d",
                                                        int4_choose_qparams_algorithm="hqq"),
                "int4mlp": lambda: Int4WeightOnlyConfig(group_size=a.group, int4_packing_format="tile_packed_to_4d",
                                                        int4_choose_qparams_algorithm="hqq")}[a.quant]()
        keep = ("lm_head", "multi_modal_projector", "ada_rms_norm")
        only = ("language_model.model.layers.", ".mlp.") if a.quant == "int4mlp" else ()  # decoder MLPs only
        quantize_(model, qcfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
                  and not any(k in fqn for k in keep) and all(o in fqn for o in only))
        torch.cuda.empty_cache()
    weights_gib = torch.cuda.memory_allocated() / 2**30
    cfg = StreamingDecodeConfig(transcription_delay_ms=480, left_pad_tokens=LEFT_PAD)
    special = resolve_qwen_streaming_special_token_ids(tok)
    lang = resolve_qwen_language_token_id(tok, "en")
    prompt_row = [special["bos_token_id"], lang] + [special["streaming_pad_token_id"]] * (PREFILL - 2)

    eng = LeanAudio8(model, fe, batch=a.batch, delay_tokens=DELAY)
    t0 = time.time()
    if a.compile:
        eng.compile_step(a.compile)
        eng.reset()
        with torch.inference_mode():
            eng._fn()  # trigger compilation + autotune outside the graph
        eng.reset()
    if a.graph:
        eng.capture()
    capture_s = time.time() - t0
    prompt = torch.tensor([prompt_row] * a.batch, device=dev)

    def run_batch(batch):
        wavs = [np.clip(sf.read(b["path"], dtype="float32")[0], -1, 1) for b in batch]
        counts = [n_windows(w, cfg) for w in wavs]
        W = max(counts)
        right = (DELAY + 1 + RIGHT_PAD_TEXT) * SAMPLES_PER_TOKEN
        L = LEFT_PAD * SAMPLES_PER_TOKEN + max(len(w) for w in wavs) + right + SAMPLES_PER_TOKEN
        streams = torch.zeros(a.batch, L, device=dev)
        for r, w in enumerate(wavs):
            streams[r, LEFT_PAD * SAMPLES_PER_TOKEN: LEFT_PAD * SAMPLES_PER_TOKEN + len(w)] = torch.from_numpy(w)
        log = torch.zeros(a.batch, W, device=dev, dtype=torch.long)
        eng.reset()
        torch.cuda.synchronize()
        tp = time.time()
        log[:, 0] = eng.prefill(streams[:, : PREFILL * SAMPLES_PER_TOKEN + LOOK_AHEAD], prompt)
        torch.cuda.synchronize()
        pre = time.time() - tp
        ts = time.time()
        for k in range(1, W):
            start = (PREFILL + k - 1) * SAMPLES_PER_TOKEN
            eng.in_audio.copy_(streams[:, start - LOOK_BACK: start + SAMPLES_PER_TOKEN + LOOK_AHEAD])
            eng.in_tok.copy_(eng.out_tok)
            log[:, k] = eng.step()
        torch.cuda.synchronize()
        steps_s = time.time() - ts
        ids = log.cpu().tolist()
        texts, gen = [], []
        for r in range(len(batch)):
            g = ids[r][: counts[r]]
            gen.append(g)
            texts.append(decode_visible_text(tokenizer=tok, generated_token_ids=g, special_ids=special)[0])
        return texts, gen, pre, steps_s, W - 1

    run_batch(items[: a.batch])  # warm-up
    torch.cuda.reset_peak_memory_stats()
    hyps, gens, pre_s, step_s, n_steps = [], [], 0.0, 0.0, 0
    for i in range(0, len(items), a.batch):
        t, g, p, s, n = run_batch(items[i: i + a.batch])
        hyps += t; gens += g; pre_s += p; step_s += s; n_steps += n

    norm = EnglishTextNormalizer()
    refs_n = [norm(x["ref"]) for x in items]
    hyps_n = [norm(h) for h in hyps]
    res = {
        "tag": a.tag, "group": a.group, "graph": a.graph, "compile": a.compile, "quant": a.quant, "corpus": a.corpus, "split": a.split,
        "n": len(items), "pool": pool, "seed": a.seed, "batch": a.batch,
        "wer_pct": round(100 * jiwer.wer(refs_n, hyps_n), 3),
        "ms_per_step": round(1000 * step_s / max(n_steps, 1), 3),
        "prefill_ms_avg": round(1000 * pre_s / max(1, -(-len(items) // a.batch)), 1),
        "audio_s": round(sum(x["dur"] for x in items), 1),
        "rtf_total": round((pre_s + step_s) / sum(x["dur"] for x in items), 4),
        "capture_s": round(capture_s, 1), "weights_gib": round(weights_gib, 2),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "torch": torch.__version__,
    }

    if a.compare_vendor:
        same, text_same, diff_examples = 0, 0, []
        for j in range(min(a.compare_vendor, len(items))):
            w = np.clip(sf.read(items[j]["path"], dtype="float32")[0], -1, 1)
            v = simulated_streaming_greedy_decode_batch(
                model=model, tokenizer=tok, feature_extractor=fe, waveforms=[w], language_token_ids=[lang],
                special_ids=special, audio_config=cfg, num_delay_tokens=[DELAY],
                right_pad_text_tokens=RIGHT_PAD_TEXT, dtype=dt, device=dev, max_new_tokens=512)[0]
            vt = v.get("generated_token_ids") or v.get("generated_ids")
            if vt is not None:
                vt = [int(x) for x in (vt.tolist() if hasattr(vt, "tolist") else vt)]
            # Vendor runs with max_new_tokens=512, i.e. keeps decoding silence past the audio end;
            # compare the overlapping prefix of token ids, and the visible text separately.
            ok = vt is not None and vt[: len(gens[j])] == gens[j]
            same += ok
            text_same += norm(v["final_text"]) == hyps_n[j]
            if not ok and len(diff_examples) < 3:
                diff_examples.append({"id": items[j]["id"], "vendor": norm(v["final_text"]), "engine": hyps_n[j]})
        res["vendor_token_prefix_match"] = f"{same}/{min(a.compare_vendor, len(items))}"
        res["vendor_text_match"] = f"{text_same}/{min(a.compare_vendor, len(items))}"
        res["vendor_diffs"] = diff_examples

    print(json.dumps(res, indent=1))
    out = HERE / "results" / f"{a.tag}-{a.quant}{a.group if 'int4' in a.quant else ''}-{'c' if a.compile else ''}{'graph' if a.graph else 'eager'}-{a.corpus}-{a.split}-n{len(items)}-b{a.batch}.json"
    out.write_text(json.dumps({**res, "utts": [{"id": x["id"], "ref": r, "hyp": h}
                                              for x, r, h in zip(items, refs_n, hyps_n)]}, indent=1))


if __name__ == "__main__":
    main()
