#!/usr/bin/env python3
"""Stream a long recording through the lean engine with the rolling ("infinite") policy on.

  python longform_bench.py data/longform-test-clean-10m-s1.wav --graph [--quant int4hqq]

Same stream the upstream vLLM server scored 3.68 % WER on (P2).  Reports WER vs the .txt reference,
per-step latency percentiles INCLUDING trims/re-bases (synced every step, like a server that
needs each token), trim/re-base counts, and emitted text tokens per audio minute (drift check).
"""
from __future__ import annotations

import argparse
import json
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
from audio8_asr_infinite.simulated_streaming_audio import decode_visible_text
from engine import LOOK_AHEAD, LOOK_BACK, PREFILL_TOKENS, SAMPLES_PER_TOKEN, LeanAudio8
from engine_bench import DELAY, LEFT_PAD, RIGHT_PAD_TEXT, n_windows

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--quant", default="none", choices=["none", "int4hqq"])
    ap.add_argument("--no-rolling", action="store_true")
    ap.add_argument("--dec-window", type=int, default=360)
    ap.add_argument("--trim-mode", default="rotate", choices=["rotate", "reencode"])
    ap.add_argument("--mem-gib", type=float, default=0.0)
    ap.add_argument("--tag", default="longform")
    a = ap.parse_args()
    a.checkpoint = resolve_checkpoint(a.checkpoint)

    if a.mem_gib:

        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    dev, dt = torch.device("cuda"), torch.bfloat16
    cfg_m = Audio8ASRInfiniteConfig.from_pretrained(a.checkpoint)
    cfg_m.semantic_vad_horizons_seconds = None
    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained(a.checkpoint, trust_remote_code=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        a.checkpoint, config=cfg_m, trust_remote_code=True, torch_dtype=dt).to(dev).eval()
    if a.quant == "int4hqq":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d",
                                              int4_choose_qparams_algorithm="hqq"),
                  filter_fn=lambda m, f: isinstance(m, torch.nn.Linear)
                  and not any(k in f for k in ("lm_head", "multi_modal_projector", "ada_rms_norm")))
        torch.cuda.empty_cache()
    special = resolve_qwen_streaming_special_token_ids(tok)
    lang = resolve_qwen_language_token_id(tok, "en")
    cfg = StreamingDecodeConfig(transcription_delay_ms=480, left_pad_tokens=LEFT_PAD)

    eng = LeanAudio8(model, fe, batch=1, delay_tokens=DELAY, rolling=not a.no_rolling,
                     dec_window=a.dec_window, dec_slots=(512 if not a.no_rolling else 16384),
                     trim_mode=a.trim_mode)
    if a.graph:
        eng.capture()
    eng.reset()

    wav = np.clip(sf.read(a.wav, dtype="float32")[0], -1, 1)
    W = n_windows(wav, cfg)
    right = (DELAY + 1 + RIGHT_PAD_TEXT) * SAMPLES_PER_TOKEN
    stream = torch.zeros(1, LEFT_PAD * SAMPLES_PER_TOKEN + len(wav) + right + SAMPLES_PER_TOKEN, device=dev)
    stream[0, LEFT_PAD * SAMPLES_PER_TOKEN: LEFT_PAD * SAMPLES_PER_TOKEN + len(wav)] = torch.from_numpy(wav)
    prompt = torch.tensor([[special["bos_token_id"], lang] + [special["streaming_pad_token_id"]] * (PREFILL_TOKENS - 2)],
                          device=dev)

    ids = [int(eng.prefill(stream[:, : PREFILL_TOKENS * SAMPLES_PER_TOKEN + LOOK_AHEAD], prompt)[0])]
    times, events = [], []
    for k in range(1, W):
        tr, rb = eng.trims, eng.rebases
        start = (PREFILL_TOKENS + k - 1) * SAMPLES_PER_TOKEN
        t = time.perf_counter()
        eng.in_audio.copy_(stream[:, start - LOOK_BACK: start + SAMPLES_PER_TOKEN + LOOK_AHEAD])
        eng.in_tok.copy_(eng.out_tok)
        ids.append(int(eng.step()[0]))  # .item()-style read = a server's per-step sync
        times.append(1000 * (time.perf_counter() - t))
        if eng.trims != tr or eng.rebases != rb:
            events.append(k)

    text = decode_visible_text(tokenizer=tok, generated_token_ids=ids, special_ids=special)[0]
    norm = EnglishTextNormalizer()
    ref = norm(Path(a.wav).with_suffix(".txt").read_text())
    specials = set(int(v) for v in special.values())
    per_min = {}
    for k, t_id in enumerate(ids):
        if t_id not in specials:
            m = int(k * 0.08 // 60)
            per_min[m] = per_min.get(m, 0) + 1
    ts = sorted(times)
    pct = lambda q: round(ts[min(len(ts) - 1, int(q * len(ts)))], 2)
    res = {
        "tag": a.tag, "audio": Path(a.wav).name, "audio_s": round(len(wav) / 16000, 1),
        "quant": a.quant, "graph": a.graph, "rolling": not a.no_rolling, "dec_window": a.dec_window, "trim_mode": a.trim_mode,
        "wer_pct": round(100 * jiwer.wer(ref, norm(text)), 3),
        "steps": len(times), "trims": eng.trims, "tower_rebases": eng.rebases,
        "step_ms_p50": pct(0.5), "step_ms_p95": pct(0.95), "step_ms_p99": pct(0.99), "step_ms_max": round(ts[-1], 2),
        "steps_over_80ms": sum(1 for x in ts if x > 80),
        "slowest_steps": sorted(((round(x, 1), k + 1, (k + 1) in events) for k, x in enumerate(times)), reverse=True)[:5],
        "maint_step_ms_max": round(max((times[k - 1] for k in events), default=0), 2),
        "text_tokens_per_min": per_min,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(res, indent=1))
    out = HERE / "results" / f"{a.tag}-{a.trim_mode}-{a.quant}-{'graph' if a.graph else 'eager'}-{Path(a.wav).stem}.json"
    out.write_text(json.dumps({**res, "text": text}, indent=1))


if __name__ == "__main__":
    main()
