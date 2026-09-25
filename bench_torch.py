#!/usr/bin/env python3
"""Audio8-ASR-Infinite torch-path bench: LibriSpeech WER + speed + VRAM.

Uses the vendor's simulated-streaming decoder unchanged (no rolling KV, so
utterances are capped at --max-sec; native context is 30 s).

  python bench_torch.py --split test-clean --n 300 --batch 16
  python bench_torch.py --split test-clean --n 20 --batch 1      # per-step latency
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
from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch

HERE = Path(__file__).resolve().parent


def load_split(split: str, max_sec: float, corpus: str = "LibriSpeech") -> list[dict]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--corpus", default="LibriSpeech", help="LibriSpeech | LibriSpeech-phone")
    ap.add_argument("--n", type=int, default=300, help="0 = whole split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--delay-ms", type=int, default=480)
    ap.add_argument("--max-sec", type=float, default=29.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--quant", default="none", choices=["none", "fp8dyn", "int8wo", "fp8wo"],
                    help="torchao weight quantization of tower + decoder linears (not lm_head/projector)")
    ap.add_argument("--mem-gib", type=float, default=0.0,
                    help="optional hard VRAM cap for this process (0 = none); useful on a shared GPU")
    args = ap.parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint)

    dtype = getattr(torch, args.dtype)
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    if args.mem_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    all_items = load_split(args.split, args.max_sec, args.corpus)
    items = list(all_items)
    random.Random(args.seed).shuffle(items)  # seeded sample, never first-N
    if args.n:
        items = items[: args.n]
    items.sort(key=lambda x: x["dur"])  # batch similar lengths together

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint, trust_remote_code=True)
    # Upstream bug (c8ba8ee): the VAD heads ship in a second file, transformers loads only
    # model.safetensors, and the strict loader then refuses the checkpoint. The heads do not
    # affect transcription, so load the ASR path without them.
    config = Audio8ASRInfiniteConfig.from_pretrained(args.checkpoint)
    config.semantic_vad_horizons_seconds = None
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        args.checkpoint, config=config, trust_remote_code=True, torch_dtype=dtype
    ).to(dev).eval()
    model.config.use_cache = True
    if args.quant != "none":
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                          Float8WeightOnlyConfig, Int8WeightOnlyConfig, PerRow, quantize_)
        qcfg = {"fp8dyn": lambda: Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                "fp8wo": Float8WeightOnlyConfig, "int8wo": Int8WeightOnlyConfig}[args.quant]()
        keep = ("lm_head", "multi_modal_projector")
        quantize_(model, qcfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
                  and not any(k in fqn for k in keep))
        torch.cuda.empty_cache()
    weights_gib = torch.cuda.memory_allocated() / 2**30
    load_s = time.time() - t0
    cfg = StreamingDecodeConfig(
        transcription_delay_ms=args.delay_ms,
        left_pad_tokens=int(getattr(model.config, "streaming_n_left_pad_tokens", 0) or 18),
    )
    special = resolve_qwen_streaming_special_token_ids(tok)
    lang = resolve_qwen_language_token_id(tok, "en")

    calls = {"lm": 0}
    model.language_model.register_forward_pre_hook(lambda *_: calls.__setitem__("lm", calls["lm"] + 1))

    def run(batch):
        wavs = [np.clip(sf.read(b["path"], dtype="float32")[0], -1, 1) for b in batch]
        return simulated_streaming_greedy_decode_batch(
            model=model, tokenizer=tok, feature_extractor=fe, waveforms=wavs,
            language_token_ids=[lang] * len(wavs), special_ids=special, audio_config=cfg,
            num_delay_tokens=[cfg.num_delay_tokens] * len(wavs),
            right_pad_text_tokens=cfg.right_pad_text_tokens, dtype=dtype, device=dev,
            max_new_tokens=cfg.max_new_tokens,
        )

    run(items[:1])  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    calls["lm"] = 0

    hyps, decode_s = [], 0.0
    for i in range(0, len(items), args.batch):
        batch = items[i : i + args.batch]
        torch.cuda.synchronize()
        t = time.time()
        out = run(batch)
        torch.cuda.synchronize()
        decode_s += time.time() - t
        hyps += [o.get("final_text", "") for o in out]

    norm = EnglishTextNormalizer()
    refs_n = [norm(x["ref"]) for x in items]
    hyps_n = [norm(h) for h in hyps]
    wer = jiwer.wer(refs_n, hyps_n)
    audio_s = sum(x["dur"] for x in items)
    res = {
        "tag": args.tag, "corpus": args.corpus, "split": args.split, "n": len(items), "pool": len(all_items),
        "seed": args.seed, "batch": args.batch, "delay_ms": args.delay_ms, "dtype": args.dtype,
        "wer_pct": round(100 * wer, 3), "audio_s": round(audio_s, 1),
        "decode_s": round(decode_s, 2), "rtf": round(decode_s / audio_s, 4),
        "lm_forward_calls": calls["lm"],
        "ms_per_step_batch": round(1000 * decode_s / max(calls["lm"], 1), 2),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "weights_gib": round(weights_gib, 2), "quant": args.quant,
        "load_s": round(load_s, 1), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    print(json.dumps(res, indent=1))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stem = f"{args.tag}-{args.quant}-{args.split}-n{len(items)}-b{args.batch}-d{args.delay_ms}"
    (out_dir / f"{stem}.json").write_text(json.dumps({
        **res,
        "utts": [{"id": x["id"], "ref": r, "hyp": h} for x, r, h in zip(items, refs_n, hyps_n)],
    }, indent=1))


if __name__ == "__main__":
    main()
