#!/usr/bin/env python3
"""How launch-bound is one Audio8 decode step?  Profile a batch-1 simulated-streaming decode
and compare summed GPU kernel time against wall time.  (wall - kernel) is what CUDA graphs
can win back.  Also counts kernel launches per step.

  python probe_launch.py [--batch 1]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from common import DEFAULT_CHECKPOINT, resolve_checkpoint
from torch.profiler import ProfilerActivity, profile
from transformers import AutoFeatureExtractor, AutoTokenizer

from audio8_asr_infinite.examples.torch_streaming_decode import StreamingDecodeConfig
from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForConditionalGeneration,
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)
from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--mem-gib", type=float, default=0.0)
a = ap.parse_args()

if a.mem_gib:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
ck = resolve_checkpoint(DEFAULT_CHECKPOINT)
cfg_m = Audio8ASRInfiniteConfig.from_pretrained(ck)
cfg_m.semantic_vad_horizons_seconds = None
tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
fe = AutoFeatureExtractor.from_pretrained(ck, trust_remote_code=True)
model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
    ck, config=cfg_m, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
cfg = StreamingDecodeConfig(transcription_delay_ms=480, left_pad_tokens=18)
special = resolve_qwen_streaming_special_token_ids(tok)
lang = resolve_qwen_language_token_id(tok, "en")

flacs = sorted((HERE / "data/LibriSpeech/test-clean").glob("*/*/*.flac"))
random.Random(3).shuffle(flacs)
wavs = []
for f in flacs:
    w = sf.read(f, dtype="float32")[0]
    if 9 < len(w) / 16000 < 12:
        wavs.append(w)
    if len(wavs) == a.batch:
        break

steps = {"n": 0}
model.language_model.register_forward_pre_hook(lambda *_: steps.__setitem__("n", steps["n"] + 1))


def run():
    return simulated_streaming_greedy_decode_batch(
        model=model, tokenizer=tok, feature_extractor=fe, waveforms=wavs,
        language_token_ids=[lang] * len(wavs), special_ids=special, audio_config=cfg,
        num_delay_tokens=[6] * len(wavs), right_pad_text_tokens=10, dtype=torch.bfloat16,
        device=torch.device("cuda"), max_new_tokens=512)


run()
torch.cuda.synchronize()
steps["n"] = 0
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.time()
    run()
    torch.cuda.synchronize()
    wall = time.time() - t
n = steps["n"]
kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
gpu_ms = sum(e.time_range.elapsed_us() for e in kern) / 1000.0
by_name = {}
for e in kern:
    by_name[e.name] = by_name.get(e.name, 0) + e.time_range.elapsed_us()
top = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:12]
res = {
    "batch": a.batch, "steps": n, "audio_s": round(sum(len(w) for w in wavs) / 16000 / a.batch, 2),
    "wall_ms_per_step": round(1000 * wall / n, 2),
    "gpu_kernel_ms_per_step": round(gpu_ms / n, 2),
    "gpu_busy_pct": round(100 * gpu_ms / (1000 * wall), 1),
    "kernel_launches_per_step": round(len(kern) / n, 1),
    "top_kernels_ms_per_step": [(k[:80], round(v / 1000 / n, 3)) for k, v in top],
}
print(json.dumps(res, indent=1))
(HERE / "results" / f"probe-launch-b{a.batch}.json").write_text(json.dumps(res, indent=1))
