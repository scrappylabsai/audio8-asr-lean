#!/usr/bin/env python3
"""Where does one engine step's GPU time go?  Profiles 40 eager steps (batch 1), buckets kernels.

  python probe_engine.py [--compile default]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from common import DEFAULT_CHECKPOINT, resolve_checkpoint
from torch.profiler import ProfilerActivity, profile
from transformers import AutoFeatureExtractor

from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import Audio8ASRInfiniteForConditionalGeneration
from engine import PREFILL_TOKENS, LeanAudio8

ap = argparse.ArgumentParser()
ap.add_argument("--compile", default="")
a = ap.parse_args()
ck = resolve_checkpoint(DEFAULT_CHECKPOINT)
cfg = Audio8ASRInfiniteConfig.from_pretrained(ck)
cfg.semantic_vad_horizons_seconds = None
fe = AutoFeatureExtractor.from_pretrained(ck, trust_remote_code=True)
model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
    ck, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
eng = LeanAudio8(model, fe, batch=1)
if a.compile:
    eng.compile_step(a.compile)
prompt = torch.full((1, PREFILL_TOKENS), int(model.config.pad_token_id), device="cuda")
with torch.inference_mode():
    eng.prefill(torch.randn(1, PREFILL_TOKENS * 1280 + 40, device="cuda") * 0.01, prompt)
    for _ in range(5):
        eng.in_audio.normal_(0, 0.01)
        eng.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(40):
            eng.step()
        torch.cuda.synchronize()

buckets, total, n = {}, 0.0, 0
rules = [("gemm/gemv", r"gemv|gemm|cutlass|sm\d+_xmma|cublas|Kernel2|splitK|scaled_mm|fp8"),
         ("attention", r"flash|fmha|attention|efficient|sdpa|softmax"),
         ("triton-fused", r"triton")]
for e in prof.events():
    if e.device_type != torch.autograd.DeviceType.CUDA:
        continue
    us = e.time_range.elapsed_us()
    total += us
    n += 1
    cat = next((c for c, rx in rules if re.search(rx, e.name, re.I)), "elementwise/other")
    b = buckets.setdefault(cat, [0.0, 0])
    b[0] += us
    b[1] += 1
res = {"compile": a.compile or "eager", "kernels_per_step": round(n / 40, 1),
       "gpu_ms_per_step": round(total / 40 / 1000, 3),
       "buckets": {k: {"ms_per_step": round(v[0] / 40 / 1000, 3), "kernels_per_step": round(v[1] / 40, 1)}
                   for k, v in sorted(buckets.items(), key=lambda kv: -kv[1][0])}}
print(json.dumps(res, indent=1))
