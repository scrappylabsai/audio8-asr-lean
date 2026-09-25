#!/usr/bin/env python3
"""Achieved weight bandwidth of the matmul kernels available for Audio8's shapes on this GPU.

Decode is weight-read bound, so GB/s of weight bytes per matmul is the number that matters.
Shapes: decoder MLP 2048->11008 / 11008->2048, attention 2048->2048, tower 1280->5120 (M=4),
lm_head 2048->151936.  M = rows (1 = one stream decode; 4 = tower frames per step; 8 = 8 streams).
"""
import json

import torch
import torch.nn.functional as F

dev = "cuda"
SHAPES = [("dec_mlp_up", 2048, 11008), ("dec_mlp_down", 11008, 2048), ("dec_attn_q", 2048, 2048),
          ("tower_mlp", 1280, 5120), ("lm_head", 2048, 151936)]


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters // 20):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


rows = []
for name, k, n in SHAPES:
    for m in (1, 4, 8):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        r = {"shape": name, "M": m, "K": k, "N": n}
        t = bench(lambda: F.linear(x, w))
        r["bf16_ms"] = round(t, 4)
        r["bf16_GBps"] = round(w.numel() * 2 / t / 1e6, 1)
        # FP8 e4m3 weight, per-row scale, activation quantized per-row (what fp8dyn does)
        w8 = w.to(torch.float8_e4m3fn)
        ws = torch.ones(1, n, device=dev, dtype=torch.float32)
        mp = max(m, 16)  # _scaled_mm wants M padded to 16 on many builds
        x8 = torch.randn(mp, k, device=dev).to(torch.float8_e4m3fn)
        xs = torch.ones(mp, 1, device=dev, dtype=torch.float32)
        try:
            t = bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16))
            r["fp8_scaled_mm_ms"] = round(t, 4)
            r["fp8_GBps"] = round(w8.numel() / t / 1e6, 1)
        except Exception as e:  # noqa: BLE001
            r["fp8_err"] = str(e)[:80]
        # int8 weight-only packed kernel
        wi = torch.randint(-127, 127, (n, k), device=dev, dtype=torch.int8)
        sc = torch.ones(n, device=dev, dtype=torch.bfloat16)
        try:
            t = bench(lambda: torch._weight_int8pack_mm(x, wi, sc))
            r["int8wo_ms"] = round(t, 4)
            r["int8_GBps"] = round(wi.numel() / t / 1e6, 1)
        except Exception as e:  # noqa: BLE001
            r["int8_err"] = str(e)[:80]
        rows.append(r)
        print(json.dumps(r))
del w, w8, wi
# Peak copy bandwidth for reference
a = torch.empty(512 * 2**20, device=dev, dtype=torch.uint8)
b = torch.empty_like(a)
t = bench(lambda: b.copy_(a), iters=40)
print(json.dumps({"device_copy_GBps": round(2 * a.numel() / t / 1e6, 1), "gpu": torch.cuda.get_device_name()}))
