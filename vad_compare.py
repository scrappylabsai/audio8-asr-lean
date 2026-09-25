#!/usr/bin/env python3
"""Compare semantic end-of-turn probabilities from two realtime runs of the same recording.

  python vad_compare.py results/rt-vllm-upstream-...json results/rt-lean-...json

Uses `selected_eot_probability` from each semantic_vad.delta event.  The two servers start their
decode at different prompt lengths (vendor served: 18 tokens, lean engine: 25), so the series are
aligned at the lag (in 80 ms steps) with the best correlation before comparing.
"""
import json
import sys

import numpy as np


def series(path):
    ev = json.load(open(path))["events"]
    return np.array([e["selected_eot_probability"] for e in ev if e.get("type") == "semantic_vad.delta"])


a, b = series(sys.argv[1]), series(sys.argv[2])
best = None
for lag in range(-20, 21):
    x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
    n = min(len(x), len(y))
    if n < 50:
        continue
    r = float(np.corrcoef(x[:n], y[:n])[0, 1])
    if best is None or r > best[1]:
        best = (lag, r, n, x[:n], y[:n])
lag, r, n, x, y = best
agree = float(np.mean((x > 0.5) == (y > 0.5)))
print(json.dumps({"steps_a": len(a), "steps_b": len(b), "best_lag_steps": lag, "pearson_r": round(r, 4),
                  "mean_abs_diff": round(float(np.mean(np.abs(x - y))), 4),
                  "eot_gt_0.5_agreement": round(agree, 4), "compared_steps": n,
                  "eot_rate_a": round(float(np.mean(x > 0.5)), 3), "eot_rate_b": round(float(np.mean(y > 0.5)), 3)},
                 indent=1))
