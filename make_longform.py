#!/usr/bin/env python3
"""Build a long-form test stream: seeded LibriSpeech utterances joined with 0.6 s gaps.

  python make_longform.py --split test-clean --minutes 10 --seed 1
Writes data/longform-<split>-<min>m-s<seed>.wav + .txt (reference, in order).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-clean")
ap.add_argument("--minutes", type=float, default=10)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--gap", type=float, default=0.6)
a = ap.parse_args()

root = HERE / "data" / "LibriSpeech" / a.split
utts = []
for t in sorted(root.glob("*/*/*.trans.txt")):
    for line in t.read_text().splitlines():
        uid, text = line.split(" ", 1)
        utts.append((t.parent / f"{uid}.flac", text))
random.Random(a.seed).shuffle(utts)

sr, target = 16000, a.minutes * 60
pieces, refs, total = [], [], 0.0
gap = np.zeros(int(a.gap * sr), dtype=np.float32)
for path, text in utts:
    wav, _ = sf.read(path, dtype="float32")
    pieces += [wav, gap]
    refs.append(text)
    total += len(wav) / sr + a.gap
    if total >= target:
        break
stem = HERE / "data" / f"longform-{a.split}-{int(a.minutes)}m-s{a.seed}"
sf.write(f"{stem}.wav", np.concatenate(pieces), sr)
Path(f"{stem}.txt").write_text(" ".join(refs) + "\n")
print(stem, f"{total:.1f}s", len(refs), "utts")
