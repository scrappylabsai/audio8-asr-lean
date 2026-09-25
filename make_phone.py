#!/usr/bin/env python3
"""Telephone-degrade a LibriSpeech split: 16 kHz -> 8 kHz -> G.711 mu-law 8-bit -> back to 16 kHz.

Writes data/LibriSpeech-phone/<split>/ with the same tree, ids and trans.txt, and the exact
original sample count, so bench_torch's seeded sample (pool filtered by duration) is identical.
  python make_phone.py test-clean test-other
"""
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

HERE = Path(__file__).resolve().parent
MU = 255.0


def degrade(src: Path) -> None:
    rel = src.relative_to(HERE / "data" / "LibriSpeech")
    dst = HERE / "data" / "LibriSpeech-phone" / rel
    x, sr = sf.read(src, dtype="float32")
    n = len(x)
    y = soxr.resample(x, sr, 8000)
    y = np.clip(y, -1, 1)
    q = np.round((np.sign(y) * np.log1p(MU * np.abs(y)) / np.log1p(MU) + 1) / 2 * 255)  # 8-bit code
    y = (q / 255) * 2 - 1
    y = np.sign(y) * np.expm1(np.abs(y) * np.log1p(MU)) / MU
    z = soxr.resample(y.astype(np.float32), 8000, sr)
    z = np.pad(z, (0, max(0, n - len(z))))[:n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, z, sr)


for split in sys.argv[1:]:
    root = HERE / "data" / "LibriSpeech" / split
    for t in root.glob("*/*/*.trans.txt"):
        d = HERE / "data" / "LibriSpeech-phone" / t.relative_to(HERE / "data" / "LibriSpeech")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(t, d)
    files = sorted(root.glob("*/*/*.flac"))
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(degrade, files, chunksize=32))
    print(split, len(files), "files")
