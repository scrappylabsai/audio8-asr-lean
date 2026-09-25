#!/usr/bin/env python3
"""Score a hyps file against the refs of an Audio8 results file (same normalizer as bench_torch).

  python score.py results/<audio8>.json results/fw-...hyps.json
"""
import json
import sys
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

base = json.loads(Path(sys.argv[1]).read_text())
other = json.loads(Path(sys.argv[2]).read_text())
norm = EnglishTextNormalizer()
refs = [u["ref"] for u in base["utts"]]  # already normalized in bench_torch
hyps = [norm(other["hyps"][u["id"]]) for u in base["utts"]]
audio_s = base["audio_s"]
print(json.dumps({
    "model": other["model"], "split": base["split"], "n": base["n"], "seed": base["seed"],
    "wer_pct": round(100 * jiwer.wer(refs, hyps), 3),
    "audio8_wer_pct": base["wer_pct"],
    "decode_s": other["decode_s"], "rtf": round(other["decode_s"] / audio_s, 4),
}, indent=1))
