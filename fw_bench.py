#!/usr/bin/env python3
"""faster-whisper on the exact utterances of an Audio8 results file (same seeded sample).

Run it in an environment with faster-whisper installed (it needs no Audio8 code).
Scoring happens in score.py.

  python fw_bench.py results/<audio8>.json --model base.en
"""
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
src = json.loads(Path(sys.argv[1]).read_text())
model_name = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "base.en"
compute = sys.argv[sys.argv.index("--compute") + 1] if "--compute" in sys.argv else "float16"
root = HERE / "data" / src.get("corpus", "LibriSpeech") / src["split"]

t0 = time.time()
model = WhisperModel(model_name, device="cuda", compute_type=compute)
load_s = time.time() - t0
paths = {u["id"]: next(root.glob(f"*/*/{u['id']}.flac")) for u in src["utts"]}
list(model.transcribe(str(paths[src["utts"][0]["id"]]), language="en")[0])  # warm-up

hyps, t = {}, time.time()
for u in src["utts"]:
    segs, _ = model.transcribe(str(paths[u["id"]]), language="en", beam_size=1)
    hyps[u["id"]] = " ".join(s.text.strip() for s in segs)
decode_s = time.time() - t

out = HERE / "results" / f"fw-{model_name}-{compute}-{Path(sys.argv[1]).stem}.hyps.json"
out.write_text(json.dumps({"model": f"faster-whisper {model_name} {compute} greedy",
                           "decode_s": round(decode_s, 2), "load_s": round(load_s, 1),
                           "hyps": hyps}, indent=1))
print(out)
