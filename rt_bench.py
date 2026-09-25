#!/usr/bin/env python3
"""Drive the upstream vLLM /v1/realtime server with their own client, then score it.

  python rt_bench.py data/longform-test-clean-10m-s1.wav --pace
Reports: WER vs the .txt reference, time from end-of-audio to transcription.done,
longest silence between deltas (stall detector), and delta chars per minute.
"""
import argparse
import asyncio
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from audio8_asr_infinite.examples.vllm_realtime_client import run

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--ws-url", default="ws://127.0.0.1:18191/v1/realtime")
ap.add_argument("--delay", type=int, default=480)
ap.add_argument("--pace", action="store_true")
ap.add_argument("--tag", default="vllm-upstream")
a = ap.parse_args()

wav = Path(a.wav)
out = Path(__file__).resolve().parent / "results" / f"rt-{a.tag}-{wav.stem}-d{a.delay}.json"
ns = argparse.Namespace(audio=str(wav), ws_url=a.ws_url, model="audio8-asr-infinite", language="en",
                        target_delay_ms=a.delay, chunk_ms=100, pace=a.pace, timeout_seconds=7200.0,
                        output=str(out))
res = asyncio.run(run(ns))
if res is None:
    res = json.loads(out.read_text())

norm = EnglishTextNormalizer()
ref = norm(wav.with_suffix(".txt").read_text())
hyp = norm(res["final_text"])
times = [e["_received_at"] for e in res["events"] if e.get("type") == "transcription.delta"]
done_at = next((e["_received_at"] for e in res["events"] if e.get("type") == "transcription.done"), None)
per_min = {}
for e in res["events"]:
    if e.get("type") == "transcription.delta":
        m = int(e["_received_at"] // 60)
        per_min[m] = per_min.get(m, 0) + len(e.get("delta") or "")
steps = sorted(e["step_ms"] for e in res["events"] if e.get("type") == "metrics.delta")
pct = lambda q: round(steps[min(len(steps) - 1, int(q * len(steps)))], 2) if steps else None
spikes = sum(1 for x in steps if x > 80.0)  # a step slower than its 80 ms audio frame
summary = {
    "tag": a.tag, "audio": wav.name, "duration_s": res["duration_seconds"], "paced": a.pace,
    "wer_pct": round(100 * jiwer.wer(ref, hyp), 3),
    "received_done": res["received_done"], "error": res["error_event"],
    "send_s": res["send_duration_seconds"], "wall_s": res["wall_seconds"],
    "done_after_audio_end_s": (round(done_at - res["send_duration_seconds"], 3) if done_at else None),
    "max_gap_between_deltas_s": round(max((b - a_ for a_, b in zip(times, times[1:])), default=0), 2),
    "step_ms_p50": pct(0.5), "step_ms_p95": pct(0.95), "step_ms_p99": pct(0.99),
    "step_ms_max": round(steps[-1], 2) if steps else None, "steps_over_80ms": spikes, "steps": len(steps),
    "delta_chars_per_min": per_min,
}
print(json.dumps(summary, indent=1))
out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
