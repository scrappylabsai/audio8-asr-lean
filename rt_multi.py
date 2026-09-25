#!/usr/bin/env python3
"""N concurrent real-time callers against a /v1/realtime server, using the vendor's own client.

  python rt_multi.py --url ws://127.0.0.1:18192/v1/realtime --stagger 7 data/longform-test-clean-3m-s{5,6,7,8}.wav

Each caller streams its file in real time (100 ms chunks), staggered by --stagger seconds.
Reports per caller: WER vs the .txt reference, final-text latency after its audio ended, the longest
silence between deltas, and the server's step_ms / active-stream counts from metrics.delta events.
"""
import argparse
import asyncio
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from audio8_asr_infinite.examples.vllm_realtime_client import run

HERE = Path(__file__).resolve().parent


async def caller(wav: str, delay_s: float, a) -> dict:
    await asyncio.sleep(delay_s)
    out = HERE / "results" / f"rt-{a.tag}-{Path(wav).stem}.json"
    ns = argparse.Namespace(audio=wav, ws_url=a.url, model="audio8-asr-infinite", language="en",
                            target_delay_ms=480, chunk_ms=100, pace=not a.no_pace, timeout_seconds=3600.0,
                            output=str(out))
    res = await run(ns)
    if res is None:
        res = json.loads(out.read_text())
    norm = EnglishTextNormalizer()
    ref = norm(Path(wav).with_suffix(".txt").read_text())
    deltas = [e["_received_at"] for e in res["events"] if e.get("type") == "transcription.delta"]
    done = next((e["_received_at"] for e in res["events"] if e.get("type") == "transcription.done"), None)
    steps = sorted(e["step_ms"] for e in res["events"] if e.get("type") == "metrics.delta")
    active = [e.get("active_streams", 1) for e in res["events"] if e.get("type") == "metrics.delta"]
    return {
        "audio": Path(wav).name, "start_s": delay_s, "duration_s": res["duration_seconds"],
        "wer_pct": round(100 * jiwer.wer(ref, norm(res["final_text"])), 3),
        "received_done": res["received_done"], "error": res["error_event"],
        "done_after_audio_end_s": round(done - res["send_duration_seconds"], 3) if done else None,
        "max_gap_between_deltas_s": round(max((b - x for x, b in zip(deltas, deltas[1:])), default=0), 2),
        "step_ms_p50": round(steps[len(steps) // 2], 2) if steps else None,
        "step_ms_max": round(steps[-1], 2) if steps else None,
        "max_active_streams": max(active) if active else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="ws://127.0.0.1:18192/v1/realtime")
    ap.add_argument("--stagger", type=float, default=7.0)
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--tag", default="lean-multi")
    a = ap.parse_args()
    rows = await asyncio.gather(*(caller(w, i * a.stagger, a) for i, w in enumerate(a.wavs)))
    print(json.dumps(rows, indent=1))
    (HERE / "results" / f"{a.tag}-summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
