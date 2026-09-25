#!/usr/bin/env python3
"""Realtime ASR service on the lean Audio8 engine, speaking the vendor's /v1/realtime protocol.

Client -> server:  session.update {model, language, target_delay_ms}
                   input_audio_buffer.commit {final: false}      start
                   input_audio_buffer.append {audio: base64 PCM16 @ 16 kHz}
                   input_audio_buffer.commit {final: true}       end of audio
Server -> client:  session.created, transcription.delta {delta}, semantic_vad.delta (end-of-turn
                   probabilities per horizon), metrics.delta, transcription.done {text, usage}, error

One engine thread owns the GPU.  Each tick, every stream whose next 80 ms window has arrived steps
together in one CUDA-graph replay; streams still waiting on network audio sit the tick out
(`set_ready`), so jitter never corrupts a stream.  Capacity = engine batch rows.

  python asr_server.py --host 127.0.0.1 --port 18192 --capacity 4 --quant int4hqq
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from common import DEFAULT_CHECKPOINT, load_model, resolve_checkpoint
from aiohttp import WSMsgType, web
from transformers import AutoFeatureExtractor, AutoTokenizer

from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)
from engine import LEFT_PAD_TOKENS, LOOK_AHEAD, LOOK_BACK, SAMPLES_PER_TOKEN, LeanAudio8, load_vad_heads

log = logging.getLogger("audio8-asr")
MODEL_ID = "audio8-asr-infinite"
RIGHT_PAD_TEXT = 10  # flush tokens after the final commit (vendor definition)
DELAYS_MS = (240, 320, 480, 560)


class Detok:
    """Incremental detokenizer over visible (non-special) tokens; O(1) per step for 24/7 streams."""

    def __init__(self, tok, skip: set[int], stop: int):
        self.tok, self.skip, self.stop = tok, skip, stop
        self.ids: list[int] = []
        self.prefix = self.read = 0
        self.text = ""
        self.stopped = False

    def _dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def push(self, t: int) -> str:
        if self.stopped or t in self.skip:
            return ""
        if t == self.stop:
            self.stopped = True
            return ""
        self.ids.append(t)
        prev = self._dec(self.ids[self.prefix:self.read])
        new = self._dec(self.ids[self.prefix:])
        if len(new) > len(prev) and not new.endswith("�"):
            delta = new[len(prev):]
            self.prefix, self.read = self.read, len(self.ids)
            if self.prefix > 64:  # keep the working window small
                self.ids = self.ids[self.prefix:]
                self.read -= self.prefix
                self.prefix = 0
            self.text += delta
            return delta
        return ""


class Session:
    def __init__(self, ws, loop):
        self.id = uuid.uuid4().hex[:12]
        self.ws, self.loop = ws, loop
        self.out: asyncio.Queue = asyncio.Queue()
        self.lang = "en"
        self.delay_tokens = 6
        self.validated = False
        self.started = self.final = self.done = False
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)  # stream samples from absolute index `base`
        self.base = 0
        self.row: int | None = None
        self.k = 0
        self.n_tokens = 0
        self.detok: Detok | None = None
        self.step_ms_ema: float | None = None

    @property
    def prefill_tokens(self) -> int:
        return LEFT_PAD_TOKENS + self.delay_tokens + 1

    def emit(self, event: dict) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, event)

    # -- stream bookkeeping (called under lock) --------------------------------
    def start_stream(self) -> None:
        self.buf = np.zeros(LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
        self.base = 0

    def end(self) -> int:
        return self.base + len(self.buf)

    def slice(self, a: int, b: int) -> np.ndarray:
        return self.buf[a - self.base: b - self.base]

    def window_bounds(self, k: int) -> tuple[int, int]:
        if k == 0:
            return 0, self.prefill_tokens * SAMPLES_PER_TOKEN + LOOK_AHEAD
        start = (self.prefill_tokens + k - 1) * SAMPLES_PER_TOKEN
        return start - LOOK_BACK, start + SAMPLES_PER_TOKEN + LOOK_AHEAD

    def drop_consumed(self) -> None:
        keep_from, _ = self.window_bounds(self.k)
        cut = keep_from - self.base
        if cut > 16000:  # trim in >= 1 s chunks
            self.buf = self.buf[cut:]
            self.base = keep_from


class EngineThread(threading.Thread):
    def __init__(self, eng: LeanAudio8, tok, special, lang_ids: dict[str, int],
                 vad_horizons: list[float] | None = None, eot_horizon: float = 1.0):
        super().__init__(daemon=True, name="audio8-engine")
        self.eng, self.tok, self.special, self.lang_ids = eng, tok, special, lang_ids
        self.vad_horizons = vad_horizons
        self.eot_index = (vad_horizons.index(eot_horizon) if vad_horizons and eot_horizon in vad_horizons
                          else 0)
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.free = list(range(eng.B))
        self.skip = {special[k] for k in ("streaming_pad_token_id", "streaming_word_token_id",
                                          "bos_token_id", "pad_token_id")}
        self.steps = 0

    def add(self, s: Session):
        with self.lock:
            self.sessions[s.id] = s

    def remove(self, s: Session):
        with self.lock:
            self.sessions.pop(s.id, None)
        self.wake.set()

    def run(self):
        torch.cuda.set_device(self.eng.dev)
        while True:
            self.wake.wait(timeout=0.05)
            self.wake.clear()
            try:
                while self.tick():
                    pass
            except Exception:  # noqa: BLE001
                log.exception("engine tick failed")

    def _prompt(self, s: Session) -> torch.Tensor:
        sp = self.special
        ids = [sp["bos_token_id"], self.lang_ids[s.lang]] + [sp["streaming_pad_token_id"]] * (s.prefill_tokens - 2)
        return torch.tensor([ids], device=self.eng.dev)

    def _finish(self, s: Session) -> None:
        self.eng.leave(s.row)
        self.free.append(s.row)
        s.row, s.done = None, True
        s.emit({"type": "transcription.done", "text": s.detok.text,
                "usage": {"prompt_tokens": s.prefill_tokens, "completion_tokens": s.n_tokens,
                          "total_tokens": s.prefill_tokens + s.n_tokens}})

    def tick(self) -> bool:
        """One scheduling round. Returns True if it stepped (so the caller loops without waiting)."""
        eng = self.eng
        with self.lock:
            live = [s for s in self.sessions.values() if s.started and not s.done]
        # Release rows of sessions that disconnected mid-stream.
        for b in range(eng.B):
            if b not in self.free and not any(s.row == b for s in live):
                eng.leave(b)
                self.free.append(b)
        ready, work = [False] * eng.B, []
        for s in live:
            with s.lock:
                if s.row is None:
                    a, e = s.window_bounds(0)
                    if s.final and s.end() < e:  # ended before the first window: nothing to decode
                        s.done = True
                        s.emit({"type": "transcription.done", "text": "",
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
                        continue
                    if not self.free or s.end() < e:
                        continue
                    first = torch.from_numpy(s.slice(a, e).copy()).to(eng.dev).unsqueeze(0)
                    s.row = self.free.pop(0)
                    s.detok = Detok(self.tok, self.skip, self.special["eos_token_id"])
                    t0 = eng.join(s.row, first, self._prompt(s), delay_tokens=s.delay_tokens)
                    self._emit_token(s, t0)
                    s.k = 1
                a, e = s.window_bounds(s.k)
                if s.final and e > s.end():
                    self._finish(s)
                    continue
                if s.end() >= e:
                    eng.in_audio[s.row].copy_(torch.from_numpy(s.slice(a, e).copy()))
                    ready[s.row] = True
                    work.append(s)
        if not work:
            eng.set_ready([False] * eng.B)
            return False
        for s in work:
            eng.in_tok[s.row] = eng.out_tok[s.row]
        eng.set_ready(ready)
        t = time.perf_counter()
        out = eng.step().tolist()
        vad = eng.out_vad.tolist() if self.vad_horizons else None
        step_ms = 1000 * (time.perf_counter() - t)
        self.steps += 1
        free_b, total_b = torch.cuda.mem_get_info()
        for s in work:
            s.k += 1
            if vad is not None:
                probs = vad[s.row]
                s.emit({"type": "semantic_vad.delta", "step_index": s.k, "horizons_seconds": self.vad_horizons,
                        "predicted_classes": [max(range(len(p)), key=p.__getitem__) for p in probs],
                        "probabilities": probs, "eot_probabilities": [p[0] for p in probs],
                        "selected_eot_horizon_seconds": self.vad_horizons[self.eot_index],
                        "selected_eot_probability": probs[self.eot_index][0]})
            self._emit_token(s, int(out[s.row]))
            s.step_ms_ema = step_ms if s.step_ms_ema is None else 0.9 * s.step_ms_ema + 0.1 * step_ms
            s.emit({"type": "metrics.delta", "step_index": s.k, "step_ms": step_ms, "audio_ms": 80.0,
                    "rtf": step_ms / 80.0, "rtf_ema": s.step_ms_ema / 80.0, "active_streams": len(work),
                    "gpu_memory_used_bytes": total_b - free_b, "gpu_memory_total_bytes": total_b,
                    "gpu_memory_used_ratio": (total_b - free_b) / total_b})
            with s.lock:
                s.drop_consumed()
        return True

    def _emit_token(self, s: Session, t: int) -> None:
        s.n_tokens += 1
        delta = s.detok.push(t)
        s.emit({"type": "transcription.delta", "delta": delta})


async def realtime(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0, heartbeat=None)
    await ws.prepare(request)
    eng_t: EngineThread = request.app["engine"]
    s = Session(ws, asyncio.get_running_loop())
    eng_t.add(s)

    async def pump():
        while True:
            ev = await s.out.get()
            if ws.closed:
                return
            await ws.send_str(json.dumps(ev))
            if ev.get("type") == "transcription.done":
                return

    sender = asyncio.create_task(pump())
    await s.out.put({"type": "session.created", "session": {"id": s.id, "model": MODEL_ID}})
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                ev = json.loads(msg.data)
            except json.JSONDecodeError:
                await s.out.put({"type": "error", "error": "Invalid JSON", "code": "invalid_json"})
                continue
            et = ev.get("type")
            if et == "session.update":
                lang = ev.get("language") or s.lang
                if lang not in eng_t.lang_ids:
                    await s.out.put({"type": "error", "error": f"language must be one of {sorted(eng_t.lang_ids)}",
                                     "code": "invalid_language"})
                    continue
                delay = ev.get("target_delay_ms")
                if delay is not None and int(delay) not in DELAYS_MS:
                    await s.out.put({"type": "error", "error": f"target_delay_ms must be one of {DELAYS_MS}",
                                     "code": "invalid_delay"})
                    continue
                with s.lock:
                    if not s.started:
                        s.lang = lang
                        if delay is not None:
                            s.delay_tokens = int(delay) // 80
                s.validated = True
            elif et == "input_audio_buffer.append":
                try:
                    pcm = np.frombuffer(base64.b64decode(ev["audio"]), dtype=np.int16).astype(np.float32) / 32768.0
                except Exception:  # noqa: BLE001
                    await s.out.put({"type": "error", "error": "Invalid audio data", "code": "invalid_audio"})
                    continue
                with s.lock:
                    if s.started and not s.final:
                        s.buf = np.concatenate([s.buf, pcm])
                eng_t.wake.set()
            elif et == "input_audio_buffer.commit":
                if not s.validated:
                    await s.out.put({"type": "error", "error": "Send session.update first",
                                     "code": "model_not_validated"})
                    continue
                with s.lock:
                    if ev.get("final"):
                        if s.started and not s.final:
                            pad = (s.delay_tokens + 1 + RIGHT_PAD_TEXT) * SAMPLES_PER_TOKEN
                            s.buf = np.concatenate([s.buf, np.zeros(pad, dtype=np.float32)])
                            s.final = True
                    elif not s.started:  # joins as soon as a row is free (queued otherwise)
                        s.start_stream()
                        s.started = True
                eng_t.wake.set()
            else:
                await s.out.put({"type": "error", "error": f"Unknown event type: {et}", "code": "unknown_event"})
        if s.started and s.final and not s.done:
            await asyncio.wait_for(sender, timeout=30)
    finally:
        eng_t.remove(s)
        sender.cancel()
    return ws


async def transcriptions(request: web.Request) -> web.Response:
    """OpenAI-compatible batch endpoint (what faster-whisper STT servers expose): multipart `file`
    (any format soundfile reads) -> {"text": ...}.  Runs as a stream whose audio is all present,
    sharing the engine rows with realtime callers."""
    import soundfile as sf
    import soxr
    eng_t: EngineThread = request.app["engine"]
    form = await request.post()
    f = form.get("file")
    if f is None or not hasattr(f, "file"):
        return web.json_response({"error": {"message": "multipart field 'file' is required"}}, status=400)
    try:
        wav, sr = sf.read(io.BytesIO(f.file.read()), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return web.json_response({"error": {"message": "could not decode audio"}}, status=400)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000)
    s = Session(None, asyncio.get_running_loop())
    lang = str(form.get("language") or "en")
    s.lang = lang if lang in eng_t.lang_ids else "en"
    s.validated = True
    with s.lock:
        s.start_stream()
        pad = (s.delay_tokens + 1 + RIGHT_PAD_TEXT) * SAMPLES_PER_TOKEN
        s.buf = np.concatenate([s.buf, np.clip(wav, -1, 1).astype(np.float32), np.zeros(pad, dtype=np.float32)])
        s.final = True
        s.started = True
    eng_t.add(s)
    eng_t.wake.set()
    try:
        while True:
            ev = await s.out.get()
            if ev["type"] == "transcription.done":
                return web.json_response({"text": ev["text"].strip()})
    finally:
        eng_t.remove(s)


async def health(request):
    e: EngineThread = request.app["engine"]
    return web.json_response({"status": "ok", "model": MODEL_ID, "capacity": e.eng.B, "trim_mode": e.eng.trim_mode,
                              "active": e.eng.B - len(e.free), "steps": e.steps, "quant": request.app["quant"]})


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})


def build_engine(a):
    dev, dt = torch.device("cuda"), torch.bfloat16
    if a.mem_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained(a.checkpoint, trust_remote_code=True)
    model = load_model(a.checkpoint, quant=a.quant, device="cuda")
    special = resolve_qwen_streaming_special_token_ids(tok)
    lang_ids = {"en": resolve_qwen_language_token_id(tok, "en"), "zh": resolve_qwen_language_token_id(tok, "zh")}
    heads, horizons = load_vad_heads(a.checkpoint, device=dev, dtype=dt)
    eng = LeanAudio8(model, fe, batch=a.capacity, rolling=True, trim_mode=a.trim_mode, vad_heads=heads)
    eng.capture()
    eng.reset()
    for b in range(eng.B):
        eng.leave(b)
    return EngineThread(eng, tok, special, lang_ids, vad_horizons=horizons, eot_horizon=a.eot_horizon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18192)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--quant", default="int4hqq", choices=["none", "int4hqq"])
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--mem-gib", type=float, default=0.0)
    ap.add_argument("--eot-horizon", type=float, default=1.0, help="horizon for selected_eot_probability")
    ap.add_argument("--trim-mode", default="reencode", choices=["rotate", "reencode"],
                    help="decoder window trim: reencode = vendor-served semantics, rotate = cheaper")
    a = ap.parse_args()
    a.checkpoint = resolve_checkpoint(a.checkpoint)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    et = build_engine(a)
    et.start()
    log.info("engine ready in %.1f s (capacity %d, %s, trim %s) | GPU now %.2f GiB, peak during load %.2f GiB",
             time.time() - t0, a.capacity, a.quant, a.trim_mode,
             torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)
    app = web.Application(client_max_size=64 * 2**20)
    app["engine"], app["quant"] = et, a.quant
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/realtime", realtime)
    app.router.add_post("/v1/audio/transcriptions", transcriptions)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
