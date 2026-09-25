#!/usr/bin/env python3
"""Independent streams on one captured graph: staggered joins, leaves at end, rolling on.

  python multistream_bench.py data/longform-test-clean-3m-s5.wav ... --starts 0,90,200,350 --graph

Each stream's transcript from the shared run is compared with the same stream decoded ALONE (same
engine, other rows idle), and both are scored against the .txt reference.  Also reports the per-tick
latency distribution against how many streams were active.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from common import DEFAULT_CHECKPOINT, resolve_checkpoint
from transformers import AutoFeatureExtractor, AutoTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

from audio8_asr_infinite.examples.torch_streaming_decode import StreamingDecodeConfig
from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForConditionalGeneration,
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)
from audio8_asr_infinite.simulated_streaming_audio import decode_visible_text
from engine import LOOK_AHEAD, LOOK_BACK, PREFILL_TOKENS, SAMPLES_PER_TOKEN, LeanAudio8
from engine_bench import DELAY, LEFT_PAD, RIGHT_PAD_TEXT, n_windows

HERE = Path(__file__).resolve().parent


class Stream:
    def __init__(self, path: str, start: int, cfg, dev):
        self.path, self.start = path, start
        wav = np.clip(sf.read(path, dtype="float32")[0], -1, 1)
        self.W = n_windows(wav, cfg)
        right = (DELAY + 1 + RIGHT_PAD_TEXT) * SAMPLES_PER_TOKEN
        s = torch.zeros(LEFT_PAD * SAMPLES_PER_TOKEN + len(wav) + right + SAMPLES_PER_TOKEN, device=dev)
        s[LEFT_PAD * SAMPLES_PER_TOKEN: LEFT_PAD * SAMPLES_PER_TOKEN + len(wav)] = torch.from_numpy(wav)
        self.audio = s
        self.k = 0          # next window index
        self.ids: list[int] = []
        self.row = None

    def first(self):
        return self.audio[: PREFILL_TOKENS * SAMPLES_PER_TOKEN + LOOK_AHEAD].unsqueeze(0)

    def window(self, k):
        start = (PREFILL_TOKENS + k - 1) * SAMPLES_PER_TOKEN
        return self.audio[start - LOOK_BACK: start + SAMPLES_PER_TOKEN + LOOK_AHEAD]


def run(eng, streams, prompt, jitter: float = 0.0, seed: int = 0):
    """jitter = per-tick probability that a live stream's next audio hasn't arrived (it sits out)."""
    import random
    rng = random.Random(seed)
    eng.reset()
    for b in range(eng.B):
        eng.leave(b)
    free = list(range(eng.B))
    ticks, t = [], 0
    pending = sorted(streams, key=lambda s: s.start)
    live: list[Stream] = []
    while pending or live:
        while pending and pending[0].start <= t and free:
            s = pending.pop(0)
            s.row = free.pop(0)
            s.ids = [eng.join(s.row, s.first(), prompt)]
            s.k = 1
            live.append(s)
        ready = [s for s in live if rng.random() >= jitter] if jitter else list(live)
        if ready:
            mask = [False] * eng.B
            for s in ready:
                eng.in_audio[s.row].copy_(s.window(s.k))
                eng.in_tok[s.row] = eng.out_tok[s.row]
                mask[s.row] = True
            eng.set_ready(mask)
            t0 = time.perf_counter()
            out = eng.step()
            host = out.tolist()  # one sync per tick: a server needs every row's token
            ticks.append((1000 * (time.perf_counter() - t0), len(ready)))
            for s in list(ready):
                s.ids.append(int(host[s.row]))
                s.k += 1
                if s.k >= s.W:
                    eng.leave(s.row)
                    free.append(s.row)
                    live.remove(s)
        t += 1
    return ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--starts", default="0,90,200,350")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--quant", default="none", choices=["none", "int4hqq"])
    ap.add_argument("--mem-gib", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--tag", default="multi")
    a = ap.parse_args()
    if a.mem_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    dev, dt = torch.device("cuda"), torch.bfloat16
    ck = resolve_checkpoint(DEFAULT_CHECKPOINT)
    cfg_m = Audio8ASRInfiniteConfig.from_pretrained(ck)
    cfg_m.semantic_vad_horizons_seconds = None
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained(ck, trust_remote_code=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        ck, config=cfg_m, trust_remote_code=True, torch_dtype=dt).to(dev).eval()
    if a.quant == "int4hqq":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d",
                                              int4_choose_qparams_algorithm="hqq"),
                  filter_fn=lambda m, f: isinstance(m, torch.nn.Linear)
                  and not any(k in f for k in ("lm_head", "multi_modal_projector", "ada_rms_norm")))
        torch.cuda.empty_cache()
    special = resolve_qwen_streaming_special_token_ids(tok)
    lang = resolve_qwen_language_token_id(tok, "en")
    cfg = StreamingDecodeConfig(transcription_delay_ms=480, left_pad_tokens=LEFT_PAD)
    prompt = torch.tensor([[special["bos_token_id"], lang] + [special["streaming_pad_token_id"]] * (PREFILL_TOKENS - 2)],
                          device=dev)
    eng = LeanAudio8(model, fe, batch=a.batch, delay_tokens=DELAY, rolling=True)
    if a.graph:
        eng.capture()
    torch.cuda.reset_peak_memory_stats()
    starts = [int(x) for x in a.starts.split(",")]
    norm = EnglishTextNormalizer()
    text = lambda ids: norm(decode_visible_text(tokenizer=tok, generated_token_ids=ids, special_ids=special)[0])

    shared = [Stream(w, st, cfg, dev) for w, st in zip(a.wavs, starts)]
    ticks = run(eng, shared, prompt, jitter=a.jitter)
    per = []
    for s in shared:
        alone = Stream(s.path, 0, cfg, dev)
        run(eng, [alone], prompt)
        ref = norm(Path(s.path).with_suffix(".txt").read_text())
        hs, ha = text(s.ids), text(alone.ids)
        per.append({"audio": Path(s.path).name, "start_tick": s.start,
                    "wer_shared": round(100 * jiwer.wer(ref, hs), 3), "wer_alone": round(100 * jiwer.wer(ref, ha), 3),
                    "text_identical": hs == ha, "token_ids_identical": s.ids == alone.ids,
                    "words_differ": jiwer.process_words(ha, hs).substitutions + jiwer.process_words(ha, hs).deletions
                    + jiwer.process_words(ha, hs).insertions})
    by_n = {}
    for ms, n in ticks:
        by_n.setdefault(n, []).append(ms)
    lat = {n: {"ticks": len(v), "p50": round(sorted(v)[len(v) // 2], 2), "max": round(max(v), 2)}
           for n, v in sorted(by_n.items())}
    res = {"tag": a.tag, "jitter": a.jitter, "quant": a.quant, "graph": a.graph, "capacity": a.batch, "streams": per,
           "tick_ms_by_active_streams": lat, "ticks_over_80ms": sum(1 for ms, _ in ticks if ms > 80),
           "trims": eng.trims, "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(res, indent=1))
    (HERE / "results" / f"{a.tag}-{a.quant}-{'graph' if a.graph else 'eager'}-b{a.batch}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
