# audio8-asr-lean

A lean streaming engine and realtime server for [Edge0's Audio8 ASR Infinite](https://huggingface.co/Edge0/Audio8-ASR-Infinite).
It uses their weights, audio frontend and projector unchanged. We rewrote the part that runs each 80 ms step.

On an RTX 5090 Laptop GPU:

| | Upstream (vLLM 0.27.1 plugin) | This engine, bf16 | This engine, int4 |
|---|---|---|---|
| LibriSpeech test-clean WER (300 seeded utterances) | 2.973 % (torch decoder) | **2.973 %, identical** | 3.137 % |
| Same clips over a telephone line (8 kHz μ-law) | 3.499 % | 3.548 % | 3.761 % |
| Time per 80 ms step, 1 stream | 20.1 ms (torch) · 21.8 ms p50 (served) | 19.2 ms | **12.4 ms** |
| Time per step, 8 streams in one batch | 24.5 ms | 24.9 ms | 22.6 ms |
| GPU memory | 12.9 GB for 1 served stream | 8.0 GB (1) · 9.1 GB (4) | **3.1 GB (1) · 4.1 GB (4)** |
| Slowest steps in a 10-minute stream | 73 ms p99, up to 187–510 ms | 58 ms p99 (one 214 ms first trim) | 127 ms p99, max 148 ms |
| Server start-up | 36 s | — | 8.6 s (peak 3.3 GB while loading) |

For comparison, faster-whisper `base.en` on the same clips: 4.42 % (clean), 5.17 % (telephone).

Built by [ScrappyLabs](https://scrappylabs.ai) with Claude Code. We filed the one upstream bug we hit as
[Edge0-AI/Audio8-ASR-Infinite#1](https://github.com/Edge0-AI/Audio8-ASR-Infinite/issues/1).

## What changed, and why

Audio8 ASR Infinite pairs a Voxtral-Realtime audio tower (32 layers) with a Qwen2.5-3B decoder (36 layers) and emits
one text token per 80 ms of audio. Upstream serves it through a vLLM plugin, but the decoder runs as the
Hugging Face module in eager mode, with the KV cache concatenated one step at a time. Profiling one step
(batch 1): 3,268 kernel launches, 16.6 ms of GPU time, most of it spent reading about 7.8 GB of weights.

`engine.py` keeps the math and changes the shape of the work:

- **Fixed-shape step.** KV caches are preallocated ring buffers (tower: 768 slots over its 750-position sliding
  window; decoder: a 16-slot prompt head plus a ring). Keys are stored with RoPE already applied, so slot order
  doesn't matter; a slot-to-position table and a mask replace any reordering. Positions and write cursors live on
  the GPU, so a whole step is captured once as a CUDA graph and replayed.
- **Per-stream constants hoisted.** The delay conditioning (`1 + ada_rms_norm(t_cond)` in every decoder layer) is
  fixed for a stream, so it's computed when the stream joins instead of every step.
- **Independent streams in one graph.** Each row of the batch has its own clock. An `active` mask is a graph
  input: a row whose next 80 ms of audio hasn't arrived sits the replay out and keeps its state. Streams join
  (a one-row prefill) and leave without recapturing. Each stream's tokens are identical to decoding it alone,
  including when streams randomly skip ticks.
- **Unlimited length.** The same rolling policy as upstream: the decoder keeps the 16-token prompt head plus the
  most recent tokens (window 360, trimmed to 323 every 38 steps), and tower positions are re-based before the
  1,500 they were trained on. Upstream effectively recomputes the kept window after each trim; `--trim-mode reencode`
  does the same with one prefill from stored input embeddings. `rotate` shifts the kept keys instead: cheaper,
  slightly less accurate.
- **int4 weights** (torchao, HQQ scales, group 128, tinygemm layout) for everything except the LM head, projector and
  delay MLPs. Batch-1 decoding is bound by weight reads, so fewer bytes means a faster step.
- **Semantic end of turn.** The checkpoint's end-of-turn heads run inside the step. Against the upstream server on
  the same recording: correlation 0.979, 99.3 % agreement on the end-of-turn call.

Long recordings (three 10-minute streams, WER): upstream server 3.07 % average, this engine 3.24 % (bf16,
reencode) and 3.43 % (int4, reencode). No drift over 10 minutes in any configuration.

## What didn't work

On sm_120 with torch 2.14:
- `torch.compile` made the step slower (19.8–21 ms vs 18.5 ms graphed eager). With FP8 weights, `max-autotune` picked
  kernels that took 81 ms per step at batch 8. Per-layer weights fit in this GPU's L2 cache, so autotuning benchmarks look far better than a real step.
  The same trap applies to hand-written microbenchmarks: rotate buffers larger than L2.
- torchao int8 weight-only: the kernel reaches about 88 GB/s here, so the step got slower (32–35 ms).
- FP8 kept WER unchanged but had no fast kernel at small batch (24 ms per step graphed, 98 ms eager).
- Finer int4 groups (64, 32) did not improve WER over group 128 with HQQ.
- All of int4's WER cost comes from the decoder MLPs. Quantizing the tower and attention too costs nothing extra.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # the CUDA build for your GPU
pip install -r requirements.txt
python asr_server.py --quant int4hqq --capacity 4            # ws://127.0.0.1:18192/v1/realtime
```

The server speaks the upstream `/v1/realtime` protocol (`session.update`, `input_audio_buffer.append` /
`commit`), so the upstream web client and `vllm_realtime_client` work unchanged. It emits `transcription.delta`,
`semantic_vad.delta`, `metrics.delta` and `transcription.done`. Weights download from the Hub on first run
(`AUDIO8_CHECKPOINT` overrides the path). With int4, loading streams layer by layer, so peak GPU memory stays
near the final footprint.

To reproduce the tables: `./get_data.sh`, then

| Script | Measures |
|---|---|
| `bench_torch.py` | upstream torch decoder: WER, speed, memory |
| `engine_bench.py` | this engine on the same clips, with `--compare-vendor N` for a token-by-token check |
| `longform_bench.py` | 10-minute streams with rolling trims (`--trim-mode`) |
| `multistream_bench.py` | staggered streams in one graph, with `--jitter`, checked against solo decoding |
| `rt_bench.py`, `rt_multi.py` | any `/v1/realtime` server, driven by the upstream client |
| `fw_bench.py`, `score.py` | faster-whisper on the same clips |
| `probe_*.py`, `vad_compare.py` | profiling, kernel bandwidth, end-of-turn comparison |

Results from our runs are in `results/`. LibriSpeech sample: seed-0 shuffle of all utterances ≤ 29 s, 300 per
split, Whisper English normalizer, corpus WER.

## Limits

- Tested on an RTX 5090 Laptop GPU (sm_120; all tables) and, from a fresh install, an RTX 4090 (sm_89): same WER, but
  there bf16 was faster than int4 (16.6 vs 22.2 ms per step). The desktop card has more memory bandwidth and the int4
  kernel is less efficient on it, so on a 4090 int4 buys memory, not speed.
- English and Chinese, as upstream. We measured English only.
- int4 re-encode trims take about 140 ms (the tinygemm kernel is built for batch 1). With 480 ms of delay that shows
  up as a brief pause in the text, not a stall. Use `--quant none` or `--trim-mode rotate` if that matters more
  than memory.
- The prompt is 25 tokens, as in upstream's reference decoder; the upstream server uses 18. That offsets
  end-of-turn events by 7–8 steps relative to it.

## License

Apache-2.0. This builds on Audio8 ASR Infinite by Edge0 (Apache-2.0); see `NOTICE`.
