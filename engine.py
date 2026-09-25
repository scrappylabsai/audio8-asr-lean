"""Lean fixed-shape streaming engine for Audio8-ASR-Infinite.

Same weights and same math as the vendor's streaming decoder, restructured so one 80 ms step has
fixed shapes and fixed tensor addresses (so the whole step is one CUDA graph):

* KV caches are preallocated rings (tower: 768 slots >= the 750-position sliding window + 3;
  decoder: `dec_slots`, split into a 16-slot stable prompt head and a ring for the tail).  Keys are
  cached with RoPE applied at their absolute position, so ring order does not matter; a per-row
  slot->position table plus a validity mask replaces any reordering.
* Every row (stream) has its own clock: positions and write cursors are [B] GPU tensors that
  advance in place.  An `active[B]` mask is a graph input: idle rows write to a scratch slot and do
  not advance, so one captured graph serves any mix of streams joining and leaving.
* The delay conditioning `1 + ada_rms_norm(t_cond)` is constant per stream: computed at join,
  not every step.
* Rolling ("infinite") policy = the vendor's served policy: decoder ledger kept at
  [16-token prompt head][recent tail], trimmed back to window-(trim_every-1) whenever it exceeds
  `dec_window`, tail keys rotated by -trim so positions stay contiguous (RoPE scores depend only on
  relative position); tower positions re-based the same way before the trained maximum (1500).
  Both run per row, eagerly, between graph replays (`maintain()`).
* The audio frontend (mel + conv stem + frame select) and the projector call the vendor's own
  functions, so the definition matches the benchmarked decoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as rope_dec
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import apply_rotary_pos_emb as rope_tower

from audio8_asr_infinite.simulated_streaming_audio import (
    _extract_streaming_features_batch,
    _group_streaming_audio_hidden_states,
    select_current_20ms_frames,
)

SAMPLES_PER_TOKEN = 1280  # 80 ms @ 16 kHz
LOOK_BACK = 840           # 52.5 ms
LOOK_AHEAD = 40           # 2.5 ms
STEP_WINDOW = LOOK_BACK + SAMPLES_PER_TOKEN + LOOK_AHEAD  # 2160 samples per steady-state step
LEFT_PAD_TOKENS = 18
PREFILL_TOKENS = LEFT_PAD_TOKENS + 6 + 1  # left pad + delay (480 ms) + 1: the first window


def load_vad_heads(checkpoint: str | Path, *, device, dtype) -> tuple[torch.nn.ModuleList, list[float]]:
    """The semantic end-of-turn heads ship in their own file (see the upstream loader bug): one
    Linear(hidden -> classes) per horizon, reading the decoder's final-norm hidden state.
    Class 0 = no speech within that horizon, i.e. end of turn."""
    from safetensors.torch import load_file
    ck = Path(checkpoint)
    cfg = json.loads((ck / "config.json").read_text())
    horizons = [float(h) for h in cfg["semantic_vad_horizons_seconds"]]
    sd = load_file(str(ck / "semantic_vad_heads.safetensors"))
    heads = torch.nn.ModuleList()
    for i in range(len(horizons)):
        w, b = sd[f"semantic_vad_heads.{i}.weight"], sd[f"semantic_vad_heads.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        heads.append(lin)
    return heads.eval(), horizons


class LeanAudio8:
    def __init__(self, model, feature_extractor, *, batch: int, delay_tokens: int = 6,
                 frame_len: int = 4, dec_slots: int = 512, tower_slots: int = 768,
                 rolling: bool = False, dec_window: int = 360, trim_every: int = 38,
                 stable_prefix: int = 16, tower_max_pos: int = 1500, trim_mode: str = "rotate",
                 vad_heads: torch.nn.ModuleList | None = None):
        self.m, self.fe, self.B, self.frame_len = model, feature_extractor, batch, frame_len
        p = next(model.parameters())
        self.dev, self.dt = p.device, p.dtype
        tw = model.audio_tower
        lm = model.language_model.model
        self.tw, self.lm, self.lm_head = tw, lm, model.language_model.lm_head
        self.window = int(tw.config.sliding_window)
        # Head geometry from the modules themselves (the tower's head_dim=64 != hidden/heads).
        ta, da = tw.layers[0].self_attn, lm.layers[0].self_attn
        self.Dt = ta.head_dim
        self.Ht = ta.q_proj.out_features // self.Dt
        self.Htkv = ta.k_proj.out_features // self.Dt
        self.Dd = da.head_dim
        self.Hq = da.q_proj.out_features // self.Dd
        self.Hkv = da.k_proj.out_features // self.Dd
        self.St, self.Sd = tower_slots, dec_slots
        assert self.St >= self.window + frame_len - 1
        B, dev, dt = batch, self.dev, self.dt
        self.delay_tokens = delay_tokens

        # Caches carry one extra scratch slot (index St / Sd): idle rows write and attend there.
        self.tk = torch.zeros(len(tw.layers), B, self.Htkv, self.St + 1, self.Dt, device=dev, dtype=dt)
        self.tv = torch.zeros_like(self.tk)
        self.dk = torch.zeros(len(lm.layers), B, self.Hkv, self.Sd + 1, self.Dd, device=dev, dtype=dt)
        self.dv = torch.zeros_like(self.dk)
        # Decoder input embeddings per slot, kept so a trim can re-encode the retained window.
        self.eh = torch.zeros(B, self.Sd + 1, lm.config.hidden_size, device=dev, dtype=dt)
        self.t_slot_pos = torch.full((B, self.St + 1), -1, device=dev, dtype=torch.long)
        self.d_slot_pos = torch.full((B, self.Sd + 1), -1, device=dev, dtype=torch.long)
        self.t_pos = torch.zeros(B, device=dev, dtype=torch.long)
        self.d_pos = torch.zeros(B, device=dev, dtype=torch.long)
        self.t_cursor = torch.zeros(B, device=dev, dtype=torch.long)
        self.d_cursor = torch.zeros(B, device=dev, dtype=torch.long)
        self.active = torch.ones(B, device=dev, dtype=torch.bool)
        self.t_scratch = torch.arange(self.St + 1, device=dev) == self.St
        self.d_scratch = torch.arange(self.Sd + 1, device=dev) == self.Sd
        self.mod = [torch.empty(B, 1, lm.config.hidden_size, device=dev, dtype=dt) for _ in lm.layers]
        self._set_delay(slice(None), delay_tokens)
        self.eos = int(model.config.eos_token_id)

        self.prefix = stable_prefix
        self.tail = self.Sd - stable_prefix
        self.rolling = rolling
        assert trim_mode in ("rotate", "reencode")
        self.trim_mode = trim_mode
        self.dec_window, self.trim_every = dec_window, trim_every
        self.dec_target = dec_window - (trim_every - 1)
        self.tower_max_pos = tower_max_pos
        assert self.tail >= dec_window, "decoder ring must hold the whole window"
        self.dec_inv_freq = lm.rotary_emb.inv_freq.float()
        self.tow_inv_freq = tw.rotary_emb.inv_freq.float()

        # Static step I/O.
        self.in_audio = torch.zeros(B, STEP_WINDOW, device=dev, dtype=torch.float32)
        self.in_tok = torch.zeros(B, device=dev, dtype=torch.long)
        self.out_tok = torch.zeros(B, device=dev, dtype=torch.long)
        self.vad_heads = vad_heads
        n_h = len(vad_heads) if vad_heads is not None else 1
        n_c = vad_heads[0].out_features if vad_heads is not None else 1
        self.out_vad = torch.zeros(B, n_h, n_c, device=dev, dtype=torch.float32)  # softmax per horizon
        self.graph = None
        self._fn = self._step_body
        self._core_fn = self._core
        self._full = self._state()
        self.reset()

    # ---------------------------------------------------------------- state
    def _state(self, b: int | None = None) -> SimpleNamespace:
        """All per-row tensors, or views of row `b` (writes go through to the base tensors)."""
        r = slice(None) if b is None else slice(b, b + 1)
        return SimpleNamespace(
            tk=self.tk[:, r], tv=self.tv[:, r], dk=self.dk[:, r], dv=self.dv[:, r], eh=self.eh[r],
            t_slot_pos=self.t_slot_pos[r], d_slot_pos=self.d_slot_pos[r],
            t_pos=self.t_pos[r], d_pos=self.d_pos[r], t_cursor=self.t_cursor[r], d_cursor=self.d_cursor[r],
            active=self.active[r], mod=[m[r] for m in self.mod], out_vad=self.out_vad[r])

    @torch.inference_mode()
    def _set_delay(self, rows: slice, delay_tokens: int) -> None:
        n = self.B if rows == slice(None) else 1
        t_cond = self.m.build_t_cond(delay_tokens, batch_size=n, device=self.dev, dtype=self.dt,
                                     frame_len=self.frame_len)
        for m, layer in zip(self.mod, self.lm.layers):
            m[rows] = 1 + layer.ada_rms_norm(t_cond).to(self.dt)

    def _reset_rows(self, rows: slice) -> None:
        self.t_slot_pos[rows] = -1
        self.d_slot_pos[rows] = -1
        for t in (self.t_pos, self.d_pos, self.t_cursor, self.d_cursor):
            t[rows] = 0

    @torch.inference_mode()
    def reset(self) -> None:
        """All rows fresh and active (the batch-bench mode: every stream starts together)."""
        self._reset_rows(slice(None))
        self.active.fill_(True)
        self._active = [True] * self.B
        self._L = [0] * self.B     # host mirror: decoder ledger length per row
        self._tp = [0] * self.B    # host mirror: next tower position per row
        self.trims = self.rebases = 0

    # ---------------------------------------------------------------- rolling maintenance
    @staticmethod
    def _rotate(keys: torch.Tensor, inv_freq: torch.Tensor, delta: int) -> torch.Tensor:
        """Shift RoPE'd keys by `delta` positions (same arithmetic as the vendor plugin)."""
        freqs = inv_freq.to(keys.device) * float(delta)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(keys.dtype), emb.sin().to(keys.dtype)
        half = keys.shape[-1] // 2
        rot = torch.cat((-keys[..., half:], keys[..., :half]), dim=-1)
        return keys * cos + rot * sin

    def _trim_decoder(self, b: int, n: int) -> None:
        if self.trim_mode == "reencode":
            return self._reencode_decoder(b, n)
        sp = self.d_slot_pos[b]
        sp.masked_fill_((sp >= self.prefix) & (sp < self.prefix + n), -1)
        idx = (sp >= self.prefix + n).nonzero().squeeze(1)
        sp.index_copy_(0, idx, sp.index_select(0, idx) - n)
        dk = self.dk[:, b]  # [L, H, S+1, D] view
        dk.index_copy_(2, idx, self._rotate(dk.index_select(2, idx), self.dec_inv_freq, -n))
        self.d_pos[b] -= n
        self._L[b] -= n
        self.trims += 1

    def _reencode_decoder(self, b: int, n: int) -> None:
        """Vendor-served semantics: drop the oldest n tail tokens, then rebuild the kept
        [prompt head][tail] from their input embeddings as a fresh prefill at positions 0..K-1,
        so no kept key carries context from evicted tokens."""
        sp = self.d_slot_pos[b]
        keep = (sp >= 0) & ((sp < self.prefix) | (sp >= self.prefix + n))
        idx = keep.nonzero().squeeze(1)
        idx = idx[torch.argsort(sp.index_select(0, idx))]
        emb = self.eh[b].index_select(0, idx).unsqueeze(0).clone()
        K = int(idx.numel())
        st = self._state(b)
        st.d_slot_pos.fill_(-1)
        st.d_pos.zero_()
        st.d_cursor.zero_()
        self._decoder(emb, st, prefill=True)
        self._L[b] = K
        self.trims += 1

    def _rebase_tower(self, b: int, delta: int) -> None:
        sp = self.t_slot_pos[b]
        sp.copy_(torch.where(sp >= 0, sp - delta, sp))
        sp.masked_fill_(sp < 0, -1)  # anything shifted below 0 was already outside the window
        tk = self.tk[:, b]
        tk.copy_(self._rotate(tk, self.tow_inv_freq, -delta))
        self.t_pos[b] -= delta
        self._tp[b] -= delta
        self.rebases += 1

    @torch.inference_mode()
    def maintain(self) -> None:
        """Host-side check after each step; acts only when a bound is crossed (no sync otherwise)."""
        if not self.rolling:
            return
        for b in range(self.B):
            if not self._active[b]:
                continue
            if self._L[b] > self.dec_window:
                self._trim_decoder(b, self._L[b] - self.dec_target)
            if self._tp[b] + self.frame_len > self.tower_max_pos:
                self._rebase_tower(b, self._tp[b] - (self.window + 2 * self.frame_len + 2))

    # ---------------------------------------------------------------- pieces
    def _frontend(self, audio: torch.Tensor, n_tokens: int) -> torch.Tensor:
        feats = _extract_streaming_features_batch(
            feature_extractor=self.fe, audio_windows=audio, sampling_rate=16000, device=self.dev,
        ).to(self.dt)
        conv = self.tw.embedder(feats)
        conv, _ = select_current_20ms_frames(conv, expected_20ms_frames=n_tokens * self.frame_len)
        return conv

    def _tower(self, x: torch.Tensor, st: SimpleNamespace) -> torch.Tensor:
        B, T, _ = x.shape
        ar = torch.arange(T, device=self.dev)
        act = st.active.unsqueeze(1)
        pos = st.t_pos.unsqueeze(1) + ar                                  # [B, T]
        cos, sin = self.tw.rotary_emb(x, pos)
        slots = torch.where(act, (st.t_cursor.unsqueeze(1) + ar) % self.St, self.St)
        st.t_slot_pos.scatter_(1, slots, torch.where(act, pos, -1))
        sp = st.t_slot_pos[:, None, None, :]
        q_pos = pos[:, None, :, None]
        mask = (sp >= 0) & (sp <= q_pos) & (sp > q_pos - self.window)
        mask = mask | ((~st.active)[:, None, None, None] & self.t_scratch)
        rows = torch.arange(B, device=self.dev).unsqueeze(1).expand(B, T)
        for i, layer in enumerate(self.tw.layers):
            a = layer.self_attn
            h = layer.self_attn_layer_norm(x)
            q = a.q_proj(h).view(B, T, self.Ht, self.Dt).transpose(1, 2)
            k = a.k_proj(h).view(B, T, self.Htkv, self.Dt).transpose(1, 2)
            v = a.v_proj(h).view(B, T, self.Htkv, self.Dt).transpose(1, 2)
            q, k = rope_tower(q, k, cos, sin)
            st.tk[i][rows, :, slots] = k.transpose(1, 2)
            st.tv[i][rows, :, slots] = v.transpose(1, 2)
            o = F.scaled_dot_product_attention(q, st.tk[i], st.tv[i], attn_mask=mask, scale=a.scaling,
                                               enable_gqa=self.Htkv != self.Ht)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
            x = x + layer.mlp(layer.final_layer_norm(x))
        step = st.active.long() * T
        st.t_pos += step
        st.t_cursor += step
        return self.tw.norm(x)

    def _project(self, hidden: torch.Tensor, n_tokens: int) -> torch.Tensor:
        g = _group_streaming_audio_hidden_states(
            model=self.m, audio_hidden_states=hidden, num_80ms_tokens=n_tokens, frame_len=self.frame_len,
        )
        return self.m.multi_modal_projector(g).to(self.dt)

    def _decoder(self, emb: torch.Tensor, st: SimpleNamespace, prefill: bool = False) -> torch.Tensor:
        B, T, _ = emb.shape
        ar = torch.arange(T, device=self.dev)
        act = st.active.unsqueeze(1)
        pos = st.d_pos.unsqueeze(1) + ar
        cos, sin = self.lm.rotary_emb(emb, pos)
        if prefill:  # prompt fills the head slots, its remainder starts the tail ring
            slots = ar.unsqueeze(0).expand(B, T)
        else:
            slots = self.prefix + (st.d_cursor.unsqueeze(1) + ar) % self.tail
        slots = torch.where(act, slots, self.Sd)
        st.d_slot_pos.scatter_(1, slots, torch.where(act, pos, -1))
        rows = torch.arange(B, device=self.dev).unsqueeze(1).expand(B, T)
        st.eh[rows, slots] = emb
        sp = st.d_slot_pos[:, None, None, :]
        mask = (sp >= 0) & (sp <= pos[:, None, :, None])
        mask = mask | ((~st.active)[:, None, None, None] & self.d_scratch)
        rows = torch.arange(B, device=self.dev).unsqueeze(1).expand(B, T)
        x = emb
        for i, layer in enumerate(self.lm.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(B, T, self.Hq, self.Dd).transpose(1, 2)
            k = a.k_proj(h).view(B, T, self.Hkv, self.Dd).transpose(1, 2)
            v = a.v_proj(h).view(B, T, self.Hkv, self.Dd).transpose(1, 2)
            q, k = rope_dec(q, k, cos, sin)
            st.dk[i][rows, :, slots] = k.transpose(1, 2)
            st.dv[i][rows, :, slots] = v.transpose(1, 2)
            o = F.scaled_dot_product_attention(q, st.dk[i], st.dv[i], attn_mask=mask,
                                               scale=a.scaling, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x) * st.mod[i])
        step = st.active.long() * T
        st.d_pos += step
        if prefill:
            st.d_cursor.copy_(torch.where(st.active, max(0, T - self.prefix), st.d_cursor))
        else:
            st.d_cursor += step
        h = self.lm.norm(x[:, -1:, :])
        logits = self.lm_head(h)[:, 0, :]
        logits[:, self.eos] = float("-inf")  # EOS always suppressed (vendor definition)
        if self.vad_heads is not None:  # semantic end-of-turn, same hidden state the LM head reads
            v = torch.stack([hd(h[:, 0, :]) for hd in self.vad_heads], dim=1)
            st.out_vad.copy_(torch.softmax(v.float(), dim=-1))
        return logits.argmax(dim=-1)

    def _prefill_rows(self, st, first_audio: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        n = int(prompt_ids.shape[1])
        audio = self._project(self._tower(self._frontend(first_audio, n), st), n)
        emb = self.lm.embed_tokens(prompt_ids) + audio
        return self._decoder(emb.to(self.dt), st, prefill=True)

    # ---------------------------------------------------------------- API
    @torch.inference_mode()
    def prefill(self, first_audio: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Batch mode: every row starts together from B x N0 samples covering the prompt window."""
        tok = self._prefill_rows(self._full, first_audio, prompt_ids)
        self.out_tok.copy_(tok)
        n = int(prompt_ids.shape[1])
        self._L = [n] * self.B
        self._tp = [n * self.frame_len] * self.B
        self.maintain()
        return tok

    @torch.inference_mode()
    def join(self, b: int, first_audio: torch.Tensor, prompt_ids: torch.Tensor,
             delay_tokens: int | None = None) -> int:
        """Start stream `b` (1 x N0 samples, 1 x prompt) while other rows keep running. Returns its
        first token.  Row views make this an eager single-row prefill into the shared caches."""
        self._reset_rows(slice(b, b + 1))
        if delay_tokens is not None and delay_tokens != self.delay_tokens:
            self._set_delay(slice(b, b + 1), delay_tokens)
        self.active[b] = True
        self._active[b] = True
        tok = self._prefill_rows(self._state(b), first_audio, prompt_ids)
        self.out_tok[b] = tok[0]
        n = int(prompt_ids.shape[1])
        self._L[b], self._tp[b] = n, n * self.frame_len
        self.maintain()
        return int(tok[0])

    def set_ready(self, ready: list[bool]) -> None:
        """Per-tick participation: rows without their next 80 ms of audio sit this replay out
        (no advance, no cache writes) without losing state.  Only rows that joined may be ready."""
        if ready != self._active:
            self._active = list(ready)
            self.active.copy_(torch.tensor(ready, device=self.dev))

    @torch.inference_mode()
    def leave(self, b: int) -> None:
        self.active[b] = False
        self._active[b] = False
        self._set_delay(slice(b, b + 1), self.delay_tokens)

    def _core(self, conv: torch.Tensor) -> None:
        audio = self._project(self._tower(conv, self._full), 1)
        emb = self.lm.embed_tokens(self.in_tok).unsqueeze(1) + audio
        # Rows sitting this replay out keep their last token: it is their next step's input.
        tok = self._decoder(emb.to(self.dt), self._full)
        self.out_tok.copy_(torch.where(self.active, tok, self.out_tok))

    def _step_body(self) -> None:
        # Frontend stays eager-traced (its vendor code holds numpy state that Dynamo can't guard);
        # it is tiny and is captured in the CUDA graph together with the core.
        self._core_fn(self._frontend(self.in_audio, 1))

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        """One 80 ms step for every active row from `in_audio` (B x 2160) and `in_tok` (B,)."""
        if self.graph is not None:
            self.graph.replay()
        else:
            self._fn()
        for b in range(self.B):
            if self._active[b]:
                self._L[b] += 1
                self._tp[b] += self.frame_len
        self.maintain()
        return self.out_tok

    def compile_step(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        """Kept for experiments; measured slower on sm_120 (see project doc)."""
        self._core_fn = torch.compile(self._core, mode=mode, dynamic=False, fullgraph=False)

    @torch.inference_mode()
    def capture(self, warmup: int = 3) -> None:
        """Record step() as one CUDA graph. Mutates state, so call reset() before real use."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._fn()
        self.graph = g
        # Warm the eager shapes a live stream hits later (join prefill, re-encode trim), so their
        # first real call doesn't pay first-use kernel setup (measured: 1.4 s on a cold trim).
        st = self._state(0)
        n = PREFILL_TOKENS
        self._prefill_rows(st, torch.zeros(1, n * SAMPLES_PER_TOKEN + LOOK_AHEAD, device=self.dev),
                           torch.zeros(1, n, device=self.dev, dtype=torch.long))
        if self.rolling and self.trim_mode == "reencode":
            st.d_slot_pos.fill_(-1)
            st.d_pos.zero_()
            st.d_cursor.zero_()
            self._decoder(torch.zeros(1, self.dec_target, self.eh.shape[-1], device=self.dev, dtype=self.dt),
                          st, prefill=True)
        torch.cuda.synchronize()
