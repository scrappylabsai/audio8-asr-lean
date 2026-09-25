"""Shared helpers: resolve the checkpoint (local dir, or download from the Hugging Face Hub)."""
import os
from pathlib import Path

DEFAULT_CHECKPOINT = os.environ.get("AUDIO8_CHECKPOINT", "Edge0/Audio8-ASR-Infinite")


def resolve_checkpoint(ref: str) -> str:
    """A local directory is used as-is; anything else is treated as a Hub repo id and downloaded
    (including `semantic_vad_heads.safetensors`, which the end-of-turn heads need)."""
    if Path(ref).expanduser().is_dir():
        return str(Path(ref).expanduser())
    from huggingface_hub import snapshot_download
    return snapshot_download(ref)


INT4_KEEP = ("lm_head", "multi_modal_projector", "ada_rms_norm")  # stay bf16


def int4_config():
    """int4 weight-only, HQQ scales, group 128, tinygemm layout (fast at batch 1; needs no extra
    packages).  The measured slim profile: see README."""
    from torchao.quantization import Int4WeightOnlyConfig
    return Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d",
                                int4_choose_qparams_algorithm="hqq")


def load_model(checkpoint: str, *, quant: str = "none", device: str = "cuda"):
    """Load the model with the ASR-only config (upstream loader bug: the VAD heads live in a second
    file; engine.load_vad_heads() reads them).  For int4, weights are loaded on the CPU and moved +
    quantized one transformer layer at a time, so GPU memory never holds the full bf16 model
    (peak ~= int4 model + one bf16 layer instead of ~8 GB)."""
    import torch
    from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
        Audio8ASRInfiniteForConditionalGeneration,
    )
    cfg = Audio8ASRInfiniteConfig.from_pretrained(checkpoint)
    cfg.semantic_vad_horizons_seconds = None
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        checkpoint, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)  # on CPU
    if quant == "none":
        return model.to(device).eval()
    if quant != "int4hqq":
        raise ValueError(f"unknown quant {quant!r}")
    from torchao.quantization import quantize_
    qcfg = int4_config()
    is_q = lambda m, f: isinstance(m, torch.nn.Linear) and not any(k in f for k in INT4_KEEP)
    for layer in list(model.audio_tower.layers) + list(model.language_model.model.layers):
        layer.to(device)
        quantize_(layer, qcfg, filter_fn=is_q)
        torch.cuda.empty_cache()
    model.to(device)  # everything else (embeddings, lm_head, norms, conv stem, projector) stays bf16
    return model.eval()
