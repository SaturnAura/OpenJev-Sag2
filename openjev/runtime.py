"""Loading helpers shared by the CLI, server, and retrieval experiments."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import QwenJevConfig
from .engine import QwenJevLite, load_backbone

MODEL_PRESETS = {
    "9b": r"C:\qwen3.5-9B",
}

CHECKPOINT_PRESETS = {
    "9b": "models/9b-mix/readout.pt",
}


def resolve_model(value: str) -> str:
    return MODEL_PRESETS.get(value.casefold(), value)


def resolve_checkpoint(value: str | None, model: str) -> str | None:
    if value:
        return str(Path(value).expanduser().resolve())
    preset = CHECKPOINT_PRESETS.get(model.casefold())
    return str(Path(preset).resolve()) if preset else None


def load_engine(
    model: str,
    checkpoint: str | None = None,
    *,
    device: str = "cuda:0",
    batch_size: int = 8,
    max_batch_tokens: int = 4096,
) -> QwenJevLite:
    """Load one text-only backbone and its trained OpenJev ``slot_head``."""

    model_path = resolve_model(model)
    checkpoint_path = resolve_checkpoint(checkpoint, model)
    if checkpoint_path and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"OpenJev head not found: {checkpoint_path}")
    config = QwenJevConfig(
        model_path=model_path,
        device=device,
        text_only=True,
        readout="slot_head" if checkpoint_path else "auto",
        readout_checkpoint=checkpoint_path,
        branch_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        state_cache_size=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    backbone = load_backbone(
        model_path,
        dtype=torch.bfloat16,
        device=device,
        text_only=True,
    )
    return QwenJevLite(backbone, tokenizer, config)


def release_engine(engine: QwenJevLite | None) -> None:
    """Release GPU and host memory before loading the next backbone."""

    if engine is None:
        return
    engine.clear_state_cache()
    engine.model = None
    engine.tokenizer = None
    engine.readout = None
    engine._auto_readouts = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

