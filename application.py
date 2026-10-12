"""HyperAI'nin kendi modelini checkpoint'ten başlatan uygulama katmanı."""

from __future__ import annotations

from pathlib import Path

from .checkpoint import load_manifest
from .registry import ModelRegistry
from .runtime import HyperAIRuntime
from .torch_checkpoint import load_hyperai_torch_v1


def create_runtime(checkpoint_dir: str | Path, device: str = "auto") -> HyperAIRuntime:
    """Yalnızca HyperAI'nin yerel ağırlıklarını kullanarak runtime oluşturur."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device auto, cpu veya cuda olmalıdır.")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(checkpoint_dir)
    registry = ModelRegistry()
    registry.register("hyperai-torch-v1", lambda manifest: load_hyperai_torch_v1(manifest, device))
    model, tokenizer = registry.load(manifest)
    return HyperAIRuntime(model, tokenizer, context_window=manifest.context_window)
