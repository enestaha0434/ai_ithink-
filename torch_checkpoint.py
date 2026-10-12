"""Kendi PyTorch HyperAI modelini manifestten yüklemek için adaptör."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .contracts import CheckpointManifest
from .model import HyperAILanguageModel, HyperAIModelConfig
from .tokenizer import load_tokenizer

MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024


def load_hyperai_torch_v1(manifest: CheckpointManifest, device: str = "cpu"):
    """``hyperai-torch-v1`` checkpoint'ini yerelde yükler.

    Ağırlık dosyası eğitim sürecinin ürettiği saf ``state_dict`` olmalıdır.
    """
    if manifest.architecture != "hyperai-torch-v1":
        raise ValueError(f"Bu yükleyici {manifest.architecture!r} mimarisi için kullanılamaz.")
    config_path = manifest.weights_path.parent / "model_config.json"
    if not config_path.is_file():
        raise ValueError(f"Model config'i bulunamadı: {config_path}")
    config = HyperAIModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if config.vocab_size != manifest.vocab_size or config.context_window != manifest.context_window:
        raise ValueError("manifest ile model_config.json sözlük/bağlam bilgisi uyuşmuyor.")
    tokenizer = load_tokenizer(manifest.tokenizer_path)
    tokenizer_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.VOCAB_SIZE
    if tokenizer_vocab_size != config.vocab_size:
        raise ValueError("Tokenizer ve model sözlük boyutları uyuşmuyor.")

    if device not in {"cpu", "cuda"}:
        raise ValueError("device yalnızca 'cpu' veya 'cuda' olabilir.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi fakat kullanılabilir PyTorch CUDA cihazı yok.")
    if manifest.weights_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Ağırlık dosyası güvenli yükleme sınırını (2 GiB) aşıyor.")
    state_dict = torch.load(manifest.weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError("Ağırlık dosyası bir state_dict içermelidir.")
    model = HyperAILanguageModel(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, tokenizer
