"""Kendi model mimarilerini güvenli ve açık biçimde kaydetmek için kayıt defteri."""

from __future__ import annotations

from collections.abc import Callable

from .contracts import AutoregressiveModel, CheckpointManifest, Tokenizer

ModelLoader = Callable[[CheckpointManifest], tuple[AutoregressiveModel, Tokenizer]]


class ModelRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, ModelLoader] = {}

    def register(self, architecture: str, loader: ModelLoader) -> None:
        if not architecture.strip():
            raise ValueError("architecture boş olamaz.")
        if architecture in self._loaders:
            raise ValueError(f"'{architecture}' mimarisi zaten kayıtlı.")
        self._loaders[architecture] = loader

    def load(self, manifest: CheckpointManifest) -> tuple[AutoregressiveModel, Tokenizer]:
        try:
            loader = self._loaders[manifest.architecture]
        except KeyError as exc:
            known = ", ".join(sorted(self._loaders)) or "henüz kayıtlı mimari yok"
            raise KeyError(f"'{manifest.architecture}' için yükleyici yok. Kayıtlı: {known}") from exc
        return loader(manifest)
