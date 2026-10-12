"""Checkpoint metadatasını doğrulayan, framework bağımsız yükleyici.

Model ağırlıklarının nasıl yükleneceği eğitim kodunun sorumluluğundadır.
Çekirdek yalnızca manifest'i JSON olarak okur; otomatik pickle yüklemez.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import CheckpointManifest


class CheckpointError(ValueError):
    pass


def load_manifest(checkpoint_dir: str | Path) -> CheckpointManifest:
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError(f"Checkpoint manifest'i bulunamadı: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"manifest.json geçerli JSON değil: {exc.msg}") from exc

    required = {"architecture", "weights", "tokenizer", "context_window", "vocab_size"}
    missing = sorted(required - raw.keys())
    if missing:
        raise CheckpointError(f"Manifest zorunlu alanları içermiyor: {', '.join(missing)}")

    if not isinstance(raw["architecture"], str) or not raw["architecture"].strip():
        raise CheckpointError("architecture boş olmayan bir metin olmalıdır.")
    if not isinstance(raw["context_window"], int) or raw["context_window"] < 1:
        raise CheckpointError("context_window pozitif bir tamsayı olmalıdır.")
    if not isinstance(raw["vocab_size"], int) or raw["vocab_size"] < 2:
        raise CheckpointError("vocab_size en az 2 olmalıdır.")

    def contained_file(value: object, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise CheckpointError(f"{label} geçerli bir dosya yolu olmalıdır.")
        path = (root / value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CheckpointError(f"{label} checkpoint klasörünün dışına çıkamaz.") from exc
        if not path.is_file():
            raise CheckpointError(f"{label} bulunamadı: {path}")
        return path

    return CheckpointManifest(
        architecture=raw["architecture"],
        weights_path=contained_file(raw["weights"], "weights"),
        tokenizer_path=contained_file(raw["tokenizer"], "tokenizer"),
        context_window=raw["context_window"],
        vocab_size=raw["vocab_size"],
        format_version=raw.get("format_version", 1),
    )
