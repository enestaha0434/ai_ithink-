"""HyperAI sohbet verisini hazırlamak için bağımlılıksız yardımcılar."""

from __future__ import annotations

import json
from pathlib import Path


class CorpusError(ValueError):
    pass


def normalize_corpus(text: str, *, deduplicate: bool = False) -> str:
    """Boş satırları kaldırır; Unicode kod noktalarını ve tekrarları korur."""
    if not isinstance(text, str):
        raise TypeError("Korpus metin olmalıdır.")
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or (deduplicate and line in seen):
            continue
        lines.append(line)
        seen.add(line)
    if not lines:
        raise CorpusError("Korpusta kullanılabilir satır yok.")
    return "\n".join(lines) + "\n"


def prepare_corpus(input_path: str | Path, output_path: str | Path, *, deduplicate: bool = False) -> dict[str, int]:
    source = Path(input_path)
    target = Path(output_path)
    if not source.is_file():
        raise CorpusError(f"Kaynak korpus bulunamadı: {source}")
    prepared = normalize_corpus(source.read_text(encoding="utf-8"), deduplicate=deduplicate)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prepared, encoding="utf-8", newline="\n")
    return {"characters": len(prepared), "lines": len(prepared.splitlines())}


def format_chat_prompt(user_message: str) -> str:
    """Inference ve eğitimde ortak kullanılan HyperAI tur başlangıcı."""
    return f"Sen: {user_message}\nHyperAI:"


def format_chat_example(user_message: str, assistant_message: str) -> str:
    """Bir sohbet çiftini metin devam ettirme modeli için tek eğitim örneğine çevirir."""
    return f"{format_chat_prompt(user_message)} {assistant_message}"


def _nonempty_lines(source: Path) -> list[str]:
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_chat_pairs(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    """Sırayla yazılmış kullanıcı/asistan satırlarını JSONL sohbet verisine çevirir.

    Aynı cümle farklı turlarda tekrar edebilir; hiçbir konuşma satırı deduplicate
    edilmez ve Unicode normalizasyonu uygulanmaz.
    """
    source, target = Path(input_path), Path(output_path)
    if not source.is_file():
        raise CorpusError(f"Kaynak korpus bulunamadı: {source}")
    if source.suffix.lower() == ".jsonl":
        # JSONL biçimi newline içeren kullanıcı veya asistan metinlerini de taşır.
        pairs = [{"user": user, "assistant": assistant} for user, assistant in load_chat_pairs(source)]
    else:
        lines = _nonempty_lines(source)
        if len(lines) < 2 or len(lines) % 2:
            raise CorpusError("Sohbet verisi sırayla kullanıcı/asistan olacak şekilde çift sayıda satır içermelidir.")
        pairs = [{"user": lines[index], "assistant": lines[index + 1]} for index in range(0, len(lines), 2)]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8", newline="\n"
    )
    return {"examples": len(pairs), "characters": sum(len(pair["user"]) + len(pair["assistant"]) for pair in pairs)}


def load_chat_pairs(path: str | Path) -> list[tuple[str, str]]:
    """JSONL sohbet verisini doğrular ve kullanıcı/asistan çiftlerini yükler."""
    source = Path(path)
    if not source.is_file():
        raise CorpusError(f"Sohbet verisi bulunamadı: {source}")
    examples: list[tuple[str, str]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{source}:{line_number} geçerli JSONL değil.") from exc
        if not isinstance(item, dict) or not isinstance(item.get("user"), str) or not isinstance(item.get("assistant"), str):
            raise CorpusError(f"{source}:{line_number} user ve assistant metin alanlarını içermelidir.")
        if not item["user"].strip() or not item["assistant"].strip():
            raise CorpusError(f"{source}:{line_number} boş sohbet turu içeremez.")
        examples.append((item["user"], item["assistant"]))
    if len(examples) < 2:
        raise CorpusError("Train/validation ayrımı için en az iki sohbet örneği gerekir.")
    return examples
