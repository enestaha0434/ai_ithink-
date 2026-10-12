"""Bağımlılıksız, yerel belge arama (RAG) katmanı.

Bu modül bir LLM ya da vektör veritabanı indirmez. Kullanıcının açıkça verdiği
metin/JSONL belgelerini küçük parçalara ayırır ve BM25 ile arar. Bulduğu metin
talimattan ziyade *alıntı kaynak* olarak değerlendirilmelidir.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", re.UNICODE)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_CHUNK_CHARS = 900


@dataclass(frozen=True)
class RetrievedPassage:
    source: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class _Chunk:
    source: str
    title: str
    text: str
    terms: Counter[str]


def _terms(text: str) -> list[str]:
    return [item.casefold() for item in TOKEN.findall(text)]


def _chunks(source: str, title: str, text: str) -> Iterable[_Chunk]:
    clean = " ".join(text.split())
    for offset in range(0, len(clean), MAX_CHUNK_CHARS):
        piece = clean[offset : offset + MAX_CHUNK_CHARS].strip()
        if len(piece) >= 20:
            yield _Chunk(source, title, piece, Counter(_terms(piece)))


def _read_jsonl(path: Path) -> Iterable[tuple[str, str]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} geçerli JSONL değil.") from exc
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or item.get("assistant")
        if isinstance(text, str):
            title = item.get("title") if isinstance(item.get("title"), str) else f"{path.name}:{line_number}"
            yield title, text


class LocalKnowledgeBase:
    """Kullanıcının verdiği belgelerden oluşan küçük BM25 arama indeksi."""

    def __init__(self, chunks: Iterable[_Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Bilgi tabanında aranabilir metin bulunamadı.")
        self.document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            self.document_frequency.update(chunk.terms.keys())
        self.average_length = sum(sum(item.terms.values()) for item in self.chunks) / len(self.chunks)

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "LocalKnowledgeBase":
        chunks: list[_Chunk] = []
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError(f"Bilgi belgesi bulunamadı: {path}")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"Bilgi belgesi 64 MiB sınırını aşıyor: {path}")
            if path.suffix.lower() == ".jsonl":
                records = _read_jsonl(path)
            elif path.suffix.lower() in {".txt", ".md"}:
                records = [(path.stem, path.read_text(encoding="utf-8"))]
            else:
                raise ValueError(f"Yalnızca .txt, .md veya .jsonl bilgi belgesi desteklenir: {path}")
            for title, text in records:
                chunks.extend(_chunks(str(path), title, text))
        return cls(chunks)

    def search(self, query: str, limit: int = 3) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if not 1 <= limit <= 10:
            raise ValueError("Arama sonuç sınırı 1..10 arasında olmalıdır.")
        query_terms = _terms(query)
        if not query_terms:
            return []
        total = len(self.chunks)
        scored: list[RetrievedPassage] = []
        for chunk in self.chunks:
            length = sum(chunk.terms.values())
            score = 0.0
            for term in set(query_terms):
                frequency = chunk.terms.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (total - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                score += inverse_frequency * frequency * 2.2 / (frequency + 1.2 * (1 - 0.75 + 0.75 * length / self.average_length))
            if score:
                scored.append(RetrievedPassage(chunk.source, chunk.title, chunk.text, score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]
