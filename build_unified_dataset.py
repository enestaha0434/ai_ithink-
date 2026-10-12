"""Tüm yerel kaynakları filtreleyip tek sürümlü HyperAI veri paketine dönüştürür (v2).

Çıktı:  <output>/pretrain/*.jsonl  -> {"text", "language", "source"}   (düz metin, ön eğitim)
        <output>/chat/*.jsonl      -> {"user", "assistant"}            (sohbet çiftleri)
        <output>/manifest.json     -> kaynak başına belge ve karakter sayıları

v1'e göre düzeltmeler:
  * 'text' sütunu olmayan parquet dosyaları atlanır (v1 bunlarda çöküyordu).
  * Tekrar kontrolü metnin kendisini değil 64 bit özetini tutar (v1 on GB RAM isterdi).
  * Uzun belgeler atılmaz, paragraf/cümle sınırlarından parçalanır.
  * Sohbet çiftleri için ayrı, hafif filtre: "Merhaba" gibi kısa mesajlar ve satır sonları korunur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SPAM = re.compile(r"\b(?:bahis|casino|slot|bet\s*sitesi|bonus|kupon|takipçi\s*kasma|follower\s*satın|porn|escort)\b", re.I)
BAD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACES = re.compile(r"[ \t\r\f\v\u00a0]+")
NEWLINES = re.compile(r"\s*\n\s*")
SENTENCE_END = re.compile(r"(?<=[.!?]) +")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def normalize(text: str) -> str:
    text = BAD_CONTROL.sub("", text)
    text = SPACES.sub(" ", text)
    return NEWLINES.sub("\n", text).strip()


def looks_bad(text: str) -> bool:
    # Bozuk web sayfalarında tek karakter tekrarları ve bağlantı yığınları yaygın.
    return bool(SPAM.search(text)) or len(set(text)) < 12 or text.count("http") > 2 or text.count("|") > 12


def split_long(text: str, max_chars: int):
    """Uzun metni paragraf (olmazsa cümle, o da olmazsa sert) sınırlarından max_chars altı parçalara böler."""
    if len(text) <= max_chars:
        yield text
        return
    chunk: list[str] = []
    size = 0
    for paragraph in text.split("\n"):
        pieces = [paragraph] if len(paragraph) <= max_chars else SENTENCE_END.split(paragraph)
        for piece in pieces:
            while len(piece) > max_chars:
                if chunk:
                    yield "\n".join(chunk)
                    chunk, size = [], 0
                yield piece[:max_chars]
                piece = piece[max_chars:]
            if chunk and size + len(piece) + 1 > max_chars:
                yield "\n".join(chunk)
                chunk, size = [], 0
            chunk.append(piece)
            size += len(piece) + 1
    if chunk:
        yield "\n".join(chunk)


def pretrain_chunks(raw: object, min_chars: int, max_chars: int):
    if not isinstance(raw, str):
        return
    text = normalize(raw)
    for chunk in split_long(text, max_chars):
        chunk = chunk.strip()
        if len(chunk) >= min_chars and not looks_bad(chunk):
            yield chunk


def clean_chat(raw: object, excludes: list[str], max_chars: int = 2048) -> str | None:
    if not isinstance(raw, str):
        return None
    text = SPACES.sub(" ", BAD_CONTROL.sub("", raw))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not 1 <= len(text) <= max_chars or SPAM.search(text) or text.count("http") > 2:
        return None
    lowered = text.lower()
    if any(word in lowered for word in excludes):
        return None
    return text


class Shards:
    def __init__(self, root: Path, stem: str, size: int = 100_000) -> None:
        self.root, self.stem, self.size, self.index, self.count, self.handle = root, stem, size, 0, 0, None
        root.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        if self.handle is None or self.count % self.size == 0:
            if self.handle:
                self.handle.close()
            path = self.root / f"{self.stem}-{self.index:05}.jsonl"
            self.index += 1
            self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def has_text_column(path: Path) -> tuple[bool, list[str]]:
    names = pq.ParquetFile(path).schema_arrow.names
    return "text" in names, names


def parquet_texts(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["text"]):
        yield from batch.column(0).to_pylist()


def story_texts(path: Path):
    parts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if "<|endoftext|>" in line:
                before, _, after = line.partition("<|endoftext|>")
                parts.append(before)
                yield "".join(parts)
                parts = [after]
            else:
                parts.append(line)
    if parts:
        yield "".join(parts)


def chat_pairs(path: Path):
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("user"), str) and isinstance(item.get("assistant"), str):
                yield item["user"], item["assistant"]


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAI birleşik veri paketini üretir.")
    parser.add_argument("--data", default="datasets", help="ham veri klasörü")
    parser.add_argument("--output", default="datasets/unified-v2")
    parser.add_argument("--include-english-stories", action="store_true")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=8000, help="bundan uzun belgeler parçalanır")
    parser.add_argument("--chat-exclude", default="", help="virgülle ayrılmış ifadeler; geçen sohbet çiftleri atlanır")
    parser.add_argument("--limit-docs", type=int, default=0, help="test için: kaynak başına en fazla bu kadar belge")
    args = parser.parse_args()

    raw, output = Path(args.data), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Çıktı dolu: {output}. Yeni bir sürüm adı seçin (ör. unified-v3) veya klasörü silin.")
    excludes = [w.strip().lower() for w in args.chat_exclude.split(",") if w.strip()]
    pre, chat = Shards(output / "pretrain", "text"), Shards(output / "chat", "pairs")
    docs, chars, seen = Counter(), Counter(), set()
    skipped_files: list[str] = []
    started = time.time()

    def add_pretrain(source: str, language: str, texts, min_chars: int) -> None:
        count = 0
        for number, raw_text in enumerate(texts, 1):
            for chunk in pretrain_chunks(raw_text, min_chars, args.max_chars):
                key = digest(chunk)
                if key in seen:
                    continue
                seen.add(key)
                pre.write({"text": chunk, "language": language, "source": source})
                docs[source] += 1
                chars[source] += len(chunk)
            count += 1
            if number % 200_000 == 0:
                log(f"  {source}: {number:,} belge okundu, {docs[source]:,} parça yazıldı ({time.time() - started:.0f} sn)")
            if args.limit_docs and count >= args.limit_docs:
                break

    def parquet_group(source: str, pattern: str) -> None:
        for path in sorted(raw.glob(pattern)):
            ok, names = has_text_column(path)
            if not ok:
                skipped_files.append(f"{path.name} (sütunlar: {', '.join(names)})")
                log(f"  ATLANDI: {path.name} -> 'text' sütunu yok, sütunlar: {names}")
                continue
            log(f"{source}: {path.name}")
            add_pretrain(source, "tr", parquet_texts(path), args.min_chars)

    parquet_group("turkish_wikipedia", "train-*.parquet")
    parquet_group("filtered_web", "tr_part_*.parquet")
    if args.include_english_stories:
        stories = raw / "TinyStoriesV2-GPT4-train.txt"
        if stories.is_file():
            log("tinystories")
            add_pretrain("tinystories", "en", story_texts(stories), 80)
        else:
            log("  uyarı: TinyStories dosyası bulunamadı")

    seen_pairs: set[int] = set()
    for candidate in (raw / "sohbet.jsonl", raw / "hf-turkish-chat-v1.jsonl", raw / "processed" / "hf-turkish-chat-v1.jsonl"):
        if not candidate.is_file():
            continue
        log(f"chat: {candidate}")
        for user, assistant in chat_pairs(candidate):
            user, assistant = clean_chat(user, excludes), clean_chat(assistant, excludes)
            if not user or not assistant:
                continue
            key = digest(user + "\x00" + assistant)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            chat.write({"user": user, "assistant": assistant})
            docs["chat"] += 1
            chars["chat"] += len(user) + len(assistant)
    pre.close(); chat.close()

    manifest = {
        "version": output.name, "records": dict(docs), "characters": dict(chars),
        "pretrain_records": pre.count, "chat_records": chat.count,
        "skipped_files": skipped_files, "english_stories": args.include_english_stories,
        "min_chars": args.min_chars, "max_chars": args.max_chars, "seconds": round(time.time() - started),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
