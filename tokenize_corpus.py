"""Birleşik veri paketini tokenizer'a çevirir: 1) hızlı BPE eğitir, 2) tüm kaynakları uint16 .bin dosyalarına yazar.

Kullanım (proje klasöründe):
  py tokenize_corpus.py --data datasets\\unified-v2 --output datasets\\tokens-v1 --vocab-size 16384

Kesilirse aynı komutu tekrar çalıştır: biten parçalar atlanır.
Her belge  [BOS] metin [EOS]  olarak yazılır; sohbet çiftleri "Sen: ...\\nHyperAI: ..." metnine çevrilir.
Belgelerin yaklaşık %1'i (içerik özetine göre, sabit) doğrulama dosyasına ayrılır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

from hyperai_core.bpe_fast import FastBPETokenizer
from hyperai_core.data import format_chat_example

_TOKENIZER: FastBPETokenizer | None = None


def log(message: str) -> None:
    print(message, flush=True)


def iter_rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def row_text(row: dict) -> tuple[str, str] | None:
    if "text" in row and isinstance(row["text"], str):
        return row.get("source", "pretrain"), row["text"]
    if isinstance(row.get("user"), str) and isinstance(row.get("assistant"), str):
        return "chat", format_chat_example(row["user"], row["assistant"])
    return None


def is_validation(text: str, percent: float) -> bool:
    value = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big")
    return value % 10_000 < percent * 100


def tokenizer_sample(shards: list[Path], quota_mb: float, seed: int):
    """Kaynak başına rastgele parçalardan, parça içinde de düzgün aralıklarla örnek verir."""
    rng = random.Random(seed)
    by_source: dict[str, list[Path]] = defaultdict(list)
    for shard in shards:
        first = next(iter_rows(shard), None)
        parsed = row_text(first) if first else None
        if parsed:
            by_source[parsed[0]].append(shard)
    quota = int(quota_mb * 1024 * 1024)
    for source, group in sorted(by_source.items()):
        rng.shuffle(group)
        used = 0
        for shard in group:
            if used >= quota:
                break
            probability = min(1.0, (quota - used) / max(1, shard.stat().st_size))
            for row in iter_rows(shard):
                parsed = row_text(row)
                if parsed and rng.random() < probability:
                    used += len(parsed[1])
                    yield parsed[1]
        log(f"  tokenizer örneği: {source}: ~{used / 1024 / 1024:.0f} MB")


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    _TOKENIZER = FastBPETokenizer.load(tokenizer_path)


def tokenize_shard(job: tuple[str, str, float]) -> dict:
    shard_path, out_root, val_percent = job
    shard, out = Path(shard_path), Path(out_root)
    done_path = out / "_done" / (shard.parent.name + "-" + shard.stem + ".json")
    if done_path.is_file():
        return json.loads(done_path.read_text(encoding="utf-8"))
    assert _TOKENIZER is not None
    bos, eos = _TOKENIZER.BOS_ID, _TOKENIZER.EOS_ID
    buffers: dict[tuple[str, str], array] = {}
    docs: dict[tuple[str, str], int] = defaultdict(int)
    for row in iter_rows(shard):
        parsed = row_text(row)
        if not parsed:
            continue
        source, text = parsed
        split = "val" if is_validation(text, val_percent) else "train"
        buffer = buffers.setdefault((source, split), array("H"))
        buffer.append(bos)
        buffer.extend(_TOKENIZER.encode(text))
        buffer.append(eos)
        docs[(source, split)] += 1
    files: dict[str, dict] = {}
    for (source, split), buffer in buffers.items():
        target = out / source / f"{shard.parent.name}-{shard.stem}.{split}.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".bin.tmp")
        with temporary.open("wb") as handle:
            buffer.tofile(handle)
        os.replace(temporary, target)
        files[str(target.relative_to(out)).replace("\\", "/")] = {
            "source": source, "split": split, "tokens": len(buffer), "documents": docs[(source, split)],
        }
    result = {"shard": str(shard), "files": files}
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="HyperAI veri paketini token dosyalarına çevir.")
    parser.add_argument("--data", required=True, help="unified-vN klasörü (pretrain/ ve chat/ içerir)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-mb", type=float, default=150, help="tokenizer eğitimi için kaynak başına örnek (MB)")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--val-percent", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.vocab_size > 65_535:
        parser.error("uint16 kullanıldığı için --vocab-size 65535'ten küçük olmalı.")

    data, out = Path(args.data), Path(args.output)
    shards = sorted((data / "pretrain").glob("*.jsonl")) + sorted((data / "chat").glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"{data} içinde pretrain/ veya chat/ JSONL parçası yok.")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / "tokenizer.json"

    if tokenizer_path.is_file():
        log(f"Tokenizer zaten var, yeniden eğitilmiyor: {tokenizer_path}")
    else:
        log(f"Tokenizer eğitiliyor (vocab={args.vocab_size}) ...")
        started = time.time()
        tokenizer = FastBPETokenizer.train(tokenizer_sample(shards, args.tokenizer_mb, args.seed), args.vocab_size)
        tokenizer.save(tokenizer_path)
        log(f"Tokenizer hazır: vocab={tokenizer.vocab_size}, {time.time() - started:.0f} sn -> {tokenizer_path}")

    jobs = [(str(shard), str(out), args.val_percent) for shard in shards]
    log(f"{len(jobs)} parça token'lanacak, {args.workers} işlemci kullanılıyor ...")
    results, started = [], time.time()
    with mp.Pool(args.workers, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        for number, result in enumerate(pool.imap_unordered(tokenize_shard, jobs), 1):
            results.append(result)
            tokens = sum(info["tokens"] for info in result["files"].values())
            elapsed = time.time() - started
            log(f"  [{number}/{len(jobs)}] {Path(result['shard']).name}: {tokens:,} token  (geçen {elapsed:.0f} sn, tahmini kalan {elapsed / number * (len(jobs) - number):.0f} sn)")

    manifest: dict = {"tokenizer": "tokenizer.json", "vocab_size": FastBPETokenizer.load(tokenizer_path).vocab_size, "sources": {}}
    for result in results:
        for relative, info in result["files"].items():
            entry = manifest["sources"].setdefault(info["source"], {"train": [], "val": [], "train_tokens": 0, "val_tokens": 0})
            entry[info["split"]].append({"file": relative, "tokens": info["tokens"]})
            entry[info["split"] + "_tokens"] += info["tokens"]
    for entry in manifest["sources"].values():
        entry["train"].sort(key=lambda item: item["file"]); entry["val"].sort(key=lambda item: item["file"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\nToken özeti:")
    for source, entry in sorted(manifest["sources"].items()):
        log(f"  {source:20s} eğitim={entry['train_tokens']:>14,}  doğrulama={entry['val_tokens']:>12,}")
    log(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
