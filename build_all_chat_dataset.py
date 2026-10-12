"""Elindeki tüm sohbet/soru-cevap kaynaklarını tek güvenli JSONL havuzunda birleştir.

Düz metin kaynakları (Vikipedi, filtreli web, TinyStories) ön-eğitim içindir;
bu araç yalnızca kullanıcı-asistan çiftine dönüştürülebilen kaynakları işler.
Her kaynak raporda ayrı sayılır. Kayıt bazında PII, bozuk şema, aşırı uzunluk,
URL ve tekrar filtrelenir; bir kaynak topluca dışlanmaz.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

from clean_chat_dataset import normalise, reject_reason


def records(path: Path) -> Iterator[dict]:
    """JSONL ve tek JSON dizi biçimlerini kabul eder."""
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            value = json.load(handle)
            if not isinstance(value, list):
                raise ValueError(f"{path}: JSON dizi bekleniyordu.")
            yield from (item for item in value if isinstance(item, dict))
            return
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def pairs(item: dict) -> Iterator[tuple[str, str]]:
    """Bilinen yerel şemalardan her insan→asistan turunu çıkarır."""
    if isinstance(item.get("user"), str) and isinstance(item.get("assistant"), str):
        yield item["user"], item["assistant"]
    if isinstance(item.get("input"), str) and isinstance(item.get("output"), str):
        yield item["input"], item["output"]
    if isinstance(item.get("question"), str) and isinstance(item.get("answer"), str):
        yield item["question"], item["answer"]
    conversation = item.get("conversations")
    if not isinstance(conversation, list):
        return
    pending: str | None = None
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role, text = turn.get("from"), turn.get("value")
        if not isinstance(text, str):
            continue
        if role in {"human", "user"}:
            pending = text
        elif role in {"gpt", "assistant"} and pending is not None:
            yield pending, text
            pending = None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tüm yerel sohbet ve QA kaynaklarını birleştirip temizler.")
    parser.add_argument("--input", action="append", required=True, help="Sohbet/QA JSON veya JSONL kaynağı; birden fazla verilebilir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", help="Varsayılan: çıktı yanında .report.json")
    parser.add_argument("--max-user-chars", type=int, default=600)
    parser.add_argument("--max-assistant-chars", type=int, default=1200)
    args = parser.parse_args()
    if args.max_user_chars < 1 or args.max_assistant_chars < 1:
        parser.error("Karakter sınırları pozitif olmalıdır.")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}")
    sources = [Path(raw) for raw in args.input]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("Kaynak bulunamadı: " + ", ".join(missing))
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    seen: set[tuple[str, str]] = set()
    totals, reasons, by_source = Counter(), Counter(), {}
    with temporary.open("x", encoding="utf-8", newline="\n") as target:
        for source in sources:
            stats = Counter()
            for item in records(source):
                stats["records"] += 1
                for user, assistant in pairs(item):
                    stats["pairs"] += 1
                    user, assistant = normalise(user), normalise(assistant)
                    reason = reject_reason(user, assistant, args.max_user_chars, args.max_assistant_chars)
                    if reason:
                        stats[reason] += 1; reasons[reason] += 1
                        continue
                    key = (user.casefold(), assistant.casefold())
                    if key in seen:
                        stats["duplicate"] += 1; reasons["duplicate"] += 1
                        continue
                    seen.add(key)
                    target.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1; totals["accepted"] += 1
            by_source[str(source)] = dict(stats)
            print(f"{source}: {stats['accepted']:,}/{stats['pairs']:,} sohbet çifti kabul edildi")
    temporary.replace(output)
    report = {
        "purpose": "all local dialogue and QA sources, cleaned record-by-record",
        "accepted_records": totals["accepted"], "sources": by_source,
        "rejection_reasons": dict(sorted(reasons.items())),
        "max_user_chars": args.max_user_chars, "max_assistant_chars": args.max_assistant_chars,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Birleşik sohbet havuzu: {output} ({totals['accepted']:,} çift)")
    print(f"Rapor: {report_path}")


if __name__ == "__main__":
    main()
