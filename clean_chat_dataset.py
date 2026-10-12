"""HyperAI sohbet ince ayarı için JSONL kaynaklarını güvenli biçimde birleştirir.

Girdi kaydı: {"user": "...", "assistant": "..."}
Çıktı, şüpheli kişisel veri veya kalitesiz uzun belge artıkları içermeyen aynı
şemadaki JSONL'dir. Bu bir PII keşif aracı değil, eğitim öncesi ihtiyatlı bir
filtre: şüpheli kayıtlar redakte edilmez, tümüyle dışarıda bırakılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# Denetim aracındaki ihtiyatlı telefon kalıbıyla aynı tutulur. Bazı sayı/tarih
# benzeri yanlış pozitifleri de dışarıda bırakabilir; sohbet ince ayarında
# mahremiyet, bu küçük veri kaybından daha önemlidir.
PHONE = re.compile(r"(?<!\d)(?:\+?90\s?)?(?:0?\d{3}[\s.-]?)?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
SECRET = re.compile(r"\b(?:api[_ -]?key|secret|password|parola|şifre)\s*[:=]\s*\S+", re.IGNORECASE)
URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def normalise(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def reject_reason(user: str, assistant: str, max_user_chars: int, max_assistant_chars: int) -> str | None:
    if not user or not assistant:
        return "empty"
    if len(user) > max_user_chars:
        return "user_too_long"
    if len(assistant) > max_assistant_chars:
        return "assistant_too_long"
    combined = user + "\n" + assistant
    if EMAIL.search(combined):
        return "email"
    if PHONE.search(combined):
        return "phone"
    if SECRET.search(combined):
        return "secret"
    if URL.search(combined):
        return "url"
    answer_lower = assistant.casefold()
    if "kaynakça" in answer_lower or "dış bağlantılar" in answer_lower:
        return "encyclopedia_artifact"
    if "as an ai language model" in answer_lower or "bir yapay zeka dil modeli olarak" in answer_lower:
        return "model_disclaimer"
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Sohbet JSONL kaynaklarını filtreleyip tek temiz fine-tune korpusuna dönüştürür.")
    parser.add_argument("--input", action="append", required=True, help="user/assistant JSONL kaynağı; birden çok kez verilebilir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", help="istatistik JSON dosyası; varsayılan output yanında .report.json")
    parser.add_argument("--max-user-chars", type=int, default=600)
    parser.add_argument("--max-assistant-chars", type=int, default=1200)
    args = parser.parse_args()
    if args.max_user_chars < 1 or args.max_assistant_chars < 1:
        parser.error("Karakter sınırları pozitif olmalıdır.")

    output = Path(args.output)
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}. Bilerek silip yeniden çalıştırın veya yeni ad verin.")
    sources = [Path(value) for value in args.input]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("Kaynak bulunamadı: " + ", ".join(missing))

    seen: set[tuple[str, str]] = set()
    reasons: Counter[str] = Counter()
    raw, accepted, invalid = 0, 0, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("x", encoding="utf-8", newline="\n") as target:
        for source in sources:
            source_rows = source_ok = 0
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    raw += 1; source_rows += 1
                    try:
                        item = json.loads(line)
                        user, assistant = normalise(item["user"]), normalise(item["assistant"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        invalid += 1; reasons["invalid_schema"] += 1
                        continue
                    reason = reject_reason(user, assistant, args.max_user_chars, args.max_assistant_chars)
                    if reason:
                        reasons[reason] += 1
                        continue
                    key = (user.casefold(), assistant.casefold())
                    if key in seen:
                        reasons["duplicate"] += 1
                        continue
                    seen.add(key)
                    rendered = json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n"
                    target.write(rendered); digest.update(rendered.encode("utf-8"))
                    accepted += 1; source_ok += 1
            print(f"{source}: {source_ok:,}/{source_rows:,} kayıt kabul edildi", flush=True)
    temporary.replace(output)
    report = {
        "sources": [str(path) for path in sources], "raw_records": raw, "accepted_records": accepted,
        "rejected_records": raw - accepted, "invalid_records": invalid, "rejection_reasons": dict(sorted(reasons.items())),
        "max_user_chars": args.max_user_chars, "max_assistant_chars": args.max_assistant_chars,
        "output_sha256": digest.hexdigest(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Temiz sohbet verisi: {output} ({accepted:,} kayıt)")
    print(f"Rapor: {report_path}")
    print("Silme nedenleri: " + ", ".join(f"{name}={count:,}" for name, count in sorted(reasons.items())))


if __name__ == "__main__":
    main()
