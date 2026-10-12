"""İncelenmiş yerel öğretmen çıktılarından izlenebilir eğitim alt kümesi üretir."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from clean_chat_dataset import normalise, reject_reason


REJECT_USER = re.compile(r"\byakın bir kavram\b", re.IGNORECASE)
REJECT_ANSWER = re.compile(
    r"\bhyperai\b|\böğretmen verisi\b|\bdoğal türkçe diyaloglar oluştur\b|"
    r"\bboşluk karakterlerinden kaçın\b",
    re.IGNORECASE,
)


def rejection_reason(user: str, assistant: str) -> str | None:
    """Sadece gözle doğrulanmış sistem sızıntısı/belirsiz şablonları çıkarır."""
    base_reason = reject_reason(user, assistant, 600, 900)
    if base_reason:
        return base_reason
    if REJECT_USER.search(user):
        return "ambiguous_prompt_template"
    if REJECT_ANSWER.search(assistant):
        return "teacher_leak_or_known_inaccuracy"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Yerel öğretmen çıktısını eğitimden önce, gözle doğrulanmış kötü kalıplardan arındırır.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = [Path(path) for path in args.input]
    output = Path(args.output)
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}")
    if missing := [str(path) for path in sources if not path.is_file()]:
        parser.error("Kaynak bulunamadı: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    digest = hashlib.sha256()
    with temporary.open("x", encoding="utf-8", newline="\n") as target:
        for source in sources:
            for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                stats["input_records"] += 1
                try:
                    item = json.loads(line)
                    user, assistant = normalise(item["user"]), normalise(item["assistant"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    stats["rejected_invalid_schema"] += 1
                    continue
                reason = rejection_reason(user, assistant)
                if reason:
                    stats[f"rejected_{reason}"] += 1
                    continue
                key = (user.casefold(), assistant.casefold())
                if key in seen:
                    stats["rejected_exact_duplicate"] += 1
                    continue
                seen.add(key)
                rendered = json.dumps({"user": user, "assistant": assistant, "provenance": item.get("provenance", {})}, ensure_ascii=False) + "\n"
                target.write(rendered)
                digest.update(rendered.encode("utf-8"))
                stats["accepted"] += 1
    temporary.replace(output)
    report = {"sources": [str(source) for source in sources], "statistics": dict(sorted(stats.items())), "output_sha256": digest.hexdigest(), "note": "This is a narrow, rule-based cleanup; sample the output before training."}
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Temiz öğretmen verisi: {output} ({stats['accepted']} kayıt)")
    print("İstatistik: " + ", ".join(f"{name}={count}" for name, count in sorted(stats.items())))


if __name__ == "__main__":
    main()
