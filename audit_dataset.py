"""JSONL sohbet verisinde olası kişisel veri izlerini raporlar; dosyayı değiştirmez."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "telefon": re.compile(r"(?<!\d)(?:\+?90\s?)?(?:0?\d{3}[\s.-]?)?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"),
    "olasi_sir": re.compile(r"\b(?:api[_-]?key|token|password|parola|şifre)\s*[:=]\s*\S+", re.IGNORECASE),
}


def load_pairs_for_audit(path: str | Path):
    """Denetim, eğitimden farklı olarak tek bir geçerli kaydı da inceleyebilir."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Denetlenecek dosya bulunamadı: {source}")
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number} geçerli JSONL değil.") from exc
        if not isinstance(item, dict) or not isinstance(item.get("user"), str) or not isinstance(item.get("assistant"), str):
            raise ValueError(f"{source}:{line_number} user ve assistant metin alanlarını içermeli.")
        yield item["user"], item["assistant"]


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAI JSONL verisindeki olası kişisel verileri raporlar.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    findings = 0
    records = 0
    for index, (user, assistant) in enumerate(load_pairs_for_audit(args.input), start=1):
        records += 1
        for field, text in (("user", user), ("assistant", assistant)):
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings += 1
                    print(f"satır {index}, {field}: olası {label}")
    print(f"Denetim tamamlandı. İncelenen kayıt: {records}; olası bulgu: {findings}")


if __name__ == "__main__":
    main()
