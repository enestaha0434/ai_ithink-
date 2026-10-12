"""Yerel öğretmen için çeşitli, kişisel veri içermeyen Türkçe prompt havuzu üretir."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import cycle
from pathlib import Path


SPECS: dict[str, tuple[list[str], list[str]]] = {
    "gundelik_sohbet": (
        ["selam", "merhaba", "hey", "naber", "nasılsın", "nasilsin", "bugün nasıl gidiyor?", "iyi akşamlar", "günaydın", "görüşürüz", "sağ ol", "teşekkür ederim", "çok teşekkürler", "bir şey sorabilir miyim?", "yardım eder misin?", "bunu anlamadım", "kısaca anlatır mısın?", "biraz karıştı", "daha basit anlatır mısın?", "buna bir örnek verir misin?", "bugün yoruldum", "canım sıkılıyor", "moralim biraz bozuk", "yeni bir şey öğrenmek istiyorum", "nereden başlamalıyım?", "bana küçük bir öneri verir misin?", "bunu tekrar eder misin?", "tamam, anladım", "bunu deneyeceğim", "iyi fikir", "sence mantıklı mı?", "buna sonra dönebilir miyiz?", "bir dakika düşüneyim", "hazırım", "devam edelim", "bugünlük bu kadar", "kendine iyi bak", "iyi geceler", "kolay gelsin", "tekrar görüşürüz"],
        ["{x}"],
    ),
    "aciklama": (
        ["algoritma", "veri", "veritabanı", "işletim sistemi", "tarayıcı", "arama motoru", "bulut depolama", "şifreleme", "iki aşamalı doğrulama", "açık kaynak", "yapay zekâ", "makine öğrenmesi", "model", "token", "API", "sunucu", "istemci", "ağ", "IP adresi", "alan adı", "RAM", "SSD", "dosya uzantısı", "yedekleme", "sürüm kontrolü"],
        ["{x} nedir? Kısa ve sade açıklar mısın?", "{x} ne işe yarar? Bir örnekle anlatır mısın?", "{x} ile ilgili en temel şeyi açıklar mısın?", "{x} neden önemlidir?"],
    ),
    "turkce": (
        ["empati", "sabır", "sorumluluk", "öz güven", "eleştiri", "hakaret", "saygı", "dürüstlük", "nezaket", "motivasyon", "alışkanlık", "odaklanma", "iletişim", "iş birliği", "merak", "hedef", "plan", "özet", "ana fikir", "örnek", "benzetme", "gerekçe", "varsayım", "tutarlılık", "bağlam"],
        ["{x} ne demek?", "{x} kavramını kısa bir örnekle açıklar mısın?", "{x} ile yakın bir kavram arasındaki farkı sade anlatır mısın?", "{x} hakkında iki cümlelik bir açıklama yazar mısın?"],
    ),
    "kod_baslangic": (
        ["değişken", "liste", "sözlük", "fonksiyon", "döngü", "koşul", "hata mesajı", "test", "debug", "JSON", "CSV", "README", "Git", "commit", "branch", "modül", "paket", "yorum satırı", "string", "integer", "boolean", "input", "output", "dosya yolu", "terminal"],
        ["Python'da {x} nedir?", "Kod yazarken {x} ne işe yarar?", "{x} konusunu yeni başlayan biri için açıklar mısın?", "{x} kullanırken dikkat edilmesi gereken bir noktayı söyler misin?"],
    ),
    "guvenli_dijital": (
        ["güçlü parola", "parola yöneticisi", "iki aşamalı doğrulama", "şüpheli bağlantı", "oltalama mesajı", "uygulama izni", "genel Wi-Fi", "yedek", "güncelleme", "kişisel bilgi", "gizlilik ayarı", "hesap kurtarma", "dosya paylaşımı", "bilinmeyen ek", "güvenilir kaynak", "tarayıcı eklentisi", "sosyal medya paylaşımı", "ekran kilidi", "virüs taraması", "indirme sitesi"],
        ["{x} konusunda temel bir güvenlik önerisi verir misin?", "{x} neden riskli veya faydalı olabilir?", "{x} ile ilgili dikkat edilmesi gerekenleri kısa yazabilir misin?", "{x} konusunda güvenli bir ilk adım ne olur?"],
    ),
    "problem_cozme": (
        ["ders çalışmak", "küçük bir proje planlamak", "not almak", "yeni bir konu öğrenmek", "hata ayıklamak", "zamanı planlamak", "dosyaları düzenlemek", "bir metni özetlemek", "sunuma hazırlanmak", "yapılacaklar listesi oluşturmak", "odaklanmak", "zor bir göreve başlamak", "geri bildirim almak", "hedef belirlemek", "alışkanlık kurmak", "araştırma yapmak", "kaynak karşılaştırmak", "bir fikri açıklamak", "soru sormak", "karar vermek"],
        ["{x} için kısa bir başlangıç planı önerir misin?", "{x} sırasında ilk üç adım ne olabilir?", "{x} işini daha düzenli yapmak için ne önerirsin?", "{x} konusunda takılırsam nasıl ilerlemeliyim?"],
    ),
}


def build_rows(count: int) -> list[dict[str, str]]:
    candidates: list[tuple[str, str]] = []
    for category, (topics, templates) in SPECS.items():
        for topic in topics:
            for template in templates:
                candidates.append((category, template.format(x=topic)))
    if count > len(candidates):
        raise ValueError(f"En fazla {len(candidates)} farklı prompt üretilebilir.")
    # Kategorileri sırayla döndürmek, ilk N kaydın tek bir konuya yığılmasını önler.
    by_category = {category: [text for item_category, text in candidates if item_category == category] for category in SPECS}
    rows: list[dict[str, str]] = []
    positions = {category: 0 for category in SPECS}
    for category in cycle(SPECS):
        index = positions[category]
        if index >= len(by_category[category]):
            continue
        positions[category] += 1
        rows.append({"id": f"{category}-{index + 1:03d}", "category": category, "user": by_category[category][index]})
        if len(rows) == count:
            return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAI için dengeli, insan yazımı şablonlardan yerel öğretmen promptları üretir.")
    parser.add_argument("--output", default="datasets/distillation-prompts-v2.jsonl")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count pozitif olmalıdır.")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}")
    try:
        rows = build_rows(args.count)
    except ValueError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    by_category = {category: sum(row["category"] == category for row in rows) for category in SPECS}
    output.with_suffix(".report.json").write_text(json.dumps({
        "purpose": "human-authored safe prompt templates for local teacher-student distillation",
        "records": len(rows), "by_category": by_category,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Genişletilmiş öğretmen prompt havuzu: {output} ({len(rows)} istek)")


if __name__ == "__main__":
    main()
