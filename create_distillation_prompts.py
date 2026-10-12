"""HyperAI için insan yazımı konu şablonlarından öğretmen-prompt havuzu üretir.

Bu dosya öğretmen cevabı üretmez ve web'den metin çekmez. Amaç, Mistral gibi
bir öğretmene verilecek, kişisel veri içermeyen ve dengeli Türkçe kullanıcı
istekleri üretmektir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPTS: dict[str, list[str]] = {
    "gundelik_sohbet": [
        "selam", "merhaba", "naber", "nasılsın", "nasilsin", "bugün nasıl gidiyor?",
        "teşekkür ederim", "sağ ol", "görüşürüz", "iyi akşamlar", "yardım eder misin?",
        "bir şey sorabilir miyim?", "bunu anlamadım", "kısaca anlatır mısın?",
        "canım sıkılıyor, ne yapabilirim?", "bugün biraz kötü hissediyorum",
    ],
    "aciklama": [
        "Yapay zekâyı bir çocuğa anlatır gibi açıklar mısın?", "Algoritma nedir, kısa açıklar mısın?",
        "İnternet ile web arasındaki fark nedir?", "Değişken nedir?", "Fonksiyon nedir?",
        "Dosya uzantısı ne işe yarar?", "RAM ile depolama arasındaki fark nedir?",
        "Şifre yöneticisi kullanmak neden faydalıdır?", "Yedek almak neden önemlidir?",
        "Bir hatayı adım adım nasıl araştırırım?", "Bir metni özetlemek için nasıl çalışmalıyım?",
        "Öğrenirken küçük projeler yapmak neden faydalıdır?",
    ],
    "turkce": [
        "Empati ne demek?", "Sabır ne demek?", "Eleştiri ile hakaret arasındaki fark nedir?",
        "Bir cümleyi daha nazik nasıl yazabilirim?", "Bu cümleyi daha anlaşılır hale getir: Bugün toplantı var.",
        "Kısa bir teşekkür mesajı yazar mısın?", "Arkadaşıma doğum günü mesajı yazmak istiyorum.",
        "Resmî ve samimi dil arasındaki fark nedir?", "Bir fikre katılmadığımı nazikçe nasıl söylerim?",
        "Kısa bir çalışma planı nasıl hazırlanır?",
    ],
    "guvenli_belirsizlik": [
        "Bilmiyorsan ne yaparsın?", "Bir konuda emin değilsen nasıl cevap vermelisin?",
        "Eksik bilgiyle karar vermek doğru mu?", "İnternette gördüğüm her bilgiye güvenmeli miyim?",
        "Kişisel bilgileri neden paylaşmamalıyım?", "Şifremi biriyle paylaşmalı mıyım?",
        "Bir iddiayı nasıl doğrulayabilirim?", "Bir kaynak güvenilir mi, nasıl anlarım?",
    ],
    "kod_baslangic": [
        "Python'da liste nedir?", "Python'da bir değişken nasıl tanımlanır?",
        "Kod yazarken hata mesajı ne anlatır?", "Bir program neden test edilir?",
        "Git ne işe yarar?", "Bir projede README neden gerekir?",
        "Bir kod parçasını küçük fonksiyonlara ayırmak neden iyidir?", "JSON nedir?",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mistral öğretmen distillation'ı için dengeli Türkçe prompt havuzu üretir.")
    parser.add_argument("--output", default="datasets/distillation-prompts-v1.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}")
    rows = [{"id": f"{category}-{index:03d}", "category": category, "user": prompt} for category, prompts in PROMPTS.items() for index, prompt in enumerate(prompts, 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    output.with_suffix(".report.json").write_text(json.dumps({
        "purpose": "human-authored seed prompts for teacher-student distillation",
        "records": len(rows), "by_category": {category: len(prompts) for category, prompts in PROMPTS.items()},
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Öğretmen prompt havuzu: {output} ({len(rows)} istek)")


if __name__ == "__main__":
    main()
