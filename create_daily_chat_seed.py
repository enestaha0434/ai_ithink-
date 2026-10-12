"""Küçük fakat denetlenebilir Türkçe gündelik sohbet çekirdeği üretir.

Bu veriler elle yazılmıştır; başka bir model çıktısı değildir. Amaç, büyük
bilgi veri setlerinin bastırdığı selamlaşma/teşekkür/vedalaşma davranışını
ayrı bir düşük riskli ince ayar aşamasında geri kazandırmaktır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PAIRS = [
    ("selam", "Selam! Nasıl yardımcı olabilirim?"),
    ("slm", "Selam! Nasıl gidiyor?"),
    ("merhaba", "Merhaba! Buradayım, ne konuşmak istersin?"),
    ("merhabalar", "Merhabalar! Sana nasıl yardımcı olabilirim?"),
    ("hey", "Hey! Buradayım."),
    ("günaydın", "Günaydın! Umarım günün iyi geçer."),
    ("iyi akşamlar", "İyi akşamlar! Nasıl yardımcı olabilirim?"),
    ("iyi geceler", "İyi geceler! Dinlenmeyi unutma."),
    ("naber", "İyiyim, teşekkürler. Sen nasılsın?"),
    ("ne haber", "İyiyim, buradayım. Senin nasıl gidiyor?"),
    ("nasılsın", "İyiyim, teşekkür ederim. Sen nasılsın?"),
    ("nasilsin", "İyiyim, teşekkür ederim. Sen nasılsın?"),
    ("nasıl gidiyor", "İyi gidiyor. Senin günün nasıl geçiyor?"),
    ("keyifler nasıl", "İyi, teşekkürler. Senin keyfin nasıl?"),
    ("sağ ol", "Rica ederim!"),
    ("sağolasın", "Ne demek, her zaman."),
    ("teşekkürler", "Rica ederim!"),
    ("teşekkür ederim", "Rica ederim, yardımcı olabildiysem ne mutlu."),
    ("eyvallah", "Eyvallah, görüşmek üzere!"),
    ("görüşürüz", "Görüşürüz! Kendine iyi bak."),
    ("bay bay", "Bay bay!"),
    ("hoşça kal", "Hoşça kal!"),
    ("yardım eder misin", "Elbette. Ne konuda yardıma ihtiyacın var?"),
    ("bir şey sorabilir miyim", "Tabii, sorunu yazabilirsin."),
    ("bunu anlamadım", "Sorun değil. Hangi kısmını birlikte açalım?"),
    ("kısaca anlat", "Elbette. Konuyu veya metni gönder, kısa anlatayım."),
    ("bilmiyorum", "Birlikte bakabiliriz. Ne hakkında emin değilsin?"),
    ("canım sıkılıyor", "İstersen biraz sohbet edebilir, bir konu seçebiliriz."),
    ("bugün kötü hissediyorum", "Üzgünüm bunu duyduğuma. İstersen ne olduğunu anlatabilirsin."),
    ("şaka yap", "Neden bilgisayar üşümüş? Çünkü pencereleri açık kalmış."),
    ("adın ne", "Ben HyperAI'yım."),
    ("sen nesin", "Ben yazıyla yardımcı olmak için eğitilen küçük bir dil modeliyim."),
    ("ne yapabiliyorsun", "Soruları yanıtlamaya, açıklama yapmaya ve metinle çalışmaya yardımcı olabilirim."),
    ("türkçe biliyor musun", "Evet, Türkçe yazışabilirim."),
    ("bana bir öneri ver", "Elbette. Ne hakkında öneri istersin?"),
    ("tamam", "Tamamdır."),
    ("olur", "Harika."),
    ("anladım", "Sevindim. İstersen devam edebiliriz."),
    ("selam hyperai", "Selam! Buradayım, nasıl yardımcı olabilirim?"),
    ("iyi misin", "İyiyim, teşekkür ederim. Sen iyi misin?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="El yazımı günlük Türkçe sohbet çekirdeğini JSONL olarak üretir.")
    parser.add_argument("--output", default="datasets/daily-chat-v1.jsonl")
    parser.add_argument("--repeat", type=int, default=1, help="Her özgün çifti eğitimde kaç kez ağırlıklandıracağı (varsayılan: 1)")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"Çıktı zaten var: {output}")
    if args.repeat < 1:
        parser.error("--repeat en az 1 olmalıdır.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Tekrar eğitim ağırlığı içindir; aynı çiftler split sırasında birlikte
    # tutulduğu için validation sızıntısı oluşturmaz. Sabit karıştırma, aynı
    # cümlelerin uzun bloklar hâlinde token dizisine düşmesini önler.
    weighted_pairs = PAIRS * args.repeat
    random.Random(42).shuffle(weighted_pairs)
    lines = [json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n" for user, assistant in weighted_pairs]
    output.write_text("".join(lines), encoding="utf-8", newline="\n")
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    output.with_suffix(".report.json").write_text(json.dumps({
        "purpose": "manually authored everyday Turkish chat behaviour seed",
        "license": "project author-created; verify final ownership before redistribution",
        "unique_records": len(PAIRS), "weighted_records": len(weighted_pairs), "repeat": args.repeat, "sha256": digest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Günlük sohbet çekirdeği: {output} ({len(PAIRS)} özgün, {len(weighted_pairs)} ağırlıklı çift)")


if __name__ == "__main__":
    main()
