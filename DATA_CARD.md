# HyperAI veri kartı

Bu dosya, her eğitim koşusundan önce veri kaynağını ve izin durumunu kaydetmek
için şablondur. Model küçük olsa da eğitim örneklerini ezberleyip aynen geri
üretebilir; özel konuşmalar, telefon numaraları ve e-posta adresleri eğitim
verisine konmamalıdır.

## Her veri seti için doldur

- **Dosya / sürüm:**
- **Kaynak ve sahibi:**
- **Kullanım izni / lisans:**
- **Toplama tarihi:**
- **Dil ve içerik türü:**
- **Örnek ve tur sayısı:**
- **Bilinen önyargılar / eksikler:**
- **Kişisel veri denetimi:** `audit_dataset.py` sonucu ve elle yapılan inceleme
- **Başka AI hizmetinden üretildiyse:** ilgili hizmetin güncel kullanım koşulu
- **Eğitimde kullanılmasına onay veren kişi:**

## Veri ilkesi

1. İzin veya lisans belirsizse veriyi kullanma.
2. Telefon, e-posta, adres, kimlik numarası, parola/token ve özel mesajları
   temizle; otomatik taramayı mutlaka elle gözden geçir.
3. Değerlendirme prompt'larını eğitim verisinden ayrı tut.
4. Türkçe veri içinde aksansız yazım, `İ/i/ı/I`, kısaltmalar, günlük dil ve
   farklı uzunluklarda cevaplar dengeli yer alsın.

## HyperAI sözlük verisi: `lexicon-v1`

- **Kaynak:** Türkçe Vikisözlük'ün resmi XML dump'ı.
- **İçerik:** Yalnızca `==Türkçe==` bölümündeki birinci seviye tanımlardan
  türetilmiş `kelime ne demek?` sohbet çiftleri. Viki biçimlendirmesi, örnek
  cümleler ve şablon kalıntıları ayıklanır.
- **Lisans:** CC BY-SA. Kaynak gösterme ve aynı lisansla paylaşma yükümlülüğü
  vardır; eğitim çıktısı/dağıtım planında bu yükümlülük ayrıca incelenmelidir.
- **İzlenebilirlik:** Oluşturulan `.report.json` kaynak SHA-256'sını, indirilen
  dump adresini ve kabul/red sayılarını saklar.

## HyperAI günlük sohbet çekirdeği: `daily-chat-v1`

- **Kaynak:** Proje içinde elle yazılmış, denetlenmiş kısa Türkçe çiftler.
- **Amaç:** Selamlaşma, teşekkür, vedalaşma ve kısa yardım istekleri için
  davranış ince ayarı. Sözlük veya ansiklopedi verisinin yerine geçmez.
- **Kural:** Bu küçük set tek başına genel eğitim verisi değildir; ayrı ve düşük
  öğrenme oranlı son aşama olarak kullanılır.
