# HyperAI

HyperAI, hazır bir sohbet modeli veya dış model API'si kullanmadan kendi
eğittiğin dil modelini çalıştırmak için düzenlenmekte olan yerel AI projesidir.

## Dürüst model kartı

HyperAI şu an eğitim altyapısı tamamlanmış, **küçük ve sıfırdan eğitilecek** bir
Türkçe metin devam ettirme/sohbet denemesidir. Birkaç MB veriyle akıcı genel
asistan olmaz; kısa kalıpları ve dar bir alanın üslubunu öğrenebilir. Yararlı
sonuç için önce izinli düz Türkçe metinle ön eğitim (başlangıçta birkaç MB,
hedef 10–50 MB), sonra birkaç bin temiz sohbet turuyla ince ayar gerekir.
Doğruluk, güvenlik, güncel bilgi, uzun muhakeme ve kişisel veri gizliliği için
garanti vermez. Eğitim verisini ezberleyebileceği için özel bilgi koymayın.

## Bugünkü çekirdek

`hyperai_core/` altında çalışan yeni çekirdek bulunur:

- eğitilebilir yerel byte-level BPE tokenizer (isteğe bağlı saf UTF-8 byte fallback),
- RoPE, SDPA ve KV cache kullanan sıfırdan eğitilebilir `hyperai-torch-v1` decoder-only Transformer,
- güvenli manifest tabanlı checkpoint doğrulaması,
- temperature, top-p ve EOS durdurma desteği olan üretim döngüsü,
- eğitimden sonra yerel checkpoint ile sohbet başlatan `run_hyperai.py`.

Bu paket model indirmez, dış API çağırmaz ve hazır bir AI sağlayıcısına bağlı
değildir.

## Model/checkpoint biçimi

Bir `hyperai-torch-v1` checkpoint klasörü şu biçimde olmalıdır:

```text
checkpoint/
├── manifest.json
├── weights/
│   ├── model.pt
│   └── model_config.json
└── tokenizer/
    └── tokenizer.json
```

`manifest.json` örneği:

```json
{
  "format_version": 1,
  "architecture": "hyperai-torch-v1",
  "weights": "weights/model.pt",
  "tokenizer": "tokenizer/tokenizer.json",
  "context_window": 512,
  "vocab_size": 260
}
```

Her epoch checkpoint'inde ayrıca optimizer, scheduler, AMP scaler ve epoch
numarasını içeren `training_state.pt` bulunur; bu dosya `--resume` için kullanılır.

## Çalıştırma

Kendi eğitim çıktın hazır olduğunda:

```powershell
python -m hyperai_core --checkpoint .\checkpoints\deneme-1\best --device auto
```

Windows'ta `python` komutu Microsoft Store yönlendirmesine gidiyorsa bunun yerine:

```powershell
.\start_hyperai.ps1 -Checkpoint .\checkpoints\deneme-1\best
```

Bu betik kullanıcı profilindeki standart Python 3.10–3.12 kurulumlarını bulur;
sistem PATH ayarını değiştirmez.

## Eğitime hazırlama ve eğitim

Hazırlayıcı, sırayla yazılmış `kullanıcı`/`HyperAI` satır çiftlerini JSONL sohbet
verisine çevirir. Aynı satırlar korunur; Unicode NFKC normalizasyonu uygulanmaz.
Eğitimde her çift şu biçimde işlenir ve sonunda özel EOS tokenı eklenir:

```text
Sen: kullanıcı mesajı
HyperAI: model yanıtı<EOS>
```

Model yalnızca `HyperAI:` yanıt bölümündeki tokenlar ve EOS için loss hesaplar;
kullanıcı prompt'u hedef olarak öğrenilmez. Inference başlangıcında da eğitimle
aynı olacak biçimde BOS tokenı ve `Sen: …\nHyperAI:` şablonu kullanılır.

Önce ham sohbet satırlarını JSONL verisine çevir:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $py prepare_dataset.py `
  --input .\datasets\seed\legacy-chat-corpus.txt `
  --output .\datasets\processed\chat-pairs-v1.jsonl
```

Ardından kendi HyperAI modelini eğit:

```powershell
& $py train_hyperai.py `
  --corpus .\datasets\processed\chat-pairs-v1.jsonl `
  --output .\checkpoints\hyperai-deneme-1 `
  --epochs 10 `
  --context-window 256 `
  --dropout 0.05 `
  --evaluation-prompts .\evaluation_prompts.example.txt
```

Eğitim komutu şunları üretir: her epoch için checkpoint (varsayılan son üçü
tutulur), en iyi validation checkpoint'i (`best/`), model konfigürasyonu,
tokenizer dosyası, doğrulanabilir `manifest.json`, adım bazlı `metrics/loss.csv`,
ek kütüphanesiz `metrics/loss.svg` eğrisi ve sabit prompt örneği. **Çıkarımda
daima `best/` kullanın**; son epoch, küçük veride aşırı öğrenmiş olabilir.
`epoch-XXX/` checkpoint'leri kesilen eğitime devam etmek içindir.

Varsayılan tokenizer, yalnızca eğitim verisinden merge öğrenen yerel
byte-level BPE'dir. Daha basit ama daha uzun diziler için `--tokenizer byte`
kullanılabilir. BPE sözlük hedefini değiştirmek için
`--tokenizer-vocab-size 8000` gibi bir değer verilebilir.

BPE eğitiminde merge'ler kelime ve boşluk parçaları dışına taşmaz; bu nedenle
prompt sonu ile yanıt başlangıcı farklı biçimde birleşmez. Eğitim, tam korpusu
her merge turunda taramak yerine özgün parça frekansları üzerinde çalışır.

Eğitim GPU bulunduğunda otomatik mixed precision kullanır; CPU'da güvenli
biçimde tam hassasiyete döner. AdamW weight decay yalnızca ağırlık
matrislerine uygulanır; embedding, norm ve bias parametreleri hariç tutulur.
Learning rate warmup sonrası cosine decay uygular.

Kesilen eğitime devam etmek için:

```powershell
& $py train_hyperai.py `
  --corpus .\datasets\processed\chat-pairs-v1.jsonl `
  --output .\checkpoints\hyperai-deneme-1 `
  --epochs 20 `
  --resume .\checkpoints\hyperai-deneme-1\epoch-010
```

Resume sırasında komut satırındaki model/tokenizer ayarları yerine checkpoint'in
`model_config.json` ve `tokenizer.json` dosyaları kullanılır. Her epoch için
checkpoint yazılır; varsayılan olarak son üç `epoch-*` checkpoint'i saklanır,
`best/` korunur. Python/Torch/CUDA RNG ile DataLoader karıştırma durumu da
kaydedilir; CPU↔GPU geçişinde AMP scaler kasıtlı olarak sıfırlanır. Aynı cihaz,
aynı sürüm ve aynı toplam epoch hedefi en tekrarlanabilir devam koşusudur.

Var olan dolu bir `--output` klasörüne yeni eğitim yazılmaz. Bilerek aynı isimli
checkpoint'leri değiştirmek gerektiğinde ayrıca `--overwrite` verin.

Eğitim verisini kullanmadan önce salt-okunur kişisel veri denetimini çalıştırın:

```powershell
& $py audit_dataset.py --input .\datasets\processed\chat-pairs-v1.jsonl
```

Kaynak, lisans/izin ve denetim sonucunu [DATA_CARD.md](DATA_CARD.md) şablonuna
yazın. Değerlendirme prompt'larını eğitim verisinden ayrı tutun; örnek dosyayı
30–50 alan-odaklı prompt'a genişletin.

Varsayılan model kasıtlı olarak küçük tutuldu (`128` genişlik, `4` katman,
`4` head, `256` token bağlam). Bu ayarlar sistemin doğru çalıştığını görmek
içindir; ciddi bir model için veri büyüdükçe donanıma göre artırılmalıdır.

Gerekli bağımlılıklar `requirements-core.txt` dosyasında PyTorch ve NumPy olarak
tanımlıdır.

## NVIDIA / CUDA GPU kullanımı

HyperAI, CUDA'lı PyTorch yüklüyse otomatik olarak RTX GPU'yu seçer ve mixed
precision kullanır. Windows'ta varsayılan PyPI `torch` paketi CPU-only olabilir;
NVIDIA GPU için resmî CUDA 12.6 indexini kullanan ayrı dosya vardır:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $py -m pip install --upgrade --force-reinstall -r requirements-cuda.txt
& $py verify_cuda.py
```

`verify_cuda.py` çıktısında GPU adı ve `CUDA tensör testi: PASS` görünmelidir.
RTX 3050'nin 4 GB VRAM'i için küçük başlangıç konfigürasyonları kullanmak uygun
olur; büyük batch veya model boyutu CUDA bellek hatasına neden olabilir.

## Büyük korpusla ön-eğitim

`datasets\\tokens-v1` paketi, hızlı BPE ile hazırlanmış büyük korpusun RAM'e
alınmadan okunabilen halidir. Kaynakların tümü eğitim havuzunda kalır; ancak
çok büyük web kaynağının kaliteli Türkçe Vikipedi ve sohbet verisini ezmemesi
için örnekleme ağırlıkları dengelenir (%45 Vikipedi, %40 filtreli web, %10
TinyStories, %5 sohbet). Bu kaynak silmek değildir: her biri eğitimde seçilir.

İlk gerçek ön-eğitim koşusunu **CMD**'de, proje klasöründeyken başlatmak için:

```cmd
py pretrain_hyperai.py --tokens datasets\tokens-v1 --output checkpoints\pretrain-v1 --steps 10000 --batch-size 4 --context-window 256 --model-dim 192 --layers 6 --heads 6 --dropout 0.05 --learning-rate 1e-3 --warmup-steps 500 --eval-batches 40 --eval-every 500 --save-every 500 --device cuda
```

Bu ayar RTX 3050 4 GB üzerinde denendi: yaklaşık 6.7 milyon parametre ve adım
başına 1.024 token kullanır. İlk 10.000 adım bir kalite/istikrar koşusudur;
sonuç iyi görünürse aynı checkpoint'ten daha yüksek bir toplam adım hedefine
devam edilir. `best/` en düşük doğrulama kaybını, `last/` en güncel devam
durumunu tutar. Kesilirse CMD'ye şu komut yazılır (örnekte hedef 50.000 adıma
çıkarılmıştır):

```cmd
py pretrain_hyperai.py --tokens datasets\tokens-v1 --output checkpoints\pretrain-v1 --resume checkpoints\pretrain-v1\last --steps 50000 --batch-size 4 --context-window 256 --model-dim 192 --layers 6 --heads 6 --dropout 0.05 --learning-rate 1e-3 --warmup-steps 500 --eval-batches 40 --eval-every 500 --save-every 500 --device cuda
```

### A100 üzerinde V4 büyük koşusu

RTX deneme checkpoint'leriyle karıştırmayın: `a100-140m-v4` ön ayarı 32k BPE
sözlükte **138.431.232 parametre**, 12 katman ve 1024 token bağlam kullanır.
Bu, mevcut 1M/6.7M deneylerin devamı değil; token paketi hazır olduğunda
sıfırdan başlayacak ayrı bir ön-eğitim koşusudur. Eski checkpoint'e `--resume`
verilmez.

```bash
python pretrain_hyperai.py \
  --preset a100-140m-v4 \
  --tokens dataset/tokens-v4 \
  --output checkpoints/hyperai-v4-140m-pretrain \
  --device cuda
```

Ön ayardaki varsayılan hedef 180.000 adımdır (16.384 token/adım, yaklaşık
2.95B örneklenmiş token). Token korpusunun gerçek boyutu ve kaynak lisansları
denetlenmeden bu komutu başlatmayın. Deneme için yalnızca hedef adım sayısı
değiştirilebilir: `--steps 500`.

`checkpoints\pretrain-v1\loss.csv` her değerlendirme anının loss, öğrenme
oranı ve token/saniye bilgisini içerir. Ön-eğitim sonunda sohbet kalitesi için
ayrıca kısa ve temiz sohbet çiftleriyle `train_hyperai.py` fine-tune koşusu
yapılır. `train_hyperai.py` artık ön-eğitim checkpoint'ini yeni bir sohbet
koşusuna başlangıç ağırlığı olarak da alabilir. Bu, model ve tokenizer'ı
sıfırdan başlatmaz; optimizer/scheduler ise sohbet aşaması için bilinçli olarak
yeniden başlar:

```cmd
py train_hyperai.py --corpus datasets\clean-chat-v2.jsonl --output checkpoints\hyperai-chat-v1 --init-checkpoint checkpoints\pretrain-v1\best --epochs 5 --batch-size 4 --learning-rate 0.0002 --context-window 256 --dropout 0.05 --device cuda --evaluation-prompts evaluation_prompts.example.txt
```

`--init-checkpoint` ile `--resume` birbirinin alternatifi iki davranıştır:
ilki ön-eğitimden *yeni* bir sohbet koşusu başlatır; ikincisi mevcut sohbet
koşusunu optimizer durumu ile devam ettirir. Ham sohbet verisini bu komuta
vermeyin: önce kişisel veri ve kalitesiz/uygunsuz cevap denetiminden geçirilmiş
`clean-chat-v2.jsonl` hazırlanmalıdır.

## Sözlük ve gündelik konuşma katmanı

Genel sohbet havuzunu daha fazla epoch çalıştırmak, kısa selamlaşmaları
öğretmek yerine modelin ansiklopedik ve tekrarlayan yanıt eğilimini artırabilir.
Bu nedenle sözlük bilgisi ile günlük diyalog iki **ayrı, sıralı** ince ayar
olarak tutulur. Türkçe Vikisözlük dump'ı zaten yereldeyse önce sözlük paketini
oluşturun:

```cmd
py prepare_wiktionary_dataset.py --input datasets\raw\trwiktionary-latest-pages-articles.xml.bz2 --output datasets\lexicon-v1.jsonl
py audit_dataset.py --input datasets\lexicon-v1.jsonl
```

Araç yalnızca Türkçe bölümün birinci seviye tanımlarını alır; kaynak SHA-256,
kaynak adresi ve CC BY-SA lisans notu `datasets\lexicon-v1.report.json` içinde
kalır. Kaynağı ve lisans yükümlülüğünü dağıtımda koruyun.

Gündelik kalıplar için, başka modelden üretilmemiş küçük ve elle denetlenebilir
çekirdek paket şununla oluşturulur:

```cmd
py create_daily_chat_seed.py --output datasets\daily-chat-v1.jsonl
py audit_dataset.py --input datasets\daily-chat-v1.jsonl
```

Önerilen sıra: önce iyi durumdaki sohbet checkpoint'inden (`hyperai-chat-v1\best`)
sözlük bilgisini düşük öğrenme oranıyla bir kez ekleyin; ardından **ayrı çıktı
klasöründe** günlük sohbet çekirdeğini kısa bir son davranış aşaması olarak
çalıştırın. `hyperai-chat-all-v2`, birleşik veri deneyi olarak saklanır; kısa
günlük sohbet testlerinde daha kötü sonuç verdiğinden başlangıç checkpoint'i
olarak seçilmez.

```cmd
py train_hyperai.py --corpus datasets\lexicon-v1.jsonl --output checkpoints\hyperai-lexicon-v1 --init-checkpoint checkpoints\hyperai-chat-v1\best --epochs 1 --batch-size 4 --learning-rate 0.00001 --context-window 256 --dropout 0.05 --device cuda --evaluation-prompts evaluation_prompts.example.txt --log-every 250

py create_daily_chat_seed.py --output datasets\daily-chat-weighted-v1.jsonl --repeat 16
py train_hyperai.py --corpus datasets\daily-chat-weighted-v1.jsonl --output checkpoints\hyperai-daily-v1 --init-checkpoint checkpoints\hyperai-lexicon-v1\best --epochs 10 --batch-size 4 --learning-rate 0.00005 --context-window 256 --dropout 0.05 --device cuda --evaluation-prompts evaluation_prompts.example.txt --log-every 20
```

İkinci eğitim komutunu yalnızca ilk komut `best/` ürettikten sonra çalıştırın.
`--repeat 16`, 40 özgün davranış örneğini silmeden/yeniden yazmadan eğitimde
yeterli ağırlığa çıkarır; gruplanmış split tüm tekrarları aynı tarafa koyar.
Günlük paket küçük olduğu için bu son aşama hızlıdır; `best/` ile `daily-chat-v1`
prompt'larını (özellikle `selam`, `naber`, `nasılsın`) elle karşılaştırmadan
modeli kullanıcıya açmayın.

## Yerel RAG ve sınırlı web bilgisi

Küçük HyperAI modeli güncel bilgiyi kendi ağırlıklarından güvenilir biçimde
çıkaramaz. Bunun için modelden bağımsız bir bilgi alma katmanı eklendi. Yerel
RAG, sadece sizin açıkça verdiğiniz `.txt`, `.md` veya `text`/`content` alanlı
`.jsonl` belgeleri üzerinde BM25 araması yapar; ek model, vektör veritabanı ya
da bulut hesabı indirmez. Bulduğu belgeler, model cevabından **ayrı** kaynak
alıntıları olarak görünür; belge metni gizlice komut/prompt olarak enjekte
edilmez.

```cmd
py run_hyperai.py --checkpoint checkpoints\hyperai-daily-v1\best --device cuda --knowledge-base docs\notlar.md --knowledge-base docs\bilgi.jsonl
```

`--wikipedia` seçeneği her soruda yalnızca sabit Türkçe Vikipedi HTTPS
uçlarından bir arama ve kısa özet ister. Keyfi URL indirmez, yönlendirme
izlemez, yanıt boyutunu sınırlar ve kaynak bağlantısını gösterir:

```cmd
py run_hyperai.py --checkpoint checkpoints\hyperai-daily-v1\best --device cuda --wikipedia
```

Bu katman modelin muhakeme kapasitesini büyütmez; güncel veya yerel bilgiyi
denetlenebilir biçimde getirir. Küçük model kaynakta olmayan bir sonucu yine de
uydurabilir; kritik bilgi için kaynak alıntısını esas alın. CLI artık varsayılan
olarak `48` token üst sınırı, `1.15` tekrar cezası ve üç-token tekrar engeli
kullanır. Bunlar modelin tekrar döngüsünü azaltan üretim korumalarıdır, eğitim
yerine geçen bir kalite iddiası değildir.

## Gemini öğretmeninden HyperAI'ye distillation

Gemini burada canlı sohbet modeli değildir: yalnızca eğitim verisi üreten
**öğretmen**dir. Çalışma zamanında yalnızca HyperAI checkpoint'i yüklenir.
Öğretmen yanıtları önce PII/URL/sır/model-açıklaması/uzunluk/tekrar filtresinden
geçer, ardından kaynak modeli ve prompt kimliği kaydedilmiş JSONL öğrenci
verisine yazılır. Hiçbir API anahtarı dosyaya, rapora veya Git'e yazılmaz.

Önce insan yazımı başlangıç prompt havuzunu üretin:

```cmd
py create_distillation_prompts.py --output datasets\distillation-prompts-v1.jsonl
```

Google AI Studio hesabınızdan bir Gemini API anahtarı oluşturun, onu **yalnızca açık CMD
oturumuna** koyun; anahtarı sohbet mesajına veya kaynak koda yapıştırmayın:

```cmd
set GEMINI_API_KEY=BURAYA_YERELDE_ANAHTARINIZ
```

İlk olarak küçük, denetlenebilir bir 20 örneklik öğretmen turu çalıştırın:

```cmd
py distill_gemini.py --prompts datasets\distillation-prompts-v1.jsonl --output datasets\distilled-chat-v1.jsonl --model gemini-3.5-flash --limit 20
py audit_dataset.py --input datasets\distilled-chat-v1.jsonl
```

Komut, Gemini'nin resmi `generateContent` HTTPS uç noktasını kullanır ve her
kayda öğretmen model/provenance bilgisi koyar. Hesabınızda hangi model adlarının
etkin olduğunu Google AI Studio'dan kontrol edin; model adı hesabınıza göre
değişebilir. Önce `datasets\distilled-chat-v1.jsonl` dosyasından örnekleri
elle inceleyin. Kalite uygunsa yeni sürüm adıyla 54 promptun tamamını üretin;
sonra bu veri, insan yazımı temiz sohbet verisiyle dengelenerek ayrı bir
HyperAI öğrenci eğitim koşusunda denenir. Öğretmen çıktısını otomatik olarak
mevcut checkpoint'lere basmayın.

Gemini `429 Too Many Requests` döndürürse kota dolmuş veya istek hızı fazla
yüksek olabilir. Araç varsayılan olarak istekler arasında 12 saniye bekler,
iki kez artan bekleme uygular ve kota sürerse kabul edilmiş kayıtları koruyup
durur. Kota yenilendiğinde aynı çıktıya güvenle devam etmek için:

```cmd
py distill_gemini.py --prompts datasets\distillation-prompts-v1.jsonl --output datasets\distilled-chat-v1.jsonl --model gemini-3.5-flash --limit 20 --resume
```

## Proje ilkesi

Bu depo yalnızca HyperAI'nin güncel ve yerel çekirdeğini içerir. Eski prototip,
kopya demo, yarım model ve kullanılmayan arayüz kodları bilinçli olarak
çıkarılmıştır. Mimari kararlar ve sıradaki geliştirme işleri için
`CORE_ROADMAP.md` dosyasına bak.
