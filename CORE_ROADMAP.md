# HyperAI mimari ve gelişim planı

Bu depo, kendi eğitilmiş HyperAI modeli için düzenlenmektedir. Haricî bir LLM,
model API'si veya hazır tokenizer çekirdeğin parçası değildir.

## Şimdi hazır olan temel

- `hyperai_core/`: framework bağımsız tokenizer/model sözleşmesi ve üretim döngüsü.
- `hyperai_core/checkpoint.py`: eğitimin yazacağı checkpoint paketini `manifest.json`
  üzerinden doğrular; otomatik pickle çalıştırmaz.
- `hyperai_core/registry.py`: kendi JAX veya PyTorch model yükleyicini, açık bir
  mimari adıyla kaydetmek için kullanılır.
- `hyperai_core/tokenizer.py`: Türkçe dahil her UTF-8 metni kayıpsız taşıyan
  fallback byte tokenizer ve eğitim verisinden BPE merge öğrenen tokenizer.
- `hyperai_core/model.py`: RoPE, SDPA, KV cache, RMSNorm ve SwiGLU kullanan
  `hyperai-torch-v1` decoder-only Transformer.
- `hyperai_core/torch_checkpoint.py`: eğitimin ürettiği saf PyTorch `state_dict`
  ağırlıklarını kendi modeline geri bağlayan adaptör.
- `tests/test_core_runtime.py`: gerçek ağırlık gerektirmeden çekirdek davranışını test eder.
- `hyperai_core/data.py` ve `prepare_dataset.py`: tekrarları koruyan, Unicode
  kod noktalarını değiştirmeyen JSONL sohbet verisi hazırlama hattı.
- `hyperai_core/training.py` ve `train_hyperai.py`: sohbet-turu EOS, BPE, aynı
  çifti sızdırmayan gruplu train-validation ayrımı, asistan-only loss, AdamW
  parametre grupları, GPU AMP, warmup/cosine, atomik per-epoch/best checkpoint,
  resume/RNG durumu, adım bazlı loss ve sabit değerlendirme çıktıları.

## Eğitim çıktısının biçimi

Eğitim döngüsü çekirdekte bulunur. İleride kendi tokenizer'ını eğitirsen aynı
arayüzü uygulaman yeterlidir. Her checkpoint klasöründe aşağıdaki gibi bir
`manifest.json` bulunur:

```json
{
  "format_version": 1,
  "architecture": "hyperai-torch-v1",
  "weights": "weights/model.pt",
  "tokenizer": "tokenizer/tokenizer.json",
  "context_window": 256,
  "vocab_size": 260
}
```

`weights/` içinde ayrıca aynı eğitimin mimari ayarlarını taşıyan
`model_config.json` bulunur. `architecture` için yükleyici, uygulama başlangıcında
`ModelRegistry` ile açıkça kaydedilir. Böylece checkpoint içeriğinden kod
çalıştırılmaz.

## Eski prototiplerden alınan kararlar

Önceki projeler incelendi ve yeni çekirdeğe yalnızca kalıcı değeri olan fikirler
aktarılmıştır:

| Alınan fikir | Yeni karşılığı |
| --- | --- |
| Yerel model eğitme/çıkarım fikri | `HyperAILanguageModel` ve `HyperAIRuntime` |
| Token tabanlı üretim | `Tokenizer` ve `AutoregressiveModel` sözleşmeleri |
| Checkpoint ile model geri yükleme | Doğrulanmış `manifest.json` + güvenli `state_dict` yükleme |
| Terminal sohbeti | `main.py` ve `run_hyperai.py` |
| Türkçe önceliği | Kayıpsız UTF-8 byte tokenizer |

Eski kaynak dosyalar, kopya demolar ve kullanılmayan arayüzler çekirdeğe
bağlı değildir; depo sade kalması için kaldırılmıştır.

## Sıradaki mühendislik işi

1. İzinli düz Türkçe metinle ön eğitim, ardından temiz sohbet turuyla ince ayar;
   her veri sürümünü `DATA_CARD.md` ile belgele.
2. 30–50 sabit, eğitimden ayrı değerlendirme prompt'u tanımla; `metrics/loss.svg`
   ile birlikte cevap kalitesini her koşuda elle karşılaştır.
3. Eğitim parametrelerini küçük deneylerle ölçüp model boyutunu RTX 3050 belleğine
   göre ayarla.
4. Çok turlu sohbet bağlamı, ardından tek yerel API ve sohbet arayüzü ekle.
