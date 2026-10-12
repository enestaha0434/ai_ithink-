"""Kendi HyperAI modelini yerel korpusla eğitmek için komut satırı aracı."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    # Eğitim örneği Türkçe/bozuk byte içerebilir; log dosyasına yönlendirmede de
    # Windows kod sayfası yüzünden epoch sonunda çökmesini engeller.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="HyperAI'nin kendi Transformer modelini eğit.")
    parser.add_argument("--corpus", required=True, help="Hazırlanmış JSONL sohbet verisi")
    parser.add_argument("--output", required=True, help="Checkpoint çıktısının yazılacağı klasör")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--context-window", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--sample-prompt", default="Merhaba, nasılsın?")
    parser.add_argument("--tokenizer", choices=("bpe", "byte"), default="bpe")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--sequence-stride", type=int, help="Eğitim blokları arası token adımı; varsayılan context/2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=250, help="Epoch içi ilerleme satırı aralığı (batch)")
    parser.add_argument("--evaluation-prompts", help="Her epoch üretilecek, satır başına bir sabit değerlendirme prompt'u")
    parser.add_argument("--overwrite", action="store_true", help="Var olan output altındaki aynı checkpoint adlarını bilerek değiştir")
    parser.add_argument("--resume", help="Devam edilecek epoch-XXX veya best checkpoint klasörü")
    parser.add_argument("--init-checkpoint", help="Ön-eğitim checkpoint'inden yeni sohbet ince ayarı başlatır; --resume ile birlikte kullanılamaz")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        from hyperai_core.model import HyperAIModelConfig
        from hyperai_core.training import TrainingConfig, train_from_corpus
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            parser.error("PyTorch kurulu değil. Önce: python -m pip install -r requirements-core.txt")
        raise

    model_config = HyperAIModelConfig(
        context_window=args.context_window,
        model_dim=args.model_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_stride=args.sequence_stride or max(1, args.context_window // 2),
        validation_fraction=args.validation_fraction,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        device=args.device,
        sample_prompt=args.sample_prompt,
        tokenizer_kind=args.tokenizer,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        warmup_steps=args.warmup_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        log_every_steps=args.log_every,
        evaluation_prompts_path=args.evaluation_prompts,
        allow_overwrite=args.overwrite,
    )
    checkpoint = train_from_corpus(args.corpus, args.output, model_config, training_config, resume_from=args.resume, init_from=args.init_checkpoint)
    print(f"Eğitim tamamlandı. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
