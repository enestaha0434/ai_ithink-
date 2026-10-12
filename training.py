"""HyperAI sohbet eğitimi: BPE, asistan-only loss, AMP, scheduler ve resume."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .contracts import GenerationSettings
from .data import CorpusError, format_chat_example, format_chat_prompt, load_chat_pairs
from .model import HyperAILanguageModel, HyperAIModelConfig
from .runtime import HyperAIRuntime
from .tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer

IGNORE_INDEX = -100
MAX_RESUME_STATE_BYTES = 2 * 1024 * 1024 * 1024
WINDOWS_RENAME_ATTEMPTS = 12


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    sequence_stride: int = 128
    validation_fraction: float = 0.2
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    sample_prompt: str = "Merhaba, nasılsın?"
    tokenizer_kind: str = "bpe"
    tokenizer_vocab_size: int = 8192
    keep_last_checkpoints: int = 3
    log_every_steps: int = 250
    evaluation_prompts_path: str | None = None
    allow_overwrite: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.sequence_stride < 1 or self.warmup_steps < 0 or self.log_every_steps < 1:
            raise ValueError("Epoch, batch, stride ve warmup değerleri geçersiz.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Optimizer ayarları geçersiz.")
        if not 0 < self.validation_fraction < 0.5 or not 0 <= self.min_lr_ratio <= 1 or self.keep_last_checkpoints < 1:
            raise ValueError("Validation, LR oranı veya checkpoint saklama değeri geçersiz.")
        if self.device not in {"auto", "cpu", "cuda"} or self.tokenizer_kind not in {"byte", "bpe"}:
            raise ValueError("device veya tokenizer_kind geçersiz.")


class TokenSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: list[int], labels: list[int], context_window: int, stride: int) -> None:
        if len(token_ids) != len(labels):
            raise ValueError("Token ve label uzunlukları eşit olmalıdır.")
        # Küçük davranış/kalibrasyon setlerinde validation bölümü bağlam
        # penceresinden kısa kalabilir. Eğitim hedeflerini değiştirmeden sona
        # PAD ve ignore-label ekleyerek tek güvenli pencere oluştururuz. Bu,
        # kısa veri için modeli daha küçük bir context ayarıyla yeniden kurma
        # zorunluluğunu kaldırır; gerçek tokenların loss'u aynen korunur.
        if len(token_ids) < context_window + 1:
            padding = context_window + 1 - len(token_ids)
            token_ids = [*token_ids, *([ByteTokenizer.PAD_ID] * padding)]
            labels = [*labels, *([IGNORE_INDEX] * padding)]
        last_start = len(token_ids) - context_window - 1
        candidates = list(range(0, last_start + 1, stride))
        if candidates[-1] != last_start:
            candidates.append(last_start)
        # Sadece prompt içeren bloklar loss'u NaN yapar; asistan/EOS hedefi olanları tut.
        self.starts = [start for start in candidates if any(label != IGNORE_INDEX for label in labels[start + 1 : start + context_window + 1])]
        if not self.starts:
            raise CorpusError("Seçilen context window içinde asistan hedefi yok.")
        self.token_ids, self.labels, self.context_window = token_ids, labels, context_window

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        return (
            torch.tensor(self.token_ids[start : start + self.context_window], dtype=torch.long),
            torch.tensor(self.labels[start + 1 : start + self.context_window + 1], dtype=torch.long),
        )


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi fakat kullanılabilir GPU bulunamadı.")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def _optimizer(model: HyperAILanguageModel, config: TrainingConfig) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 and "embedding" not in name else no_decay).append(parameter)
    return AdamW([{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate, betas=(0.9, 0.95))


def _scheduler(optimizer: AdamW, total_steps: int, config: TrainingConfig) -> LambdaLR:
    warmup = min(config.warmup_steps, max(0, total_steps - 1))
    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, factor)


def _loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction=reduction)


def _evaluate(model: HyperAILanguageModel, loader: DataLoader, device: torch.device, amp: bool) -> float:
    model.eval(); total_loss = 0.0; total_tokens = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs.to(device)); loss = _loss(logits, targets, reduction="sum")
            total_loss += loss.item(); total_tokens += (targets != IGNORE_INDEX).sum().item()
    return total_loss / total_tokens


def _replace_directory_with_retry(source: Path, destination: Path) -> None:
    """Kısa süreli Windows/antivirüs klasör kilitlerinde atomik yazımı korur."""
    last_error: PermissionError | None = None
    for attempt in range(WINDOWS_RENAME_ATTEMPTS):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(2.0, 0.08 * (2**attempt)))
    assert last_error is not None
    raise last_error


def _write_checkpoint(output_dir: Path, name: str, model: HyperAILanguageModel, tokenizer, model_config: HyperAIModelConfig, training_config: TrainingConfig, report: dict[str, object], optimizer: AdamW, scheduler: LambdaLR, scaler, completed_epoch: int, global_step: int, data_generator: torch.Generator) -> Path:
    """Checkpoint'i staging dizinine yazar, sonra dizin düzeyinde değiştirir.

    Güç kesilmesi/yetersiz disk alanı eski checkpoint'i bozmaz; en kötü durumda
    yalnızca silinebilecek `.staging-*` ya da `.previous-*` dizini kalır.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    checkpoint_dir = output_dir / f".{name}.staging-{uuid.uuid4().hex}"
    weights_dir, tokenizer_dir = checkpoint_dir / "weights", checkpoint_dir / "tokenizer"
    weights_dir.mkdir(parents=True, exist_ok=True); tokenizer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_dir / "model.pt")
    torch.save({
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "completed_epoch": completed_epoch, "global_step": global_step,
        "python_random_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_generator_state": data_generator.get_state(),
    }, checkpoint_dir / "training_state.pt")
    (weights_dir / "model_config.json").write_text(json.dumps(model_config.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save(tokenizer_dir / "tokenizer.json")
    (checkpoint_dir / "manifest.json").write_text(json.dumps({"format_version": 1, "architecture": "hyperai-torch-v1", "weights": "weights/model.pt", "tokenizer": "tokenizer/tokenizer.json", "context_window": model_config.context_window, "vocab_size": model_config.vocab_size}, indent=2), encoding="utf-8")
    (checkpoint_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (checkpoint_dir / "training_config.json").write_text(json.dumps(asdict(training_config), ensure_ascii=False, indent=2), encoding="utf-8")
    # Dizinler Windows'ta doğrudan replace edilemez. Önce eski sürümü geri
    # alınabilir bir ada taşır, sonra eksiksiz staging dizinini yerine koyarız.
    previous = output_dir / f".{name}.previous-{uuid.uuid4().hex}"
    try:
        if target.exists():
            _replace_directory_with_retry(target, previous)
        _replace_directory_with_retry(checkpoint_dir, target)
    except Exception:
        if previous.exists() and not target.exists():
            _replace_directory_with_retry(previous, target)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return target


def _load_resume(resume_from: Path, model: HyperAILanguageModel, optimizer: AdamW, scheduler: LambdaLR, scaler, data_generator: torch.Generator) -> tuple[int, int, dict[str, object]]:
    state_path, weights_path = resume_from / "training_state.pt", resume_from / "weights" / "model.pt"
    if not state_path.is_file() or not weights_path.is_file():
        raise CorpusError("Resume checkpoint'i training_state.pt ve weights/model.pt içermelidir.")
    if state_path.stat().st_size > MAX_RESUME_STATE_BYTES or weights_path.stat().st_size > MAX_RESUME_STATE_BYTES:
        raise CorpusError("Resume checkpoint dosyası güvenli yükleme sınırını (2 GiB) aşıyor.")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    if scaler.is_enabled() and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    # CPU -> GPU ve GPU -> CPU devamlarında AMP scaler'ı taşınmaz; geri kalan
    # durum yine de aynen geri yüklenir.
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    if "data_generator_state" in state:
        data_generator.set_state(state["data_generator_state"])
    report_path = resume_from / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {"epochs": []}
    return int(state["completed_epoch"]), int(state["global_step"]), report


def _resume_artifacts(path: Path):
    config_path, tokenizer_path = path / "weights" / "model_config.json", path / "tokenizer" / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise CorpusError("Resume checkpoint model_config.json ve tokenizer.json içermelidir.")
    return HyperAIModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8"))), load_tokenizer(tokenizer_path)


def _load_initial_checkpoint(path: Path, model: HyperAILanguageModel) -> None:
    """Ön-eğitim ağırlıklarını optimizer durumunu taşımadan sohbet eğitimine al."""
    weights_path = path / "weights" / "model.pt"
    if not weights_path.is_file():
        raise CorpusError("Başlangıç checkpoint'inde weights/model.pt bulunamadı.")
    if weights_path.stat().st_size > MAX_RESUME_STATE_BYTES:
        raise CorpusError("Başlangıç ağırlıkları güvenli 2 GiB yükleme sınırını aşıyor.")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)


def _prune_epoch_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted((path for path in output_dir.glob("epoch-*") if path.is_dir()), key=lambda path: path.name)
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def _split_examples(examples: list[tuple[str, str]], validation_fraction: float, seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Aynı sohbet çiftini bütünüyle tek tarafa koyarak validation sızıntısını önler."""
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for example in examples:
        groups.setdefault(example, []).append(example)
    if len(groups) < 2:
        raise CorpusError("Train/validation ayrımı için en az iki farklı sohbet çifti gerekir.")
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)
    target = max(1, round(len(examples) * validation_fraction))
    validation_groups: list[list[tuple[str, str]]] = []
    validation_size = 0
    for group in grouped:
        if validation_size < target:
            validation_groups.append(group)
            validation_size += len(group)
    train_groups = grouped[len(validation_groups) :]
    if not train_groups:
        # En küçük grubu validation'da bırakıp diğerlerini eğitime ayır.
        train_groups = [validation_groups.pop()]
    validation = [example for group in validation_groups for example in group]
    train = [example for group in train_groups for example in group]
    if not validation or not train:
        raise CorpusError("Gruplu train/validation ayrımı üretilemedi.")
    return train, validation


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_evaluation_prompts(path: str | None) -> list[str]:
    if path is None:
        return []
    source = Path(path)
    if not source.is_file():
        raise CorpusError(f"Değerlendirme prompt dosyası bulunamadı: {source}")
    prompts = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not prompts:
        raise CorpusError("Değerlendirme prompt dosyasında kullanılabilir prompt yok.")
    return prompts


def _append_metric(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("global_step", "epoch", "phase", "loss", "learning_rate"))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _write_loss_chart(metrics_path: Path, chart_path: Path) -> None:
    """Ek grafik kütüphanesi olmadan eğitim loss'undan küçük bir SVG üretir."""
    if not metrics_path.is_file():
        return
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row.get("phase") == "train"]
    if len(train_rows) < 2:
        return
    values = [float(row["loss"]) for row in train_rows]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    points = " ".join(f"{40 + 720 * index / (len(values) - 1):.1f},{270 - 230 * (value - low) / span:.1f}" for index, value in enumerate(values))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320" viewBox="0 0 800 320"><rect width="100%" height="100%" fill="white"/><text x="40" y="25" font-family="sans-serif" font-size="16">HyperAI training loss</text><line x1="40" y1="270" x2="760" y2="270" stroke="#666"/><line x1="40" y1="40" x2="40" y2="270" stroke="#666"/><polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}"/><text x="4" y="48" font-family="sans-serif" font-size="12">{high:.4f}</text><text x="4" y="272" font-family="sans-serif" font-size="12">{low:.4f}</text></svg>'''
    temporary = chart_path.with_suffix(".svg.tmp")
    temporary.write_text(svg, encoding="utf-8")
    os.replace(temporary, chart_path)


def _console_print(message: str) -> None:
    """Kütüphane doğrudan çağrılsa da Windows kod sayfasında log çökmesin."""
    encoding = sys.stdout.encoding or "utf-8"
    # Eğitim uzun sürebilir. Windows/CMD yönlendirmesinde çıktının tamponda
    # kalması, çalışan bir GPU koşusunu donmuş gibi gösteriyordu.
    print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def train_from_corpus(corpus_path: str | Path, output_dir: str | Path, model_config: HyperAIModelConfig, training_config: TrainingConfig, resume_from: str | Path | None = None, init_from: str | Path | None = None) -> Path:
    """JSONL sohbet verisini asistan cevabı odaklı loss ile eğitir.

    ``init_from`` ön-eğitim ağırlıkları ve aynı tokenizer ile yeni bir sohbet
    koşusu başlatır; ``resume_from`` ise mevcut sohbet koşusunun optimizer ve
    scheduler durumunu da koruyarak devam eder. İkisi aynı anda kullanılamaz.
    """
    corpus = Path(corpus_path)
    _console_print(f"Sohbet verisi yükleniyor: {corpus}")
    examples = load_chat_pairs(corpus)
    _console_print(f"{len(examples):,} sohbet çifti yüklendi; train/validation ayrılıyor.")
    train_examples, validation_examples = _split_examples(examples, training_config.validation_fraction, training_config.seed)
    root = Path(output_dir)
    resume_path = Path(resume_from) if resume_from else None
    init_path = Path(init_from) if init_from else None
    if resume_path and init_path:
        raise CorpusError("--resume ve --init-checkpoint aynı anda kullanılamaz.")
    if root.exists() and any(root.iterdir()) and not resume_path and not training_config.allow_overwrite:
        raise CorpusError(f"Çıktı klasörü zaten dolu: {root}. Yeni koşu için farklı --output kullanın; bilerek değiştirmek için --overwrite verin.")
    if resume_path and not resume_path.is_dir():
        raise CorpusError(f"Resume checkpoint'i bulunamadı: {resume_path}")
    if init_path and not init_path.is_dir():
        raise CorpusError(f"Başlangıç checkpoint'i bulunamadı: {init_path}")
    if resume_path:
        model_config, tokenizer = _resume_artifacts(resume_path)
    elif init_path:
        model_config, tokenizer = _resume_artifacts(init_path)
    else:
        texts = [format_chat_example(user, assistant) for user, assistant in train_examples]
        tokenizer = ByteTokenizer() if training_config.tokenizer_kind == "byte" else BPETokenizer.train(texts, training_config.tokenizer_vocab_size)
        tokenizer_vocab_size = tokenizer.VOCAB_SIZE if isinstance(tokenizer, ByteTokenizer) else tokenizer.vocab_size
        model_config = HyperAIModelConfig(**{**model_config.to_dict(), "vocab_size": tokenizer_vocab_size})

    def tokenise(conversations: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
        tokens: list[int] = []; labels: list[int] = []
        for user, assistant in conversations:
            prompt_ids = tokenizer.encode(format_chat_prompt(user))
            answer_ids = tokenizer.encode(" " + assistant)
            tokens.extend([ByteTokenizer.BOS_ID, *prompt_ids, *answer_ids, ByteTokenizer.EOS_ID])
            labels.extend([IGNORE_INDEX] * (1 + len(prompt_ids)) + answer_ids + [ByteTokenizer.EOS_ID])
        return tokens, labels
    _console_print("Sohbet çiftleri token dizilerine çevriliyor; büyük korpuslarda bu aşama birkaç dakika sürebilir.")
    train_ids, train_labels = tokenise(train_examples); validation_ids, validation_labels = tokenise(validation_examples)
    _console_print(f"Tokenizasyon tamamlandı: train={len(train_ids):,}, validation={len(validation_ids):,} token.")
    train_set = TokenSequenceDataset(train_ids, train_labels, model_config.context_window, training_config.sequence_stride)
    validation_set = TokenSequenceDataset(validation_ids, validation_labels, model_config.context_window, training_config.sequence_stride)
    random.seed(training_config.seed); torch.manual_seed(training_config.seed)
    device = _select_device(training_config.device); amp = device.type == "cuda"
    if amp: torch.cuda.manual_seed_all(training_config.seed)
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(train_set, batch_size=training_config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=training_config.batch_size, shuffle=False)
    model = HyperAILanguageModel(model_config).to(device); optimizer = _optimizer(model, training_config)
    scheduler = _scheduler(optimizer, training_config.epochs * len(train_loader), training_config); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, global_step, previous_report = (0, 0, None)
    if resume_path:
        start_epoch, global_step, previous_report = _load_resume(resume_path, model, optimizer, scheduler, scaler, generator); model.to(device)
    elif init_path:
        _load_initial_checkpoint(init_path, model); model.to(device)
    evaluation_prompts = _load_evaluation_prompts(training_config.evaluation_prompts_path)
    report: dict[str, object] = {
        "device": str(device), "amp": amp, "tokenizer": type(tokenizer).__name__, "vocab_size": model_config.vocab_size,
        "train_examples": len(train_examples), "validation_examples": len(validation_examples), "unique_examples": len(set(examples)),
        "train_tokens": len(train_ids), "validation_tokens": len(validation_ids), "epochs": [],
        "provenance": {"corpus_path": str(corpus), "corpus_sha256": _file_sha256(corpus), "git_commit": _git_commit(), "torch_version": torch.__version__, "python_version": sys.version, "init_checkpoint": str(init_path) if init_path else None},
    }
    if previous_report: report["epochs"] = previous_report.get("epochs", [])
    best_loss = min((float(item["validation_loss"]) for item in report["epochs"]), default=float("inf"))  # type: ignore[index]
    for epoch in range(start_epoch + 1, training_config.epochs + 1):
        model.train(); total_loss = 0.0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True); targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp): logits = model(inputs.to(device)); loss = _loss(logits, targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            # AMP taşması olursa GradScaler optimizer.step'i atlar. O durumda
            # scheduler'ı ilerletmek PyTorch uyarısına ve LR'ın model güncellemesi
            # olmadan kaymasına neden olur; yalnızca gerçek optimizer adımından
            # sonra scheduler.step çağırıyoruz.
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            optimizer_updated = not amp or scaler.get_scale() >= previous_scale
            if optimizer_updated:
                scheduler.step()
                global_step += 1
            total_loss += loss.item()
            if batch_index % training_config.log_every_steps == 0 or batch_index == len(train_loader):
                _console_print(
                    f"Epoch {epoch}/{training_config.epochs} | batch {batch_index:,}/{len(train_loader):,} "
                    f"| loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )
            _append_metric(root / "metrics" / "loss.csv", {"global_step": global_step, "epoch": epoch, "phase": "train", "loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})
        validation_loss = _evaluate(model, validation_loader, device, amp)
        runtime = HyperAIRuntime(model, tokenizer, model_config.context_window)
        sample = runtime.generate(format_chat_prompt(training_config.sample_prompt), GenerationSettings(max_new_tokens=48, temperature=0.8, top_p=0.95, seed=training_config.seed, stop_token_ids=frozenset({ByteTokenizer.EOS_ID})))
        _append_metric(root / "metrics" / "loss.csv", {"global_step": global_step, "epoch": epoch, "phase": "validation", "loss": validation_loss, "learning_rate": scheduler.get_last_lr()[0]})
        evaluations = []
        for prompt in evaluation_prompts:
            output = runtime.generate(format_chat_prompt(prompt), GenerationSettings(max_new_tokens=48, temperature=0.8, top_p=0.95, seed=training_config.seed, stop_token_ids=frozenset({ByteTokenizer.EOS_ID})))
            evaluations.append({"prompt": prompt, "response": output.text})
        if evaluations:
            evaluation_path = root / "evaluations" / f"epoch-{epoch:03}.jsonl"
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = evaluation_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in evaluations), encoding="utf-8")
            os.replace(temporary, evaluation_path)
        _write_loss_chart(root / "metrics" / "loss.csv", root / "metrics" / "loss.svg")
        epoch_report = {"epoch": epoch, "train_loss": total_loss / len(train_loader), "validation_loss": validation_loss, "learning_rate": scheduler.get_last_lr()[0], "sample_prompt": training_config.sample_prompt, "sample_response": sample.text, "evaluation_count": len(evaluations)}
        report["epochs"].append(epoch_report)  # type: ignore[index]
        _write_checkpoint(root, f"epoch-{epoch:03}", model, tokenizer, model_config, training_config, report, optimizer, scheduler, scaler, epoch, global_step, generator); _prune_epoch_checkpoints(root, training_config.keep_last_checkpoints)
        if validation_loss < best_loss:
            best_loss = validation_loss; _write_checkpoint(root, "best", model, tokenizer, model_config, training_config, report, optimizer, scheduler, scaler, epoch, global_step, generator)
        _console_print(f"Epoch {epoch}/{training_config.epochs} | train_loss={epoch_report['train_loss']:.4f} | val_loss={validation_loss:.4f} | lr={epoch_report['learning_rate']:.2e} | sample={sample.text!r}")
    return root / "best"
