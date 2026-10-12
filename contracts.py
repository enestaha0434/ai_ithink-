"""HyperAI model ve tokenizer sözleşmeleri.

Bu dosya kasıtlı olarak JAX, PyTorch, Transformers veya haricî bir model
sağlayıcısına bağımlı değildir. Eğitim kodu hangi altyapıyı kullanırsa
kullansın bu küçük arayüzleri uygulayarak çekirdeğe bağlanabilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Sana ait tokenizer'ın sağlaması gereken en küçük arayüz."""

    def encode(self, text: str) -> list[int]:
        """Metni token kimliklerine dönüştürür."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Token kimliklerini metne dönüştürür."""


@runtime_checkable
class AutoregressiveModel(Protocol):
    """Eğitilmiş yerel model adaptörü.

    Model; JAX, PyTorch veya saf Python arkasında çalışabilir. Çekirdeğin
    ihtiyacı yalnızca bağlam için bir sonraki token logit'leridir.
    """

    def next_token_logits(self, token_ids: Sequence[int]) -> Sequence[float]:
        """Verilen bağlam için sözlük boyutunda ham logit döndürür."""


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: int | None = None
    stop_token_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens en az 1 olmalıdır.")
        if self.temperature <= 0 or self.repetition_penalty < 1:
            raise ValueError("temperature sıfırdan büyük olmalıdır.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 0 ile 1 arasında olmalıdır.")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size negatif olamaz.")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    stopped: bool


@dataclass(frozen=True)
class CheckpointManifest:
    """Bir eğitimin ürettiği checkpoint paketinin kamuya açık metadatası."""

    architecture: str
    weights_path: Path
    tokenizer_path: Path
    context_window: int
    vocab_size: int
    format_version: int = 1
