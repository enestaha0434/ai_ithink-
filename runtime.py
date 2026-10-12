"""Yerel HyperAI modeli için üretim döngüsü."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .contracts import AutoregressiveModel, GenerationResult, GenerationSettings, Tokenizer


class HyperAIRuntime:
    """Tokenizer ve senin eğittiğin modeli bir araya getirir.

    Bu sınıf hiçbir model indirmez, ağa bağlanmaz ve eğitim yapmaz.
    """

    def __init__(self, model: AutoregressiveModel, tokenizer: Tokenizer, context_window: int) -> None:
        if context_window < 1:
            raise ValueError("context_window en az 1 olmalıdır.")
        if not isinstance(model, AutoregressiveModel):
            raise TypeError("Model next_token_logits(token_ids) metodunu uygulamalıdır.")
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("Tokenizer encode(text) ve decode(token_ids) metodlarını uygulamalıdır.")
        self.model = model
        self.tokenizer = tokenizer
        self.context_window = context_window

    def generate(self, prompt: str, settings: GenerationSettings | None = None) -> GenerationResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt metin olmalıdır.")
        settings = settings or GenerationSettings()
        # Eğitim örnekleri BOS ile başladığı için inference başlangıcı da aynı
        # dağılımdan gelmelidir.
        encoded_prompt = self.tokenizer.encode(prompt)
        if not encoded_prompt:
            raise ValueError("Prompt tokenizer tarafından boş üretildi.")
        if not hasattr(self.tokenizer, "BOS_ID"):
            raise TypeError("Sohbet tokenizer'ı BOS_ID özel tokenını tanımlamalıdır.")
        prompt_tokens = [self.tokenizer.BOS_ID, *encoded_prompt]

        tokens = list(prompt_tokens[-self.context_window :])
        produced: list[int] = []
        rng = random.Random(settings.seed)
        stopped = False
        cached_decode = hasattr(self.model, "prefill") and hasattr(self.model, "decode_with_cache")
        if cached_decode:
            logits, cache = self.model.prefill(tokens)  # type: ignore[attr-defined]
        for _ in range(settings.max_new_tokens):
            if not cached_decode:
                logits = self.model.next_token_logits(tokens[-self.context_window :])
            adjusted = _apply_repetition_penalty(logits, produced, settings.repetition_penalty)
            forbidden = _repeated_ngram_candidates(produced, settings.no_repeat_ngram_size)
            # EOS her zaman izinli kalır: tekrar filtresi modelin doğru yerde
            # susmasını engellememelidir.
            forbidden.difference_update(settings.stop_token_ids)
            token_id = _sample(adjusted, settings.temperature, settings.top_p, rng, forbidden)
            if token_id in settings.stop_token_ids:
                stopped = True
                break
            tokens.append(token_id)
            produced.append(token_id)
            if cached_decode and len(tokens) < self.context_window:
                logits, cache = self.model.decode_with_cache(token_id, cache)  # type: ignore[attr-defined]
            elif cached_decode and len(tokens) >= self.context_window:
                # Cache sabit bağlam penceresini geçtiğinde güvenli biçimde yeniden kurulur.
                tokens = tokens[-self.context_window :]
                logits, cache = self.model.prefill(tokens)  # type: ignore[attr-defined]

        return GenerationResult(
            text=self.tokenizer.decode(produced),
            prompt_token_count=len(prompt_tokens),
            generated_token_ids=tuple(produced),
            stopped=stopped,
        )


def _apply_repetition_penalty(logits: Sequence[float], produced: Sequence[int], penalty: float) -> list[float]:
    """Daha önce üretilen tokenların logitini hafifçe azaltır.

    Bu bir doğruluk veya muhakeme mekanizması değildir; küçük modelin aynı kısa
    parçayı sonsuza kadar yazmasını önleyen çıkarım katmanı korumasıdır.
    """
    adjusted = list(logits)
    if penalty == 1:
        return adjusted
    for token_id in set(produced):
        if 0 <= token_id < len(adjusted):
            adjusted[token_id] = adjusted[token_id] / penalty if adjusted[token_id] >= 0 else adjusted[token_id] * penalty
    return adjusted


def _repeated_ngram_candidates(produced: Sequence[int], size: int) -> set[int]:
    """Sıradaki token eklenirse yinelenecek n-gramların son tokenlarını bulur."""
    if size < 2 or len(produced) < size - 1:
        return set()
    prefix = tuple(produced[-(size - 1) :])
    return {produced[index + size - 1] for index in range(len(produced) - size + 1) if tuple(produced[index : index + size - 1]) == prefix}


def _sample(logits: Sequence[float], temperature: float, top_p: float, rng: random.Random, forbidden: set[int] | None = None) -> int:
    if len(logits) < 2:
        raise ValueError("Model en az iki token için logit döndürmelidir.")
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("Model sonlu logit değerleri döndürmelidir.")

    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    blocked = forbidden or set()
    ranked = sorted(((token_id, weight) for token_id, weight in enumerate(weights) if token_id not in blocked), key=lambda item: item[1], reverse=True)
    if not ranked:
        raise ValueError("Tüm tokenlar tekrar filtresi tarafından engellendi.")

    candidates: list[tuple[int, float]] = []
    cumulative = 0.0
    for token_id, weight in ranked:
        probability = weight / total
        candidates.append((token_id, probability))
        cumulative += probability
        if cumulative >= top_p:
            break

    threshold = rng.random()
    running = 0.0
    normalization = sum(probability for _, probability in candidates)
    for token_id, probability in candidates:
        running += probability / normalization
        if threshold <= running:
            return token_id
    return candidates[-1][0]
