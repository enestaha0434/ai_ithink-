"""HyperAI'nin RoPE, SDPA ve KV cache destekli causal Transformer modeli."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KVCache = tuple[Tensor, Tensor]
PastKeyValues = tuple[KVCache | None, ...]


@dataclass(frozen=True)
class HyperAIModelConfig:
    vocab_size: int = 260
    context_window: int = 512
    model_dim: int = 384
    num_layers: int = 8
    num_heads: int = 6
    mlp_multiplier: int = 4
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.vocab_size < 8 or self.context_window < 2:
            raise ValueError("vocab_size en az 8, context_window en az 2 olmalıdır.")
        # Checkpoint metadatası yerel JSON'dan geldiği için, yanlışlıkla devasa
        # tensör ayırmayı engelleyen makul bir üst sınır da uygularız.
        if self.vocab_size > 131_072 or self.context_window > 16_384:
            raise ValueError("vocab_size veya context_window güvenli üst sınırı aşıyor.")
        if self.model_dim < 8 or self.model_dim % self.num_heads or (self.model_dim // self.num_heads) % 2:
            raise ValueError("model_dim num_heads'a bölünmeli ve head boyutu çift olmalıdır.")
        if self.num_layers < 1 or self.num_layers > 128 or self.mlp_multiplier < 1 or self.mlp_multiplier > 16 or self.rope_theta <= 1:
            raise ValueError("Model konfigürasyonu geçersiz.")
        if self.model_dim > 8_192 or self.num_heads > 128:
            raise ValueError("Model boyutu güvenli üst sınırı aşıyor.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout [0, 1) aralığında olmalıdır.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "HyperAIModelConfig":
        expected = set(cls.__dataclass_fields__)
        unexpected, missing = set(raw) - expected, expected - set(raw)
        if unexpected or missing:
            details = (["eksik: " + ", ".join(sorted(missing))] if missing else []) + (["bilinmeyen: " + ", ".join(sorted(unexpected))] if unexpected else [])
            raise ValueError("Geçersiz model config: " + "; ".join(details))
        return cls(**raw)  # type: ignore[arg-type]


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(variance + self.epsilon).to(values.dtype) * self.weight


def _apply_rope(values: Tensor, positions: Tensor, theta: float) -> Tensor:
    """[batch, heads, sequence, head_dim] tensörüne Rotary Position Embedding uygular."""
    head_dim = values.size(-1)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
    cos, sin = angles.cos().to(values.dtype)[None, None, :, :], angles.sin().to(values.dtype)[None, None, :, :]
    even, odd = values[..., 0::2], values[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: HyperAIModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.rope_theta = config.rope_theta
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.dropout = config.dropout

    def forward(self, values: Tensor, past_key_value: KVCache | None = None, use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        batch, sequence_length, model_dim = values.shape
        qkv = self.qkv(values).view(batch, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = (item.transpose(1, 2) for item in qkv.unbind(dim=2))
        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)
        positions = torch.arange(past_length, past_length + sequence_length, device=values.device)
        query, key = _apply_rope(query, positions, self.rope_theta), _apply_rope(key, positions, self.rope_theta)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None

        if past_length == 0:
            attention = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            query_positions = torch.arange(past_length, past_length + sequence_length, device=values.device)[:, None]
            key_positions = torch.arange(key.size(-2), device=values.device)[None, :]
            causal_mask = key_positions <= query_positions
            attention = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask, dropout_p=self.dropout if self.training else 0.0)
        output = attention.transpose(1, 2).contiguous().view(batch, sequence_length, model_dim)
        return self.output(output), present


class SwiGLU(nn.Module):
    def __init__(self, config: HyperAIModelConfig) -> None:
        super().__init__()
        hidden_dim = config.model_dim * config.mlp_multiplier
        self.gate, self.value = nn.Linear(config.model_dim, hidden_dim, bias=False), nn.Linear(config.model_dim, hidden_dim, bias=False)
        self.output, self.dropout = nn.Linear(hidden_dim, config.model_dim, bias=False), nn.Dropout(config.dropout)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(self.output(F.silu(self.gate(values)) * self.value(values)))


class TransformerBlock(nn.Module):
    def __init__(self, config: HyperAIModelConfig) -> None:
        super().__init__()
        self.attention_norm, self.attention = RMSNorm(config.model_dim), CausalSelfAttention(config)
        self.mlp_norm, self.mlp = RMSNorm(config.model_dim), SwiGLU(config)

    def forward(self, values: Tensor, past_key_value: KVCache | None = None, use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        attention, present = self.attention(self.attention_norm(values), past_key_value, use_cache)
        values = values + attention
        return values + self.mlp(self.mlp_norm(values)), present


class HyperAILanguageModel(nn.Module):
    """Sıfırdan eğitilen, RoPE ve opsiyonel KV cache kullanan HyperAI modeli."""

    def __init__(self, config: HyperAIModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm, self.output = RMSNorm(config.model_dim), nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, past_key_values: PastKeyValues | None = None, use_cache: bool = False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids şekli [batch, sequence] olmalıdır.")
        _, sequence_length = input_ids.shape
        past_length = 0 if not past_key_values or past_key_values[0] is None else past_key_values[0][0].size(-2)
        if not 1 <= sequence_length or past_length + sequence_length > self.config.context_window:
            raise ValueError(f"Toplam dizi uzunluğu 1..{self.config.context_window} arasında olmalıdır.")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        # Token aralık kontrolü eğitim döngüsünde yapılır; burada min/max .item()
        # çağrıları GPU'yu her adımda CPU ile senkronize edeceği için yapılmaz.
        values, presents = self.token_embedding(input_ids), []
        for index, block in enumerate(self.blocks):
            prior = None if past_key_values is None else past_key_values[index]
            values, present = block(values, prior, use_cache)
            if use_cache:
                presents.append(present)
        logits = self.output(self.final_norm(values))
        return (logits, tuple(presents)) if use_cache else logits

    @torch.inference_mode()
    def prefill(self, token_ids: Sequence[int]) -> tuple[list[float], PastKeyValues]:
        if not token_ids:
            raise ValueError("Model bağlamı boş olamaz.")
        device = next(self.parameters()).device
        self.eval()
        logits, cache = self(torch.tensor([list(token_ids)[-self.config.context_window :]], device=device), use_cache=True)
        return logits[0, -1].float().cpu().tolist(), cache

    @torch.inference_mode()
    def decode_with_cache(self, token_id: int, cache: PastKeyValues) -> tuple[list[float], PastKeyValues]:
        device = next(self.parameters()).device
        if cache[0] is not None and cache[0][0].size(-2) >= self.config.context_window:
            raise ValueError("KV cache context window'a ulaştı; yeniden prefill gerekir.")
        self.eval()
        logits, updated = self(torch.tensor([[token_id]], device=device), past_key_values=cache, use_cache=True)
        return logits[0, -1].float().cpu().tolist(), updated

    @torch.inference_mode()
    def next_token_logits(self, token_ids: Sequence[int]) -> list[float]:
        logits, _ = self.prefill(token_ids)
        return logits
