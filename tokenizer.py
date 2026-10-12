"""HyperAI'nin yerel byte ve eğitilebilir byte-level BPE tokenizer'ları."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


class _BaseByteTokenizer:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    BYTE_OFFSET = 4
    BASE_VOCAB_SIZE = BYTE_OFFSET + 256

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = bytearray()
        for token_id in token_ids:
            if token_id in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            token = self._token_bytes(token_id)
            raw.extend(token if token is not None else "�".encode("utf-8"))
        return bytes(raw).decode("utf-8", errors="replace")

    def _token_bytes(self, token_id: int) -> bytes | None:
        if self.BYTE_OFFSET <= token_id < self.BASE_VOCAB_SIZE:
            return bytes([token_id - self.BYTE_OFFSET])
        return None


class ByteTokenizer(_BaseByteTokenizer):
    """Eğitim gerektirmeyen, Unicode'u kayıpsız taşıyan UTF-8 tokenizer."""

    VOCAB_SIZE = _BaseByteTokenizer.BASE_VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text metin olmalıdır.")
        token_ids = [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]
        if add_bos:
            token_ids.insert(0, self.BOS_ID)
        if add_eos:
            token_ids.append(self.EOS_ID)
        return token_ids

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "hyperai-byte-v1", "vocab_size": self.VOCAB_SIZE}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        try:
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Tokenizer dosyası okunamadı: {path}") from exc
        if metadata != {"type": "hyperai-byte-v1", "vocab_size": cls.VOCAB_SIZE}:
            raise ValueError("Bu dosya hyperai-byte-v1 tokenizer'ı değil.")
        return cls()


class BPETokenizer(_BaseByteTokenizer):
    """Haricî kütüphane gerektirmeyen, Türkçe için eğitilebilir byte-level BPE."""

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_table: dict[int, bytes] = {
            token_id: bytes([token_id - self.BYTE_OFFSET])
            for token_id in range(self.BYTE_OFFSET, self.BASE_VOCAB_SIZE)
        }
        for index, (left, right) in enumerate(self.merges):
            if left not in self.token_table or right not in self.token_table:
                raise ValueError("BPE merge listesi geçersiz token içeriyor.")
            self.token_table[self.BASE_VOCAB_SIZE + index] = self.token_table[left] + self.token_table[right]
        self.vocab_size = self.BASE_VOCAB_SIZE + len(self.merges)

    def _token_bytes(self, token_id: int) -> bytes | None:
        return self.token_table.get(token_id)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text metin olmalıdır.")
        tokens: list[int] = []
        # Merge'ler kelime/boşluk parçası dışına taşmaz. Böylece eğitimdeki
        # "HyperAI: cevap" ile çıkarımdaki "HyperAI:" sınırı kararlı kalır.
        for piece in self._pretokenize(text):
            tokens.extend(self._merge_sequence([byte + self.BYTE_OFFSET for byte in piece.encode("utf-8")]))
        if add_bos:
            tokens.insert(0, self.BOS_ID)
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return re.findall(r"\S+|\s+", text, flags=re.UNICODE)

    def _merge_sequence(self, tokens: list[int]) -> list[int]:
        while len(tokens) > 1:
            ranked = [(self.merge_ranks[pair], position) for position, pair in enumerate(zip(tokens, tokens[1:])) if pair in self.merge_ranks]
            if not ranked:
                break
            _, position = min(ranked)
            merged_id = self.BASE_VOCAB_SIZE + self.merge_ranks[(tokens[position], tokens[position + 1])]
            tokens[position : position + 2] = [merged_id]
        return tokens

    @classmethod
    def train(cls, texts: Iterable[str], target_vocab_size: int = 8192, min_frequency: int = 2) -> "BPETokenizer":
        if target_vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError(f"target_vocab_size en az {cls.BASE_VOCAB_SIZE} olmalıdır.")
        # Korpusun tamamını her merge turunda taramak yerine özgün kelime/boşluk
        # parçalarının frekans tablosu üzerinde çalışır. Bu hem hızlıdır hem de
        # merge'lerin sohbet alan etiketlerini aşmasını engeller.
        sequences: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in cls._pretokenize(text):
                if piece:
                    sequences[tuple(byte + cls.BYTE_OFFSET for byte in piece.encode("utf-8"))] += 1
        if not sequences:
            raise ValueError("BPE eğitimi için metin yok.")
        merges: list[tuple[int, int]] = []
        while cls.BASE_VOCAB_SIZE + len(merges) < target_vocab_size:
            frequencies: Counter[tuple[int, int]] = Counter()
            for sequence, count in sequences.items():
                frequencies.update({pair: count for pair in zip(sequence, sequence[1:])})
            if not frequencies:
                break
            pair, frequency = frequencies.most_common(1)[0]
            if frequency < min_frequency:
                break
            merged_id = cls.BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            updated_sequences: Counter[tuple[int, ...]] = Counter()
            for sequence, count in sequences.items():
                position, updated = 0, []
                while position < len(sequence):
                    if position + 1 < len(sequence) and (sequence[position], sequence[position + 1]) == pair:
                        updated.append(merged_id)
                        position += 2
                    else:
                        updated.append(sequence[position])
                        position += 1
                updated_sequences[tuple(updated)] += count
            sequences = updated_sequences
        return cls(merges)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "hyperai-byte-bpe-v1", "vocab_size": self.vocab_size, "merges": self.merges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        try:
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
            merges = [tuple(pair) for pair in metadata["merges"]]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"BPE tokenizer dosyası okunamadı: {path}") from exc
        if len(merges) > 130_812:  # BASE_VOCAB_SIZE + merges <= model config sınırı
            raise ValueError("BPE merge sayısı güvenli üst sınırı aşıyor.")
        tokenizer = cls(merges)
        if metadata.get("type") != "hyperai-byte-bpe-v1" or metadata.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Bu dosya geçerli hyperai-byte-bpe-v1 tokenizer'ı değil.")
        return tokenizer


def load_tokenizer(path: str | Path):
    """Manifest'teki tokenizer dosyasını türüne göre yerelde yükler."""
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("type") == "hyperai-byte-v1":
        return ByteTokenizer.load(path)
    if metadata.get("type") == "hyperai-byte-bpe-v1":
        return BPETokenizer.load(path)
    if metadata.get("type") == "hyperai-byte-bpe-v2":
        from .bpe_fast import FastBPETokenizer  # büyük veri için hızlı, ön-bölmeli BPE

        return FastBPETokenizer.load(path)
    raise ValueError("Bilinmeyen HyperAI tokenizer türü.")
