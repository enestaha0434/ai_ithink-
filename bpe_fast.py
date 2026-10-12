"""Hızlı, ön-bölmeli byte-level BPE (hyperai-byte-bpe-v2).

- Metin önce kelime/boşluk/noktalama parçalarına bölünür; birleştirmeler bu sınırları aşmaz.
  Böylece "HyperAI:" gibi bir prompt sonu, tam metindeki aynı yerle aynı token'lara ayrışır.
- Eğitim, kelime frekans tablosu üzerinde çalışır (tüm metin üzerinde değil), bu yüzden
  büyük korpusta bile dakikalar sürer.
- Kayıpsızdır: decode(encode(x)) == x (geçerli UTF-8 metin için).
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

PATTERN = r" ?[^\W\d_]+| ?\d| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
_REGEX = re.compile(PATTERN)


class FastBPETokenizer:
    PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
    BYTE_OFFSET = 4
    BASE_VOCAB_SIZE = BYTE_OFFSET + 256
    TYPE = "hyperai-byte-bpe-v2"
    CACHE_LIMIT = 1_000_000

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_table: dict[int, bytes] = {
            i: bytes([i - self.BYTE_OFFSET]) for i in range(self.BYTE_OFFSET, self.BASE_VOCAB_SIZE)
        }
        for index, (left, right) in enumerate(self.merges):
            if left not in self.token_table or right not in self.token_table:
                raise ValueError("BPE merge listesi geçersiz token içeriyor.")
            self.token_table[self.BASE_VOCAB_SIZE + index] = self.token_table[left] + self.token_table[right]
        self.vocab_size = self.BASE_VOCAB_SIZE + len(self.merges)
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ encode / decode
    def _encode_word(self, word: str) -> tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [byte + self.BYTE_OFFSET for byte in word.encode("utf-8")]
        ranks = self.merge_ranks
        while len(ids) > 1:
            best_rank, best_pos = None, -1
            for position in range(len(ids) - 1):
                rank = ranks.get((ids[position], ids[position + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, position
            if best_rank is None:
                break
            ids[best_pos : best_pos + 2] = [self.BASE_VOCAB_SIZE + best_rank]
        result = tuple(ids)
        if len(self._cache) < self.CACHE_LIMIT:
            self._cache[word] = result
        return result

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text metin olmalıdır.")
        tokens: list[int] = [self.BOS_ID] if add_bos else []
        for word in _REGEX.findall(text):
            tokens.extend(self._encode_word(word))
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = bytearray()
        for token_id in token_ids:
            if token_id in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            token = self.token_table.get(token_id)
            raw.extend(token if token is not None else "�".encode("utf-8"))
        return bytes(raw).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 16384,
        min_frequency: int = 2,
        max_chars: int | None = None,
    ) -> "FastBPETokenizer":
        if vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size en az {cls.BASE_VOCAB_SIZE} olmalıdır.")
        word_counts: Counter[str] = Counter()
        seen = 0
        for text in texts:
            if max_chars is not None and seen >= max_chars:
                break
            seen += len(text)
            word_counts.update(_REGEX.findall(text))
        if not word_counts:
            raise ValueError("BPE eğitimi için metin yok.")

        words: list[list[int]] = []
        freqs: list[int] = []
        for word, count in word_counts.items():
            words.append([byte + cls.BYTE_OFFSET for byte in word.encode("utf-8")])
            freqs.append(count)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for index, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[index]
                pair_words[pair].add(index)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while cls.BASE_VOCAB_SIZE + len(merges) < vocab_size and heap:
            negative, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current != -negative:
                if current > 0:
                    heapq.heappush(heap, (-current, pair))
                continue
            if current < min_frequency:
                break
            new_id = cls.BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            left, right = pair
            touched: set[tuple[int, int]] = set()
            for index in list(pair_words.pop(pair, ())):
                word = words[index]
                merged: list[int] = []
                position, changed = 0, False
                while position < len(word):
                    if position + 1 < len(word) and word[position] == left and word[position + 1] == right:
                        merged.append(new_id)
                        position += 2
                        changed = True
                    else:
                        merged.append(word[position])
                        position += 1
                if not changed:
                    continue
                frequency = freqs[index]
                for old_pair in zip(word, word[1:]):
                    pair_counts[old_pair] -= frequency
                for new_pair in zip(merged, merged[1:]):
                    pair_counts[new_pair] += frequency
                    pair_words[new_pair].add(index)
                    touched.add(new_pair)
                words[index] = merged
            pair_counts.pop(pair, None)
            for new_pair in touched:
                count = pair_counts.get(new_pair, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, new_pair))
        return cls(merges)

    # ------------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"type": self.TYPE, "vocab_size": self.vocab_size, "pattern": PATTERN, "merges": self.merges},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FastBPETokenizer":
        try:
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
            merges = [tuple(pair) for pair in metadata["merges"]]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"BPE tokenizer dosyası okunamadı: {path}") from exc
        if metadata.get("type") != cls.TYPE or metadata.get("pattern") != PATTERN:
            raise ValueError("Bu dosya geçerli hyperai-byte-bpe-v2 tokenizer'ı değil.")
        tokenizer = cls(merges)
        if metadata.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Tokenizer dosyasındaki vocab_size merge listesiyle uyuşmuyor.")
        return tokenizer
