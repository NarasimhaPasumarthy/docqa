"""A small, dependency-free BM25 index with JSON persistence.

BM25 needs no embedding model, vector database or GPU, and it is deterministic, which makes
retrieval quality cheap to measure in CI. The Retriever surface (upsert/search) is narrow
enough to swap in a vector or hybrid index later.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its me my of on or "
    "our so than that the their them there these they this to was we what when where "
    "which who why will with you your".split()
)


def _stem(token: str) -> str:
    """Very light stemming: fold common plural/verb suffixes so 'days' matches 'day'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._rebuild()

    def __len__(self) -> int:
        return len(self._chunks)

    def upsert_source(self, source: str, chunks: list[Chunk]) -> None:
        """Replace every chunk of `source` with `chunks`, so re-ingesting a file is idempotent."""
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.source != source}
        self._chunks.update((c.id, c) for c in chunks)
        self._rebuild()

    def remove_source(self, source: str) -> int:
        before = len(self._chunks)
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.source != source}
        self._rebuild()
        return before - len(self._chunks)

    def sources(self) -> dict[str, int]:
        return dict(Counter(c.source for c in self._chunks.values()))

    def _rebuild(self) -> None:
        # The source name is indexed too, so "security policy" finds security-policy.md.
        self._tf = {
            cid: Counter(tokenize(f"{c.source} {c.text}")) for cid, c in self._chunks.items()
        }
        self._lengths = {cid: sum(tf.values()) for cid, tf in self._tf.items()}
        self._avg_len = sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        self._df: Counter[str] = Counter()
        for tf in self._tf.values():
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self._chunks), self._df[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not terms or not self._chunks:
            return []
        scored = []
        for cid, tf in self._tf.items():
            norm = self.k1 * (1 - self.b + self.b * self._lengths[cid] / self._avg_len)
            score = sum(
                self._idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append(ScoredChunk(self._chunks[cid], score))
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "chunks": [c.to_dict() for c in self._chunks.values()]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a corrupt index

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        index = cls()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            index._chunks = {c["id"]: Chunk(**c) for c in payload["chunks"]}
            index._rebuild()
        return index
