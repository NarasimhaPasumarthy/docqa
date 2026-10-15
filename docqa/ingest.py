"""Load, chunk and index documents."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .chunking import chunk_text
from .loaders import extract_text, iter_files
from .retriever import BM25Index


def ingest_document(index: BM25Index, source: str, text: str) -> int:
    """Index one document under `source`, replacing any earlier version. Returns chunk count."""
    chunks = chunk_text(text, source)
    if not chunks:
        raise ValueError(f"{source}: no extractable text")
    index.upsert_source(source, chunks)
    return len(chunks)


def ingest_paths(index: BM25Index, paths: Iterable[Path]) -> dict[str, int]:
    counts = {}
    for path in iter_files(paths):
        text = extract_text(path.name, path.read_bytes())
        counts[path.name] = ingest_document(index, path.name, text)
    return counts
