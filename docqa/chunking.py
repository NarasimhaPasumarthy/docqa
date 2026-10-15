"""Split documents into paragraph-aligned chunks with a small overlap."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    position: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Break a paragraph longer than max_chars on sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        # A single sentence longer than the limit gets hard-wrapped.
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str, source: str, max_chars: int = 1200, overlap_chars: int = 200
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most max_chars.

    When a chunk is closed, its last paragraph is repeated at the start of the next one
    if it is short (<= overlap_chars), so a heading or lead-in sentence keeps its context.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    paragraphs: list[str] = []
    for para in _PARAGRAPH_BREAK.split(text):
        para = para.strip()
        if para:
            paragraphs.extend(_split_long(para, max_chars))

    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        if current and size + len(para) > max_chars:
            groups.append(current)
            tail = current[-1]
            carry = len(tail) <= overlap_chars and len(tail) + 2 + len(para) <= max_chars
            current = [tail] if carry else []
            size = len(tail) + 2 if carry else 0
        current.append(para)
        size += len(para) + 2
    if current:
        groups.append(current)

    return [
        Chunk(id=f"{source}#{i}", source=source, position=i, text="\n\n".join(group))
        for i, group in enumerate(groups)
    ]
