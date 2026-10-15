"""Turn files (Markdown, plain text, PDF) into plain text."""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


class UnsupportedFileType(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(
            f"{filename}: unsupported type; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield supported files from a mix of file and directory paths, in a stable order."""
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield child
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(path)
