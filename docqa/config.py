"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    index_path: Path = Path("data/index.json")
    top_k: int = 5
    seed_dir: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        seed = os.getenv("DOCQA_SEED_DIR")
        return cls(
            model=os.getenv("DOCQA_MODEL", cls.model),
            effort=os.getenv("DOCQA_EFFORT", cls.effort),
            index_path=Path(os.getenv("DOCQA_INDEX_PATH", str(cls.index_path))),
            top_k=int(os.getenv("DOCQA_TOP_K", str(cls.top_k))),
            seed_dir=Path(seed) if seed else None,
        )
