from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True, frozen=True)
class Settings:
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))
