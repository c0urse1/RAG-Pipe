from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> Sequence[float]:
        ...

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[Sequence[float]]:
        ...
