from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RerankerPort(ABC):
    @abstractmethod
    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        ...
