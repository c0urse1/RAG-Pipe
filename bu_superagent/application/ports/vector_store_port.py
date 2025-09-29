from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from ...domain.types import DocumentId, ChunkId


class VectorStorePort(ABC):
    @abstractmethod
    def upsert(self, ids: Sequence[ChunkId], embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict] | None = None) -> None:
        ...

    @abstractmethod
    def query(self, embedding: Sequence[float], top_k: int) -> list[tuple[ChunkId, float, dict]]:
        ...
