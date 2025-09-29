from __future__ import annotations

from typing import Sequence

from ...application.ports.vector_store_port import VectorStorePort, RetrievedChunk


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._texts: list[str] = []
        self._metas: list[dict] = []

    def add(self, texts: Sequence[str], metadatas: Sequence[dict]) -> None:
        for t, m in zip(texts, metadatas):
            self._texts.append(str(t))
            self._metas.append(dict(m))

    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence[RetrievedChunk]:
        # Placeholder: no scoring logic yet; return empty list
        return []

    def persist(self) -> None:  # pragma: no cover - no-op for in-memory
        pass
