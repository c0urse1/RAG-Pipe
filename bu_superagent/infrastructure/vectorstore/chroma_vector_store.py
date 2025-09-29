from __future__ import annotations

from typing import Sequence, Tuple

from ...application.ports.vector_store_port import VectorStorePort


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, ids: Sequence[str], embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict] | None = None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        for i, e, m in zip(ids, embeddings, metadatas):
            self._store[str(i)] = (list(e), dict(m))
            self._store[str(i)][1].setdefault("text", m.get("text", ""))

    def query(self, embedding: Sequence[float], top_k: int) -> list[tuple[str, float, dict]]:
        # naive cosine on stored embeddings
        import math

        def cos(a: Sequence[float], b: Sequence[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return (dot / (na * nb) + 1.0) / 2.0

        scored = [
            (k, cos(v[0], embedding), v[1])
            for k, v in self._store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
