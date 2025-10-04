from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Import domain model and re-export for convenience
from bu_superagent.domain.models import RetrievedChunk

__all__ = ["RetrievedChunk", "VectorStorePort"]


@runtime_checkable
class VectorStorePort(Protocol):
    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None: ...

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def ensure_collection(self, name: str, dim: int) -> None: ...
