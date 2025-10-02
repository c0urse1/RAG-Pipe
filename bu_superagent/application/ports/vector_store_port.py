from typing import Protocol, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict


class VectorStorePort(Protocol):
    def upsert(
        self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
    ) -> None: ...

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def ensure_collection(self, name: str, dim: int) -> None: ...
