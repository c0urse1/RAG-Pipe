from dataclasses import dataclass
from typing import Protocol
from collections.abc import Sequence


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict[str, object]


class VectorStorePort(Protocol):
    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None: ...

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def ensure_collection(self, name: str, dim: int) -> None: ...
