from typing import Protocol, Sequence, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Mapping[str, object]


class VectorStorePort(Protocol):
    def add(self, texts: Sequence[str], metadatas: Sequence[Mapping[str, object]]) -> None: ...
    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence[RetrievedChunk]: ...
    def persist(self) -> None: ...
