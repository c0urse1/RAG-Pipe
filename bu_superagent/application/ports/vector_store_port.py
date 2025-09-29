from typing import Protocol, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class VectorStorePort(Protocol):
    def add(self, texts: Sequence[str], metadatas: Sequence[dict]) -> None: ...
    def search(self, query_embedding: Sequence[float], top_k: int) -> Sequence[RetrievedChunk]: ...
    def persist(self) -> None: ...
