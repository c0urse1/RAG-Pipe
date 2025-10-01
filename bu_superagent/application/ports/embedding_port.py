from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
    def embed_query(self, text: str) -> Sequence[float]: ...
