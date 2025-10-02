from typing import Literal, Protocol
from collections.abc import Sequence

EmbeddingKind = Literal["e5", "mxbai", "jina"]


class EmbeddingPort(Protocol):
    def embed_texts(
        self, texts: Sequence[str], kind: EmbeddingKind = "mxbai"
    ) -> list[list[float]]: ...
    def embed_query(self, text: str, kind: EmbeddingKind = "mxbai") -> list[float]: ...
