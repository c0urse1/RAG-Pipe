from __future__ import annotations

from typing import Sequence

from ...application.ports.embedding_port import EmbeddingPort


class SimpleHashEmbedding(EmbeddingPort):
    """Deterministic tiny embedding using character hashing.

    This is a placeholder; swap with sentence-transformers later.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0] * self.dim
        if not text:
            return [0.0] * self.dim
        for i, ch in enumerate(text):
            vec[(i + ord(ch)) % self.dim] += 1
        # L2 normalize
        import math

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embed(text)

    def embed_texts(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self._embed(t) for t in texts]
