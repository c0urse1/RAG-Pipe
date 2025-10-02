"""
Sentence-Transformers embedding adapter (skeleton)

Implements EmbeddingPort using sentence-transformers. Kept as a placeholder
to avoid side-effects during import and to preserve current test behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...application.ports.embedding_port import EmbeddingKind


class SentenceTransformersEmbedding:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # lazy; load in first call in a real implementation

    def embed_texts(self, texts: Sequence[str], kind: EmbeddingKind = "mxbai") -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str, kind: EmbeddingKind = "mxbai") -> list[float]:
        raise NotImplementedError
