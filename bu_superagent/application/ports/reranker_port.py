"""Reranker port for semantic reranking of retrieved candidates.

Why (SAM): Application defines the interface (port), infrastructure provides
concrete adapters. Port returns Result type for explicit error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class RerankerPort(ABC):
    """Port for semantic reranking of query-passage pairs.

    Cross-encoders score query-passage pairs directly for higher accuracy
    than bi-encoder similarity. Used in 2-stage RAG: retrieve → rerank.
    """

    @abstractmethod
    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Score query-candidate pairs for relevance.

        Args:
            query: Query text
            candidates: Candidate texts to score against query

        Returns:
            List of scores (higher = more relevant). Length matches candidates.
            Scores may be raw logits or normalized probabilities depending on adapter.

        Raises:
            RuntimeError: If scoring fails (wrapped to RerankerError by use case)

        Note:
            Infrastructure adapters handle actual scoring. Use case wraps errors
            in domain RerankerError via try/except for proper error layering.
        """
        ...
