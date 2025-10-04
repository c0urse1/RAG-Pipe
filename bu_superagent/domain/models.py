"""Domain models for RAG pipeline data structures.

These are portable, testable domain POJOs used by the application layer
to orchestrate retrieval, ranking, and citation logic without depending
on infrastructure details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk retrieved from the vector store with similarity score."""

    id: str
    text: str
    metadata: dict[str, Any]
    vector: list[float] | None
    score: float  # similarity score from vector store


@dataclass(frozen=True)
class RankedChunk:
    """A retrieved chunk with its reranking position."""

    chunk: RetrievedChunk
    rank: int


@dataclass(frozen=True)
class Citation:
    """Citation reference for a generated answer."""

    chunk_id: str
    source: str
    score: float
