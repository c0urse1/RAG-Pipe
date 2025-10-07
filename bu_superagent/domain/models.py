# bu_superagent/domain/models.py
# Domain models must be pure (no I/O, no external libs)
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Immutable domain entity that represents a retrieved passage/chunk.

    - id:        stable identifier for the chunk (document id + local id)
    - text:      the visible chunk text (may be empty if PII rules disable storage)
    - vector:    the L2-normalized embedding vector (tuple) or None if not available
    - metadata:  immutable metadata mapping (source, uri, page, section, etc.)
    - score:     similarity/relevance score set by retrieval/reranking (None if unset)

    NOTE: Domain layer must not import external libs. Keep types standard-library only.
    """

    id: str
    text: str
    vector: tuple[float, ...] | None
    metadata: Mapping[str, Any]
    score: float | None = None


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
