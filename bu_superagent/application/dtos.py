"""Application DTOs for ingest and query with hybrid support.

Why: Saubere Input-Verträge; Query erweitert um Hybrid/Reranker Flags.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class IngestRequest:
    """Request for document ingestion with sharding support."""

    collection: str
    shard_key: str
    docs: list[dict[str, Any]]  # each: {"id": str, "text": str, "meta": dict}


@dataclass
class QueryRequest:
    """Request for RAG query with retrieval and generation parameters."""

    collection: str
    question: str
    top_k: int = 5
    use_mmr: bool = True
    use_reranker: bool = False
    use_hybrid: bool = False
    confidence_threshold: float = 0.25
