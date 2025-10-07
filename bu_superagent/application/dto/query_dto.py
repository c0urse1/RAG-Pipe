# bu_superagent/application/dto/query_dto.py
from __future__ import annotations

from dataclasses import dataclass

from bu_superagent.domain.models import Citation


@dataclass(frozen=True)
class QueryRequest:
    """
    DTO for querying the knowledge base.

    - question: user question (non-empty)
    - top_k: number of chunks to return for answering
    - pre_rerank_k: how many candidates to fetch before dedup and optional rerank
    - use_reranker: whether to apply the optional cross-encoder reranker (if provided)
    - confidence_threshold: minimum top score required to proceed with answering
    """

    question: str
    top_k: int = 5
    pre_rerank_k: int = 20
    use_reranker: bool = False
    confidence_threshold: float = 0.30


@dataclass(frozen=True)
class RAGAnswer:
    """Complete RAG answer with generated text and citations."""

    text: str
    citations: list[Citation]
