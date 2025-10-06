"""Query DTOs for RAG pipeline requests and responses."""

from dataclasses import dataclass

from bu_superagent.domain.models import Citation


@dataclass(frozen=True)
class QueryRequest:
    """Request for RAG query with retrieval and generation parameters."""

    question: str
    top_k: int = 5
    mmr: bool = True  # Use Maximal Marginal Relevance for diversity
    mmr_lambda: float = 0.5  # Trade-off: 0=diversity, 1=relevance
    confidence_threshold: float = 0.35  # Minimum confidence for answering
    use_reranker: bool = False  # Apply cross-encoder reranking (if reranker port provided)
    pre_rerank_k: int = 20  # Candidate pool size for reranking (>= top_k; ignored if !use_reranker)


@dataclass(frozen=True)
class RAGAnswer:
    """Complete RAG answer with generated text and citations."""

    text: str
    citations: list[Citation]
