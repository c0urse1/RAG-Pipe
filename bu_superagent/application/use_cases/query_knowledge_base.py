"""Query knowledge base use case with confidence-gate and RAG pipeline."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from bu_superagent.application.dto.query_dto import QueryRequest, RAGAnswer
from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.llm_port import LLMPort
from bu_superagent.application.ports.reranker_port import RerankerPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.domain.errors import (
    LowConfidenceError,
    RerankerError,
    RetrievalError,
    ValidationError,
)
from bu_superagent.domain.models import Citation, RetrievedChunk
from bu_superagent.domain.services.ranking import deduplicate_by_cosine, mmr, passes_confidence
from bu_superagent.domain.services.reranking import sort_by_scores_desc

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """Result type for explicit error handling without exceptions.

    Avoids silent failures and makes success/failure paths explicit.
    """

    ok: bool
    value: T | None = None
    error: E | None = None

    @staticmethod
    def success(val: T) -> "Result[T, E]":
        """Create a successful result."""
        return Result(ok=True, value=val)

    @staticmethod
    def failure(err: E) -> "Result[T, E]":
        """Create a failed result."""
        return Result(ok=False, error=err)


class QueryKnowledgeBase:
    """Orchestrates: validate → embed → search → [rerank] → dedup/MMR → confidence → LLM.

    Why (SAM): The use case orchestrates ports, applies pure domain services
    (reranking, MMR/dedup), enforces the confidence gate (business rule), and never
    calls infra libraries directly. The Result type avoids silent failures.

    RAG Pipeline:
    1. Validate input
    2. Embed query (EmbeddingPort)
    3. Retrieve candidates (VectorStorePort) - oversample for reranking/MMR
    4. Optional cross-encoder reranking (RerankerPort) - BEFORE dedup/MMR
    5. Domain post-processing: dedup + optional MMR
    6. Confidence gate check
    7. LLM generation or extractive fallback

    Confidence-Gate reference: Under threshold → escalate with complete context.
    """

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort | None = None,
        reranker: RerankerPort | None = None,
    ) -> None:
        """Initialize query use case with required ports.

        Args:
            embedding: Port for embedding text queries
            vector_store: Port for vector similarity search
            llm: Optional port for answer generation (extractive fallback if None)
            reranker: Optional port for cross-encoder reranking (2-stage RAG)
        """
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    def _build_prompt(self, q: str, chunks: list[RetrievedChunk]) -> str:
        """Build LLM prompt with question and retrieved context.

        Args:
            q: The user's question
            chunks: Retrieved and ranked chunks for context

        Returns:
            Formatted prompt string with instructions and context
        """
        parts = []
        for i, c in enumerate(chunks):
            source = c.metadata.get("source_path", c.metadata.get("source", "?"))
            parts.append(f"[{i+1}] {c.text}\nSOURCE={source}")
        context = "\n\n".join(parts)
        # Keep prompt minimal & explicit for grounded answers with references
        return (
            "You are a careful assistant. Answer the question strictly using the provided context. "
            "Cite sources as [#] using the index of the snippets.\n\n"
            f"Question:\n{q}\n\nContext:\n{context}\n\nAnswer:"
        )

    def execute(self, req: QueryRequest) -> Result[RAGAnswer, Exception]:
        """Execute RAG query pipeline with optional reranking and confidence gating.

        Pipeline stages:
        1. Validate input
        2. Embed query
        3. Retrieve candidates (oversample for reranking/MMR)
        4. Optional cross-encoder reranking (if use_reranker=True and port provided)
        5. Domain post-processing (dedup + optional MMR)
        6. Confidence gate check
        7. LLM generation (or extractive fallback)
        8. Shape citations

        Args:
            req: Query request with question and parameters

        Returns:
            Result containing either RAGAnswer on success or Exception on failure
        """
        # 1) Validate
        if not req.question or req.top_k <= 0:
            return Result.failure(ValidationError("question must be non-empty and top_k>0"))

        # 2) Embed query (EmbeddingPort should map infra errors already)
        try:
            q_vec = self.embedding.embed_query(req.question)
        except Exception as e:  # noqa: BLE001
            return Result.failure(e)

        # 3) Retrieve candidates (oversample for reranking/MMR)
        # Use larger pool if reranking to give cross-encoder more candidates
        search_k = max(req.top_k * 4, req.pre_rerank_k if req.use_reranker else 20)
        try:
            candidates = self.vector_store.search(q_vec, top_k=search_k)
        except Exception as e:  # noqa: BLE001
            return Result.failure(e)

        if not candidates:
            return Result.failure(RetrievalError("no candidates returned from vector store"))

        # 4) Optional cross-encoder reranking (BEFORE dedup/MMR for best accuracy)
        if req.use_reranker and self.reranker is not None and candidates:
            try:
                texts = [c.text for c in candidates]
                scores = self.reranker.score(req.question, texts)
                # Reorder candidates by cross-encoder scores (pure domain function)
                candidates = sort_by_scores_desc(candidates, scores)
            except Exception as e:  # noqa: BLE001
                # Wrap infrastructure errors in domain error
                return Result.failure(RerankerError(detail=f"Cross-encoder reranking failed: {e}"))

        # 5) Domain post-processing: dedup + optional MMR
        deduped = deduplicate_by_cosine(candidates, threshold=0.95)
        final = mmr(deduped, q_vec, req.top_k, req.mmr_lambda) if req.mmr else deduped[: req.top_k]

        # 5) Confidence gate (domain)
        ok, score = passes_confidence(final, req.confidence_threshold)
        if not ok:
            return Result.failure(
                LowConfidenceError(
                    message="confidence below threshold; escalate to human review",
                    top_score=score,
                    threshold=req.confidence_threshold,
                )
            )

        # 6) LLM answer generation (if LLMPort provided)
        if self.llm is None:
            # Extractive fallback (no LLM): return concatenated snippet with citations
            citations = [
                Citation(
                    chunk_id=c.id,
                    source=c.metadata.get("source_path", c.metadata.get("source", "?")),
                    score=c.score,
                )
                for c in final
            ]
            text = " ".join(c.text for c in final)
            return Result.success(RAGAnswer(text=text, citations=citations))

        # Generate answer with LLM
        try:
            prompt = self._build_prompt(req.question, final)
            llm_resp = self.llm.generate(prompt)
        except Exception as e:  # noqa: BLE001
            return Result.failure(e)

        # 7) Shape citations from final retrieval set
        citations = [
            Citation(
                chunk_id=c.id,
                source=c.metadata.get("source_path", c.metadata.get("source", "?")),
                score=c.score,
            )
            for c in final
        ]
        return Result.success(RAGAnswer(text=llm_resp, citations=citations))
