# bu_superagent/application/use_cases/query_knowledge_base.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.llm_port import LLMPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.domain.errors import (
    EmbeddingError,
    LLMError,
    LowConfidenceError,
    RetrievalError,
    ValidationError,
    VectorStoreError,
)
from bu_superagent.domain.models import RetrievedChunk
from bu_superagent.domain.result import Result
from bu_superagent.domain.services.ranking import deduplicate_by_cosine

if TYPE_CHECKING:
    # Optional: only if this port exists in your repo
    from bu_superagent.application.ports.reranker_port import RerankerPort


@dataclass(frozen=True)
class RAGAnswer:
    text: str
    sources: Sequence[RetrievedChunk]


class QueryKnowledgeBase:
    """
    Application Use-Case orchestrating the domain for querying the RAG KB.
    No I/O, uses only ports; handles errors via Result[T, E].
    """

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort | None = None,
        reranker: RerankerPort | None = None,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    def execute(self, req: QueryRequest) -> Result[RAGAnswer, Exception]:
        # 1) Validate
        if not req.question or not req.question.strip():
            return Result.failure(ValidationError("question must not be empty"))
        if req.top_k <= 0:
            return Result.failure(ValidationError("top_k must be > 0"))

        # 2) Embed query
        try:
            q_vec = self.embedding.embed_query(req.question)
        except Exception as ex:
            return Result.failure(EmbeddingError(f"embedding failed: {ex}"))

        # 3) Retrieve candidates (overfetch)
        candidate_k = max(req.pre_rerank_k, req.top_k * 4, 20)
        try:
            candidates = list(self.vector_store.search(q_vec, candidate_k))
        except Exception as ex:
            return Result.failure(VectorStoreError(f"vector search failed: {ex}"))

        if not candidates:
            return Result.failure(RetrievalError("no candidates returned"))

        # 4) Optional rerank (if provided and requested)
        if self.reranker and req.use_reranker:
            try:
                candidates = list(self.reranker.rerank(req.question, candidates))
            except Exception as ex:
                return Result.failure(VectorStoreError(f"rerank failed: {ex}"))

        # 5) Dedup & slice top_k
        deduped = deduplicate_by_cosine(candidates, threshold=0.95)
        final = deduped[: req.top_k]

        # 6) Confidence gate
        top_score = final[0].score if final and final[0].score is not None else 0.0
        if not final or top_score < req.confidence_threshold:
            return Result.failure(
                LowConfidenceError(
                    f"top score {top_score:.3f} below threshold {req.confidence_threshold:.3f}"
                )
            )

        # 7) Answer: extractive fallback or generative via LLM
        if self.llm is None:
            text = "\n\n".join(c.text for c in final if (c.text or "").strip())
            return Result.success(RAGAnswer(text=text, sources=final))

        try:
            ctx = "\n\n".join(
                f"[{i+1}] {c.text}" for i, c in enumerate(final) if (c.text or "").strip()
            )
            prompt = (
                "You are a helpful assistant. Use ONLY the provided context.\n\n"
                f"Question: {req.question}\n\n"
                f"Context:\n{ctx}\n\n"
                "Answer:"
            )
            text = self.llm.generate(prompt)
            return Result.success(RAGAnswer(text=text, sources=final))
        except Exception as ex:
            return Result.failure(LLMError(f"llm generation failed: {ex}"))
