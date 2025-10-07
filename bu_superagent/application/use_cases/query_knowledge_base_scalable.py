"""Query knowledge base use case with hybrid fusion and confidence gate.

Why: Multi-stage Retrieval (Vector → optional Hybrid RRF → optional MMR),
     Confidence-Gate als letzte Schranke.
"""

from typing import Any

from bu_superagent.application.dtos import QueryRequest
from bu_superagent.application.scalable_ports import EmbeddingPort, VectorStorePort
from bu_superagent.domain.errors import (
    DomainError,
    LowConfidenceError,
    RetrievalError,
    ValidationError,
)
from bu_superagent.domain.services.ranking import mmr, passes_confidence, rrf
from bu_superagent.domain.types import Result, Vector


class QueryKnowledgeBaseScalable:
    """Scalable query orchestration with hybrid fusion and confidence gating.

    Pipeline:
    1. Validate input (question, top_k)
    2. Embed query vector
    3. Vector search (oversample for fusion)
    4. Optional hybrid: RRF fusion with lexical search
    5. Optional MMR for diversity
    6. Confidence gate check
    7. Return results or LowConfidenceError

    Why: Multi-stage retrieval enables hybrid search, MMR diversity,
         and confidence-based answer quality control.
    """

    def __init__(
        self,
        embed: EmbeddingPort,
        vs: VectorStorePort,
        lexical: VectorStorePort | None = None,
    ) -> None:
        """Initialize with embedding, vector store, and optional lexical search.

        Args:
            embed: Port for text embedding
            vs: Port for vector similarity search
            lexical: Optional port for lexical/keyword search (hybrid mode)
        """
        self.embed = embed
        self.vs = vs
        self.lexical = lexical

    def execute(self, req: QueryRequest) -> Result[dict[str, Any], DomainError]:
        """Execute query pipeline with hybrid fusion and confidence gate.

        Args:
            req: QueryRequest with question, top_k, flags

        Returns:
            Result with answer dict or DomainError
        """
        # Validate input
        if not req.question or req.top_k <= 0:
            return Result.failure(ValidationError("invalid input"))

        # Embed query
        r_vec = self.embed.embed_texts([req.question])
        if not r_vec.ok:
            assert r_vec.error is not None
            return Result.failure(r_vec.error)
        assert r_vec.value is not None
        qv: Vector = r_vec.value[0]

        # Vector search (oversample for fusion/MMR)
        r_nn = self.vs.search(req.collection, qv, top_k=max(req.top_k * 4, 20))
        if not r_nn.ok:
            assert r_nn.error is not None
            return Result.failure(r_nn.error)
        assert r_nn.value is not None
        vec_hits = r_nn.value  # [{"id": "...", "score": float, "meta": {...}}]

        # Optional hybrid fusion with lexical search
        if req.use_hybrid and self.lexical:
            # Lexical search returns ranked ids with ranks
            r_lex = self.lexical.search(req.collection, qv, top_k=max(req.top_k * 4, 20))
            if not r_lex.ok:
                assert r_lex.error is not None
                return Result.failure(r_lex.error)
            assert r_lex.value is not None

            # RRF fusion: combine vector and lexical rankings
            fused = rrf(
                ranks=[
                    [(h["id"], i + 1) for i, h in enumerate(vec_hits)],
                    [(h["id"], i + 1) for i, h in enumerate(r_lex.value)],
                ]
            )
            fused_ids = [doc_id for doc_id, _ in fused][: max(req.top_k * 2, 10)]
            vec_hits = [h for h in vec_hits if h["id"] in fused_ids]

        # Sort by score descending
        vec_hits.sort(key=lambda x: x["score"], reverse=True)

        # Optional MMR for diversity
        if req.use_mmr:
            mmr_input = [(h["id"], h["score"]) for h in vec_hits]
            mmr_out = mmr(mmr_input, k=req.top_k)
            ids = {id_ for id_, _ in mmr_out}
            final = [h for h in vec_hits if h["id"] in ids][: req.top_k]
        else:
            final = vec_hits[: req.top_k]

        # Validate results
        if not final:
            return Result.failure(RetrievalError("no candidates"))

        # Confidence gate
        top_score = float(final[0]["score"])
        if not passes_confidence(top_score, req.confidence_threshold):
            return Result.failure(
                LowConfidenceError(
                    message=f"top={top_score}, thr={req.confidence_threshold}",
                    top_score=top_score,
                    threshold=req.confidence_threshold,
                )
            )

        return Result.success({"answers": final})
