from __future__ import annotations

from typing import Sequence

from ...domain.services.relevance_scoring import cosine_similarity
from ..dto.query_dto import QueryDTO
from ..ports.embedding_port import EmbeddingPort
from ..ports.reranker_port import RerankerPort
from ..ports.vector_store_port import VectorStorePort


class QueryKnowledgeBaseUseCase:
    def __init__(self, embedder: EmbeddingPort, vector_store: VectorStorePort, reranker: RerankerPort | None = None) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def execute(self, query: QueryDTO) -> list[dict]:
        q_emb = self.embedder.embed_text(query.text)
        candidates = self.vector_store.query(q_emb, top_k=query.top_k)

        results = [
            {"chunk_id": cid, "score": score, "metadata": meta}
            for cid, score, meta in candidates
        ]

        if self.reranker and candidates:
            texts = [c[2].get("text", "") for c in candidates]
            rerank_scores = self.reranker.score(query.text, texts)
            for r, s in zip(results, rerank_scores):
                # combine naive: average
                r["score"] = (r["score"] + s) / 2.0
            results.sort(key=lambda x: x["score"], reverse=True)

        return results
