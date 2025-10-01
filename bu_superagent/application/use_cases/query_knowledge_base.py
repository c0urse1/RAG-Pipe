from dataclasses import dataclass

from ..dto.query_dto import QueryRequest
from ..ports.embedding_port import EmbeddingPort
from ..ports.vector_store_port import VectorStorePort


@dataclass
class QueryKnowledgeBase:
    vector_store: VectorStorePort
    embedding: EmbeddingPort

    def execute(self, req: QueryRequest) -> None:
        """Orchestriert Domain (später: Dedup, MMR, Reranking). Keine direkten Infra-Imports."""
        # Platzhalter – keine Implementierung
        raise NotImplementedError
