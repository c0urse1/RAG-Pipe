from dataclasses import dataclass
from ..ports.vector_store_port import VectorStorePort
from ..ports.embedding_port import EmbeddingPort
from ..dto.query_dto import QueryRequest


@dataclass
class QueryKnowledgeBase:
    vector_store: VectorStorePort
    embedding: EmbeddingPort

    def execute(self, req: QueryRequest):
        """Orchestriert Domain (später: Dedup, MMR, Reranking). Keine direkten Infra-Imports."""
        # Platzhalter – keine Implementierung
        raise NotImplementedError
