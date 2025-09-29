from __future__ import annotations

from .settings import Settings
from ..infrastructure.embeddings.hf_sentence_transformers import SimpleHashEmbedding
from ..infrastructure.vectorstore.chroma_vector_store import InMemoryVectorStore
from ..infrastructure.reranking.cross_encoder import SimpleOverlapReranker
from ..application.use_cases.ingest_documents import IngestDocumentsUseCase
from ..application.use_cases.query_knowledge_base import QueryKnowledgeBaseUseCase


def build_application():
    settings = Settings()
    embedder = SimpleHashEmbedding(dim=settings.embedding_dim)
    vs = InMemoryVectorStore()
    reranker = SimpleOverlapReranker()

    ingest = IngestDocumentsUseCase(embedder, vs)
    query = QueryKnowledgeBaseUseCase(embedder, vs, reranker)

    return {"ingest": ingest, "query": query}
