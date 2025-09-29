from __future__ import annotations

from .settings import Settings
from ..infrastructure.embeddings.hf_sentence_transformers import SimpleHashEmbedding
from ..infrastructure.vectorstore.chroma_vector_store import InMemoryVectorStore
from ..application.use_cases.ingest_documents import IngestDocumentsUseCase
from ..application.use_cases.query_knowledge_base import QueryKnowledgeBase


def build_application():
    settings = Settings()
    embedder = SimpleHashEmbedding(dim=settings.embedding_dim)
    vs = InMemoryVectorStore()

    ingest = IngestDocumentsUseCase(embedder, vs)
    query = QueryKnowledgeBase(vector_store=vs, embedding=embedder)

    return {"ingest": ingest, "query": query}
