from __future__ import annotations

from ..infrastructure.vectorstore.chroma_vector_store import InMemoryVectorStore
from ..application.use_cases.ingest_documents import IngestDocumentsUseCase
from ..application.use_cases.query_knowledge_base import QueryKnowledgeBase

"""
# Composition Root: injiziert konkrete Adapter in Use-Cases.
# Wichtig: keine Geschäftslogik, nur Wiring.
"""


class _NoopEmbedder:
    def embed_texts(self, texts):  # type: ignore[no-untyped-def]
        return []

    def embed_query(self, text):  # type: ignore[no-untyped-def]
        return []


def build_application():
    embedder = _NoopEmbedder()
    vs = InMemoryVectorStore()
    ingest = IngestDocumentsUseCase(embedder, vs)
    query = QueryKnowledgeBase(vector_store=vs, embedding=embedder)
    return {"ingest": ingest, "query": query}


def build_query_use_case():  # retained for future wiring steps
    raise NotImplementedError
