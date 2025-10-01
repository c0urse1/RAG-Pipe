import pytest
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.application.dto.query_dto import QueryRequest


class _NoopEmbedder:
    def embed_texts(self, texts):  # type: ignore[no-untyped-def]
        return []

    def embed_query(self, text):  # type: ignore[no-untyped-def]
        return []


class _NoopVectorStore:
    def add(self, texts, metadatas):  # type: ignore[no-untyped-def]
        pass

    def search(self, query_embedding, top_k):  # type: ignore[no-untyped-def]
        return []

    def persist(self):  # type: ignore[no-untyped-def]
        pass


def test_query_placeholder_raises():
    uc = QueryKnowledgeBase(vector_store=_NoopVectorStore(), embedding=_NoopEmbedder())
    with pytest.raises(NotImplementedError):
        uc.execute(QueryRequest(question="beta", top_k=2))
