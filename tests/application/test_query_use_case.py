import pytest
from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase


class DummyVectorStore:
    def add(self, texts, metadatas): ...
    def search(self, query_embedding, top_k):
        return []

    def persist(self): ...


class DummyEmbedding:
    def embed_texts(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 4


def test_query_use_case_contract():
    uc = QueryKnowledgeBase(vector_store=DummyVectorStore(), embedding=DummyEmbedding())
    with pytest.raises(NotImplementedError):
        uc.execute(QueryRequest(question="Was ist BU?"))
