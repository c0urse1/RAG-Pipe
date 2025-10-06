"""Tests for query DTOs."""

from bu_superagent.application.dto.query_dto import QueryRequest, RAGAnswer
from bu_superagent.domain.models import Citation


class TestQueryRequest:
    def test_query_request_defaults(self) -> None:
        """QueryRequest should have sensible defaults."""
        req = QueryRequest(question="What is RAG?")

        assert req.question == "What is RAG?"
        assert req.top_k == 5
        assert req.mmr is True
        assert req.mmr_lambda == 0.5
        assert req.confidence_threshold == 0.35
        assert req.use_reranker is False  # Default: disabled for performance
        assert req.pre_rerank_k == 20  # Default candidate pool for reranking

    def test_query_request_custom_values(self) -> None:
        """QueryRequest should accept custom values."""
        req = QueryRequest(
            question="Test question",
            top_k=10,
            mmr=False,
            mmr_lambda=0.8,
            confidence_threshold=0.5,
            use_reranker=False,
        )

        assert req.question == "Test question"
        assert req.top_k == 10
        assert req.mmr is False
        assert req.mmr_lambda == 0.8
        assert req.confidence_threshold == 0.5
        assert req.use_reranker is False

    def test_query_request_is_frozen(self) -> None:
        """QueryRequest should be immutable."""
        import pytest

        req = QueryRequest(question="Test")

        with pytest.raises(AttributeError):
            req.top_k = 10  # type: ignore[misc]


class TestRAGAnswer:
    def test_rag_answer_creation(self) -> None:
        """RAGAnswer should store text and citations."""
        citations = [
            Citation(chunk_id="c1", source="doc1.pdf", score=0.9),
            Citation(chunk_id="c2", source="doc2.pdf", score=0.8),
        ]
        answer = RAGAnswer(text="The answer is 42.", citations=citations)

        assert answer.text == "The answer is 42."
        assert len(answer.citations) == 2
        assert answer.citations[0].chunk_id == "c1"
        assert answer.citations[1].source == "doc2.pdf"

    def test_rag_answer_empty_citations(self) -> None:
        """RAGAnswer should allow empty citations list."""
        answer = RAGAnswer(text="Low confidence answer", citations=[])

        assert answer.text == "Low confidence answer"
        assert answer.citations == []

    def test_rag_answer_is_frozen(self) -> None:
        """RAGAnswer should be immutable."""
        import pytest

        answer = RAGAnswer(text="Test", citations=[])

        with pytest.raises(AttributeError):
            answer.text = "Modified"  # type: ignore[misc]
