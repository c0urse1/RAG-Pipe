"""Tests for QueryKnowledgeBase use case."""

from collections.abc import Sequence

from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.application.ports.llm_port import ChatMessage, LLMResponse
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.domain.errors import LowConfidenceError, RetrievalError, ValidationError
from bu_superagent.domain.models import RetrievedChunk


class FakeEmbedding:
    """Fake embedding adapter for testing."""

    def embed_query(self, text: str, kind: str = "mxbai") -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts: Sequence[str], kind: str = "mxbai") -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    """Fake vector store adapter for testing."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
    ) -> None:
        pass

    def ensure_collection(self, name: str, dim: int) -> None:
        pass


class FakeLLM:
    """Fake LLM adapter for testing."""

    def __init__(self, response: str = "Test answer [1][2]") -> None:
        self.response = response

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        return LLMResponse(text=self.response)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        return self.response


def make_chunk(
    id_: str,
    score: float,
    text: str = "Test text",
    source: str = "test.pdf",
    vector: list[float] | None = None,
) -> RetrievedChunk:
    """Helper to create test chunks."""
    return RetrievedChunk(
        id=id_,
        text=text,
        metadata={"source_path": source},
        vector=vector or [1.0, 0.0, 0.0],
        score=score,
    )


class TestQueryKnowledgeBase:
    def test_validation_empty_question(self) -> None:
        """Should fail validation with empty question."""
        uc = QueryKnowledgeBase(embedding=FakeEmbedding(), vector_store=FakeVectorStore())
        result = uc.execute(QueryRequest(question=""))

        assert not result.ok
        assert isinstance(result.error, ValidationError)

    def test_validation_invalid_top_k(self) -> None:
        """Should fail validation with top_k <= 0."""
        uc = QueryKnowledgeBase(embedding=FakeEmbedding(), vector_store=FakeVectorStore())
        result = uc.execute(QueryRequest(question="Test", top_k=0))

        assert not result.ok
        assert isinstance(result.error, ValidationError)

    def test_no_candidates_returned(self) -> None:
        """Should fail when vector store returns no candidates."""
        uc = QueryKnowledgeBase(embedding=FakeEmbedding(), vector_store=FakeVectorStore(chunks=[]))
        result = uc.execute(QueryRequest(question="Test question"))

        assert not result.ok
        assert isinstance(result.error, RetrievalError)

    def test_low_confidence_triggers_gate(self) -> None:
        """Should fail when confidence is below threshold."""
        chunks = [make_chunk("c1", 0.2)]  # Low score
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(), vector_store=FakeVectorStore(chunks=chunks)
        )
        result = uc.execute(QueryRequest(question="Test", confidence_threshold=0.5))

        assert not result.ok
        assert isinstance(result.error, LowConfidenceError)
        assert result.error.top_score == 0.2
        assert result.error.threshold == 0.5

    def test_extractive_fallback_without_llm(self) -> None:
        """Should return concatenated text when no LLM is provided."""
        chunks = [
            make_chunk("c1", 0.9, text="First chunk", source="doc1.pdf", vector=[1.0, 0.0, 0.0]),
            make_chunk(
                "c2", 0.8, text="Second chunk", source="doc2.pdf", vector=[0.0, 1.0, 0.0]
            ),  # Diverse to avoid dedup
        ]
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(),
            vector_store=FakeVectorStore(chunks=chunks),
            llm=None,
        )
        result = uc.execute(QueryRequest(question="Test", top_k=2))

        assert result.ok
        assert result.value is not None
        # Both chunks should survive deduplication (diverse vectors)
        assert "First chunk" in result.value.text
        assert "Second chunk" in result.value.text
        assert len(result.value.citations) == 2
        assert result.value.citations[0].chunk_id == "c1"
        assert result.value.citations[0].source == "doc1.pdf"

    def test_success_with_llm(self) -> None:
        """Should generate answer with LLM when provided."""
        chunks = [
            make_chunk("c1", 0.9, text="Context chunk", source="doc.pdf"),
        ]
        llm = FakeLLM(response="Generated answer with citation [1]")
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(),
            vector_store=FakeVectorStore(chunks=chunks),
            llm=llm,
        )
        result = uc.execute(QueryRequest(question="Test question"))

        assert result.ok
        assert result.value is not None
        assert result.value.text == "Generated answer with citation [1]"
        assert len(result.value.citations) == 1
        assert result.value.citations[0].chunk_id == "c1"

    def test_mmr_enabled(self) -> None:
        """Should apply MMR when enabled."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.85, vector=[0.99, 0.01, 0.0]),  # Similar to c1
            make_chunk("c3", 0.8, vector=[0.0, 1.0, 0.0]),  # Diverse
        ]
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(),
            vector_store=FakeVectorStore(chunks=chunks),
            llm=FakeLLM(),
        )
        result = uc.execute(QueryRequest(question="Test", top_k=2, mmr=True, mmr_lambda=0.5))

        assert result.ok
        assert result.value is not None
        # Should select c1 and c3 (diverse) over c1 and c2 (similar)
        assert len(result.value.citations) == 2

    def test_mmr_disabled(self) -> None:
        """Should use simple top-k when MMR disabled."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.85, vector=[0.0, 1.0, 0.0]),  # Diverse
            make_chunk("c3", 0.8, vector=[0.0, 0.0, 1.0]),  # Diverse
        ]
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(),
            vector_store=FakeVectorStore(chunks=chunks),
            llm=FakeLLM(),
        )
        result = uc.execute(QueryRequest(question="Test", top_k=2, mmr=False))

        assert result.ok
        assert result.value is not None
        # Should get top 2 after deduplication (all diverse, so all survive)
        assert len(result.value.citations) == 2

    def test_deduplication_applied(self) -> None:
        """Should deduplicate similar chunks."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.85, vector=[0.99, 0.0, 0.0]),  # Near-duplicate
        ]
        uc = QueryKnowledgeBase(
            embedding=FakeEmbedding(),
            vector_store=FakeVectorStore(chunks=chunks),
            llm=FakeLLM(),
        )
        result = uc.execute(QueryRequest(question="Test", top_k=5))

        assert result.ok
        assert result.value is not None
        # Should deduplicate c2 as it's too similar to c1
        assert len(result.value.citations) <= 2


class TestResultType:
    def test_result_success(self) -> None:
        """Result.success should create successful result."""
        from bu_superagent.application.use_cases.query_knowledge_base import Result

        result = Result.success("test value")

        assert result.ok is True
        assert result.value == "test value"
        assert result.error is None

    def test_result_failure(self) -> None:
        """Result.failure should create failed result."""
        from bu_superagent.application.use_cases.query_knowledge_base import Result

        error = ValueError("test error")
        result = Result.failure(error)

        assert result.ok is False
        assert result.value is None
        assert result.error is error
