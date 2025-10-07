"""Tests for QueryKnowledgeBase use case with reranker integration.

Tests verify the application layer orchestration of the reranking step
in the RAG pipeline, using fake ports following the existing test patterns.
"""

from dataclasses import dataclass

from bu_superagent.application.dto.query_dto import QueryRequest, RAGAnswer
from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.reranker_port import RerankerPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.domain.models import RetrievedChunk

# --- Fake Ports for Testing ---


@dataclass
class FakeEmbedding(EmbeddingPort):
    """Fake embedding port returning fixed vectors."""

    def embed_query(self, text: str, kind: str = "mxbai") -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str], kind: str = "mxbai") -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass
class FakeVectorStore(VectorStorePort):
    """Fake vector store returning predefined chunks."""

    chunks: list[RetrievedChunk]

    def search(self, query_vec: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]

    def ensure_collection(self, name: str, dim: int) -> None:
        pass

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        pass


@dataclass
class FakeReranker(RerankerPort):
    """Fake reranker returning predefined scores or simulating failure."""

    scores: list[float]
    should_fail: bool = False

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if self.should_fail:
            raise RuntimeError("Fake reranker failure")
        return self.scores[: len(candidates)]


# --- Tests ---


def test_reranker_reorders_candidates():
    """Test that reranker reorders candidates by cross-encoder scores."""
    # Arrange: 4 chunks with initial retrieval order
    chunks = [
        RetrievedChunk(
            id=f"chunk{i}", text=f"text{i}", metadata={"source": f"doc{i}"}, score=0.5, vector=None
        )
        for i in range(4)
    ]
    vector_store = FakeVectorStore(chunks=chunks)

    # Fake reranker scores reorder: text2 > text0 > text1 > text3
    reranker = FakeReranker(scores=[0.5, 0.2, 0.9, 0.1])

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,  # Extractive mode
        reranker=reranker,
    )

    # Act: Query with reranking enabled, request top 2
    req = QueryRequest(
        question="test query",
        top_k=2,
        use_reranker=True,
        pre_rerank_k=4,
        confidence_threshold=0.0,  # Bypass confidence gate
    )
    result = uc.execute(req)

    # Assert: Reranker reordered chunks (text2, text0 are top 2)
    assert result.ok
    answer = result.value
    assert isinstance(answer, RAGAnswer)
    # Check citations preserve reranked order
    assert len(answer.citations) == 2
    assert answer.citations[0].chunk_id == "chunk2"  # Highest score (0.9)
    assert answer.citations[1].chunk_id == "chunk0"  # Second highest (0.5)


def test_reranker_disabled_uses_retrieval_order():
    """Test that disabling reranker preserves vector search ranking."""
    chunks = [
        RetrievedChunk(
            id=f"chunk{i}",
            text=f"text{i}",
            metadata={"source": f"doc{i}"},
            score=0.5 - i * 0.1,
            vector=None,
        )
        for i in range(3)
    ]
    vector_store = FakeVectorStore(chunks=chunks)

    # Reranker present but disabled
    reranker = FakeReranker(scores=[0.9, 0.5, 0.1])

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,
        reranker=reranker,
    )

    req = QueryRequest(
        question="test",
        top_k=2,
        use_reranker=False,  # Disabled
        confidence_threshold=0.0,
    )
    result = uc.execute(req)

    assert result.ok
    answer = result.value
    # Should preserve retrieval order (chunk0, chunk1)
    assert answer.citations[0].chunk_id == "chunk0"
    assert answer.citations[1].chunk_id == "chunk1"


def test_reranker_failure_returns_error():
    """Test that reranker failures are properly wrapped in domain error."""
    chunks = [RetrievedChunk(id="c1", text="text", metadata={}, score=0.5, vector=None)]
    vector_store = FakeVectorStore(chunks=chunks)
    reranker = FakeReranker(scores=[], should_fail=True)

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,
        reranker=reranker,
    )

    req = QueryRequest(question="test", top_k=1, use_reranker=True, confidence_threshold=0.0)
    result = uc.execute(req)

    # Assert: Failure wrapped in RerankerError
    assert not result.ok
    assert result.error is not None
    from bu_superagent.domain.errors import RerankerError

    assert isinstance(result.error, RerankerError)
    assert "fake reranker failure" in result.error.detail.lower()


def test_reranker_none_with_flag_skips_reranking():
    """Test that use_reranker=True with no reranker port skips reranking gracefully."""
    chunks = [
        RetrievedChunk(id=f"c{i}", text=f"t{i}", metadata={}, score=0.5, vector=None)
        for i in range(3)
    ]
    vector_store = FakeVectorStore(chunks=chunks)

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,
        reranker=None,  # No reranker port
    )

    req = QueryRequest(
        question="test",
        top_k=2,
        use_reranker=True,  # Flag enabled but port is None
        confidence_threshold=0.0,
    )
    result = uc.execute(req)

    # Assert: Should succeed without reranking
    assert result.ok
    assert len(result.value.citations) == 2


def test_reranker_with_empty_candidates():
    """Test that empty candidate list is handled gracefully."""
    vector_store = FakeVectorStore(chunks=[])
    reranker = FakeReranker(scores=[])

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,
        reranker=reranker,
    )

    req = QueryRequest(question="test", top_k=5, use_reranker=True, confidence_threshold=0.0)
    result = uc.execute(req)

    # Assert: Should fail with retrieval error (no candidates)
    assert not result.ok
    assert "no candidates" in str(result.error).lower()


def test_reranker_integration_with_dedup():
    """Test that reranking happens BEFORE deduplication in the pipeline."""
    # Arrange: Create chunks that would be reordered by reranker
    chunks = [
        RetrievedChunk(
            id=f"chunk{i}",
            text=f"text{i}",
            metadata={"source": "doc"},
            score=0.8 - i * 0.1,
            vector=[float(i), 0.0, 0.0],  # Vectors for diversity
        )
        for i in range(5)
    ]
    vector_store = FakeVectorStore(chunks=chunks)

    # Reranker reverses order: chunk4 > chunk3 > chunk2 > chunk1 > chunk0
    reranker = FakeReranker(scores=[0.1, 0.2, 0.3, 0.4, 0.5])

    uc = QueryKnowledgeBase(
        embedding=FakeEmbedding(),
        vector_store=vector_store,
        llm=None,
        reranker=reranker,
    )

    req = QueryRequest(
        question="test",
        top_k=3,
        use_reranker=True,
        confidence_threshold=0.0,
    )
    result = uc.execute(req)

    # Assert: Reranking should happen before dedup
    # Top chunk after reranking should be chunk4 (highest reranker score)
    assert result.ok
    # At least the first result should reflect reranking
    assert result.value.citations[0].chunk_id == "chunk4"
