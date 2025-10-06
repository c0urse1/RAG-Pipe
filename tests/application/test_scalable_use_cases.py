"""Application layer tests with fake adapters (no infrastructure).

Why: Test use case orchestration logic without real embeddings/vector stores.
"""

from bu_superagent.application.dtos import IngestRequest, QueryRequest
from bu_superagent.application.use_cases.ingest_documents_parallel import IngestDocumentsParallel
from bu_superagent.application.use_cases.query_knowledge_base_scalable import (
    QueryKnowledgeBaseScalable,
)
from bu_superagent.domain.errors import LowConfidenceError, RetrievalError, ValidationError
from bu_superagent.domain.types import Result


class FakeEmbed:
    """Fake embedding port for testing."""

    def __init__(self, vectors=None):
        self.vectors = vectors or [[(1.0, 0.0, 0.0)]]

    def embed_texts(self, texts):
        """Return pre-configured vectors."""
        return Result.success(self.vectors[0] if texts else [])


class FakeVS:
    """Fake vector store port for testing."""

    def __init__(self, search_results=None, fail_search=False, fail_upsert=False):
        self.search_results = search_results or []
        self.fail_search = fail_search
        self.fail_upsert = fail_upsert
        self.upserted = []

    def search(self, collection, vector, top_k, filters=None):
        """Return pre-configured search results."""
        if self.fail_search:
            return Result.failure(RetrievalError("search failed"))
        return Result.success(self.search_results[:top_k])

    def upsert(self, collection, ids, vectors, metadata):
        """Record upserted data."""
        if self.fail_upsert:
            return Result.failure(RetrievalError("upsert failed"))
        self.upserted.append({"ids": ids, "vectors": vectors, "metadata": metadata})
        return Result.success(None)


class FakeWorkQueue:
    """Fake work queue port for testing."""

    def __init__(self):
        self.enqueued = []
        self.dequeued = []
        self.acked = []

    def enqueue(self, topic, payload):
        """Record enqueued tasks."""
        task_id = f"task-{len(self.enqueued)}"
        self.enqueued.append({"topic": topic, "payload": payload, "id": task_id})
        return Result.success(task_id)

    def dequeue_batch(self, topic, max_n):
        """Return dequeued tasks."""
        batch = self.dequeued[:max_n]
        return Result.success(batch)

    def ack(self, topic, ack_ids):
        """Record acknowledged tasks."""
        self.acked.extend(ack_ids)
        return Result.success(None)


def test_query_hybrid_confidence_gate():
    """Test query with hybrid fusion and confidence gate."""
    # Setup: high-confidence results
    embed = FakeEmbed(vectors=[[(1.0, 0.0, 0.0)]])
    vs = FakeVS(
        search_results=[
            {"id": "doc1", "score": 0.95, "meta": {"text": "answer 1"}},
            {"id": "doc2", "score": 0.85, "meta": {"text": "answer 2"}},
        ]
    )
    lexical = FakeVS(
        search_results=[
            {"id": "doc2", "score": 0.90, "meta": {"text": "answer 2"}},
            {"id": "doc3", "score": 0.80, "meta": {"text": "answer 3"}},
        ]
    )

    uc = QueryKnowledgeBaseScalable(embed=embed, vs=vs, lexical=lexical)
    req = QueryRequest(
        collection="test",
        question="What is RAG?",
        top_k=2,
        use_hybrid=True,
        use_mmr=False,
        confidence_threshold=0.7,
    )

    result = uc.execute(req)

    assert result.ok
    assert "answers" in result.value
    assert len(result.value["answers"]) > 0


def test_query_low_confidence_fails():
    """Test query fails when confidence threshold not met."""
    embed = FakeEmbed(vectors=[[(1.0, 0.0, 0.0)]])
    vs = FakeVS(
        search_results=[
            {"id": "doc1", "score": 0.3, "meta": {"text": "low confidence"}},
        ]
    )

    uc = QueryKnowledgeBaseScalable(embed=embed, vs=vs)
    req = QueryRequest(
        collection="test",
        question="What is RAG?",
        top_k=5,
        confidence_threshold=0.5,  # Higher than top score
    )

    result = uc.execute(req)

    assert not result.ok
    assert isinstance(result.error, LowConfidenceError)


def test_query_no_results_fails():
    """Test query fails when no candidates found."""
    embed = FakeEmbed(vectors=[[(1.0, 0.0, 0.0)]])
    vs = FakeVS(search_results=[])  # Empty results

    uc = QueryKnowledgeBaseScalable(embed=embed, vs=vs)
    req = QueryRequest(
        collection="test",
        question="What is RAG?",
        top_k=5,
    )

    result = uc.execute(req)

    assert not result.ok
    assert isinstance(result.error, RetrievalError)


def test_query_invalid_input_fails():
    """Test query fails with invalid input."""
    embed = FakeEmbed()
    vs = FakeVS()

    uc = QueryKnowledgeBaseScalable(embed=embed, vs=vs)
    req = QueryRequest(
        collection="test",
        question="",  # Empty question
        top_k=5,
    )

    result = uc.execute(req)

    assert not result.ok
    assert isinstance(result.error, ValidationError)


def test_ingest_batches_512_and_propagates_errors():
    """Test ingest splits into 512-doc batches and propagates errors."""
    embed = FakeEmbed(vectors=[[(1.0, 0.0, 0.0)] * 512])
    vs = FakeVS()
    wq = FakeWorkQueue()

    uc = IngestDocumentsParallel(embed=embed, vs=vs, wq=wq)

    # Create 1000 docs -> should split into 2 batches (512 + 488)
    docs = [{"id": f"doc{i}", "text": f"text {i}", "meta": {}} for i in range(1000)]
    req = IngestRequest(collection="test", shard_key="shard1", docs=docs)

    # Test planning
    plan_result = uc.plan(req)
    assert plan_result.ok
    batches = plan_result.value
    assert len(batches) == 2
    assert len(batches[0]["docs"]) == 512
    assert len(batches[1]["docs"]) == 488

    # Test execution
    exec_result = uc.execute_batch("test", batches[0])
    assert exec_result.ok
    assert len(vs.upserted) == 1
    assert len(vs.upserted[0]["ids"]) == 512


def test_ingest_empty_docs_fails():
    """Test ingest fails with empty document list."""
    embed = FakeEmbed()
    vs = FakeVS()
    wq = FakeWorkQueue()

    uc = IngestDocumentsParallel(embed=embed, vs=vs, wq=wq)
    req = IngestRequest(collection="test", shard_key="shard1", docs=[])

    result = uc.plan(req)

    assert not result.ok
    assert isinstance(result.error, ValidationError)


def test_ingest_embed_error_propagates():
    """Test ingest propagates embedding errors."""
    embed = FakeEmbed()
    embed.embed_texts = lambda texts: Result.failure(RetrievalError("embed failed"))
    vs = FakeVS()
    wq = FakeWorkQueue()

    uc = IngestDocumentsParallel(embed=embed, vs=vs, wq=wq)
    batch = {"docs": [{"id": "doc1", "text": "text", "meta": {}}]}

    result = uc.execute_batch("test", batch)

    assert not result.ok
    assert isinstance(result.error, RetrievalError)


def test_ingest_upsert_error_propagates():
    """Test ingest propagates vector store errors."""
    embed = FakeEmbed(vectors=[[(1.0, 0.0, 0.0)]])
    vs = FakeVS(fail_upsert=True)
    wq = FakeWorkQueue()

    uc = IngestDocumentsParallel(embed=embed, vs=vs, wq=wq)
    batch = {"docs": [{"id": "doc1", "text": "text", "meta": {}}]}

    result = uc.execute_batch("test", batch)

    assert not result.ok
    assert isinstance(result.error, RetrievalError)
