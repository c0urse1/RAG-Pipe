"""Qdrant adapter contract tests with Docker.

Why: Verhindert Regressions beim Wechsel der Vector-DB.
"""

import pytest

from bu_superagent.infrastructure.vectorstore.qdrant_adapter import (
    QdrantConfig,
    QdrantVectorStoreAdapter,
)


@pytest.fixture
def qdrant_adapter():
    """Fixture for Qdrant adapter with Docker container.

    Requires: docker-compose up qdrant
    Or: docker run -p 6333:6333 qdrant/qdrant
    """
    cfg = QdrantConfig(
        url="http://localhost:6333",
        api_key=None,
        prefer_grpc=False,
        timeout_s=10,
    )
    adapter = QdrantVectorStoreAdapter(cfg)
    yield adapter
    # Cleanup handled by test functions


@pytest.mark.slow
@pytest.mark.integration
def test_upsert_and_search_round_trip(qdrant_adapter):
    """Test full round-trip: create collection, upsert, search."""
    collection = "test_roundtrip"
    dim = 128

    # Ensure collection
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=dim,
        shards=1,
        replicas=1,
        metric="cosine",
    )
    assert result.ok, f"Failed to create collection: {result.error}"

    # Upsert vectors
    ids = ["doc1", "doc2", "doc3"]
    vectors = [
        tuple([1.0] + [0.0] * 127),  # Vector at origin + x-axis
        tuple([0.0, 1.0] + [0.0] * 126),  # Vector at origin + y-axis
        tuple([1.0, 0.0] + [0.0] * 126),  # Similar to doc1
    ]
    metadata = [
        {"text": "document 1", "source": "test"},
        {"text": "document 2", "source": "test"},
        {"text": "document 3", "source": "test"},
    ]

    result = qdrant_adapter.upsert(collection, ids, vectors, metadata)
    assert result.ok, f"Failed to upsert: {result.error}"

    # Search for similar to doc1
    query_vector = tuple([1.0] + [0.0] * 127)
    result = qdrant_adapter.search(collection, query_vector, top_k=2)
    assert result.ok, f"Failed to search: {result.error}"

    hits = result.value
    assert len(hits) == 2
    assert hits[0]["id"] in ["doc1", "doc3"]  # Most similar
    assert hits[0]["score"] > 0.9  # High cosine similarity


@pytest.mark.slow
@pytest.mark.integration
def test_collection_settings_shards_replicas(qdrant_adapter):
    """Test collection creation with sharding and replication."""
    collection = "test_shards_replicas"
    dim = 64

    # Create collection with multiple shards (simulates horizontal scaling)
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=dim,
        shards=2,  # Split data across 2 shards
        replicas=1,  # No replication (single node test)
        metric="cosine",
    )
    assert result.ok, f"Failed to create collection: {result.error}"

    # Verify collection exists (idempotent call)
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=dim,
        shards=2,
        replicas=1,
        metric="cosine",
    )
    assert result.ok, f"Failed on idempotent ensure: {result.error}"

    # Test dimension mismatch error
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=128,  # Wrong dimension
        shards=2,
        replicas=1,
        metric="cosine",
    )
    assert not result.ok, "Should fail on dimension mismatch"


@pytest.mark.slow
@pytest.mark.integration
def test_quantization_on_off(qdrant_adapter):
    """Test quantization configuration for compression."""
    collection = "test_quantization"
    dim = 256

    # Create collection
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=dim,
        shards=1,
        replicas=1,
        metric="cosine",
    )
    assert result.ok, f"Failed to create collection: {result.error}"

    # Enable scalar quantization (INT8)
    result = qdrant_adapter.set_quantization(
        name=collection,
        kind="scalar",
        params={"quantile": 0.99, "always_ram": True},
    )
    assert result.ok, f"Failed to set quantization: {result.error}"

    # Upsert some vectors to verify it works with quantization
    ids = ["q1", "q2"]
    vectors = [
        tuple([0.5] * dim),
        tuple([0.3] * dim),
    ]
    metadata = [{"text": "test1"}, {"text": "test2"}]

    result = qdrant_adapter.upsert(collection, ids, vectors, metadata)
    assert result.ok, f"Failed to upsert with quantization: {result.error}"

    # Search still works
    result = qdrant_adapter.search(collection, vectors[0], top_k=1)
    assert result.ok, f"Failed to search with quantization: {result.error}"
    assert len(result.value) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_search_with_filters(qdrant_adapter):
    """Test metadata filtering during search."""
    collection = "test_filters"
    dim = 64

    # Create collection
    result = qdrant_adapter.ensure_collection(
        name=collection,
        dim=dim,
        shards=1,
        replicas=1,
        metric="cosine",
    )
    assert result.ok

    # Upsert vectors with different metadata
    ids = ["f1", "f2", "f3"]
    vectors = [
        tuple([1.0] + [0.0] * 63),
        tuple([0.9, 0.1] + [0.0] * 62),
        tuple([0.8, 0.2] + [0.0] * 62),
    ]
    metadata = [
        {"category": "A", "year": "2023"},
        {"category": "B", "year": "2023"},
        {"category": "A", "year": "2024"},
    ]

    result = qdrant_adapter.upsert(collection, ids, vectors, metadata)
    assert result.ok

    # Search with filter: only category A
    query_vector = tuple([1.0] + [0.0] * 63)
    result = qdrant_adapter.search(
        collection,
        query_vector,
        top_k=10,
        filters={"category": "A"},
    )
    assert result.ok
    hits = result.value
    assert len(hits) == 2  # Only f1 and f3
    assert all(h["meta"]["category"] == "A" for h in hits)
