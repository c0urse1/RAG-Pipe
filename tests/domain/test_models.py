"""Tests for domain models (RankedChunk, Citation)."""

from bu_superagent.domain.models import Citation, RankedChunk, RetrievedChunk


def test_retrieved_chunk_creation():
    """RetrievedChunk should store all required fields."""
    chunk = RetrievedChunk(
        id="chunk_1",
        text="Sample text",
        metadata={"doc_id": "doc1"},
        vector=[0.1, 0.2, 0.3],
        score=0.95,
    )

    assert chunk.id == "chunk_1"
    assert chunk.text == "Sample text"
    assert chunk.metadata == {"doc_id": "doc1"}
    assert chunk.vector == [0.1, 0.2, 0.3]
    assert chunk.score == 0.95


def test_retrieved_chunk_optional_vector():
    """RetrievedChunk should allow None for vector field."""
    chunk = RetrievedChunk(
        id="chunk_1",
        text="Sample text",
        metadata={},
        vector=None,
        score=0.95,
    )

    assert chunk.vector is None


def test_ranked_chunk_creation():
    """RankedChunk should wrap a RetrievedChunk with a rank."""
    chunk = RetrievedChunk(
        id="chunk_1",
        text="Sample text",
        metadata={"doc_id": "doc1"},
        vector=None,
        score=0.95,
    )
    ranked = RankedChunk(chunk=chunk, rank=1)

    assert ranked.chunk.id == "chunk_1"
    assert ranked.chunk.text == "Sample text"
    assert ranked.rank == 1


def test_ranked_chunk_is_frozen():
    """RankedChunk should be immutable."""
    chunk = RetrievedChunk(
        id="chunk_1",
        text="Sample",
        metadata={},
        vector=None,
        score=0.8,
    )
    ranked = RankedChunk(chunk=chunk, rank=2)

    import pytest

    with pytest.raises(AttributeError):
        ranked.rank = 5  # type: ignore[misc]


def test_citation_creation():
    """Citation should store chunk_id, source, and score."""
    citation = Citation(
        chunk_id="chunk_42",
        source="document.pdf",
        score=0.88,
    )

    assert citation.chunk_id == "chunk_42"
    assert citation.source == "document.pdf"
    assert citation.score == 0.88


def test_citation_is_frozen():
    """Citation should be immutable."""
    citation = Citation(
        chunk_id="c1",
        source="doc.txt",
        score=0.75,
    )

    import pytest

    with pytest.raises(AttributeError):
        citation.score = 0.99  # type: ignore[misc]
