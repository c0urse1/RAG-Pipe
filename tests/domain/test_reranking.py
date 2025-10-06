"""Tests for pure domain reranking functions.

These tests verify deterministic, pure functions with no infrastructure dependencies.
"""

from bu_superagent.domain.services.reranking import minmax_normalize, sort_by_scores_desc


def test_minmax_normalize_basic():
    """Test basic minmax normalization to [0,1] range."""
    result = minmax_normalize([1.0, 2.0, 3.0])
    assert result == [0.0, 0.5, 1.0]


def test_minmax_normalize_empty():
    """Test empty input returns empty list."""
    assert minmax_normalize([]) == []


def test_minmax_normalize_single():
    """Test single value normalizes to midpoint 0.5."""
    assert minmax_normalize([5.0]) == [0.5]


def test_minmax_normalize_equal_values():
    """Test all equal values normalize to midpoint 0.5."""
    assert minmax_normalize([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]


def test_minmax_normalize_negative():
    """Test normalization works with negative scores."""
    result = minmax_normalize([-1.0, 0.0, 1.0])
    assert result == [0.0, 0.5, 1.0]


def test_minmax_normalize_large_range():
    """Test normalization with large value range."""
    result = minmax_normalize([0.0, 100.0, 50.0])
    assert result == [0.0, 1.0, 0.5]


def test_sort_by_scores_desc_basic():
    """Test stable descending sort by scores."""
    items = ["a", "b", "c"]
    scores = [0.2, 0.9, 0.5]
    assert sort_by_scores_desc(items, scores) == ["b", "c", "a"]


def test_sort_by_scores_desc_empty():
    """Test empty inputs return empty list."""
    assert sort_by_scores_desc([], []) == []


def test_sort_by_scores_desc_single():
    """Test single item returns unchanged."""
    assert sort_by_scores_desc(["x"], [0.5]) == ["x"]


def test_sort_by_scores_desc_stable():
    """Test stable sort preserves order for equal scores."""
    items = ["a", "b", "c", "d"]
    scores = [0.5, 0.9, 0.5, 0.9]
    result = sort_by_scores_desc(items, scores)
    # Items with 0.9 should come first (b, d), then items with 0.5 (a, c)
    # Stable sort preserves original order within equal scores
    assert result == ["b", "d", "a", "c"]


def test_sort_by_scores_desc_all_equal():
    """Test sorting with all equal scores preserves order."""
    items = ["x", "y", "z"]
    scores = [0.5, 0.5, 0.5]
    assert sort_by_scores_desc(items, scores) == ["x", "y", "z"]


def test_sort_by_scores_desc_negative_scores():
    """Test sorting works with negative scores."""
    items = ["a", "b", "c"]
    scores = [-1.0, -0.5, -2.0]
    assert sort_by_scores_desc(items, scores) == ["b", "a", "c"]


def test_sort_by_scores_desc_with_chunks():
    """Test sorting with domain model objects (chunks)."""
    from bu_superagent.domain.models import RetrievedChunk

    chunks = [
        RetrievedChunk(id="1", text="low", metadata={}, score=0.2, vector=None),
        RetrievedChunk(id="2", text="high", metadata={}, score=0.9, vector=None),
        RetrievedChunk(id="3", text="mid", metadata={}, score=0.5, vector=None),
    ]
    scores = [0.2, 0.9, 0.5]
    result = sort_by_scores_desc(chunks, scores)
    assert [c.id for c in result] == ["2", "3", "1"]
    assert [c.text for c in result] == ["high", "mid", "low"]
