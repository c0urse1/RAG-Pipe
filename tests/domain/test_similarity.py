"""Domain tests for similarity functions.

Why: Early safety nets for Precision/Recall-critical logic.
"""

from bu_superagent.domain.similarity import cosine, deduplicate_by_cosine, zscore_normalize


def test_cosine_similarity():
    """Test basic cosine similarity calculation."""
    u = (1.0, 0.0, 0.0)
    v = (1.0, 0.0, 0.0)
    assert abs(cosine(u, v) - 1.0) < 1e-6

    u = (1.0, 0.0, 0.0)
    v = (0.0, 1.0, 0.0)
    assert abs(cosine(u, v) - 0.0) < 1e-6


def test_dedup_keeps_far_vectors():
    """Test deduplication keeps vectors that are sufficiently different."""
    items = [
        ("a", (1.0, 0.0, 0.0)),
        ("b", (0.0, 1.0, 0.0)),  # orthogonal to a, should be kept
        ("c", (1.0, 0.01, 0.0)),  # very similar to a, should be removed
    ]
    result = deduplicate_by_cosine(items, threshold=0.95)
    assert len(result) == 2
    assert result[0][0] == "a"
    assert result[1][0] == "b"


def test_dedup_empty_list():
    """Test deduplication with empty input."""
    result = deduplicate_by_cosine([], threshold=0.95)
    assert result == []


def test_dedup_single_item():
    """Test deduplication with single item."""
    items = [("a", (1.0, 0.0, 0.0))]
    result = deduplicate_by_cosine(items, threshold=0.95)
    assert len(result) == 1
    assert result[0][0] == "a"


def test_zscore_normalize_stability():
    """Test z-score normalization stability."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = zscore_normalize(scores)

    # Check mean is approximately 0
    mean = sum(normalized) / len(normalized)
    assert abs(mean) < 1e-6

    # Check std dev is approximately 1
    variance = sum((s - mean) ** 2 for s in normalized) / (len(normalized) - 1)
    std = variance**0.5
    assert abs(std - 1.0) < 1e-6


def test_zscore_normalize_empty():
    """Test z-score normalization with empty input."""
    result = zscore_normalize([])
    assert result == []


def test_zscore_normalize_single():
    """Test z-score normalization with single value."""
    result = zscore_normalize([5.0])
    assert len(result) == 1
    # Single value normalizes to 0
    assert abs(result[0]) < 1e-6


def test_zscore_normalize_constant():
    """Test z-score normalization with constant values."""
    scores = [5.0, 5.0, 5.0, 5.0]
    result = zscore_normalize(scores)
    # All constant values should normalize to 0
    assert all(abs(s) < 1e-6 for s in result)
