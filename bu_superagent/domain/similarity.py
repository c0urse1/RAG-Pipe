"""Pure similarity and deduplication functions for scaling.

Why: Dedup/Normalization are pure functions → perfect for Domain.
Preparation for Hybrid-Fusion and Confidence-Gate.
"""

from math import sqrt

from .types import Score, Vector


def cosine(u: Vector, v: Vector) -> Score:
    """Compute cosine similarity between two vectors.

    Args:
        u: First vector
        v: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    dot = sum(a * b for a, b in zip(u, v, strict=False))
    nu = sqrt(sum(a * a for a in u)) or 1.0
    nv = sqrt(sum(b * b for b in v)) or 1.0
    return dot / (nu * nv)


def deduplicate_by_cosine(
    items: list[tuple[str, Vector]], threshold: float = 0.95
) -> list[tuple[str, Vector]]:
    """Deduplicate items by cosine similarity threshold.

    Args:
        items: List of (id, vector) tuples
        threshold: Cosine similarity threshold (default 0.95)

    Returns:
        Deduplicated list maintaining first occurrence of similar items
    """
    kept: list[tuple[str, Vector]] = []
    for id_i, vec_i in items:
        if all(cosine(vec_i, vj) < threshold for _, vj in kept):
            kept.append((id_i, vec_i))
    return kept


def zscore_normalize(scores: list[Score]) -> list[Score]:
    """Normalize scores using z-score normalization.

    Args:
        scores: List of scores to normalize

    Returns:
        Z-score normalized scores
    """
    if not scores:
        return scores
    mu = sum(scores) / len(scores)
    var = sum((s - mu) ** 2 for s in scores) / max(len(scores) - 1, 1)
    std = (var**0.5) or 1.0
    return [(s - mu) / std for s in scores]
