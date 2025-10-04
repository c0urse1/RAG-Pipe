"""Domain services for ranking, deduplication, and confidence scoring.

Why (SAM): Deterministic algorithms belong to domain/services to be unit tested
without infrastructure dependencies.

Functions:
- deduplicate_by_cosine: Stable deduplication using cosine similarity threshold
- mmr: Maximal Marginal Relevance for diversity-aware ranking
- passes_confidence: Check if top score meets confidence threshold
- top_score: Extract highest score from chunks

Reference: Confidence-Gate architecture - under threshold triggers escalation
with complete context package.
"""

from __future__ import annotations

from bu_superagent.domain.models import RetrievedChunk


def _cosine(u: list[float], v: list[float]) -> float:
    """Compute cosine similarity between two L2-normalized vectors.

    Args:
        u: First vector (L2-normalized, same length as v expected)
        v: Second vector (L2-normalized, same length as u expected)

    Returns:
        Cosine similarity (dot product for normalized vectors)

    Note:
        Assumes both vectors are L2-normalized and have the same length.
        If lengths differ, computation truncates to shorter length (zip behavior).
    """
    return sum(a * b for a, b in zip(u, v, strict=False))


def deduplicate_by_cosine(
    chunks: list[RetrievedChunk], threshold: float = 0.95
) -> list[RetrievedChunk]:
    """Stable deduplication: keep first occurrence, drop near-duplicates.

    Args:
        chunks: List of retrieved chunks with optional vectors
        threshold: Cosine similarity threshold (default 0.95)

    Returns:
        Deduplicated list maintaining original order

    Note:
        Chunks without vectors are always kept.
    """
    kept: list[RetrievedChunk] = []
    for c in chunks:
        if not c.vector:
            kept.append(c)
            continue
        if all((kc.vector is None) or _cosine(c.vector, kc.vector) < threshold for kc in kept):
            kept.append(c)
    return kept


def mmr(
    candidates: list[RetrievedChunk],
    query_vec: list[float],
    k: int,
    lambda_mult: float = 0.5,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance for diversity-aware ranking.

    Args:
        candidates: List of candidate chunks with vectors
        query_vec: Query vector (L2-normalized)
        k: Number of chunks to select
        lambda_mult: Trade-off parameter (0=diversity, 1=relevance)

    Returns:
        Top-k chunks balancing relevance and diversity

    Note:
        Assumes all vectors are L2-normalized.
        Falls back to chunk.score if vector is missing.
    """
    selected: list[RetrievedChunk] = []
    remaining = candidates[:]
    while remaining and len(selected) < k:
        best = None
        best_score = float("-inf")
        for c in remaining:
            relevance = _cosine(query_vec, c.vector) if c.vector else c.score
            diversity = 0.0
            if selected:
                diversity = max(
                    _cosine(c.vector, s.vector) if c.vector and s.vector else 0.0 for s in selected
                )
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            selected.append(best)
            remaining.remove(best)
    return selected


def top_score(chunks: list[RetrievedChunk]) -> float:
    """Extract the highest score from a list of chunks.

    Args:
        chunks: List of retrieved chunks

    Returns:
        Score of first chunk, or 0.0 if list is empty
    """
    return chunks[0].score if chunks else 0.0


def passes_confidence(chunks: list[RetrievedChunk], threshold: float) -> tuple[bool, float]:
    """Check if top score meets confidence threshold.

    Args:
        chunks: List of retrieved chunks (assumed sorted by score)
        threshold: Minimum acceptable confidence score

    Returns:
        Tuple of (passes, score) where:
        - passes: True if top_score >= threshold
        - score: The top score value

    Note:
        Part of Confidence-Gate architecture: under threshold triggers
        escalation with complete context package.
    """
    s = top_score(chunks)
    return (s >= threshold, s)
