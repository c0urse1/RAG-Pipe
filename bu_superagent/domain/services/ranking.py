"""Pure ranking policies for scaling.

Why: MMR/RRF/Confidence are purely deterministic → Domain stays independent of Infra.
"""

from bu_superagent.domain.types import Score


def mmr(
    candidates: list[tuple[str, Score]], lambda_diversity: float = 0.7, k: int = 10
) -> list[tuple[str, Score]]:
    """Maximal Marginal Relevance for diversity-aware ranking.

    Assumes candidates sorted by score desc; uses id-only diversity proxy for
    simplicity in domain.

    Args:
        candidates: List of (id, score) tuples sorted by score descending
        lambda_diversity: Trade-off parameter (0=diversity, 1=relevance)
        k: Number of items to select

    Returns:
        Top-k items balancing relevance and diversity
    """
    selected: list[tuple[str, Score]] = []
    remaining = candidates[:]
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        # naive diversity proxy: penalize same doc prefix
        def mmr_score(item):
            sim_to_sel = max(
                1.0 if item[0].split(":")[0] == s[0].split(":")[0] else 0.0 for s in selected
            )
            return lambda_diversity * item[1] - (1 - lambda_diversity) * sim_to_sel

        best = max(remaining, key=mmr_score)
        remaining.remove(best)
        selected.append(best)
    return selected


def rrf(ranks: list[list[tuple[str, int]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion for hybrid (lexical + vector) retrieval.

    Args:
        ranks: List of rank lists, each containing (doc_id, rank) tuples
        k: Constant for RRF formula (default 60)

    Returns:
        Fused ranking as list of (doc_id, score) tuples sorted by score descending
    """
    scores = {}
    for ranklist in ranks:
        for doc_id, rank in ranklist:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def passes_confidence(top_score: Score, threshold: Score) -> bool:
    """Check if top score meets confidence threshold.

    Args:
        top_score: The highest score from retrieval
        threshold: Minimum acceptable confidence score

    Returns:
        True if top_score >= threshold, False otherwise
    """
    return top_score >= threshold
