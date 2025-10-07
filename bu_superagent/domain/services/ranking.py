# bu_superagent/domain/services/ranking.py
# Pure domain services: no I/O, deterministic, no external libraries.
from __future__ import annotations

import math
from collections.abc import Sequence

from bu_superagent.domain.models import RetrievedChunk


def _cos_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute cosine similarity without external libs.
    Works with non-perfectly normalized vectors by normalizing on the fly.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def deduplicate_by_cosine(
    chunks: Sequence[RetrievedChunk],
    threshold: float = 0.95,
) -> list[RetrievedChunk]:
    """
    Remove near-duplicates by cosine similarity (>= threshold).
    Falls back to text-based exact match if vectors are missing.

    - O(n^2) but n is small after initial ANN retrieval; acceptable for domain.
    - Keeps the first occurrence; drops later ones considered near-duplicates.
    """
    seen: list[RetrievedChunk] = []
    for c in chunks:
        is_dup = False
        # Prefer vector-based duplicate detection if vectors present.
        if c.vector is not None:
            for s in seen:
                if s.vector is None:
                    continue
                if _cos_sim(c.vector, s.vector) >= threshold:
                    is_dup = True
                    break
        else:
            # Fallback: dedup by normalized text if no vector available
            norm_c = (c.text or "").strip().casefold()
            for s in seen:
                if (s.text or "").strip().casefold() == norm_c and norm_c != "":
                    is_dup = True
                    break

        if not is_dup:
            seen.append(c)
    return seen


def mmr(
    query_vec: Sequence[float],
    candidates: Sequence[RetrievedChunk],
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[RetrievedChunk]:
    """
    Maximal Marginal Relevance (MMR) selection with pure-Python cosine.
    Assumes candidate vectors are L2-ish; we normalize on-the-fly anyway.

    NOTE:
    - Keep here only if the product actually uses MMR.
    - If the application removes MMR, delete this function and its tests.
    """
    if top_k <= 0:
        return []

    # Precompute candidate similarities to the query
    sims: list[tuple[int, float]] = []
    for idx, c in enumerate(candidates):
        if c.vector is None:
            sims.append((idx, -1.0))
        else:
            sims.append((idx, _cos_sim(query_vec, c.vector)))

    selected: list[int] = []
    remaining: list[int] = [i for i in range(len(candidates))]

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -1.0

        for i in remaining:
            query_rel = sims[i][1]
            diversity = 0.0
            if selected:
                # max similarity to any already selected item (diversity penalty)
                max_sim_to_selected = max(
                    [
                        (
                            _cos_sim(
                                candidates[i].vector or (),
                                candidates[j].vector or (),
                            )
                            if (candidates[i].vector and candidates[j].vector)
                            else 0.0
                        )
                        for j in selected
                    ]
                )
                diversity = max_sim_to_selected

            score = lambda_mult * query_rel - (1.0 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)  # type: ignore[arg-type]
        remaining = [i for i in remaining if i != best_idx]

    return [candidates[i] for i in selected]
