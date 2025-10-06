"""Pure domain functions for reranking operations.

Why (SAM): These are deterministic, pure functions with no I/O or external
dependencies. They belong in the domain layer for unit testing without
infrastructure concerns.

Functions:
- minmax_normalize: Scale scores linearly to [0,1] range
- sort_by_scores_desc: Stable descending sort by scores
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def minmax_normalize(scores: Sequence[float]) -> list[float]:
    """Pure function: scales scores linearly to [0,1].

    Args:
        scores: Raw scores to normalize

    Returns:
        Normalized scores in [0,1] range. Empty input returns empty list.
        All equal scores return 0.5 for each (midpoint).

    Examples:
        >>> minmax_normalize([1.0, 2.0, 3.0])
        [0.0, 0.5, 1.0]
        >>> minmax_normalize([5.0, 5.0])
        [0.5, 0.5]
        >>> minmax_normalize([])
        []

    Note:
        Deterministic and testable without any infrastructure dependencies.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def sort_by_scores_desc(items: Sequence[T], scores: Sequence[float]) -> list[T]:
    """Stable sort (descending) by scores, pure & deterministic.

    Args:
        items: Items to sort (any type)
        scores: Corresponding scores (must have same length as items)

    Returns:
        Items sorted by scores in descending order (highest first).
        Maintains stable sort property (equal scores preserve original order).

    Examples:
        >>> sort_by_scores_desc(["a", "b", "c"], [0.2, 0.9, 0.5])
        ["b", "c", "a"]

    Note:
        This is a pure domain function with no side effects.
        Useful for reranking retrieved chunks by cross-encoder scores.
    """
    pairs: list[tuple[float, T]] = list(zip(scores, items, strict=False))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [it for _, it in pairs]
