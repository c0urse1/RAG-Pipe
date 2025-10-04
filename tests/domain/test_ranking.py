"""Tests for domain ranking services (MMR, dedup, confidence)."""

import pytest

from bu_superagent.domain.models import RetrievedChunk
from bu_superagent.domain.services.ranking import (
    deduplicate_by_cosine,
    mmr,
    passes_confidence,
    top_score,
)


# Helper to create test chunks
def make_chunk(id_: str, score: float, vector: list[float] | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=id_,
        text=f"Text for {id_}",
        metadata={"doc_id": id_},
        vector=vector,
        score=score,
    )


class TestTopScore:
    @pytest.mark.parametrize(
        "chunks,expected",
        [
            ([make_chunk("c1", 0.95), make_chunk("c2", 0.85)], 0.95),
            ([], 0.0),
            ([make_chunk("c1", 0.42)], 0.42),
            ([make_chunk("c1", 0.95), make_chunk("c2", 0.85), make_chunk("c3", 0.75)], 0.95),
        ],
    )
    def test_top_score_parametrized(self, chunks: list[RetrievedChunk], expected: float) -> None:
        """top_score should extract highest score from chunks or 0.0 if empty."""
        assert top_score(chunks) == expected


class TestPassesConfidence:
    def test_passes_confidence_above_threshold(self) -> None:
        """passes_confidence should return True when score >= threshold."""
        chunks = [make_chunk("c1", 0.8), make_chunk("c2", 0.6)]
        passes, score = passes_confidence(chunks, threshold=0.7)
        assert passes is True
        assert score == 0.8

    def test_passes_confidence_below_threshold(self) -> None:
        """passes_confidence should return False when score < threshold."""
        chunks = [make_chunk("c1", 0.5), make_chunk("c2", 0.4)]
        passes, score = passes_confidence(chunks, threshold=0.6)
        assert passes is False
        assert score == 0.5

    def test_passes_confidence_exactly_at_threshold(self) -> None:
        """passes_confidence should return True when score == threshold."""
        chunks = [make_chunk("c1", 0.75)]
        passes, score = passes_confidence(chunks, threshold=0.75)
        assert passes is True
        assert score == 0.75

    def test_passes_confidence_empty_list(self) -> None:
        """passes_confidence should return False for empty list."""
        passes, score = passes_confidence([], threshold=0.5)
        assert passes is False
        assert score == 0.0


class TestDeduplicateByCosine:
    def test_deduplicate_keeps_first_drops_duplicates(self) -> None:
        """Stable dedup should keep first occurrence, drop near-duplicates."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.99, 0.01, 0.0]),  # very similar
            make_chunk("c3", 0.7, vector=[0.0, 1.0, 0.0]),  # different
        ]
        result = deduplicate_by_cosine(chunks, threshold=0.95)
        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c3"

    def test_deduplicate_keeps_all_if_diverse(self) -> None:
        """Should keep all chunks if they're diverse."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.0, 1.0, 0.0]),
            make_chunk("c3", 0.7, vector=[0.0, 0.0, 1.0]),
        ]
        result = deduplicate_by_cosine(chunks, threshold=0.95)
        assert len(result) == 3

    def test_deduplicate_keeps_chunks_without_vectors(self) -> None:
        """Chunks without vectors should always be kept."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=None),
            make_chunk("c3", 0.7, vector=None),
        ]
        result = deduplicate_by_cosine(chunks, threshold=0.95)
        assert len(result) == 3

    def test_deduplicate_empty_list(self) -> None:
        """Should handle empty list."""
        result = deduplicate_by_cosine([], threshold=0.95)
        assert result == []

    def test_deduplicate_custom_threshold(self) -> None:
        """Should respect custom threshold."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.9, 0.1, 0.0]),  # cosine ~0.9
        ]
        # High threshold (0.95) should keep both
        result_high = deduplicate_by_cosine(chunks, threshold=0.95)
        assert len(result_high) == 2

        # Low threshold (0.85) should drop c2
        result_low = deduplicate_by_cosine(chunks, threshold=0.85)
        assert len(result_low) == 1
        assert result_low[0].id == "c1"

    def test_deduplicate_threshold_equal_is_duplicate(self) -> None:
        """Boundary case: cosine == threshold exactly should drop duplicate."""
        # cos([1.0, 0.0, 0.0], [0.95, 0.0, 0.0]) = 0.95 exactly
        # Should be treated as duplicate and c2 should be dropped
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.95, 0.0, 0.0]),
        ]
        result = deduplicate_by_cosine(chunks, threshold=0.95)
        assert [c.id for c in result] == ["c1"]


class TestMMR:
    def test_mmr_selects_top_k(self) -> None:
        """MMR should select exactly k chunks."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.0, 1.0, 0.0]),
            make_chunk("c3", 0.7, vector=[0.0, 0.0, 1.0]),
            make_chunk("c4", 0.6, vector=[0.5, 0.5, 0.0]),
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=2, lambda_mult=0.5)
        assert len(result) == 2

    def test_mmr_pure_relevance(self) -> None:
        """MMR with lambda=1.0 should prioritize relevance only."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.9, 0.1, 0.0]),
            make_chunk("c3", 0.7, vector=[0.0, 1.0, 0.0]),
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=2, lambda_mult=1.0)
        # Should pick most relevant first (c1, c2)
        assert result[0].id == "c1"
        assert result[1].id == "c2"

    def test_mmr_pure_diversity(self) -> None:
        """MMR with lambda=0.0 should prioritize diversity."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.99, 0.01, 0.0]),  # very similar to c1
            make_chunk("c3", 0.7, vector=[0.0, 1.0, 0.0]),  # orthogonal to c1
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=2, lambda_mult=0.0)
        # Should pick c1 first (highest relevance), then c3 (most diverse)
        assert result[0].id == "c1"
        assert result[1].id == "c3"

    def test_mmr_handles_missing_vectors(self) -> None:
        """MMR should fall back to score when vector is missing."""
        chunks = [
            make_chunk("c1", 0.9, vector=None),
            make_chunk("c2", 0.8, vector=[0.0, 1.0, 0.0]),
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=2, lambda_mult=0.5)
        assert len(result) == 2

    def test_mmr_k_larger_than_candidates(self) -> None:
        """MMR should return all candidates if k > len(candidates)."""
        chunks = [
            make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[0.0, 1.0, 0.0]),
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=10, lambda_mult=0.5)
        assert len(result) == 2

    def test_mmr_empty_candidates(self) -> None:
        """MMR should handle empty candidate list."""
        result = mmr([], query_vec=[1.0, 0.0, 0.0], k=5, lambda_mult=0.5)
        assert result == []

    def test_mmr_k_zero(self) -> None:
        """MMR with k=0 should return empty list."""
        chunks = [make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0])]
        result = mmr(chunks, query_vec=[1.0, 0.0, 0.0], k=0, lambda_mult=0.5)
        assert result == []

    def test_mmr_negative_k_returns_empty(self) -> None:
        """MMR with negative k should robustly return empty list."""
        chunks = [make_chunk("c1", 0.9, vector=[1.0, 0.0, 0.0])]
        result = mmr(chunks, query_vec=[1.0, 0.0, 0.0], k=-1, lambda_mult=0.5)
        assert result == []

    def test_mmr_tie_break_is_stable(self) -> None:
        """MMR with identical scores/vectors should keep first occurrence (stable)."""
        chunks = [
            make_chunk("c1", 0.8, vector=[1.0, 0.0, 0.0]),
            make_chunk("c2", 0.8, vector=[1.0, 0.0, 0.0]),  # identical to c1
        ]
        query_vec = [1.0, 0.0, 0.0]
        result = mmr(chunks, query_vec, k=1, lambda_mult=1.0)
        assert [c.id for c in result] == ["c1"]  # keep first → stable

    def test_mmr_all_missing_vectors_falls_back_to_score(self) -> None:
        """MMR should fall back to score when all vectors are missing."""
        chunks = [
            make_chunk("a", 0.9, vector=None),
            make_chunk("b", 0.8, vector=None),
            make_chunk("c", 0.7, vector=None),
        ]
        result = mmr(chunks, query_vec=[1.0, 0.0, 0.0], k=2, lambda_mult=0.5)
        assert [c.id for c in result] == ["a", "b"]


class TestCosineHelper:
    def test_cosine_orthogonal_vectors(self) -> None:
        """Cosine of orthogonal vectors should be 0."""
        from bu_superagent.domain.services.ranking import _cosine

        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        assert abs(_cosine(u, v)) < 1e-6

    def test_cosine_identical_vectors(self) -> None:
        """Cosine of identical vectors should be 1."""
        from bu_superagent.domain.services.ranking import _cosine

        u = [1.0, 0.0, 0.0]
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine(u, v) - 1.0) < 1e-6

    def test_cosine_opposite_vectors(self) -> None:
        """Cosine of opposite vectors should be -1."""
        from bu_superagent.domain.services.ranking import _cosine

        u = [1.0, 0.0, 0.0]
        v = [-1.0, 0.0, 0.0]
        assert abs(_cosine(u, v) - (-1.0)) < 1e-6

    def test_cosine_length_mismatch_truncates(self) -> None:
        """Cosine with mismatched lengths should truncate to shorter (zip behavior)."""
        from bu_superagent.domain.services.ranking import _cosine

        # zip truncates to shorter length without strict=True
        # [1.0] * [1.0] = 1.0, second element ignored
        result = _cosine([1.0, 0.0], [1.0])
        assert abs(result - 1.0) < 1e-6
