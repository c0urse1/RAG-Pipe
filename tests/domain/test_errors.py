"""Tests for domain errors, especially confidence gating."""

import pytest

from bu_superagent.domain.errors import (
    DomainError,
    LowConfidenceError,
    RetrievalError,
    ValidationError,
)


def test_low_confidence_error_is_domain_error():
    """LowConfidenceError should be a DomainError."""
    err = LowConfidenceError(message="Too low", top_score=0.3, threshold=0.5)
    assert isinstance(err, DomainError)


def test_low_confidence_error_dataclass_fields():
    """LowConfidenceError should carry confidence metrics."""
    err = LowConfidenceError(message="Confidence too low", top_score=0.35, threshold=0.6)
    assert err.message == "Confidence too low"
    assert err.top_score == 0.35
    assert err.threshold == 0.6


def test_low_confidence_error_is_frozen():
    """LowConfidenceError should be immutable (frozen dataclass)."""
    err = LowConfidenceError(message="Low", top_score=0.2, threshold=0.5)
    with pytest.raises(AttributeError):
        err.top_score = 0.9  # type: ignore[misc]


def test_retrieval_error_is_domain_error():
    """RetrievalError should be a DomainError."""
    err = RetrievalError("Failed to retrieve")
    assert isinstance(err, DomainError)
    assert str(err) == "Failed to retrieve"


def test_validation_error_still_works():
    """Ensure existing ValidationError is not broken."""
    err = ValidationError("Invalid input")
    assert isinstance(err, DomainError)
    assert str(err) == "Invalid input"
