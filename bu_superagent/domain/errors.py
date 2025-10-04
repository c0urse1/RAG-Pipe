from dataclasses import dataclass


class DomainError(Exception):
    """Base class for domain-specific errors."""


class ValidationError(DomainError):
    """Invalid input/domain state."""


class EmbeddingError(DomainError):
    """Embedding backend failed or is misconfigured."""


class VectorStoreError(DomainError):
    """Vector store backend failed or is misconfigured."""


@dataclass(frozen=True)
class LowConfidenceError(DomainError):
    """Retrieval confidence below acceptable threshold."""

    message: str
    top_score: float
    threshold: float


class RetrievalError(DomainError):
    """Generic retrieval failure (after infra errors were mapped)."""
