"""Domain errors (typed) for scale.

Why: Unified error family for Application layer, without Infra leaks.
"""

from dataclasses import dataclass


class DomainError(Exception):
    """Base class for domain-specific errors."""


class ValidationError(DomainError):
    """Invalid input/domain state."""


class RetrievalError(DomainError):
    """Generic retrieval failure (after infra errors were mapped)."""


@dataclass(frozen=True)
class LowConfidenceError(DomainError):
    """Retrieval confidence below acceptable threshold."""

    message: str
    top_score: float
    threshold: float


class RateLimitExceeded(DomainError):
    """Rate limit exceeded for API or service."""


class QuotaExceeded(DomainError):
    """Quota exceeded for tenant or resource."""


# Infrastructure-mapped errors (kept for backward compatibility)
class EmbeddingError(DomainError):
    """Embedding backend failed or is misconfigured."""


class VectorStoreError(DomainError):
    """Vector store backend failed or is misconfigured."""


class LLMError(DomainError):
    """LLM backend failed or is misconfigured."""


class DocumentError(DomainError):
    """Document loading/parsing failed."""


@dataclass(frozen=True)
class RerankerError(DomainError):
    """Reranking operation failed or is misconfigured."""

    detail: str = ""
