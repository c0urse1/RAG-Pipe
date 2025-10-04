class DomainError(Exception):
    """Base class for domain-specific errors."""


class ValidationError(DomainError):
    """Invalid input/domain state."""


class EmbeddingError(DomainError):
    """Embedding backend failed or is misconfigured."""


class VectorStoreError(DomainError):
    """Vector store backend failed or is misconfigured."""
