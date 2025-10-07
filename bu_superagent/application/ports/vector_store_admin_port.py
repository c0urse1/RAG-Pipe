"""Vector store admin port for scale-critical operations."""

from typing import Any, Protocol

from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result


class VectorStoreAdminPort(Protocol):
    """Port for vector store admin operations (sharding, replication, quantization)."""

    def ensure_collection(
        self,
        name: str,
        dim: int,
        shards: int,
        replicas: int,
        metric: str = "cosine",
    ) -> Result[None, DomainError]:
        """Create or ensure collection exists with specified configuration."""
        ...

    def set_quantization(
        self, name: str, kind: str, params: dict[str, Any]
    ) -> Result[None, DomainError]:
        """Set quantization configuration for collection."""
        ...

    def set_search_params(self, name: str, params: dict[str, Any]) -> Result[None, DomainError]:
        """Set search parameters for collection."""
        ...
