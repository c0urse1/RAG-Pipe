"""Blob store port for object storage operations."""

from typing import Any, Protocol

from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result


class BlobStorePort(Protocol):
    """Port for blob storage operations."""

    def put(self, key: str, data: bytes, meta: dict[str, Any]) -> Result[str, DomainError]:
        """Put blob data with metadata. Returns storage key."""
        ...

    def get(self, key: str) -> Result[bytes, DomainError]:
        """Get blob data by key."""
        ...
