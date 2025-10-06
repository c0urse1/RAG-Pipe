"""Scale-critical application ports.

Why: Skalierung braucht Admin-Fähigkeiten (Shards/Replicas/Quantization),
     Queueing und Monitoring – alles als Ports, um Infra zu kapseln.
"""

from typing import Protocol

from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result, Vector


class EmbeddingPort(Protocol):
    """Port for text embedding operations."""

    def embed_texts(self, texts: list[str]) -> Result[list[Vector], DomainError]:
        """Embed multiple texts into vectors."""
        ...


class VectorStorePort(Protocol):
    """Port for basic vector store operations."""

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[Vector],
        metadata: list[dict],
    ) -> Result[None, DomainError]:
        """Upsert vectors with metadata into collection."""
        ...

    def search(
        self,
        collection: str,
        vector: Vector,
        top_k: int,
        filters: dict | None = None,
    ) -> Result[list[dict], DomainError]:
        """Search for similar vectors in collection."""
        ...


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

    def set_quantization(self, name: str, kind: str, params: dict) -> Result[None, DomainError]:
        """Set quantization configuration for collection."""
        ...

    def set_search_params(self, name: str, params: dict) -> Result[None, DomainError]:
        """Set search parameters for collection."""
        ...


class WorkQueuePort(Protocol):
    """Port for async work queue operations."""

    def enqueue(self, topic: str, payload: dict) -> Result[str, DomainError]:
        """Enqueue a task to topic. Returns task ID."""
        ...

    def dequeue_batch(self, topic: str, max_n: int) -> Result[list[dict], DomainError]:
        """Dequeue up to max_n tasks from topic."""
        ...

    def ack(self, topic: str, ack_ids: list[str]) -> Result[None, DomainError]:
        """Acknowledge completion of tasks."""
        ...


class BlobStorePort(Protocol):
    """Port for blob storage operations."""

    def put(self, key: str, data: bytes, meta: dict) -> Result[str, DomainError]:
        """Put blob data with metadata. Returns storage key."""
        ...

    def get(self, key: str) -> Result[bytes, DomainError]:
        """Get blob data by key."""
        ...


class TelemetryPort(Protocol):
    """Port for telemetry and monitoring."""

    def incr(self, name: str, tags: dict) -> None:
        """Increment a counter metric."""
        ...

    def observe(self, name: str, value: float, tags: dict) -> None:
        """Observe a value for histogram/summary metric."""
        ...
