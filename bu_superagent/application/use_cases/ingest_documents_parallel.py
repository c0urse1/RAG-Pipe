"""Parallel document ingestion use case with batch processing.

Why: Ingestion in 512er-Batches (GPU-freundlich), strikt Result-basiert,
     Queue-fähig (optional wq.enqueue im Orchestrator-Use-Case).
"""

from typing import Any, Protocol

from bu_superagent.application.dtos import IngestRequest
from bu_superagent.application.ports.work_queue_port import WorkQueuePort
from bu_superagent.domain.errors import DomainError, ValidationError
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
        metadata: list[dict[str, Any]],
    ) -> Result[None, DomainError]:
        """Upsert vectors with metadata into collection."""
        ...

    def search(
        self,
        collection: str,
        vector: Vector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> Result[list[dict[str, Any]], DomainError]:
        """Search for similar vectors in collection."""
        ...


class IngestDocumentsParallel:
    """Parallel document ingestion with batch planning and execution."""

    def __init__(
        self,
        embed: EmbeddingPort,
        vs: VectorStorePort,
        wq: WorkQueuePort,
    ) -> None:
        """Initialize with embedding, vector store, and work queue ports."""
        self.embed = embed
        self.vs = vs
        self.wq = wq

    def plan(self, req: IngestRequest) -> Result[list[dict[str, Any]], ValidationError]:
        """Plan ingestion by splitting documents into GPU-friendly batches.

        Args:
            req: IngestRequest with docs to process

        Returns:
            Result with list of batch dicts (each: {"docs": [...]})
            or ValidationError if input invalid
        """
        if not req.docs:
            return Result.failure(ValidationError("empty docs"))

        # Split into batches (pure planning)
        batches = []
        batch = []
        for d in req.docs:
            batch.append(d)
            if len(batch) >= 512:
                batches.append({"docs": batch})
                batch = []
        if batch:
            batches.append({"docs": batch})

        return Result.success(batches)

    def execute_batch(self, collection: str, batch: dict[str, Any]) -> Result[None, DomainError]:
        """Execute a single batch: embed texts and upsert to vector store.

        Args:
            collection: Target collection name
            batch: Batch dict with "docs" key containing document list

        Returns:
            Result with None on success, or DomainError on failure
        """
        texts = [d["text"] for d in batch["docs"]]
        ids = [d["id"] for d in batch["docs"]]
        metas = [d["meta"] for d in batch["docs"]]

        # Embed texts
        r_vecs = self.embed.embed_texts(texts)
        if not r_vecs.ok:
            assert r_vecs.error is not None
            return Result.failure(r_vecs.error)  # typed

        # Upsert to vector store
        assert r_vecs.value is not None
        r_up = self.vs.upsert(collection, ids, r_vecs.value, metas)
        if not r_up.ok:
            assert r_up.error is not None
            return Result.failure(r_up.error)

        return Result.success(None)
