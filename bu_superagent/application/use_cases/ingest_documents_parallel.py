"""Parallel document ingestion use case with batch processing.

Why: Ingestion in 512er-Batches (GPU-freundlich), strikt Result-basiert,
     Queue-fähig (optional wq.enqueue im Orchestrator-Use-Case).
"""

from bu_superagent.application.ports import EmbeddingPort, VectorStorePort, WorkQueuePort
from bu_superagent.domain.errors import DomainError, ValidationError
from bu_superagent.domain.types import Result


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

    def plan(self, req: any) -> Result[list[dict[str, any]], ValidationError]:  # type: ignore[misc]
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

    def execute_batch(self, collection: str, batch: dict[str, any]) -> Result[None, DomainError]:  # type: ignore[misc]
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
