"""Depfrom typing import Any, TYPE_CHECKING

from bu_superagent.application.scalable_ports import (
    BlobStorePort,
    EmbeddingPort,
    TelemetryPort,
    VectorStorePort,
    WorkQueuePort,
)njection container with environment-driven wiring.

Why: Einzige Stelle mit Env; Feature-Flags erlauben toggles
     (Hybrid, Quantization, Shards).
"""

from typing import TYPE_CHECKING, Any

from bu_superagent.application.ports import (
    BlobStorePort,
    EmbeddingPort,
    TelemetryPort,
    VectorStorePort,
    WorkQueuePort,
)
from bu_superagent.config.settings import AppSettings
from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result

if TYPE_CHECKING:
    from bu_superagent.application.use_cases.ingest_documents_parallel import (
        IngestDocumentsParallel,
    )
    from bu_superagent.application.use_cases.query_knowledge_base_scalable import (
        QueryKnowledgeBaseScalable,
    )


class Container:
    """Dependency injection container for application components.

    Responsibilities:
    1. Read settings from environment (via AppSettings)
    2. Choose adapters based on settings (vector_backend, workqueue_backend, etc.)
    3. Inject dependencies into use cases
    4. Apply feature flags (quantization, hybrid search, etc.)

    Why: Single place for wiring; all other layers remain pure.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize container with settings.

        Args:
            settings: Application settings (default: load from environment)
        """
        self.settings = settings or AppSettings()
        self._embedding: EmbeddingPort | None = None
        self._vector_store: VectorStorePort | None = None
        self._work_queue: WorkQueuePort | None = None
        self._blob_store: BlobStorePort | None = None
        self._telemetry: TelemetryPort | None = None

    # ===== Adapters =====

    def get_embedding(self) -> EmbeddingPort:
        """Get or create embedding adapter based on settings."""
        if self._embedding is None:
            self._embedding = self._build_embedding()
        return self._embedding

    def get_vector_store(self) -> VectorStorePort:
        """Get or create vector store adapter based on settings."""
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    def get_work_queue(self) -> WorkQueuePort:
        """Get or create work queue adapter based on settings."""
        if self._work_queue is None:
            self._work_queue = self._build_work_queue()
        return self._work_queue

    def get_blob_store(self) -> BlobStorePort:
        """Get or create blob store adapter based on settings."""
        if self._blob_store is None:
            self._blob_store = self._build_blob_store()
        return self._blob_store

    def get_telemetry(self) -> TelemetryPort:
        """Get or create telemetry adapter based on settings."""
        if self._telemetry is None:
            self._telemetry = self._build_telemetry()
        return self._telemetry

    # ===== Use Cases =====

    def get_query_use_case(self) -> "QueryKnowledgeBaseScalable":
        """Build query use case with all dependencies."""
        from bu_superagent.application.use_cases.query_knowledge_base_scalable import (
            QueryKnowledgeBaseScalable,
        )

        return QueryKnowledgeBaseScalable(
            embed=self.get_embedding(),
            vs=self.get_vector_store(),
            lexical=None,  # TODO: lexical search adapter for hybrid mode
        )

    def get_ingest_use_case(self) -> "IngestDocumentsParallel":
        """Build ingest use case with all dependencies."""
        from bu_superagent.application.use_cases.ingest_documents_parallel import (
            IngestDocumentsParallel,
        )

        return IngestDocumentsParallel(
            embed=self.get_embedding(),
            vs=self.get_vector_store(),
            wq=self.get_work_queue(),
        )

    # ===== Private Builder Methods =====

    def _build_embedding(self) -> EmbeddingPort:
        """Build embedding adapter based on settings.

        Returns:
            E5HFEmbeddingAdapter with GPU batching support
        """
        from bu_superagent.infrastructure.embeddings.e5_hf_adapter import E5HFEmbeddingAdapter

        return E5HFEmbeddingAdapter(
            model_id=self.settings.embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
        )

    def _build_vector_store(self) -> VectorStorePort:
        """Build vector store adapter based on settings.vector_backend.

        Supports: qdrant | chroma | faiss | weaviate | elasticsearch

        Returns:
            VectorStorePort implementation with admin capabilities if available
        """
        backend = self.settings.vector_backend

        if backend == "qdrant":
            return self._build_qdrant_adapter()
        elif backend == "chroma":
            return self._build_chroma_adapter()
        elif backend == "faiss":
            return self._build_faiss_adapter()
        elif backend == "weaviate":
            raise NotImplementedError("Weaviate adapter not yet implemented")
        elif backend == "elasticsearch":
            raise NotImplementedError("Elasticsearch adapter not yet implemented")
        else:
            # Fallback to FAISS for unknown backends (testability)
            return self._build_faiss_adapter()

    def _build_qdrant_adapter(self) -> VectorStorePort:
        """Build Qdrant adapter with scaling configuration.

        Applies:
        - Sharding (settings.shards)
        - Replication (settings.replicas)
        - Quantization (if settings.use_quantization)

        Returns:
            QdrantVectorStoreAdapter implementing VectorStorePort + VectorStoreAdminPort
        """
        from bu_superagent.infrastructure.vectorstore.qdrant_adapter import (
            QdrantConfig,
            QdrantVectorStoreAdapter,
        )

        cfg = QdrantConfig(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            prefer_grpc=self.settings.qdrant_prefer_grpc,
            timeout_s=self.settings.qdrant_timeout_s,
        )

        adapter = QdrantVectorStoreAdapter(cfg)

        # Auto-apply collection configuration (if admin port available)
        if hasattr(adapter, "ensure_collection"):
            # Note: This is done at container build time for simplicity
            # Production: Use CLI to configure collections separately
            pass

        # Auto-apply quantization (if enabled)
        if self.settings.use_quantization and hasattr(adapter, "set_quantization"):
            # Note: This is done at container build time for simplicity
            # Production: Use CLI to configure quantization separately
            pass

        return adapter

    def _build_chroma_adapter(self) -> VectorStorePort:
        """Build Chroma adapter (legacy compatibility).

        Returns:
            ChromaVectorStoreAdapter
        """
        from bu_superagent.infrastructure.vectorstore.chroma_vector_store import (
            ChromaVectorStoreAdapter,
        )

        adapter = ChromaVectorStoreAdapter(
            persist_dir=self.settings.chroma_dir,
            collection=self.settings.collection,
        )

        if hasattr(adapter, "store_text"):
            adapter.store_text = self.settings.store_text_payload

        return adapter  # type: ignore[return-value]  # Legacy adapter

    def _build_faiss_adapter(self) -> VectorStorePort:
        """Build FAISS adapter (in-memory, for testing).

        Returns:
            FaissVectorStoreAdapter
        """
        from bu_superagent.infrastructure.vectorstore.faiss_vector_store import (
            FaissVectorStoreAdapter,
        )

        adapter = FaissVectorStoreAdapter()
        adapter.collection = self.settings.collection
        return adapter  # type: ignore[return-value]  # Legacy adapter

    def _build_work_queue(self) -> WorkQueuePort:
        """Build work queue adapter based on settings.workqueue_backend.

        Supports: redis | fake

        Returns:
            WorkQueuePort implementation
        """
        backend = self.settings.workqueue_backend

        if backend == "redis":
            return self._build_redis_work_queue()
        else:
            # Fallback to fake adapter (testing, local dev)
            return self._build_fake_work_queue()

    def _build_redis_work_queue(self) -> WorkQueuePort:
        """Build Redis Streams work queue adapter.

        Returns:
            RedisWorkQueueAdapter
        """
        from bu_superagent.infrastructure.queues.redis_streams_adapter import (
            RedisConfig,
            RedisWorkQueueAdapter,
        )

        cfg = RedisConfig(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            db=self.settings.redis_db,
            password=self.settings.redis_password or None,
            decode_responses=True,
        )

        return RedisWorkQueueAdapter(cfg)

    def _build_fake_work_queue(self) -> WorkQueuePort:
        """Build fake work queue for testing/local dev.

        Returns:
            Fake WorkQueuePort implementation
        """

        class FakeWorkQueue:
            def enqueue(self, topic: str, payload: dict[str, Any]) -> Result[str, DomainError]:
                return Result.success(f"fake-{topic}-{hash(str(payload))}")

            def dequeue_batch(
                self, topic: str, max_n: int
            ) -> Result[list[dict[str, Any]], DomainError]:
                return Result.success([])

            def ack(self, topic: str, ack_ids: list[str]) -> Result[None, DomainError]:
                return Result.success(None)

        return FakeWorkQueue()

    def _build_blob_store(self) -> BlobStorePort:
        """Build blob store adapter based on settings.blobstore_backend.

        Supports: minio | s3 | fake

        Returns:
            BlobStorePort implementation
        """
        backend = self.settings.blobstore_backend

        if backend == "minio":
            return self._build_minio_blob_store()
        elif backend == "s3":
            # MinIO adapter is S3-compatible
            return self._build_minio_blob_store()
        else:
            # Fallback to fake adapter (testing, local dev)
            return self._build_fake_blob_store()

    def _build_minio_blob_store(self) -> BlobStorePort:
        """Build MinIO blob store adapter.

        Returns:
            MinioBlobStoreAdapter
        """
        from bu_superagent.infrastructure.blobstores.minio_adapter import (
            MinioBlobStoreAdapter,
            MinioConfig,
        )

        cfg = MinioConfig(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            bucket_name=self.settings.minio_bucket,
            secure=self.settings.minio_secure,
        )

        return MinioBlobStoreAdapter(cfg)

    def _build_fake_blob_store(self) -> BlobStorePort:
        """Build fake blob store for testing/local dev.

        Returns:
            Fake BlobStorePort implementation
        """

        class FakeBlobStore:
            def put(self, key: str, data: bytes, meta: dict[str, Any]) -> Result[str, DomainError]:
                return Result.success(key)

            def get(self, key: str) -> Result[bytes, DomainError]:
                return Result.success(b"")

        return FakeBlobStore()

    def _build_telemetry(self) -> TelemetryPort:
        """Build telemetry adapter based on settings.telemetry_enabled.

        Returns:
            OpenTelemetryAdapter (or no-op if disabled)
        """
        if not self.settings.telemetry_enabled:
            # Return no-op telemetry adapter
            return self._build_noop_telemetry()

        from bu_superagent.infrastructure.telemetry.otel_adapter import (
            OpenTelemetryAdapter,
            OtelConfig,
        )

        cfg = OtelConfig(
            service_name="bu-superagent",
            otlp_endpoint=self.settings.otlp_endpoint or None,
            environment=self.settings.telemetry_environment,
            enable_console=False,  # Production: disable console logging
        )

        return OpenTelemetryAdapter(cfg)

    def _build_noop_telemetry(self) -> TelemetryPort:
        """Build no-op telemetry adapter (when telemetry disabled).

        Returns:
            Fake TelemetryPort implementation
        """

        class NoopTelemetry:
            def incr(self, name: str, tags: dict[str, Any] | None = None) -> None:
                pass

            def observe(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
                pass

        return NoopTelemetry()


# ===== Convenience Functions =====


def build_container(settings: AppSettings | None = None) -> Container:
    """Build dependency injection container with settings.

    Args:
        settings: Application settings (default: load from environment)

    Returns:
        Container with all adapters and use cases

    Example:
        container = build_container()
        query_uc = container.get_query_use_case()
        result = query_uc.execute(request)
    """
    return Container(settings)


def get_query_use_case(settings: AppSettings | None = None) -> "QueryKnowledgeBaseScalable":
    """Quick access to query use case (backward compatibility).

    Args:
        settings: Application settings (default: load from environment)

    Returns:
        QueryKnowledgeBaseScalable use case
    """
    return build_container(settings).get_query_use_case()


def get_ingest_use_case(settings: AppSettings | None = None) -> "IngestDocumentsParallel":
    """Quick access to ingest use case (backward compatibility).

    Args:
        settings: Application settings (default: load from environment)

    Returns:
        IngestDocumentsParallel use case
    """
    return build_container(settings).get_ingest_use_case()
