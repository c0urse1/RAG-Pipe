from bu_superagent.application.ports.clock_port import ClockPort
from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.llm_port import LLMPort
from bu_superagent.application.ports.reranker_port import RerankerPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.application.use_cases.ingest_documents import IngestDocuments
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.config.settings import AppSettings
from bu_superagent.infrastructure.embeddings.hf_sentence_transformers import HFEmbeddingAdapter
from bu_superagent.infrastructure.llm.vllm_openai_adapter import VLLMOpenAIAdapter
from bu_superagent.infrastructure.parsing.pdf_text_extractor import PDFTextExtractorAdapter
from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter
from bu_superagent.infrastructure.time.system_clock import SystemClock
from bu_superagent.infrastructure.vectorstore.chroma_vector_store import ChromaVectorStoreAdapter
from bu_superagent.infrastructure.vectorstore.faiss_vector_store import FaissVectorStoreAdapter
from bu_superagent.infrastructure.vectorstore.qdrant_vector_store import QdrantVectorStoreAdapter


def build_embedding(settings: AppSettings) -> EmbeddingPort:
    return HFEmbeddingAdapter(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )


def build_vector_store(settings: AppSettings) -> VectorStorePort:
    backend = settings.vector_backend

    if backend == "chroma":
        chroma_adapter = ChromaVectorStoreAdapter(
            persist_dir=settings.chroma_dir,
            collection=settings.collection,
        )
        if hasattr(chroma_adapter, "store_text"):
            chroma_adapter.store_text = settings.store_text_payload
        return chroma_adapter

    if backend == "qdrant":
        return QdrantVectorStoreAdapter(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.collection,
        )

    faiss_adapter = FaissVectorStoreAdapter()
    faiss_adapter.collection = settings.collection
    return faiss_adapter


def build_llm(settings: AppSettings) -> LLMPort:
    return VLLMOpenAIAdapter(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def build_clock() -> ClockPort:
    """Build clock adapter for time operations.

    Returns:
        SystemClock providing real UTC time for production use.

    Note:
        Tests should inject FakeClock or similar test doubles instead.
    """
    return SystemClock()


def build_reranker(settings: AppSettings) -> RerankerPort:
    """Build reranker adapter for semantic reranking.

    Returns:
        CrossEncoderAdapter for 2-stage retrieve→rerank RAG pattern.
        Uses cross-encoder model for higher accuracy than bi-encoder retrieval.

    Environment variables:
        RERANKER_MODEL: Model name (default: BAAI/bge-reranker-v2-m3)
        RERANKER_DEVICE: cpu or cuda (default: cpu)
        RERANKER_APPLY_SIGMOID: Apply sigmoid to scores (default: true)

    Note:
        Cross-encoders are slower than bi-encoders, so rerank only Top-K
        results (e.g., K=100) from initial retrieval, not the entire corpus.
    """
    return CrossEncoderAdapter(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        apply_sigmoid=settings.reranker_apply_sigmoid,
    )


def build_embedding_adapter(settings: AppSettings) -> EmbeddingPort:
    """Backward-compatible alias for legacy tests."""

    return build_embedding(settings)


def build_llm_adapter(settings: AppSettings) -> LLMPort:
    """Backward-compatible alias for legacy tests."""

    return build_llm(settings)


def build_ingest_use_case() -> IngestDocuments:
    settings = AppSettings()
    return IngestDocuments(
        loader=PDFTextExtractorAdapter(),
        embedding=build_embedding(settings),
        vector_store=build_vector_store(settings),
    )


def build_query_use_case(with_llm: bool = True, with_reranker: bool = False) -> QueryKnowledgeBase:
    """Build QueryKnowledgeBase use case.

    Args:
        with_llm: If True, includes LLM for generative answers.
                  If False, uses extractive fallback (concatenated chunks).
        with_reranker: If True, adds cross-encoder reranking after retrieval.
                       Improves accuracy but adds latency (2-stage RAG pattern).
    """
    settings = AppSettings()
    return QueryKnowledgeBase(
        vector_store=build_vector_store(settings),
        embedding=build_embedding(settings),
        llm=build_llm(settings) if with_llm else None,
        reranker=build_reranker(settings) if with_reranker else None,
    )


def build_work_queue(settings: AppSettings):
    """Build work queue adapter for async task distribution.

    Returns:
        Fake work queue adapter (real Redis adapter requires redis-py).
        Replace with RedisWorkQueueAdapter for production.
    """

    # Placeholder: return fake adapter for now
    # Production: from bu_superagent.infrastructure.queues.redis_streams_adapter import (
    #     RedisWorkQueueAdapter,
    # )
    class FakeWorkQueue:
        def enqueue(self, topic, payload):
            from bu_superagent.domain.types import Result

            return Result.success(f"fake-{topic}-{hash(str(payload))}")

        def dequeue_batch(self, topic, max_n):
            from bu_superagent.domain.types import Result

            return Result.success([])

        def ack(self, topic, ack_ids):
            from bu_superagent.domain.types import Result

            return Result.success(None)

    return FakeWorkQueue()


def build_blob_store(settings: AppSettings):
    """Build blob store adapter for large document storage.

    Returns:
        Fake blob store adapter (real MinIO adapter requires minio-py).
        Replace with MinioBlobStoreAdapter for production.
    """

    # Placeholder: return fake adapter for now
    # Production: from bu_superagent.infrastructure.blobstores.minio_adapter import (
    #     MinioBlobStoreAdapter,
    # )
    class FakeBlobStore:
        def put(self, key, data, meta):
            from bu_superagent.domain.types import Result

            return Result.success(key)

        def get(self, key):
            from bu_superagent.domain.types import Result

            return Result.success(b"")

    return FakeBlobStore()


def build_telemetry(settings: AppSettings):
    """Build telemetry adapter for metrics and tracing.

    Returns:
        OpenTelemetryAdapter (gracefully degrades if opentelemetry-sdk not installed).
    """
    from bu_superagent.infrastructure.telemetry.otel_adapter import OpenTelemetryAdapter, OtelConfig

    cfg = OtelConfig(
        service_name="bu-superagent",
        otlp_endpoint=None,  # Set via env var or settings
        environment="production",
        enable_console=False,
    )
    return OpenTelemetryAdapter(cfg)
