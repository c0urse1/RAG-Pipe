from bu_superagent.application.ports.clock_port import ClockPort
from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.llm_port import LLMPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.application.use_cases.ingest_documents import IngestDocuments
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.config.settings import AppSettings
from bu_superagent.infrastructure.embeddings.hf_sentence_transformers import HFEmbeddingAdapter
from bu_superagent.infrastructure.llm.vllm_openai_adapter import VLLMOpenAIAdapter
from bu_superagent.infrastructure.parsing.pdf_text_extractor import PDFTextExtractorAdapter
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


def build_query_use_case(with_llm: bool = True) -> QueryKnowledgeBase:
    """Build QueryKnowledgeBase use case.

    Args:
        with_llm: If True, includes LLM for generative answers.
                  If False, uses extractive fallback (concatenated chunks).
    """
    settings = AppSettings()
    return QueryKnowledgeBase(
        vector_store=build_vector_store(settings),
        embedding=build_embedding(settings),
        llm=build_llm(settings) if with_llm else None,
    )
