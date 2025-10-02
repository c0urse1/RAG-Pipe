"""
# Composition Root: injiziert konkrete Adapter in Use-Cases.
# Wichtig: keine Geschäftslogik, nur Wiring.
"""

from bu_superagent.config.settings import AppSettings
from bu_superagent.infrastructure.llm.vllm_openai_adapter import VLLMOpenAIAdapter
from bu_superagent.infrastructure.embeddings.sentence_transformers_adapter import (
    SentenceTransformersEmbeddingAdapter,
)
from bu_superagent.infrastructure.vectorstore.qdrant_vector_store import (
    QdrantVectorStoreAdapter,
)
from bu_superagent.infrastructure.vectorstore.chroma_vector_store import (
    ChromaVectorStoreAdapter,
)
from bu_superagent.infrastructure.vectorstore.faiss_vector_store import (
    FaissVectorStoreAdapter,
)
from bu_superagent.infrastructure.parsing.pdf_text_extractor import PDFTextExtractorAdapter
from bu_superagent.application.use_cases.ingest_documents import IngestDocuments
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase


def _build_vector_store(s: AppSettings):
    vb = (s.vector_backend or "").lower()
    if vb == "qdrant":
        return QdrantVectorStoreAdapter(host=s.qdrant_host, port=s.qdrant_port, collection=s.qdrant_collection)
    if vb == "faiss":
        return FaissVectorStoreAdapter(collection=s.qdrant_collection)
    if vb == "chroma":
        return ChromaVectorStoreAdapter(persist_dir=s.chroma_dir, collection=s.qdrant_collection)
    # Fallback: choose FAISS (no external service) to keep tests local-only
    return FaissVectorStoreAdapter(collection=s.qdrant_collection)


def build_ingest_use_case() -> IngestDocuments:
    s = AppSettings()
    loader = PDFTextExtractorAdapter()
    emb = SentenceTransformersEmbeddingAdapter(
        model_mxbai=s.embedding_primary,
        model_jina=s.embedding_fallback,
        model_e5=s.embedding_e5,
        device="cuda",
    )
    vs = _build_vector_store(s)
    return IngestDocuments(loader=loader, embedding=emb, vector_store=vs)


def build_query_use_case() -> QueryKnowledgeBase:
    s = AppSettings()

    llm = VLLMOpenAIAdapter(base_url=s.vllm_base_url, model=s.vllm_model)
    emb = SentenceTransformersEmbeddingAdapter(
        model_mxbai=s.embedding_primary,
        model_jina=s.embedding_fallback,
        model_e5=s.embedding_e5,
        device="cuda",  # falls verfügbar, sonst "cpu"
    )
    vs = _build_vector_store(s)
    # Optionally ensure collection for stores that need upfront sizing (Qdrant); Chroma/FAISS handle internally
    if isinstance(vs, QdrantVectorStoreAdapter):
        vs.ensure_collection(s.qdrant_collection, s.embedding_dim)

    # Use-Case braucht nur Ports; keine Technologie im Use-Case selbst.
    return QueryKnowledgeBase(vector_store=vs, embedding=emb)


# Optional: separate builders (keine Geschäftslogik, nur Wiring)
def build_embedding_adapter(settings: AppSettings | None = None) -> SentenceTransformersEmbeddingAdapter:
    s = settings or AppSettings()
    return SentenceTransformersEmbeddingAdapter(
        model_mxbai=s.embedding_primary,
        model_jina=s.embedding_fallback,
        model_e5=s.embedding_e5,
    )


def build_vector_store_adapter(settings: AppSettings | None = None) -> QdrantVectorStoreAdapter:
    s = settings or AppSettings()
    adapter = QdrantVectorStoreAdapter(
        host=s.qdrant_host, port=s.qdrant_port, collection=s.qdrant_collection
    )
    return adapter


def build_llm_adapter(settings: AppSettings | None = None) -> VLLMOpenAIAdapter:
    s = settings or AppSettings()
    return VLLMOpenAIAdapter(base_url=s.vllm_base_url, api_key=s.llm_api_key, model=s.vllm_model)
