from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class AppSettings:
    # Legacy/defaults retained for compatibility
    vector_dir: str = "var/vector_store/e5_large"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = ""  # vLLM may ignore; keep for compatibility
    llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "kb_chunks_de_1024d"))
    qdrant_distance: str = "Cosine"
    embedding_dim: int = 1024

    # New preferred names per 4.5
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding_primary: str = "mixedbread-ai/mxbai-embed-de-large-v1"
    embedding_fallback: str = "jinaai/jina-embeddings-v2-base-de"
    embedding_e5: str = "intfloat/multilingual-e5-large-instruct"

    # Vector backend switching
    vector_backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "chroma"))  # "qdrant" | "chroma" | "faiss"
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "var/chroma/e5_1024d"))
