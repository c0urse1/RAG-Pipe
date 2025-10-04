import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppSettings:
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
        )
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))

    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "var/chroma/e5_1024d"))
    collection: str = field(
        default_factory=lambda: os.getenv("VECTOR_COLLECTION", "kb_chunks_de_1024d")
    )
    store_text_payload: bool = field(
        default_factory=lambda: os.getenv("STORE_TEXT_PAYLOAD", "false").lower() == "true"
    )
    vector_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "chroma").lower()
    )

    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))

    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    )
