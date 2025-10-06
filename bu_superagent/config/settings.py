"""Application settings with environment-driven configuration.

Why: Einzige Stelle mit Env; Feature-Flags erlauben toggles
     (Hybrid, Quantization, Shards).
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppSettings:
    """Application settings loaded from environment variables.

    This is the ONLY place where environment variables are read.
    All other layers receive settings via dependency injection.

    Feature Flags:
    - use_quantization: Enable vector compression (4-32x memory savings)
    - query_use_hybrid: Enable hybrid search (vector + lexical fusion)
    - store_text_payload: Store full text in vector DB (vs. external blob store)
    """

    # ===== Vector Store Configuration =====
    vector_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "qdrant").lower()
    )
    # Supported: "qdrant" | "chroma" | "faiss" | "weaviate" | "elasticsearch"

    # Qdrant-specific
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_prefer_grpc: bool = field(
        default_factory=lambda: os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    )
    qdrant_timeout_s: int = field(default_factory=lambda: int(os.getenv("QDRANT_TIMEOUT_S", "30")))

    # Chroma-specific (legacy)
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "var/chroma/e5_1024d"))

    # Collection settings
    collection: str = field(
        default_factory=lambda: os.getenv("VECTOR_COLLECTION", "kb_chunks_de_1024d")
    )
    store_text_payload: bool = field(
        default_factory=lambda: os.getenv("STORE_TEXT_PAYLOAD", "false").lower() == "true"
    )

    # ===== Scaling Configuration =====
    shards: int = field(default_factory=lambda: int(os.getenv("VECTOR_SHARDS", "6")))
    replicas: int = field(default_factory=lambda: int(os.getenv("VECTOR_REPLICAS", "2")))

    # Quantization (compression)
    use_quantization: bool = field(
        default_factory=lambda: os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    )
    quantization_kind: str = field(
        default_factory=lambda: os.getenv("QUANTIZATION_KIND", "scalar").lower()
    )
    # Supported: "scalar" (INT8, 4x compression) | "product" (16x) | "binary" (32x)

    # ===== Embedding Configuration =====
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
        )
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    # Supported: "cpu" | "cuda" | "mps"

    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
    )
    # GPU-optimal batch size (512 for RTX 3090/4090)

    # ===== Work Queue Configuration =====
    workqueue_backend: str = field(
        default_factory=lambda: os.getenv("WORKQUEUE_BACKEND", "fake").lower()
    )
    # Supported: "redis" | "fake" (for testing)

    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    # ===== Blob Storage Configuration =====
    blobstore_backend: str = field(
        default_factory=lambda: os.getenv("BLOBSTORE_BACKEND", "fake").lower()
    )
    # Supported: "minio" | "s3" | "fake" (for testing)

    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000")
    )
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "rag-documents"))
    minio_secure: bool = field(
        default_factory=lambda: os.getenv("MINIO_SECURE", "true").lower() == "true"
    )

    # ===== LLM Configuration =====
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    )

    # ===== Reranker Configuration =====
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    reranker_device: str = field(default_factory=lambda: os.getenv("RERANKER_DEVICE", "cpu"))
    reranker_apply_sigmoid: bool = field(
        default_factory=lambda: os.getenv("RERANKER_APPLY_SIGMOID", "true").lower() == "true"
    )

    # ===== Query Configuration (Feature Flags) =====
    query_use_hybrid: bool = field(
        default_factory=lambda: os.getenv("QUERY_USE_HYBRID", "true").lower() == "true"
    )
    # Enable hybrid search (vector + lexical fusion with RRF)

    query_use_mmr: bool = field(
        default_factory=lambda: os.getenv("QUERY_USE_MMR", "true").lower() == "true"
    )
    # Enable MMR for diversity

    query_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("QUERY_CONFIDENCE_THRESHOLD", "0.25"))
    )
    # Minimum confidence for answering (0.0-1.0)

    # ===== Telemetry Configuration =====
    telemetry_enabled: bool = field(
        default_factory=lambda: os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    )
    otlp_endpoint: str = field(default_factory=lambda: os.getenv("OTLP_ENDPOINT", ""))
    # Empty string = no OTLP export (console only if telemetry_enabled)

    telemetry_environment: str = field(
        default_factory=lambda: os.getenv("TELEMETRY_ENVIRONMENT", "production")
    )

    # ===== Legacy Compatibility =====
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    # Legacy: Use qdrant_url instead (http://host:port)

    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    # Legacy: Use qdrant_url instead
