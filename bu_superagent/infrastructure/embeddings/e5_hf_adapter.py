"""E5 HuggingFace embedding adapter with GPU batching and streaming.

Why: Große Batches + GPU → hoher Durchsatz beim Ingest.
"""

from importlib import import_module
from typing import Any, Protocol

from bu_superagent.domain.errors import DomainError, EmbeddingError
from bu_superagent.domain.types import Result, Vector


class EmbeddingPort(Protocol):
    """Port for text embedding operations."""

    def embed_texts(self, texts: list[str]) -> Result[list[Vector], DomainError]:
        """Embed multiple texts into vectors."""
        ...


def _prefix_e5_passage(text: str) -> str:
    """Prefix text for E5 passage encoding.

    E5 models require specific instruction prefixes for optimal performance.
    """
    return f"passage: {text}"


def _prefix_e5_query(text: str) -> str:
    """Prefix text for E5 query encoding.

    E5 models require specific instruction prefixes for optimal performance.
    """
    return f"query: {text}"


class E5HFEmbeddingAdapter(EmbeddingPort):
    """E5 embedding adapter with GPU batching for high-throughput ingestion.

    Features:
    - Lazy model loading with caching
    - GPU batch processing (configurable batch size)
    - Automatic E5 instruction prefixing
    - L2 normalization for cosine similarity
    - Streaming support for large datasets

    Why: Large batches + GPU enable high throughput during ingestion.
         E5-instruct models provide strong multilingual performance.
    """

    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
        batch_size: int = 512,
    ) -> None:
        """Initialize E5 embedding adapter.

        Args:
            model_id: HuggingFace model ID (default: E5 multilingual instruct)
            device: Device for inference ("cuda", "cpu", "mps")
            batch_size: Batch size for GPU processing (512 is GPU-optimal)
        """
        self._model_id = model_id
        self._device = device
        self._bs = batch_size
        self._model = None  # Lazy loading

    def _load_model(self) -> Any:
        """Load sentence-transformers model with lazy import.

        Returns:
            SentenceTransformer model instance

        Raises:
            EmbeddingError: If model loading fails
        """
        try:
            # Lazy import for testability
            st_module = import_module("sentence_transformers")
            SentenceTransformer = st_module.SentenceTransformer

            # Load model with device placement
            model = SentenceTransformer(self._model_id, device=self._device)

            return model

        except Exception as ex:
            raise EmbeddingError(f"load failed: {ex}") from ex

    def embed_texts(self, texts: list[str]) -> Result[list[Vector], DomainError]:
        """Embed multiple texts with GPU batching and normalization.

        Pipeline:
        1. Prefix texts with E5 passage instruction
        2. Batch encode with GPU (batch_size chunks)
        3. L2 normalize embeddings for cosine similarity
        4. Convert to tuples (domain Vector type)

        Args:
            texts: List of text strings to embed

        Returns:
            Result with list of Vector tuples or EmbeddingError
        """
        try:
            # Lazy load model on first use
            if self._model is None:
                self._model = self._load_model()

            # Prefix texts with E5 passage instruction
            prefixed = [_prefix_e5_passage(t) for t in texts]

            # Batch encode with GPU
            # sentence-transformers handles batching internally
            embeddings = self._model.encode(  # type: ignore[attr-defined]
                prefixed,
                batch_size=self._bs,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=False,  # Disable progress for production
                convert_to_numpy=True,  # NumPy arrays for efficiency
            )

            # Convert to domain Vector type (tuples)
            vectors: list[Vector] = [tuple(emb.tolist()) for emb in embeddings]

            return Result.success(vectors)

        except EmbeddingError:
            # Re-raise domain errors
            return Result.failure(EmbeddingError("embed failed: already wrapped"))
        except Exception as ex:
            return Result.failure(EmbeddingError(f"embed failed: {ex}"))

    def embed_query(self, query: str) -> Result[Vector, DomainError]:
        """Embed single query text with E5 query instruction.

        Args:
            query: Query text to embed

        Returns:
            Result with Vector tuple or EmbeddingError
        """
        try:
            # Lazy load model on first use
            if self._model is None:
                self._model = self._load_model()

            # Prefix query with E5 query instruction
            prefixed = _prefix_e5_query(query)

            # Encode single query
            embedding = self._model.encode(  # type: ignore[attr-defined]
                prefixed,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert to domain Vector type (tuple)
            vector: Vector = tuple(embedding.tolist())

            return Result.success(vector)

        except EmbeddingError:
            # Re-raise domain errors
            return Result.failure(EmbeddingError("embed failed: already wrapped"))
        except Exception as ex:
            return Result.failure(EmbeddingError(f"embed failed: {ex}"))

    def embed_texts_stream(
        self, texts: list[str], chunk_size: int = 1024
    ) -> Result[list[Vector], DomainError]:
        """Embed texts with streaming for very large datasets.

        Processes texts in chunks to avoid OOM on GPU. Useful for
        ingesting millions of documents.

        Args:
            texts: List of text strings to embed
            chunk_size: Size of each streaming chunk (default: 1024)

        Returns:
            Result with list of Vector tuples or EmbeddingError
        """
        try:
            all_vectors: list[Vector] = []

            # Process in chunks
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                result = self.embed_texts(chunk)

                if not result.ok:
                    return result  # Propagate error

                assert result.value is not None
                all_vectors.extend(result.value)

            return Result.success(all_vectors)

        except Exception as ex:
            return Result.failure(EmbeddingError(f"streaming embed failed: {ex}"))
