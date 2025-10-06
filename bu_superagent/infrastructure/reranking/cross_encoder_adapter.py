"""Cross-encoder reranker adapter using sentence-transformers.

This adapter implements semantic reranking using cross-encoder models,
which score query-candidate pairs directly for higher accuracy than
bi-encoder retrieval alone.

Why (SAM): Infrastructure adapters wrap external libraries with lazy imports,
cache expensive resources (models), and map errors to domain exceptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from ...application.ports.reranker_port import RerankerPort


class CrossEncoderAdapter(RerankerPort):
    """Cross-encoder reranker using sentence-transformers CrossEncoder.

    Cross-encoders score query-document pairs directly (not via separate vectors),
    providing higher accuracy than bi-encoder retrieval. This is a 2-stage RAG
    pattern: retrieve Top-K with fast bi-encoder → rerank with slow cross-encoder.

    Why (SAM):
    - Lazy imports via importlib for testability (fake modules in tests)
    - Model caching to avoid reloading on each call
    - Error wrapping to domain exceptions (infrastructure → domain mapping)
    - Sigmoid applied for BGE models (raw logits → [0,1] scores)

    Recommended models:
    - BAAI/bge-reranker-v2-m3 (multilingual, high accuracy, apply sigmoid)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (English, fast, direct scores)

    Args:
        model_name: HuggingFace model identifier for cross-encoder
        device: Computation device ("cpu" or "cuda")
        apply_sigmoid: Apply sigmoid to raw logits (True for BGE models)

    Raises:
        RuntimeError: If model loading fails (wrapped to domain error by caller)
    """

    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        apply_sigmoid: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.apply_sigmoid = apply_sigmoid
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Lazy-load and cache cross-encoder model.

        Returns:
            Loaded CrossEncoder model instance

        Raises:
            RuntimeError: If sentence_transformers import or model load fails
        """
        cache_key = f"{self.model_name}:{self.device}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            # Lazy import for testability
            st_module = import_module("sentence_transformers")
            CrossEncoder = st_module.CrossEncoder

            model = CrossEncoder(self.model_name, device=self.device)
            self._model_cache[cache_key] = model
            return model

        except Exception as e:
            msg = f"Failed to load cross-encoder model {self.model_name}: {e}"
            raise RuntimeError(msg) from e

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Score query-candidate pairs using cross-encoder.

        Args:
            query: Query text
            candidates: List of candidate texts to score

        Returns:
            List of relevance scores (higher = more relevant).
            Scores are in [0,1] range if apply_sigmoid=True, else raw logits.

        Raises:
            RuntimeError: If model loading or scoring fails

        Note:
            Empty candidates list returns empty scores list.
            Cross-encoder scoring is O(n) with query length × candidate length,
            so rerank only Top-K results (e.g., K=100) from retrieval.
        """
        if not candidates:
            return []

        if self._model is None:
            self._model = self._load_model()

        try:
            # Build query-candidate pairs for cross-encoder
            pairs = [(query, str(c)) for c in candidates]

            # Score pairs
            scores = self._model.predict(
                pairs,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Apply sigmoid for BGE models (raw logits → [0,1] probabilities)
            if self.apply_sigmoid:
                import_module("torch")  # Ensure torch is available
                np = import_module("numpy")
                scores = 1.0 / (1.0 + np.exp(-scores))

            return list(scores.tolist())

        except Exception as e:
            msg = f"Cross-encoder scoring failed for query '{query[:50]}...': {e}"
            raise RuntimeError(msg) from e
