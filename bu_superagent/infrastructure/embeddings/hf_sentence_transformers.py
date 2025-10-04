from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Sequence as SequenceType
from dataclasses import dataclass
from typing import Any, cast

from bu_superagent.application.ports.embedding_port import EmbeddingKind, EmbeddingPort
from bu_superagent.domain.errors import EmbeddingError

# Lazy import for testability (allow monkeypatching fake SentenceTransformer)
SentenceTransformer: Any | None
try:  # pragma: no cover - exercised via tests with monkeypatch
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None
else:  # pragma: no cover - exercised in integration
    SentenceTransformer = _SentenceTransformer


def _e5_query_prefix(query: str) -> str:
    return f"Instruct: Retrieve relevant passages for the query.\nQuery: {query}"


def _e5_passage_prefix(passage: str) -> str:
    return f"Passage: {passage}"


@dataclass
class HFEmbeddingAdapter(EmbeddingPort):
    """HuggingFace Sentence-Transformers adapter for multilingual E5 embeddings."""

    model_name: str = "intfloat/multilingual-e5-large-instruct"
    device: str = "cpu"  # switch to "cuda" when available
    local_files_only: bool = False  # support offline deployments
    _model: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise EmbeddingError("sentence-transformers not installed.")
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        except Exception as ex:  # noqa: BLE001
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {ex}"
            ) from ex

    def embed_texts(self, texts: Sequence[str], kind: EmbeddingKind = "e5") -> list[list[float]]:
        if kind != "e5":
            raise EmbeddingError(f"Unsupported kind '{kind}', only 'e5' supported here.")
        self._ensure_model()
        model = self._model
        if model is None:  # pragma: no cover - defensive guard
            raise EmbeddingError("Embedding model failed to initialize.")
        try:
            inputs = [_e5_passage_prefix(t) for t in texts]
            raw_vectors = model.encode(
                inputs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as ex:  # noqa: BLE001
            raise EmbeddingError(f"Embedding texts failed: {ex}") from ex
        vectors = cast(SequenceType[SequenceType[float]], raw_vectors)
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, text: str, kind: EmbeddingKind = "e5") -> list[float]:
        if kind != "e5":
            raise EmbeddingError(f"Unsupported kind '{kind}', only 'e5' supported here.")
        self._ensure_model()
        model = self._model
        if model is None:  # pragma: no cover - defensive guard
            raise EmbeddingError("Embedding model failed to initialize.")
        try:
            query = _e5_query_prefix(text)
            raw_vector = model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as ex:  # noqa: BLE001
            raise EmbeddingError(f"Embedding query failed: {ex}") from ex
        vector = cast(SequenceType[float], raw_vector)
        return [float(x) for x in vector]
