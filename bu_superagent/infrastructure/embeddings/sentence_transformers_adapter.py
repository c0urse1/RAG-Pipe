from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bu_superagent.application.ports.embedding_port import EmbeddingKind, EmbeddingPort

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def _prefix_e5_query(text: str) -> str:
    return f"Instruct: Retrieve relevant passages for the query.\nQuery: {text}"


def _prefix_e5_passage(text: str) -> str:
    return f"Passage: {text}"


@dataclass
class SentenceTransformersEmbeddingAdapter(EmbeddingPort):
    # default Primary (de-stark)
    model_mxbai: str = "mixedbread-ai/mxbai-embed-de-large-v1"
    # Fallback/Hybrid
    model_jina: str = "jinaai/jina-embeddings-v2-base-de"
    # optional: e5 für Kompatibilität
    model_e5: str = "intfloat/multilingual-e5-large-instruct"
    device: str = "cpu"  # "cuda" falls verfügbar

    def __post_init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _get(self, name: str) -> Any:
        if name not in self._cache:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as ex:  # pragma: no cover
                raise RuntimeError("sentence-transformers not installed") from ex
            self._cache[name] = SentenceTransformer(name, device=self.device)
        return self._cache[name]

    def embed_texts(self, texts: Sequence[str], kind: EmbeddingKind = "mxbai") -> list[list[float]]:
        if kind == "e5":
            model = self._get(self.model_e5)
            inputs = [_prefix_e5_passage(t) for t in texts]
            return [
                v.tolist()
                for v in model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
            ]
        elif kind == "jina":
            model = self._get(self.model_jina)
            return [
                v.tolist()
                for v in model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
            ]
        else:  # "mxbai"
            model = self._get(self.model_mxbai)
            return [
                v.tolist()
                for v in model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
            ]

    def embed_query(self, text: str, kind: EmbeddingKind = "mxbai") -> list[float]:
        if kind == "e5":
            model = self._get(self.model_e5)
            q = _prefix_e5_query(text)
            return model.encode(q, normalize_embeddings=True).tolist()
        elif kind == "jina":
            model = self._get(self.model_jina)
            return model.encode(text, normalize_embeddings=True).tolist()
        else:
            model = self._get(self.model_mxbai)
            return model.encode(text, normalize_embeddings=True).tolist()
