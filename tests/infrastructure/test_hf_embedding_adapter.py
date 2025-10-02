import math

import numpy as np
import pytest

from bu_superagent.domain.errors import EmbeddingError
from bu_superagent.infrastructure.embeddings import hf_sentence_transformers as hf_mod


class _FakeST:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        """Stand-in for sentence_transformers.SentenceTransformer."""

    def encode(
        self,
        inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        def _vec():
            vec = np.ones(1024, dtype=np.float32)
            if normalize_embeddings:
                vec = vec / max(np.linalg.norm(vec), 1e-8)
            if convert_to_numpy:
                return vec
            return vec.tolist()

        if isinstance(inputs, str):
            return _vec()
        return np.stack([_vec() for _ in inputs])


def test_embed_query_and_texts_are_normalized(monkeypatch):
    monkeypatch.setattr(hf_mod, "SentenceTransformer", _FakeST, raising=True)
    adapter = hf_mod.HFEmbeddingAdapter(device="cpu")

    query_vector = adapter.embed_query("Hallo Welt")
    assert len(query_vector) == 1024
    norm = math.sqrt(sum(x * x for x in query_vector))
    assert 0.999 < norm < 1.001

    text_vectors = adapter.embed_texts(["eins", "zwei"])
    assert len(text_vectors) == 2
    assert all(len(vec) == 1024 for vec in text_vectors)
    for vec in text_vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert 0.999 < norm < 1.001


def test_unsupported_kind_raises(monkeypatch):
    monkeypatch.setattr(hf_mod, "SentenceTransformer", _FakeST, raising=True)
    adapter = hf_mod.HFEmbeddingAdapter(device="cpu")
    with pytest.raises(EmbeddingError):
        adapter.embed_query("x", kind="mxbai")
    with pytest.raises(EmbeddingError):
        adapter.embed_texts(["y"], kind="mxbai")
