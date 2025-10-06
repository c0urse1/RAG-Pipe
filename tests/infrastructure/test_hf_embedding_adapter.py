import math
import sys
from importlib import reload

import pytest

from bu_superagent.domain.errors import EmbeddingError


# Fake numpy for testing without dependencies
class _FakeArray:
    """Minimal fake for numpy array operations."""

    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]

    def __iter__(self):
        return iter(self.data)

    def tolist(self):
        return self.data


class _FakeNumpy:
    """Minimal fake for numpy module."""

    float32 = float

    @staticmethod
    def ones(size, dtype=float):
        return _FakeArray([1.0] * size)

    @staticmethod
    def stack(arrays):
        return _FakeArray([arr.data for arr in arrays])

    class linalg:
        @staticmethod
        def norm(vec):
            if isinstance(vec, _FakeArray):
                return math.sqrt(sum(x * x for x in vec.data))
            return math.sqrt(sum(x * x for x in vec))


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
        np = _FakeNumpy()

        def _vec():
            vec = np.ones(1024, dtype=np.float32)
            if normalize_embeddings:
                norm_val = max(np.linalg.norm(vec), 1e-8)
                vec = _FakeArray([x / norm_val for x in vec.data])
            if convert_to_numpy:
                return vec
            return vec.tolist()

        if isinstance(inputs, str):
            return _vec()
        return np.stack([_vec() for _ in inputs])


@pytest.fixture(autouse=True)
def _setup_fake_modules(monkeypatch):
    """Setup fake numpy and sentence_transformers for all tests."""
    # Save original modules
    original_modules = sys.modules.copy()

    # Inject fake numpy
    monkeypatch.setitem(sys.modules, "numpy", _FakeNumpy())

    # Create fake sentence_transformers with our fake ST class
    fake_st_module = type(sys)("sentence_transformers")
    fake_st_module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    # Reload the module under test to use fake dependencies
    import bu_superagent.infrastructure.embeddings.hf_sentence_transformers as hf_mod

    reload(hf_mod)

    yield hf_mod

    # Cleanup: restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


def test_embed_query_and_texts_are_normalized(_setup_fake_modules):
    hf_mod = _setup_fake_modules
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


def test_unsupported_kind_raises(_setup_fake_modules):
    hf_mod = _setup_fake_modules
    adapter = hf_mod.HFEmbeddingAdapter(device="cpu")
    with pytest.raises(EmbeddingError):
        adapter.embed_query("x", kind="mxbai")
    with pytest.raises(EmbeddingError):
        adapter.embed_texts(["y"], kind="mxbai")
