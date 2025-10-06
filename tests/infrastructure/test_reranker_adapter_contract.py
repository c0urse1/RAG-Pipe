"""Contract tests for reranker adapters using fake modules.

Following the fake module pattern from test_vectorstore_adapters_fake.py.
"""

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _cleanup_sys_modules():
    """Save and restore sys.modules state to prevent test pollution."""
    # Save original modules
    saved_modules = {
        "sentence_transformers": sys.modules.get("sentence_transformers"),
        "numpy": sys.modules.get("numpy"),
        "torch": sys.modules.get("torch"),
    }

    yield  # Run test

    # Restore original modules (or remove if they weren't there)
    for name, original in saved_modules.items():
        if original is not None:
            sys.modules[name] = original
        else:
            sys.modules.pop(name, None)


def test_cross_encoder_adapter_with_fake_module(monkeypatch):
    """Test CrossEncoderAdapter with fake sentence_transformers module."""

    # Create fake CrossEncoder that returns deterministic scores
    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            self.model_name = model_name
            self.device = device

        def predict(self, pairs, convert_to_numpy=True, show_progress_bar=False):
            """Return fake scores based on pair count."""

            class FakeArray:
                def __init__(self, values):
                    self.values = values

                def tolist(self):
                    return self.values

            # Return scores: first pair gets 0.9, second 0.5, third 0.1
            scores = [0.9 - i * 0.4 for i in range(len(pairs))]
            return FakeArray(scores)

    class FakeSentenceTransformers:
        CrossEncoder = FakeCrossEncoder

    # Fake numpy for sigmoid computation
    class FakeNumpy:
        # Fake bool_ for pytest.approx
        bool_ = bool

        @staticmethod
        def exp(x):
            """Fake exp for testing sigmoid."""
            import math

            if hasattr(x, "__iter__") and not isinstance(x, str):
                return [math.exp(val) for val in x]
            return math.exp(x)

        @staticmethod
        def isscalar(obj):
            """Check if obj is scalar (needed by pytest.approx)."""
            return isinstance(obj, int | float)

    # Inject fake modules
    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())
    monkeypatch.setitem(sys.modules, "torch", type("FakeTorch", (), {})())  # Mock torch

    # Import (or reload) adapter module with fakes in place
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    # Test: Basic scoring with fake module
    adapter = CrossEncoderAdapter(
        model_name="BAAI/bge-reranker-v2-m3", device="cpu", apply_sigmoid=False
    )

    query = "What is the capital of Germany?"
    candidates = ["Berlin is the capital.", "Munich is a city.", "Hamburg is a port."]

    scores = adapter.score(query, candidates)

    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # First candidate should have highest score (0.9)
    assert scores[0] == 0.9
    assert scores[1] == 0.5
    assert scores[2] == pytest.approx(0.1, abs=0.01)


def test_cross_encoder_adapter_model_caching(monkeypatch):
    """Test that models are cached per (model_name, device) key."""

    load_count = {"count": 0}

    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            load_count["count"] += 1
            self.model_name = model_name

        def predict(self, pairs, **kwargs):
            class FakeArray:
                values = [0.5] * len(pairs)

                def tolist(self):
                    return self.values

            return FakeArray()

    class FakeSentenceTransformers:
        CrossEncoder = FakeCrossEncoder

    # Mock numpy (for potential sigmoid even with apply_sigmoid=False by default)
    class FakeNumpy:
        bool_ = bool

        @staticmethod
        def exp(x):
            import math

            if hasattr(x, "__iter__") and not isinstance(x, str):
                return [math.exp(val) for val in x]
            return math.exp(x)

        @staticmethod
        def isscalar(obj):
            return isinstance(obj, int | float)

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())
    monkeypatch.setitem(sys.modules, "torch", type("FakeTorch", (), {})())  # Mock torch
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())  # Mock numpy

    # Reload adapter module with fakes
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    # Clear cache for test isolation
    CrossEncoderAdapter._model_cache.clear()

    # First instantiation should load model
    adapter1 = CrossEncoderAdapter(model_name="model-a", device="cpu", apply_sigmoid=False)
    adapter1.score("q", ["p1"])
    assert load_count["count"] == 1

    # Same model+device should reuse cached model
    adapter2 = CrossEncoderAdapter(model_name="model-a", device="cpu", apply_sigmoid=False)
    adapter2.score("q", ["p2"])
    assert load_count["count"] == 1  # No additional load

    # Different device should load new model
    adapter3 = CrossEncoderAdapter(model_name="model-a", device="cuda", apply_sigmoid=False)
    adapter3.score("q", ["p3"])
    assert load_count["count"] == 2

    # Different model name should load new model (defaults to apply_sigmoid=True, needs numpy)
    adapter4 = CrossEncoderAdapter(model_name="model-b", device="cpu", apply_sigmoid=False)
    adapter4.score("q", ["p4"])
    assert load_count["count"] == 3


def test_cross_encoder_adapter_empty_candidates(monkeypatch):
    """Test that empty candidates list returns empty scores list."""

    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            pass

        def predict(self, pairs, **kwargs):
            msg = "predict should not be called for empty candidates"
            raise AssertionError(msg)

    class FakeSentenceTransformers:
        CrossEncoder = FakeCrossEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())

    # Reload adapter module
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    adapter = CrossEncoderAdapter()
    scores = adapter.score("query", [])
    assert scores == []


def test_cross_encoder_adapter_error_handling(monkeypatch):
    """Test that adapter wraps prediction errors in RuntimeError."""

    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            pass

        def predict(self, pairs, **kwargs):
            raise ValueError("Fake prediction error")

    class FakeSentenceTransformers:
        CrossEncoder = FakeCrossEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())

    # Reload adapter module
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    adapter = CrossEncoderAdapter(apply_sigmoid=False)
    with pytest.raises(RuntimeError, match="Cross-encoder scoring failed"):
        adapter.score("query", ["p1", "p2", "p3"])


def test_cross_encoder_adapter_model_load_failure(monkeypatch):
    """Test that adapter handles model load failures gracefully."""

    class FakeSentenceTransformers:
        @staticmethod
        def CrossEncoder(model_name, device="cpu"):
            raise OSError("Fake model load error")

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())

    # Reload adapter module
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    adapter = CrossEncoderAdapter(apply_sigmoid=False)
    with pytest.raises(RuntimeError, match="Failed to load cross-encoder model"):
        adapter.score("query", ["p1"])


def test_cross_encoder_adapter_sigmoid_application(monkeypatch):
    """Test sigmoid is applied when apply_sigmoid=True."""

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def __neg__(self):
            """Support unary negation for sigmoid: -scores"""
            return FakeArray([-v for v in self.values])

        def __iter__(self):
            """Make iterable for numpy operations"""
            return iter(self.values)

        def __radd__(self, other):
            """Support scalar + array: 1.0 + np.exp(-scores)"""
            if isinstance(other, int | float):
                return FakeArray([other + v for v in self.values])
            return NotImplemented

        def __rtruediv__(self, other):
            """Support scalar / array: 1.0 / (1.0 + ...)"""
            if isinstance(other, int | float):
                return FakeArray([other / v for v in self.values])
            return NotImplemented

        def tolist(self):
            return self.values

    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            pass

        def predict(self, pairs, **kwargs):
            """Return raw logits for testing sigmoid."""
            return FakeArray([0.0, 1.0, -1.0])

    class FakeNumpy:
        @staticmethod
        def exp(x):
            """Simple exp for testing - returns FakeArray for array inputs."""
            import math

            if hasattr(x, "__iter__") and not isinstance(x, str):
                # Return FakeArray for array-like inputs
                result_values = [math.exp(val) for val in x]
                return FakeArray(result_values)
            return math.exp(x)

        @staticmethod
        def isscalar(obj):
            """Check if obj is scalar."""
            return isinstance(obj, int | float)

    class FakeSentenceTransformers:
        CrossEncoder = FakeCrossEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformers())
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())
    monkeypatch.setitem(sys.modules, "torch", type("FakeTorch", (), {})())

    # Reload adapter module with fakes
    import bu_superagent.infrastructure.reranking.cross_encoder_adapter

    importlib.reload(bu_superagent.infrastructure.reranking.cross_encoder_adapter)

    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    # Test with sigmoid enabled
    adapter = CrossEncoderAdapter(apply_sigmoid=True)
    scores = adapter.score("query", ["p1", "p2", "p3"])

    # Sigmoid should transform logits: sigmoid(0)=0.5, sigmoid(1)≈0.73, sigmoid(-1)≈0.27
    assert len(scores) == 3
    assert 0.4 < scores[0] < 0.6  # sigmoid(0) ≈ 0.5
    assert 0.7 < scores[1] < 0.8  # sigmoid(1) ≈ 0.73
    assert 0.2 < scores[2] < 0.3  # sigmoid(-1) ≈ 0.27


def test_cross_encoder_adapter_lazy_import_pattern(monkeypatch):
    """Test that sentence_transformers is imported lazily (not at module load)."""
    # This test verifies imports happen at runtime, not import time

    import_count = {"count": 0}
    import builtins

    original_import = builtins.__import__

    def counting_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            import_count["count"] += 1
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting_import)

    # Import adapter module - should NOT import sentence_transformers yet
    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import (  # noqa: F401
        CrossEncoderAdapter,
    )

    assert import_count["count"] == 0, "sentence_transformers imported at module load (not lazy)"

    # Only when creating adapter should import happen
    # (We'd need a real fake module here, so skip actual instantiation)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Integration test - skip on Windows to avoid dependency issues",
)
def test_cross_encoder_adapter_integration_with_real_model():
    """Integration test with real sentence-transformers (marked as slow)."""
    from bu_superagent.infrastructure.reranking.cross_encoder_adapter import CrossEncoderAdapter

    # Use lightweight model for testing
    adapter = CrossEncoderAdapter(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", device="cpu")

    query = "What is the capital of France?"
    candidates = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is in Germany.",
    ]

    scores = adapter.score(query, candidates)

    # Basic sanity checks
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # First candidate should score higher than others
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
