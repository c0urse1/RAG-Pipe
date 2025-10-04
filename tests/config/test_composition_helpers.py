"""Tests for composition root (dependency injection/wiring).

The composition root is the ONLY place that:
1. Reads environment variables (via AppSettings)
2. Instantiates concrete infrastructure adapters
3. Wires dependencies into use cases

This ensures application/domain layers remain pure and testable.
"""

import os

from bu_superagent.application.ports.embedding_port import EmbeddingPort
from bu_superagent.application.ports.llm_port import LLMPort
from bu_superagent.application.ports.vector_store_port import VectorStorePort
from bu_superagent.application.use_cases.ingest_documents import IngestDocuments
from bu_superagent.application.use_cases.query_knowledge_base import QueryKnowledgeBase
from bu_superagent.config.composition import (
    build_embedding,
    build_embedding_adapter,
    build_ingest_use_case,
    build_llm,
    build_llm_adapter,
    build_query_use_case,
    build_vector_store,
)
from bu_superagent.config.settings import AppSettings


class TestCompositionRoot:
    """Test that composition root properly wires dependencies."""

    def test_build_embedding_returns_port_implementation(self) -> None:
        """build_embedding() should return an EmbeddingPort implementation."""
        os.environ["VECTOR_BACKEND"] = "faiss"  # Use FAISS for tests (no deps)
        settings = AppSettings()
        adapter = build_embedding(settings)

        assert adapter is not None
        assert isinstance(adapter, EmbeddingPort)

    def test_build_vector_store_returns_port_implementation(self) -> None:
        """build_vector_store() should return a VectorStorePort implementation."""
        os.environ["VECTOR_BACKEND"] = "faiss"  # Use FAISS for tests (no deps)
        settings = AppSettings()
        adapter = build_vector_store(settings)

        assert adapter is not None
        assert isinstance(adapter, VectorStorePort)

    def test_build_llm_returns_port_implementation(self) -> None:
        """build_llm() should return an LLMPort implementation."""
        settings = AppSettings()
        adapter = build_llm(settings)

        assert adapter is not None
        assert isinstance(adapter, LLMPort)

    def test_build_query_use_case_wires_all_dependencies(self) -> None:
        """build_query_use_case() should wire embedding, vector_store, and llm."""
        os.environ["VECTOR_BACKEND"] = "faiss"  # Use FAISS for tests (no deps)
        use_case = build_query_use_case(with_llm=True)

        assert isinstance(use_case, QueryKnowledgeBase)
        assert use_case.embedding is not None
        assert use_case.vector_store is not None
        assert use_case.llm is not None

    def test_build_query_use_case_without_llm(self) -> None:
        """build_query_use_case(with_llm=False) should omit LLM (extractive mode)."""
        os.environ["VECTOR_BACKEND"] = "faiss"  # Use FAISS for tests (no deps)
        use_case = build_query_use_case(with_llm=False)

        assert isinstance(use_case, QueryKnowledgeBase)
        assert use_case.embedding is not None
        assert use_case.vector_store is not None
        assert use_case.llm is None  # Extractive fallback

    def test_build_ingest_use_case_wires_dependencies(self) -> None:
        """build_ingest_use_case() should wire loader, embedding, and vector_store."""
        os.environ["VECTOR_BACKEND"] = "faiss"  # Use FAISS for tests (no deps)
        use_case = build_ingest_use_case()

        assert isinstance(use_case, IngestDocuments)
        assert use_case.loader is not None
        assert use_case.embedding is not None
        assert use_case.vector_store is not None


class TestBackwardCompatibility:
    """Test backward-compatible aliases for legacy tests."""

    def test_build_embedding_adapter_is_alias(self) -> None:
        """build_embedding_adapter() should be an alias for build_embedding()."""
        settings = AppSettings()
        emb1 = build_embedding(settings)
        emb2 = build_embedding_adapter(settings)

        # Both should return the same type
        assert type(emb1).__name__ == type(emb2).__name__

    def test_build_llm_adapter_is_alias(self) -> None:
        """build_llm_adapter() should be an alias for build_llm()."""
        settings = AppSettings()
        llm1 = build_llm(settings)
        llm2 = build_llm_adapter(settings)

        # Both should return the same type
        assert type(llm1).__name__ == type(llm2).__name__


class TestSettingsIsolation:
    """Test that settings properly isolate environment variable reading."""

    def test_settings_uses_defaults_without_env(self, monkeypatch) -> None:
        """AppSettings should use defaults when env vars are not set."""
        # Clear all relevant env vars
        for key in [
            "EMBEDDING_MODEL",
            "EMBEDDING_DEVICE",
            "VECTOR_BACKEND",
            "LLM_BASE_URL",
            "LLM_MODEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        settings = AppSettings()

        # Check defaults
        assert "e5" in settings.embedding_model.lower()
        assert settings.embedding_device == "cpu"
        assert settings.vector_backend == "chroma"
        assert "localhost" in settings.llm_base_url
        assert "llama" in settings.llm_model.lower()

    def test_settings_respects_env_overrides(self, monkeypatch) -> None:
        """AppSettings should respect environment variable overrides."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("LLM_MODEL", "custom-llm")

        settings = AppSettings()

        assert settings.embedding_model == "custom-model"
        assert settings.embedding_device == "cuda"
        assert settings.vector_backend == "qdrant"
        assert settings.llm_model == "custom-llm"


class TestNoSideEffects:
    """Test that builders don't trigger heavy operations (network, model loading)."""

    def test_builders_return_instances_without_side_effects(self) -> None:
        """Builders should instantiate adapters without triggering I/O."""
        s = AppSettings()
        emb = build_embedding_adapter(s)
        llm = build_llm_adapter(s)

        # Nur Existenz prüfen; keine Netzwerk- oder Modell-Ladevorgänge
        assert emb is not None
        assert llm is not None
