import os
from dataclasses import dataclass

import importlib


def test_builders_with_faiss_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    comp = importlib.import_module("bu_superagent.config.composition")
    importlib.reload(comp)

    q = comp.build_query_use_case()
    i = comp.build_ingest_use_case()

    # Adapter type should be the FAISS skeleton (no heavy deps imported here)
    assert "FaissVectorStoreAdapter" in type(q.vector_store).__name__
    assert "FaissVectorStoreAdapter" in type(i.vector_store).__name__


def test_builders_with_chroma_backend(monkeypatch):
    # Patch the Chroma adapter in composition to avoid importing chromadb
    monkeypatch.setenv("VECTOR_BACKEND", "chroma")
    comp = importlib.import_module("bu_superagent.config.composition")

    @dataclass
    class DummyChroma:
        persist_dir: str = ""
        collection: str = ""
        def ensure_collection(self, *_args, **_kwargs):
            return None
        def upsert(self, *_args, **_kwargs):
            return None
        def search(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(comp, "ChromaVectorStoreAdapter", DummyChroma, raising=True)

    q = comp.build_query_use_case()
    i = comp.build_ingest_use_case()
    assert isinstance(q.vector_store, DummyChroma)
    assert isinstance(i.vector_store, DummyChroma)


def test_builders_with_qdrant_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    comp = importlib.import_module("bu_superagent.config.composition")

    @dataclass
    class DummyQdrant:
        host: str = "localhost"
        port: int = 6333
        collection: str = "kb"
        def ensure_collection(self, *_args, **_kwargs):
            return None
        def upsert(self, *_args, **_kwargs):
            return None
        def search(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(comp, "QdrantVectorStoreAdapter", DummyQdrant, raising=True)

    q = comp.build_query_use_case()
    i = comp.build_ingest_use_case()
    assert isinstance(q.vector_store, DummyQdrant)
    assert isinstance(i.vector_store, DummyQdrant)
