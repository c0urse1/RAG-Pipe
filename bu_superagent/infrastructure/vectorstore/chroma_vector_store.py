from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort
from bu_superagent.domain.errors import VectorStoreError

try:  # pragma: no cover - exercised via tests with monkeypatch
    import chromadb
except Exception:  # noqa: BLE001
    chromadb = None


@dataclass
class ChromaVectorStoreAdapter(VectorStorePort):
    persist_dir: str = "var/chroma/e5_1024d"
    collection: str = "kb_chunks_de_1024d"
    store_text: bool = True  # regulatorische Option: Text explizit deaktivierbar
    _client: Any | None = None
    _coll: Any | None = None

    def __post_init__(self) -> None:
        if chromadb is None:
            raise VectorStoreError("chromadb not installed.")
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Failed to init Chroma at '{self.persist_dir}': {ex}") from ex

    def ensure_collection(self, name: str, dim: int) -> None:
        del dim  # Chroma collections are dimensionless; embeddings carry their own size
        if self._client is None:
            raise VectorStoreError("Chroma client not initialized.")
        self.collection = name
        try:
            self._coll = self._client.get_or_create_collection(
                name=self.collection,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Failed to ensure collection '{name}': {ex}") from ex

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        if self._coll is None:
            raise VectorStoreError("Collection not initialized. Call ensure_collection first.")
        try:
            documents = (
                [p.get("text", "") for p in payloads] if self.store_text else [""] * len(payloads)
            )
            self._coll.add(
                ids=list(ids),
                embeddings=[list(vec) for vec in vectors],
                metadatas=list(payloads),
                documents=documents,
            )
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Upsert failed: {ex}") from ex

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        if self._coll is None:
            raise VectorStoreError("Collection not initialized. Call ensure_collection first.")
        try:
            result = cast(
                dict[str, list[list[Any]]],
                self._coll.query(
                    query_embeddings=[list(query_vector)],
                    n_results=top_k,
                ),
            )
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {ex}") from ex

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            score = 1.0 - distance  # Chroma liefert Distanz (kleiner = besser)
            text = documents[idx] if idx < len(documents) and documents[idx] is not None else ""
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] is not None else {}
            chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(text),
                    score=score,
                    metadata=metadata,
                )
            )
        return chunks

    def persist(self) -> None:
        if self._client is None:
            raise VectorStoreError("Chroma client not initialized.")
        try:
            self._client.persist()
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Persist failed: {ex}") from ex
