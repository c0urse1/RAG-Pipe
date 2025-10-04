from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort
from bu_superagent.domain.errors import VectorStoreError


@dataclass
class QdrantVectorStoreAdapter(VectorStorePort):
    host: str = "localhost"
    port: int = 6333
    collection: str = "kb_chunks"
    _cli: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            client_mod = import_module("qdrant_client")
            QdrantClient = client_mod.QdrantClient
        except Exception as ex:  # pragma: no cover
            raise VectorStoreError("qdrant-client not available; install runtime deps") from ex
        self._cli = QdrantClient(host=self.host, port=self.port)

    def ensure_collection(self, name: str, dim: int) -> None:
        self.collection = name
        try:
            models_mod = import_module("qdrant_client.models")
            Distance = models_mod.Distance
            VectorParams = models_mod.VectorParams
        except Exception as ex:  # pragma: no cover
            raise VectorStoreError(
                "qdrant-client models not available; install runtime deps"
            ) from ex
        if self._cli is None:
            raise VectorStoreError("Qdrant client not initialized")
        try:
            self._cli.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Failed to recreate collection '{name}': {ex}") from ex

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None:
        try:
            models_mod = import_module("qdrant_client.models")
            PointStruct = models_mod.PointStruct
        except Exception as ex:  # pragma: no cover
            raise VectorStoreError(
                "qdrant-client models not available; install runtime deps"
            ) from ex
        if self._cli is None:
            raise VectorStoreError("Qdrant client not initialized")
        try:
            points = [
                PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(len(ids))
            ]
            self._cli.upsert(collection_name=self.collection, points=points)
        except VectorStoreError:
            raise  # Re-raise domain errors
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Upsert failed: {ex}") from ex

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        if self._cli is None:
            raise VectorStoreError("Qdrant client not initialized")
        try:
            rs: Any = self._cli.search(
                collection_name=self.collection, query_vector=query_vector, limit=top_k
            )
            return [
                RetrievedChunk(
                    id=str(p.id),
                    text=p.payload.get("text", ""),
                    metadata=p.payload,
                    vector=None,  # Qdrant doesn't return vectors by default
                    score=p.score,
                )
                for p in rs
            ]
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {ex}") from ex
