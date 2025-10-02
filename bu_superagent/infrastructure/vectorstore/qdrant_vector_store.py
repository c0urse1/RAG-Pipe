from collections.abc import Sequence
from dataclasses import dataclass

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort


@dataclass
class QdrantVectorStoreAdapter(VectorStorePort):
    host: str = "localhost"
    port: int = 6333
    collection: str = "kb_chunks"

    def __post_init__(self) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("qdrant-client not available; install runtime deps") from ex
        self._cli = QdrantClient(host=self.host, port=self.port)

    def ensure_collection(self, name: str, dim: int) -> None:
        self.collection = name
        try:
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("qdrant-client models not available; install runtime deps") from ex
        self._cli.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None:
        try:
            from qdrant_client.models import PointStruct  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("qdrant-client models not available; install runtime deps") from ex
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))
        ]
        self._cli.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        rs = self._cli.search(
            collection_name=self.collection, query_vector=query_vector, limit=top_k
        )
        return [
            RetrievedChunk(
                id=str(p.id), text=p.payload.get("text", ""), score=p.score, metadata=p.payload
            )
            for p in rs
        ]
