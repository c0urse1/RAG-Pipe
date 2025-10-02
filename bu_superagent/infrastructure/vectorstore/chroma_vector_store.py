from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort


@dataclass
class ChromaVectorStoreAdapter(VectorStorePort):
    persist_dir: str = "var/chroma/e5_1024d"
    collection: str = "kb_chunks_de_1024d"

    def __post_init__(
        self,
    ) -> None:  # pragma: no cover - lazy import path exercised in tests indirectly
        try:
            import chromadb  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("chromadb not available; install runtime deps") from ex
        self._chromadb = chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._coll = None

    def ensure_collection(self, name: str, dim: int) -> None:
        self.collection = name
        # Embeddings supplied manually; use cosine space to match normalized embeddings
        self._coll = self._client.get_or_create_collection(
            name=self.collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None:
        assert self._coll is not None, "Collection not initialized. Call ensure_collection first."
        docs = [p.get("text", "") for p in payloads]
        self._coll.add(ids=list(ids), metadatas=list(payloads), documents=docs, embeddings=list(vectors))

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        assert self._coll is not None, "Collection not initialized. Call ensure_collection first."
        res = self._coll.query(query_embeddings=[list(query_vector)], n_results=top_k)
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                RetrievedChunk(
                    id=str(ids[i]),
                    text=docs[i] or "",
                    score=float(dists[i]),
                    metadata=metas[i] or {},
                )
            )
        return out
