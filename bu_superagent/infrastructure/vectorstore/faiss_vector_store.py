from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort


@dataclass
class FaissVectorStoreAdapter(VectorStorePort):
    index: object | None = field(default=None, init=False)
    payloads: Dict[int, dict] = field(default_factory=dict, init=False)
    id_map: Dict[int, str] = field(default_factory=dict, init=False)
    next_idx: int = 0
    collection: str = "kb_chunks_de_1024d"

    def _require_modules(self):  # pragma: no cover
        try:
            import numpy as np  # type: ignore
            import faiss  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("faiss-cpu and numpy are required for FaissVectorStoreAdapter") from ex
        return np, faiss

    def ensure_collection(self, name: str, dim: int) -> None:
        self.collection = name
        np, faiss = self._require_modules()
        self.index = faiss.IndexFlatIP(dim)  # inner product ≈ cosine for L2-normalized vectors

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]) -> None:
        assert self.index is not None, "Collection not initialized. Call ensure_collection first."
        np, _faiss = self._require_modules()
        arr = np.array(vectors, dtype=np.float32)
        n = arr.shape[0]
        base = self.next_idx
        # type: ignore[attr-defined]
        self.index.add(arr)  # pyright: ignore[reportAttributeAccessIssue]
        for i in range(n):
            self.id_map[base + i] = ids[i]
            self.payloads[base + i] = payloads[i]
        self.next_idx += n

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[RetrievedChunk]:
        assert self.index is not None, "Collection not initialized. Call ensure_collection first."
        np, _faiss = self._require_modules()
        q = np.array([query_vector], dtype=np.float32)
        # type: ignore[attr-defined]
        scores, idxs = self.index.search(q, top_k)  # pyright: ignore[reportAttributeAccessIssue]
        out: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if int(idx) == -1:
                continue
            rid = self.id_map[int(idx)]
            meta = self.payloads[int(idx)]
            out.append(RetrievedChunk(id=rid, text=meta.get("text", ""), score=float(score), metadata=meta))
        return out
