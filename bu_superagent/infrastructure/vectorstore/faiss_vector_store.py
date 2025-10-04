from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

from bu_superagent.application.ports.vector_store_port import RetrievedChunk, VectorStorePort
from bu_superagent.domain.errors import VectorStoreError


@dataclass
class FaissVectorStoreAdapter(VectorStorePort):
    index: Any | None = field(default=None, init=False)
    payloads: dict[int, dict[str, object]] = field(default_factory=dict, init=False)
    id_map: dict[int, str] = field(default_factory=dict, init=False)
    next_idx: int = 0
    collection: str = "kb_chunks_de_1024d"

    def _require_modules(self) -> tuple[Any, Any]:  # pragma: no cover
        try:
            faiss = import_module("faiss")
            np = import_module("numpy")
        except Exception as ex:  # pragma: no cover
            raise VectorStoreError(
                "faiss-cpu and numpy are required for FaissVectorStoreAdapter"
            ) from ex
        return cast(Any, np), cast(Any, faiss)

    def ensure_collection(self, name: str, dim: int) -> None:
        self.collection = name
        try:
            np, faiss = self._require_modules()
            self.index = faiss.IndexFlatIP(dim)  # inner product ≈ cosine for L2-normalized vectors
        except VectorStoreError:
            raise  # Re-raise domain errors
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Failed to create FAISS index: {ex}") from ex

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, object]],
    ) -> None:
        if self.index is None:
            raise VectorStoreError("Collection not initialized. Call ensure_collection first.")
        try:
            np, _faiss = self._require_modules()
            arr = np.array(vectors, dtype=np.float32)
            n = arr.shape[0]
            base = self.next_idx
            self.index.add(arr)
            for i in range(n):
                self.id_map[base + i] = ids[i]
                self.payloads[base + i] = payloads[i]
            self.next_idx += n
        except VectorStoreError:
            raise  # Re-raise domain errors
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Upsert failed: {ex}") from ex

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None:
            raise VectorStoreError("Collection not initialized. Call ensure_collection first.")
        try:
            np, _faiss = self._require_modules()
            q = np.array([query_vector], dtype=np.float32)
            scores, idxs = self.index.search(q, top_k)
            out: list[RetrievedChunk] = []
            for score, idx in zip(scores[0], idxs[0], strict=False):
                if int(idx) == -1:
                    continue
                rid = self.id_map[int(idx)]
                meta = self.payloads[int(idx)]
                out.append(
                    RetrievedChunk(
                        id=rid,
                        text=str(meta.get("text", "")),
                        metadata=meta,
                        vector=None,  # FAISS doesn't store vectors separately
                        score=float(score),
                    )
                )
            return out
        except VectorStoreError:
            raise  # Re-raise domain errors
        except Exception as ex:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {ex}") from ex
