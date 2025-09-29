from __future__ import annotations

from typing import Sequence

from ...domain.types import ChunkId, DocumentId
from ..dto.ingest_dto import IngestDocumentDTO
from ..ports.embedding_port import EmbeddingPort
from ..ports.vector_store_port import VectorStorePort


def simple_split(text: str, chunk_size: int = 300) -> list[str]:
    # naive splitter to keep demo deterministic
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]


class IngestDocumentsUseCase:
    def __init__(self, embedder: EmbeddingPort, vector_store: VectorStorePort) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def execute(self, docs: Sequence[IngestDocumentDTO]) -> None:
        ids: list[ChunkId] = []
        chunks: list[str] = []
        metas: list[dict] = []
        for d in docs:
            parts = simple_split(d.content)
            for idx, part in enumerate(parts):
                ids.append(ChunkId(f"{d.id}:{idx}"))
                chunks.append(part)
                metas.append({"doc_id": d.id, "title": d.title, "chunk_index": idx})

        embeddings = self.embedder.embed_texts(chunks)
        self.vector_store.upsert(ids, embeddings, metas)
