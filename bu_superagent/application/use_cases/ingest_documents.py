from __future__ import annotations

from typing import Sequence

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
        chunks: list[str] = []
        metas: list[dict] = []
        for d in docs:
            parts = simple_split(d.content)
            for idx, part in enumerate(parts):
                chunks.append(part)
                metas.append({"doc_id": d.id, "title": d.title, "chunk_index": idx})

        # Placeholder stage: store raw chunks and metadata; embeddings used only at query time
        self.vector_store.add(chunks, metas)
