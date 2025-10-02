from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...domain.services.chunking import ChunkingParams, chunk_text_semantic
from ..dto.ingest_dto import IngestDocumentDTO, IngestDocumentRequest
from ..ports.document_loader_port import DocumentLoaderPort
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
        metas: list[Mapping[str, object]] = []
        for d in docs:
            parts = simple_split(d.content)
            for idx, part in enumerate(parts):
                chunks.append(part)
                metas.append({"doc_id": d.id, "title": d.title, "chunk_index": idx})

        # Placeholder stage: store raw chunks and metadata; embeddings used only at query time
        self.vector_store.add(chunks, metas)


@dataclass
class IngestDocuments:
    loader: DocumentLoaderPort
    embedding: EmbeddingPort
    vector_store: VectorStorePort

    def execute(self, req: IngestDocumentRequest) -> int:
        # 1) Quelle laden
        payload = self.loader.load(req.path)
        text = payload.text

        # 2) Chunken (pure Domain)
        params = ChunkingParams(
            target_chars=req.target_chars,
            overlap_chars=req.overlap_chars,
            max_overhang=req.max_overhang,
            merge_threshold=req.merge_threshold,
            inject_section_titles=req.inject_section_titles,
        )
        chunks = chunk_text_semantic(text, params)

        if not chunks:
            return 0

        # 3) Embeddings
        texts = [c.text for c in chunks]
        vectors = self.embedding.embed_texts(texts, kind=req.embedding_kind)  # L2-normalized

        # 4) Persistenz in VectorStore (mit Metadaten)
        ids = [f"{req.doc_id}::chunk::{i}" for i in range(len(chunks))]
        payloads = []
        for i, c in enumerate(chunks):
            payloads.append({
                "doc_id": req.doc_id,
                "chunk_index": i,
                "section_title": c.section_title,
                "text": c.text,  # für Auditing & ggf. Reranker
                "source_path": payload.source_path,
                "title": payload.title,
            })

        self.vector_store.ensure_collection(req.collection, dim=len(vectors[0]))
        self.vector_store.upsert(ids=ids, vectors=vectors, payloads=payloads)

        return len(chunks)
