from __future__ import annotations

from dataclasses import dataclass

from bu_superagent.application.ports.embedding_port import EmbeddingKind


@dataclass(slots=True, frozen=True)
class IngestDocumentDTO:
    id: str
    title: str
    content: str


@dataclass(frozen=True)
class IngestDocumentRequest:
    doc_id: str
    path: str  # Pfad zur Quelle (PDF/TXT)
    collection: str = "kb_chunks_de_1024d"
    target_chars: int = 1000
    overlap_chars: int = 150
    max_overhang: int = 200
    merge_threshold: int = 500
    inject_section_titles: bool = True
    embedding_kind: EmbeddingKind = "mxbai"  # oder "e5"/"jina"
