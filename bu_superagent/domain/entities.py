from dataclasses import dataclass

from .value_objects import ChunkId, DocumentId


@dataclass(frozen=True)
class DocumentChunk:
    id: ChunkId
    document_id: DocumentId
    text: str
    section: str | None
    page: int | None
