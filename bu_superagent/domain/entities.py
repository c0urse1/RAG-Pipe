from dataclasses import dataclass
from typing import Optional
from .value_objects import ChunkId, DocumentId


@dataclass(frozen=True)
class DocumentChunk:
    id: ChunkId
    document_id: DocumentId
    text: str
    section: Optional[str]
    page: Optional[int]
