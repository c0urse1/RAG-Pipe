from dataclasses import dataclass

from bu_superagent.application.dto.ingest_dto import IngestDocumentRequest
from bu_superagent.application.ports.document_loader_port import DocumentPayload
from bu_superagent.application.use_cases.ingest_documents import IngestDocuments


@dataclass
class FakeLoader:
    text: str = "Titel\n\nDas ist ein Text. Er ist kurz."

    def load(self, path: str) -> DocumentPayload:  # type: ignore[override]
        return DocumentPayload(text=self.text, title="Titel", source_path=path)


class FakeEmbed:
    def embed_texts(self, texts, kind="mxbai"):
        return [[0.0, 1.0, 0.0] for _ in texts]


class FakeVS:
    def __init__(self):
        self.upserts = []
        self.collections = []

    def ensure_collection(self, name, dim):
        self.collections.append((name, dim))

    def upsert(self, ids, vectors, payloads):
        self.upserts.append((ids, vectors, payloads))


def test_ingest_documents_end_to_end_small():
    uc = IngestDocuments(loader=FakeLoader(), embedding=FakeEmbed(), vector_store=FakeVS())
    n = uc.execute(
        IngestDocumentRequest(doc_id="D1", path="/tmp/doc.txt", target_chars=60, overlap_chars=10)
    )
    assert n > 0
