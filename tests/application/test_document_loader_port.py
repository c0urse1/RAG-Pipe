from dataclasses import dataclass

from bu_superagent.application.ports.document_loader_port import DocumentLoaderPort, DocumentPayload


@dataclass
class DummyLoader(DocumentLoaderPort):
    def load(self, path: str) -> DocumentPayload:  # type: ignore[override]
        return DocumentPayload(text=f"content:{path}", title="T", source_path=path)


def test_document_loader_port_contract():
    loader = DummyLoader()
    dp = loader.load("/tmp/x.txt")
    assert dp.text.startswith("content:")
    assert dp.title == "T"
    assert dp.source_path.endswith("x.txt")
