from bu_superagent.config.composition import build_application
from bu_superagent.application.dto.ingest_dto import IngestDocumentDTO


def test_ingest_single_document():
    app = build_application()
    app["ingest"].execute([IngestDocumentDTO(id="doc1", title="Doc", content="hello world")])
    # subsequent query should return something
    # No assertion on query here; ensure ingest didn't raise
    assert True
