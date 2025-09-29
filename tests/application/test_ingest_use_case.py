from bu_superagent.config.composition import build_application
from bu_superagent.application.dto.ingest_dto import IngestDocumentDTO
from bu_superagent.application.dto.query_dto import QueryDTO


def test_ingest_single_document():
    app = build_application()
    app["ingest"].execute([IngestDocumentDTO(id="doc1", title="Doc", content="hello world")])
    # subsequent query should return something
    # use a small top_k just to ensure we get back a list
    res = app["query"].execute(QueryDTO(text="hello", top_k=3))
    assert isinstance(res, list)
