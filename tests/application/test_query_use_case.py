from bu_superagent.config.composition import build_application
from bu_superagent.application.dto.ingest_dto import IngestDocumentDTO
from bu_superagent.application.dto.query_dto import QueryDTO


def test_query_returns_results():
    app = build_application()
    app["ingest"].execute([IngestDocumentDTO(id="d1", title="T", content="alpha beta gamma")])
    results = app["query"].execute(QueryDTO(text="beta", top_k=2))
    assert len(results) >= 1
    assert "score" in results[0]
