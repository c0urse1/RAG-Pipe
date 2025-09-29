import pytest
from bu_superagent.config.composition import build_application
from bu_superagent.application.dto.ingest_dto import IngestDocumentDTO
from bu_superagent.application.dto.query_dto import QueryRequest


def test_query_placeholder_raises():
    app = build_application()
    app["ingest"].execute([IngestDocumentDTO(id="d1", title="T", content="alpha beta gamma")])
    with pytest.raises(NotImplementedError):
        app["query"].execute(QueryRequest(question="beta", top_k=2))
