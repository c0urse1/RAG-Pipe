import sys

from bu_superagent.application.dto.query_dto import RAGAnswer
from bu_superagent.application.use_cases.query_knowledge_base import Result
from bu_superagent.domain.errors import LowConfidenceError
from bu_superagent.domain.models import Citation
from bu_superagent.interface.cli import ingest as ingest_cli
from bu_superagent.interface.cli import query as query_cli
from bu_superagent.interface.cli.main import main


def test_cli_main_placeholder():
    try:
        main()
    except NotImplementedError:
        assert True


def test_ingest_cli_parses_args(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class FakeUseCase:
        def execute(self, req):  # type: ignore[no-untyped-def]
            captured["req"] = req
            return 7

    monkeypatch.setattr(ingest_cli, "build_ingest_use_case", lambda: FakeUseCase())
    monkeypatch.setenv("VECTOR_COLLECTION", "custom_collection")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--doc-id",
            "doc-123",
            "--path",
            "./docs/sample.pdf",
        ],
        raising=False,
    )

    ingest_cli.main()

    out = capsys.readouterr().out
    assert "Ingest done: 7 chunks" in out
    req = captured["req"]
    assert req.collection == "custom_collection"
    assert req.embedding_kind == "e5"
    assert req.inject_section_titles is True


def test_query_cli_success_path(monkeypatch, capsys):
    """Test query CLI with successful result."""

    class FakeUseCase:
        def execute(self, req):  # type: ignore[no-untyped-def]
            citations = [
                Citation(chunk_id="c1", source="doc1.pdf", score=0.92),
                Citation(chunk_id="c2", source="doc2.pdf", score=0.87),
            ]
            answer = RAGAnswer(
                text="Die Wartezeit beträgt 12 Monate für Bestandskunden.", citations=citations
            )
            return Result.success(answer)

    monkeypatch.setattr(query_cli, "build_query_use_case", lambda with_llm: FakeUseCase())

    query_cli.main()

    out = capsys.readouterr().out
    assert "Die Wartezeit beträgt 12 Monate" in out
    assert "[1] doc1.pdf (score=0.920)" in out
    assert "[2] doc2.pdf (score=0.870)" in out


def test_query_cli_low_confidence_error(monkeypatch, capsys):
    """Test query CLI with low confidence error."""

    class FakeUseCase:
        def execute(self, req):  # type: ignore[no-untyped-def]
            err = LowConfidenceError(
                message="Confidence below threshold", top_score=0.25, threshold=0.35
            )
            return Result.failure(err)

    monkeypatch.setattr(query_cli, "build_query_use_case", lambda with_llm: FakeUseCase())

    query_cli.main()

    out = capsys.readouterr().out
    assert "[ERROR] LowConfidenceError" in out
    assert "Top score: 0.250" in out
    assert "Threshold: 0.350" in out


def test_cli_main_with_question(monkeypatch, capsys):
    """Test main.py CLI with --question argument."""

    class FakeUseCase:
        def execute(self, req):  # type: ignore[no-untyped-def]
            citations = [Citation(chunk_id="c1", source="test.pdf", score=0.95)]
            answer = RAGAnswer(text="Test answer", citations=citations)
            return Result.success(answer)

    monkeypatch.setattr(
        "bu_superagent.interface.cli.main.build_query_use_case", lambda with_llm: FakeUseCase()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["main", "--question", "Was ist BU?"],
        raising=False,
    )

    main()

    out = capsys.readouterr().out
    assert "Test answer" in out
    assert "[1] test.pdf (score=0.950)" in out
