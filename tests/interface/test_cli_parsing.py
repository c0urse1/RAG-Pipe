import sys

from bu_superagent.interface.cli import ingest as ingest_cli
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
