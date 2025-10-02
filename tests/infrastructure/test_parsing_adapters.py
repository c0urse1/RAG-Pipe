import os
import tempfile

import pytest

from bu_superagent.infrastructure.parsing.pdf_text_extractor import (
    PDFTextExtractorAdapter,
    PlainTextLoaderAdapter,
)


def test_plain_text_loader_reads_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("Hallo Welt\n\nZweite Zeile")
        loader = PlainTextLoaderAdapter()
        payload = loader.load(p)
        assert "Hallo Welt" in payload.text
        assert payload.source_path == p


def test_pdf_loader_errors_for_missing_dependency_or_file():
    adapter = PDFTextExtractorAdapter()
    # When pypdf missing or file invalid, should raise RuntimeError (no external side effects)
    with pytest.raises(RuntimeError):
        adapter.load("/non/existent.pdf")


def test_plain_text_loader_missing_file_raises_runtimeerror():
    loader = PlainTextLoaderAdapter()
    with pytest.raises(RuntimeError):
        loader.load("/no/such/file.txt")
