from __future__ import annotations

from dataclasses import dataclass

from bu_superagent.application.ports.document_loader_port import DocumentLoaderPort, DocumentPayload


@dataclass
class PlainTextLoaderAdapter(DocumentLoaderPort):
    def load(self, path: str) -> DocumentPayload:  # type: ignore[override]
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            return DocumentPayload(text=text, title=None, source_path=path)
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(f"TXT load failed: {ex}") from ex


@dataclass
class PDFTextExtractorAdapter(DocumentLoaderPort):
    def load(self, path: str) -> DocumentPayload:  # type: ignore[override]
        try:
            from pypdf import PdfReader  # lazy import to avoid hard dependency in tests
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("pypdf is not installed") from ex

        try:
            reader = PdfReader(path)
            pages: list[str] = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n\n".join(pages).strip()
            title = reader.metadata.title if getattr(reader, "metadata", None) else None
            return DocumentPayload(text=text, title=title, source_path=path)
        except Exception as ex:  # noqa: BLE001
            # Optional: Fallback über 'unstructured' o. OCR – hier bewusst weggelassen
            raise RuntimeError(f"PDF parse failed: {ex}") from ex
