## Copilot Instructions – BU Superagent (Strict Architecture Mode)

**Purpose:** Maintain the Clean Architecture RAG skeleton (German BU domain). Domain stays pure (no I/O, deterministic); Application orchestrates; Infrastructure adapts tech; Interface/config only wire things together.

### Architecture snapshot
- `domain/`: Entities + value objects; services like `services/chunking.py` perform semantic chunking (section → paragraph → sentence → overlap merge) with tunable `ChunkingParams`. `services/relevance_scoring.py::cosine_similarity` is intentionally `NotImplemented`.
- `application/`: Use cases and ports. `use_cases/ingest_documents.py::IngestDocuments.execute` is the active ingestion orchestrator (returns chunk count); `QueryKnowledgeBase.execute` remains a placeholder that must keep raising `NotImplemented`.
- `infrastructure/`: Each adapter implements a port and lazily imports its dependency (see tests); no domain/app code here.
- `interface/`: CLI/HTTP shells only. `cli/ingest.py` builds the ingest use case; `cli/main.py` deliberately raises `NotImplementedError` if no `--question` is provided (tests rely on it).
- `config/`: `composition.py` chooses adapters; `settings.py::AppSettings` centralises env/config knobs.

### Core contracts & behaviors
- Vector store port (`application/ports/vector_store_port.py`) requires `ensure_collection` → `upsert` → `search`; tests in `tests/infrastructure/test_vectorstore_adapters_fake.py` assert this sequence, so always call `ensure_collection` before persisting.
- Embedding port (`application/ports/embedding_port.py`) supports kinds `"mxbai"|"jina"|"e5"`. The Sentence Transformers adapter prefixes E5 queries/passages (`_prefix_e5_query/_passage`) and caches models per name.
- Document loading uses `DocumentLoaderPort.load` returning `DocumentPayload`; `PlainTextLoaderAdapter` trims UTF-8 files while `PDFTextExtractorAdapter` wraps `pypdf`, both raising `RuntimeError` on failure.
- `IngestDocuments.execute` pipeline: load → chunk via domain service → embed texts (outputs L2-normalized vectors) → `ensure_collection` + `upsert` with payloads carrying `doc_id`, `chunk_index`, `section_title`, and source metadata.

### Infrastructure adapters & wiring
- `_build_vector_store` in `config/composition.py` honours `VECTOR_BACKEND` (`qdrant` | `chroma` | `faiss`) and falls back to FAISS for testability; default env setting is `chroma` but unknown values pick FAISS.
- Qdrant adapter recreates the collection with cosine distance each time; Chroma adapter persists under `var/chroma/...`; FAISS adapter holds an in-memory `IndexFlatIP`. All import modules via `import_module` so fake modules in tests can replace them.
- Embedding builder in composition pins `device="cuda"`; switch to `cpu` locally if no GPU is available before instantiating `SentenceTransformersEmbeddingAdapter`.
- Simple overlap reranker (`infrastructure/reranking/cross_encoder.py`) implements `RerankerPort` but is not wired yet—keep it pure and self-contained when extending.

### Developer workflow (Windows cmd examples)
- Create venv + install dev deps: `python -m venv .venv` → `.venv\Scripts\pip.exe install -e ".[dev]"`.
- Lint/format: run `lint.cmd` (Ruff configured for lint + style) or `.venv\Scripts\python.exe -m ruff .` as needed.
- Types & layering: `.venv\Scripts\python.exe -m mypy` and `.venv\Scripts\lint-imports.exe --config importlinter.ini` keep import boundaries intact.
- Tests: `.venv\Scripts\python.exe -m pytest -q`. Active suites cover domain chunking (`tests/domain/test_chunking.py`), ingestion flow (`tests/application/test_ingest_use_case_v2.py`), CLI placeholders, and infra adapter fakes. Leave NotImplemented sentinels in place where asserted.

### Guardrails & examples
- **Do not remove/rename files** without checking Port relations. If a file is an unused duplicate adapter, remove it. Utilities belong under `infrastructure/<tech>/utils.py` and are only imported by adapters.
- Never import infrastructure from domain/application; respect inward-only dependencies enforced by importlinter.
- Follow the chunking patterns and tests in `tests/domain/test_chunking.py` when modifying `chunk_text_semantic`; stay deterministic and keep overlap/tail semantics.
- For new adapters, copy the lazy-import/error-wrapping style and add contract tests under `tests/infrastructure/` similar to existing fakes.
- When extending interface/CLI, keep it thin: parse args, call composition builders, and hand off to use cases.
