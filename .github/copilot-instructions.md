## Copilot Instructions – BU Superagent (Strict Architecture Mode)

Purpose: Strict Clean Architecture RAG skeleton (German BU domain). Keep Domain pure and IO-free; orchestrate in Application; wire tech in Infrastructure/Config; expose only via Interface.

Do not remove/rename files without checking Port relations. If a file is an unused duplicate adapter, remove it. Utilities belong under `infrastructure/<tech>/utils.py` and are imported only by adapters.

Architecture (inward-only deps)
- `domain/`: Entities/Value Objects/Services (pure). Example: `domain/services/relevance_scoring.py::cosine_similarity` is a pure placeholder raising NotImplemented.
- `domain/services/chunking.py`: Pure semantic chunking (sections → paragraphs → sentences → pack with overlap/title injection → merge). No I/O or external libs; tune via `ChunkingParams`.
- `application/`: Use cases + Ports + DTOs. Examples: `application/use_cases/query_knowledge_base.py::QueryKnowledgeBase` raises NotImplemented; `ingest_documents.py::simple_split` (300 chars, naive).
- `infrastructure/`: Adapters implement Ports. Qdrant adapter in `infrastructure/vectorstore/qdrant_vector_store.py`; Chroma file exists but is a stub.
 - `infrastructure/`: Adapters implement Ports. Qdrant (`vectorstore/qdrant_vector_store.py`), Chroma (`vectorstore/chroma_vector_store.py`, persistent), and FAISS skeleton (`vectorstore/faiss_vector_store.py`).
- `interface/`: CLI/HTTP are thin. See `interface/cli/main.py` for arg parsing and use-case invocation.
- `config/`: Composition & settings only. See `config/composition.py::build_query_use_case` and `config/settings.py::AppSettings`.

Contracts and DTOs (authoritative)
- Vector store Port: `application/ports/vector_store_port.py` exposes `ensure_collection(name, dim)`, `upsert(ids, vectors, payloads)`, `search(query_vector, top_k)->list[RetrievedChunk]`.
- Embedding Port: `application/ports/embedding_port.py` exposes `embed_texts(texts, kind)` and `embed_query(text, kind)` where kind in {"mxbai","jina","e5"}.
- DTOs: `application/dto/query_dto.py::QueryRequest(question, top_k=5, use_reranker=True)` and `application/dto/ingest_dto.py::IngestDocumentDTO(id,title,content)`.
 - Document loading: `application/ports/document_loader_port.py` defines `DocumentPayload(text,title,source_path)` and `DocumentLoaderPort.load(path)->DocumentPayload`.
 - Ingestion request: `application/dto/ingest_dto.py::IngestDocumentRequest(doc_id,path,collection,target_chars,overlap_chars,max_overhang,merge_threshold,inject_section_titles,embedding_kind)`.

Adapters and defaults (current wiring)
- Embeddings via Sentence-Transformers: `infrastructure/embeddings/sentence_transformers_adapter.py` supports MXBAI (primary), Jina (fallback), and E5 (with E5-specific prefixes). Normalized outputs; 1024D by default.
- Vector store default: Qdrant (`qdrant-client`). `config/settings.py::AppSettings` provides `qdrant_host/port/collection` and `embedding_dim`. `ensure_collection` uses cosine distance.
 - Chroma: persistent on-disk via `PersistentClient(path=...)`; cosine space in collection metadata.
 - FAISS: in-memory IndexFlatIP; optional dep (`faiss-cpu`, `numpy`), basic skeleton without persistence.
- LLM (not yet used in use cases): vLLM-compatible OpenAI adapter in `infrastructure/llm/` wired in `config/composition.py`.
 - Parsing adapters: `infrastructure/parsing/pdf_text_extractor.py` contains `PlainTextLoaderAdapter` (UTF-8 .txt) and `PDFTextExtractorAdapter` (pypdf). Both implement `DocumentLoaderPort` and raise `RuntimeError` on failures.

Important caveats
- `ingest_documents.py` currently calls `vector_store.add(...)` as a placeholder; the Port defines `upsert(...)`. If you implement ingestion, use `upsert` and keep tests green (ingest tests are skipped).
- `QueryKnowledgeBase.execute` and `relevance_scoring.cosine_similarity` intentionally raise NotImplemented; tests assert this behavior.

Developer workflow (Windows cmd examples)
- Create venv + dev deps: `python -m venv .venv` → `.venv\Scripts\pip.exe install -e ".[dev]"`.
- Lint (Ruff): `lint.cmd` or `.venv\Scripts\python.exe -m ruff .`; Format: Black/Isort/Ruff Format.
- Types: `.venv\Scripts\python.exe -m mypy`; Imports layering: `.venv\Scripts\lint-imports.exe --config importlinter.ini`.
- Tests: `.venv\Scripts\python.exe -m pytest -q`. Active tests: application/domain assertions for NotImplemented; infra/ingest tests are skipped.

Patterns you should follow
- New use case: define DTOs under `application/dto/`, Ports in `application/ports/`, orchestrate in `application/use_cases/`. Test with fakes (see `tests/application/test_query_use_case.py`).
- New adapter: implement under `infrastructure/<tech>/` and translate external exceptions to typed errors; add adapter contract tests in `tests/infrastructure/`.
- Interface stays thin: parse/format only; call use cases created in `config/composition.py`.

Examples you can follow
- Pure domain service: `domain/services/chunking.py` with tests in `tests/domain/test_chunking.py` (section detection, packing with overlap, fallback paths).

Quick review before you commit
- Inward-only imports respected? Any app/domain → infra leaks? Ports tech-agnostic? Placeholders preserved where tests expect NotImplemented? Settings wired only in config?
