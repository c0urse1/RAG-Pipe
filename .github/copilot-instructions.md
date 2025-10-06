## Copilot Instructions – BU Superagent (Strict Architecture Mode)

**Purpose:** Maintain the Clean Architecture RAG skeleton (German BU domain). Domain stays pure (no I/O, deterministic); Application orchestrates; Infrastructure adapts tech; Interface/config only wire things together.

### Architecture snapshot
- `domain/`: Pure business logic—entities, value objects, domain services. **No I/O, no env access, no infrastructure imports**. `services/chunking.py` does semantic chunking (section → paragraph → sentence → overlap merge) with tunable `ChunkingParams`. Uses regex patterns to detect German document headings (Markdown `#`, numbered sections `1.1.1`, ALL-CAPS sections). `services/ranking.py` provides deterministic algorithms: `deduplicate_by_cosine`, `mmr` (Maximal Marginal Relevance), and `passes_confidence` for confidence-gate pattern. `services/relevance_scoring.py::cosine_similarity` is intentionally `NotImplemented` (placeholder for future implementation). Domain models (`models.py`): `RetrievedChunk`, `Citation`, `RankedChunk` are pure POJOs. Domain errors (`errors.py`): typed errors like `LowConfidenceError`, `ValidationError`, `EmbeddingError` enable business-meaningful failures without leaking infra details.
- `application/`: Use cases + ports (interfaces). `use_cases/ingest_documents.py::IngestDocuments.execute` orchestrates ingestion pipeline (returns chunk count). `use_cases/query_knowledge_base.py::QueryKnowledgeBase.execute` orchestrates: validate → embed → search → [optional rerank] → dedup → optional MMR → confidence gate → LLM generation. Uses `Result[T, E]` type with `.ok` property and `.success(val)`/`.failure(err)` factory methods for explicit error handling. All use cases receive dependencies via constructor (ports), never import infrastructure directly.
- `infrastructure/`: Adapters implementing ports. **Lazy imports via `importlib.import_module`** for testability—allows tests to inject fake modules via `monkeypatch`. Adapters wrap errors in domain exceptions (`RuntimeError` → `EmbeddingError`). Examples: `embeddings/hf_sentence_transformers.py` prefixes E5 queries/passages and caches models; `vectorstore/{chroma,qdrant,faiss}_vector_store.py` implement `VectorStorePort`; `parsing/pdf_text_extractor.py` wraps pypdf.
- `interface/`: CLI/HTTP shells only. Entry point: `bu-superagent` command (defined in `pyproject.toml`). `cli/main.py` deliberately raises `NotImplementedError` if `--question` is missing (tests assert this). Keep thin: parse args → call composition builders → delegate to use cases → format output.
- `config/`: **ONLY place for env/DI/wiring**. `composition.py` provides builder functions (`build_embedding`, `build_vector_store`, `build_llm`, `build_reranker`, `build_query_use_case`) that read `settings.py::AppSettings` and wire concrete adapters. `build_vector_store` respects `VECTOR_BACKEND` (`qdrant`|`chroma`|`faiss`) and falls back to FAISS for unknown values. All env vars read lazily via `field(default_factory=lambda: os.getenv(...))`.

### Core contracts & behaviors
- **Vector store port** (`application/ports/vector_store_port.py`) requires strict call sequence: `ensure_collection` → `upsert` → `search`. Tests in `tests/infrastructure/test_vectorstore_adapters_fake.py` assert this ordering—**always call `ensure_collection` before persisting**. Adapters return `RetrievedChunk` with `vector=None` by default (saves bandwidth; domain model supports vectors for MMR/diversity).
- **Embedding port** (`application/ports/embedding_port.py`) supports kinds `"mxbai"|"jina"|"e5"`. The Sentence Transformers adapter (`embeddings/sentence_transformers_adapter.py`) prefixes E5 queries ("query: {text}") and passages ("passage: {text}") automatically via `_prefix_e5_query/_passage` helpers. Models are cached per name. **All embeddings are L2-normalized** before returning.
- **Document loading** uses `DocumentLoaderPort.load` returning `DocumentPayload`. `PlainTextLoaderAdapter` trims UTF-8 files; `PDFTextExtractorAdapter` wraps pypdf. Both raise `RuntimeError` on failure (adapters wrap as `DocumentError`).
- **Ingestion pipeline** (`IngestDocuments.execute`): load → chunk via domain service (`chunk_text_semantic`) → embed texts → `ensure_collection` + `upsert` with payloads carrying `doc_id`, `chunk_index`, `section_title`, source metadata. Returns total chunk count.
- **Query pipeline** (`QueryKnowledgeBase.execute`): validate → embed query → vector search → **[optional rerank]** → deduplicate (`deduplicate_by_cosine`) → optional MMR (`mmr` for diversity) → confidence gate (`passes_confidence`) → LLM generation if above threshold, else return `LowConfidenceError` with context. Uses `Result[RAGAnswer, DomainError]` type.
- **Reranker port** (`application/ports/reranker_port.py`) provides semantic reranking via cross-encoders. `score(query, candidates)` returns relevance scores (higher = more relevant). Positioned AFTER vector search, BEFORE dedup/MMR for maximum accuracy. Opt-in via `use_reranker=True` in QueryRequest (default: False). `pre_rerank_k` controls candidate pool size before reranking (default: 20, recommended: max(top_k*4, 20)).

### Infrastructure adapters & wiring
- **Vector backend selection**: `build_vector_store` in `config/composition.py` honours `VECTOR_BACKEND` env var (`qdrant` | `chroma` | `faiss`, default `chroma`). **Unknown values fall back to FAISS** for testability. Qdrant adapter recreates collection with cosine distance each time; Chroma persists under `var/chroma/...`; FAISS holds in-memory `IndexFlatIP`. **Windows note**: chromadb requires Visual C++ Build Tools—use `VECTOR_BACKEND=faiss` or install from [visualstudio.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- **Lazy imports pattern**: All infrastructure adapters use `importlib.import_module` instead of direct imports (e.g., `chromadb = import_module("chromadb")`). This enables test isolation via `monkeypatch.setitem(sys.modules, "chromadb", FakeModule())`. See `tests/infrastructure/test_vectorstore_adapters_fake.py` for examples.
- **Embedding device config**: `build_embedding` in `composition.py` reads `EMBEDDING_DEVICE` env var (default `cpu`). Production may use `cuda`; adjust settings before adapter instantiation. Sentence Transformers adapter caches models by name in `_model_cache`.
- **LLM adapter**: `VLLMOpenAIAdapter` wraps OpenAI-compatible API (vLLM, Ollama, etc.). Reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` from settings. Default points to local vLLM server (`http://localhost:8000/v1`).
- **Reranker adapter**: `CrossEncoderAdapter` in `infrastructure/reranking/cross_encoder_adapter.py` implements cross-encoder reranking. Uses lazy imports (`importlib.import_module`) for sentence-transformers. Model caching per (model_name, device) key. Sigmoid application for BGE models (converts raw logits → [0,1] probabilities). Default model: `BAAI/bge-reranker-v2-m3` (multilingual). Settings: `reranker_model`, `reranker_device` (cpu/cuda), `reranker_apply_sigmoid` (true for BGE). Wraps errors as `RuntimeError`, use case wraps in domain `RerankerError`.

### Developer workflow (Windows cmd examples)
- **Setup**: `python -m venv .venv` → `.venv\Scripts\pip.exe install -e ".[dev]"` (installs package in editable mode with dev deps). **Note**: Vector stores (chromadb, qdrant, faiss) are optional dependencies—install with `.[dev,vectorstores]` if needed (requires Visual C++ Build Tools on Windows for chromadb's hnswlib compilation).
- **Lint/format**: `lint.cmd` (wrapper for Ruff) or `.venv\Scripts\python.exe -m ruff .` directly. Ruff handles both linting and formatting (replaces black/isort in practice).
- **Type checking**: `.venv\Scripts\python.exe -m mypy` (configured via `mypy.ini`, strict mode enabled).
- **Import boundaries**: `.venv\Scripts\lint-imports.exe --config importlinter.ini` validates layering contracts (runs in pre-commit).
- **Tests**: `.venv\Scripts\python.exe -m pytest` (defaults to `-q --cov=bu_superagent --cov-report=term-missing --cov-fail-under=75` from `pyproject.toml`). Active suites cover domain chunking (`tests/domain/test_chunking.py`), ingestion flow (`tests/application/test_ingest_use_case_v2.py`), CLI parsing, reranker integration, and infra adapter fakes.
- **Run CLI**: `.venv\Scripts\bu-superagent --help` (entry point defined in `pyproject.toml[project.scripts]`).
- **Docker services**: `docker-compose up -d` starts Qdrant + vLLM (see `docker-compose.yml` for LLM model config).

### Guardrails & examples
- **File management**: Do not remove/rename files without checking Port relations. If a file is an unused duplicate adapter, remove it. Utilities belong under `infrastructure/<tech>/utils.py` and are only imported by adapters.
- **Layering enforcement**: `importlinter.ini` defines strict contracts. Application can import domain; infrastructure/interface are independent; interface CANNOT import domain directly (only via application). Run `.venv\Scripts\lint-imports.exe --config importlinter.ini` to validate. Never import infrastructure from domain/application.
- **NotImplemented sentinels**: `domain/services/relevance_scoring.py::cosine_similarity` deliberately raises `NotImplemented` (placeholder); `cli/main.py` raises `NotImplementedError` when `--question` is missing (test assertion). **Do not remove these—tests rely on them**.
- **Result type pattern**: Use cases return `Result[T, E]` with explicit `Result.success(value)` or `Result.failure(error)` factory methods. Check `result.ok` before accessing `result.value`/`result.error`. The type is defined in `application/use_cases/query_knowledge_base.py` (local definition) and `domain/types.py` (domain-level). No silent failures or uncaught exceptions in business logic.
- **Domain chunking**: `chunk_text_semantic` in `domain/services/chunking.py` is deterministic. Follow tests in `tests/domain/test_chunking.py` when modifying—maintain overlap/tail semantics and section-aware merging.
- **Adapter patterns**: New adapters must (1) use `import_module` for lazy imports, (2) wrap errors in domain exceptions (e.g., `RuntimeError` → `EmbeddingError`), (3) add contract tests under `tests/infrastructure/` with fake modules. See `test_vectorstore_adapters_fake.py` for reference.
- **Interface shells**: Keep `interface/cli/` and `interface/http/` thin. Parse inputs → call composition builders (`build_query_use_case`, etc.) → format outputs. No business logic here.

### Quality gates & architecture guards
- **Import linter**: `importlinter.ini` enforces boundaries. Infrastructure → Application is allowed (adapters implement ports). Domain/Application remain 3rd-party-free. Run `.venv\Scripts\lint-imports.exe --config importlinter.ini` before commits.
- **Typed errors + Result[T,E]**: All infrastructure exceptions wrapped as domain errors (`EmbeddingError`, `RerankerError`, `DocumentError`). Use cases propagate via `Result.failure(error)`. Never let infrastructure exceptions leak to interface layer.
- **Test markers**: Slow tests (model downloads, real API calls) marked with `@pytest.mark.slow`. Skip in fast CI: `pytest -m "not slow"`. Reranker contract tests use `@pytest.mark.skipif(sys.platform == "win32")` for integration tests.
- **Pre-commit hooks**: isort, ruff (lint+format), black, mypy, import-linter run automatically. All must pass before commit. Use `git commit --no-verify` only for emergency hotfixes.
- **Coverage threshold**: 75% minimum enforced via pytest-cov. Current: 78%+. Domain layer should approach 100%, application 80%+, infrastructure 70%+ (adapters have integration tests).

### Reranker operational tips (scaling to production)
- **Pool size**: `pre_rerank_k = max(top_k*4, 20)` is sensible default. Tune for latency/quality tradeoff:
  - Too small: misses relevant candidates (quality loss)
  - Too large: slow reranking, diminishing returns (latency penalty)
  - Example: `top_k=5` → retrieve 20 candidates, rerank, return top 5
- **Model choice**: Start with `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed (<100ms/query on CPU). Switch to `BAAI/bge-reranker-v2-m3` for high-quality multilingual settings (German BU domain). BGE models require `apply_sigmoid=true` (converts raw logits to probabilities).
- **Batching**: CrossEncoder batch size 16-64 depending on VRAM. Sentence-transformers handles batching internally. For high-throughput: consider async processing with queue.
- **Caching**: Model downloads cached in `~/.cache/huggingface/` by default. CI/CD should cache this directory between runs to avoid re-downloads. First run may take 1-2min for model download.
- **Device config**: `reranker_device=cpu` for dev/test. `reranker_device=cuda` for production if GPU available. Reranking is 10-50x faster on GPU.
- **Performance**: Cross-encoder reranking adds 50-200ms latency on CPU, 5-20ms on GPU. Only enable when quality is critical. Disable for latency-sensitive apps.

### Testing strategy
- **Domain tests**: Pure functions, deterministic, 100% coverage. No mocks needed. Test edge cases (empty inputs, boundary values).
- **Application tests**: Use case orchestration with fake ports. Verify Result types, error propagation, pipeline ordering. Cover happy path + error scenarios.
- **Infrastructure tests**: Contract tests with fake modules (`monkeypatch.setitem(sys.modules)`). Verify lazy imports, model caching, error wrapping. Mark integration tests as `@pytest.mark.slow`.
- **Interface tests**: CLI arg parsing only. Verify flags map to QueryRequest fields. No business logic.
- **Run fast tests**: `pytest -q` (excludes slow tests by default)
- **Run all tests**: `pytest -q -m ""` (includes slow tests)
- **Run with coverage**: `pytest --cov=bu_superagent --cov-report=term-missing`
