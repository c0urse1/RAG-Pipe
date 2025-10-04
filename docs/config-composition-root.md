"""Composition Root (Dependency Injection) - Configuration Layer

This module serves as the ONLY place in the application that:
1. Reads environment variables (via AppSettings)
2. Instantiates concrete infrastructure adapters
3. Wires dependencies into use cases

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE PRINCIPLE: Composition Root (SAM - Stable Abstractions Minimal)
═══════════════════════════════════════════════════════════════════════════════

The composition root ensures:
✅ Application/Domain layers remain PURE (no env access, no concrete types)
✅ Infrastructure adapters are constructed ONCE at startup
✅ Environment configuration is CENTRALIZED in AppSettings
✅ Dependencies flow INWARD (infrastructure → application → domain)

═══════════════════════════════════════════════════════════════════════════════
LAYER RESPONSIBILITIES
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIG LAYER (composition.py + settings.py)                                 │
│ ───────────────────────────────────────────────────────────────────────────│
│ • Reads environment variables (ONLY HERE!)                                  │
│ • Instantiates concrete infrastructure adapters                             │
│ • Wires dependencies into use cases                                         │
│ • Provides builder functions for CLI/HTTP layers                            │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓ constructs & injects
┌─────────────────────────────────────────────────────────────────────────────┐
│ APPLICATION LAYER (use_cases/ + ports/)                                     │
│ ───────────────────────────────────────────────────────────────────────────│
│ • Receives dependencies via constructor (ports/interfaces)                  │
│ • Orchestrates domain services and infrastructure calls                     │
│ • Returns Result types (no exceptions in control flow)                      │
│ • PURE: No env access, no concrete infrastructure types                     │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓ uses
┌─────────────────────────────────────────────────────────────────────────────┐
│ DOMAIN LAYER (services/ + models/ + errors/)                                │
│ ───────────────────────────────────────────────────────────────────────────│
│ • Pure business logic (deterministic, testable)                             │
│ • No I/O, no env access, no infrastructure dependencies                     │
│ • Domain models, value objects, domain services                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
BUILDER FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

build_embedding(settings: AppSettings) -> EmbeddingPort
    Constructs embedding adapter from settings.
    Environment variables:
    - EMBEDDING_MODEL: Model name (default: intfloat/multilingual-e5-large-instruct)
    - EMBEDDING_DEVICE: cpu or cuda (default: cpu)

build_vector_store(settings: AppSettings) -> VectorStorePort
    Constructs vector store adapter based on backend selection.
    Environment variables:
    - VECTOR_BACKEND: chroma | qdrant | faiss (default: chroma)
    - CHROMA_DIR: Persistence directory for Chroma (default: var/chroma/e5_1024d)
    - QDRANT_HOST: Qdrant server host (default: localhost)
    - QDRANT_PORT: Qdrant server port (default: 6333)
    - VECTOR_COLLECTION: Collection name (default: kb_chunks_de_1024d)

build_llm(settings: AppSettings) -> LLMPort
    Constructs LLM adapter for generative answers.
    Environment variables:
    - LLM_BASE_URL: OpenAI-compatible API endpoint (default: http://localhost:8000/v1)
    - LLM_API_KEY: API key (default: EMPTY)
    - LLM_MODEL: Model name (default: meta-llama/Meta-Llama-3.1-8B-Instruct)

build_query_use_case(with_llm: bool = True) -> QueryKnowledgeBase
    Wires all dependencies for query execution.

    Args:
        with_llm: If True, includes LLM for generative RAG answers.
                  If False, uses extractive fallback (concatenated chunks).

    Returns:
        QueryKnowledgeBase use case with all dependencies wired.

build_ingest_use_case() -> IngestDocuments
    Wires all dependencies for document ingestion.

    Returns:
        IngestDocuments use case with loader, embedding, and vector store.

═══════════════════════════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

# CLI Handler (interface/cli/main.py)
from bu_superagent.config.composition import build_query_use_case

def main():
    # Composition root constructs use case with all dependencies
    use_case = build_query_use_case(with_llm=True)

    # Use case orchestrates application logic
    result = use_case.execute(request)

    # Interface formats output (thin shell)
    if result.ok:
        print(result.value.text)

# HTTP Handler (future: interface/http/api.py)
from bu_superagent.config.composition import build_query_use_case

@app.post("/query")
def query_endpoint(request: QueryRequest):
    use_case = build_query_use_case(with_llm=True)
    result = use_case.execute(request)
    return result

═══════════════════════════════════════════════════════════════════════════════
TESTING STRATEGY
═══════════════════════════════════════════════════════════════════════════════

Unit Tests (Application/Domain):
    • Mock ports/interfaces
    • Test business logic in isolation
    • Fast, deterministic, no external dependencies

Integration Tests (Composition):
    • Test that builders wire dependencies correctly
    • Use monkeypatch to test env var handling
    • Verify adapters implement correct ports

End-to-End Tests (Future):
    • Test with real infrastructure (Qdrant, OpenAI)
    • Set env vars for test environment
    • Validate full RAG pipeline

═══════════════════════════════════════════════════════════════════════════════
ENVIRONMENT VARIABLE REFERENCE
═══════════════════════════════════════════════════════════════════════════════

# Embedding Configuration
export EMBEDDING_MODEL="intfloat/multilingual-e5-large-instruct"
export EMBEDDING_DEVICE="cpu"  # or "cuda"

# Vector Store Configuration
export VECTOR_BACKEND="chroma"  # or "qdrant" or "faiss"
export CHROMA_DIR="var/chroma/e5_1024d"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export VECTOR_COLLECTION="kb_chunks_de_1024d"
export STORE_TEXT_PAYLOAD="false"

# LLM Configuration
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

═══════════════════════════════════════════════════════════════════════════════
WHY THIS PATTERN? (SAM Principles)
═══════════════════════════════════════════════════════════════════════════════

1. **Testability**
   - Application logic can be tested with mock ports
   - No need to mock environment variables in unit tests
   - Integration tests can verify wiring separately

2. **Flexibility**
   - Easy to swap implementations (chroma → qdrant)
   - Change via environment variables, no code changes
   - New adapters don't affect application layer

3. **Clarity**
   - Single source of truth for dependency construction
   - Clear separation: config constructs, application orchestrates
   - Environment variables documented in one place

4. **Production Readiness**
   - All infrastructure concerns isolated in config
   - Easy to configure for different environments (dev/staging/prod)
   - No env access leaking into business logic

═══════════════════════════════════════════════════════════════════════════════
See Also:
- bu_superagent/config/settings.py: Environment variable definitions
- bu_superagent/application/use_cases/: Use case implementations
- bu_superagent/application/ports/: Port interfaces
- tests/config/: Composition root tests
═══════════════════════════════════════════════════════════════════════════════
"""
