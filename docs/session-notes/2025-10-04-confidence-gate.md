# Confidence-Gate Branch Implementation

**Date:** October 4, 2025
**Branch:** `Confidence-Gate`
**Base:** `feat/infra-adapters`

## Summary

This branch implements domain-level error handling and data models to support confidence gating in the RAG pipeline, following SAM (Simple Architecture Mindset) principles.

## Changes Made

### 1. Domain Errors (`bu_superagent/domain/errors.py`)

**Added:**
- `LowConfidenceError` - Frozen dataclass for signaling when retrieval confidence falls below threshold
  - Fields: `message`, `top_score`, `threshold`
  - Enables typed business-meaningful failures without leaking infrastructure details
- `RetrievalError` - Generic retrieval failure after infrastructure errors are mapped

**Why:** Typed domain errors are the only way the application can signal business-meaningful failures without leaking infra details (SAM principle).

### 2. Domain Models (`bu_superagent/domain/models.py` - NEW)

**Created:**
- `RetrievedChunk` - A chunk retrieved from vector store with similarity score
  - Fields: `id: str`, `text: str`, `metadata: dict[str, Any]`, `vector: list[float] | None`, `score: float`
  - **Moved from application/ports** to domain layer (proper SAM architecture)
- `RankedChunk` - Wraps `RetrievedChunk` with reranking position
  - Fields: `chunk: RetrievedChunk`, `rank: int`
- `Citation` - Citation reference for generated answers
  - Fields: `chunk_id: str`, `source: str`, `score: float`

**Why:** Application orchestrates with simple DTO-like domain POJOs. Keep them portable & testable (SAM principle).

**Architecture Fix:** `RetrievedChunk` was originally in `application/ports/vector_store_port.py`, but according to SAM it should be a pure domain model. The port now imports and re-exports it from domain.

### 3. Infrastructure Updates

**Modified all vector store adapters** to include `vector` field in `RetrievedChunk`:
- `infrastructure/vectorstore/chroma_vector_store.py` - Added `vector=None` (Chroma doesn't return vectors by default)
- `infrastructure/vectorstore/qdrant_vector_store.py` - Added `vector=None` (Qdrant doesn't return vectors by default)
- `infrastructure/vectorstore/faiss_vector_store.py` - Added `vector=None` (FAISS doesn't store vectors separately)

**Why:** Adapters set `vector=None` by default to save bandwidth, but the domain model supports it for use cases that need it.

### 4. Tests

**Added:**
- `tests/domain/test_errors.py` - Comprehensive tests for domain errors (5 tests)
  - Tests `LowConfidenceError` dataclass behavior, immutability, and inheritance
  - Tests `RetrievalError` basic functionality
  - Ensures backward compatibility with existing `ValidationError`
- `tests/domain/test_models.py` - Tests for domain models (6 tests)
  - Tests `RetrievedChunk` creation with all fields including optional `vector`
  - Tests `RankedChunk` creation and immutability
  - Tests `Citation` creation and immutability

**Coverage:** 100% on new domain code, all infrastructure tests pass

### 5. Documentation (`.github/copilot-instructions.md`)

**Updated:**
- Restructured for clarity with clear sections
- Added missing guideline about file removal/renaming
- More specific examples from the codebase
- Improved actionable patterns for AI agents

## Verification

✅ All domain tests pass (18 tests)
✅ All infrastructure tests pass (3 tests)
✅ Ruff linting passes
✅ MyPy type checking passes
✅ Import layering constraints respected (0 contracts broken)
✅ 100% test coverage on new domain code (86.5% overall domain coverage)

## Next Steps

These domain errors and models enable implementing:
1. Confidence-based filtering in retrieval use cases
2. Reranking pipeline with `RankedChunk`
3. Citation tracking with `Citation` model
4. Graceful error handling that respects Clean Architecture boundaries

## Architecture Compliance

✅ Domain remains pure (no I/O, no external deps)
✅ Errors are typed and business-meaningful
✅ Models are immutable frozen dataclasses
✅ No infrastructure leakage into domain
✅ Follows existing patterns in codebase
