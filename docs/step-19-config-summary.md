# Step 19: Environment-Driven Configuration Summary

**Status**: ✅ Complete
**Date**: October 6, 2025

## Overview

Implemented comprehensive environment-driven configuration with feature flags and dependency injection container, following the principle: **"Config is the ONLY layer that reads environment variables."**

## Files Created/Modified

### 1. `config/settings.py` - Enhanced Settings
**Before**: Basic settings with ~10 environment variables
**After**: Comprehensive settings with 40+ environment variables organized by concern

**New Configuration Sections:**
- ✅ Vector Store (backend, URL, API key, GRPC, timeout)
- ✅ Scaling (shards, replicas, quantization)
- ✅ Embeddings (model, device, batch size)
- ✅ Work Queue (backend, Redis config)
- ✅ Blob Storage (backend, MinIO config)
- ✅ Query Features (hybrid, MMR, confidence threshold)
- ✅ Telemetry (enabled, OTLP endpoint, environment)

**Key Features:**
- All env vars have sensible defaults
- Feature flags for optional capabilities (quantization, hybrid, MMR)
- Frozen dataclass (immutable after creation)
- Lazy evaluation via `default_factory` (env read on access)

### 2. `config/compose.py` - DI Container
**New File**: Complete dependency injection container

**Container Class Responsibilities:**
1. Read settings from environment (via AppSettings)
2. Choose adapters based on settings (vector_backend, workqueue_backend, etc.)
3. Inject dependencies into use cases
4. Apply feature flags (quantization, hybrid search, etc.)

**Adapter Selection Logic:**
```python
vector_backend:
  qdrant → QdrantVectorStoreAdapter (with sharding/replication/quantization)
  chroma → ChromaVectorStoreAdapter (legacy)
  faiss → FaissVectorStoreAdapter (in-memory)
  weaviate → NotImplementedError
  elasticsearch → NotImplementedError

workqueue_backend:
  redis → RedisWorkQueueAdapter
  fake → FakeWorkQueue (testing/dev)

blobstore_backend:
  minio → MinioBlobStoreAdapter
  s3 → MinioBlobStoreAdapter (S3-compatible)
  fake → FakeBlobStore (testing/dev)

telemetry_enabled:
  true → OpenTelemetryAdapter
  false → NoopTelemetry
```

**Lazy Loading:**
- Adapters created on first use
- Cached for subsequent calls
- Avoids initialization overhead

**Convenience Functions:**
```python
container = build_container()
query_uc = container.get_query_use_case()
ingest_uc = container.get_ingest_use_case()

# Or quick access:
query_uc = get_query_use_case()
ingest_uc = get_ingest_use_case()
```

### 3. `.env.example` - Configuration Template
**New File**: Comprehensive environment template with 40+ variables

**Organization:**
- Vector Store Configuration (backend, Qdrant, Chroma, collection)
- Scaling Configuration (shards, replicas, quantization)
- Embedding Configuration (model, device, batch size)
- Work Queue Configuration (backend, Redis)
- Blob Storage Configuration (backend, MinIO/S3)
- LLM Configuration (base URL, API key, model)
- Reranker Configuration (model, device, sigmoid)
- Query Configuration (hybrid, MMR, confidence)
- Telemetry Configuration (enabled, OTLP, environment)
- Legacy Compatibility (old QDRANT_HOST/PORT)

**Key Features:**
- Detailed comments explaining each setting
- Recommendations for production vs. dev
- Alternatives for each configurable component
- Safe defaults for local development

### 4. `interface/http/api.py` - Updated HTTP API
**Change**: Replaced global use case instances with DI container

**Before:**
```python
settings = AppSettings()
embed = build_embedding(settings)
vs = build_vector_store(settings)
wq = build_work_queue(settings)
query_uc = QueryKnowledgeBaseScalable(embed=embed, vs=vs)
ingest_uc = IngestDocumentsParallel(embed=embed, vs=vs, wq=wq)
```

**After:**
```python
from bu_superagent.config.compose import build_container

container = build_container()
# Use cases fetched on demand:
query_uc = container.get_query_use_case()
ingest_uc = container.get_ingest_use_case()
```

**Benefits:**
- Single source of truth (container)
- Cleaner startup code
- Easier testing (inject fake container)
- Lazy adapter initialization

---

## Configuration Examples

### Development (Local)
```bash
# Minimal config for local development
VECTOR_BACKEND=faiss           # In-memory (no Docker)
EMBEDDING_DEVICE=cpu           # No GPU required
WORKQUEUE_BACKEND=fake         # No Redis required
BLOBSTORE_BACKEND=fake         # No MinIO required
TELEMETRY_ENABLED=false        # No observability overhead
```

### Staging (Docker Compose)
```bash
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
VECTOR_SHARDS=2
VECTOR_REPLICAS=1
USE_QUANTIZATION=true
QUANTIZATION_KIND=scalar

EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=512

WORKQUEUE_BACKEND=redis
REDIS_HOST=localhost

BLOBSTORE_BACKEND=minio
MINIO_ENDPOINT=localhost:9000

TELEMETRY_ENABLED=true
OTLP_ENDPOINT=
```

### Production (Kubernetes)
```bash
VECTOR_BACKEND=qdrant
QDRANT_URL=http://qdrant-cluster:6333
QDRANT_API_KEY=${QDRANT_SECRET}
QDRANT_PREFER_GRPC=true
VECTOR_SHARDS=6
VECTOR_REPLICAS=2
USE_QUANTIZATION=true
QUANTIZATION_KIND=scalar

EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=1024

WORKQUEUE_BACKEND=redis
REDIS_HOST=redis-cluster
REDIS_PASSWORD=${REDIS_SECRET}

BLOBSTORE_BACKEND=s3
MINIO_ENDPOINT=s3.amazonaws.com
MINIO_ACCESS_KEY=${AWS_ACCESS_KEY}
MINIO_SECRET_KEY=${AWS_SECRET_KEY}
MINIO_BUCKET=prod-rag-documents
MINIO_SECURE=true

QUERY_USE_HYBRID=true
QUERY_USE_MMR=true
QUERY_CONFIDENCE_THRESHOLD=0.35

TELEMETRY_ENABLED=true
OTLP_ENDPOINT=http://otel-collector:4317
TELEMETRY_ENVIRONMENT=production
```

---

## Feature Flags

### 1. Vector Store Scaling
**USE_QUANTIZATION** (default: true)
- `true` - Enable compression (4-32x memory savings)
- `false` - Store full-precision vectors

**QUANTIZATION_KIND** (default: scalar)
- `scalar` - INT8 quantization (4x compression, minimal quality loss)
- `product` - Product quantization (16x compression)
- `binary` - Binary quantization (32x compression)

**VECTOR_SHARDS** (default: 6)
- Number of shards for horizontal scaling
- Recommendation: 1 shard per node

**VECTOR_REPLICAS** (default: 2)
- Replication factor for high availability
- Recommendation: 2 for production

### 2. Query Features
**QUERY_USE_HYBRID** (default: true)
- `true` - Enable hybrid search (vector + lexical fusion with RRF)
- `false` - Vector-only search

**QUERY_USE_MMR** (default: true)
- `true` - Enable MMR for diversity
- `false` - Pure similarity ranking

**QUERY_CONFIDENCE_THRESHOLD** (default: 0.25)
- Minimum confidence for answering (0.0-1.0)
- Lower = more answers, higher = more reliable

### 3. Storage Strategy
**STORE_TEXT_PAYLOAD** (default: false)
- `true` - Store full text in vector DB (simpler)
- `false` - Store only vectors, use blob store (scalable)

### 4. Telemetry
**TELEMETRY_ENABLED** (default: true)
- `true` - Enable OpenTelemetry metrics
- `false` - No-op telemetry adapter

**OTLP_ENDPOINT** (default: empty)
- Empty = console logging only (dev)
- URL = export to OTLP collector (production)

---

## Architecture Principles

### 1. Single Source of Truth
**Environment variables ONLY in config layer**
- ✅ `config/settings.py` reads env vars
- ❌ No env access in domain/application/infrastructure/interface

### 2. Dependency Injection
**Container manages all wiring**
- Adapters chosen based on settings
- Use cases receive dependencies via constructor
- No service locator anti-pattern

### 3. Feature Flags
**Toggle capabilities without code changes**
- Quantization on/off
- Hybrid search on/off
- MMR on/off
- Telemetry on/off

### 4. Defaults for Local Dev
**Zero-config for getting started**
- FAISS (in-memory) as default vector store
- Fake adapters for queue/blob store
- CPU for embeddings
- No telemetry overhead

### 5. Production-Ready Scaling
**Environment-driven optimization**
- Qdrant with sharding/replication
- GPU batching for embeddings
- Redis for work queue
- MinIO/S3 for blob storage
- OpenTelemetry for observability

---

## Testing Strategy

### Unit Tests
Mock container in tests:
```python
def test_http_query_endpoint():
    fake_container = FakeContainer()
    app.dependency_overrides[get_container] = lambda: fake_container
    # Test endpoint logic without real adapters
```

### Integration Tests
Real adapters with test config:
```python
settings = AppSettings(
    vector_backend="faiss",
    workqueue_backend="fake",
    blobstore_backend="fake",
)
container = build_container(settings)
# Test with in-memory adapters
```

### Contract Tests
Real infrastructure with Docker:
```bash
VECTOR_BACKEND=qdrant QDRANT_URL=http://localhost:6333 \
pytest tests/infrastructure/test_qdrant_adapter_contract.py
```

---

## Migration Guide

### From Legacy composition.py
**Before:**
```python
from bu_superagent.config.composition import (
    build_embedding,
    build_vector_store,
    build_query_use_case,
)

settings = AppSettings()
embed = build_embedding(settings)
vs = build_vector_store(settings)
uc = build_query_use_case()
```

**After:**
```python
from bu_superagent.config.compose import build_container

container = build_container()
uc = container.get_query_use_case()
# Adapters created lazily inside container
```

### Benefits
- ✅ Single source of truth (container)
- ✅ Lazy initialization (only create what's used)
- ✅ Easier testing (inject fake container)
- ✅ Cleaner code (no manual wiring)

---

## Performance Impact

### Memory Optimization
- **Quantization**: 4-32x memory savings (configurable)
- **Lazy loading**: Only initialize used adapters
- **Sharding**: Distribute data across nodes

### Throughput Optimization
- **GPU batching**: 512-1024 doc batches (configurable)
- **GRPC**: Optional for Qdrant (lower latency)
- **Replication**: Load balancing across replicas

### Observability
- **OpenTelemetry**: Vendor-neutral metrics
- **Feature flags**: Toggle overhead on/off
- **Graceful degradation**: No-ops if disabled

---

## Next Steps

1. ✅ **Step 19 Complete**: Environment-driven config with DI container
2. 🔄 **Documentation**: Update README with environment variables
3. 🔄 **Testing**: Add container tests with various configs
4. 🔄 **Deployment**: Create Kubernetes ConfigMaps/Secrets
5. 🔄 **Monitoring**: Set up Grafana dashboards for feature flags

---

**Why**: Einzige Stelle mit Env; Feature-Flags erlauben toggles (Hybrid, Quantization, Shards) ✓
