"""HTTP API for query and async ingestion.

Why: Konsumierbare API ohne Business-Logik; pure Delegation.
"""

from typing import Any

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as err:
    raise ImportError(
        "FastAPI not installed. Install with: pip install 'bu-superagent[http]'"
    ) from err

from bu_superagent.application.dtos import IngestRequest, QueryRequest


# Pydantic models for request/response validation
class QueryRequestModel(BaseModel):
    """Request model for /v1/query endpoint."""

    collection: str
    question: str
    top_k: int = 5
    use_mmr: bool = True
    use_reranker: bool = False
    use_hybrid: bool = False
    confidence_threshold: float = 0.25


class QueryResponseModel(BaseModel):
    """Response model for /v1/query endpoint."""

    status: str
    answers: list[dict[str, Any]] | None = None
    error: str | None = None


class IngestRequestModel(BaseModel):
    """Request model for /v1/ingest endpoint."""

    collection: str
    shard_key: str
    docs: list[dict[str, Any]]  # each: {"id": str, "text": str, "meta": dict}


class IngestResponseModel(BaseModel):
    """Response model for /v1/ingest endpoint."""

    status: str
    job_id: str
    message: str


class JobStatusResponseModel(BaseModel):
    """Response model for /v1/jobs/{id} endpoint."""

    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: str | None = None


# Global state (initialized on startup)
app = FastAPI(title="BU Superagent RAG API", version="1.0.0")
container: Any | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize dependencies on startup with DI container."""
    global container

    from bu_superagent.config.compose import build_container

    container = build_container()
    # Container will lazy-load adapters on first use


@app.post("/v1/query", response_model=QueryResponseModel)
async def query(req: QueryRequestModel) -> QueryResponseModel:
    """Synchronous query endpoint with hybrid search and confidence gate.

    Args:
        req: Query request with question and retrieval parameters

    Returns:
        Query response with answers or error

    Example:
        POST /v1/query
        {
            "collection": "kb_chunks",
            "question": "What is RAG?",
            "top_k": 5,
            "use_mmr": true,
            "use_hybrid": false,
            "confidence_threshold": 0.25
        }
    """
    if container is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get use case from container
        query_uc = container.get_query_use_case()

        # Convert Pydantic model to application DTO
        dto = QueryRequest(
            collection=req.collection,
            question=req.question,
            top_k=req.top_k,
            use_mmr=req.use_mmr,
            use_reranker=req.use_reranker,
            use_hybrid=req.use_hybrid,
            confidence_threshold=req.confidence_threshold,
        )

        # Execute use case
        result = query_uc.execute(dto)

        if result.ok:
            return QueryResponseModel(
                status="success",
                answers=result.value.get("answers", []),
            )
        else:
            return QueryResponseModel(
                status="error",
                error=str(result.error),
            )

    except Exception as ex:
        return QueryResponseModel(
            status="error",
            error=f"Internal error: {ex}",
        )


@app.post("/v1/ingest", response_model=IngestResponseModel)
async def ingest(req: IngestRequestModel, background_tasks: BackgroundTasks) -> IngestResponseModel:
    """Async ingestion endpoint with job queuing.

    Enqueues document batches for background processing.
    Returns job ID for status tracking.

    Args:
        req: Ingest request with collection, shard_key, and documents
        background_tasks: FastAPI background tasks

    Returns:
        Ingest response with job ID

    Example:
        POST /v1/ingest
        {
            "collection": "kb_chunks",
            "shard_key": "tenant1",
            "docs": [
                {"id": "doc1", "text": "content", "meta": {"source": "web"}},
                {"id": "doc2", "text": "more content", "meta": {}}
            ]
        }
    """
    if container is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get use case from container
        ingest_uc = container.get_ingest_use_case()

        # Convert Pydantic model to application DTO
        dto = IngestRequest(
            collection=req.collection,
            shard_key=req.shard_key,
            docs=req.docs,
        )

        # Plan batches
        result = ingest_uc.plan(dto)
        if not result.ok:
            return IngestResponseModel(
                status="error",
                job_id="",
                message=str(result.error),
            )

        batches = result.value

        # Enqueue batches (simplified: single job ID for all batches)
        job_id = f"job-{hash(req.shard_key)}-{len(req.docs)}"

        # Background processing (in real system: queue each batch)
        def process_batches():
            for batch in batches:
                result = ingest_uc.execute_batch(req.collection, batch)
                if not result.ok:
                    print(f"Batch failed: {result.error}")

        background_tasks.add_task(process_batches)

        return IngestResponseModel(
            status="accepted",
            job_id=job_id,
            message=f"Enqueued {len(batches)} batches ({len(req.docs)} docs)",
        )

    except Exception as ex:
        return IngestResponseModel(
            status="error",
            job_id="",
            message=f"Internal error: {ex}",
        )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponseModel)
async def get_job_status(job_id: str) -> JobStatusResponseModel:
    """Get status of async ingestion job.

    Args:
        job_id: Job ID returned from /v1/ingest

    Returns:
        Job status response

    Example:
        GET /v1/jobs/job-123456-1000
    """
    # Simplified: real implementation would query job store
    return JobStatusResponseModel(
        job_id=job_id,
        status="processing",
        message="Job status tracking not implemented (use queue backend)",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "bu-superagent"}
