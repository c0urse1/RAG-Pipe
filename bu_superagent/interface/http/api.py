from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ...config.composition import build_application
from ...application.dto.ingest_dto import IngestDocumentDTO
from ...application.dto.query_dto import QueryRequest


app = FastAPI(title="BU Superagent API")
container = build_application()


class IngestRequest(BaseModel):
    id: str
    title: str
    content: str


class QueryBody(BaseModel):
    text: str
    top_k: int = 5


@app.post("/ingest")
def ingest(req: IngestRequest):
    container["ingest"].execute([IngestDocumentDTO(**req.model_dump())])
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryBody):
    try:
        results = container["query"].execute(QueryRequest(question=req.text, top_k=req.top_k))
    except NotImplementedError:
        results = []
    return {"results": results}
