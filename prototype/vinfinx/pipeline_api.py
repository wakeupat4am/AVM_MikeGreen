"""Loopback pipeline API for the demo website (docs/PIPELINE_API_PLAN.md). CPU only; reached through the SSH tunnel.

Routes are thin: the reads and writes, with the same checks as the command line, are in `ai/api/service.py`.
A request with an `Origin` header must come from an allowed origin; a write (POST) must have one (A2). A GET
without one is a link the reviewer opened (a stored PDF) or a request through the tunnel from the server itself.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ai.api import service
from ai.api.service import ApiError, Store
from ai.corpus.index import Index

API_PATH = os.environ.get("VINFINX_PIPELINE_PATH", "vinfinx-pipeline").strip("/")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "VINFINX_ALLOWED_ORIGINS",
        "http://127.0.0.1:4173,http://localhost:4173",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="VinFinx pipeline API", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
store = Store()
_indexes: dict[str, Index] = {}
_index_lock = threading.Lock()


def index_for(version: str) -> Index:
    """One Index per version, kept for the process: repeated questions reuse their embedding."""
    store.version_dir(version)
    with _index_lock:
        if version not in _indexes:
            _indexes[version] = Index(version, base=store.corpus / "versions")
        return _indexes[version]


@app.middleware("http")
async def check_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        return await call_next(request)
    if (origin is not None and origin not in ALLOWED_ORIGINS) or (request.method == "POST" and origin is None):
        return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
    return await call_next(request)


@app.exception_handler(ApiError)
def api_error(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"detail": exc.message, "problems": exc.problems}, status_code=exc.status)


class SourceRow(BaseModel):
    source_id: str
    decision: str | None
    note: str | None = None


class SourceProgress(BaseModel):
    reviewer: str
    rows: list[SourceRow] = Field(min_length=1, max_length=1000)


class SignRequest(BaseModel):
    reviewer: str
    version: str


class GoldRow(BaseModel):
    question_id: str
    chunk_id: str
    label: str | None
    note: str | None = None


class GoldProgress(BaseModel):
    reviewer: str
    rows: list[GoldRow] = Field(min_length=1, max_length=1000)


class GoldSubmit(BaseModel):
    reviewer: str


P = f"/{API_PATH}"


@app.get(f"{P}/health")
def health() -> dict[str, Any]:
    return {"status": "ready", "service": "pipeline", "corpus": str(store.corpus), "reviewers": service.REVIEWERS}


@app.get(f"{P}/source-runs")
def source_runs() -> list[dict]:
    return service.source_runs(store)


@app.get(f"{P}/source-runs/{{run_id}}")
def source_run(run_id: str) -> dict:
    return service.source_run(store, run_id)


@app.post(f"{P}/source-runs/{{run_id}}/progress")
def source_progress(run_id: str, body: SourceProgress) -> dict:
    return service.save_source_progress(store, run_id, body.reviewer, [r.model_dump() for r in body.rows])


@app.post(f"{P}/source-runs/{{run_id}}/sign")
def sign(run_id: str, body: SignRequest) -> dict:
    return service.sign_source_list(store, run_id, body.reviewer, body.version)


@app.get(f"{P}/source-runs/{{run_id}}/review.csv")
def source_csv(run_id: str) -> PlainTextResponse:
    return PlainTextResponse(service.source_review_csv(store, run_id), media_type="text/csv")


@app.get(f"{P}/source-lists")
def source_lists() -> list[dict]:
    return service.source_lists(store)


@app.get(f"{P}/source-lists/{{name}}")
def source_list(name: str) -> dict:
    return service.source_list(store, f"source_list/{name}")  # name is "v0.1"


@app.get(f"{P}/originals/{{sha256}}")
def original(sha256: str) -> FileResponse:
    return FileResponse(service.original_path(store, sha256), media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


@app.get(f"{P}/corpus")
def corpus_versions() -> list[dict]:
    return service.corpus_versions(store)


@app.get(f"{P}/corpus/{{version}}")
def corpus_version(version: str) -> dict:
    return service.corpus_version(store, version)


@app.get(f"{P}/corpus/{{version}}/gold")
def gold(version: str) -> dict:
    return service.gold_review(store, version)


@app.post(f"{P}/corpus/{{version}}/gold/progress")
def gold_progress(version: str, body: GoldProgress) -> dict:
    return service.save_gold_progress(store, version, body.reviewer, [r.model_dump() for r in body.rows])


@app.post(f"{P}/corpus/{{version}}/gold/submit")
def gold_submit(version: str, body: GoldSubmit) -> dict:
    return service.submit_gold_review(store, version, body.reviewer, index_for)


@app.get(f"{P}/corpus/{{version}}/gold/review.csv")
def gold_csv(version: str) -> PlainTextResponse:
    return PlainTextResponse(service.gold_review_csv(store, version), media_type="text/csv")


@app.get(f"{P}/corpus/{{version}}/search")
def search(version: str, q: str, mode: str = "hybrid") -> dict:
    try:
        return service.search(store, index_for(version), q, mode)
    except ApiError:
        raise
    except Exception as exc:  # e.g. OpenAI unreachable: say so instead of a bare 500
        raise HTTPException(status_code=502, detail=f"search failed: {type(exc).__name__}") from exc
