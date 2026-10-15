"""HTTP API and web UI. Run with: uvicorn docqa.api:create_app --factory"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .ingest import ingest_document, ingest_paths
from .llm import Answerer, ClaudeAnswerer, MissingCredentialsError
from .loaders import UnsupportedFileType, extract_text
from .retriever import BM25Index

log = logging.getLogger("docqa")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    n: int
    source: str
    chunk_id: str
    quote: str


class RetrievedOut(BaseModel):
    chunk_id: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]
    model: str | None
    latency_ms: int


def create_app(settings: Settings | None = None, answerer: Answerer | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    index = BM25Index.load(settings.index_path)
    if not len(index) and settings.seed_dir:
        counts = ingest_paths(index, [settings.seed_dir])
        index.save(settings.index_path)
        log.info("seeded index from %s: %s", settings.seed_dir, counts)
    answerer = answerer or ClaudeAnswerer(settings.model, settings.effort)
    # Sync endpoints run in a threadpool; uploads and deletes rebuild the index in place.
    lock = threading.Lock()

    app = FastAPI(
        title="DocQA",
        version=__version__,
        description="Question answering over company documents, with citations.",
    )

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(index), "model": settings.model}

    @app.get("/documents")
    def list_documents() -> dict[str, int]:
        return index.sources()

    @app.post("/documents", status_code=201)
    def upload_documents(files: list[UploadFile]) -> dict[str, int]:
        texts = {}
        for f in files:
            data = f.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename} is larger than 10 MB")
            try:
                texts[f.filename] = extract_text(f.filename or "", data)
            except UnsupportedFileType as e:
                raise HTTPException(415, str(e)) from e
        counts = {}
        with lock:
            for name, text in texts.items():
                try:
                    counts[name] = ingest_document(index, name, text)
                except ValueError as e:
                    raise HTTPException(422, str(e)) from e
            index.save(settings.index_path)
        return counts

    @app.delete("/documents/{source}")
    def delete_document(source: str) -> dict[str, int]:
        with lock:
            removed = index.remove_source(source)
            if not removed:
                raise HTTPException(404, f"{source} is not indexed")
            index.save(settings.index_path)
        return {"removed_chunks": removed}

    @app.post("/ask")
    def ask(req: AskRequest) -> AskResponse:
        start = time.perf_counter()
        hits = index.search(req.question, k=req.top_k or settings.top_k)
        try:
            result = answerer.answer(req.question, [h.chunk for h in hits])
        except MissingCredentialsError as e:
            raise HTTPException(503, str(e)) from e
        except anthropic.AuthenticationError as e:
            raise HTTPException(503, "Claude API key is missing or invalid") from e
        except anthropic.RateLimitError as e:
            raise HTTPException(429, "Rate limited by the Claude API; retry shortly") from e
        except anthropic.APIConnectionError as e:
            raise HTTPException(502, "Could not reach the Claude API") from e
        except anthropic.APIStatusError as e:
            log.warning("Claude API error %s (request %s)", e.status_code, e.request_id)
            raise HTTPException(502, f"Claude API error ({e.status_code})") from e
        return AskResponse(
            answer=result.text,
            citations=[
                CitationOut(n=i, source=c.source, chunk_id=c.chunk_id, quote=c.quote)
                for i, c in enumerate(result.citations, start=1)
            ],
            retrieved=[
                RetrievedOut(chunk_id=h.chunk.id, source=h.chunk.source, score=round(h.score, 3))
                for h in hits
            ],
            model=result.model,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

    return app
