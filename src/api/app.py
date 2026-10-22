"""FastAPI app: JSON API plus the single-page frontend at http://localhost:8000.

Run (inside the .venv, from the project root):
    uvicorn src.api.app:app --port 8000
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import METADATA_FILE, PROJECT_ROOT, WEB_DIR
from src.metadata import MetadataStore
from src.search.engine import KnowledgeService
from src.storage.mongodb_store import MongoKnowledgeStore, connect_from_environment, sha256_file
from src.textutil import collapse_whitespace

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

logger = logging.getLogger("knowledge_map.api")


class MetadataUpdate(BaseModel):
    title: str | None = None
    author: str | None = None
    year: str | None = None


class AskRequest(BaseModel):
    question: str


class JobManager:
    """Runs one processing job at a time in a background thread and tracks progress."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="km-job")
        self._lock = threading.Lock()

    def submit(self, filename: str, document_id: str, work: Callable[[Callable[[str, float], None]], Any]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "filename": filename, "document_id": document_id, "status": "queued",
               "stage": "queued", "progress": 0.0, "error": None, "created_at": time.time()}
        with self._lock:
            self.jobs[job_id] = job

        def progress(stage: str, fraction: float) -> None:
            job.update(stage=stage, progress=round(max(0.0, min(1.0, fraction)), 3))

        def run() -> None:
            job["status"] = "running"
            try:
                work(progress)
                job.update(status="done", stage="done", progress=1.0)
            except Exception as exc:  # reported to the UI
                logger.exception("job %s failed", job_id)
                job.update(status="failed", error=f"{type(exc).__name__}: {exc}")

        self._executor.submit(run)
        return job

    def for_document(self, document_id: str) -> dict[str, Any] | None:
        matches = [job for job in self.jobs.values() if job["document_id"] == document_id]
        return max(matches, key=lambda job: job["created_at"]) if matches else None


def _highlight_rects(page: Any, quote: str) -> list[Any]:
    """Find the quote on the PDF page; falls back to 5-word pieces when line breaks differ."""
    needle = collapse_whitespace(quote)
    rects = page.search_for(needle) if needle else []
    if rects:
        return rects
    words = needle.split()
    found = []
    for start in range(0, len(words), 5):
        piece = " ".join(words[start:start + 5])
        if len(piece) >= 8:
            found += page.search_for(piece)
    return found


def create_app(store: MongoKnowledgeStore | None = None, metadata: MetadataStore | None = None,
               pipeline_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Build the app. Tests pass a mongomock-backed store and a fake pipeline."""
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        if store is None:
            client, database = connect_from_environment()
            state["store"] = MongoKnowledgeStore(database, PROJECT_ROOT)
        else:
            state["store"] = store
        state["metadata"] = metadata or MetadataStore(METADATA_FILE)
        state["service"] = KnowledgeService(state["store"], state["metadata"])
        state["jobs"] = JobManager()
        state["pipeline"] = None
        yield
        if client is not None:
            client.close()

    app = FastAPI(title="Knowledge Map", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    def service() -> KnowledgeService:
        return state["service"]

    def get_pipeline() -> Any:
        if state["pipeline"] is None:
            if pipeline_factory is not None:
                state["pipeline"] = pipeline_factory()
            else:
                from src.pipeline import Pipeline
                state["pipeline"] = Pipeline(state["store"], state["metadata"])
        return state["pipeline"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            state["store"].database.command("ping")
            mongo = "ok"
        except Exception as exc:  # pragma: no cover - depends on server
            mongo = f"error: {exc}"
        return {"status": "ok", "mongodb": mongo, "cache_loaded_at": service().loaded_at}

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return service().stats()

    @app.get("/api/documents")
    def documents() -> list[dict[str, Any]]:
        store_ = state["store"]
        rows = []
        known = set()
        for document in store_.documents.find({}, {"filename": 1, "page_count": 1}):
            known.add(document["_id"])
            meta = state["metadata"].get(document["_id"])
            pages = store_.content_units.count_documents({"document_id": document["_id"], "unit_type": "page"})
            job = state["jobs"].for_document(document["_id"])
            status = job["status"] if job and job["status"] in ("queued", "running", "failed") else (
                "processed" if pages == document.get("page_count") else "incomplete")
            rows.append({
                "id": document["_id"], "filename": document["filename"], "pages": document.get("page_count", 0),
                "images": store_.content_units.count_documents({"document_id": document["_id"], "unit_type": "image"}),
                "concepts": store_.concepts.count_documents({"document_ids": document["_id"]}),
                "title": meta.get("title"), "author": meta.get("author"), "year": meta.get("year"),
                "detected_from": meta.get("detected_from", {}), "user_edited": meta.get("user_edited", []),
                "status": status, "stage": job["stage"] if job else None,
                "progress": job["progress"] if job else 1.0, "error": job["error"] if job else None,
            })
        for job in state["jobs"].jobs.values():
            if job["document_id"] not in known:
                rows.append({"id": job["document_id"], "filename": job["filename"], "pages": 0, "images": 0,
                             "concepts": 0, "title": None, "author": None, "year": None, "detected_from": {},
                             "user_edited": [], "status": job["status"], "stage": job["stage"],
                             "progress": job["progress"], "error": job["error"]})
        return rows

    @app.put("/api/documents/{document_id}/metadata")
    def update_metadata(document_id: str, body: MetadataUpdate) -> dict[str, Any]:
        if state["store"].documents.find_one({"_id": document_id}, {"_id": 1}) is None:
            raise HTTPException(404, "Unknown document")
        changes = {key: value for key, value in body.model_dump().items() if value is not None}
        return state["metadata"].update(document_id, changes)

    @app.post("/api/documents/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = Path(file.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are accepted")
        temp_dir = Path(tempfile.mkdtemp(prefix="km-upload-"))
        temp_path = temp_dir / filename
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        with temp_path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(400, "The file is not a PDF")
        document_id = sha256_file(temp_path)

        def work(progress: Callable[[str, float], None]) -> None:
            try:
                pipeline = get_pipeline()
                pipeline.ingest_file(temp_path, filename, progress)
                pipeline.knowledge_stages(progress)
                progress("refreshing caches", 0.99)
                service().load()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return state["jobs"].submit(filename, document_id, work)

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return sorted(state["jobs"].jobs.values(), key=lambda job: -job["created_at"])

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        if job_id not in state["jobs"].jobs:
            raise HTTPException(404, "Unknown job")
        return state["jobs"].jobs[job_id]

    @app.get("/api/concepts/autocomplete")
    def autocomplete(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)) -> list[dict[str, Any]]:
        return service().autocomplete(q, limit)

    @app.get("/api/concepts/search")
    def concept_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> list[dict]:
        found = [service().concept_summary(concept_id) for concept_id in service().match_concepts(q)]
        ids = {item["id"] for item in found}
        found += [item for item in service().autocomplete(q, limit) if item["id"] not in ids]
        return found[:limit]

    @app.get("/api/concepts/{concept_id}")
    def concept(concept_id: str) -> dict[str, Any]:
        detail = service().concept_detail(concept_id)
        if detail is None:
            raise HTTPException(404, "Unknown concept")
        return detail

    @app.get("/api/graph")
    def graph(document_id: str | None = None, type: str | None = None,
              min_weight: float = Query(0.0, ge=0), limit: int = Query(250, ge=1, le=3000),
              focus: str | None = None) -> dict[str, Any]:
        return service().graph_payload(document_id or None, type or None, min_weight, limit, focus)

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50),
               document_id: str | None = None) -> dict[str, Any]:
        return service().search(q, limit, document_id or None)

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Question is empty")
        return service().ask(question)

    @app.get("/api/pages/{unit_id}")
    def page(unit_id: str) -> dict[str, Any]:
        record = service().pages.get(unit_id)
        if record is None:
            raise HTTPException(404, "Unknown page")
        return {"id": unit_id, **record, "citation": service().citation(record["document_id"], record["page_number"])}

    @app.get("/api/pages/{unit_id}/image")
    def page_image(unit_id: str, quote: str | None = None, zoom: float = Query(1.6, ge=0.5, le=4.0)) -> Response:
        record = service().pages.get(unit_id)
        if record is None or fitz is None:
            raise HTTPException(404, "Unknown page")
        document = state["store"].documents.find_one({"_id": record["document_id"]}, {"file_storage": 1})
        path = PROJECT_ROOT / document["file_storage"]["uri"] if document else None
        if path is None or not path.is_file():
            raise HTTPException(404, "Source PDF not found")
        with fitz.open(path) as pdf:
            pdf_page = pdf[record["page_number"] - 1]
            if quote:
                for rect in _highlight_rects(pdf_page, quote):
                    pdf_page.add_highlight_annot(rect)
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=True)
            return Response(pixmap.tobytes("png"), media_type="image/png")

    @app.get("/api/images/{unit_id}")
    def image(unit_id: str) -> FileResponse:
        unit = state["store"].content_units.find_one({"_id": unit_id, "unit_type": "image"}, {"image_uri": 1})
        path = PROJECT_ROOT / unit["image_uri"] if unit and unit.get("image_uri") else None
        if path is None or not path.is_file():
            raise HTTPException(404, "Unknown image")
        return FileResponse(path)

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
