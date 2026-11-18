"""FastAPI app: JSON API plus the single-page frontend at http://localhost:8000.

Run (inside the .venv, from the project root):
    uvicorn src.api.app:app --port 8000

This module only translates HTTP to ``orchestration.services.KnowledgeApplication``
calls (the same services the CLI uses) and runs long work as background jobs.
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
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import METADATA_FILE, PROJECT_ROOT, WATCH_DIR, WATCH_ENABLED, WATCH_INTERVAL, WEB_DIR
from src.envcheck import warn_if_needed
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore, connect_from_environment, sha256_file
from src.knowledge_repository.projects import DEFAULT_PROJECT, ProjectError
from src.orchestration.documents import ACCEPTED_FORMATS, accepted_name, upload_problem
from src.orchestration.services import KnowledgeApplication, NotFound, ProjectBusy
from src.orchestration.watcher import FolderWatcher, WatchService, process_events
from src.retrieval.engine import KnowledgeService
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


class ProjectBody(BaseModel):
    name: str


class AskRequest(BaseModel):
    question: str


class AdjudicationBody(BaseModel):
    reviewer: str
    decision: Literal["accept", "reject"]
    note: str = ""


class ReviewerNote(BaseModel):
    reviewer: str
    note: str = ""


class MergeBody(BaseModel):
    a: str                                  # concept ID or name
    b: str
    decision: Literal["same", "different"]
    reviewer: str
    note: str = ""


class OntologyBody(BaseModel):
    actor: str
    types: list[dict[str, Any]]


class ProposalBody(BaseModel):
    agent: str
    kind: Literal["concept", "relationship", "claim"]
    unit_id: str
    data: dict[str, Any]


class JobManager:
    """Runs one processing job at a time in a background thread and tracks progress."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="km-job")
        self._lock = threading.Lock()

    def submit(self, filename: str, document_id: str | None, work: Callable[[Callable[[str, float], None]], Any],
               kind: str = "upload", project_id: str | None = DEFAULT_PROJECT) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "kind": kind, "project_id": project_id, "filename": filename,
               "document_id": document_id, "status": "queued",
               "stage": "queued", "progress": 0.0, "error": None, "created_at": time.time(), "result": None}
        with self._lock:
            self.jobs[job_id] = job

        def progress(stage: str, fraction: float) -> None:
            job.update(stage=stage, progress=round(max(0.0, min(1.0, fraction)), 3))

        def run() -> None:
            job["status"] = "running"
            try:
                result = work(progress)
                job.update(status="done", stage="done", progress=1.0,
                           result=result if isinstance(result, (dict, list)) else None)
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
               pipeline_factory: Callable[[], Any] | None = None,
               watch_folder: Path | None = WATCH_DIR if WATCH_ENABLED else None,
               watch_interval: float = WATCH_INTERVAL) -> FastAPI:
    """Build the app. Tests pass a mongomock-backed store, a fake pipeline and their own watch folder."""
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        if store is None:
            warn_if_needed()
            client, database = connect_from_environment()
            active_store = MongoKnowledgeStore(database, PROJECT_ROOT)
        else:
            active_store = store
        state["app"] = KnowledgeApplication(active_store, metadata or MetadataStore(METADATA_FILE),
                                            pipeline_factory)
        state["jobs"] = JobManager()
        state["watch"] = None
        if watch_folder is not None:
            watch = WatchService(FolderWatcher(active_store, watch_folder), submit_watch_events, watch_interval)
            state["watch"] = watch
            watch.start()
        yield
        if state["watch"] is not None:
            state["watch"].stop()
        if client is not None:
            client.close()

    app = FastAPI(title="Knowledge Map", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.middleware("http")
    async def revalidate_frontend(request: Any, call_next: Callable[[Any], Any]) -> Any:
        """Make browsers re-check the page, CSS and JS so they never keep a stale copy (ETag keeps it cheap)."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def services() -> KnowledgeApplication:
        return state["app"]

    def call(action: Callable[[], Any]) -> Any:
        """Run a service call and map its errors to HTTP status codes."""
        try:
            return action()
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ProjectError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectBusy as exc:
            raise HTTPException(409, str(exc)) from exc

    def retriever(project: str) -> KnowledgeService:
        return call(lambda: services().retriever(project))

    def submit_watch_events(events: list[Any]) -> dict[str, Any]:
        watch = state["watch"]

        def work(progress: Callable[[str, float], None]) -> None:
            try:
                process_events(services().pipeline, watch.watcher, events, progress)
                progress("refreshing caches", 0.99)
                for project in {event.project_id for event in events}:
                    services().reload(project)
            finally:
                watch.busy.clear()

        label = ", ".join(f"{event.kind} {event.relative_path}" for event in events)
        return state["jobs"].submit(f"watched folder: {label}"[:200], None, work, kind="watch", project_id=None)

    def job(kind: str, label: str, document_id: str | None, project: str | None,
            work: Callable[[Callable[[str, float], None]], Any], temp_dir: Path | None = None) -> dict:
        def run(progress: Callable[[str, float], None]) -> Any:
            try:
                return work(progress)
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)
        return state["jobs"].submit(label, document_id, run, kind=kind, project_id=project)

    async def save_upload(file: UploadFile) -> tuple[Path, Path, str, str]:
        """Store an uploaded PDF, DOCX, HTML or Markdown file in a temporary folder; returns
        (folder, path, filename, sha256). The content must match the file type."""
        filename = Path(file.filename or "document.pdf").name
        if not accepted_name(filename):
            raise HTTPException(400, f"Accepted files: {ACCEPTED_FORMATS}")
        temp_dir = Path(tempfile.mkdtemp(prefix="km-upload-"))
        temp_path = temp_dir / filename
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        problem = upload_problem(temp_path)
        if problem:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(400, problem)
        return temp_dir, temp_path, filename, sha256_file(temp_path)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            services().store.database.command("ping")
            mongo = "ok"
        except Exception as exc:  # pragma: no cover - depends on server
            mongo = f"error: {exc}"
        return {"status": "ok", "mongodb": mongo, "cache_loaded_at": retriever(DEFAULT_PROJECT).loaded_at}

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        """Which model serves which capability (nothing is loaded by this call)."""
        router = getattr(services().pipeline, "router", None)
        return router.describe() if router is not None else {}

    # ------------------------------------------------------------------ projects
    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return services().list_projects()

    @app.post("/api/projects")
    def create_project(body: ProjectBody) -> dict[str, Any]:
        return call(lambda: services().create_project(body.name))

    @app.put("/api/projects/{project_id}")
    def rename_project(project_id: str, body: ProjectBody) -> dict[str, Any]:
        return call(lambda: services().rename_project(project_id, body.name))

    @app.delete("/api/projects/{project_id}")
    def delete_project_endpoint(project_id: str) -> dict[str, Any]:
        """Delete a project with all its documents and graph (background job). Other projects are untouched."""
        project = call(lambda: services().project(project_id))
        if project_id == DEFAULT_PROJECT:
            raise HTTPException(400, "The default project cannot be deleted")
        return job("project_delete", project["name"], None, project_id,
                   lambda progress: services().delete_project(project_id, progress))

    @app.post("/api/projects/{project_id}/rebuild")
    def rebuild_project(project_id: str) -> dict[str, Any]:
        """Rebuild one project's graph, index and governance from stored extractions (no model calls)."""
        project = call(lambda: services().project(project_id))
        return job("rebuild", project["name"], None, project_id, lambda progress: services().rebuild(project_id, progress))

    @app.post("/api/projects/{project_id}/validate")
    def validate_endpoint(project_id: str, source: str | None = None) -> list[dict[str, Any]]:
        return call(lambda: services().validate(project_id, source))

    @app.get("/api/projects/{project_id}/stages")
    def stages(project_id: str) -> dict[str, Any]:
        return call(lambda: services().stage_status(project_id))

    # ------------------------------------------------------------------ documents
    @app.get("/api/stats")
    def stats(project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return retriever(project).stats()

    @app.get("/api/documents")
    def documents(project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        rows = call(lambda: services().list_documents(project, state["jobs"].for_document))
        known = {row["id"] for row in rows}
        for queued in state["jobs"].jobs.values():
            # Uploads that are still being processed are shown before their document exists.
            if queued["kind"] == "upload" and queued.get("project_id") == project and queued["document_id"] not in known \
                    and queued["status"] != "done":
                rows.append({"id": queued["document_id"], "filename": queued["filename"], "pages": 0, "images": 0,
                             "concepts": 0, "title": None, "author": None, "year": None, "detected_from": {},
                             "user_edited": [], "sources": [], "structure": None, "status": queued["status"],
                             "stage": queued["stage"], "progress": queued["progress"], "error": queued["error"]})
        return rows

    @app.put("/api/documents/{document_id}/metadata")
    def update_metadata(document_id: str, body: MetadataUpdate, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().update_metadata(project, document_id, body.model_dump()))

    @app.get("/api/documents/{document_id}/versions")
    def versions(document_id: str, project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().document_versions(project, document_id))

    @app.post("/api/documents/{document_id}/structure")
    def structure(document_id: str, project: str = Query(DEFAULT_PROJECT),
                  parser: str | None = Query(None, pattern="^(auto|docling|pymupdf)$")) -> dict[str, Any]:
        """Parse sections, tables, figures and equations (background job; stored page text is unchanged)."""
        document = call(lambda: services().document(project, document_id))
        return job("structure", document["filename"], document_id, project,
                   lambda progress: services().parse_structure(project, document_id, parser))

    @app.post("/api/documents/upload")
    async def upload(file: UploadFile = File(...), project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        """Add a document to the project."""
        call(lambda: services().project(project))
        temp_dir, temp_path, filename, sha = await save_upload(file)
        return job("upload", filename, services().upload_document_id(project, sha), project,
                   lambda progress: services().ingest(project, temp_path, filename, progress), temp_dir)

    @app.put("/api/documents/{document_id}/file")
    async def replace_file(document_id: str, file: UploadFile = File(...),
                           project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        """Edit a document by uploading a new version of its PDF; it stays in its project, metadata edits carry over."""
        call(lambda: services().document(project, document_id))
        temp_dir, temp_path, filename, _ = await save_upload(file)
        return job("replace", filename, document_id, project,
                   lambda progress: services().replace(project, document_id, temp_path, filename, progress), temp_dir)

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        """Delete a document and everything derived from it, then rebuild that project's concepts, edges and index."""
        document = call(lambda: services().document(project, document_id))
        return job("delete", document["filename"], document_id, project,
                   lambda progress: services().delete_document(project, document_id, progress))

    @app.get("/api/watch")
    def watch_status() -> dict[str, Any]:
        watch = state["watch"]
        return watch.status() if watch is not None else {"enabled": False}

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return sorted(state["jobs"].jobs.values(), key=lambda item: -item["created_at"])

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        if job_id not in state["jobs"].jobs:
            raise HTTPException(404, "Unknown job")
        return state["jobs"].jobs[job_id]

    # ------------------------------------------------------------------ knowledge governance
    @app.get("/api/knowledge/summary")
    def knowledge_summary(project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().knowledge_summary(project))

    @app.get("/api/knowledge/{item_id}/validation")
    def validation(item_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().validation_record(project, item_id))

    @app.post("/api/knowledge/{item_id}/adjudicate")
    def adjudicate(item_id: str, body: AdjudicationBody, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().adjudicate(project, item_id, body.reviewer, body.decision, body.note))

    # ------------------------------------------------------------------ retrieval
    @app.get("/api/concepts/autocomplete")
    def autocomplete(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50),
                     project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return retriever(project).autocomplete(q, limit)

    @app.get("/api/concepts/search")
    def concept_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100),
                       project: str = Query(DEFAULT_PROJECT)) -> list[dict]:
        svc = retriever(project)
        found = [svc.concept_summary(concept_id) for concept_id in svc.match_concepts(q)]
        ids = {item["id"] for item in found}
        found += [item for item in svc.autocomplete(q, limit) if item["id"] not in ids]
        return found[:limit]

    @app.get("/api/concepts/{concept_id}")
    def concept(concept_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        detail = retriever(project).concept_detail(concept_id)   # None for other projects' concepts
        if detail is None:
            raise HTTPException(404, "Unknown concept")
        return detail

    @app.get("/api/graph")
    def graph(document_id: str | None = None, type: str | None = None,
              min_weight: float = Query(0.0, ge=0), limit: int = Query(250, ge=1, le=3000),
              focus: str | None = None, source: str | None = None,
              project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        """The map: ``source`` gliner (default, as before) or concept_first (governed concepts and claims)."""
        return call(lambda: services().graph_payload(project, source or "gliner", document_id or None, type or None,
                                                     min_weight, limit, focus))

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50),
               document_id: str | None = None, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return retriever(project).search(q, limit, document_id or None)

    @app.post("/api/ask")
    def ask(body: AskRequest, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        svc = retriever(project)
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Question is empty")
        return svc.ask(question)

    @app.get("/api/pages/{unit_id}")
    def page(unit_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        svc = retriever(project)
        record = svc.pages.get(unit_id)          # only this project's pages are loaded
        if record is None:
            raise HTTPException(404, "Unknown page")
        return {"id": unit_id, **record, "citation": svc.citation(record["document_id"], record["page_number"])}

    @app.get("/api/pages/{unit_id}/image")
    def page_image(unit_id: str, quote: str | None = None, zoom: float = Query(1.6, ge=0.5, le=4.0),
                   project: str = Query(DEFAULT_PROJECT)) -> Response:
        if fitz is None:
            raise HTTPException(404, "Unknown page")
        path, page_number = call(lambda: services().page_source(project, unit_id))
        with fitz.open(path) as pdf:
            pdf_page = pdf[page_number - 1]
            if quote:
                for rect in _highlight_rects(pdf_page, quote):
                    pdf_page.add_highlight_annot(rect)
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=True)
            return Response(pixmap.tobytes("png"), media_type="image/png")

    @app.get("/api/images/{unit_id}")
    def image(unit_id: str, project: str = Query(DEFAULT_PROJECT)) -> FileResponse:
        return FileResponse(call(lambda: services().image_file(project, unit_id)))

    # ------------------------------------------------------------------ hybrid retrieval and reasoning
    @app.get("/api/projects/{project_id}/context")
    def project_context(project_id: str) -> dict[str, Any]:
        return call(lambda: services().project_context(project_id))

    @app.get("/api/search/hybrid")
    def hybrid_search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50),
                      source: str | None = None, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().search_hybrid(project, q.strip(), limit, source))

    @app.post("/api/research")
    def research(body: AskRequest, source: str | None = None, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Question is empty")
        return call(lambda: services().research(project, question, source))

    @app.post("/api/projects/{project_id}/semantic-index")
    def semantic_index(project_id: str) -> dict[str, Any]:
        """Embed the project's passages with the local embedding model (background job)."""
        project = call(lambda: services().project(project_id))
        return job("semantic_index", project["name"], None, project_id,
                   lambda progress: services().build_semantic_index(project_id))

    @app.post("/api/projects/{project_id}/concept-graph")
    def concept_graph(project_id: str) -> dict[str, Any]:
        """Rebuild the concept-first graph from stored extractions and proposals (no model call)."""
        project = call(lambda: services().project(project_id))
        return job("concept_graph", project["name"], None, project_id,
                   lambda progress: services().rebuild_concept_graph(project_id))

    # ------------------------------------------------------------------ knowledge exploration
    @app.get("/api/knowledge/concepts")
    def knowledge_concepts(q: str = Query(..., min_length=1), source: str | None = None,
                           project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().find_concepts(project, q, source))

    @app.get("/api/knowledge/concepts/{concept_id}")
    def knowledge_concept(concept_id: str, source: str | None = None,
                          project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().concept_knowledge(project, concept_id, source))

    @app.get("/api/knowledge/graph/explore")
    def explore(concept: list[str] = Query(...), hops: int = Query(1, ge=1, le=3), source: str | None = None,
                project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().explore(project, concept, hops, source))

    @app.get("/api/knowledge/graph/path")
    def path(a: str, b: str, source: str | None = None, max_hops: int = Query(4, ge=1, le=6),
             project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().connection(project, a, b, source, max_hops))

    @app.get("/api/knowledge/claims")
    def claims(concept: str | None = None, kind: str | None = None, modality: str | None = None,
               project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().claims(project, concept, kind, modality))

    @app.get("/api/knowledge/conflicts")
    def conflicts(source: str | None = None, project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().conflicts(project, source))

    @app.get("/api/knowledge/items/{item_id}/evidence")
    def item_evidence(item_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().evidence(project, item_id))

    # ------------------------------------------------------------------ governance and review
    @app.get("/api/knowledge/review")
    def review_queue(project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().review_queue(project))

    @app.post("/api/knowledge/{item_id}/revoke")
    def revoke(item_id: str, body: ReviewerNote, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().revoke_adjudication(project, item_id, body.reviewer, body.note))

    @app.get("/api/knowledge/merges")
    def merges(project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().merge_decisions(project))

    @app.post("/api/knowledge/merges")
    def merge(body: MergeBody, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().review_merge(project, body.a, body.b, body.decision, body.reviewer, body.note))

    @app.post("/api/knowledge/merges/{decision_id}/revert")
    def revert_merge(decision_id: str, body: ReviewerNote, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().revert_merge(project, decision_id, body.reviewer, body.note))

    @app.get("/api/ontology")
    def ontology(project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().ontology(project))

    @app.post("/api/ontology")
    def extend_ontology(body: OntologyBody, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().extend_ontology(project, body.types, body.actor))

    @app.get("/api/proposals")
    def proposals(status: str | None = None, agent: str | None = None,
                  project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().proposals(project, status, agent))

    @app.post("/api/proposals")
    def propose(body: ProposalBody, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().propose(project, body.agent, body.kind, body.unit_id, body.data))

    @app.post("/api/proposals/{proposal_id}/withdraw")
    def withdraw(proposal_id: str, body: ReviewerNote, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().withdraw_proposal(project, proposal_id, body.reviewer))

    @app.get("/api/history")
    def history(item: str | None = None, actor: str | None = None, limit: int = Query(100, ge=1, le=1000),
                project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().history(project, item, actor, limit))

    @app.get("/api/projects/{project_id}/diagnostics")
    def diagnostics(project_id: str, probe_models: bool = False) -> dict[str, Any]:
        return call(lambda: services().diagnostics(project_id, probe_models))

    @app.get("/api/agents/activity")
    def agent_activity(agent: str | None = None, limit: int = Query(100, ge=1, le=1000),
                       project: str = Query(DEFAULT_PROJECT)) -> list[dict[str, Any]]:
        return call(lambda: services().agent_activity(project, agent, limit))

    @app.get("/api/documents/{document_id}/quality")
    def document_quality(document_id: str, project: str = Query(DEFAULT_PROJECT)) -> dict[str, Any]:
        return call(lambda: services().document_quality(project, document_id))

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
