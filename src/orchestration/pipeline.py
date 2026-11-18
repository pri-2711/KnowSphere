"""Pipeline stages for one project: ingest PDFs (no model), extract and merge GLiNER concepts, build
relationships, the keyword index and the concept-first graph, then run governance. Every stage is
idempotent and recorded in ``stage_runs``. The command line interface is ``src.api.cli``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from src.config import CAPTION_MODEL, EXTRACTED_DIR, IMAGE_DIR, RAW_PDF_DIR, REPORT_DIR, UPLOAD_DIR
from src.knowledge_repository.projects import DEFAULT_PROJECT, is_default
from src.document_intelligence import text_formats
from src.orchestration.documents import (delete_document, ingest_pdf, ingest_text_file, is_tombstoned, remove_source,
                                         stage_upload)
from src.knowledge_extraction.image_links import link_images_to_concepts
from src.knowledge_extraction.gliner_concepts import extract_concepts
from src.model_runtime.gliner import GlinerExtractor
from src.knowledge_extraction.merging import merge_concepts
from src.model_runtime.vision import VitCaptioner
from src.knowledge_extraction.relationships import store_relationships
from src.knowledge_repository.metadata import MetadataStore
from src.evidence_quality.service import validate_project
from src.knowledge_repository.canonical import GLINER
from src.knowledge_repository.repository import MongoKnowledgeRepository
from src.model_runtime.router import ModelRouter
from src.orchestration.kg_pipeline import KGPipeline
from src.orchestration.stages import StageTracker
from src.retrieval.lexical_index import build_index
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore, sha256_file

logger = logging.getLogger("knowledge_map.pipeline")
Progress = Callable[[str, float], None]


def _noop(stage: str, fraction: float) -> None:
    return None


class Pipeline:
    """Runs the stages in order. Every stage is idempotent and skips unchanged input."""

    def __init__(self, store: MongoKnowledgeStore, metadata: MetadataStore, caption_images: bool = False,
                 extractor: GlinerExtractor | None = None, embedder: Any = None,
                 upload_dir: Path = UPLOAD_DIR, image_dir: Path = IMAGE_DIR, report_dir: Path = REPORT_DIR,
                 router: ModelRouter | None = None) -> None:
        self.store = store
        self.metadata = metadata
        self.router = router or ModelRouter()
        self.repository = MongoKnowledgeRepository(store, metadata)
        self.tracker = StageTracker(store.database)
        self.captioner = VitCaptioner(CAPTION_MODEL) if caption_images else None
        self._extractor = extractor
        self._embedder = embedder
        self.upload_dir = upload_dir
        self.image_dir = image_dir
        self.report_dir = report_dir

    @property
    def extractor(self) -> Any:
        """The entity span model (GLiNER) from the router, or None when that optional stage is off."""
        if self._extractor is None:
            self._extractor = self.router.span_model()
        return self._extractor

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = self.router.embedder()
        return self._embedder

    def ingest_raw_books(self, progress: Progress = _noop) -> list[dict[str, Any]]:
        """Ingest data/raw_pdfs (read only). Books deleted in the app stay deleted."""
        results = []
        for pdf in sorted(RAW_PDF_DIR.glob("*.pdf"), key=lambda path: path.name.casefold()):
            document_id = sha256_file(pdf)
            if is_tombstoned(self.store, document_id):
                results.append({"document_id": document_id, "filename": pdf.name, "skipped": "deleted in the app"})
                continue
            results.append(ingest_pdf(self.store, self.metadata, pdf, EXTRACTED_DIR, self.image_dir, self.captioner,
                                      progress, source=f"raw_pdfs/{pdf.name}"))
        return results

    def ingest_file(self, source: Path, filename: str | None = None, progress: Progress = _noop,
                    source_label: str = "upload", project_id: str = DEFAULT_PROJECT,
                    previous_document_id: str | None = None) -> dict[str, Any]:
        """Add a PDF to one project (upload, CLI or watched folder). Copied to data/uploads/<document id>/."""
        pdf_path, _ = stage_upload(source, filename or source.name, self.upload_dir, project_id)
        if text_formats.supported(pdf_path):          # DOCX, HTML, Markdown: logical pages, no images
            progress("reading document", 0.1)
            result = ingest_text_file(self.store, self.metadata, pdf_path, pdf_path.parent / "extracted",
                                      source_label, project_id)
        else:
            result = ingest_pdf(self.store, self.metadata, pdf_path, pdf_path.parent / "extracted", self.image_dir,
                                self.captioner, progress, source=source_label, project_id=project_id)
        self.record_version(result["document_id"], project_id, pdf_path.name, previous_document_id)
        return result

    def record_version(self, document_id: str, project_id: str, filename: str,
                       previous_document_id: str | None = None) -> None:
        document = self.store.documents.find_one({"_id": document_id}, {"source_sha256": 1})
        if document is not None:
            self.repository.record_version(project_id, document_id, document.get("source_sha256") or document_id,
                                           filename, previous_document_id, parser="pymupdf")

    def project_of(self, document_id: str) -> str:
        document = self.store.documents.find_one({"_id": document_id}, {"project_id": 1})
        return (document or {}).get("project_id") or DEFAULT_PROJECT

    def delete(self, document_id: str, progress: Progress = _noop) -> dict[str, Any]:
        """Delete a document and all derived records; it will not be re-added automatically."""
        progress("deleting document", 0.1)
        return delete_document(self.store, self.metadata, document_id, self.upload_dir, self.image_dir)

    def replace(self, document_id: str, source: Path, filename: str | None = None, progress: Progress = _noop,
                source_label: str = "upload") -> dict[str, Any]:
        """Add a new version of a document (same project), keep the user's metadata edits, delete the old one."""
        project_id = self.project_of(document_id)
        added = self.ingest_file(source, filename, progress, source_label, project_id, previous_document_id=document_id)
        new_id = added["document_id"]
        if new_id != document_id:
            self.metadata.carry_over_edits(document_id, new_id)
            progress("deleting old version", 0.42)
            added["replaced"] = self.delete(document_id)
        return added

    def remove_source(self, document_id: str, source: str, progress: Progress = _noop) -> dict[str, Any]:
        progress("removing source", 0.1)
        return remove_source(self.store, self.metadata, document_id, source, self.upload_dir, self.image_dir)

    def knowledge_stages(self, progress: Progress = _noop, show_bar: bool = False,
                         project_id: str = DEFAULT_PROJECT) -> dict[str, Any]:
        """Rebuild one project's concepts, links, relationships, index and concept-first graph."""
        timings: dict[str, Any] = {}
        report = "merge_report.csv" if is_default(project_id) else f"merge_report_{project_id}.csv"

        def timed(name: str, fraction: float, action: Callable[[], Any]) -> Any:
            progress(name, fraction)
            started = time.perf_counter()
            output = self.tracker.run(project_id, name, action)
            timings[name] = {"seconds": round(time.perf_counter() - started, 1), "result": output}
            logger.info("%s finished: %s", name, output)
            return output

        def concept_progress(done: int, total: int) -> None:
            progress(f"finding concepts ({done}/{total} pages)", 0.45 + 0.35 * (done / total if total else 1))

        if self.extractor is not None:      # GLiNER is optional (KM_USE_GLINER=0 turns this graph off)
            timed("concepts", 0.45, lambda: extract_concepts(self.store, self.extractor, progress=concept_progress,
                                                              show_bar=show_bar, project_id=project_id))
            timed("merge", 0.8, lambda: merge_concepts(self.store, self.embedder, self.report_dir / report,
                                                       project_id=project_id))
            timed("link_images", 0.88, lambda: link_images_to_concepts(self.store, project_id=project_id))
            timed("relations", 0.9, lambda: store_relationships(self.store, project_id))
            timed("validate", 0.92, lambda: validate_project(self.repository, project_id, GLINER))
        timed("index", 0.96, lambda: build_index(self.store, project_id))
        # Concept-first graph: rebuilt from stored model extractions only (no model calls), so a deleted or
        # replaced document also leaves that graph. New text is extracted with `python -m src.orchestration.kg_pipeline`.
        timed("kg_graph", 0.98, lambda: KGPipeline(self.store, embedder=self.embedder, metadata=self.metadata,
                                                    report_dir=self.report_dir).build_graph(project_id))
        progress("done", 1.0)
        return timings
