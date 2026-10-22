"""Ingest PDF files: existing extraction, MongoDB storage, metadata detection, and images.

A document is identified by the SHA-256 of its PDF bytes. When that hash is
already stored with all its pages, extraction and ingestion are skipped.
Uploaded files are kept in ``data/uploads/<hash prefix>/``; ``data/raw_pdfs`` is
only read.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from src import pdf_extraction
from src.config import IMAGE_DIR, UPLOAD_DIR
from src.ingest.images import Captioner, extract_images
from src.metadata import MetadataStore, detect_metadata
from src.storage.mongodb_store import MongoKnowledgeStore, sha256_file

logger = logging.getLogger("knowledge_map.ingest")
Progress = Callable[[str, float], None]


def _noop(stage: str, fraction: float) -> None:
    return None


def is_ingested(store: MongoKnowledgeStore, document_id: str) -> bool:
    document = store.documents.find_one({"_id": document_id}, {"page_count": 1})
    if not document:
        return False
    pages = store.content_units.count_documents({"document_id": document_id, "unit_type": "page"})
    return pages > 0 and pages == document.get("page_count")


def stage_upload(source: Path, filename: str, upload_root: Path = UPLOAD_DIR) -> tuple[Path, str]:
    """Copy an uploaded file into its own folder named by hash. Returns (pdf path, document id)."""
    document_id = sha256_file(source)
    folder = upload_root / document_id[:16]
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    target = folder / safe_name
    if not target.exists():
        shutil.copyfile(source, target)
    return target, document_id


def ingest_pdf(
    store: MongoKnowledgeStore,
    metadata: MetadataStore,
    pdf_path: Path,
    extracted_dir: Path,
    image_root: Path = IMAGE_DIR,
    captioner: Captioner | None = None,
    progress: Progress = _noop,
) -> dict[str, Any]:
    """Extract (only if needed), store pages, detect metadata, and extract images for one PDF."""
    document_id = sha256_file(pdf_path)
    result: dict[str, Any] = {"document_id": document_id, "filename": pdf_path.name}
    if is_ingested(store, document_id):
        result["pages"] = "unchanged"
    else:
        sidecar = extracted_dir / f"{pdf_path.stem}.pages.jsonl"
        if not sidecar.is_file():
            progress("extracting text", 0.05)
            pdf_extraction.run(pdf_path.parent, extracted_dir)
        progress("storing pages", 0.3)
        result["pages"] = store.ingest_extracted_directory(extracted_dir, pdf_path.parent)["content_units"]
    if not metadata.get(document_id):
        progress("detecting metadata", 0.35)
        metadata.record_detection(document_id, pdf_path.name, detect_metadata(pdf_path))
    progress("extracting images", 0.4)
    result["images"] = extract_images(store, document_id, pdf_path, image_root, captioner)
    return result
