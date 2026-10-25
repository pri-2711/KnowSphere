"""Document lifecycle: add (ingest), replace with a new version, and delete with all derived data.

A document is identified by the SHA-256 of its PDF bytes. When that hash is
already stored with all its pages, extraction and ingestion are skipped.
Uploaded and watched files are copied to ``data/uploads/<hash prefix>/``;
``data/raw_pdfs`` is only read, never changed or deleted.

Each document lists its ``sources`` (for example ``upload``, ``raw_pdfs/book1.pdf``,
``watch:report.pdf``). A document deleted in the app gets a tombstone in
``deleted_documents`` so that ``pipeline run`` and the folder watcher do not add
the same file back. Adding the file again on purpose (upload) removes the tombstone.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src import pdf_extraction
from src.config import IMAGE_DIR, UPLOAD_DIR
from src.ingest.images import Captioner, extract_images
from src.metadata import MetadataStore, detect_metadata
from src.storage.mongodb_store import MongoKnowledgeStore, sha256_file

logger = logging.getLogger("knowledge_map.ingest")
Progress = Callable[[str, float], None]
TOMBSTONES = "deleted_documents"

# Collections whose records belong to one document (field that holds the document ID).
_DOCUMENT_SCOPED = (
    ("content_units", "document_id"),
    ("assets", "document_id"),
    ("extraction_state", "document_id"),
    ("concept_mentions", "document_id"),
    ("page_terms", "document_id"),
    ("image_state", "_id"),
)


def _noop(stage: str, fraction: float) -> None:
    return None


def is_ingested(store: MongoKnowledgeStore, document_id: str) -> bool:
    document = store.documents.find_one({"_id": document_id}, {"page_count": 1})
    if not document:
        return False
    pages = store.content_units.count_documents({"document_id": document_id, "unit_type": "page"})
    return pages > 0 and pages == document.get("page_count")


def is_tombstoned(store: MongoKnowledgeStore, document_id: str) -> bool:
    return store.database[TOMBSTONES].find_one({"_id": document_id}, {"_id": 1}) is not None


def stage_upload(source: Path, filename: str, upload_root: Path = UPLOAD_DIR) -> tuple[Path, str]:
    """Copy a file into its own folder named by hash. Returns (pdf path, document id)."""
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
    source: str = "upload",
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
        result["pages"] = store.ingest_extracted_directory(
            extracted_dir, pdf_path.parent, only_stems={pdf_path.stem})["content_units"]
    store.documents.update_one({"_id": document_id}, {"$addToSet": {"sources": source}})
    store.database[TOMBSTONES].delete_one({"_id": document_id})
    if not metadata.get(document_id):
        progress("detecting metadata", 0.35)
        metadata.record_detection(document_id, pdf_path.name, detect_metadata(pdf_path))
    progress("extracting images", 0.4)
    result["images"] = extract_images(store, document_id, pdf_path, image_root, captioner)
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def delete_document(
    store: MongoKnowledgeStore,
    metadata: MetadataStore | None,
    document_id: str,
    upload_root: Path = UPLOAD_DIR,
    image_root: Path = IMAGE_DIR,
    tombstone: bool = True,
) -> dict[str, Any]:
    """Remove a document and everything derived from it. Files in data/raw_pdfs are never touched.

    Concept nodes, relationships and the search index are rebuilt afterwards by the
    knowledge stages (merge, relations, index), which drop anything that lost its source.
    """
    document = store.documents.find_one({"_id": document_id})
    if document is None:
        return {"document_id": document_id, "deleted": False}
    removed: dict[str, int] = {}
    for collection, field in _DOCUMENT_SCOPED:
        removed[collection] = store.database[collection].delete_many({field: document_id}).deleted_count
    store.documents.delete_one({"_id": document_id})
    if tombstone:
        store.database[TOMBSTONES].update_one(
            {"_id": document_id},
            {"$set": {"filename": document.get("filename"), "deleted_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    for folder in (upload_root / document_id[:16], image_root / document_id[:16]):
        if folder.is_dir() and (_inside(folder, upload_root) or _inside(folder, image_root)):
            shutil.rmtree(folder, ignore_errors=True)
    if metadata is not None:
        metadata.remove(document_id)
    logger.info("deleted document %s (%s): %s", document_id, document.get("filename"), removed)
    return {"document_id": document_id, "filename": document.get("filename"), "deleted": True, "removed": removed}


def remove_source(
    store: MongoKnowledgeStore,
    metadata: MetadataStore | None,
    document_id: str,
    source: str,
    upload_root: Path = UPLOAD_DIR,
    image_root: Path = IMAGE_DIR,
) -> dict[str, Any]:
    """Forget one source of a document; delete the document when no source is left."""
    store.documents.update_one({"_id": document_id}, {"$pull": {"sources": source}})
    document = store.documents.find_one({"_id": document_id}, {"sources": 1})
    if document is None:
        return {"document_id": document_id, "deleted": False}
    if document.get("sources"):
        return {"document_id": document_id, "deleted": False, "remaining_sources": document["sources"]}
    return delete_document(store, metadata, document_id, upload_root, image_root, tombstone=False)
