"""Command line pipeline: ingest PDFs, extract and merge concepts, build relationships and the index.

Examples (from the project root, inside the .venv):
    python -m src.pipeline run                 # both books in data/raw_pdfs, then every stage
    python -m src.pipeline ingest my.pdf       # add a new PDF, then every stage
    python -m src.pipeline watch               # keep data/inbox in sync (add, change, remove PDFs)
    python -m src.pipeline documents           # list document IDs
    python -m src.pipeline delete ID_OR_NAME   # delete a document and rebuild
    python -m src.pipeline concepts|merge|relations|index|export|stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src.config import (CAPTION_MODEL, EMBEDDING_MODEL, EXTRACTED_DIR, GLINER_MODEL, IMAGE_DIR, METADATA_FILE,
                        RAW_PDF_DIR, REPORT_DIR, UPLOAD_DIR, VAULT_DIR, PROJECT_ROOT, WATCH_DIR,
                        WATCH_INTERVAL)
from src.envcheck import warn_if_needed
from src.ingest.documents import delete_document, ingest_pdf, is_tombstoned, remove_source, stage_upload
from src.ingest.images import VitCaptioner, link_images_to_concepts
from src.knowledge.concept_extraction import GlinerExtractor, configure_logging, extract_concepts
from src.knowledge.merging import FastEmbedder, merge_concepts
from src.knowledge.relationships import store_relationships
from src.metadata import MetadataStore
from src.search.index import build_index
from src.storage.mongodb_store import MongoKnowledgeStore, connect_from_environment, sha256_file

logger = logging.getLogger("knowledge_map.pipeline")
Progress = Callable[[str, float], None]


def _noop(stage: str, fraction: float) -> None:
    return None


class Pipeline:
    """Runs the stages in order. Every stage is idempotent and skips unchanged input."""

    def __init__(self, store: MongoKnowledgeStore, metadata: MetadataStore, caption_images: bool = False,
                 extractor: GlinerExtractor | None = None, embedder: Any = None,
                 upload_dir: Path = UPLOAD_DIR, image_dir: Path = IMAGE_DIR, report_dir: Path = REPORT_DIR) -> None:
        self.store = store
        self.metadata = metadata
        self.captioner = VitCaptioner(CAPTION_MODEL) if caption_images else None
        self._extractor = extractor
        self._embedder = embedder
        self.upload_dir = upload_dir
        self.image_dir = image_dir
        self.report_dir = report_dir

    @property
    def extractor(self) -> GlinerExtractor:
        if self._extractor is None:
            self._extractor = GlinerExtractor(GLINER_MODEL)
        return self._extractor

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = FastEmbedder(EMBEDDING_MODEL)
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
                    source_label: str = "upload") -> dict[str, Any]:
        """Add a PDF (upload, CLI or watched folder). The file is copied to data/uploads/<hash>/."""
        pdf_path, _ = stage_upload(source, filename or source.name, self.upload_dir)
        return ingest_pdf(self.store, self.metadata, pdf_path, pdf_path.parent / "extracted", self.image_dir,
                          self.captioner, progress, source=source_label)

    def delete(self, document_id: str, progress: Progress = _noop) -> dict[str, Any]:
        """Delete a document and all derived records; it will not be re-added automatically."""
        progress("deleting document", 0.1)
        return delete_document(self.store, self.metadata, document_id, self.upload_dir, self.image_dir)

    def replace(self, document_id: str, source: Path, filename: str | None = None, progress: Progress = _noop,
                source_label: str = "upload") -> dict[str, Any]:
        """Add a new version of a document, keep the user's metadata edits, and delete the old version."""
        added = self.ingest_file(source, filename, progress, source_label)
        new_id = added["document_id"]
        if new_id != document_id:
            self.metadata.carry_over_edits(document_id, new_id)
            progress("deleting old version", 0.42)
            added["replaced"] = self.delete(document_id)
        return added

    def remove_source(self, document_id: str, source: str, progress: Progress = _noop) -> dict[str, Any]:
        progress("removing source", 0.1)
        return remove_source(self.store, self.metadata, document_id, source, self.upload_dir, self.image_dir)

    def knowledge_stages(self, progress: Progress = _noop, show_bar: bool = False) -> dict[str, Any]:
        timings: dict[str, Any] = {}

        def timed(name: str, fraction: float, action: Callable[[], Any]) -> Any:
            progress(name, fraction)
            started = time.perf_counter()
            output = action()
            timings[name] = {"seconds": round(time.perf_counter() - started, 1), "result": output}
            logger.info("%s finished: %s", name, output)
            return output

        def concept_progress(done: int, total: int) -> None:
            progress(f"finding concepts ({done}/{total} pages)", 0.45 + 0.35 * (done / total if total else 1))

        timed("concepts", 0.45, lambda: extract_concepts(self.store, self.extractor, progress=concept_progress,
                                                          show_bar=show_bar))
        timed("merge", 0.8, lambda: merge_concepts(self.store, self.embedder, self.report_dir / "merge_report.csv"))
        timed("link_images", 0.88, lambda: link_images_to_concepts(self.store))
        timed("relations", 0.9, lambda: store_relationships(self.store))
        timed("index", 0.96, lambda: build_index(self.store))
        progress("done", 1.0)
        return timings


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge Map pipeline")
    parser.add_argument("--caption-images", action="store_true",
                        help="Also caption images with a ViT-GPT2 model (off by default; stored separately)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Ingest data/raw_pdfs and run every stage")
    ingest = sub.add_parser("ingest", help="Ingest one or more PDF files, then run every stage")
    ingest.add_argument("pdfs", nargs="+", type=Path)
    for name, text in (("concepts", "Extract concept mentions (resumable)"), ("merge", "Merge concepts into nodes"),
                       ("relations", "Build relationships"), ("index", "Build the keyword index"),
                       ("export", "Write the Obsidian vault"), ("stats", "Print collection counts and graph stats"),
                       ("metadata", "Detect citation metadata for documents that have none")):
        sub.add_parser(name, help=text)
    sub.add_parser("documents", help="List documents with ID, pages and sources")
    delete = sub.add_parser("delete", help="Delete documents (by ID, ID prefix or filename), then rebuild")
    delete.add_argument("documents", nargs="+")
    watch = sub.add_parser("watch", help=f"Watch a folder for added, changed or removed PDFs (default {WATCH_DIR})")
    watch.add_argument("--folder", type=Path, default=WATCH_DIR)
    watch.add_argument("--interval", type=float, default=WATCH_INTERVAL, help="Seconds between scans")
    watch.add_argument("--once", action="store_true", help="Scan twice (to confirm files are stable), apply, exit")
    args = parser.parse_args(argv)

    configure_logging()
    warn_if_needed()
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database, PROJECT_ROOT)
        metadata = MetadataStore(METADATA_FILE)
        pipeline = Pipeline(store, metadata, caption_images=args.caption_images)
        started = time.perf_counter()
        if args.command in ("run", "ingest"):
            if args.command == "run":
                ingested = pipeline.ingest_raw_books()
            else:
                ingested = [pipeline.ingest_file(path) for path in args.pdfs]
            ingest_seconds = round(time.perf_counter() - started, 1)
            timings = pipeline.knowledge_stages(show_bar=True)
            report = {"ingest": ingested, "ingest_seconds": ingest_seconds, "stages": timings,
                      "total_seconds": round(time.perf_counter() - started, 1)}
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "pipeline_run.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            _print(report)
        elif args.command == "documents":
            _print([{"id": document["_id"], "filename": document["filename"], "pages": document.get("page_count"),
                     "sources": document.get("sources", [])} for document in store.documents.find()])
        elif args.command == "delete":
            results = []
            for key in args.documents:
                matches = [document["_id"] for document in store.documents.find({}, {"filename": 1})
                           if document["_id"].startswith(key) or document["filename"] == key]
                if len(matches) != 1:
                    raise ValueError(f"'{key}' matches {len(matches)} documents; use the ID from 'documents'")
                results.append(pipeline.delete(matches[0]))
            _print({"deleted": results, "stages": pipeline.knowledge_stages()})
        elif args.command == "watch":
            from src.ingest.watcher import FolderWatcher, process_events
            watcher = FolderWatcher(store, args.folder)
            print(f"Watching {args.folder.resolve()} every {args.interval} s. Press Ctrl+C to stop.")
            scans = 0
            while True:
                events = watcher.scan()
                scans += 1
                if events:
                    _print(process_events(pipeline, watcher, events))
                if args.once and scans >= 2:
                    break
                time.sleep(args.interval if not args.once else 0.5)
        elif args.command == "concepts":
            _print(extract_concepts(store, pipeline.extractor, show_bar=True))
        elif args.command == "merge":
            _print(merge_concepts(store, pipeline.embedder, REPORT_DIR / "merge_report.csv"))
            _print(link_images_to_concepts(store))
        elif args.command == "relations":
            _print(store_relationships(store))
        elif args.command == "index":
            _print(build_index(store))
        elif args.command == "export":
            from src.export.obsidian_exporter import export_vault
            _print(export_vault(store, metadata, VAULT_DIR))
        elif args.command == "metadata":
            from src.metadata import detect_metadata
            for document in store.documents.find():
                if not metadata.get(document["_id"]):
                    path = PROJECT_ROOT / document["file_storage"]["uri"]
                    metadata.record_detection(document["_id"], document["filename"], detect_metadata(path))
            _print(metadata.load())
        elif args.command == "stats":
            from src.graph.graph_builder import graph_stats, load_graph
            counts = {name: database[name].count_documents({}) for name in sorted(database.list_collection_names())}
            _print({"collections": counts, "graph": graph_stats(load_graph(store))})
        return 0
    except Exception as exc:
        logger.exception("pipeline failed")
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
