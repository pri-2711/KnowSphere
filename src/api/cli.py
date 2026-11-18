"""Command line interface. Uses the same application services as the web API.

Examples (from the project root, inside the .venv):
    python -m src.api.cli run                 # both books in data/raw_pdfs, then every stage
    python -m src.api.cli ingest my.pdf       # add a new PDF, then every stage
    python -m src.api.cli watch               # keep data/inbox in sync (add, change, remove PDFs)
    python -m src.api.cli documents           # list document IDs
    python -m src.api.cli delete ID_OR_NAME   # delete a document and rebuild
    python -m src.api.cli concepts|merge|relations|index|export|stats
    python -m src.api.cli structure [--parser docling|pymupdf]   # sections, tables, figures, equations
    python -m src.api.cli validate|rebuild|status|resume|models
    python -m src.api.cli versions DOCUMENT_ID
    python -m src.api.cli search "QUERY" | research "QUESTION"     # hybrid retrieval, evidence-first answers
    python -m src.api.cli diagnostics [--probe-models] | review | history | activity
    python -m src.api.cli semantic-index        # embed passages with the local embedding model
    python -m src.api.cli backup FOLDER | restore FOLDER        # one project as Extended JSON (never overwrites)
    python -m src.api.cli evaluate DATASET.json # retrieval Recall@10 and citation correctness
    add --project PROJECT_ID to work on a project other than the default one
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.config import METADATA_FILE, PROJECT_ROOT, REPORT_DIR, VAULT_DIR, WATCH_DIR, WATCH_INTERVAL
from src.envcheck import warn_if_needed
from src.knowledge_extraction.gliner_concepts import configure_logging, extract_concepts
from src.knowledge_extraction.image_links import link_images_to_concepts
from src.knowledge_extraction.merging import merge_concepts
from src.knowledge_extraction.relationships import store_relationships
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore, connect_from_environment
from src.knowledge_repository.projects import DEFAULT_PROJECT, ProjectStore, is_default, migrate, project_filter
from src.orchestration.pipeline import Pipeline
from src.orchestration.services import KnowledgeApplication
from src.retrieval.lexical_index import build_index


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge Map pipeline")
    parser.add_argument("--caption-images", action="store_true",
                        help="Also caption images with a ViT-GPT2 model (off by default; stored separately)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="Project ID the command works on (default: default). See the 'projects' command.")
    sub = parser.add_subparsers(dest="command", required=True)
    projects_cmd = sub.add_parser("projects", help="List projects, or create one with --create NAME")
    projects_cmd.add_argument("--create", metavar="NAME", help="Create a project with this name")
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
    structure = sub.add_parser("structure", help="Parse sections, tables, figures and equations (no model)")
    structure.add_argument("--document", help="Only this document ID")
    structure.add_argument("--parser", choices=("auto", "docling", "pymupdf"), default=None)
    validate = sub.add_parser("validate", help="Run evidence governance on the project's knowledge")
    validate.add_argument("--source", choices=("gliner", "concept_first"), default=None)
    sub.add_parser("rebuild", help="Rebuild the project's graphs, index and governance (no ingestion)")
    sub.add_parser("status", help="Show stage runs and knowledge statuses of the project")
    sub.add_parser("resume", help="Rerun the knowledge stages if a previous run did not finish")
    versions = sub.add_parser("versions", help="Show the version history of a document")
    versions.add_argument("document")
    sub.add_parser("models", help="Show which model serves each capability (loads nothing)")
    search = sub.add_parser("search", help="Hybrid search (lexical, semantic when indexed, graph)")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    research = sub.add_parser("research", help="Evidence-first answer: passages, claims, conflicts, conversions")
    research.add_argument("question")
    diagnose = sub.add_parser("diagnostics", help="Health of the project and platform")
    diagnose.add_argument("--probe-models", action="store_true", help="Also ask Ollama which models are pulled")
    sub.add_parser("review", help="Items waiting for a reviewer")
    sub.add_parser("history", help="Decisions and contributions, newest first")
    sub.add_parser("activity", help="Agent calls (MCP), newest first")
    sub.add_parser("semantic-index", help="Embed page passages for semantic retrieval (runs the embedding model)")
    backup_cmd = sub.add_parser("backup", help="Back up this project into a new folder")
    backup_cmd.add_argument("folder", type=Path)
    backup_cmd.add_argument("--all-projects", action="store_true")
    restore_cmd = sub.add_parser("restore", help="Restore a backup folder (refuses to overwrite)")
    restore_cmd.add_argument("folder", type=Path)
    bench = sub.add_parser("benchmark", help="Run Pass 1 with a text model on N stored chunks and measure it "
                                             "(REAL INFERENCE; stores nothing)")
    bench.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2.5:7b")
    bench.add_argument("--chunks", type=int, default=10)
    bench.add_argument("--dataset", type=Path, help="Evaluation dataset with 'extraction' labels for quality")
    evaluate = sub.add_parser("evaluate", help="Evaluate retrieval and citations against a dataset (no model)")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--method", choices=["lexical", "hybrid"], default="hybrid")
    args = parser.parse_args(argv)

    configure_logging()
    warn_if_needed()
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database, PROJECT_ROOT)
        migrate(database)
        projects = ProjectStore(database)
        project = args.project
        projects.get(project)                      # unknown project: stop before doing anything
        scope = project_filter(project)
        metadata = MetadataStore(METADATA_FILE)
        pipeline = Pipeline(store, metadata, caption_images=args.caption_images)
        app = KnowledgeApplication(store, metadata, lambda: pipeline)   # the same services the web API uses
        started = time.perf_counter()
        if args.command == "projects":
            if args.create:
                app.create_project(args.create)
            _print(app.list_projects())
        elif args.command == "structure":
            _print(app.parse_structure(project, args.document, args.parser))
        elif args.command == "validate":
            _print(app.validate(project, args.source))
        elif args.command == "rebuild":
            _print(app.rebuild(project))
        elif args.command == "status":
            _print({**app.stage_status(project), "knowledge": app.knowledge_summary(project)})
        elif args.command == "resume":
            unfinished = app.stage_status(project)["unfinished"]
            _print({"unfinished": unfinished, "rebuild": app.rebuild(project) if unfinished else "nothing to resume"})
        elif args.command == "versions":
            _print(app.document_versions(project, args.document))
        elif args.command == "models":
            _print(pipeline.router.describe())
        elif args.command == "search":
            _print(app.search_hybrid(project, args.query, args.limit))
        elif args.command == "research":
            _print(app.research(project, args.question))
        elif args.command == "diagnostics":
            _print(app.diagnostics(project, args.probe_models))
        elif args.command == "review":
            _print(app.review_queue(project))
        elif args.command == "history":
            _print(app.history(project))
        elif args.command == "activity":
            _print(app.agent_activity(project))
        elif args.command == "semantic-index":
            _print(app.build_semantic_index(project))
        elif args.command == "backup":
            from src.knowledge_repository.backup import backup
            _print(backup(database, args.folder, None if args.all_projects else project))
        elif args.command == "restore":
            from src.knowledge_repository.backup import restore
            _print(restore(database, args.folder))
        elif args.command == "benchmark":
            from src.orchestration.benchmark import run_benchmark
            labels = json.loads(args.dataset.read_text(encoding="utf-8")).get("extraction") if args.dataset else None
            result = run_benchmark(store, project, args.model, args.chunks, labels)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / f"benchmark_{args.model.replace(':', '_')}.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8")
            _print({k: v for k, v in result.items() if k != "per_chunk"})
        elif args.command == "evaluate":
            from src.orchestration.evaluation import evaluate_dataset
            _print(evaluate_dataset(app, project, args.dataset, args.method))
        elif args.command in ("run", "ingest"):
            if args.command == "run":
                if not is_default(project):
                    raise ValueError("'run' ingests data/raw_pdfs into the default project; use 'ingest' for others")
                ingested = pipeline.ingest_raw_books()
            else:
                ingested = [pipeline.ingest_file(path, project_id=project) for path in args.pdfs]
            ingest_seconds = round(time.perf_counter() - started, 1)
            timings = pipeline.knowledge_stages(show_bar=True, project_id=project)
            report = {"ingest": ingested, "ingest_seconds": ingest_seconds, "stages": timings,
                      "total_seconds": round(time.perf_counter() - started, 1)}
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "pipeline_run.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            _print(report)
        elif args.command == "documents":
            _print([{"id": document["_id"], "filename": document["filename"], "pages": document.get("page_count"),
                     "sources": document.get("sources", [])} for document in store.documents.find(scope)])
        elif args.command == "delete":
            results = []
            for key in args.documents:
                matches = [document["_id"] for document in store.documents.find(scope, {"filename": 1})
                           if document["_id"].startswith(key) or document["filename"] == key]
                if len(matches) != 1:
                    raise ValueError(f"'{key}' matches {len(matches)} documents; use the ID from 'documents'")
                results.append(pipeline.delete(matches[0]))
            _print({"deleted": results, "stages": app.rebuild(project)})
        elif args.command == "watch":
            from src.orchestration.watcher import FolderWatcher, process_events
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
            _print(extract_concepts(store, pipeline.extractor, show_bar=True, project_id=project))
        elif args.command == "merge":
            report = "merge_report.csv" if is_default(project) else f"merge_report_{project}.csv"
            _print(merge_concepts(store, pipeline.embedder, REPORT_DIR / report, project_id=project))
            _print(link_images_to_concepts(store, project_id=project))
        elif args.command == "relations":
            _print(store_relationships(store, project))
        elif args.command == "index":
            _print(build_index(store, project))
        elif args.command == "export":
            from src.api.obsidian_exporter import export_vault
            vault = VAULT_DIR if is_default(project) else VAULT_DIR.parent / f"obsidian_vault_{project}"
            _print(export_vault(store, metadata, vault, project))
        elif args.command == "metadata":
            from src.knowledge_repository.metadata import detect_metadata
            for document in store.documents.find(scope):
                if not metadata.get(document["_id"]):
                    path = PROJECT_ROOT / document["file_storage"]["uri"]
                    metadata.record_detection(document["_id"], document["filename"], detect_metadata(path))
            _print(metadata.load())
        elif args.command == "stats":
            from src.retrieval.graph import graph_stats, load_graph
            counts = {name: database[name].count_documents({}) for name in sorted(database.list_collection_names())}
            _print({"project": project, "collections": counts, "graph": graph_stats(load_graph(store, project))})
        return 0
    except Exception as exc:
        logging.getLogger("knowledge_map.cli").exception("pipeline failed")
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
