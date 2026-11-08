"""Orchestration of the concept-first pipeline. Coordination only: each step lives in its own module.

    python -m src.kg.orchestrator --document "ICES-2023-74.pdf"
    python -m src.kg.orchestrator --document ICES-2023-74.pdf --document BVAD_2.15.22-final.pdf
    python -m src.kg.orchestrator --all
    python -m src.kg.orchestrator --rebuild-only        # Pass 2 + validation + save, no LLM calls
    add --project PROJECT_ID to work on a project other than the default one

Steps: load stored pages -> segment -> Pass 1 for chunks without a current
extraction (failed chunks are retried on the next run) -> Pass 2 over every
extracted chunk of every document -> graph validation -> save -> summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from src.config import METADATA_FILE, PROJECT_ROOT, REPORT_DIR
from src.kg.config import KGConfig
from src.kg.enrichment import Embedder, enrich
from src.kg.extraction import FAILED, extract_chunk, extraction_is_current
from src.kg.llm import LLMClient, LLMFatalError, create_client
from src.kg.segmentation import segment_document
from src.kg.store import KGStore
from src.kg.validation import validate_graph
from src.metadata import MetadataStore
from src.projects import DEFAULT_PROJECT, ProjectStore, is_default, migrate
from src.storage.mongodb_store import MongoKnowledgeStore, connect_from_environment

logger = logging.getLogger("knowledge_map.kg")
Progress = Callable[[str, int, int], None]


class KGPipeline:
    def __init__(self, store: MongoKnowledgeStore, config: KGConfig | None = None, llm: LLMClient | None = None,
                 embedder: Embedder | None = None, metadata: MetadataStore | None = None,
                 report_dir: Path | None = None) -> None:
        self.config = config or KGConfig()
        self.kg = KGStore(store)
        self.llm = llm
        self.embedder = embedder
        self.metadata = metadata
        self.report_dir = report_dir   # where the merge report is written; None = do not write

    def segment(self, document_ids: Iterable[str]) -> dict[str, Any]:
        totals = {"documents": 0, "chunks": 0, "toc_chunks": 0, "created": 0, "updated": 0, "unchanged": 0,
                  "removed": 0}
        for document_id in document_ids:
            pages = self.kg.pages(document_id)
            project_id = (pages[0].get("project_id") if pages else None) or DEFAULT_PROJECT
            chunks = segment_document(document_id, pages, self.config.chunking, project_id)
            stats = self.kg.save_chunks(document_id, chunks)
            totals["documents"] += 1
            totals["chunks"] += len(chunks)
            totals["toc_chunks"] += sum(1 for chunk in chunks if chunk["kind"] == "toc")
            for key in ("created", "updated", "unchanged", "removed"):
                totals[key] += stats[key]
        return totals

    def extract(self, document_ids: Iterable[str], limit: int | None = None,
                progress: Progress | None = None) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Pass 1 needs an LLM client.")
        chunks = self.kg.chunks(document_ids, kind="text")
        todo = [chunk for chunk in chunks if not extraction_is_current(chunk, self.llm.model)]
        reused = len(chunks) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        stats = {"chunks": len(chunks), "reused": reused, "processed": 0, "failed": 0, "recovery_attempts": 0, "recovered": 0,
                 "local_concepts": 0, "local_relationships": 0, "rejected_items": 0}
        for index, chunk in enumerate(todo, start=1):
            title = (self.metadata.get(chunk["document_id"]).get("title") if self.metadata else None)
            record = extract_chunk(chunk, self.llm, self.config.max_retries, title)
            self.kg.save_extraction(chunk["_id"], record)
            stats["processed"] += 1
            stats["recovery_attempts"] += record["retries"]
            if record["status"] == FAILED:
                stats["failed"] += 1
                logger.warning("chunk %s failed after %d attempts: %s", chunk["_id"], len(record["attempts"]),
                               record["error"])
            elif record["retries"]:
                stats["recovered"] += 1
            stats["local_concepts"] += sum(1 for c in record["concepts"] if c["status"] != "rejected")
            stats["local_relationships"] += sum(1 for r in record["relationships"] if r["status"] != "rejected")
            stats["rejected_items"] += sum(1 for item in record["concepts"] + record["relationships"]
                                           if item["status"] == "rejected")
            if progress:
                progress("extracting", index, len(todo))
        return stats

    def build_graph(self, project_id: str = DEFAULT_PROJECT) -> dict[str, Any]:
        """Pass 2 + graph validation + save over one project's extracted chunks. No LLM calls."""
        chunks = self.kg.chunks(project_id=project_id)
        concepts, relationships, stats = enrich(chunks, self.config.enrichment, self.embedder, project_id)
        concepts, relationships, validation = validate_graph(concepts, relationships, {c["_id"] for c in chunks})
        saved = self.kg.save_graph(concepts, relationships, project_id)
        merge_log = stats.pop("merge_log")
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            name = "kg_merge_report.json" if is_default(project_id) else f"kg_merge_report_{project_id}.json"
            (self.report_dir / name).write_text(
                json.dumps(merge_log, indent=2, ensure_ascii=False), encoding="utf-8")
        return {**stats, "validation": validation, **saved}

    def run(self, document_ids: list[str], limit: int | None = None, progress: Progress | None = None) -> dict[str, Any]:
        """Run every step for documents of one project (mixing projects is refused)."""
        started = time.perf_counter()
        projects = {(d.get("project_id") or DEFAULT_PROJECT)
                    for d in self.kg.store.documents.find({"_id": {"$in": document_ids}}, {"project_id": 1})}
        if len(projects) > 1:
            raise ValueError("One run works on one project; these documents belong to several projects")
        project_id = projects.pop() if projects else DEFAULT_PROJECT
        summary: dict[str, Any] = {"project_id": project_id, "documents": document_ids}
        summary["segmentation"] = self.segment(document_ids)
        summary["pass1"] = self.extract(document_ids, limit, progress)
        summary["pass2"] = self.build_graph(project_id)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        return summary


def _embedder(config: KGConfig) -> Embedder | None:
    if not config.enrichment.use_embeddings:
        return None
    from src.config import EMBEDDING_MODEL
    from src.knowledge.merging import FastEmbedder
    return FastEmbedder(EMBEDDING_MODEL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concept-first knowledge graph pipeline")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--document", action="append", help="Document filename, ID or ID prefix (repeatable)")
    target.add_argument("--all", action="store_true", help="Every stored document")
    target.add_argument("--rebuild-only", action="store_true", help="Pass 2 + validation + save only (no LLM)")
    parser.add_argument("--limit-chunks", type=int, help="Send at most N chunks to the LLM in this run")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project ID (default: default)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = KGConfig()
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database, PROJECT_ROOT)
        migrate(database)
        ProjectStore(database).get(args.project)
        if args.rebuild_only:
            pipeline = KGPipeline(store, config, embedder=_embedder(config), report_dir=REPORT_DIR)
            print(json.dumps(pipeline.build_graph(args.project), indent=2, default=str))
            return 0
        pipeline = KGPipeline(store, config, create_client(config.llm), _embedder(config), MetadataStore(METADATA_FILE),
                              REPORT_DIR)
        documents = pipeline.kg.all_documents(args.project) if args.all else             pipeline.kg.resolve_documents(args.document, args.project)

        def progress(stage: str, done: int, total: int) -> None:
            print(f"\r{stage}: {done}/{total} chunks", end="", file=sys.stderr, flush=True)

        summary = pipeline.run([document["_id"] for document in documents], args.limit_chunks, progress)
        print(file=sys.stderr)
        summary["documents"] = [document["filename"] for document in documents]
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except LLMFatalError as exc:
        print(f"\nLLM setup problem, run stopped: {exc}\nSet ANTHROPIC_API_KEY in .env (see README).", file=sys.stderr)
        return 3
    except Exception as exc:
        logger.exception("kg pipeline failed")
        print(f"KG pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
