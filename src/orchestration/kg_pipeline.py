"""Orchestration of the concept-first pipeline. Coordination only: each step lives in its own module.

    Command line: ``python -m src.api.kg_cli`` (see that module).

Steps: load stored pages -> segment -> Pass 1 for chunks without a current
extraction (failed chunks are retried on the next run) -> Pass 2 over every
extracted chunk and figure -> graph validation -> save -> claims -> governance -> summary.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from src.config import KGConfig
from src.contracts.interfaces import Embedder, TextModel, VisionModel
from src.document_intelligence.segmentation import segment_document
from src.evidence_quality.service import validate_project
from src.evidence_quality.validation import validate_graph
from src.knowledge_extraction.claims import build_claims, claims_are_current, extract_claims_chunk
from src.knowledge_extraction.contributions import contribution_units
from src.knowledge_extraction.enrichment import enrich
from src.knowledge_extraction.resolution import ConceptResolver
from src.knowledge_extraction.llm_extraction import FAILED, extract_chunk, extraction_is_current
from src.knowledge_extraction.visual import OCR, extract_image_unit, visual_is_current
from src.knowledge_repository.canonical import CONCEPT_FIRST
from src.knowledge_repository.kg_store import KGStore
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore
from src.knowledge_repository.projects import DEFAULT_PROJECT, is_default, project_filter
from src.knowledge_repository.repository import MongoKnowledgeRepository

logger = logging.getLogger("knowledge_map.kg")
Progress = Callable[[str, int, int], None]


class KGPipeline:
    def __init__(self, store: MongoKnowledgeStore, config: KGConfig | None = None, llm: TextModel | None = None,
                 embedder: Embedder | None = None, metadata: MetadataStore | None = None,
                 report_dir: Path | None = None, vision: VisionModel | None = None, ocr: OCR | None = None) -> None:
        self.config = config or KGConfig()
        self.kg = KGStore(store)
        self.repository = MongoKnowledgeRepository(store, metadata)
        self.llm = llm
        self.vision = vision
        self.ocr = ocr
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

    def _title(self, document_id: str) -> str | None:
        return self.metadata.get(document_id).get("title") if self.metadata else None

    def extract_claims(self, document_ids: Iterable[str], limit: int | None = None,
                       progress: Progress | None = None) -> dict[str, Any]:
        """Claim candidates per text chunk; reuses current results, retries failed chunks only."""
        if self.llm is None:
            raise RuntimeError("Claim extraction needs a text model.")
        chunks = self.kg.chunks(document_ids, kind="text")
        todo = [chunk for chunk in chunks if not claims_are_current(chunk, self.llm.model)][:limit]
        stats = {"chunks": len(chunks), "reused": len(chunks) - len(todo) if limit is None else None,
                 "processed": 0, "failed": 0, "claims": 0, "rejected_claims": 0}
        for index, chunk in enumerate(todo, start=1):
            record = extract_claims_chunk(chunk, self.llm, self.config.max_retries, self._title(chunk["document_id"]))
            self.kg.save_extraction(chunk["_id"], record, "claims_extraction")
            stats["processed"] += 1
            stats["failed"] += record["status"] == FAILED
            stats["claims"] += sum(1 for claim in record["claims"] if claim["status"] != "rejected")
            stats["rejected_claims"] += sum(1 for claim in record["claims"] if claim["status"] == "rejected")
            if progress:
                progress("claims", index, len(todo))
        return stats

    def extract_visual(self, document_ids: Iterable[str], limit: int | None = None,
                       progress: Progress | None = None) -> dict[str, Any]:
        """Concept candidates per image unit with the vision model, grounded on caption, context and OCR."""
        if self.vision is None:
            raise RuntimeError("Visual extraction needs a vision model.")
        units = self.kg.image_units(document_ids)
        todo = [unit for unit in units if not visual_is_current(unit, self.vision.model)][:limit]
        stats = {"images": len(units), "processed": 0, "failed": 0, "missing_files": 0, "local_concepts": 0}
        for index, unit in enumerate(todo, start=1):
            path = self.kg.store.project_root / unit["image_uri"] if unit.get("image_uri") else None
            if path is None or not path.is_file():
                stats["missing_files"] += 1
                continue
            fields = extract_image_unit(unit, path, self.vision, self.config.max_retries, self.ocr,
                                        self._title(unit["document_id"]))
            self.kg.save_unit_fields(unit["_id"], fields)
            stats["processed"] += 1
            stats["failed"] += fields["extraction"]["status"] == FAILED
            stats["local_concepts"] += sum(1 for c in fields["extraction"]["concepts"] if c["status"] != "rejected")
            if progress:
                progress("images", index, len(todo))
        return stats

    def build_graph(self, project_id: str = DEFAULT_PROJECT) -> dict[str, Any]:
        """Pass 2 + graph validation + save + claims + governance over one project.

        The only model calls are canonical resolution questions (when a text model is set); without
        one, stored resolution decisions are reused and nothing is asked.
        """
        units = self.kg.extracted_units(project_id) + self._contributions(project_id)
        enrichment = self.config.enrichment
        resolver = ConceptResolver(self.llm, self.kg, project_id, self.config.max_retries,
                                   enrichment.resolution_max_pairs, enabled=enrichment.resolve)
        concepts, relationships, stats = enrich(units, enrichment, self.embedder, project_id, resolver)
        concepts, relationships, validation = validate_graph(concepts, relationships, {u["_id"] for u in units})
        saved = self.kg.save_graph(concepts, relationships, project_id)
        saved.update(self.kg.save_claims(build_claims(units, concepts, project_id), project_id))
        saved["governance"] = validate_project(self.repository, project_id, CONCEPT_FIRST)
        merge_log = stats.pop("merge_log")
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            name = "kg_merge_report.json" if is_default(project_id) else f"kg_merge_report_{project_id}.json"
            (self.report_dir / name).write_text(
                json.dumps(merge_log, indent=2, ensure_ascii=False), encoding="utf-8")
        return {**stats, "validation": validation, **saved}

    def _contributions(self, project_id: str) -> list[dict[str, Any]]:
        """Submitted proposals of agents and people as pseudo units (candidates like any extraction)."""
        proposals = self.repository.proposals(project_id, status="submitted")
        if not proposals:
            return []
        ids = sorted({proposal.unit_id for proposal in proposals})
        units = {unit["_id"]: unit for unit in self.kg.units.find({"_id": {"$in": ids}, **project_filter(project_id)})}
        return contribution_units(proposals, units, self.repository.ontology(project_id))

    def run(self, document_ids: list[str], limit: int | None = None, progress: Progress | None = None,
            concepts: bool = True, claims: bool = False, visual: bool = False) -> dict[str, Any]:
        """Run every step for documents of one project (mixing projects is refused)."""
        started = time.perf_counter()
        projects = {(d.get("project_id") or DEFAULT_PROJECT)
                    for d in self.kg.store.documents.find({"_id": {"$in": document_ids}}, {"project_id": 1})}
        if len(projects) > 1:
            raise ValueError("One run works on one project; these documents belong to several projects")
        project_id = projects.pop() if projects else DEFAULT_PROJECT
        summary: dict[str, Any] = {"project_id": project_id, "documents": document_ids}
        summary["segmentation"] = self.segment(document_ids)
        if concepts:
            summary["pass1"] = self.extract(document_ids, limit, progress)
        if claims:
            summary["claims"] = self.extract_claims(document_ids, limit, progress)
        if visual:
            summary["visual"] = self.extract_visual(document_ids, limit, progress)
        summary["pass2"] = self.build_graph(project_id)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        return summary
