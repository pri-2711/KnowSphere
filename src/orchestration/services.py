"""Application services: the single entry point for the web API, the CLI and a future MCP server.

Interfaces (HTTP, command line, MCP) only translate requests into these calls and
results into responses; the rules live here. Every operation is scoped to one
project and raises ``NotFound`` for records of another project.

    projects       list / create / rename / delete
    documents      list, metadata, versions, ingest, replace, delete (ingestion runs no model)
    processing     structure parsing, knowledge rebuild, validation, stage status
    knowledge      governance status, review queue, adjudication and its revocation, reviewer merges and
                   splits and their reversal, ontology, proposals, decision history
    retrieval      per project: lexical ``KnowledgeService``, the governed graph, the semantic index and
                   the hybrid retriever (search, research answers, graph exploration, claims, conflicts)
"""

from __future__ import annotations

import functools
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.config import IMAGE_DIR, PROJECT_ROOT, UPLOAD_DIR
from src.contracts.ids import stable_id
from src.contracts.ontology import Ontology, OntologyType
from src.contracts.schemas import Adjudication, Proposal
from src.knowledge_extraction.contributions import check as check_proposal
from src.knowledge_extraction.resolution import reviewer_record_id
from src.knowledge_repository.kg_store import CONCEPTS as KG_CONCEPTS, KGStore
from src.knowledge_repository.locks import ProjectBusy, ProjectLocks  # noqa: F401  (ProjectBusy re-exported)
from src.textutil import concept_key
from src.document_intelligence.parsers import select_parser
from src.document_intelligence.quality import assess
from src.evidence_quality.service import validate_project
from src.knowledge_repository.canonical import COLLECTIONS, SOURCES
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore
from src.knowledge_repository.projects import (DEFAULT_PROJECT, ProjectError, ProjectStore, document_id_for, migrate,
                                               project_filter)
from src.knowledge_repository.repository import MongoKnowledgeRepository
from src.orchestration.documents import delete_project
from src.orchestration.stages import StageTracker
from src.retrieval.engine import KnowledgeService
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.knowledge_graph import GovernedGraph
from src.retrieval.semantic import SemanticIndex, index_pages

Progress = Callable[[str, float], None]


class NotFound(LookupError):
    """A project, document or item that does not exist in the requested project."""


def _locked(operation: str) -> Callable:
    """Run a writing operation under the project's lock (one writer per project, across processes)."""
    def wrap(method: Callable) -> Callable:
        @functools.wraps(method)
        def inner(self: "KnowledgeApplication", project_id: str, *args: Any, **kwargs: Any) -> Any:
            with self.locks.hold(project_id, operation):
                return method(self, project_id, *args, **kwargs)
        return inner
    return wrap


def _noop(stage: str, fraction: float) -> None:
    return None


class KnowledgeApplication:
    def __init__(self, store: MongoKnowledgeStore, metadata: MetadataStore,
                 pipeline_factory: Callable[[], Any] | None = None) -> None:
        self.store = store
        self.metadata = metadata
        migrate(store.database)                        # adds project_id "default" where missing; deletes nothing
        self.projects = ProjectStore(store.database)
        self.repository = MongoKnowledgeRepository(store, metadata)
        self.tracker = StageTracker(store.database)
        self.locks = ProjectLocks(store.database)
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._retrievers: dict[str, KnowledgeService] = {}
        self._graphs: dict[tuple[str, str], GovernedGraph] = {}
        self._semantic: dict[str, SemanticIndex] = {}
        self._embedder: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ shared helpers
    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            if self._pipeline_factory is not None:
                self._pipeline = self._pipeline_factory()
            else:
                from src.orchestration.pipeline import Pipeline
                self._pipeline = Pipeline(self.store, self.metadata)
        return self._pipeline

    def project(self, project_id: str) -> dict[str, Any]:
        try:
            return self.projects.get(project_id)
        except ProjectError as exc:
            raise NotFound(str(exc)) from exc

    def document(self, project_id: str, document_id: str) -> dict[str, Any]:
        self.project(project_id)
        document = self.store.documents.find_one({"_id": document_id, **project_filter(project_id)})
        if document is None:
            raise NotFound("Unknown document in this project")
        return document

    # ------------------------------------------------------------ retrieval
    def retriever(self, project_id: str = DEFAULT_PROJECT) -> KnowledgeService:
        """The project's in-memory retrieval service; it only ever holds that project's records."""
        self.project(project_id)
        with self._lock:
            cached = self._retrievers.get(project_id)
            if cached is None:
                cached = KnowledgeService(self.store, self.metadata, project_id)
                self._retrievers[project_id] = cached
            return cached

    def reload(self, project_id: str) -> None:
        with self._lock:
            self._retrievers.pop(project_id, None)
            self._semantic.pop(project_id, None)
            for key in [key for key in self._graphs if key[0] == project_id]:
                self._graphs.pop(key)
        if project_id in {p["_id"] for p in self.projects.list()}:
            self.retriever(project_id)

    def graph_source(self, project_id: str, source: str | None = None) -> str:
        """The graph a request reads: the one asked for, else concept-first when it has concepts, else GLiNER."""
        if source:
            if source not in SOURCES:
                raise ValueError(f"Unknown graph source '{source}'")
            return source
        has_kg = self.store.database[KG_CONCEPTS].find_one(project_filter(project_id), {"_id": 1})
        return "concept_first" if has_kg else "gliner"

    def knowledge_graph(self, project_id: str, source: str | None = None) -> GovernedGraph:
        """The project's governed graph (rejected items left out), read through the repository."""
        self.project(project_id)
        source = self.graph_source(project_id, source)
        with self._lock:
            graph = self._graphs.get((project_id, source))
        if graph is None:
            graph = GovernedGraph(self.repository.knowledge(project_id, source),
                                  self.repository.adjudications(project_id))
            with self._lock:
                self._graphs[(project_id, source)] = graph
        return graph

    @property
    def embedder(self) -> Any:
        """The embedder for semantic retrieval: the pipeline's (a local model), loaded on first use."""
        if self._embedder is None:
            self._embedder = getattr(self.pipeline, "embedder", None)
        return self._embedder

    def semantic_index(self, project_id: str) -> SemanticIndex:
        with self._lock:
            index = self._semantic.get(project_id)
        if index is None:
            index = SemanticIndex(self.store.database, project_id, self.embedder_name())
            index.load()
            with self._lock:
                self._semantic[project_id] = index
        return index

    def embedder_name(self) -> str:
        from src.config import EMBEDDING_MODEL
        return getattr(self._embedder, "model_name", None) or EMBEDDING_MODEL

    def hybrid(self, project_id: str, source: str | None = None) -> HybridRetriever:
        """Lexical + semantic (when the project has an index) + graph retrieval over one project."""
        index = self.semantic_index(project_id)
        return HybridRetriever(self.retriever(project_id), self.knowledge_graph(project_id, source), index,
                               self.embedder if index.ready else None)

    @_locked("build_semantic_index")
    def build_semantic_index(self, project_id: str) -> dict[str, Any]:
        """Embed the project's page passages (incremental). Runs the embedding model: an explicit step."""
        self.project(project_id)
        if self.embedder is None:
            raise RuntimeError("No embedder is configured for semantic retrieval")
        pages = self.store.content_units.find({"unit_type": "page", **project_filter(project_id)}, {"text": 1})
        index = SemanticIndex(self.store.database, project_id, self.embedder_name())
        result = self.tracker.run(project_id, "semantic_index", lambda: index.build(index_pages(pages), self.embedder))
        with self._lock:
            self._semantic.pop(project_id, None)
        return result

    def graph_payload(self, project_id: str, source: str | None, document_id: str | None, concept_type: str | None,
                      min_weight: float, limit: int, focus: str | None) -> dict[str, Any]:
        """Map data: the GLiNER graph from the lexical cache (as before), the concept-first graph governed."""
        source = self.graph_source(project_id, source)
        if source == "gliner":
            return {**self.retriever(project_id).graph_payload(document_id, concept_type, min_weight, limit, focus),
                    "source": source}
        return self.knowledge_graph(project_id, source).payload(document_id, concept_type, min_weight, limit, focus)

    def search_hybrid(self, project_id: str, query: str, limit: int = 10, source: str | None = None) -> dict[str, Any]:
        return self.hybrid(project_id, source).search(query, limit)

    def research(self, project_id: str, question: str, source: str | None = None) -> dict[str, Any]:
        return self.hybrid(project_id, source).research(question)

    def explore(self, project_id: str, concept_ids: list[str], hops: int = 1, source: str | None = None,
                limit: int = 60) -> dict[str, Any]:
        graph = self.knowledge_graph(project_id, source)
        missing = [c for c in concept_ids if c not in graph.concepts]
        if missing:
            raise NotFound(f"Unknown concept in this project's {graph.source} graph: {missing[0]}")
        return {"source": graph.source, **graph.neighbourhood(concept_ids, max(1, min(hops, 3)), limit)}

    def connection(self, project_id: str, a: str, b: str, source: str | None = None,
                   max_hops: int = 4) -> dict[str, Any]:
        graph = self.knowledge_graph(project_id, source)
        return {"source": graph.source, "path": graph.path(a, b, max_hops)}

    def concept_knowledge(self, project_id: str, concept_id: str, source: str | None = None) -> dict[str, Any]:
        """One governed concept by its canonical ID: the governed detail, plus (concept-first graph) the names
        it was written as, its documents and pages, and how it was resolved (merges, decisions, open questions)."""
        graph = self.knowledge_graph(project_id, source)
        detail = graph.concept_detail(concept_id)
        if detail is None:
            raise NotFound("Unknown concept in this project")
        result: dict[str, Any] = {"source": graph.source, **detail}
        record = self.store.database[KG_CONCEPTS].find_one({"_id": concept_id, **project_filter(project_id)}) \
            if graph.source == "concept_first" else None
        if record is not None:
            result.update(self._resolution_detail(project_id, record, graph))
        return result

    def _resolution_detail(self, project_id: str, record: dict[str, Any], graph: GovernedGraph) -> dict[str, Any]:
        mentions = record.get("mentions", [])
        names = Counter(m.get("name") for m in mentions if m.get("name"))
        titles = {d.id: d.title or d.filename for d in self.repository.documents(project_id)}
        pages: dict[str, set] = {}
        for mention in mentions:
            pages.setdefault(mention["document_id"], set()).add(mention.get("page_number"))
        keys = {record.get("normalized_name")} | {m["key"] for m in record.get("merged_from", [])}
        decisions = [{"names": r.get("names"), "decision": r.get("decision"), "reason": r.get("reason"),
                      "by": f"reviewer:{r['reviewer']}" if r.get("source") == "reviewer" else r.get("model"),
                      "id": r["_id"]}
                     for r in self.store.database["kg_resolutions"].find(project_filter(project_id))
                     if keys & set(r.get("keys", []))]
        reasons = record.get("merge_reasons", {})
        return {
            "canonical_name": record.get("canonical_name"),
            "original_names": [{"name": name, "mentions": count} for name, count in names.most_common()],
            "documents_detail": [{"document_id": d, "title": titles.get(d, d),
                                  "pages": sorted(p for p in pages[d] if p is not None)} for d in sorted(pages)],
            "merge_history": [{**m, "reason": reasons.get(m["name"], "")} for m in record.get("merged_from", [])],
            "resolution_decisions": decisions,
            "possible_duplicates": [{"id": d, "name": graph.concepts[d].name if d in graph.concepts else d}
                                    for d in record.get("possible_duplicates", [])],
        }

    def find_concepts(self, project_id: str, text: str, source: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        graph = self.knowledge_graph(project_id, source)
        matched = graph.match(text)
        if not matched:
            needle = text.strip().lower()
            matched = [c.id for c in graph.concepts.values()
                       if needle and (needle in c.name.lower() or any(needle in a.lower() for a in c.aliases))]
        return [graph.summary(c) for c in matched[:limit]]

    def claims(self, project_id: str, concept_id: str | None = None, kind: str | None = None,
               modality: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        graph = self.knowledge_graph(project_id, "concept_first")
        ids = graph.claims_by_concept.get(concept_id, []) if concept_id else list(graph.claims)
        rows = [graph.claim(graph.claims[c]) for c in ids]
        return [r for r in rows if (kind is None or r["kind"] == kind)
                and (modality is None or r["modality"] == modality)][:limit]

    def conflicts(self, project_id: str, source: str | None = None) -> list[dict[str, Any]]:
        return self.knowledge_graph(project_id, source).all_conflicts()

    def evidence(self, project_id: str, item_id: str) -> dict[str, Any]:
        """The evidence of any knowledge item (concept, relationship or claim), with each quote re-checked
        against the stored text of the unit it cites."""
        self.project(project_id)
        for source in SOURCES:
            knowledge = self.repository.knowledge(project_id, source)
            for item in (*knowledge.concepts, *knowledge.relationships, *knowledge.claims):
                if item.id == item_id:
                    units = self.repository.units(project_id, [e.source.content_unit_id for e in item.evidence])
                    rows = []
                    for e in item.evidence:
                        unit = units.get(e.source.content_unit_id)
                        rows.append({"quote": e.quote, "document_id": e.source.document_id,
                                     "content_unit_id": e.source.content_unit_id, "page_number": e.source.page_number,
                                     "start": e.source.char_start, "end": e.source.char_end,
                                     "method": e.provenance.method, "model": e.provenance.model,
                                     "verified_in_source": bool(unit and e.quote in unit.text)})
                    return {"item_id": item_id, "source": source, "status": item.status.value,
                            "uncertainty": item.uncertainty, "evidence": rows}
        raise NotFound("Unknown knowledge item in this project")

    def project_context(self, project_id: str) -> dict[str, Any]:
        """A compact overview of a project for people and agents: documents, knowledge, review, activity."""
        project = self.project(project_id)
        graph = self.knowledge_graph(project_id)
        top = sorted(graph.concepts, key=lambda c: -graph.undirected.degree(c))[:15]
        return {"project": {"id": project["_id"], "name": project["name"]},
                "documents": [{"id": d.id, "filename": d.filename, "title": d.title, "pages": d.page_count}
                              for d in self.repository.documents(project_id)],
                "graph_source": graph.source, "concepts": len(graph.concepts),
                "relationships": len(graph.relationships), "claims": len(graph.claims),
                "conflicts": len(graph.conflicts), "central_concepts": [graph.summary(c) for c in top],
                "status": self.knowledge_summary(project_id),
                "review_queue": len(self.review_queue(project_id)),
                "ontology": self.repository.ontology(project_id).tree(),
                "recent_activity": self.history(project_id, limit=10)}

    def page_source(self, project_id: str, unit_id: str) -> tuple[Path, int]:
        """The source PDF and page number of one of this project's pages."""
        record = self.retriever(project_id).pages.get(unit_id)
        if record is None:
            raise NotFound("Unknown page")
        document = self.store.documents.find_one({"_id": record["document_id"]}, {"file_storage": 1, "format": 1})
        if document and (document.get("format") or "pdf") != "pdf":
            raise NotFound("This document has no page images (it is not a PDF)")
        path = PROJECT_ROOT / document["file_storage"]["uri"] if document else None
        if path is None or not path.is_file():
            raise NotFound("Source PDF not found")
        return path, record["page_number"]

    def image_file(self, project_id: str, unit_id: str) -> Path:
        self.project(project_id)
        unit = self.store.content_units.find_one({"_id": unit_id, "unit_type": "image", **project_filter(project_id)},
                                                 {"image_uri": 1})
        path = PROJECT_ROOT / unit["image_uri"] if unit and unit.get("image_uri") else None
        if path is None or not path.is_file():
            raise NotFound("Unknown image")
        return path

    @staticmethod
    def upload_document_id(project_id: str, sha256: str) -> str:
        return document_id_for(project_id, sha256)

    # ------------------------------------------------------------ projects
    def list_projects(self) -> list[dict[str, Any]]:
        return [{"id": p["_id"], "name": p["name"], "is_default": p["_id"] == DEFAULT_PROJECT,
                 "documents": self.store.documents.count_documents(project_filter(p["_id"])),
                 "concepts": self.store.concepts.count_documents(project_filter(p["_id"]))}
                for p in self.projects.list()]

    def create_project(self, name: str) -> dict[str, Any]:
        project = self.projects.create(name)           # ProjectError for an invalid or duplicate name
        return {"id": project["_id"], "name": project["name"], "is_default": False, "documents": 0, "concepts": 0}

    def rename_project(self, project_id: str, name: str) -> dict[str, Any]:
        self.project(project_id)
        project = self.projects.rename(project_id, name)
        return {"id": project["_id"], "name": project["name"]}

    @_locked("delete_project")
    def delete_project(self, project_id: str, progress: Progress = _noop) -> dict[str, Any]:
        self.project(project_id)
        if project_id == DEFAULT_PROJECT:
            raise ProjectError("The default project cannot be deleted")
        progress("deleting project", 0.2)
        result = delete_project(self.store, self.metadata, project_id,
                                getattr(self.pipeline, "upload_dir", UPLOAD_DIR),
                                getattr(self.pipeline, "image_dir", IMAGE_DIR))
        with self._lock:
            self._retrievers.pop(project_id, None)
        return result

    # ------------------------------------------------------------ documents
    def list_documents(self, project_id: str,
                       job_for: Callable[[str], dict[str, Any] | None] | None = None) -> list[dict[str, Any]]:
        self.project(project_id)
        rows = []
        for document in self.store.documents.find(project_filter(project_id),
                                                  {"filename": 1, "page_count": 1, "sources": 1, "structure": 1,
                                                   "current_version_id": 1}):
            meta = self.metadata.get(document["_id"])
            units = self.store.content_units
            pages = units.count_documents({"document_id": document["_id"], "unit_type": "page"})
            job = job_for(document["_id"]) if job_for else None
            status = job["status"] if job and job["status"] in ("queued", "running", "failed") else (
                "processed" if pages == document.get("page_count") else "incomplete")
            rows.append({
                "id": document["_id"], "filename": document["filename"], "pages": document.get("page_count", 0),
                "images": units.count_documents({"document_id": document["_id"], "unit_type": "image"}),
                "concepts": self.store.concepts.count_documents({"document_ids": document["_id"]}),
                "title": meta.get("title"), "author": meta.get("author"), "year": meta.get("year"),
                "detected_from": meta.get("detected_from", {}), "user_edited": meta.get("user_edited", []),
                "sources": document.get("sources", []), "structure": document.get("structure"),
                "status": status, "stage": job["stage"] if job else None,
                "format": document.get("format") or "pdf",
                "kind": assess(units.find({"document_id": document["_id"], "unit_type": "page"},
                                          {"extraction_method": 1, "quality_status": 1, "_id": 0}))["kind"],
                "progress": job["progress"] if job else 1.0, "error": job["error"] if job else None,
            })
        return rows

    def document_quality(self, project_id: str, document_id: str) -> dict[str, Any]:
        """Digital, scanned or mixed, page methods and statuses, and warnings for one document."""
        document = self.document(project_id, document_id)
        pages = self.store.content_units.find({"document_id": document_id, "unit_type": "page"},
                                              {"text": 1, "extraction_method": 1, "quality_status": 1})
        return {"document_id": document_id, "filename": document["filename"],
                "format": document.get("format") or "pdf", **assess(pages)}

    def update_metadata(self, project_id: str, document_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self.document(project_id, document_id)
        return self.metadata.update(document_id, {k: v for k, v in changes.items() if v is not None})

    def document_versions(self, project_id: str, document_id: str) -> list[dict[str, Any]]:
        document = self.document(project_id, document_id)
        lineage = document.get("lineage_id") or document_id
        return [version.model_dump(mode="json") for version in self.repository.document_versions(project_id, lineage)]

    @_locked("ingest")
    def ingest(self, project_id: str, path: Path, filename: str, progress: Progress = _noop,
               rebuild: bool = True) -> dict[str, Any]:
        """Add a PDF: text, pages, metadata and images (no model). Then rebuild the project's knowledge."""
        self.project(project_id)
        result = self.pipeline.ingest_file(path, filename, progress, project_id=project_id)
        if rebuild:
            self.rebuild(project_id, progress)
        return result

    @_locked("replace")
    def replace(self, project_id: str, document_id: str, path: Path, filename: str,
                progress: Progress = _noop) -> dict[str, Any]:
        self.document(project_id, document_id)
        result = self.pipeline.replace(document_id, path, filename, progress)
        self.rebuild(project_id, progress)
        return result

    @_locked("delete_document")
    def delete_document(self, project_id: str, document_id: str, progress: Progress = _noop) -> dict[str, Any]:
        self.document(project_id, document_id)
        result = self.pipeline.delete(document_id, progress)
        self.rebuild(project_id, progress)
        return result

    # ------------------------------------------------------------ processing
    @_locked("rebuild")
    def rebuild(self, project_id: str, progress: Progress = _noop) -> dict[str, Any]:
        """Project-specific rebuild of concepts, relationships, index, concept-first graph and governance."""
        self.project(project_id)
        result = self.pipeline.knowledge_stages(progress, project_id=project_id)
        progress("refreshing caches", 0.99)
        self.reload(project_id)
        return result

    @_locked("parse_structure")
    def parse_structure(self, project_id: str, document_id: str | None = None,
                        parser: str | None = None) -> list[dict[str, Any]]:
        """Run a structure parser (Docling or PyMuPDF) and store sections, tables, figures, equations."""
        self.project(project_id)
        chosen = select_parser(parser)
        query = {**project_filter(project_id), **({"_id": document_id} if document_id else {})}
        results = []
        for document in self.store.documents.find(query, {"file_storage": 1, "current_version_id": 1}):
            path = PROJECT_ROOT / document["file_storage"]["uri"]
            if not path.is_file():
                results.append({"document_id": document["_id"], "error": "source PDF not found"})
                continue
            parsed = self.tracker.run(project_id, f"structure:{chosen.name}", lambda: chosen.parse(
                path, project_id, document["_id"], document.get("current_version_id")), document["_id"])
            results.append({"document_id": document["_id"], "parser": chosen.name,
                            **self.repository.save_structure(parsed)})
        if document_id and not results:
            raise NotFound("Unknown document in this project")
        return results

    @_locked("validate")
    def validate(self, project_id: str, source: str | None = None) -> list[dict[str, Any]]:
        """Unified knowledge validation (governance) for one or both graph sources of a project."""
        self.project(project_id)
        sources = [source] if source else list(SOURCES)
        return [self.tracker.run(project_id, f"validate:{name}",
                                 lambda name=name: validate_project(self.repository, project_id, name))
                for name in sources]

    def stage_status(self, project_id: str) -> dict[str, Any]:
        self.project(project_id)
        return {"project_id": project_id, "stages": self.tracker.status(project_id),
                "unfinished": self.tracker.unfinished(project_id)}

    # ------------------------------------------------------------ knowledge governance
    def knowledge_summary(self, project_id: str) -> dict[str, Any]:
        self.project(project_id)
        summary: dict[str, Any] = {}
        for source, collections in COLLECTIONS.items():
            summary[source] = {item_type: dict(Counter(record.get("knowledge_status") or "unvalidated"
                                                       for record in self.store.database[name].find(
                                                           project_filter(project_id), {"knowledge_status": 1})))
                               for item_type, name in collections.items()}
        return {"project_id": project_id, **summary}

    @_locked("adjudicate")
    def adjudicate(self, project_id: str, item_id: str, reviewer: str, decision: str, note: str = "") -> dict[str, Any]:
        """Record a reviewer decision for an item of this project, then re-run governance for its graph."""
        self.project(project_id)
        for source, collections in COLLECTIONS.items():
            for name in collections.values():
                if self.store.database[name].find_one({"_id": item_id, **project_filter(project_id)}, {"_id": 1}):
                    adjudication = Adjudication(reviewer=reviewer, decision=decision, note=note)
                    previous = self.repository.adjudications(project_id).get(item_id)
                    self.repository.adjudicate(project_id, item_id, adjudication)
                    self.repository.append_history(project_id, f"reviewer:{reviewer}", "adjudicate", item_id, {
                        "decision": decision, "note": note, "source": source,
                        "previous": previous.model_dump(mode="json") if previous else None})
                    validate_project(self.repository, project_id, source)
                    self.reload(project_id)
                    return {"item_id": item_id, "source": source, **adjudication.model_dump(mode="json")}
        raise NotFound("Unknown knowledge item in this project")

    @_locked("revoke_adjudication")
    def revoke_adjudication(self, project_id: str, item_id: str, reviewer: str, note: str = "") -> dict[str, Any]:
        """Undo a reviewer decision: governance decides the item again. The decision stays in the history."""
        self.project(project_id)
        revoked = self.repository.revoke_adjudication(project_id, item_id)
        if revoked is None:
            raise NotFound("This item has no reviewer decision in this project")
        self.repository.append_history(project_id, f"reviewer:{reviewer}", "revoke_adjudication", item_id,
                                       {"revoked": revoked.model_dump(mode="json"), "note": note})
        for source in SOURCES:
            validate_project(self.repository, project_id, source)
        self.reload(project_id)
        return {"item_id": item_id, "revoked": revoked.model_dump(mode="json")}

    def review_queue(self, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Governed concept-first items waiting for a person: needs_review first, with their failed checks."""
        self.project(project_id)
        rows = []
        for item_type, name in COLLECTIONS["concept_first"].items():
            for record in self.store.database[name].find({"knowledge_status": "needs_review",
                                                          **project_filter(project_id)}).limit(limit):
                rows.append({
                    "id": record["_id"], "item_type": item_type,
                    "label": record.get("canonical_name") or record.get("statement")
                    or f"{record.get('source_id')} {record.get('relation_type')} {record.get('target_id')}",
                    "failed_checks": (record.get("governance") or {}).get("failed_checks", []),
                    "possible_duplicates": record.get("possible_duplicates", []),
                    "documents": record.get("document_ids") or [record.get("document_id")]})
        return rows[:limit]

    # ------------------------------------------------------------ reviewer merges (reversible)
    def _concept_key(self, project_id: str, ref: str) -> str:
        """A name key from a concept-first concept ID, or from a name or alias as written."""
        record = self.store.database[KG_CONCEPTS].find_one({"_id": ref, **project_filter(project_id)},
                                                           {"normalized_name": 1})
        key = record["normalized_name"] if record else concept_key(ref)
        if not key:
            raise NotFound("Unknown concept")
        return key

    @_locked("review_merge")
    def review_merge(self, project_id: str, a: str, b: str, decision: str, reviewer: str,
                     note: str = "") -> dict[str, Any]:
        """Merge two concepts ("same") or keep two names apart ("different"), then rebuild the graph.

        ``a`` and ``b`` are concept IDs or names (a name splits that alias out of the concept it was
        merged into). The decision overrides every automatic rule and can be reverted.
        """
        self.project(project_id)
        if decision not in ("same", "different"):
            raise ValueError("decision must be 'same' or 'different'")
        key_a, key_b = sorted((self._concept_key(project_id, a), self._concept_key(project_id, b)))
        if key_a == key_b:
            raise ValueError("both refer to the same name")
        record = {"_id": reviewer_record_id(project_id, key_a, key_b), "project_id": project_id, "source": "reviewer",
                  "keys": [key_a, key_b], "names": [a, b], "decision": decision, "canonical_name": "",
                  "reason": note or f"decided by {reviewer}", "reviewer": reviewer}
        kg = KGStore(self.store)
        kg.save_reviewer_resolution(record)
        self.repository.append_history(project_id, f"reviewer:{reviewer}",
                                       "merge" if decision == "same" else "keep_apart", record["_id"],
                                       {"keys": [key_a, key_b], "names": [a, b], "note": note})
        return {"decision": record, "graph": self.rebuild_concept_graph(project_id)}

    @_locked("revert_merge")
    def revert_merge(self, project_id: str, decision_id: str, reviewer: str, note: str = "") -> dict[str, Any]:
        self.project(project_id)
        removed = KGStore(self.store).remove_reviewer_resolution(decision_id)
        if removed is None or removed.get("project_id") != project_id:
            raise NotFound("No reviewer merge decision with this ID in this project")
        self.repository.append_history(project_id, f"reviewer:{reviewer}", "revert_merge", decision_id,
                                       {"reverted": {k: v for k, v in removed.items() if k != "_id"}, "note": note})
        return {"reverted": decision_id, "graph": self.rebuild_concept_graph(project_id)}

    def merge_decisions(self, project_id: str) -> list[dict[str, Any]]:
        self.project(project_id)
        return [{"id": r["_id"], **{k: v for k, v in r.items() if k != "_id"}}
                for r in KGStore(self.store).reviewer_resolutions(project_id)]

    @_locked("rebuild_concept_graph")
    def rebuild_concept_graph(self, project_id: str) -> dict[str, Any]:
        """Pass 2, claims and governance of the concept-first graph from stored results (no model call)."""
        from src.orchestration.kg_pipeline import KGPipeline
        embedder = getattr(self.pipeline, "embedder", None)    # the same name embeddings as a full rebuild
        result = self.tracker.run(project_id, "kg_graph", lambda: KGPipeline(
            self.store, embedder=embedder, metadata=self.metadata).build_graph(project_id))
        self.reload(project_id)
        return {key: result[key] for key in ("canonical_concepts", "relationships", "concepts_merged",
                                             "needs_review_pairs") if key in result}

    # ------------------------------------------------------------ ontology
    def ontology(self, project_id: str) -> dict[str, Any]:
        self.project(project_id)
        ontology = self.repository.ontology(project_id)
        return {"project_id": project_id, "types": [t.model_dump(mode="json") for t in ontology.types],
                "tree": ontology.tree()}

    @_locked("extend_ontology")
    def extend_ontology(self, project_id: str, types: list[dict[str, Any]], actor: str) -> dict[str, Any]:
        """Add domain types (each a specialisation of a known type). Invalid input changes nothing."""
        self.project(project_id)
        current = self.repository.ontology(project_id)
        known = {(t.category, t.name) for t in current.extensions}
        added = [OntologyType.model_validate(t) for t in types]
        added = [t for t in added if (t.category, t.name) not in known]
        ontology = Ontology(project_id=project_id, extensions=[*current.extensions, *added])   # validates
        self.repository.save_ontology(ontology)
        if added:
            self.repository.append_history(project_id, actor, "ontology_extend", None,
                                           {"types": [t.model_dump(mode="json") for t in added]})
        return self.ontology(project_id)

    # ------------------------------------------------------------ proposals (agents and people)
    def propose(self, project_id: str, agent: str, kind: str, unit_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Store a knowledge proposal. A proposal whose quote is not in the unit is rejected at once;
        the others enter the next graph build as candidates and governance decides."""
        self.project(project_id)
        unit = self.store.content_units.find_one({"_id": unit_id, "unit_type": {"$in": ["page", "chunk"]},
                                                  **project_filter(project_id)})
        if unit is None:
            raise NotFound("Unknown page or chunk in this project")
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        proposal = Proposal(id=stable_id("proposal", project_id, agent, kind, unit_id, payload), project_id=project_id,
                            agent=agent, kind=kind, unit_id=unit_id, data=data)
        issues = check_proposal(proposal, unit, self.repository.ontology(project_id))
        proposal = proposal.model_copy(update={"status": "rejected" if issues else "submitted", "issues": issues})
        self.repository.save_proposal(proposal)
        self.repository.append_history(project_id, f"agent:{agent}", "propose", proposal.id,
                                       {"kind": kind, "unit_id": unit_id, "status": proposal.status, "issues": issues})
        return proposal.model_dump(mode="json")

    def proposals(self, project_id: str, status: str | None = None, agent: str | None = None) -> list[dict[str, Any]]:
        self.project(project_id)
        return [p.model_dump(mode="json") for p in self.repository.proposals(project_id, status, agent)]

    def withdraw_proposal(self, project_id: str, proposal_id: str, actor: str) -> dict[str, Any]:
        self.project(project_id)
        if not self.repository.set_proposal_status(project_id, proposal_id, "withdrawn"):
            raise NotFound("Unknown proposal in this project")
        self.repository.append_history(project_id, actor, "withdraw_proposal", proposal_id)
        return {"id": proposal_id, "status": "withdrawn"}

    def diagnostics(self, project_id: str, probe_models: bool = False) -> dict[str, Any]:
        """Health of one project and of the platform. ``probe_models`` also asks Ollama which models it has
        (a local status request; nothing is loaded)."""
        self.project(project_id)
        scope = project_filter(project_id)
        database = self.store.database
        documents = list(self.store.documents.find(scope, {"filename": 1}))
        kinds = Counter(assess(self.store.content_units.find({"document_id": d["_id"], "unit_type": "page"},
                                                             {"extraction_method": 1, "quality_status": 1,
                                                              "_id": 0}))["kind"] for d in documents)
        counts = {name: database[name].count_documents(scope) for name in (
            "content_units", "concepts", "relationships", "kg_concepts", "kg_relationships", "kg_claims",
            "proposals", "semantic_index", "validation_records", "adjudications", "decision_history")}
        unfinished = self.tracker.unfinished(project_id)
        chunks = self.store.content_units.count_documents({"unit_type": "chunk", **scope})
        extracted = self.store.content_units.count_documents({"unit_type": "chunk", "extraction.status": "extracted",
                                                              **scope})
        report: dict[str, Any] = {
            "project_id": project_id, "database": {"name": database.name, "ping": self._ping()},
            "documents": len(documents), "document_kinds": dict(kinds), "records": counts,
            "concept_first": {"chunks": chunks, "extracted": extracted, "pending": chunks - extracted},
            "unfinished_stages": [s["stage"] for s in unfinished], "lock": self.locks.current(project_id),
            "review_queue": len(self.review_queue(project_id)),
            "semantic_index_ready": counts["semantic_index"] > 0,
        }
        router = getattr(self.pipeline, "router", None)
        if router is not None:
            report["models"] = router.describe()
        if probe_models:
            from src.model_runtime.ollama import ollama_status
            report["ollama"] = ollama_status()
        problems = []
        if not report["database"]["ping"]:
            problems.append("database does not answer")
        if unfinished:
            problems.append(f"{len(unfinished)} stage(s) did not finish: run 'resume'")
        if kinds.get("scanned") or kinds.get("mixed"):
            problems.append("some documents are scanned: OCR text may contain errors")
        if chunks and extracted < chunks:
            problems.append(f"{chunks - extracted} chunk(s) have no concept extraction yet")
        if probe_models and not report["ollama"].get("reachable"):
            problems.append("Ollama is not reachable")
        report["problems"] = problems
        return report

    def _ping(self) -> bool:
        try:
            self.store.database.command("ping")
            return True
        except Exception:
            return False

    def agent_activity(self, project_id: str, agent: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        self.project(project_id)
        return self.repository.activity(project_id, agent, limit)

    def history(self, project_id: str, item_id: str | None = None, actor: str | None = None,
                limit: int = 100) -> list[dict[str, Any]]:
        self.project(project_id)
        return [entry.model_dump(mode="json") for entry in self.repository.history(project_id, item_id, actor, limit)]

    def validation_record(self, project_id: str, item_id: str) -> dict[str, Any]:
        self.project(project_id)
        record = self.store.database["validation_records"].find_one({"item_id": item_id, **project_filter(project_id)})
        if record is None:
            raise NotFound("No validation record for this item in this project")
        return {key: value for key, value in record.items() if key != "_id"}
