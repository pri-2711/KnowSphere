"""MongoDB implementation of ``contracts.interfaces.KnowledgeRepository``.

The repository is the canonical source of truth. It reads the existing
collections through ``canonical`` (so stored data needs no migration) and owns
the newer collections:

    document_versions    one record per file version of a document (lineage across replacements)
    validation_records   latest governance outcome per item (rebuilt per project and graph source)
    adjudications        reviewer decisions; never rebuilt, so they survive every rerun
    kg_claims            claims (written by the concept-first graph build)
    decision_history     append-only record of every decision and contribution (never rebuilt or deleted)
    ontologies           a project's ontology extensions
    proposals            knowledge proposed by agents or people (candidates for Pass 2)
    agent_activity       audit record of every agent call (append only)

Governance writes ``knowledge_status`` and a short ``governance`` summary onto the
item records themselves, so retrieval can filter without a join.
"""

from __future__ import annotations

import itertools
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import ValidationError

from src.contracts.ids import stable_id
from src.contracts.interfaces import KnowledgeSet, UnitRecord
from src.contracts.ontology import Ontology, OntologyType
from src.contracts.schemas import (STRUCTURE_TYPES, Adjudication, ContentUnit, Document, DocumentVersion, HistoryEntry,
                                   KnowledgeStatus, ParsedDocument, Proposal, ValidationRecord)
from src.knowledge_repository.canonical import COLLECTIONS, READERS, SOURCES, content_unit
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore, upsert_all
from src.knowledge_repository.projects import project_filter

VERSIONS, VALIDATIONS, ADJUDICATIONS = "document_versions", "validation_records", "adjudications"
HISTORY, ONTOLOGIES, PROPOSALS = "decision_history", "ontologies", "proposals"
ACTIVITY = "agent_activity"
_COUNTER = itertools.count()


def _sequence() -> int:
    """Strictly increasing within a process, also when the clock does not advance between two calls."""
    return time.time_ns() // 1000 * 1000 + next(_COUNTER) % 1000      # microseconds, fits 64 bits


class MongoKnowledgeRepository:
    def __init__(self, store: MongoKnowledgeStore, metadata: MetadataStore | None = None) -> None:
        self.store = store
        self.db = store.database
        self.metadata = metadata

    # ------------------------------------------------------------ documents and versions
    def documents(self, project_id: str) -> list[Document]:
        result = []
        for record in self.store.documents.find(project_filter(project_id)).sort("filename", 1):
            meta = self.metadata.get(record["_id"]) if self.metadata else {}
            result.append(Document(
                id=record["_id"], project_id=record.get("project_id") or project_id, filename=record["filename"],
                sha256=record.get("source_sha256") or record["_id"], page_count=record.get("page_count") or 0,
                title=meta.get("title"), author=meta.get("author"), year=meta.get("year"),
                current_version_id=record.get("current_version_id"), sources=record.get("sources", [])))
        return result

    def document_versions(self, project_id: str, lineage_id: str | None = None) -> list[DocumentVersion]:
        query = {**project_filter(project_id), **({"lineage_id": lineage_id} if lineage_id else {})}
        return [DocumentVersion.model_validate({**{k: v for k, v in r.items() if k != "_id"}, "id": r["_id"]})
                for r in self.db[VERSIONS].find(query).sort([("lineage_id", 1), ("version", 1)])]

    def record_version(self, project_id: str, document_id: str, sha256: str, filename: str,
                       previous_document_id: str | None = None, parser: str | None = None) -> DocumentVersion:
        """Idempotent: the same file in the same project is always the same version record."""
        existing = self.db[VERSIONS].find_one({"document_id": document_id, **project_filter(project_id)})
        if existing is not None:
            return DocumentVersion.model_validate({**{k: v for k, v in existing.items() if k != "_id"},
                                                   "id": existing["_id"]})
        lineage, number = document_id, 1
        if previous_document_id:
            previous = self.db[VERSIONS].find_one({"document_id": previous_document_id, **project_filter(project_id)})
            if previous is not None:
                lineage, number = previous["lineage_id"], previous["version"] + 1
            else:
                lineage, number = previous_document_id, 2
        version = DocumentVersion(id=stable_id("document_version", project_id, document_id), project_id=project_id,
                                  document_id=document_id, lineage_id=lineage, version=number, sha256=sha256,
                                  filename=filename, parser=parser)
        self.db[VERSIONS].update_one({"_id": version.id},
                                     {"$set": version.model_dump(mode="python", exclude={"id"})}, upsert=True)
        if previous_document_id:
            self.db[VERSIONS].update_many({"document_id": previous_document_id, **project_filter(project_id)},
                                          {"$set": {"superseded_by": document_id}})
        self.store.documents.update_one({"_id": document_id}, {"$set": {"current_version_id": version.id,
                                                                         "lineage_id": lineage}})
        return version

    # ------------------------------------------------------------ content
    def content_units(self, project_id: str, unit_types: Iterable[str] | None = None,
                      document_id: str | None = None) -> list[ContentUnit]:
        query: dict[str, Any] = dict(project_filter(project_id))
        if unit_types is not None:
            query["unit_type"] = {"$in": list(unit_types)}
        if document_id is not None:
            query["document_id"] = document_id
        return [content_unit(r) for r in self.store.content_units.find(query).sort([("document_id", 1),
                                                                                    ("page_number", 1)])]

    def save_structure(self, parsed: ParsedDocument) -> dict[str, int]:
        """Replace a document's structure units (sections, tables, figures, equations). Pages are not touched."""
        scope = {"document_id": parsed.document_id, "unit_type": {"$in": [t.value for t in STRUCTURE_TYPES]},
                 **project_filter(parsed.project_id)}
        removed = self.store.content_units.delete_many(scope).deleted_count
        counts: Counter = Counter()
        now = datetime.now(timezone.utc)
        for unit in parsed.units:
            if unit.project_id != parsed.project_id or unit.document_id != parsed.document_id:
                raise ValueError("a parsed unit belongs to another document or project")
            counts[unit.unit_type.value] += 1
            record = unit.model_dump(mode="json", exclude={"id"})
            record.update(unit_type=unit.unit_type.value, unit_index=counts[unit.unit_type.value],
                          extraction_method=unit.provenance.method, content_type=unit.unit_type.value, created_at=now)
            self.store.content_units.update_one({"_id": unit.id}, {"$set": record}, upsert=True)
        self.store.documents.update_one({"_id": parsed.document_id}, {"$set": {"structure": {
            "parser": parsed.parser.method, "parser_version": parsed.parser.model, "units": dict(counts),
            "warnings": parsed.warnings[:20], "parsed_at": now}}})
        return {"removed": removed, **dict(counts)}

    # ------------------------------------------------------------ knowledge
    def knowledge(self, project_id: str, source: str) -> KnowledgeSet:
        if source not in SOURCES:
            raise ValueError(f"Unknown graph source '{source}'")
        result = KnowledgeSet(project_id=project_id, source=source)
        targets = {"concept": result.concepts, "relationship": result.relationships, "claim": result.claims}
        for item_type, collection in COLLECTIONS[source].items():
            reader = READERS[(source, item_type)]
            for record in self.db[collection].find(project_filter(project_id)):
                try:
                    targets[item_type].append(reader(record))
                except (ValidationError, KeyError, TypeError, ValueError) as exc:
                    result.invalid.append((record["_id"], item_type, f"{type(exc).__name__}: {exc}"))
        return result

    def units(self, project_id: str, unit_ids: Iterable[str]) -> dict[str, UnitRecord]:
        """Owner and stored text of units; image units are checked against their OCR-grounded text."""
        found = {}
        for record in self.store.content_units.find({"_id": {"$in": sorted(set(unit_ids))}, **project_filter(project_id)},
                                                    {"document_id": 1, "page_number": 1, "text": 1,
                                                     "grounding_text": 1}):
            found[record["_id"]] = UnitRecord(record["document_id"], record.get("page_number"),
                                              record.get("grounding_text") or record.get("text") or "")
        return found

    def page_counts(self, project_id: str) -> dict[str, int]:
        return {r["_id"]: r.get("page_count") or 0
                for r in self.store.documents.find(project_filter(project_id), {"page_count": 1})}

    def adjudications(self, project_id: str) -> dict[str, Adjudication]:
        return {r["item_id"]: Adjudication.model_validate(r["adjudication"])
                for r in self.db[ADJUDICATIONS].find(project_filter(project_id))}

    def adjudicate(self, project_id: str, item_id: str, adjudication: Adjudication) -> None:
        self.db[ADJUDICATIONS].update_one(
            {"_id": stable_id("adjudication", project_id, item_id)},
            {"$set": {"project_id": project_id, "item_id": item_id, "adjudication": adjudication.model_dump(mode="python")}},
            upsert=True)

    def revoke_adjudication(self, project_id: str, item_id: str) -> Adjudication | None:
        """Remove the active decision (its history entry keeps it). Returns the decision that was active."""
        record = self.db[ADJUDICATIONS].find_one_and_delete({"_id": stable_id("adjudication", project_id, item_id)})
        return Adjudication.model_validate(record["adjudication"]) if record else None

    # ------------------------------------------------------------ history (append only)
    def append_history(self, project_id: str, actor: str, action: str, item_id: str | None = None,
                       details: dict[str, Any] | None = None) -> HistoryEntry:
        at = datetime.now(timezone.utc)
        entry = HistoryEntry(id=stable_id("history", project_id, actor, action, item_id, at.isoformat()),
                             project_id=project_id, actor=actor, action=action, item_id=item_id,
                             details=details or {}, at=at)
        self.db[HISTORY].insert_one({"_id": entry.id, **entry.model_dump(mode="python", exclude={"id"}),
                                     "seq": _sequence()})          # stable order within one millisecond
        return entry

    def history(self, project_id: str, item_id: str | None = None, actor_prefix: str | None = None,
                limit: int = 100) -> list[HistoryEntry]:
        query: dict[str, Any] = dict(project_filter(project_id))
        if item_id:
            query["item_id"] = item_id
        if actor_prefix:
            query["actor"] = {"$regex": "^" + re.escape(actor_prefix)}
        records = self.db[HISTORY].find(query).sort([("at", -1), ("seq", -1)]).limit(limit)
        return [HistoryEntry.model_validate({**{k: v for k, v in r.items() if k not in ("_id", "seq")}, "id": r["_id"]})
                for r in records]

    # ------------------------------------------------------------ agent activity (append only)
    def log_activity(self, record: dict[str, Any]) -> None:
        at = datetime.now(timezone.utc)
        self.db[ACTIVITY].insert_one({"_id": stable_id("activity", record.get("project_id"), record.get("agent"),
                                                       record.get("tool"), at.isoformat(), id(record)),
                                      **record, "at": at, "seq": _sequence()})

    def activity(self, project_id: str, agent: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query: dict[str, Any] = dict(project_filter(project_id))
        if agent:
            query["agent"] = agent
        return [{k: v for k, v in r.items() if k not in ("_id", "seq")}
                for r in self.db[ACTIVITY].find(query).sort([("at", -1), ("seq", -1)]).limit(limit)]

    # ------------------------------------------------------------ ontology
    def ontology(self, project_id: str) -> Ontology:
        record = self.db[ONTOLOGIES].find_one({"_id": stable_id("ontology", project_id)}) or {}
        return Ontology(project_id=project_id, extensions=[OntologyType.model_validate(t)
                                                          for t in record.get("extensions", [])])

    def save_ontology(self, ontology: Ontology) -> None:
        self.db[ONTOLOGIES].update_one({"_id": stable_id("ontology", ontology.project_id)}, {"$set": {
            "project_id": ontology.project_id,
            "extensions": [t.model_dump(mode="json") for t in ontology.extensions]}}, upsert=True)

    # ------------------------------------------------------------ proposals
    def proposals(self, project_id: str, status: str | None = None, agent: str | None = None) -> list[Proposal]:
        query: dict[str, Any] = dict(project_filter(project_id))
        if status:
            query["status"] = status
        if agent:
            query["agent"] = agent
        return [Proposal.model_validate({**{k: v for k, v in r.items() if k != "_id"}, "id": r["_id"]})
                for r in self.db[PROPOSALS].find(query).sort("created_at", 1)]

    def save_proposal(self, proposal: Proposal) -> None:
        """Idempotent: the same agent proposing the same thing twice keeps one record."""
        record = proposal.model_dump(mode="python", exclude={"id", "created_at"})
        self.db[PROPOSALS].update_one({"_id": proposal.id}, {"$set": record,
                                                             "$setOnInsert": {"created_at": proposal.created_at}},
                                      upsert=True)

    def set_proposal_status(self, project_id: str, proposal_id: str, status: str) -> bool:
        return self.db[PROPOSALS].update_one({"_id": proposal_id, **project_filter(project_id)},
                                             {"$set": {"status": status}}).matched_count == 1

    def save_validation(self, project_id: str, source: str, records: list[ValidationRecord],
                        statuses: dict[str, KnowledgeStatus]) -> dict[str, int]:
        by_item = {record.item_id: record for record in records}
        updated = 0
        for item_type, collection in COLLECTIONS[source].items():
            for record in self.db[collection].find(project_filter(project_id), {"_id": 1}):
                validation = by_item.get(record["_id"])
                if validation is None:
                    continue
                failed = [check.check for check in validation.checks if not check.passed]
                self.db[collection].update_one({"_id": record["_id"]}, {"$set": {
                    "knowledge_status": statuses[record["_id"]].value,
                    "governance": {"policy_version": validation.policy_version, "failed_checks": failed,
                                   "validated_at": validation.created_at}}})
                updated += 1
        rows = [{"_id": r.id, **r.model_dump(mode="json", exclude={"id"}), "graph": source} for r in records]
        stale = upsert_all(self.db[VALIDATIONS], rows, {**project_filter(project_id), "graph": source})
        self.db[VALIDATIONS].create_index([("project_id", 1), ("item_id", 1)], name="ix_validation_item")
        counts = Counter(status.value for status in statuses.values())
        return {"items_updated": updated, "records": len(rows), "stale_records_removed": len(stale), **counts}
