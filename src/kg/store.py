"""MongoDB persistence for the concept-first pipeline.

- Semantic chunks are ``content_units`` records with ``unit_type: "chunk"`` (the
  existing collection and unique index; ``unit_index`` is the chunk's start offset
  on its first page). Pass 1 results are stored on the chunk as ``extraction``.
- Canonical concepts go to ``kg_concepts`` and relationships to ``kg_relationships``,
  with evidence and provenance embedded. They are separate from ``concepts`` and
  ``relationships`` because the GLiNER stage rebuilds those collections as a whole.

All writes use stable IDs and upserts; the graph collections are rebuilt from the
stored chunk extractions, so re-running never creates duplicates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.knowledge.merging import upsert_all
from src.projects import DEFAULT_PROJECT, project_filter
from src.storage.mongodb_store import MongoKnowledgeStore

CONCEPTS = "kg_concepts"
RELATIONSHIPS = "kg_relationships"


class KGStore:
    def __init__(self, store: MongoKnowledgeStore) -> None:
        self.store = store
        self.units = store.content_units
        self.concepts = store.database[CONCEPTS]
        self.relationships = store.database[RELATIONSHIPS]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        self.units.create_index([("unit_type", 1), ("document_id", 1)], name="ix_unit_type_document")
        self.concepts.create_index([("normalized_name", 1)], name="ix_kg_concept_name")
        self.concepts.create_index([("document_ids", 1)], name="ix_kg_concept_documents")
        self.concepts.create_index([("mentions.chunk_id", 1)], name="ix_kg_concept_chunk")
        self.relationships.create_index([("source_id", 1)], name="ix_kg_rel_source")
        self.relationships.create_index([("target_id", 1)], name="ix_kg_rel_target")
        self.relationships.create_index([("relation_type", 1)], name="ix_kg_rel_type")

    # ------------------------------------------------------------ documents and pages
    def resolve_documents(self, keys: Iterable[str], project_id: str = DEFAULT_PROJECT) -> list[dict[str, Any]]:
        """Find a project's documents by ID, ID prefix or filename; each key must match exactly one."""
        documents = list(self.store.documents.find(project_filter(project_id),
                                                   {"filename": 1, "page_count": 1, "project_id": 1}))
        resolved = []
        for key in keys:
            matches = [d for d in documents if d["_id"] == key or d["_id"].startswith(key) or d["filename"] == key]
            if len(matches) != 1:
                raise ValueError(f"'{key}' matches {len(matches)} documents; use an ID or exact filename")
            resolved.append(matches[0])
        return resolved

    def all_documents(self, project_id: str = DEFAULT_PROJECT) -> list[dict[str, Any]]:
        return list(self.store.documents.find(project_filter(project_id),
                                              {"filename": 1, "page_count": 1, "project_id": 1}).sort("filename", 1))

    def pages(self, document_id: str) -> list[dict[str, Any]]:
        return list(self.units.find({"document_id": document_id, "unit_type": "page"},
                                    {"text": 1, "page_number": 1, "document_id": 1, "project_id": 1}).sort("page_number", 1))

    # ------------------------------------------------------------ chunks
    def save_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> dict[str, int]:
        """Upsert a document's chunks; keep stored extractions whose chunk text is unchanged."""
        existing = {unit["_id"]: unit for unit in self.units.find(
            {"document_id": document_id, "unit_type": "chunk"}, {"text_sha256": 1, "chunker_version": 1})}
        keep = {chunk["_id"] for chunk in chunks}
        stale = [unit_id for unit_id in existing if unit_id not in keep]
        if stale:   # removed first so the unique (document, page, type, index) key stays free
            self.units.delete_many({"_id": {"$in": stale}})
        stats = {"created": 0, "updated": 0, "unchanged": 0, "removed": len(stale)}
        now = datetime.now(timezone.utc)
        for chunk in chunks:
            payload = {key: value for key, value in chunk.items() if key != "_id"}
            payload["quality_status"] = "usable"
            payload["content_type"] = "chunk"
            previous = existing.get(chunk["_id"])
            update: dict[str, Any] = {"$set": payload, "$setOnInsert": {"created_at": now}}
            if previous is None:
                stats["created"] += 1
            elif previous.get("text_sha256") != chunk["text_sha256"]:
                update["$unset"] = {"extraction": ""}    # text changed: the old result no longer applies
                stats["updated"] += 1
            elif previous.get("chunker_version") != chunk["chunker_version"]:
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
            self.units.update_one({"_id": chunk["_id"]}, update, upsert=True)
        return stats

    def chunks(self, document_ids: Iterable[str] | None = None, kind: str | None = None,
               project_id: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"unit_type": "chunk"}
        if project_id is not None:
            query.update(project_filter(project_id))
        if document_ids is not None:
            query["document_id"] = {"$in": list(document_ids)}
        if kind is not None:
            query["kind"] = kind
        return list(self.units.find(query).sort([("document_id", 1), ("chunk_index", 1)]))

    def save_extraction(self, chunk_id: str, record: Mapping[str, Any]) -> None:
        self.units.update_one({"_id": chunk_id}, {"$set": {"extraction": dict(record)}})

    # ------------------------------------------------------------ graph
    def save_graph(self, concepts: list[dict[str, Any]], relationships: list[dict[str, Any]],
                   project_id: str = DEFAULT_PROJECT) -> dict[str, int]:
        """Replace one project's graph; other projects' records are never touched."""
        scope = project_filter(project_id)
        for record in (*concepts, *relationships):
            record["project_id"] = project_id
        removed_concepts = upsert_all(self.concepts, concepts, scope)
        removed_relationships = upsert_all(self.relationships, relationships, scope)
        return {"concepts_written": len(concepts), "relationships_written": len(relationships),
                "stale_concepts_removed": len(removed_concepts),
                "stale_relationships_removed": len(removed_relationships)}
