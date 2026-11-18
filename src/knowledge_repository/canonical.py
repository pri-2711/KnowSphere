"""Map stored MongoDB records to canonical models (``contracts.schemas``) and back.

Two graph sources exist and stay separate (the GLiNER stage rebuilds its own
collections): ``gliner`` (``concepts``, ``relationships``) and ``concept_first``
(``kg_concepts``, ``kg_relationships``, ``kg_claims``). Both become the same
canonical ``Concept`` / ``Relationship`` / ``Claim`` models, so governance,
retrieval and future interfaces (MCP) read one shape.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.schemas import (Claim, Concept, ContentUnit, Evidence, KnowledgeStatus, Provenance, Relationship,
                                   SourceRef, UnitType)
from src.knowledge_repository.projects import DEFAULT_PROJECT

GLINER, CONCEPT_FIRST = "gliner", "concept_first"
SOURCES = (GLINER, CONCEPT_FIRST)
COLLECTIONS = {
    GLINER: {"concept": "concepts", "relationship": "relationships"},
    CONCEPT_FIRST: {"concept": "kg_concepts", "relationship": "kg_relationships", "claim": "kg_claims"},
}
MAX_EVIDENCE = 25            # quotes per item read for governance (the stored record keeps all)
_ENTITY_TYPES = {"organization", "document", "system component", "entity", "standard", "artifact"}
_COMPONENT = "knowledge_extraction"


def _status(record: Mapping[str, Any]) -> KnowledgeStatus:
    try:
        return KnowledgeStatus(record.get("knowledge_status") or "candidate")
    except ValueError:
        return KnowledgeStatus.CANDIDATE


def _evidence(project_id: str, document_id: str, unit_id: str | None, page: int | None, quote: str | None,
              start: int | None, end: int | None, provenance: Provenance,
              unit_type: UnitType = UnitType.PAGE) -> Evidence | None:
    if not quote or not quote.strip() or not unit_id:
        return None
    if start is None or end is None or end <= start:
        start = end = None
    return Evidence(source=SourceRef(project_id=project_id, document_id=document_id, content_unit_id=unit_id,
                                     unit_type=unit_type, page_number=page or None, char_start=start, char_end=end),
                    quote=quote, provenance=provenance)


def _project(record: Mapping[str, Any]) -> str:
    return record.get("project_id") or DEFAULT_PROJECT


# ------------------------------------------------------------------ GLiNER graph
def gliner_concept(record: Mapping[str, Any]) -> Concept:
    project_id = _project(record)
    provenance = Provenance(component=_COMPONENT, method="gliner", model=record.get("merge_version"))
    evidence: list[Evidence] = []
    for mention in record.get("mentions", []):
        for quote in mention.get("quotes", [])[:2]:
            item = _evidence(project_id, mention["document_id"], mention.get("content_unit_id"),
                             mention.get("page_number"), quote.get("quote"), quote.get("start"), quote.get("end"),
                             provenance)
            if item is not None:
                evidence.append(item)
        if len(evidence) >= MAX_EVIDENCE:
            break
    description = record.get("description") or {}
    concept_type = record.get("concept_type") or "concept"
    return Concept(id=record["_id"], project_id=project_id, status=_status(record), evidence=evidence,
                   provenance=[provenance], kind="entity" if concept_type in _ENTITY_TYPES else "concept",
                   name=record["canonical_name"], normalized_name=record.get("normalized_name") or record["canonical_name"],
                   concept_type=concept_type, aliases=list(record.get("aliases", []))[:100],
                   description=description.get("quote") if isinstance(description, dict) else description)


def gliner_relationship(record: Mapping[str, Any]) -> Relationship:
    project_id = _project(record)
    provenance = Provenance(component=_COMPONENT, method="cooccurrence", model=record.get("version"))
    evidence = [item for item in (
        _evidence(project_id, e["document_id"], e.get("content_unit_id"), e.get("page_number"), e.get("quote"),
                  e.get("start"), e.get("end"), provenance) for e in record.get("evidence", [])[:MAX_EVIDENCE]) if item]
    confidence = record.get("confidence")
    return Relationship(id=record["_id"], project_id=project_id, status=_status(record), evidence=evidence,
                        provenance=[provenance], source_id=record["source_id"], target_id=record["target_id"],
                        relation_type=record["relation_type"], weight=record.get("weight"),
                        confidence=float(confidence) if confidence is not None else None)


# ------------------------------------------------------------------ concept-first graph
def _kg_evidence(project_id: str, item: Mapping[str, Any] | None, provenance: Provenance) -> Evidence | None:
    """Concept-first evidence cites the page (with page offsets) when it can, else the chunk."""
    if not item or not item.get("quote"):
        return None
    if item.get("content_unit_id"):
        return _evidence(project_id, item["document_id"], item["content_unit_id"], item.get("page_number"),
                         item["quote"], item.get("page_start"), item.get("page_end"), provenance,
                         UnitType.IMAGE if provenance.method == "vision" else UnitType.PAGE)
    return _evidence(project_id, item["document_id"], item.get("chunk_id"), item.get("page_number"), item["quote"],
                     item.get("chunk_start"), item.get("chunk_end"), provenance, UnitType.CHUNK)


def _kg_provenance(item: Mapping[str, Any]) -> Provenance:
    return Provenance(component=_COMPONENT, method=item.get("method") or "llm", model=item.get("model"),
                      prompt_version=item.get("prompt_version"))


def kg_concept(record: Mapping[str, Any]) -> Concept:
    project_id = _project(record)
    evidence = []
    methods: dict[str, Provenance] = {}
    for mention in record.get("mentions", []):
        provenance = _kg_provenance(mention)
        methods.setdefault(provenance.method, provenance)
        item = _kg_evidence(project_id, mention.get("evidence"), provenance)
        if item is not None and len(evidence) < MAX_EVIDENCE:
            evidence.append(item)
    description = record.get("description") or {}
    return Concept(id=record["_id"], project_id=project_id, status=_status(record), evidence=evidence,
                   provenance=list(methods.values()) or [Provenance(component=_COMPONENT, method="llm")],
                   kind="entity" if record.get("concept_type") in _ENTITY_TYPES else "concept",
                   name=record["canonical_name"], normalized_name=record["normalized_name"],
                   concept_type=record.get("concept_type") or "concept", aliases=list(record.get("aliases", [])),
                   possible_duplicates=list(record.get("possible_duplicates", [])),
                   description=description.get("text") if isinstance(description, dict) else description)


def kg_relationship(record: Mapping[str, Any]) -> Relationship:
    project_id = _project(record)
    evidence = [item for item in (_kg_evidence(project_id, e, _kg_provenance(e))
                                  for e in record.get("evidence", [])[:MAX_EVIDENCE]) if item]
    return Relationship(id=record["_id"], project_id=project_id, status=_status(record), evidence=evidence,
                        provenance=[Provenance(component=_COMPONENT, method="llm")], source_id=record["source_id"],
                        target_id=record["target_id"], relation_type=record["relation_type"],
                        weight=float(record.get("evidence_count") or len(evidence)))


def kg_claim(record: Mapping[str, Any]) -> Claim:
    """Claims are stored as canonical model dumps, plus storage fields that are dropped here."""
    data = {key: value for key, value in record.items()
            if key not in ("_id", "created_at", "knowledge_status", "governance", "graph", "document_id", "chunk_id")}
    data["id"] = record["_id"]
    data["status"] = _status(record)
    return Claim.model_validate(data)


READERS = {
    (GLINER, "concept"): gliner_concept, (GLINER, "relationship"): gliner_relationship,
    (CONCEPT_FIRST, "concept"): kg_concept, (CONCEPT_FIRST, "relationship"): kg_relationship,
    (CONCEPT_FIRST, "claim"): kg_claim,
}


# ------------------------------------------------------------------ content units
_UNIT_METHODS = {"page": "pymupdf", "chunk": "semantic-segmentation", "image": "pymupdf"}


def content_unit(record: Mapping[str, Any]) -> ContentUnit:
    unit_type = UnitType(record.get("unit_type", "page"))
    stored = record.get("provenance")
    if isinstance(stored, dict):
        provenance = Provenance.model_validate(stored)
    else:
        provenance = Provenance(component="document_intelligence",
                                method=record.get("extraction_method") or _UNIT_METHODS.get(unit_type.value, "unknown"))
    page = record.get("page_number")
    section = record.get("section_path") or ([record["section"]] if record.get("section") else [])
    return ContentUnit(id=record["_id"], project_id=_project(record), document_id=record["document_id"],
                       document_version_id=record.get("document_version_id"), unit_type=unit_type,
                       page_number=page if isinstance(page, int) and page >= 1 else None,
                       page_end=record.get("page_end"), section_path=section, text=record.get("text") or "",
                       bbox=record.get("bbox"), parent_id=record.get("parent_id"), provenance=provenance,
                       quality_status=record.get("quality_status") or "usable")
