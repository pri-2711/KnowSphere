"""Unified knowledge validation for one project and one graph source, through the repository interface."""

from __future__ import annotations

from typing import Any

from src.contracts.interfaces import KnowledgeRepository
from src.evidence_quality.governance import POLICY_VERSION, GovernanceContext, govern, schema_rejection


def validate_project(repository: KnowledgeRepository, project_id: str, source: str) -> dict[str, Any]:
    """Load canonical knowledge, run every governance check, store statuses and validation records."""
    knowledge = repository.knowledge(project_id, source)
    items = [*knowledge.concepts, *knowledge.relationships, *knowledge.claims]
    unit_ids = {evidence.source.content_unit_id for item in items for evidence in item.evidence}
    context = GovernanceContext(
        project_id=project_id, page_counts=repository.page_counts(project_id),
        units=repository.units(project_id, unit_ids), concept_ids={c.id for c in knowledge.concepts},
        concepts=knowledge.concepts, claims=knowledge.claims, relationships=knowledge.relationships,
        adjudications=repository.adjudications(project_id))
    records, statuses = govern(items, context)
    for item_id, item_type, error in knowledge.invalid:
        record = schema_rejection(project_id, item_id, item_type, error)
        records.append(record)
        statuses[item_id] = record.status_after
    saved = repository.save_validation(project_id, source, records, statuses)
    return {"project_id": project_id, "source": source, "policy_version": POLICY_VERSION,
            "concepts": len(knowledge.concepts), "relationships": len(knowledge.relationships),
            "claims": len(knowledge.claims), "schema_rejections": len(knowledge.invalid), **saved}
