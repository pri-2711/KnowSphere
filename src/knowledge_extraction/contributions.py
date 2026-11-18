"""Knowledge proposed by agents (MCP clients) or people, turned into ordinary Pass 1 candidates.

A proposal names one content unit (a page or a semantic chunk of the project) and quotes it. It
is checked at once with the same local rules as model output (``validate_local``, ``local_claim``):
a quote that is not in the unit rejects it. Accepted proposals of a unit are combined into a
pseudo unit with an ``extraction`` and a ``claims_extraction`` record (method "agent"), which Pass 2
merges with everything else; governance then decides their status like any other candidate. So an
agent can never write accepted knowledge directly.

A relationship proposal brings its two endpoints as concepts quoted by the same evidence, because
local validation only accepts relationships between concepts of the same unit.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.contracts.ontology import Ontology
from src.contracts.schemas import Proposal
from src.evidence_quality.validation import validate_local
from src.knowledge_extraction.claims import EXTRACTED, local_claim

METHOD = "agent"


def unit_as_chunk(unit: Mapping[str, Any]) -> dict[str, Any]:
    """A stored page or chunk in the shape the local validation rules read."""
    text = unit.get("text") or ""
    spans = unit.get("spans") or [{"content_unit_id": unit["_id"], "page_number": unit.get("page_number"),
                                   "start": 0, "end": len(text), "chunk_start": 0, "chunk_end": len(text)}]
    return {"_id": unit["_id"], "document_id": unit["document_id"], "text": text, "spans": spans,
            "page_number": unit.get("page_number"), "page_end": unit.get("page_end") or unit.get("page_number"),
            "text_sha256": unit.get("text_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()}


def _answer(proposals: Iterable[Proposal], ontology: Ontology | None) -> tuple[dict[str, list], list[dict]]:
    concepts, relationships, claims = [], [], []
    for proposal in proposals:
        data = dict(proposal.data)
        if proposal.kind == "concept":
            core = ontology.core_of(str(data.get("type", ""))) if ontology else None
            concepts.append({"name": data.get("name", ""), "type": core or data.get("type", ""),
                             "description": data.get("description", ""), "aliases": data.get("aliases") or [],
                             "evidence": data.get("evidence", "")})
        elif proposal.kind == "relationship":
            relationships.append({key: data.get(key, "") for key in ("source", "type", "target", "evidence")})
            for end in ("source", "target"):
                concepts.append({"name": data.get(end, ""), "type": "concept", "description": "", "aliases": [],
                                 "evidence": data.get("evidence", "")})
        else:
            claims.append(data)
    return {"concepts": concepts, "relationships": relationships}, claims


def check(proposal: Proposal, unit: Mapping[str, Any], ontology: Ontology | None = None) -> list[str]:
    """Problems that reject a proposal before it is stored (empty when it can enter Pass 2)."""
    chunk = unit_as_chunk(unit)
    answer, claims = _answer([proposal], ontology)
    issues: list[str] = []
    if proposal.kind == "claim":
        for claim in (local_claim(raw, chunk) for raw in claims):
            if claim["status"] == "rejected":
                issues += claim["issues"]
        return issues
    concepts, relationships = validate_local(answer, chunk)
    items = relationships if proposal.kind == "relationship" else concepts
    for item in items:
        if item["status"] == "rejected":
            issues += item["issues"]
    if not items:
        issues.append("nothing usable in the proposal")
    return issues


def contribution_units(proposals: Iterable[Proposal], units: Mapping[str, Mapping[str, Any]],
                       ontology: Ontology | None = None) -> list[dict[str, Any]]:
    """One pseudo unit per content unit and agent with submitted proposals, ready for Pass 2 and claims.
    Grouping by agent keeps each mention's provenance on the agent that proposed it."""
    by_unit: dict[tuple[str, str], list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        if proposal.status == "submitted" and proposal.unit_id in units:
            by_unit[(proposal.unit_id, proposal.agent)].append(proposal)
    result = []
    now = datetime.now(timezone.utc)
    for (unit_id, agent), items in sorted(by_unit.items()):
        chunk = unit_as_chunk(units[unit_id])
        answer, claims = _answer(sorted(items, key=lambda p: p.id), ontology)
        concepts, relationships = validate_local(answer, chunk)
        base = {"method": METHOD, "model": agent, "prompt_version": "proposal-1", "text_sha256": chunk["text_sha256"],
                "status": EXTRACTED, "extracted_at": now}
        result.append({**chunk, "unit_type": "proposal",
                       "extraction": {**base, "concepts": concepts, "relationships": relationships},
                       "claims_extraction": {**base, "claims": [local_claim(raw, chunk) for raw in claims]}})
    return result
