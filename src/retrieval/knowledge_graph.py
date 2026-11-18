"""The governed knowledge graph for retrieval: canonical models from the repository, never rejected items.

Built from ``KnowledgeRepository.knowledge(project, source)``, so it serves either graph (the GLiNER
graph or the concept-first graph with claims) through the same contracts governance checked.

- ``accepted``: verified, corroborated, or adjudicated with "accept"; ``unconfirmed``: candidate or
  needs_review. Rejected items (by a check or by a reviewer) are left out.
- graph retrieval: concepts named in a text, their neighbourhood (multi-hop) and shortest paths,
  each step with its evidence.
- claims about a concept, grouped as facts, interpretations and hypotheses, and the conflicts
  governance found between claims or between relationships.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

import networkx as nx

from src.contracts.interfaces import KnowledgeSet
from src.contracts.schemas import Adjudication, Claim, Concept, Evidence, KnowledgeItem, KnowledgeStatus, Relationship
from src.evidence_quality.governance import find_contradictions
from src.textutil import concept_key

ACCEPTED = {KnowledgeStatus.VERIFIED, KnowledgeStatus.CORROBORATED}


def evidence_payload(evidence: Evidence) -> dict[str, Any]:
    source = evidence.source
    return {"quote": evidence.quote, "document_id": source.document_id, "content_unit_id": source.content_unit_id,
            "page_number": source.page_number, "start": source.char_start, "end": source.char_end,
            "method": evidence.provenance.method, "model": evidence.provenance.model}


class GovernedGraph:
    def __init__(self, knowledge: KnowledgeSet, adjudications: Mapping[str, Adjudication] | None = None) -> None:
        self.project_id = knowledge.project_id
        self.source = knowledge.source
        self.adjudications = dict(adjudications or {})
        shown = [c for c in knowledge.concepts if self._shown(c)]
        self.concepts: dict[str, Concept] = {c.id: c for c in shown}
        self.relationships: dict[str, Relationship] = {
            r.id: r for r in knowledge.relationships
            if self._shown(r) and r.source_id in self.concepts and r.target_id in self.concepts}
        self.claims: dict[str, Claim] = {c.id: c for c in knowledge.claims if self._shown(c)}
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(self.concepts)
        for relationship in self.relationships.values():
            self.graph.add_edge(relationship.source_id, relationship.target_id, key=relationship.id)
        self.undirected = nx.Graph(self.graph.to_undirected(as_view=True))
        names: dict[str, set[str]] = defaultdict(set)
        for concept in self.concepts.values():
            for name in {concept.name, *concept.aliases}:
                if concept_key(name):
                    names[concept_key(name)].add(concept.id)
        self.by_key = {key: next(iter(ids)) for key, ids in names.items() if len(ids) == 1}   # ambiguous left out
        self.claims_by_concept: dict[str, list[str]] = defaultdict(list)
        for claim in self.claims.values():
            for concept_id in {claim.subject_concept_id, claim.object_concept_id} - {None}:
                self.claims_by_concept[concept_id].append(claim.id)
        self.conflicts = find_contradictions(list(self.claims.values()), list(self.relationships.values()))

    # ------------------------------------------------------------ status
    def _shown(self, item: KnowledgeItem) -> bool:
        if item.status == KnowledgeStatus.REJECTED:
            return False
        adjudication = self.adjudications.get(item.id)
        return not (item.status == KnowledgeStatus.ADJUDICATED and adjudication and adjudication.decision == "reject")

    def standing(self, item: KnowledgeItem) -> str:
        """"accepted" or "unconfirmed"; rejected items are never in the graph."""
        if item.status in ACCEPTED:
            return "accepted"
        adjudication = self.adjudications.get(item.id)
        if item.status == KnowledgeStatus.ADJUDICATED and adjudication and adjudication.decision == "accept":
            return "accepted"
        return "unconfirmed"

    # ------------------------------------------------------------ lookup
    def match(self, text: str, max_words: int = 6) -> list[str]:
        """Concepts whose name or alias occurs in the text, longest names first."""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9&\-]*", text)
        found: list[str] = []
        for size in range(min(max_words, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                concept_id = self.by_key.get(concept_key(" ".join(words[start:start + size])))
                if concept_id and concept_id not in found:
                    found.append(concept_id)
        return found

    def summary(self, concept_id: str) -> dict[str, Any]:
        concept = self.concepts[concept_id]
        return {"id": concept.id, "name": concept.name, "type": concept.concept_type, "status": concept.status.value,
                "standing": self.standing(concept), "aliases": concept.aliases[:20],
                "documents": sorted(concept.document_ids), "degree": self.undirected.degree(concept_id)}

    def edge(self, relationship: Relationship) -> dict[str, Any]:
        return {"id": relationship.id, "source": relationship.source_id, "target": relationship.target_id,
                "source_name": self.concepts[relationship.source_id].name,
                "target_name": self.concepts[relationship.target_id].name, "type": relationship.relation_type, "status": relationship.status.value,
                "standing": self.standing(relationship), "conflicts": self.conflicts.get(relationship.id, []),
                "evidence": [evidence_payload(e) for e in relationship.evidence[:3]]}

    def claim(self, claim: Claim) -> dict[str, Any]:
        return {"id": claim.id, "statement": claim.statement, "subject": claim.subject, "predicate": claim.predicate,
                "object": claim.object, "kind": claim.kind, "modality": claim.modality, "polarity": claim.polarity,
                "status": claim.status.value, "standing": self.standing(claim), "uncertainty": claim.uncertainty,
                "quantities": [q.model_dump(mode="json", exclude={"schema_version"}) for q in claim.quantities],
                "conflicts": self.conflicts.get(claim.id, []),
                "evidence": [evidence_payload(e) for e in claim.evidence[:3]]}

    # ------------------------------------------------------------ graph retrieval
    def neighbourhood(self, concept_ids: Iterable[str], hops: int = 1, limit: int = 60) -> dict[str, Any]:
        """Concepts within ``hops`` of the given ones (breadth first, nearest first) and the edges between them."""
        frontier = [c for c in concept_ids if c in self.concepts]
        distance = {c: 0 for c in frontier}
        for depth in range(1, hops + 1):
            following = []
            for node in frontier:
                for neighbour in sorted(self.undirected.neighbors(node), key=lambda n: -self.undirected.degree(n)):
                    if neighbour not in distance and len(distance) < limit:
                        distance[neighbour] = depth
                        following.append(neighbour)
            frontier = following
        edges = [self.edge(self.relationships[key]) for a, b, key in self.graph.edges(keys=True)
                 if a in distance and b in distance]
        return {"nodes": [{**self.summary(n), "distance": d} for n, d in sorted(distance.items(), key=lambda x: x[1])],
                "edges": edges}

    def path(self, a: str, b: str, max_hops: int = 4) -> list[dict[str, Any]] | None:
        """The shortest connection between two concepts, each step with its relationship and evidence."""
        if a not in self.concepts or b not in self.concepts:
            return None
        try:
            nodes = nx.shortest_path(self.undirected, a, b)
        except nx.NetworkXNoPath:
            return None
        if len(nodes) - 1 > max_hops:
            return None
        steps = []
        for left, right in zip(nodes, nodes[1:]):
            keys = [(left, right, k) for k in self.graph.get_edge_data(left, right, default={})] + \
                   [(right, left, k) for k in self.graph.get_edge_data(right, left, default={})]
            best = sorted(keys, key=lambda item: (self.standing(self.relationships[item[2]]) != "accepted", item[2]))[0]
            steps.append({"from": self.summary(left), "to": self.summary(right),
                          "relationship": self.edge(self.relationships[best[2]])})
        return steps

    def evidence_units(self, concept_ids: Iterable[str]) -> list[str]:
        """Content units that evidence the concepts, in order of the concepts given."""
        units: list[str] = []
        for concept_id in concept_ids:
            for evidence in self.concepts[concept_id].evidence:
                if evidence.source.content_unit_id not in units:
                    units.append(evidence.source.content_unit_id)
        return units

    def related_units(self, concept_ids: Iterable[str]) -> list[str]:
        """Units that evidence relationships touching the concepts (one hop of graph context)."""
        wanted = set(concept_ids)
        units: list[str] = []
        for relationship in self.relationships.values():
            if relationship.source_id in wanted or relationship.target_id in wanted:
                for evidence in relationship.evidence:
                    if evidence.source.content_unit_id not in units:
                        units.append(evidence.source.content_unit_id)
        return units

    def concept_detail(self, concept_id: str) -> dict[str, Any] | None:
        concept = self.concepts.get(concept_id)
        if concept is None:
            return None
        edges = [self.edge(r) for r in self.relationships.values() if concept_id in (r.source_id, r.target_id)]
        claims = [self.claim(self.claims[c]) for c in self.claims_by_concept.get(concept_id, [])]
        return {**self.summary(concept_id), "description": concept.description,
                "uncertainty": concept.uncertainty, "evidence": [evidence_payload(e) for e in concept.evidence[:10]],
                "relationships": edges, "claims": group_claims(claims),
                "conflicts": [c for c in claims if c["conflicts"]] + [e for e in edges if e["conflicts"]]}

    def payload(self, document_id: str | None = None, concept_type: str | None = None, min_weight: float = 0.0,
                limit: int = 250, focus: str | None = None) -> dict[str, Any]:
        """Map elements in the shape of ``KnowledgeService.graph_payload`` (Cytoscape), most connected first."""
        rank = nx.degree_centrality(self.undirected) if len(self.undirected) > 1 else {}   # no SciPy needed
        nodes = [c for c in self.concepts.values()
                 if (document_id is None or document_id in c.document_ids)
                 and (concept_type is None or c.concept_type == concept_type)]
        nodes.sort(key=lambda c: (-rank.get(c.id, 0.0), -len(c.evidence), c.id))
        chosen = {c.id for c in nodes[:limit]}
        if focus in self.concepts:
            chosen.add(focus)
            chosen.update(list(self.undirected.neighbors(focus))[:40])
        elements = [{"data": {"id": c, "label": self.concepts[c].name, "type": self.concepts[c].concept_type,
                              "mentions": len(self.concepts[c].evidence), "degree": self.undirected.degree(c),
                              "importance": rank.get(c, 0.0) or 1e-6, "community": 0,
                              "status": self.concepts[c].status.value}} for c in sorted(chosen)]
        edges = 0
        for relationship in self.relationships.values():
            weight = float(len(relationship.evidence))
            if relationship.source_id in chosen and relationship.target_id in chosen and weight >= min_weight:
                elements.append({"data": {"id": relationship.id, "source": relationship.source_id,
                                          "target": relationship.target_id, "relation": relationship.relation_type,
                                          "weight": weight, "confidence": relationship.confidence,
                                          "status": relationship.status.value}})
                edges += 1
        return {"elements": elements, "node_count": len(chosen), "edge_count": edges, "source": self.source,
                "total_nodes": len(self.concepts), "total_edges": len(self.relationships),
                "types": sorted({c.concept_type for c in self.concepts.values()})}

    def all_conflicts(self) -> list[dict[str, Any]]:
        rows = []
        for item_id, descriptions in sorted(self.conflicts.items()):
            if item_id in self.claims:
                rows.append({"item_type": "claim", **self.claim(self.claims[item_id])})
            elif item_id in self.relationships:
                rows.append({"item_type": "relationship", **self.edge(self.relationships[item_id])})
        return rows


def group_claims(claims: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Accepted claims by modality (facts, interpretations, hypotheses); the rest as unconfirmed."""
    groups: dict[str, list[Mapping[str, Any]]] = {"facts": [], "interpretations": [], "hypotheses": [],
                                                  "unconfirmed": []}
    names = {"fact": "facts", "interpretation": "interpretations", "hypothesis": "hypotheses"}
    for claim in claims:
        groups["unconfirmed" if claim["standing"] != "accepted" else names[claim["modality"]]].append(claim)
    return groups
