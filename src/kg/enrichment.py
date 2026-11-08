"""Pass 2: cross-document enrichment.

Local concepts from all chunks (all documents) are grouped by their normalized
name key, then groups are merged when they are the same idea:

1. ``alias``: a group's name is an alias another passage gave for a concept
   ("Systems Engineering (SE)"). An alias claimed for two different concepts is
   ambiguous and merges nothing.
2. ``fuzzy``: spelling or OCR variants (rapidfuzz ratio).
3. ``embedding``: form variants found by name embeddings (fastembed), optional.

Fuzzy and embedding merges use the same conservative guards as the existing
merging stage (types, numbers, negation, sub-phrases, word variants) and must
also hold between the two clusters' canonical names. Related but distinct
concepts ("design review" vs "preliminary design review") stay separate.

Relationships are never created here. Pass 1 relationships are re-pointed to the
merged concepts and duplicates are combined, keeping every piece of evidence.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.kg.config import EnrichmentConfig
from src.kg.schema import best
from src.knowledge.merging import _digits, _negates, _only_variants
from src.projects import DEFAULT_PROJECT, scoped_id
from src.textutil import concept_key, normalize_concept, singularize, stem

# Types that may describe the same thing; others merge only with themselves.
TYPE_GROUPS = {"concept": "idea", "principle": "idea", "method": "activity", "process": "activity",
               "entity": "entity", "artifact": "artifact", "standard": "artifact"}


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Any:
        """(n, d) array of L2-normalized vectors."""


@dataclass
class Group:
    """All local mentions that share one normalized name key."""
    key: str
    mentions: list[dict[str, Any]] = field(default_factory=list)
    surfaces: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    alias_keys: set[str] = field(default_factory=set)
    alias_surfaces: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Most used form after folding case and plurals; shown as written, preferring singular and capitalized."""
        folded: Counter = Counter()
        for surface, count in self.surfaces.items():
            folded[normalize_concept(surface)] += count
        form = sorted(folded.items(), key=lambda item: (-item[1], item[0]))[0][0]
        originals = [(surface, count) for surface, count in self.surfaces.items() if normalize_concept(surface) == form]
        return sorted(originals, key=lambda item: (-item[1], item[0].lower() != form, not item[0][:1].isupper(),
                                                   item[0]))[0][0]

    @property
    def concept_type(self) -> str:
        return sorted(self.types.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @property
    def words(self) -> set[str]:
        return {singularize(word) for word in re.findall(r"[a-z0-9]+", self.name.lower().replace("-", " "))}


def collect_groups(chunks: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Group], list[dict[str, Any]]]:
    """Accepted local concepts grouped by key, plus all accepted local relationships with their chunk."""
    groups: dict[str, Group] = {}
    relations: list[dict[str, Any]] = []
    for chunk in chunks:
        record = chunk.get("extraction") or {}
        if record.get("status") != "extracted":
            continue
        for concept in record.get("concepts", []):
            if concept["status"] == "rejected":
                continue
            group = groups.setdefault(concept["key"], Group(concept["key"]))
            evidence = concept.get("evidence")
            group.mentions.append({
                "chunk_id": chunk["_id"], "document_id": chunk["document_id"],
                "page_number": evidence["page_number"] if evidence and evidence.get("page_number") else chunk["page_number"],
                "name": concept["name"], "type": concept["type"], "description": concept["description"],
                "evidence": evidence, "status": concept["status"], "issues": concept.get("issues", []),
            })
            group.surfaces[concept["name"]] += 1
            group.types[concept["type"]] += 1
            for alias in concept.get("aliases", []):
                group.alias_keys.add(concept_key(alias))
                group.alias_surfaces.add(alias)
        for relation in record.get("relationships", []):
            if relation["status"] != "rejected":
                relations.append({**relation, "chunk_id": chunk["_id"], "document_id": chunk["document_id"],
                                  "page_number": chunk["page_number"]})
    return groups, relations


def guard(a: Group, b: Group, rule: str) -> str | None:
    """Why two groups must not merge, or None."""
    if rule != "alias" and TYPE_GROUPS.get(a.concept_type) != TYPE_GROUPS.get(b.concept_type):
        return f"incompatible types ({a.concept_type} vs {b.concept_type})"
    if _digits(a.name.lower()) != _digits(b.name.lower()):
        return "different numbers"
    words_a, words_b = a.words, b.words
    if _negates(words_a, words_b) or _negates(words_b, words_a):
        return "negation"
    if rule == "alias":
        return None
    if words_a != words_b and (words_a < words_b or words_b < words_a):
        return "one name is a sub-phrase of the other"
    if rule == "embedding":
        if not ({stem(w) for w in words_a} & {stem(w) for w in words_b}):
            return "no shared word"
        if not _only_variants(words_a - words_b, words_b - words_a):
            return "the differing words are not variants of each other"
    return None


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}
        self.rank = {key: index for index, key in enumerate(keys)}   # lower rank = more mentions

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        keep, drop = sorted((root_a, root_b), key=lambda key: self.rank[key])
        self.parent[drop] = keep


# ---------------------------------------------------------------- matchers
def alias_candidates(groups: Mapping[str, Group]) -> tuple[list[tuple[str, float, str, str]], list[dict[str, Any]]]:
    claims: dict[str, set[str]] = defaultdict(set)
    for group in groups.values():
        for alias_key in group.alias_keys:
            if alias_key in groups and alias_key != group.key:
                claims[alias_key].add(group.key)
    candidates, rejected = [], []
    for alias_key, owners in sorted(claims.items()):
        if len(owners) > 1:   # the same short name means different things in different places
            rejected.append({"rule": "alias", "score": 1.0, "a": groups[alias_key].name,
                             "b": ", ".join(sorted(groups[owner].name for owner in owners)),
                             "decision": "rejected", "reason": "ambiguous alias claimed by several concepts"})
            continue
        candidates.append(("alias", 1.0, next(iter(owners)), alias_key))
    return candidates, rejected


def fuzzy_candidates(groups: Mapping[str, Group], config: EnrichmentConfig) -> list[tuple[str, float, str, str]]:
    from rapidfuzz import fuzz, process
    keys = [key for key, group in groups.items() if len(group.name) >= config.fuzzy_min_length]
    if len(keys) < 2:
        return []
    names = [groups[key].name.lower() for key in keys]
    matrix = process.cdist(names, names, scorer=fuzz.ratio, score_cutoff=config.fuzzy_threshold, workers=-1)
    rows, cols = matrix.nonzero()
    return [("fuzzy", float(matrix[r, c]), keys[r], keys[c]) for r, c in zip(rows.tolist(), cols.tolist()) if r < c]


def embedding_candidates(groups: Mapping[str, Group], embedder: Embedder,
                         config: EnrichmentConfig) -> tuple[list[tuple[str, float, str, str]], dict[str, Any]]:
    import numpy as np
    keys = list(groups)
    if len(keys) < 2:
        return [], {}
    vectors = embedder.embed([groups[key].name.lower() for key in keys])
    by_key = {key: vectors[index] for index, key in enumerate(keys)}
    candidates = []
    for start in range(0, len(keys), 1024):
        scores = vectors[start:start + 1024] @ vectors.T
        rows, cols = np.nonzero(scores >= config.embedding_threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if start + r < c:
                candidates.append(("embedding", float(scores[r, c]), keys[start + r], keys[c]))
    return candidates, by_key


def cluster(groups: Mapping[str, Group], config: EnrichmentConfig,
            embedder: Embedder | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Map every group key to its cluster's canonical key, and log every merge decision."""
    # Canonical choice: never a name another passage calls an alias ("SE"), then most mentions,
    # then the more complete (longer) name.
    alias_names = {alias_key for group in groups.values() for alias_key in group.alias_keys if alias_key != group.key}
    order = sorted(groups, key=lambda key: (key in alias_names, -len(groups[key].mentions),
                                            -len(groups[key].name), key))
    union = _UnionFind(order)
    alias, log = alias_candidates(groups)
    candidates = alias + sorted(fuzzy_candidates(groups, config), key=lambda c: -c[1])
    vectors: dict[str, Any] = {}
    if embedder is not None and config.use_embeddings:
        found, vectors = embedding_candidates(groups, embedder, config)
        candidates += sorted(found, key=lambda c: -c[1])
    for rule, score, a, b in candidates:
        root_a, root_b = union.find(a), union.find(b)
        if root_a == root_b:
            continue
        reason = guard(groups[a], groups[b], rule)
        if reason is None and (root_a, root_b) != (a, b):
            reason = guard(groups[root_a], groups[root_b], rule)
            if reason is None and rule == "fuzzy":
                from rapidfuzz import fuzz
                if fuzz.ratio(groups[root_a].name.lower(), groups[root_b].name.lower()) < config.fuzzy_threshold:
                    reason = "canonical names are not similar enough"
            if reason is None and rule == "embedding" and \
                    float(vectors[root_a] @ vectors[root_b]) < config.embedding_threshold:
                reason = "canonical names are not similar enough"
            if reason:
                reason = f"cluster check: {reason}"
        log.append({"rule": rule, "score": round(score, 4), "a": groups[a].name, "b": groups[b].name,
                    "decision": "rejected" if reason else "merged", "reason": reason or ""})
        if not reason:
            union.union(a, b)
    return {key: union.find(key) for key in groups}, log


# ---------------------------------------------------------------- build graph records
def _choose_description(mentions: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = sorted((m for m in mentions if m["description"]),
                    key=lambda m: (m["status"] != "supported", m["evidence"] is None, m["document_id"],
                                   m["page_number"], m["chunk_id"]))
    if not ranked:
        return None
    m = ranked[0]
    return {"text": m["description"], "generated": True, "chunk_id": m["chunk_id"], "document_id": m["document_id"],
            "page_number": m["page_number"], "evidence": m["evidence"]}


def build_concepts(groups: Mapping[str, Group], canonical_of: Mapping[str, str],
                   log: list[dict[str, Any]], project_id: str = DEFAULT_PROJECT) -> list[dict[str, Any]]:
    members: dict[str, list[str]] = defaultdict(list)
    for key, root in canonical_of.items():
        members[root].append(key)
    merged_rules = {}
    for entry in log:
        if entry["decision"] == "merged":
            merged_rules.setdefault(entry["b"], entry["rule"])
            merged_rules.setdefault(entry["a"], entry["rule"])
    concepts = []
    for root, keys in members.items():
        canonical = groups[root]
        mentions = sorted((m for key in keys for m in groups[key].mentions),
                          key=lambda m: (m["document_id"], m["page_number"], m["chunk_id"]))
        types: Counter = Counter()
        surfaces: set[str] = set()
        for key in keys:
            types.update(groups[key].types)
            surfaces |= set(groups[key].surfaces) | groups[key].alias_surfaces
        name = canonical.name
        seen = {name.lower()}
        aliases = []
        for surface in sorted(surfaces, key=str.lower):
            if surface.lower() not in seen:
                seen.add(surface.lower())
                aliases.append(surface)
        concepts.append({
            "_id": scoped_id("kg_concept", project_id, root),
            "canonical_name": name,
            "normalized_name": root,
            "concept_type": sorted(types.items(), key=lambda item: (-item[1], item[0]))[0][0],
            "aliases": aliases,
            "description": _choose_description(mentions),
            "mentions": mentions,
            "mention_count": len(mentions),
            "document_ids": sorted({m["document_id"] for m in mentions}),
            "status": best(*(m["status"] for m in mentions)),
            "merged_from": [{"key": key, "name": groups[key].name, "rule": merged_rules.get(groups[key].name, "name")}
                            for key in sorted(keys) if key != root],
        })
    return concepts


def build_relationships(relations: list[dict[str, Any]], canonical_of: Mapping[str, str],
                        project_id: str = DEFAULT_PROJECT) -> tuple[list[dict[str, Any]], dict[str, int]]:
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved = self_loops = 0
    for relation in relations:
        source_root, target_root = canonical_of.get(relation["source_key"]), canonical_of.get(relation["target_key"])
        if source_root is None or target_root is None:
            unresolved += 1
            continue
        if source_root == target_root:
            self_loops += 1     # both ends were merged into one concept
            continue
        source_id, target_id = scoped_id("kg_concept", project_id, source_root), scoped_id("kg_concept", project_id, target_root)
        key = (source_id, relation["type"], target_id)
        evidence = relation.get("evidence") or {"chunk_id": relation["chunk_id"], "document_id": relation["document_id"],
                                                "page_number": relation["page_number"], "quote": None}
        item = {**evidence, "status": relation["status"], "stated_as": f"{relation['source']} {relation['type']} {relation['target']}"}
        edge = edges.get(key)
        if edge is None:
            edges[key] = {"_id": scoped_id("kg_relationship", project_id, source_id, relation["type"], target_id),
                          "source_id": source_id, "target_id": target_id, "relation_type": relation["type"],
                          "status": relation["status"], "evidence": [item]}
        else:
            edge["status"] = best(edge["status"], relation["status"])
            edge["evidence"].append(item)
    for edge in edges.values():
        edge["evidence"].sort(key=lambda e: (e["status"] != "supported", e["document_id"], e["page_number"] or 0))
        edge["document_ids"] = sorted({e["document_id"] for e in edge["evidence"]})
        edge["cross_document"] = len(edge["document_ids"]) > 1
        edge["evidence_count"] = len(edge["evidence"])
    return list(edges.values()), {"unresolved_endpoints": unresolved, "self_loops_after_merge": self_loops}


def enrich(chunks: Iterable[Mapping[str, Any]], config: EnrichmentConfig, embedder: Embedder | None = None,
           project_id: str = DEFAULT_PROJECT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run Pass 2 over one project's extracted chunks. Returns (concepts, relationships, stats incl. merge log).

    The caller passes only that project's chunks; IDs are project-scoped, so the same name in another
    project is a different concept and can never be merged or connected with this one.
    """
    groups, relations = collect_groups(chunks)
    canonical_of, log = cluster(groups, config, embedder)
    concepts = build_concepts(groups, canonical_of, log, project_id)
    relationships, edge_stats = build_relationships(relations, canonical_of, project_id)
    merges = Counter(entry["rule"] for entry in log if entry["decision"] == "merged")
    stats = {
        "local_mentions": sum(len(group.mentions) for group in groups.values()),
        "distinct_names": len(groups),
        "canonical_concepts": len(concepts),
        "concepts_merged": sum(merges.values()),
        "merges_by_rule": dict(merges),
        "merges_rejected": sum(1 for entry in log if entry["decision"] == "rejected"),
        "concepts_in_several_documents": sum(1 for concept in concepts if len(concept["document_ids"]) > 1),
        "relationships": len(relationships),
        "cross_document_relationships": sum(1 for edge in relationships if edge["cross_document"]),
        **edge_stats,
        "merge_log": log,
    }
    return concepts, relationships, stats
