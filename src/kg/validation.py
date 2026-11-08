"""Validation of extracted concepts and relationships.

Every item ends with a status: ``supported`` (it passed every check and its quote
was found in the source), ``needs_review`` (usable but something is doubtful),
or ``rejected`` (kept in the chunk's extraction record for debugging, never
written to the graph). Rules are plain functions in the ``*_RULES`` lists; add a
function to add a check.

Local rules run per chunk on Pass 1 output. Graph rules run after Pass 2 on the
canonical concepts and relationships.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from src.kg.schema import CONCEPT_TYPES, RELATION_TYPES, Evidence, LocalConcept, LocalRelationship, best, worst
from src.kg.segmentation import chunk_offset_to_page
from src.textutil import concept_key

MAX_NAME_WORDS = 8
_LINE_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")


# ----------------------------------------------------------------- quote search
def _normalized(text: str) -> tuple[str, list[int]]:
    """Lowercase, single-spaced text with line-break hyphens joined, plus a map back to raw offsets."""
    out: list[str] = []
    index_map: list[int] = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char == "-" and i + 1 < length and text[i + 1] in "\r\n" and out and out[-1].isalnum():
            j = i + 1
            while j < length and text[j].isspace():
                j += 1
            if j < length and text[j].isalnum():
                i = j                      # "require-\nments" -> "requirements"
                continue
        if char.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                index_map.append(i)
            i += 1
            continue
        out.append(char.lower())
        index_map.append(i)
        i += 1
    return "".join(out), index_map


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quote in text, tolerating case, whitespace and line-break hyphenation. Returns raw offsets."""
    needle, _ = _normalized(quote.strip().strip('"“”'))
    needle = needle.strip()
    if len(needle) < 3:
        return None
    haystack, index_map = _normalized(text)
    position = haystack.find(needle)
    if position == -1:
        return None
    start = index_map[position]
    end = index_map[position + len(needle) - 1] + 1
    return start, end


def contains_name(text: str, names: Iterable[str]) -> bool:
    haystack, _ = _normalized(text)
    for name in names:
        needle, _ = _normalized(name)
        needle = needle.strip()
        if needle and re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"s?(?![a-z0-9])", haystack):
            return True
    return False


def make_evidence(chunk: Mapping[str, Any], quote: str) -> Evidence | None:
    found = locate(chunk["text"], quote)
    if found is None:
        return None
    start, end = found
    page = chunk_offset_to_page(chunk, start, end) or {}
    return {"chunk_id": chunk["_id"], "document_id": chunk["document_id"], "chunk_start": start, "chunk_end": end,
            "quote": chunk["text"][start:end], **page}


# ----------------------------------------------------------------- local rules
ConceptRule = Callable[[LocalConcept, Mapping[str, Any]], None]
RelationRule = Callable[[LocalRelationship, Mapping[str, Any], Mapping[str, LocalConcept]], None]


def _flag(item: dict[str, Any], status: str, issue: str) -> None:
    item["status"] = worst(item.get("status", "supported"), status)
    item.setdefault("issues", []).append(issue)


def rule_concept_name(concept: LocalConcept, chunk: Mapping[str, Any]) -> None:
    name = concept["name"]
    if len(name) < 2 or not any(char.isalpha() for char in name) or len(name.split()) > MAX_NAME_WORDS \
            or not concept["key"]:
        _flag(concept, "rejected", "name is not a usable concept name")


def rule_concept_type(concept: LocalConcept, chunk: Mapping[str, Any]) -> None:
    if concept["type"] not in CONCEPT_TYPES:
        _flag(concept, "needs_review", f"unknown concept type '{concept['type']}', stored as 'concept'")
        concept["type"] = "concept"


def rule_concept_evidence(concept: LocalConcept, chunk: Mapping[str, Any]) -> None:
    names = [concept["name"], *concept["aliases"]]
    if concept["evidence"] is None:
        if contains_name(chunk["text"], names):
            _flag(concept, "needs_review", "evidence quote not found in the chunk, but the name is")
        else:
            _flag(concept, "rejected", "neither the evidence quote nor the name occurs in the chunk")
    elif not contains_name(concept["evidence"]["quote"], names):
        _flag(concept, "needs_review", "evidence quote does not mention the concept")


def rule_concept_description(concept: LocalConcept, chunk: Mapping[str, Any]) -> None:
    if not concept["description"].strip():
        _flag(concept, "needs_review", "empty description")


CONCEPT_RULES: list[ConceptRule] = [rule_concept_name, rule_concept_type, rule_concept_evidence,
                                    rule_concept_description]


def rule_relation_type(relation: LocalRelationship, chunk: Mapping[str, Any],
                       concepts: Mapping[str, LocalConcept]) -> None:
    if relation["type"] not in RELATION_TYPES:
        _flag(relation, "rejected", f"relation type '{relation['type']}' is not in the vocabulary")


def rule_relation_endpoints(relation: LocalRelationship, chunk: Mapping[str, Any],
                            concepts: Mapping[str, LocalConcept]) -> None:
    for end in ("source", "target"):
        key = relation[f"{end}_key"]
        if key not in concepts or concepts[key]["status"] == "rejected":
            _flag(relation, "rejected", f"{end} '{relation[end]}' is not an accepted concept of this chunk")
    if relation["source_key"] and relation["source_key"] == relation["target_key"]:
        _flag(relation, "rejected", "source and target are the same concept")


def rule_relation_evidence(relation: LocalRelationship, chunk: Mapping[str, Any],
                           concepts: Mapping[str, LocalConcept]) -> None:
    names = {}
    for end in ("source", "target"):
        concept = concepts.get(relation[f"{end}_key"])
        names[end] = [relation[end]] + ([concept["name"], *concept["aliases"]] if concept else [])
    if relation["evidence"] is None:
        if contains_name(chunk["text"], names["source"]) and contains_name(chunk["text"], names["target"]):
            _flag(relation, "needs_review", "evidence quote not found in the chunk")
        else:
            _flag(relation, "rejected", "evidence quote not found and an endpoint does not occur in the chunk")
        return
    quote = relation["evidence"]["quote"]
    if not (contains_name(quote, names["source"]) and contains_name(quote, names["target"])):
        _flag(relation, "needs_review", "evidence quote does not mention both concepts")


RELATION_RULES: list[RelationRule] = [rule_relation_type, rule_relation_endpoints, rule_relation_evidence]


def validate_local(data: Mapping[str, Any], chunk: Mapping[str, Any]) -> tuple[list[LocalConcept], list[LocalRelationship]]:
    """Validate one chunk's parsed answer; duplicates are merged, every item gets a status."""
    concepts: dict[str, LocalConcept] = {}
    alias_to_key: dict[str, str] = {}
    for raw in data.get("concepts", []):
        name = re.sub(r"\s+", " ", str(raw.get("name", ""))).strip()
        aliases = [re.sub(r"\s+", " ", a).strip() for a in raw.get("aliases") or [] if a and a.strip()]
        supported = [a for a in aliases if contains_name(chunk["text"], [a]) and concept_key(a) != concept_key(name)]
        concept: LocalConcept = {
            "name": name, "key": concept_key(name), "type": str(raw.get("type", "")).strip(),
            "description": str(raw.get("description", "")).strip(), "aliases": supported,
            "unsupported_aliases": [a for a in aliases if a not in supported and concept_key(a) != concept_key(name)],
            "evidence": make_evidence(chunk, str(raw.get("evidence", ""))), "status": "supported", "issues": [],
        }
        for rule in CONCEPT_RULES:
            rule(concept, chunk)
        existing = concepts.get(concept["key"])
        if existing is None:
            concepts[concept["key"]] = concept
        else:   # duplicate in one answer: keep one item with everything it said
            existing["aliases"] = sorted(set(existing["aliases"]) | set(concept["aliases"]))
            if existing["evidence"] is None and concept["evidence"] is not None:
                existing["evidence"] = concept["evidence"]
            existing["status"] = best(existing["status"], concept["status"])
            existing.setdefault("issues", []).append("duplicate concept in the same answer merged")
    for key, concept in concepts.items():
        if concept["status"] != "rejected":
            for alias in concept["aliases"]:
                alias_to_key.setdefault(concept_key(alias), key)

    relations: dict[tuple[str, str, str], LocalRelationship] = {}
    rejected: list[LocalRelationship] = []
    for raw in data.get("relationships", []):
        source, target = str(raw.get("source", "")).strip(), str(raw.get("target", "")).strip()
        source_key = alias_to_key.get(concept_key(source), concept_key(source))
        target_key = alias_to_key.get(concept_key(target), concept_key(target))
        relation: LocalRelationship = {
            "source": source, "target": target, "source_key": source_key, "target_key": target_key,
            "type": str(raw.get("type", "")).strip(), "evidence": make_evidence(chunk, str(raw.get("evidence", ""))),
            "status": "supported", "issues": [],
        }
        for rule in RELATION_RULES:
            rule(relation, chunk, concepts)
        if relation["status"] == "rejected":
            rejected.append(relation)
            continue
        key = (source_key, relation["type"], target_key)
        if key in relations:
            relations[key]["status"] = best(relations[key]["status"], relation["status"])
        else:
            relations[key] = relation
    return list(concepts.values()), list(relations.values()) + rejected


# ----------------------------------------------------------------- graph rules
GraphRule = Callable[[dict[str, Any], dict[str, Any]], None]


def graph_rule_endpoints(relationship: dict[str, Any], context: dict[str, Any]) -> None:
    for end in ("source_id", "target_id"):
        if relationship[end] not in context["concept_ids"]:
            _flag(relationship, "rejected", f"{end} does not exist")


def graph_rule_vocabulary(relationship: dict[str, Any], context: dict[str, Any]) -> None:
    if relationship["relation_type"] not in RELATION_TYPES:
        _flag(relationship, "rejected", "relation type not in vocabulary")


def graph_rule_sources(relationship: dict[str, Any], context: dict[str, Any]) -> None:
    valid = [item for item in relationship["evidence"] if item.get("chunk_id") in context["chunk_ids"]]
    if not valid:
        _flag(relationship, "rejected", "no evidence points to an existing chunk")
    relationship["evidence"] = valid


GRAPH_RULES: list[GraphRule] = [graph_rule_endpoints, graph_rule_vocabulary, graph_rule_sources]


def validate_graph(concepts: list[dict[str, Any]], relationships: list[dict[str, Any]],
                   chunk_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Check canonical concepts and relationships; rejected relationships are dropped from the graph."""
    for concept in concepts:
        concept["mentions"] = [m for m in concept["mentions"] if m["chunk_id"] in chunk_ids]
    concepts = [concept for concept in concepts if concept["mentions"]]
    context = {"concept_ids": {concept["_id"] for concept in concepts}, "chunk_ids": chunk_ids}
    kept = []
    rejected = 0
    for relationship in relationships:
        for rule in GRAPH_RULES:
            rule(relationship, context)
        if relationship["status"] == "rejected":
            rejected += 1
        else:
            kept.append(relationship)
    counts = {"concepts_supported": sum(c["status"] == "supported" for c in concepts),
              "concepts_needs_review": sum(c["status"] == "needs_review" for c in concepts),
              "relationships_supported": sum(r["status"] == "supported" for r in kept),
              "relationships_needs_review": sum(r["status"] == "needs_review" for r in kept),
              "relationships_rejected": rejected}
    return concepts, kept, counts
