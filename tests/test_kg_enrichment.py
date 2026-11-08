"""Pass 2: cross-document normalization, aliases, distinct concepts, relationship re-pointing."""

from __future__ import annotations

from conftest import FakeEmbedder

from src.kg.config import EnrichmentConfig
from src.kg.enrichment import enrich
from src.storage.mongodb_store import stable_id
from src.textutil import concept_key

CONFIG = EnrichmentConfig()


def _concept(name, type_="method", aliases=(), status="supported", doc="d1", page=1):
    return {"name": name, "key": concept_key(name), "type": type_, "description": f"About {name}.",
            "aliases": list(aliases), "status": status, "issues": [],
            "evidence": {"chunk_id": "c", "document_id": doc, "page_number": page, "quote": name}}


def _relation(source, type_, target, status="supported"):
    return {"source": source, "target": target, "source_key": concept_key(source), "target_key": concept_key(target),
            "type": type_, "status": status, "issues": [], "evidence": None}


def _chunk(chunk_id, document_id, concepts, relationships=(), status="extracted"):
    return {"_id": chunk_id, "document_id": document_id, "page_number": 1,
            "extraction": {"status": status, "concepts": concepts, "relationships": list(relationships)}}


def _by_name(concepts):
    return {concept["canonical_name"].lower(): concept for concept in concepts}


def test_same_concept_in_two_documents_becomes_one_node_with_all_provenance() -> None:
    chunks = [_chunk("a1", "doc-a", [_concept("Supervised learning", doc="doc-a")]),
              _chunk("b1", "doc-b", [_concept("supervised learning", doc="doc-b"),
                                     _concept("Supervised Learnings", doc="doc-b")])]
    concepts, _, stats = enrich(chunks, CONFIG)
    assert len(concepts) == 1
    node = concepts[0]
    assert node["_id"] == stable_id("kg_concept", "supervisedlearning")
    assert node["document_ids"] == ["doc-a", "doc-b"] and node["mention_count"] == 3
    assert {m["chunk_id"] for m in node["mentions"]} == {"a1", "b1"}
    assert {m["name"] for m in node["mentions"]} == {"Supervised learning", "supervised learning", "Supervised Learnings"}
    assert stats["concepts_in_several_documents"] == 1


def test_alias_merge_and_relationship_re_pointing() -> None:
    chunks = [
        _chunk("a1", "doc-a", [_concept("Systems Engineering", "concept", aliases=["SE"]),
                               _concept("Requirements Analysis", "process")],
               [_relation("Systems Engineering", "includes", "Requirements Analysis")]),
        _chunk("b1", "doc-b", [_concept("SE", "concept", doc="doc-b"), _concept("Requirements Analysis", "process",
                                                                                 doc="doc-b")],
               [_relation("SE", "includes", "Requirements Analysis")]),
    ]
    concepts, relationships, stats = enrich(chunks, CONFIG)
    nodes = _by_name(concepts)
    assert set(nodes) == {"systems engineering", "requirements analysis"}
    assert "SE" in nodes["systems engineering"]["aliases"]
    assert stats["merges_by_rule"] == {"alias": 1}
    assert len(relationships) == 1
    edge = relationships[0]
    assert edge["source_id"] == nodes["systems engineering"]["_id"] and edge["relation_type"] == "includes"
    assert edge["evidence_count"] == 2 and edge["cross_document"] is True
    assert edge["document_ids"] == ["doc-a", "doc-b"]


def test_ambiguous_alias_merges_nothing() -> None:
    chunks = [_chunk("a1", "doc-a", [_concept("Systems Engineering", "concept", aliases=["SE"])]),
              _chunk("b1", "doc-b", [_concept("Software Engineering", "concept", aliases=["SE"])]),
              _chunk("c1", "doc-c", [_concept("SE", "concept")])]
    concepts, _, stats = enrich(chunks, CONFIG)
    assert len(concepts) == 3 and stats["concepts_merged"] == 0
    assert any(entry["reason"] == "ambiguous alias claimed by several concepts" for entry in stats["merge_log"])


def test_related_but_distinct_concepts_stay_separate() -> None:
    names = ["supervised learning", "unsupervised learning", "design review", "preliminary design review",
             "phase 1 test", "phase 2 test"]
    chunks = [_chunk(f"c{i}", "doc-a", [_concept(name)]) for i, name in enumerate(names)]
    embedder = FakeEmbedder({"unsupervised learning": "supervised learning",
                             "preliminary design review": "design review", "phase 2 test": "phase 1 test"})
    concepts, relationships, stats = enrich(chunks, CONFIG, embedder)
    assert len(concepts) == len(names) and stats["concepts_merged"] == 0
    reasons = {entry["reason"] for entry in stats["merge_log"]}
    assert {"negation", "one name is a sub-phrase of the other", "different numbers"} <= reasons
    assert relationships == []      # similar names never create an edge by themselves


def test_spelling_variants_merge_with_fuzzy_matching() -> None:
    chunks = [_chunk("a1", "doc-a", [_concept("Environmental control"), _concept("Environmental control")]),
              _chunk("b1", "doc-b", [_concept("Enviromental control")])]   # OCR/spelling slip
    concepts, _, stats = enrich(chunks, CONFIG)
    assert len(concepts) == 1 and stats["merges_by_rule"] == {"fuzzy": 1}
    assert "Enviromental control" in concepts[0]["aliases"]


def test_failed_chunks_rejected_items_and_self_loops_are_left_out() -> None:
    chunks = [
        _chunk("a1", "doc-a", [_concept("Thermal control"), _concept("Heat pipe", "artifact"),
                               _concept("Ghost", status="rejected")],
               [_relation("Heat pipe", "example_of", "Thermal control"),
                _relation("Heat pipe", "uses", "Ghost", status="rejected")]),
        _chunk("a2", "doc-a", [_concept("Never stored")], status="failed"),
        _chunk("b1", "doc-b", [_concept("Thermal Controls"), _concept("thermal control")],
               [_relation("Thermal Controls", "related_to", "thermal control")]),
    ]
    concepts, relationships, stats = enrich(chunks, CONFIG)
    assert set(_by_name(concepts)) == {"thermal control", "heat pipe"}
    assert [(r["relation_type"]) for r in relationships] == ["example_of"]
    assert stats["self_loops_after_merge"] == 1
