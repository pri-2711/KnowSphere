"""Concept merging: hash, fuzzy and embedding rules, cases that must not merge, report, idempotency."""

from __future__ import annotations

import csv

from conftest import FakeEmbedder

from src.knowledge.merging import merge_concepts
from src.storage.mongodb_store import stable_id


def _mention(store, name, unit, label="process", count=2, forms=None, document_id="doc-a"):
    page = int(unit[1:])
    store.content_units.update_one({"_id": unit}, {"$setOnInsert": {
        "document_id": document_id, "unit_type": "page", "page_number": page, "unit_index": page,
        "text": f"{name} is described on this page."}}, upsert=True)
    store.database["concept_mentions"].insert_one({
        "_id": stable_id("concept_mention", name, unit), "name": name, "document_id": document_id,
        "content_unit_id": unit, "page_number": page, "surface_forms": [[form, 1] for form in (forms or [name])],
        "label_scores": [[label, 0.9]], "count": count, "max_score": 0.9, "sentences": [[0, 10]],
    })


def _names_by_node(store):
    return {node["normalized_name"]: sorted(node["member_names"]) for node in store.concepts.find()}


def test_alias_hash_fuzzy_and_embedding_merges(store, tmp_path) -> None:
    _mention(store, "sub-system", "u1", forms=["Sub-System"], count=5)
    _mention(store, "subsystem", "u2")
    _mention(store, "requirements analysis", "u1", count=6)
    _mention(store, "requirments analysis", "u3")                 # OCR typo
    _mention(store, "verification and validation", "u1", count=4)
    _mention(store, "validation and verification", "u2")
    embedder = FakeEmbedder({"validation and verification": "verification and validation"})
    report = tmp_path / "merge_report.csv"
    summary = merge_concepts(store, embedder, report)

    nodes = _names_by_node(store)
    assert nodes["sub-system"] == ["sub-system", "subsystem"]
    assert nodes["requirements analysis"] == ["requirements analysis", "requirments analysis"]
    assert nodes["verification and validation"] == ["validation and verification", "verification and validation"]
    assert summary["merges"] == {"alias_hash": 1, "fuzzy": 1, "embedding": 1}

    node = store.concepts.find_one({"_id": stable_id("concept", "requirements analysis")})
    assert "requirments analysis" in node["aliases"]
    assert {mention["content_unit_id"] for mention in node["mentions"]} == {"u1", "u3"}
    rows = list(csv.DictReader(report.open(encoding="utf-8")))
    assert {row["rule"] for row in rows if row["decision"] == "merged"} == {"alias_hash", "fuzzy", "embedding"}


def test_conservative_cases_must_not_merge(store, tmp_path) -> None:
    _mention(store, "design review", "u1", count=5)
    _mention(store, "preliminary design review", "u2")
    _mention(store, "functional requirement", "u1", count=4)
    _mention(store, "non-functional requirement", "u2")
    _mention(store, "acquisition office", "u1", label="organization", count=3)
    _mention(store, "acquisition process", "u2", label="process")
    _mention(store, "phase 1 review", "u1", count=3)
    _mention(store, "phase 2 review", "u2")
    # The fake embedder makes every pair look nearly identical, so only the guards can stop them.
    embedder = FakeEmbedder({"preliminary design review": "design review",
                             "non-functional requirement": "functional requirement",
                             "acquisition process": "acquisition office",
                             "phase 2 review": "phase 1 review"})
    report = tmp_path / "merge_report.csv"
    summary = merge_concepts(store, embedder, report)

    assert summary["concepts"] == 8
    assert summary["merges"] == {}
    reasons = {(row["name_a"], row["name_b"]): row["reason"]
               for row in csv.DictReader(report.open(encoding="utf-8")) if row["decision"] == "rejected"}
    assert "sub-phrase" in reasons[("design review", "preliminary design review")]
    assert reasons[("functional requirement", "non-functional requirement")] == "negation"
    assert "incompatible types" in reasons[("acquisition office", "acquisition process")]
    assert reasons[("phase 1 review", "phase 2 review")] == "different numbers"


def test_minimum_mentions_and_idempotent_rebuild(store) -> None:
    _mention(store, "design review", "u1", count=3)
    _mention(store, "one-off term", "u2", count=1)
    first = merge_concepts(store, None)
    ids_first = sorted(node["_id"] for node in store.concepts.find())
    second = merge_concepts(store, None)
    ids_second = sorted(node["_id"] for node in store.concepts.find())
    assert first["raw_names"] == 2 and first["concepts"] == 1
    assert ids_first == ids_second == [stable_id("concept", "design review")]
    assert second["stale_removed"] == 0
    assert store.database["concept_aliases"].find_one({"_id": "designreview"})["concept_id"] == ids_first[0]
