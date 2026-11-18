"""Canonical resolution of an abbreviation and its full form (NASA) in both graphs, and the governed concept
detail served for a node. Models are scripted (SIMULATED); no real inference runs."""

from __future__ import annotations

import pytest
from conftest import FakeEmbedder
from fastapi.testclient import TestClient
from governed_fixture import CONFIG, FULL, PAGES, ScriptedText
from kg_fakes import insert_corpus, page_id

from src.api.app import create_app
from src.knowledge_extraction.merging import cluster_candidates, defined_abbreviations, load_candidates, merge_concepts
from src.knowledge_extraction.relationships import store_relationships
from src.knowledge_repository.kg_store import CONCEPTS, RELATIONSHIPS
from src.orchestration.kg_pipeline import KGPipeline
from src.orchestration.services import KnowledgeApplication, NotFound
from src.textutil import concept_key


# ---------------------------------------------------------------- the definition detector
def test_definitions_are_read_from_the_text_in_both_directions_and_only_when_initials_match() -> None:
    found = defined_abbreviations([
        "The National Aeronautics and Space Administration (NASA) funds research.",
        "ICD (Interface Control Document) lists every signal. Requirements are measured in kilograms (kg).",
        "The pump (see Figure 3) and the Model-Based Systems Engineering (MBSE) approach.",
        "The station (ISS) is in orbit."])                          # letters do not match the words before
    assert found == {
        "nasa": {concept_key(FULL): f"{FULL} (NASA)"},
        "icd": {concept_key("Interface Control Document"): "ICD (Interface Control Document)"},
        "mbse": {concept_key("Model-Based Systems Engineering"): "Model-Based Systems Engineering (MBSE)"}}


# ---------------------------------------------------------------- the GLiNER graph ("extracted mentions")
def _mention(name, form, unit, document, page, count, label="organization"):
    return {"_id": f"m-{name}-{unit}", "name": name, "document_id": document, "project_id": "default",
            "content_unit_id": unit, "page_number": page, "surface_forms": [[form, count]],
            "label_scores": [[label, 0.9]], "count": count, "sentences": [[0, 40]]}


def test_extracted_mentions_graph_merges_nasa_into_its_full_form(store) -> None:
    insert_corpus(store, PAGES)
    mentions = store.database["concept_mentions"]
    mentions.insert_many([
        _mention("nasa", "NASA", page_id("doc-m", 1), "doc-m", 1, 6),
        _mention("nasa", "NASA", page_id("doc-n", 1), "doc-n", 1, 1),
        _mention(FULL.lower(), FULL, page_id("doc-n", 1), "doc-n", 1, 2),
        _mention("life support system", "life support systems", page_id("doc-n", 1), "doc-n", 1, 2, "system component"),
        _mention("life support system", "Life support systems", page_id("doc-n", 2), "doc-n", 2, 2, "system component")])
    summary = merge_concepts(store, None)
    assert summary["merges"].get("abbreviation_defined") == 1
    nodes = {c["canonical_name"]: c for c in store.concepts.find()}
    assert {name.lower() for name in nodes} == {FULL.lower(), "life support systems"}   # one agency node, not two
    agency = nodes[FULL]
    assert "NASA" in agency["aliases"] and agency["mention_count"] == 9      # every mention kept
    assert agency["document_ids"] == ["doc-m", "doc-n"]
    alias = store.database["concept_aliases"].find_one({"key": "nasa"})
    assert alias["concept_id"] == agency["_id"]                                # the abbreviation finds the node
    store_relationships(store)
    ids = {c["_id"] for c in store.concepts.find()}
    for edge in store.database["relationships"].find():
        assert edge["source_id"] in ids and edge["target_id"] in ids          # no edge points at a removed node


def test_an_abbreviation_defined_two_ways_is_never_merged(store) -> None:
    from src.knowledge_extraction.merging import Candidate
    candidates = {name: Candidate(name, count=3) for name in ("se", "systems engineering", "software engineering")}
    for candidate in candidates.values():
        candidate.labels["engineering concept"] = 1.0
        candidate.forms[candidate.name] = 3
    definitions = defined_abbreviations(["Systems Engineering (SE) balances risk.",
                                         "Software Engineering (SE) writes code."])
    clusters, log = cluster_candidates(candidates, None, definitions)
    assert sorted(len(c) for c in clusters) == [1, 1, 1]
    assert {e["reason"] for e in log if e["rule"] == "abbreviation_defined"} == {
        "ambiguous: defined with several full forms in this project"}


# ---------------------------------------------------------------- the governed graph
@pytest.fixture
def governed(store, metadata):
    insert_corpus(store, PAGES)
    model = ScriptedText()
    KGPipeline(store, CONFIG, model, FakeEmbedder()).run(["doc-m", "doc-n"])
    return KnowledgeApplication(store, metadata, lambda: None), model


def test_governed_graph_has_one_canonical_node_with_aliases_evidence_and_relationships(governed, store) -> None:
    app, model = governed
    nodes = {c["canonical_name"]: c for c in store.database[CONCEPTS].find()}
    assert "NASA" not in nodes and FULL in nodes                               # merged, not relabelled
    agency = nodes[FULL]
    assert agency["aliases"] == ["NASA"] and agency["document_ids"] == ["doc-m", "doc-n"]
    assert agency["merged_from"] == [{"key": "nasa", "name": "NASA", "rule": "abbreviation_defined"}]
    assert {(m["document_id"], m["page_number"], m["evidence"]["quote"], m["method"], m["model"])
            for m in agency["mentions"]} == {
        ("doc-n", 1, "The National Aeronautics and Space Administration (NASA) funds research", "llm", "scripted-fixture"),
        ("doc-m", 1, "NASA requires two independent oxygen sources", "llm", "scripted-fixture")}
    edges = list(store.database[RELATIONSHIPS].find())
    assert {e["relation_type"] for e in edges} == {"defines", "related_to"}
    assert {e["source_id"] for e in edges} == {agency["_id"]}                  # both re-pointed to the canonical node
    defines = next(e for e in edges if e["relation_type"] == "defines")
    assert defines["evidence"][0]["stated_as"] == "NASA defines Oxygen source"  # what the source said, kept
    chunks = store.content_units.count_documents({"unit_type": "chunk"})
    assert model.calls == chunks == 2                                          # Pass 1 only: resolution asked nothing
    KGPipeline(store, CONFIG, None, FakeEmbedder()).build_graph()               # a rebuild without any model
    assert {c["canonical_name"] for c in store.database[CONCEPTS].find()} == set(nodes)


def test_concept_detail_by_canonical_id_shows_names_documents_and_merge_history(governed, store) -> None:
    app, _ = governed
    agency = store.database[CONCEPTS].find_one({"canonical_name": FULL})
    detail = app.concept_knowledge("default", agency["_id"], "concept_first")
    assert detail["name"] == FULL and detail["type"] == "entity" and detail["aliases"] == ["NASA"]
    assert {n["name"] for n in detail["original_names"]} == {"NASA", FULL}
    assert [(d["document_id"], d["pages"]) for d in detail["documents_detail"]] == [("doc-m", [1]), ("doc-n", [1])]
    assert detail["merge_history"][0]["rule"] == "abbreviation_defined"
    assert detail["merge_history"][0]["reason"] == f"defined in the text: {FULL} (NASA)"
    assert {e["target_name"] for e in detail["relationships"]} == {"Oxygen source", "Life support system"}
    assert detail["status"] in ("verified", "corroborated") and detail["description"]
    assert len(detail["evidence"]) == 2
    with pytest.raises(NotFound):
        app.concept_knowledge("default", "kg_concept:does-not-exist", "concept_first")


def test_a_reviewer_can_keep_the_abbreviation_apart_and_revert_it(governed, store) -> None:
    app, _ = governed
    agency = store.database[CONCEPTS].find_one({"canonical_name": FULL})
    decision = app.review_merge("default", agency["_id"], "NASA", "different", "ana", "here NASA means a program")
    assert {c["canonical_name"] for c in store.database[CONCEPTS].find()} >= {FULL, "NASA"}
    app.revert_merge("default", decision["decision"]["_id"], "ana")
    assert "NASA" not in {c["canonical_name"] for c in store.database[CONCEPTS].find()}


def test_api_serves_the_canonical_record_and_keeps_the_two_graphs_apart(governed, store, metadata) -> None:
    agency = store.database[CONCEPTS].find_one({"canonical_name": FULL})
    with TestClient(create_app(store=store, metadata=metadata, pipeline_factory=lambda: None,
                               watch_folder=None)) as client:
        governed_map = client.get("/api/graph", params={"source": "concept_first"}).json()
        labels = [e["data"]["label"] for e in governed_map["elements"] if "label" in e["data"]]
        assert FULL in labels and "NASA" not in labels
        detail = client.get(f"/api/knowledge/concepts/{agency['_id']}", params={"source": "concept_first"}).json()
        assert detail["name"] == FULL and detail["merge_history"]
        assert client.get("/api/knowledge/concepts/kg_concept:nasa", params={"source": "concept_first"}).status_code == 404
        extracted = client.get("/api/graph", params={"source": "gliner"}).json()
        assert extracted["source"] == "gliner" and extracted["total_nodes"] == 0   # no GLiNER run here: separate graph
