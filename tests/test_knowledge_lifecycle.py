"""One knowledge lifecycle, verified through the defined interfaces (all models are fakes).

ingestion (no model) -> structure -> text, claim and visual candidates -> enrichment -> governance
-> repository -> retrieval, API and CLI-facing application services; plus versions and stage recovery.
"""

from __future__ import annotations

import json
import time

import pytest
from conftest import FakeEmbedder, FakeGliner
from fastapi.testclient import TestClient
from kg_fakes import ANSWERS, DOC_X, DOC_Y, insert_corpus, page_id
from test_lifecycle import TEXT_A, TEXT_B, make_pdf

from src.api.app import create_app
from src.config import ChunkingConfig, KGConfig
from src.contracts.interfaces import LLMResponse
from src.evidence_quality.service import validate_project
from src.model_runtime.gliner import GlinerExtractor
from src.knowledge_repository.canonical import CONCEPT_FIRST, GLINER
from src.knowledge_repository.kg_store import CLAIMS, CONCEPTS, RELATIONSHIPS
from src.knowledge_repository.repository import MongoKnowledgeRepository
from src.orchestration.kg_pipeline import KGPipeline
from src.orchestration.pipeline import Pipeline
from src.orchestration.services import KnowledgeApplication, NotFound
from src.retrieval.engine import KnowledgeService

CONFIG = KGConfig(chunking=ChunkingConfig(min_chars=60, target_chars=1000, max_chars=2000), max_retries=1)
CLAIMS_X = [
    {"subject": "Supervised learning", "predicate": "uses", "object": "labeled data",
     "statement": "Supervised learning uses labeled data.", "polarity": "affirmed",
     "evidence": "Supervised learning is a type of machine learning that uses labeled data."},
    {"subject": "Supervised learning", "predicate": "needs", "object": "500 examples",     # number not in the text
     "statement": "Supervised learning needs 500 examples.", "polarity": "affirmed",
     "evidence": "Supervised learning is a type of machine learning"},
]


class FakeTextModel:
    """One text model for both concept and claim prompts (the schema tells which answer is wanted)."""
    model = "fake-qwen"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, messages, schema):
        self.calls += 1
        passage = messages[0]["content"]
        if "claims" in schema["properties"]:
            return LLMResponse(json.dumps({"claims": CLAIMS_X if "Supervised learning is a type" in passage else []}),
                               "end_turn", self.model)
        answer = next((a for marker, a in ANSWERS.items() if marker in passage), {"concepts": [], "relationships": []})
        return LLMResponse(json.dumps(answer), "end_turn", self.model)


class FakeVision:
    model = "fake-qwen-vl"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, image_path, system, prompt, schema):
        self.calls += 1
        assert "WPA" in prompt                    # the OCR text reached the model as grounding
        answer = {"description": "A loop diagram.", "concepts": [
            {"name": "Neural network", "type": "artifact", "description": "Shown as the model box.",
             "aliases": [], "evidence": "Neural network"},
            {"name": "Water Processor Assembly", "type": "artifact", "description": "Labelled unit.",
             "aliases": ["WPA"], "evidence": "WPA"},
            {"name": "Quantum Router", "type": "artifact", "description": "Invented.", "aliases": [],
             "evidence": "Quantum Router"}], "relationships": []}
        return LLMResponse(json.dumps(answer), "end_turn", self.model)


@pytest.fixture
def corpus(store, tmp_path):
    insert_corpus(store)
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG not decoded by the fakes")
    store.content_units.insert_one({
        "_id": "image-x-1", "document_id": DOC_X, "unit_type": "image", "unit_index": 1, "page_number": 1,
        "caption": "Figure 3 Neural network training loop", "context_text": "", "text": "Figure 3",
        "image_uri": str(image), "quality_status": "usable"})
    return store


def _kg(store, text=None, vision=None):
    return KGPipeline(store, CONFIG, text or FakeTextModel(), FakeEmbedder(), vision=vision or FakeVision(),
                      ocr=lambda path: "WPA\nNeural network")


def test_text_claims_and_figures_flow_into_one_governed_graph(corpus) -> None:
    text, vision = FakeTextModel(), FakeVision()
    summary = _kg(corpus, text, vision).run([DOC_X, DOC_Y], claims=True, visual=True)
    assert summary["claims"]["claims"] == 2 and summary["visual"]["processed"] == 1

    image = corpus.content_units.find_one({"_id": "image-x-1"})
    assert image["ocr_text"] == "WPA\nNeural network" and image["interpretation"]["generated"] is True
    by_name = {c["name"]: c for c in image["extraction"]["concepts"]}
    assert by_name["Quantum Router"]["status"] == "rejected"           # not in caption, context or OCR
    assert by_name["Water Processor Assembly"]["status"] != "rejected"

    concepts = {c["canonical_name"]: c for c in corpus.database[CONCEPTS].find()}
    assert "Quantum Router" not in concepts
    network = concepts["Neural network"]
    assert {m["method"] for m in network["mentions"]} == {"llm", "vision"}   # text and figure, one concept
    assert concepts["Supervised Learning"]["knowledge_status"] == "corroborated"   # two documents
    assert concepts["Machine learning"]["knowledge_status"] == "verified"
    assert all(c.get("knowledge_status") for c in concepts.values())
    assert corpus.database[RELATIONSHIPS].find_one({"relation_type": "uses"})["knowledge_status"] == "corroborated"

    claims = {c["statement"]: c for c in corpus.database[CLAIMS].find()}
    assert claims["Supervised learning uses labeled data."]["knowledge_status"] == "verified"
    assert claims["Supervised learning uses labeled data."]["subject_concept_id"] == concepts["Supervised Learning"]["_id"]
    invented = claims["Supervised learning needs 500 examples."]
    assert invented["knowledge_status"] == "rejected"
    record = corpus.database["validation_records"].find_one({"item_id": invented["_id"]})
    assert any(c["check"] == "numeric_consistency" and not c["passed"] for c in record["checks"])

    # Rerun: nothing is sent to a model again, and nothing is duplicated.
    calls = (text.calls, vision.calls)
    counts = [corpus.database[name].count_documents({}) for name in (CONCEPTS, CLAIMS, "validation_records")]
    _kg(corpus, text, vision).run([DOC_X, DOC_Y], claims=True, visual=True)
    assert (text.calls, vision.calls) == calls
    assert [corpus.database[name].count_documents({}) for name in (CONCEPTS, CLAIMS, "validation_records")] == counts


def test_governance_runs_on_the_gliner_graph_and_retrieval_hides_rejected(seeded, metadata) -> None:
    repository = MongoKnowledgeRepository(seeded, metadata)
    result = validate_project(repository, "default", GLINER)
    assert result["concepts"] == seeded.concepts.count_documents({}) and result["schema_rejections"] == 0
    assert seeded.concepts.count_documents({"knowledge_status": {"$in": ["verified", "corroborated"]}}) > 0
    target = seeded.concepts.find_one({"canonical_name": {"$regex": "^risk management$", "$options": "i"}})
    # Break the evidence of one concept: governance must reject it and retrieval must stop showing it.
    mentions = target["mentions"]
    for mention in mentions:
        for quote in mention["quotes"]:
            quote["quote"] = "text that is not on the page"
    seeded.concepts.update_one({"_id": target["_id"]}, {"$set": {"mentions": mentions}})
    validate_project(repository, "default", GLINER)
    assert seeded.concepts.find_one({"_id": target["_id"]})["knowledge_status"] == "rejected"
    service = KnowledgeService(seeded, metadata)
    assert target["_id"] not in service.concepts and service.concept_detail(target["_id"]) is None
    visible = service.concept_detail(next(iter(service.concepts)))
    assert visible["status"] in ("verified", "corroborated", "needs_review")


def test_application_services_adjudication_isolation_and_recovery(corpus, metadata) -> None:
    _kg(corpus).run([DOC_X, DOC_Y])
    app = KnowledgeApplication(corpus, metadata, lambda: None)
    other = app.create_project("Other")["id"]
    summary = app.knowledge_summary("default")
    assert summary[CONCEPT_FIRST]["concept"].get("corroborated", 0) >= 1
    assert app.knowledge_summary(other)[CONCEPT_FIRST]["concept"] == {}

    concept = corpus.database[CONCEPTS].find_one({"canonical_name": "Machine learning"})
    with pytest.raises(NotFound):
        app.adjudicate(other, concept["_id"], "analyst", "reject")      # another project's item
    app.adjudicate("default", concept["_id"], "analyst", "reject", "too generic")
    _kg(corpus).build_graph("default")                                  # a full rebuild keeps the decision
    assert corpus.database[CONCEPTS].find_one({"_id": concept["_id"]})["knowledge_status"] == "adjudicated"
    assert app.validation_record("default", concept["_id"])["status_after"] == "adjudicated"

    def broken():
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        app.tracker.run("default", "index", broken)
    assert [s["stage"] for s in app.stage_status("default")["unfinished"]] == ["index"]
    app.tracker.run("default", "index", lambda: "ok")
    assert app.stage_status("default")["unfinished"] == []
    assert app.stage_status(other)["stages"] == []


def test_replace_records_a_version_lineage(store, metadata, tmp_path) -> None:
    pipeline = Pipeline(store, metadata, extractor=GlinerExtractor(model=FakeGliner(), threshold=0.5),
                        embedder=FakeEmbedder(), upload_dir=tmp_path / "uploads", image_dir=tmp_path / "images",
                        report_dir=tmp_path / "reports")
    first = pipeline.ingest_file(make_pdf(tmp_path / "plan.pdf", TEXT_A))["document_id"]
    second = pipeline.replace(first, make_pdf(tmp_path / "plan-v2.pdf", TEXT_B), "plan.pdf")["document_id"]
    app = KnowledgeApplication(store, metadata, lambda: pipeline)
    versions = app.document_versions("default", second)
    assert [(v["version"], v["document_id"]) for v in versions] == [(1, first), (2, second)]
    assert versions[0]["superseded_by"] == second and versions[0]["lineage_id"] == versions[1]["lineage_id"]
    assert store.documents.find_one({"_id": first}) is None             # the old version's content is gone


def _wait(client, job_id):
    for _ in range(100):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    return state


def test_api_exposes_governance_through_the_same_services(corpus, metadata, tmp_path) -> None:
    _kg(corpus).run([DOC_X, DOC_Y], claims=True)
    app = create_app(store=corpus, metadata=metadata, pipeline_factory=lambda: None, watch_folder=None)
    with TestClient(app) as client:
        summary = client.get("/api/knowledge/summary").json()
        assert summary["concept_first"]["claim"] == {"verified": 1, "rejected": 1}
        concept = corpus.database[CONCEPTS].find_one({"canonical_name": "Labeled data"})
        validation = client.get(f"/api/knowledge/{concept['_id']}/validation").json()
        assert validation["status_after"] == "corroborated" and validation["checks"][0]["check"] == "schema"
        assert client.post(f"/api/knowledge/{concept['_id']}/adjudicate",
                           json={"reviewer": "analyst", "decision": "accept"}).status_code == 200
        assert client.get(f"/api/knowledge/{concept['_id']}/validation").json()["status_after"] == "adjudicated"
        assert client.post("/api/knowledge/unknown/adjudicate",
                           json={"reviewer": "a", "decision": "accept"}).status_code == 404
        results = client.post("/api/projects/default/validate").json()
        assert {r["source"] for r in results} == {"gliner", "concept_first"}
        assert client.get("/api/projects/default/stages").json()["unfinished"] == []
        assert client.get("/api/projects/missing/stages").status_code == 404
        assert client.get(f"/api/knowledge/{concept['_id']}/validation", params={"project": "missing"}).status_code == 404
        page = client.get(f"/api/pages/{page_id(DOC_X, 1)}").json()
        assert page["page_number"] == 1
