"""Multi-project support: management, migration, document mapping, isolation, retrieval and safe deletion.

GLiNER, the embedding model and the LLM are mocked; PDFs are generated with PyMuPDF.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from conftest import FakeEmbedder, FakeGliner
from fastapi.testclient import TestClient
from kg_fakes import ScriptedLLM
from test_lifecycle import TEXT_A, TEXT_B, make_pdf

from src.api.app import create_app
from src.ingest.documents import delete_project
from src.ingest.watcher import FolderWatcher, process_events
from src.kg.config import ChunkingConfig, KGConfig
from src.kg.orchestrator import KGPipeline
from src.knowledge.concept_extraction import GlinerExtractor
from src.pipeline import Pipeline
from src.projects import DEFAULT_PROJECT, ProjectError, ProjectStore, migrate, project_filter
from src.search.engine import NOT_FOUND, KnowledgeService
from src.storage.mongodb_store import sha256_file

ONLY_IN_B = ("Verification and validation confirms that the product meets stakeholder needs before delivery. "
             "Verification and validation reviews every test result and every inspection record in detail.")


@pytest.fixture
def projects(store) -> ProjectStore:
    return ProjectStore(store.database)


@pytest.fixture
def pipeline(store, metadata, tmp_path) -> Pipeline:
    return Pipeline(store, metadata, extractor=GlinerExtractor(model=FakeGliner(), threshold=0.5),
                    embedder=FakeEmbedder(), upload_dir=tmp_path / "uploads", image_dir=tmp_path / "images",
                    report_dir=tmp_path / "reports")


@pytest.fixture
def two_projects(store, projects, pipeline, tmp_path):
    """Project A (the default) and project B hold the SAME PDF; B also has a document of its own."""
    shared = make_pdf(tmp_path / "shared.pdf", TEXT_A, TEXT_A)
    b = projects.create("Project B")["_id"]
    doc_a = pipeline.ingest_file(shared)["document_id"]
    doc_b = pipeline.ingest_file(shared, project_id=b)["document_id"]
    doc_b_only = pipeline.ingest_file(make_pdf(tmp_path / "vv.pdf", ONLY_IN_B, TEXT_B), project_id=b)["document_id"]
    pipeline.knowledge_stages()
    pipeline.knowledge_stages(project_id=b)
    return {"a": DEFAULT_PROJECT, "b": b, "doc_a": doc_a, "doc_b": doc_b, "doc_b_only": doc_b_only}


def _counts(store, project_id):
    names = ("documents", "content_units", "assets", "concept_mentions", "concepts", "concept_aliases",
             "relationships", "page_terms", "search_index", "extraction_state")
    return {name: store.database[name].count_documents(project_filter(project_id)) for name in names}


# ------------------------------------------------------------------ management and migration
def test_project_management(projects) -> None:
    created = projects.create("  Thermal   Study ")
    assert created["name"] == "Thermal Study" and created["_id"] != DEFAULT_PROJECT
    with pytest.raises(ProjectError):
        projects.create("thermal study")                    # names are unique, case-insensitive
    with pytest.raises(ProjectError):
        projects.create("   ")
    assert projects.rename(created["_id"], "Thermal control")["name"] == "Thermal control"
    assert [p["_id"] for p in projects.list()][0] == DEFAULT_PROJECT
    with pytest.raises(ProjectError):
        projects.remove_record(DEFAULT_PROJECT)             # the default project is protected
    with pytest.raises(ProjectError):
        projects.get("does-not-exist")


def test_migration_only_adds_the_default_project(store) -> None:
    store.documents.insert_one({"_id": "old", "source_sha256": "old", "filename": "old.pdf", "page_count": 1})
    store.content_units.insert_one({"_id": "u1", "document_id": "old", "unit_type": "page", "page_number": 1,
                                    "unit_index": 1, "text": "untouched text"})
    store.concepts.insert_one({"_id": "c1", "canonical_name": "Old concept", "project_id": "kept"})
    updated = migrate(store.database)
    assert updated["documents"] == 1 and updated["content_units"] == 1 and updated["concepts"] == 0
    assert store.content_units.find_one({"_id": "u1"}) == {
        "_id": "u1", "document_id": "old", "unit_type": "page", "page_number": 1, "unit_index": 1,
        "text": "untouched text", "project_id": DEFAULT_PROJECT}
    assert store.concepts.find_one({"_id": "c1"})["project_id"] == "kept"   # existing values are not overwritten
    assert migrate(store.database)["documents"] == 0                       # running it again changes nothing


# ------------------------------------------------------------------ document mapping
def test_same_pdf_in_two_projects_is_two_independent_documents(store, two_projects, tmp_path) -> None:
    doc_a, doc_b = two_projects["doc_a"], two_projects["doc_b"]
    assert doc_a == sha256_file(tmp_path / "shared.pdf") and doc_b != doc_a     # default keeps the plain hash
    assert store.documents.find_one({"_id": doc_a})["project_id"] == DEFAULT_PROJECT
    assert store.documents.find_one({"_id": doc_b})["project_id"] == two_projects["b"]
    for unit in store.content_units.find({"document_id": doc_b}):
        assert unit["project_id"] == two_projects["b"]
    assert {d["_id"] for d in store.documents.find(project_filter(two_projects["b"]))} == \
        {doc_b, two_projects["doc_b_only"]}


# ------------------------------------------------------------------ graph isolation
def test_identical_concepts_in_two_projects_never_merge_or_connect(store, two_projects) -> None:
    a, b = two_projects["a"], two_projects["b"]
    in_a = store.concepts.find_one({"normalized_name": "systems engineering", **project_filter(a)})
    in_b = store.concepts.find_one({"normalized_name": "systems engineering", **project_filter(b)})
    assert in_a and in_b and in_a["_id"] != in_b["_id"]                       # same name, two separate nodes
    project_of = {c["_id"]: c.get("project_id") or DEFAULT_PROJECT for c in store.concepts.find()}
    units_of = {u["_id"]: u.get("project_id") or DEFAULT_PROJECT for u in store.content_units.find()}
    for concept in store.concepts.find():
        assert all(units_of[m["content_unit_id"]] == project_of[concept["_id"]] for m in concept["mentions"])
    for edge in store.database["relationships"].find():
        assert project_of[edge["source_id"]] == project_of[edge["target_id"]] == edge["project_id"]
        for evidence in edge["evidence"]:
            assert units_of[evidence["content_unit_id"]] == edge["project_id"]
    # Aliases and index terms are separate records per project.
    aliases = list(store.database["concept_aliases"].find({"key": "systemengineering"}))
    assert sorted(alias["project_id"] for alias in aliases) == sorted([DEFAULT_PROJECT, b])   # one record each
    assert len({alias["_id"] for alias in aliases}) == 2 and len({alias["concept_id"] for alias in aliases}) == 2
    assert store.database["search_index"].count_documents({"term": "verif", "project_id": b}) == 1
    assert store.database["search_index"].count_documents({"_id": "verif"}) == 0          # not in A's index


def test_concept_first_graph_is_project_scoped(store, projects, tmp_path) -> None:
    b = projects.create("KG B")["_id"]
    text = ("Supervised learning is a type of machine learning that uses labeled data. Labeled data consists "
            "of input examples paired with the correct output, prepared by people who know the answers.")
    for project_id, document_id in ((DEFAULT_PROJECT, "kg-a"), (b, "kg-b")):
        store.documents.insert_one({"_id": document_id, "source_sha256": document_id, "filename": "same.pdf",
                                    "page_count": 1, "project_id": project_id})
        store.content_units.insert_one({"_id": f"{document_id}-p1", "document_id": document_id,
                                        "project_id": project_id, "unit_type": "page", "unit_index": 1,
                                        "page_number": 1, "text": text, "quality_status": "usable"})
    config = KGConfig(chunking=ChunkingConfig(min_chars=60, target_chars=1000, max_chars=2000))
    KGPipeline(store, config, ScriptedLLM(), FakeEmbedder()).run(["kg-a"])
    KGPipeline(store, config, ScriptedLLM(), FakeEmbedder()).run(["kg-b"])
    concepts = list(store.database["kg_concepts"].find({"normalized_name": "supervisedlearning"}))
    assert len(concepts) == 2 and concepts[0]["_id"] != concepts[1]["_id"]
    for concept in concepts:
        assert concept["document_ids"] == (["kg-a"] if concept["project_id"] == DEFAULT_PROJECT else ["kg-b"])
    project_of = {c["_id"]: c["project_id"] for c in store.database["kg_concepts"].find()}
    for edge in store.database["kg_relationships"].find():
        assert project_of[edge["source_id"]] == project_of[edge["target_id"]] == edge["project_id"]
    # Rebuilding one project leaves the other one's graph exactly as it was.
    before = list(store.database["kg_concepts"].find(project_filter(b)))
    KGPipeline(store, config, embedder=FakeEmbedder()).build_graph(DEFAULT_PROJECT)
    assert list(store.database["kg_concepts"].find(project_filter(b))) == before
    with pytest.raises(ValueError):
        KGPipeline(store, config, ScriptedLLM(), FakeEmbedder()).run(["kg-a", "kg-b"])


# ------------------------------------------------------------------ retrieval isolation
def test_search_and_answers_only_use_the_selected_project(store, metadata, two_projects) -> None:
    a, b = two_projects["a"], two_projects["b"]
    service_a, service_b = KnowledgeService(store, metadata, a), KnowledgeService(store, metadata, b)
    question = "What does verification and validation confirm?"
    answer_b = service_b.ask(question)
    assert answer_b["answer_found"]
    assert {s["document_id"] for s in answer_b["sentences"]} <= {two_projects["doc_b"], two_projects["doc_b_only"]}
    assert service_a.ask(question)["message"] == NOT_FOUND                     # B's content is invisible in A
    assert all(r["document_id"] == two_projects["doc_a"] for r in service_a.search("systems engineering")["results"])
    concept_in_b = next(iter(service_b.concepts))
    assert service_a.concept_detail(concept_in_b) is None                      # no cross-project lookup
    assert not set(service_a.pages) & set(service_b.pages)
    assert service_a.stats()["documents"] == 1 and service_b.stats()["documents"] == 2


# ------------------------------------------------------------------ changes stay inside one project
def test_deleting_a_document_only_changes_its_project(store, pipeline, two_projects) -> None:
    before_a = _counts(store, two_projects["a"])
    pipeline.delete(two_projects["doc_b_only"])
    pipeline.knowledge_stages(project_id=two_projects["b"])
    assert _counts(store, two_projects["a"]) == before_a
    assert store.documents.count_documents(project_filter(two_projects["b"])) == 1


def test_deleting_a_project_removes_only_that_project(store, metadata, two_projects, tmp_path) -> None:
    a, b = two_projects["a"], two_projects["b"]
    before_a = _counts(store, a)
    concepts_a = list(store.concepts.find(project_filter(a)))
    result = delete_project(store, metadata, b, tmp_path / "uploads", tmp_path / "images")
    assert result["documents_deleted"] == 2
    assert all(count == 0 for count in _counts(store, b).values())
    assert _counts(store, a) == before_a and list(store.concepts.find(project_filter(a))) == concepts_a
    assert store.database["projects"].find_one({"_id": b}) is None
    assert not (tmp_path / "uploads" / two_projects["doc_b"][:16]).exists()
    assert (tmp_path / "uploads" / two_projects["doc_a"][:16]).exists()
    with pytest.raises(ProjectError):
        delete_project(store, metadata, DEFAULT_PROJECT, tmp_path / "uploads", tmp_path / "images")
    assert store.documents.count_documents(project_filter(a)) == 1


def test_watched_subfolder_goes_to_its_project(store, projects, pipeline, tmp_path) -> None:
    b = projects.create("Watched B")["_id"]
    inbox = tmp_path / "inbox"
    (inbox / b).mkdir(parents=True)
    make_pdf(inbox / b / "notes.pdf", TEXT_A)
    make_pdf(inbox / "top.pdf", TEXT_B)
    watcher = FolderWatcher(store, inbox)
    watcher.scan()
    results = process_events(pipeline, watcher, watcher.scan())
    assert {(r["path"], r["project_id"]) for r in results} == {(f"{b}/notes.pdf", b), ("top.pdf", DEFAULT_PROJECT)}
    assert store.documents.find_one({"filename": "notes.pdf"})["project_id"] == b
    assert store.documents.find_one({"filename": "top.pdf"})["project_id"] == DEFAULT_PROJECT


# ------------------------------------------------------------------ API
class _ApiPipeline:
    """Fake pipeline for the API: records which project each call was for."""

    def __init__(self, store) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []

    def ingest_file(self, source: Path, filename: str, progress, project_id: str = DEFAULT_PROJECT) -> None:
        self.calls.append(("ingest", project_id))

    def knowledge_stages(self, progress, project_id: str = DEFAULT_PROJECT) -> None:
        self.calls.append(("stages", project_id))


def _wait(client, job_id):
    for _ in range(100):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    return state


def test_api_projects_and_isolation(store, metadata, two_projects, tmp_path) -> None:
    fake = _ApiPipeline(store)
    app = create_app(store=store, metadata=metadata, pipeline_factory=lambda: fake, watch_folder=None)
    a, b = two_projects["a"], two_projects["b"]
    with TestClient(app) as client:
        listed = {p["id"]: p for p in client.get("/api/projects").json()}
        assert listed[a]["is_default"] and listed[b]["documents"] == 2
        created = client.post("/api/projects", json={"name": "API project"}).json()
        assert client.post("/api/projects", json={"name": "api PROJECT"}).status_code == 400
        assert client.put(f"/api/projects/{created['id']}", json={"name": "Renamed"}).json()["name"] == "Renamed"

        assert {d["id"] for d in client.get("/api/documents", params={"project": b}).json()} == \
            {two_projects["doc_b"], two_projects["doc_b_only"]}
        assert client.get("/api/documents", params={"project": "nope"}).status_code == 404
        concept_b = client.get("/api/concepts/autocomplete", params={"q": "verif", "project": b}).json()[0]["id"]
        assert client.get(f"/api/concepts/{concept_b}", params={"project": b}).status_code == 200
        assert client.get(f"/api/concepts/{concept_b}", params={"project": a}).status_code == 404
        assert client.get("/api/concepts/autocomplete", params={"q": "verif", "project": a}).json() == []
        # A document cannot be changed through another project.
        assert client.delete(f"/api/documents/{two_projects['doc_b']}", params={"project": a}).status_code == 404
        answer = client.post("/api/ask", params={"project": a}, json={"question": "verification and validation"}).json()
        assert answer["message"] == NOT_FOUND

        job = client.post("/api/documents/upload", params={"project": created["id"]},
                          files={"file": ("new.pdf", b"%PDF-1.4 test", "application/pdf")}).json()
        assert _wait(client, job["id"])["status"] == "done"
        assert fake.calls == [("ingest", created["id"]), ("stages", created["id"])]

        assert client.delete(f"/api/projects/{a}").status_code == 400
        job = client.delete(f"/api/projects/{b}").json()
        assert _wait(client, job["id"])["status"] == "done"
        assert b not in {p["id"] for p in client.get("/api/projects").json()}
        assert client.get("/api/stats", params={"project": a}).json()["documents"] == 1
