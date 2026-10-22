"""API smoke tests with mongomock data and a fake processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import fitz
import pytest
from conftest import page_id
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.search.engine import NOT_FOUND


class FakePipeline:
    """Records the upload and adds a document record, like a real ingest would."""

    def __init__(self, store) -> None:
        self.store = store
        self.files: list[str] = []

    def ingest_file(self, source: Path, filename: str, progress) -> None:
        progress("storing pages", 0.3)
        self.files.append(filename)
        self.store.documents.insert_one({"_id": "uploaded-doc", "source_sha256": "uploaded-doc",
                                        "filename": filename, "page_count": 0})

    def knowledge_stages(self, progress) -> None:
        progress("finding concepts", 0.6)


@pytest.fixture
def client(seeded, metadata, tmp_path):
    pdf = tmp_path / "doc-a.pdf"
    doc = fitz.open()
    for number in range(4):
        doc.new_page().insert_text((72, 100), f"Page {number + 1}. Requirements analysis is a part of systems engineering.")
    doc.save(pdf)
    doc.close()
    seeded.documents.update_one({"_id": "doc-a"}, {"$set": {"file_storage": {"uri": str(pdf)}}})
    pipeline = FakePipeline(seeded)
    app = create_app(store=seeded, metadata=metadata, pipeline_factory=lambda: pipeline)
    with TestClient(app) as test_client:
        test_client.pipeline = pipeline
        yield test_client


def test_health_stats_and_frontend(client) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    stats = client.get("/api/stats").json()
    assert stats["pages"] == 5 and stats["concepts"] > 0 and stats["relationships"] > 0
    page = client.get("/")
    assert page.status_code == 200 and "Knowledge Map" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/cytoscape.min.js").status_code == 200


def test_documents_and_metadata_edit(client) -> None:
    rows = {row["id"]: row for row in client.get("/api/documents").json()}
    assert rows["doc-a"]["status"] == "processed" and rows["doc-a"]["title"] == "Systems Handbook"
    response = client.put("/api/documents/doc-b/metadata", json={"author": "Defense Press"})
    assert response.status_code == 200 and response.json()["author"] == "Defense Press"
    assert client.put("/api/documents/missing/metadata", json={"title": "x"}).status_code == 404
    answer = client.post("/api/ask", json={"question": "verification and validation"}).json()
    assert "[Engineering Fundamentals | Defense Press | 2001, p. 1]" in {s["citation"] for s in answer["sentences"]}


def test_concept_endpoints(client) -> None:
    suggestions = client.get("/api/concepts/autocomplete", params={"q": "risk"}).json()
    assert suggestions[0]["name"].lower() == "risk management"
    found = client.get("/api/concepts/search", params={"q": "systems engineering"}).json()
    assert found[0]["name"].lower() == "systems engineering"
    detail = client.get(f"/api/concepts/{found[0]['id']}").json()
    assert detail["description"]["citation"].startswith("[")
    assert detail["related"] and detail["related"][0]["evidence"]
    assert detail["sources"] and all(source["citation"] for source in detail["sources"])
    assert client.get("/api/concepts/concept:missing").status_code == 404


def test_graph_endpoint_with_filters(client) -> None:
    full = client.get("/api/graph").json()
    nodes = [el for el in full["elements"] if "source" not in el["data"]]
    edges = [el for el in full["elements"] if "source" in el["data"]]
    assert nodes and edges and full["types"]
    only_b = client.get("/api/graph", params={"document_id": "doc-b"}).json()
    assert only_b["node_count"] < full["node_count"]
    heavy = client.get("/api/graph", params={"min_weight": 100}).json()
    assert heavy["edge_count"] == 0


def test_search_and_ask(client) -> None:
    search = client.get("/api/search", params={"q": "design review"}).json()
    assert search["results"] and search["results"][0]["highlights"]
    answer = client.post("/api/ask", json={"question": "Which review is held early in the program?"}).json()
    assert answer["answer_found"] and answer["groups"][0]["citation"] == "[Systems Handbook | Jane Doe | 2012, p. 2]"
    missing = client.post("/api/ask", json={"question": "What is the capital of France?"}).json()
    assert missing == {**missing, "answer_found": False, "message": NOT_FOUND, "groups": [], "sentences": []}
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400


def test_page_text_and_rendered_image(client) -> None:
    unit = page_id("doc-a", 1)
    page = client.get(f"/api/pages/{unit}").json()
    assert page["page_number"] == 1 and page["citation"] == "[Systems Handbook | Jane Doe | 2012, p. 1]"
    image = client.get(f"/api/pages/{unit}/image", params={"quote": "Requirements analysis is a part"})
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/api/pages/unknown/image").status_code == 404


def test_upload_runs_a_background_job(client) -> None:
    assert client.post("/api/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    fake_pdf = client.post("/api/documents/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert fake_pdf.status_code == 400
    job = client.post("/api/documents/upload",
                      files={"file": ("new.pdf", b"%PDF-1.4 minimal test bytes", "application/pdf")}).json()
    for _ in range(50):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "done" and state["progress"] == 1.0
    assert client.pipeline.files == ["new.pdf"]
    assert any(row["filename"] == "new.pdf" for row in client.get("/api/documents").json())
