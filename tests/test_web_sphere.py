"""The Knowledge Sphere renderers (src/web/sphere.js) run on real API responses: research answers, hybrid
search, governed concept panels. Executed with Node.js and minimal stand-ins for app.js helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_hybrid_retrieval import _concept, app  # noqa: F401  (fixture)

from src.api.app import create_app

WEB = Path(__file__).resolve().parent.parent / "src" / "web"
HELPERS = """
const NOT_FOUND = "Not found in the uploaded documents.";
const $ = () => null; let currentProject = "default";
const escapeHtml = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const highlighted = (q) => escapeHtml(q);
const citationButton = (label, unit, quote) => `<button class="citation" data-unit="${escapeHtml(unit)}">${escapeHtml(label)}</button>`;
const api = async () => { throw new Error("no network in this test"); };
"""


def _render(payloads: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    script = (WEB / "types.js").read_text(encoding="utf-8") + "\nconst Types = globalThis.KnowledgeTypes;\n" + HELPERS \
        + (WEB / "sphere.js").read_text(encoding="utf-8") + f"""
const p = {json.dumps(payloads)};
console.log(JSON.stringify({{research: Sphere.renderResearch(p.research), abstained: Sphere.renderResearch(p.abstained),
  hybrid: Sphere.renderHybrid(p.hybrid), concept: Sphere.renderKnowledgeConcept(p.concept)}}));"""
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_scripts_load_before_the_code_that_uses_them() -> None:
    """app.js calls Sphere.setup() while starting; loading sphere.js after it stopped the whole app
    (no project list, documents stuck on "Loading")."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    order = [page.index(f"/static/{name}") for name in ("types.js", "sphere.js", "app.js")]
    assert order == sorted(order)


def test_sphere_renders_real_responses_with_citations_and_escaping(app, store, metadata) -> None:
    with TestClient(create_app(store=store, metadata=metadata, pipeline_factory=lambda: None,
                               watch_folder=None)) as client:
        payloads = {
            "research": client.post("/api/research", json={"question": "oxygen per crew member in g/day"}).json(),
            "abstained": client.post("/api/research", json={"question": "Who won the 1998 world cup?"}).json(),
            "hybrid": client.get("/api/search/hybrid", params={"q": "zeolite"}).json(),
            "concept": client.get(f"/api/knowledge/concepts/{_concept(store, 'Electrolysis')}").json(),
        }
    payloads["concept"]["description"] = "<script>alert(1)</script>"          # document text is never markup
    html = _render(payloads)
    assert "Unresolved conflicts (2)" in html["research"] and "840 g/day" in html["research"]
    assert 'class="citation"' in html["research"] and "Supporting passages" in html["research"]
    assert "Not found in the uploaded documents." in html["abstained"]
    assert "zeolite" in html["hybrid"].lower() and "found by lexical" in html["hybrid"]
    assert "Interpretations (the source infers or hedges)" in html["concept"]
    assert "<script>" not in html["concept"] and "&lt;script&gt;" in html["concept"]
