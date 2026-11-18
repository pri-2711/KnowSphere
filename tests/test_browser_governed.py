"""Browser regression test: the real app in headless Chrome (CDP), real mouse clicks on governed nodes.

Skipped when Chrome or Edge is not installed. The app runs in this process on a mongomock database
with the SIMULATED governed NASA corpus (scripted model) and a second project.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import mongomock
import pytest
from conftest import FakeEmbedder
from governed_fixture import CONFIG, FULL, PAGES, ScriptedText
from kg_fakes import insert_corpus

from src.api.app import create_app
from src.knowledge_repository.kg_store import CONCEPTS
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore
from src.knowledge_repository.projects import ProjectStore
from src.orchestration.kg_pipeline import KGPipeline

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            shutil.which("google-chrome") or "", shutil.which("chromium") or ""]
BROWSER = next((path for path in BROWSERS if path and Path(path).is_file()), None)
pytestmark = pytest.mark.skipif(BROWSER is None, reason="no Chrome or Edge for the browser test")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn
    store = MongoKnowledgeStore(mongomock.MongoClient()["browser"], Path.cwd())
    metadata = MetadataStore(tmp_path_factory.mktemp("meta") / "metadata.json")
    insert_corpus(store, PAGES)
    KGPipeline(store, CONFIG, ScriptedText(), FakeEmbedder()).run(["doc-m", "doc-n"])
    other = ProjectStore(store.database).create("Other")["_id"]
    app = create_app(store=store, metadata=metadata, pipeline_factory=lambda: None, watch_folder=None)
    port = _free_port()
    instance = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    threading.Thread(target=instance.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/", store, other
    instance.should_exit = True


class Page:
    """Minimal CDP client: evaluate JavaScript, send real mouse clicks, record requests and exceptions."""

    def __init__(self, ws):
        self.ws, self.counter, self.pending, self.events = ws, 0, {}, []

    async def reader(self):
        async for raw in self.ws:
            message = json.loads(raw)
            if message.get("id") in self.pending:
                self.pending.pop(message["id"]).set_result(message)
            else:
                self.events.append(message)

    async def send(self, method, **params):
        self.counter += 1
        future = asyncio.get_running_loop().create_future()
        self.pending[self.counter] = future
        await self.ws.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        return await asyncio.wait_for(future, 30)

    async def js(self, body):
        result = await self.send("Runtime.evaluate", expression=f"(async () => {{ {body} }})()", awaitPromise=True,
                                 returnByValue=True)
        return result["result"]["result"].get("value")

    async def click(self, x, y):
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)

    def requests(self, part):
        return [e["params"]["request"]["url"] for e in self.events
                if e.get("method") == "Network.requestWillBeSent" and part in e["params"]["request"]["url"]]

    def exceptions(self):
        return [e for e in self.events if e.get("method") == "Runtime.exceptionThrown"]


async def _session(url, scenario):
    import websockets
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="km-browser-test-")
    browser = subprocess.Popen([BROWSER, "--headless=new", f"--remote-debugging-port={port}",
                                f"--user-data-dir={profile}", "--no-first-run", "--window-size=1400,1000",
                                "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
                target = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            page = Page(ws)
            task = asyncio.create_task(page.reader())
            for domain in ("Runtime", "Network", "Page"):
                await page.send(f"{domain}.enable")
            await page.send("Page.navigate", url=url + "#map")
            await asyncio.sleep(3)
            result = await scenario(page)
            task.cancel()
            return result
    finally:
        browser.kill()
        shutil.rmtree(profile, ignore_errors=True)


WAIT = "await new Promise(r => setTimeout(r, {}));"


def test_clicking_a_governed_node_shows_its_canonical_record_and_never_a_stale_one(server) -> None:
    url, store, other = server
    canonical = store.database[CONCEPTS].find_one({"canonical_name": FULL})["_id"]

    async def scenario(page):
        await page.js("const f = document.querySelector('#filter-source'); f.value = 'concept_first'; "
                      "f.dispatchEvent(new Event('change')); " + WAIT.format(3000))
        labels = await page.js("return cy.nodes().map(n => n.data('label'));")
        where = await page.js(f"const n = cy.nodes().filter(x => x.data('label') === {json.dumps(FULL)}); "
                              "const p = n.renderedPosition(); const r = document.querySelector('#cy')"
                              ".getBoundingClientRect(); return [r.left + p.x, r.top + p.y];")
        await page.click(*where)
        await asyncio.sleep(2.5)
        panel = await page.js("return document.querySelector('#panel').innerText;")
        selected = await page.js("return cy.$(':selected').map(n => n.id());")
        # A concept request still in flight when the project changes must not fill the new project's panel.
        stale = await page.js(f"selectNode({json.dumps(canonical)}); "
                              f"const s = document.querySelector('#project-select'); s.value = {json.dumps(other)}; "
                              "s.dispatchEvent(new Event('change')); " + WAIT.format(2500) +
                              "return document.querySelector('#panel').innerText;")
        extracted = await page.js("const g = document.querySelector('#filter-source'); g.value = 'gliner'; "
                                  "g.dispatchEvent(new Event('change')); " + WAIT.format(2000) +
                                  "return document.querySelector('#graph-info').innerText;")
        return labels, panel, selected, stale, extracted, page.requests("/api/knowledge/concepts/"), page.exceptions()

    labels, panel, selected, stale, extracted, requests, exceptions = asyncio.run(_session(url, scenario))
    assert FULL in labels and "NASA" not in labels                              # one canonical node
    assert selected == [canonical]                                             # identified by ID, not label
    assert any(canonical.replace(":", "%3A") in r and "source=concept_first" in r for r in requests)
    for expected in (FULL, "NASA", "entity", "abbreviation defined in the text", f"{FULL} (NASA)",
                     "Oxygen source", "defines", "related_to", "p. 1", "NASA requires two independent oxygen sources"):
        assert expected in panel, expected                                     # rendered from the API response
    assert FULL not in stale and "Click a concept" in stale                    # no stale concept after switching
    assert "no concepts yet" in extracted                                      # the other graph keeps its behaviour
    assert exceptions == []
