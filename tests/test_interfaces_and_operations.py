"""MCP server (protocol, tools, scoped permissions, audit), project locks, backup and restore, structured
logging, evaluation metrics and diagnostics. All models are fakes; the MCP server runs in process."""

from __future__ import annotations

import io
import json
import logging
import threading
from datetime import datetime, timedelta, timezone

import mongomock
import pytest
from fastapi.testclient import TestClient
from kg_fakes import page_id
from test_hybrid_retrieval import _concept, app  # noqa: F401  (fixture: governed corpus with claims)

from src.api.app import create_app
from src.api.mcp_server import MCPServer
from src.knowledge_repository.backup import backup, restore
from src.knowledge_repository.locks import ProjectBusy, ProjectLocks
from src.observability import JsonFormatter
from src.orchestration.agents import AgentPolicy
from src.orchestration.evaluation import evaluate_extraction, evaluate_knowledge_citations, evaluate_retrieval

POLICY = AgentPolicy({"default": {"projects": ["default"], "scopes": ["read"]},
                      "agents": {"claude-code": {"projects": ["*"], "scopes": ["read", "propose"]},
                                 "outsider": {"projects": ["other"], "scopes": ["read"]}}})


def rpc(server, method, params=None, request_id=1):
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})


def tool(server, name, **arguments):
    result = rpc(server, "tools/call", {"name": name, "arguments": arguments})["result"]
    return json.loads(result["content"][0]["text"]), result["isError"]


# ---------------------------------------------------------------- MCP
def test_mcp_protocol_handshake_and_tool_listing(app) -> None:
    server = MCPServer(app, POLICY, "codex")
    init = rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "codex", "version": "1"}})["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["capabilities"]["tools"] == {"listChanged": False}
    assert rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in rpc(server, "tools/list")["result"]["tools"]}
    assert names == {"list_projects", "project_context", "search", "research", "get_document", "get_page",
                     "find_concepts", "get_concept", "get_claims", "get_evidence", "explore_graph", "find_path",
                     "find_contradictions", "propose_knowledge"}
    assert rpc(server, "no/such/method")["error"]["code"] == -32601
    streamed = io.StringIO()
    server.serve(io.StringIO('{"jsonrpc":"2.0","id":7,"method":"ping"}\nnot json\n'), streamed)
    lines = [json.loads(line) for line in streamed.getvalue().splitlines()]
    assert lines[0] == {"jsonrpc": "2.0", "id": 7, "result": {}} and lines[1]["error"]["code"] == -32700


def test_mcp_tools_route_through_services_with_citations(app, store) -> None:
    server = MCPServer(app, POLICY, "codex")
    found, error = tool(server, "search", query="zeolite beds")
    assert not error and found["results"][0]["content_unit_id"] == page_id("doc-a", 2)
    answer, _ = tool(server, "research", question="oxygen per crew member in g/day")
    assert answer["calculations"] and answer["conflicts"]
    electrolysis = _concept(store, "Electrolysis")
    detail, _ = tool(server, "get_concept", concept_id=electrolysis)
    assert detail["name"] == "Electrolysis"
    path, _ = tool(server, "find_path", from_concept=_concept(store, "Oxygen Generation System"),
                   to_concept=_concept(store, "Water"))
    assert len(path["path"]) == 2
    conflicts, _ = tool(server, "find_contradictions")
    assert len(conflicts) == 2
    page, _ = tool(server, "get_page", unit_id=page_id("doc-b", 2))
    assert page["text"].startswith("Atmosphere revitalization")
    assert tool(server, "list_projects")[0] == [p for p in app.list_projects() if p["id"] == "default"]
    missing, error = tool(server, "get_concept", concept_id="nope")
    assert error and "NotFound" in missing["error"]


def test_mcp_permissions_are_enforced_and_every_call_is_audited(app) -> None:
    reader, writer, outsider = (MCPServer(app, POLICY, name) for name in ("codex", "claude-code", "outsider"))
    data = {"name": "Zeolite bed", "type": "artifact", "description": "Adsorbs CO2.", "aliases": [],
            "evidence": "Carbon dioxide removal uses zeolite beds"}
    denied, error = tool(reader, "propose_knowledge", kind="concept", unit_id=page_id("doc-a", 2), data=data)
    assert error and "may not propose" in denied["error"]
    proposal, error = tool(writer, "propose_knowledge", kind="concept", unit_id=page_id("doc-a", 2), data=data)
    assert not error and proposal["status"] == "submitted" and proposal["agent"] == "claude-code"
    blocked, error = tool(outsider, "search", query="oxygen")
    assert error and "may not read" in blocked["error"]
    assert tool(outsider, "list_projects")[0] == []
    activity = app.agent_activity("default")
    assert [(a["agent"], a["tool"], a["outcome"]) for a in activity[:3]] == [
        ("outsider", "search", "denied"), ("claude-code", "propose_knowledge", "ok"),
        ("codex", "propose_knowledge", "denied")]
    with pytest.raises(ValueError):
        AgentPolicy({"agents": {"x": {"projects": ["*"], "scopes": ["admin"]}}})


# ---------------------------------------------------------------- locks
def test_one_writer_per_project_across_threads_and_expired_locks() -> None:
    database = mongomock.MongoClient()["locks"]
    locks = ProjectLocks(database)
    errors = []
    with locks.hold("p", "rebuild"):
        with locks.hold("p", "ingest"):                     # same thread: re-entrant
            pass
        thread = threading.Thread(target=lambda: errors.append(_try(locks)))
        thread.start()
        thread.join()
        with locks.hold("other", "rebuild"):                # other projects are independent
            pass
    assert isinstance(errors[0], ProjectBusy) and "rebuild" in str(errors[0])
    assert locks.current("p") is None                       # released
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    database["project_locks"].insert_one({"_id": "p", "token": "dead", "operation": "crashed", "owner": "x",
                                          "acquired_at": past, "expires_at": past})
    with locks.hold("p", "rebuild"):                        # an expired lock is taken over
        assert locks.current("p")["operation"] == "rebuild"


def _try(locks):
    try:
        with locks.hold("p", "validate"):
            return None
    except ProjectBusy as exc:
        return exc


def test_api_answers_409_while_another_process_writes(app, store, metadata) -> None:
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    store.database["project_locks"].insert_one({"_id": "default", "token": "t", "operation": "rebuild",
                                                "owner": "cli:4242", "acquired_at": future, "expires_at": future})
    with TestClient(create_app(store=store, metadata=metadata, pipeline_factory=lambda: None,
                               watch_folder=None)) as client:
        response = client.post("/api/projects/default/validate")
        assert response.status_code == 409 and "cli:4242" in response.json()["detail"]
        assert client.get("/api/search/hybrid", params={"q": "oxygen"}).status_code == 200   # reads continue


# ---------------------------------------------------------------- backup and restore
def test_project_backup_restores_into_an_empty_database_and_never_overwrites(app, store, tmp_path) -> None:
    manifest = backup(store.database, tmp_path / "b1", "default")
    assert manifest["collections"]["kg_concepts"]["records"] == store.database["kg_concepts"].count_documents({})
    target = mongomock.MongoClient()["restored"]
    restored = restore(target, tmp_path / "b1")
    for name in ("kg_concepts", "kg_claims", "content_units", "documents", "projects"):
        assert restored[name] == target[name].count_documents({}) == store.database[name].count_documents(
            {"_id": "default"} if name == "projects" else {})
    original = store.database["kg_claims"].find_one()
    assert target["kg_claims"].find_one({"_id": original["_id"]}) == original      # types survive (dates, nested)
    with pytest.raises(FileExistsError):
        restore(target, tmp_path / "b1")                                           # never overwrites
    with pytest.raises(FileExistsError):
        backup(store.database, tmp_path / "b1", "default")                         # never writes into a backup
    damaged = tmp_path / "b1" / "kg_claims.jsonl"
    damaged.write_text(damaged.read_text(encoding="utf-8").replace("oxygen", "OXYGEN"), encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum"):
        restore(mongomock.MongoClient()["fresh"], tmp_path / "b1")


# ---------------------------------------------------------------- logging, evaluation, diagnostics
def test_json_logs_carry_context_fields() -> None:
    record = logging.LogRecord("knowledge_map.api", logging.INFO, __file__, 1, "rebuild finished", None, None)
    record.project = "default"
    record.ms = 12.5
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "rebuild finished" and entry["project"] == "default" and entry["ms"] == 12.5
    assert entry["level"] == "INFO" and entry["time"].endswith("+00:00")


def test_evaluation_metrics_on_a_labelled_corpus(app) -> None:
    retrieval = evaluate_retrieval(app, "default", [
        {"query": "zeolite beds regenerated", "relevant": [{"document": "doc-a.pdf", "page": 2}]},
        {"query": "oxygen consumed by each crew member", "relevant": [{"document": "doc-a", "page": 1},
                                                                       {"document": "doc-b", "page": 1}]},
        {"query": "unlabelled", "relevant": [{"document": "missing.pdf", "page": 1}]}])
    assert retrieval["queries"] == 2 and retrieval["recall_at_10"] == 1.0 and retrieval["mrr"] == 1.0
    assert retrieval["citation_correctness"] == 1.0 and "error" in retrieval["per_query"][2]
    extraction = evaluate_extraction(app, "default", [
        {"document": "doc-a.pdf", "page": 1, "concepts": ["Oxygen Generation System", "Electrolysis", "Water", "Hydrogen"],
         "relationships": [["Oxygen Generation System", "uses", "Electrolysis"]]}])
    assert extraction["concept_precision"] == 1.0 and extraction["concept_recall"] == 0.75
    assert extraction["relationship_recall"] == 1.0
    assert extraction["per_page"][0]["missed"] == ["hydrogen"]
    assert evaluate_knowledge_citations(app, "default")["citation_correctness"] == 1.0


def test_diagnostics_report_the_projects_health(app) -> None:
    report = app.diagnostics("default")
    assert report["documents"] == 2 and report["document_kinds"] == {"digital": 2}
    assert report["concept_first"]["pending"] == 0 and report["records"]["kg_claims"] == 3
    assert report["lock"] is None and isinstance(report["problems"], list)


def test_benchmark_measures_latency_failures_memory_and_quality(app, store) -> None:
    from src.contracts.interfaces import LLMResponse
    from src.orchestration.benchmark import benchmark_text_model
    from test_hybrid_retrieval import FakeText

    class Flaky(FakeText):
        def complete(self, system, messages, schema):
            if "zeolite" in messages[0]["content"]:
                return LLMResponse("not json at all", "end_turn", self.model)      # every attempt fails
            return super().complete(system, messages, schema)

    chunks = list(store.content_units.find({"unit_type": "chunk"}).sort("_id", 1))
    readings = iter([1000.0, 4200.0] * len(chunks))
    result = benchmark_text_model(Flaky(), chunks, max_retries=1, sampler=lambda: next(readings),
                                  gold={("doc-a", 1): ["Oxygen Generation System", "Electrolysis", "Water", "Hydrogen"]})
    assert result["chunks"] == len(chunks) == 4 and result["failure_rate"] == 0.25
    assert result["peak_server_rss_mb"] == 4200.0 and result["latency_seconds"]["median"] is not None
    assert result["quality"] == {"concept_precision": 1.0, "concept_recall": 0.75, "labelled_chunks": 1}
    assert store.content_units.count_documents({"extraction.model": "fake-qwen", "unit_type": "chunk"}) == 4  # untouched


def test_upserts_never_set_a_field_twice_which_real_mongodb_rejects() -> None:
    """mongomock accepts {"$set": {"created_at"}, "$setOnInsert": {"created_at"}}; MongoDB answers error 40.
    Governance writes validation records that carry created_at, which broke every rebuild on a real server."""
    from src.knowledge_repository.mongodb_store import upsert_all

    class StrictCollection:
        def __init__(self):
            self.inner = mongomock.MongoClient()["strict"]["records"]

        def update_one(self, query, update, upsert=False):
            assert not set(update.get("$set", {})) & set(update.get("$setOnInsert", {})), update
            return self.inner.update_one(query, update, upsert=upsert)

        def find(self, *args, **kwargs):
            return self.inner.find(*args, **kwargs)

        def delete_many(self, query):
            return self.inner.delete_many(query)

    collection = StrictCollection()
    first = datetime(2026, 1, 1, tzinfo=timezone.utc)
    upsert_all(collection, [{"_id": "a", "value": 1, "created_at": first}])
    upsert_all(collection, [{"_id": "a", "value": 2, "created_at": datetime.now(timezone.utc)}])
    stored = collection.inner.find_one({"_id": "a"})
    assert stored["value"] == 2 and stored["created_at"].replace(tzinfo=timezone.utc) == first   # first time kept
