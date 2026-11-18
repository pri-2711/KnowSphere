"""Canonical contracts and the model runtime (Ollama provider through a fake transport, router policy)."""

from __future__ import annotations

import base64
import json

import pytest
from pydantic import ValidationError

from src.config import LLMConfig, RuntimeConfig
from src.contracts.interfaces import LLMError, LLMFatalError, SpanModel, TextModel, VisionModel
from src.contracts.schemas import Claim, Concept, Evidence, KnowledgeStatus, Provenance, Relationship, SourceRef
from src.model_runtime.ollama import OllamaModel
from src.model_runtime.router import ModelRouter, ModelUnavailable, ResourceGate

PROV = Provenance(component="test", method="llm")


def _evidence(project: str = "p1", quote: str = "a quote") -> Evidence:
    return Evidence(source=SourceRef(project_id=project, document_id="d", content_unit_id="u", page_number=1,
                                     char_start=0, char_end=7), quote=quote, provenance=PROV)


def test_knowledge_items_keep_project_isolation_and_source_offsets() -> None:
    concept = Concept(id="c", project_id="p1", name="Water recovery", normalized_name="water recovery",
                      evidence=[_evidence()])
    assert concept.status is KnowledgeStatus.CANDIDATE and concept.document_ids == {"d"}
    with pytest.raises(ValidationError, match="own project"):
        Concept(id="c", project_id="p1", name="X", normalized_name="x", evidence=[_evidence("p2")])
    with pytest.raises(ValidationError):
        SourceRef(project_id="p1", document_id="d", content_unit_id="u", char_start=5)      # half an offset
    with pytest.raises(ValidationError):
        Relationship(id="r", project_id="p1", source_id="a", target_id="a", relation_type="uses")
    with pytest.raises(ValidationError):
        Claim(id="k", project_id="p1", subject="s", predicate="p", statement="x", unexpected=1)
    assert Claim(id="k", project_id="p1", subject="s", predicate="p", statement="x").schema_version == "knowpal-1"


class FakeTransport:
    def __init__(self, status: int = 200, answer: dict | None = None) -> None:
        self.status = status
        self.answer = answer if answer is not None else {"message": {"content": '{"concepts": []}'},
                                                         "done_reason": "stop", "model": "qwen2.5:7b",
                                                         "prompt_eval_count": 11, "eval_count": 5}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, body: dict, timeout: float):
        self.calls.append((url, body))
        return self.status, self.answer, json.dumps(self.answer)


def test_ollama_text_and_vision_follow_the_shared_protocols(tmp_path) -> None:
    transport = FakeTransport()
    model = OllamaModel("qwen2.5:7b", RuntimeConfig(), transport)
    assert isinstance(model, TextModel) and isinstance(model, VisionModel)
    schema = {"type": "object"}
    response = model.complete("system text", [{"role": "user", "content": "passage"}], schema)
    url, body = transport.calls[0]
    assert url.endswith("/api/chat") and body["format"] == schema and body["options"]["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "system text"} and body["stream"] is False
    assert response.text == '{"concepts": []}' and response.stop_reason == "end_turn"
    assert response.usage == {"input_tokens": 11, "output_tokens": 5}

    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG fake")
    model.describe(image, "sys", "prompt", schema)
    sent = transport.calls[1][1]["messages"][1]
    assert sent["images"] == [base64.b64encode(b"\x89PNG fake").decode()] and sent["content"] == "prompt"

    cut = OllamaModel("m", transport=FakeTransport(answer={"message": {"content": "{"}, "done_reason": "length"}))
    assert cut.complete("s", [{"role": "user", "content": "x"}], schema).stop_reason == "max_tokens"
    with pytest.raises(LLMFatalError, match="ollama pull"):
        OllamaModel("missing", transport=FakeTransport(404, {"error": "not found"})).complete("s", [], schema)
    with pytest.raises(LLMError) as error:
        OllamaModel("m", transport=FakeTransport(500, {"error": "boom"})).complete("s", [], schema)
    assert error.value.retryable


def test_router_routes_capabilities_without_model_specific_callers() -> None:
    transport = FakeTransport()
    router = ModelRouter(LLMConfig(provider="ollama", model="qwen2.5:7b"),
                         RuntimeConfig(vision_model="qwen2.5vl:3b", ram_budget_gb=10, use_gliner=False),
                         transport=transport)
    assert router.text_model().model == "qwen2.5:7b" and router.vision_model().model == "qwen2.5vl:3b"
    assert router.span_model() is None                                    # GLiNER is optional
    assert router.describe()["text"] == {"provider": "ollama", "model": "qwen2.5:7b"}
    claude = ModelRouter(LLMConfig(provider="anthropic", model="claude-opus-5-5"), RuntimeConfig())
    assert claude.text_model().model == "claude-opus-5-5" and claude.text_model().gate is None   # hosted: no gate
    with pytest.raises(ModelUnavailable):
        ModelRouter(LLMConfig(provider="unknown", model="x"), RuntimeConfig()).text_model()
    with pytest.raises(ModelUnavailable, match="budget"):
        ModelRouter(LLMConfig(provider="ollama", model="qwen2.5:14b"), RuntimeConfig(ram_budget_gb=8)).text_model()
    fake_spans = type("Spans", (), {"model_name": "fake", "extract_many": lambda self, texts: [[] for _ in texts]})()
    assert isinstance(ModelRouter(overrides={"spans": fake_spans}).span_model(), SpanModel)


def test_local_models_run_one_at_a_time_and_unload_on_switch() -> None:
    transport = FakeTransport()
    router = ModelRouter(LLMConfig(provider="ollama", model="qwen2.5:7b"),
                         RuntimeConfig(vision_model="qwen2.5vl:3b"), transport=transport)
    text, vision = router.text_model(), router.vision_model()
    text.complete("s", [{"role": "user", "content": "a"}], {})
    text.complete("s", [{"role": "user", "content": "b"}], {})
    assert router.gate.switches == 0                                      # same model stays loaded
    vision.inner.describe = lambda *args: text.inner.complete("s", [], {})   # avoid reading an image file
    vision.describe(None, "s", "p", {})
    unloads = [body for url, body in transport.calls if url.endswith("/api/generate")]
    assert router.gate.switches == 1 and unloads == [{"model": "qwen2.5:7b", "keep_alive": 0}]
    assert isinstance(router.gate, ResourceGate)
