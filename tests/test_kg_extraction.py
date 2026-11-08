"""Pass 1 with a mocked LLM: evidence, validation statuses, recovery and retry limits."""

from __future__ import annotations

import json

import pytest
from kg_fakes import DOC_X, PAGES, ScriptedLLM, concept, page_id, relation

from src.kg.config import ChunkingConfig
from src.kg.extraction import EXTRACTED, FAILED, extract_chunk, extraction_is_current
from src.kg.llm import LLMError, LLMFatalError, LLMResponse
from src.kg.recovery import parse_output
from src.kg.segmentation import segment_document

CONFIG = ChunkingConfig(min_chars=60, target_chars=1000, max_chars=2000)


@pytest.fixture
def chunk() -> dict:
    pages = [{"_id": page_id(DOC_X, 1), "page_number": 1, "text": PAGES[(DOC_X, 1)]}]
    return segment_document(DOC_X, pages, CONFIG)[0]


def _answer(concepts: list, relationships: list) -> ScriptedLLM:
    return ScriptedLLM({"Supervised learning is a type": {"concepts": concepts, "relationships": relationships}})


def test_concepts_and_relationships_carry_located_evidence(chunk) -> None:
    record = extract_chunk(chunk, ScriptedLLM(), max_retries=2)
    assert record["status"] == EXTRACTED and record["retries"] == 0
    by_name = {c["name"]: c for c in record["concepts"]}
    supervised = by_name["Supervised learning"]
    assert supervised["status"] == "supported" and supervised["type"] == "method"
    evidence = supervised["evidence"]
    page_text = PAGES[(DOC_X, 1)]
    assert evidence["content_unit_id"] == page_id(DOC_X, 1) and evidence["page_number"] == 1
    assert page_text[evidence["page_start"]:evidence["page_end"]] == evidence["quote"]
    assert chunk["text"][evidence["chunk_start"]:evidence["chunk_end"]] == evidence["quote"]
    uses = next(r for r in record["relationships"] if r["type"] == "uses")
    assert uses["status"] == "supported" and uses["source_key"] == supervised["key"]
    assert extraction_is_current({**chunk, "extraction": record}, "fake-model")
    assert not extraction_is_current({**chunk, "extraction": record}, "another-model")


def test_validation_statuses(chunk) -> None:
    llm = _answer(
        [concept("Supervised learning", "method", "Learning from labels.",
                 "Supervised learning is a type of machine learning", aliases=["SVL", "Supervised learning"]),
         concept("supervised learning", "method", "Duplicate of the first.", "uses labeled data"),
         concept("Labeled data", "concept", "Data with answers.", "labels are expensive to collect"),   # paraphrase
         concept("Quantum annealing", "method", "Not in the passage.", "Quantum annealing is great"),  # invented
         concept("Unsupervised learning", "technique", "Finds structure.", "Unsupervised learning finds structure")],
        [relation("Supervised learning", "uses", "Labeled data",
                  "Supervised learning is a type of machine learning that uses labeled data."),
         relation("Supervised learning", "contradicts", "Labeled data", "uses labeled data"),
         relation("Supervised learning", "uses", "Quantum annealing", "uses labeled data"),
         relation("Supervised learning", "is_a", "supervised learning", "Supervised learning is a type"),
         relation("Unsupervised learning", "related_to", "Labeled data", "finds structure in unlabeled data")],
    )
    record = extract_chunk(chunk, llm, max_retries=0)
    concepts = {c["name"]: c for c in record["concepts"]}
    assert len([c for c in record["concepts"] if c["key"] == "supervisedlearning"]) == 1      # duplicate merged
    assert concepts["Supervised learning"]["unsupported_aliases"] == ["SVL"]                  # alias not in text
    assert concepts["Labeled data"]["status"] == "needs_review"                               # quote not found
    assert concepts["Quantum annealing"]["status"] == "rejected"                              # not in the text
    assert concepts["Unsupervised learning"]["type"] == "concept"                             # unknown type
    assert concepts["Unsupervised learning"]["status"] == "needs_review"
    statuses = [(r["type"], r["target"], r["status"]) for r in record["relationships"]]
    assert ("uses", "Labeled data", "supported") in statuses
    assert ("contradicts", "Labeled data", "rejected") in statuses                           # not in vocabulary
    assert ("uses", "Quantum annealing", "rejected") in statuses                             # rejected endpoint
    assert ("is_a", "supervised learning", "rejected") in statuses                           # self-loop
    assert ("related_to", "Labeled data", "needs_review") in statuses                        # quote lacks a name


def test_local_repair_needs_no_retry() -> None:
    fenced = LLMResponse('```json\n{"concepts": [], "relationships": [],}\n```', stop_reason="end_turn")
    data, error, repairs = parse_output(fenced)
    assert error is None and data == {"concepts": [], "relationships": []}
    assert repairs == ["removed code fence", "removed trailing commas"]


def test_invalid_json_is_recovered_and_raw_answers_are_kept(chunk) -> None:
    good = json.dumps({"concepts": [concept("Labeled data", "concept", "Data with answers.",
                                            "Labeled data consists of input examples")], "relationships": []})
    answers = iter([LLMResponse('{"concepts": [ {"name": "Labeled data"', stop_reason="end_turn"),
                    LLMResponse(good, stop_reason="end_turn")])
    llm = ScriptedLLM(responder=lambda messages: next(answers))
    record = extract_chunk(chunk, llm, max_retries=2)
    assert record["status"] == EXTRACTED and record["retries"] == 1 and len(llm.calls) == 2
    assert record["attempts"][0]["raw"].startswith('{"concepts"') and "invalid JSON" in record["attempts"][0]["error"]
    retry_messages = llm.calls[1]
    assert retry_messages[1]["role"] == "assistant" and "invalid JSON" in retry_messages[2]["content"]
    assert [c["name"] for c in record["concepts"]] == ["Labeled data"]


def test_retry_limit_and_failure_is_marked_for_review(chunk) -> None:
    llm = ScriptedLLM(responder=lambda messages: LLMResponse('{"concepts": "oops"}', stop_reason="end_turn"))
    record = extract_chunk(chunk, llm, max_retries=2)
    assert record["status"] == FAILED and record["status_detail"] == "needs_review"
    assert len(llm.calls) == 3 and len(record["attempts"]) == 3
    assert "'concepts' must be a list" in record["error"]
    assert record["concepts"] == [] and record["relationships"] == []     # nothing invented


def test_truncated_output_is_retried_and_refusal_is_not(chunk) -> None:
    answers = iter([LLMResponse('{"concepts": [', stop_reason="max_tokens"),
                    LLMResponse('{"concepts": [], "relationships": []}', stop_reason="end_turn")])
    llm = ScriptedLLM(responder=lambda messages: next(answers))
    assert extract_chunk(chunk, llm, max_retries=1)["status"] == EXTRACTED
    refusing = ScriptedLLM(responder=lambda messages: LLMResponse("", stop_reason="refusal"))
    record = extract_chunk(chunk, refusing, max_retries=3)
    assert record["status"] == FAILED and len(refusing.calls) == 1


def test_transient_errors_retry_permanent_errors_stop_and_fatal_errors_raise(chunk) -> None:
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("RateLimitError (after SDK retries)")
        return LLMResponse('{"concepts": [], "relationships": []}', stop_reason="end_turn")

    assert extract_chunk(chunk, ScriptedLLM(responder=flaky), max_retries=2)["status"] == EXTRACTED

    def bad_request(messages):
        raise LLMError("BadRequestError", retryable=False)

    llm = ScriptedLLM(responder=bad_request)
    assert extract_chunk(chunk, llm, max_retries=3)["status"] == FAILED and len(llm.calls) == 1

    def fatal(messages):
        raise LLMFatalError("AuthenticationError")

    with pytest.raises(LLMFatalError):
        extract_chunk(chunk, ScriptedLLM(responder=fatal), max_retries=3)
