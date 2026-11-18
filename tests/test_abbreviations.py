"""Abbreviations in Pass 1: domain context in the prompt, expansions checked against the passage,
and one canonical concept (full form, abbreviation as alias) across documents. The model is a fake."""

from __future__ import annotations

from conftest import FakeEmbedder
from kg_fakes import ScriptedLLM, concept, insert_corpus

from src.config import ChunkingConfig, KGConfig
from src.contracts.vocabulary import DOMAIN_ABBREVIATIONS, looks_like_abbreviation
from src.evidence_quality.validation import validate_local
from src.knowledge_extraction.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from src.knowledge_repository.kg_store import CONCEPTS
from src.orchestration.kg_pipeline import KGPipeline


def _chunk(text: str) -> dict:
    span = {"content_unit_id": "page-1", "page_number": 1, "start": 0, "end": len(text),
            "chunk_start": 0, "chunk_end": len(text)}
    return {"_id": "chunk-1", "document_id": "doc-1", "text": text, "spans": [span], "page_number": 1, "page_end": 1}


def _one(name, aliases, evidence, text):
    concepts, _ = validate_local({"concepts": [concept(name, "concept", f"About {name}.", evidence, aliases)],
                                  "relationships": []}, _chunk(text))
    return concepts[0]


def test_prompt_gives_the_domain_and_the_abbreviation_rules() -> None:
    assert PROMPT_VERSION != "kg-extract-1"          # older extractions are redone with the new rules
    for subject in ("Space technology", "Aerospace engineering", "Spacecraft and satellite systems",
                    "NASA technical documentation", "requirements engineering, verification and validation"):
        assert subject.lower() in SYSTEM_PROMPT.lower()
    for short, (full, _) in DOMAIN_ABBREVIATIONS.items():
        assert f"- {short} = {full}" in SYSTEM_PROMPT
    assert "SE = Systems Engineering (only when the passage supports this meaning)" in SYSTEM_PROMPT
    assert "Never guess" in SYSTEM_PROMPT and "abbreviation goes in \"aliases\"" in SYSTEM_PROMPT


def test_standard_expansions_are_accepted_from_the_abbreviation_alone() -> None:
    text = "NASA requires an ICD for every interface. MBSE and V&V start in Phase A."
    for name, alias, quote in (("National Aeronautics and Space Administration", "NASA", "NASA requires an ICD"),
                               ("Interface Control Document", "ICD", "an ICD for every interface"),
                               ("Model-Based Systems Engineering", "MBSE", "MBSE and V&V start"),
                               ("Verification and Validation", "V&V", "MBSE and V&V start")):
        item = _one(name, [alias], quote, text)
        assert item["status"] == "supported", (name, item["issues"])
        assert item["aliases"] == [alias]


def test_context_dependent_and_guessed_expansions_go_to_review() -> None:
    unsupported = _one("Systems Engineering", ["SE"], "SE reviews start early", "SE reviews start early in the program.")
    assert unsupported["status"] == "needs_review"
    assert "expansion of 'SE'" in unsupported["issues"][0]
    stated = _one("Systems Engineering", ["SE"], "Systems Engineering (SE) reviews start early",
                  "Systems Engineering (SE) reviews start early in the program.")
    assert stated["status"] == "supported"
    guessed = _one("Critical Design Review", ["CDR"], "The CDR closed", "The CDR closed all open items.")
    assert guessed["status"] == "needs_review"
    defined = _one("Critical Design Review", ["CDR"], "The Critical Design Review (CDR) closed",
                   "The Critical Design Review (CDR) closed all open items.")
    assert defined["status"] == "supported"


def test_unresolved_abbreviations_stay_as_they_are() -> None:
    item = _one("ECLSS", [], "The ECLSS recycles water", "The ECLSS recycles water and air on the station.")
    assert item["status"] == "supported" and item["name"] == "ECLSS" and item["aliases"] == []
    assert looks_like_abbreviation("V&V") and looks_like_abbreviation("ECLSS") and not looks_like_abbreviation("Water")
    assert not looks_like_abbreviation("Systems Engineering")


PAGES = {
    ("doc-1", 1): ("Model-Based Systems Engineering (MBSE) replaces documents with system models. "
                   "Verification and Validation (V&V) confirms that the spacecraft meets its requirements."),
    ("doc-2", 1): ("MBSE tools hold the interface definitions for the satellite bus. "
                   "V&V of the payload starts after integration and testing."),
}
ANSWERS = {
    "replaces documents": {"concepts": [
        concept("Model-Based Systems Engineering", "method", "Engineering with system models.",
                "Model-Based Systems Engineering (MBSE) replaces documents with system models.", ["MBSE"]),
        concept("Verification and Validation", "process", "Confirms the spacecraft meets its requirements.",
                "Verification and Validation (V&V) confirms that the spacecraft meets its requirements", ["V&V"])],
        "relationships": []},
    "MBSE tools hold": {"concepts": [
        concept("Model-Based Systems Engineering", "method", "Its tools hold interface definitions.",
                "MBSE tools hold the interface definitions", ["MBSE"]),
        concept("Verification and Validation", "process", "Starts after integration and testing.",
                "V&V of the payload starts after integration and testing", ["V&V"])],
        "relationships": []},
}


def test_one_canonical_concept_per_term_across_documents(store) -> None:
    insert_corpus(store, PAGES)
    config = KGConfig(chunking=ChunkingConfig(min_chars=60, target_chars=1000, max_chars=2000), max_retries=1)
    llm = ScriptedLLM(ANSWERS)
    for _ in range(2):                                  # the second run reuses results: same records
        KGPipeline(store, config, llm, FakeEmbedder()).run(["doc-1", "doc-2"])
    nodes = {c["canonical_name"]: c for c in store.database[CONCEPTS].find()}
    assert set(nodes) == {"Model-Based Systems Engineering", "Verification and Validation"}
    mbse = nodes["Model-Based Systems Engineering"]
    assert mbse["aliases"] == ["MBSE"] and mbse["document_ids"] == ["doc-1", "doc-2"]
    assert {m["evidence"]["quote"] for m in mbse["mentions"]} == {
        "Model-Based Systems Engineering (MBSE) replaces documents with system models.",
        "MBSE tools hold the interface definitions"}
    assert nodes["Verification and Validation"]["aliases"] == ["V&V"]
    assert all(m["status"] == "supported" for node in nodes.values() for m in node["mentions"])
    assert len(llm.calls) == 2                          # two chunks, extracted once
