"""Canonical concept resolution in Pass 2: the text model judges candidate pairs from their passages.
The model is a fake that answers extraction prompts by passage marker and resolution prompts by name pair."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from conftest import FakeEmbedder
from kg_fakes import ScriptedLLM, concept, insert_corpus, relation

from src.config import ChunkingConfig, KGConfig
from src.knowledge_extraction.resolution import ConceptResolver, Side, is_initialism
from src.knowledge_repository.kg_store import CONCEPTS, RELATIONSHIPS, RESOLUTIONS, KGStore
from src.model_runtime.llm import LLMResponse
from src.orchestration.kg_pipeline import KGPipeline

CONFIG = KGConfig(chunking=ChunkingConfig(min_chars=40, target_chars=1000, max_chars=2000), max_retries=1)


class Judge(ScriptedLLM):
    """Pass 1 answers by marker; resolution answers from {frozenset(names): (decision, canonical)}."""

    def __init__(self, answers: dict[str, Any], verdicts: dict[frozenset, tuple[str, str]]) -> None:
        super().__init__(answers, responder=self._respond)
        self.verdicts = verdicts
        self.questions: list[frozenset] = []

    def _respond(self, messages: Sequence[dict[str, Any]]) -> LLMResponse:
        text = messages[0]["content"]
        names = re.findall(r"^Term [12]: (.+)$", text, re.MULTILINE)
        if not names:
            return ScriptedLLM(self.answers).complete("", messages, {})
        pair = frozenset(names)
        self.questions.append(pair)
        decision, canonical = self.verdicts.get(pair, ("different", names[0]))
        return LLMResponse(json.dumps({"decision": decision, "canonical_name": canonical, "reason": "fake"}),
                           stop_reason="end_turn", model=self.model)


def _run(store, pages, answers, verdicts, embedder=None, runs=1):
    insert_corpus(store, pages)
    judge = Judge(answers, verdicts)
    for _ in range(runs):
        KGPipeline(store, CONFIG, judge, embedder or FakeEmbedder()).run(sorted({d for d, _ in pages}))
    return judge, {c["canonical_name"]: c for c in store.database[CONCEPTS].find()}


ECLSS_PAGES = {
    ("doc-1", 1): "The Environmental Control and Life Support System recycles water for the crew on long missions.",
    ("doc-2", 4): "The ECLSS uses an oxygen generator to supply breathable air in the cabin of the station.",
}
ECLSS_ANSWERS = {
    "recycles water": {"concepts": [
        concept("Environmental Control and Life Support System", "artifact", "Keeps the crew alive.",
                "The Environmental Control and Life Support System recycles water"),
        concept("Water recycling", "process", "Reuse of water.", "recycles water for the crew")],
        "relationships": [relation("Environmental Control and Life Support System", "includes", "Water recycling",
                                   "The Environmental Control and Life Support System recycles water")]},
    "oxygen generator": {"concepts": [
        concept("ECLSS", "artifact", "Station life support.", "The ECLSS uses an oxygen generator"),
        concept("Oxygen generator", "artifact", "Supplies breathable air.", "an oxygen generator to supply breathable air")],
        "relationships": [relation("ECLSS", "includes", "Oxygen generator", "The ECLSS uses an oxygen generator")]},
}
FULL = "Environmental Control and Life Support System"


def test_initialisms_are_found_by_a_general_rule() -> None:
    assert is_initialism("ECLSS", FULL) and is_initialism("V&V", "Verification and Validation")
    assert is_initialism("MBSE", "Model-Based Systems Engineering") and is_initialism("JPL", "Jet Propulsion Laboratory")
    assert not is_initialism("ECLSS", "Electrical Power System") and not is_initialism("Water", "Waste Air Tank")


def test_abbreviation_across_documents_becomes_one_node_with_provenance(store) -> None:
    judge, nodes = _run(store, ECLSS_PAGES, ECLSS_ANSWERS, {frozenset({FULL, "ECLSS"}): ("same", FULL)})
    assert FULL in nodes and "ECLSS" not in nodes
    node = nodes[FULL]
    assert node["aliases"] == ["ECLSS"] and node["document_ids"] == ["doc-1", "doc-2"]
    assert {(m["document_id"], m["page_number"], m["evidence"]["quote"]) for m in node["mentions"]} == {
        ("doc-1", 1, "The Environmental Control and Life Support System recycles water"),
        ("doc-2", 4, "The ECLSS uses an oxygen generator")}
    assert node["merged_from"] == [{"key": "eclss", "name": "ECLSS", "rule": "llm_abbreviation"}]
    edges = list(store.database[RELATIONSHIPS].find())
    assert len(edges) == 2 and {edge["source_id"] for edge in edges} == {node["_id"]}     # both re-pointed
    oxygen = next(edge for edge in edges if edge["target_id"] == nodes["Oxygen generator"]["_id"])
    assert oxygen["evidence"][0]["stated_as"] == "ECLSS includes Oxygen generator"     # as written, kept
    assert judge.questions == [frozenset({FULL, "ECLSS"})]


def test_resolution_is_idempotent_and_reused_without_a_model(store) -> None:
    judge, first = _run(store, ECLSS_PAGES, ECLSS_ANSWERS, {frozenset({FULL, "ECLSS"}): ("same", FULL)}, runs=2)
    assert len(judge.questions) == 1                       # the second run reuses the stored decision
    assert store.database[RESOLUTIONS].count_documents({}) == 1
    KGPipeline(store, CONFIG, None, FakeEmbedder()).build_graph()      # a rebuild without any model
    again = {c["canonical_name"]: c for c in store.database[CONCEPTS].find()}
    assert set(again) == set(first) and again[FULL]["aliases"] == ["ECLSS"]


SYNONYM_PAGES = {
    ("doc-1", 1): "The spacecraft bus carries the payload and provides power and pointing for the mission.",
    ("doc-2", 1): "Each satellite platform carries the payload and provides power, while a pressure sensor and "
                  "a pressure regulator keep the propellant tank at 2 MPa.",
}
SYNONYM_ANSWERS = {
    "spacecraft bus carries": {"concepts": [
        concept("Spacecraft bus", "artifact", "Carries the payload.", "The spacecraft bus carries the payload")],
        "relationships": []},
    "satellite platform carries": {"concepts": [
        concept("Satellite platform", "artifact", "Carries the payload.", "Each satellite platform carries the payload"),
        concept("Pressure sensor", "artifact", "Measures pressure.", "a pressure sensor"),
        concept("Pressure regulator", "artifact", "Controls pressure.", "a pressure regulator keep the propellant tank")],
        "relationships": []},
}


def test_synonyms_merge_and_similar_components_stay_separate(store) -> None:
    embedder = FakeEmbedder({"satellite platform": "spacecraft bus", "pressure regulator": "pressure sensor"})
    verdicts = {frozenset({"Spacecraft bus", "Satellite platform"}): ("same", "Spacecraft bus"),
                frozenset({"Pressure sensor", "Pressure regulator"}): ("different", "Pressure sensor")}
    judge, nodes = _run(store, SYNONYM_PAGES, SYNONYM_ANSWERS, verdicts, embedder)
    assert set(nodes) == {"Spacecraft bus", "Pressure sensor", "Pressure regulator"}
    assert nodes["Spacecraft bus"]["aliases"] == ["Satellite platform"]
    assert nodes["Spacecraft bus"]["document_ids"] == ["doc-1", "doc-2"]
    assert nodes["Pressure sensor"]["possible_duplicates"] == []                     # "different" is final
    assert set(judge.questions) == set(verdicts)


AMBIGUOUS_PAGES = {
    ("doc-1", 1): "Systems Engineering (SE) balances cost, schedule and risk across the whole spacecraft.",
    ("doc-2", 1): "Software Engineering (SE) produces the flight software that runs on the onboard computer.",
    ("doc-3", 1): "The SE team signed off the interface review before the launch campaign started.",
}
AMBIGUOUS_ANSWERS = {
    "balances cost": {"concepts": [concept("Systems Engineering", "method", "Balances cost and risk.",
                                           "Systems Engineering (SE) balances cost", ["SE"])], "relationships": []},
    "flight software": {"concepts": [concept("Software Engineering", "method", "Produces flight software.",
                                             "Software Engineering (SE) produces the flight software", ["SE"])],
                        "relationships": []},
    "signed off": {"concepts": [concept("SE", "method", "A team.", "The SE team signed off")], "relationships": []},
}


def test_ambiguous_abbreviation_is_left_for_review_not_guessed(store) -> None:
    verdicts = {frozenset({"SE", "Systems Engineering"}): ("same", "Systems Engineering"),
                frozenset({"SE", "Software Engineering"}): ("same", "Software Engineering")}
    _, nodes = _run(store, AMBIGUOUS_PAGES, AMBIGUOUS_ANSWERS, verdicts)
    assert set(nodes) == {"SE", "Systems Engineering", "Software Engineering"}     # nothing merged
    bare = nodes["SE"]
    assert set(bare["possible_duplicates"]) == {nodes["Systems Engineering"]["_id"], nodes["Software Engineering"]["_id"]}
    assert nodes["Systems Engineering"]["possible_duplicates"] == [bare["_id"]]
    assert bare["knowledge_status"] == "needs_review"                                # governance sees it
    checks = store.database["validation_records"].find_one({"item_id": bare["_id"]})["checks"]
    assert any(check["check"] == "possible_duplicate" and not check["passed"] for check in checks)


def test_unsure_and_number_differences_never_merge(store) -> None:
    pages = {("doc-1", 1): "Stage 1 burns for two minutes before separation of the launch vehicle stack.",
             ("doc-2", 1): "Stage 2 burns for six minutes and places the payload into the parking orbit."}
    answers = {"two minutes": {"concepts": [concept("Stage 1", "artifact", "First stage.", "Stage 1 burns")],
                               "relationships": []},
               "six minutes": {"concepts": [concept("Stage 2", "artifact", "Second stage.", "Stage 2 burns")],
                               "relationships": []}}
    judge, nodes = _run(store, pages, answers, {frozenset({"Stage 1", "Stage 2"}): ("same", "Stage 1")},
                        FakeEmbedder({"stage 2": "stage 1"}))
    assert set(nodes) == {"Stage 1", "Stage 2"} and judge.questions == []            # never even asked


def test_decisions_belong_to_one_project(store) -> None:
    kg = KGStore(store)
    a, b = Side("eclss", "ECLSS", "artifact", ["ECLSS"], []), Side("full", FULL, "artifact", [FULL], [])
    ConceptResolver(Judge({}, {frozenset({FULL, "ECLSS"}): ("same", FULL)}), kg, "alpha").judge(a, b)
    assert ConceptResolver(None, kg, "alpha").judge(a, b)["decision"] == "same"
    assert ConceptResolver(None, kg, "beta").judge(a, b) is None                    # another project: not reused
