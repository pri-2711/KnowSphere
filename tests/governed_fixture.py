"""A small governed (concept-first) corpus for the NASA canonical-resolution case.

SIMULATED: the text model is scripted (no real inference). It extracts "NASA" and its full form
from different documents and never states that they are the same, so only canonical resolution
(the definition in doc-n, page 1) can join them.
"""

from __future__ import annotations

import json

from kg_fakes import concept, relation

from src.config import ChunkingConfig, KGConfig
from src.contracts.interfaces import LLMResponse

CONFIG = KGConfig(chunking=ChunkingConfig(min_chars=40, target_chars=1000, max_chars=2000), max_retries=1)
FULL = "National Aeronautics and Space Administration"
PAGES = {
    ("doc-n", 1): "The National Aeronautics and Space Administration (NASA) funds research on life support "
                  "systems for crewed missions.",
    ("doc-n", 2): "Life support systems recycle water and oxygen for the crew on long missions.",
    ("doc-m", 1): "NASA requires two independent oxygen sources on every crewed vehicle that leaves low orbit.",
}
ANSWERS = {
    "funds research": {
        "concepts": [concept(FULL, "entity", "The United States space agency.",
                             "The National Aeronautics and Space Administration (NASA) funds research"),
                     concept("Life support system", "artifact", "Keeps the crew alive.",
                             "life support systems for crewed missions")],
        "relationships": [relation(FULL, "related_to", "Life support system",
                                   "National Aeronautics and Space Administration (NASA) funds research on life "
                                   "support systems")]},
    "recycle water": {
        "concepts": [concept("Life support system", "artifact", "Recycles water and oxygen.",
                             "Life support systems recycle water and oxygen")],
        "relationships": []},
    "two independent": {
        "concepts": [concept("NASA", "entity", "Agency that sets crew vehicle requirements.",
                             "NASA requires two independent oxygen sources"),
                     concept("Oxygen source", "artifact", "Supplies oxygen.", "two independent oxygen sources")],
        "relationships": [relation("NASA", "defines", "Oxygen source", "NASA requires two independent oxygen sources")]},
}


class ScriptedText:
    """Answers Pass 1 by passage marker; answers every resolution question with "unsure"."""
    model = "scripted-fixture"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, messages, schema):
        self.calls += 1
        text = messages[0]["content"]
        if "decision" in schema.get("properties", {}):
            return LLMResponse(json.dumps({"decision": "unsure", "canonical_name": "", "reason": "scripted"}),
                               "end_turn", self.model)
        if "claims" in schema.get("properties", {}):
            return LLMResponse(json.dumps({"claims": []}), "end_turn", self.model)
        answer = next((a for marker, a in ANSWERS.items() if marker in text), {"concepts": [], "relationships": []})
        return LLMResponse(json.dumps(answer), "end_turn", self.model)
