"""Relationships: evidence on every edge, verbatim quotes, lexical type upgrades, thresholds."""

from __future__ import annotations

import re

from src.knowledge.relationships import _name_pattern, detect_relation, npmi
from src.storage.mongodb_store import stable_id


def _id(name: str) -> str:
    return stable_id("concept", name)


def test_every_edge_has_verbatim_evidence(seeded) -> None:
    edges = list(seeded.database["relationships"].find())
    assert edges
    pages = {unit["_id"]: unit["text"] for unit in seeded.content_units.find()}
    for edge in edges:
        assert edge["source_id"] and edge["target_id"] and edge["relation_type"]
        assert 0.0 <= edge["confidence"] <= 1.0
        assert edge["evidence"], "every edge needs at least one evidence quote"
        for item in edge["evidence"]:
            assert {"document_id", "page_number", "content_unit_id", "quote"} <= set(item)
            assert pages[item["content_unit_id"]][item["start"]:item["end"]] == item["quote"]


def test_lexical_patterns_upgrade_relation_types(seeded) -> None:
    part_of = seeded.database["relationships"].find_one({"relation_type": "part_of",
                                                         "source_id": _id("requirements analysis")})
    assert part_of is not None and part_of["target_id"] == _id("systems engineering")
    includes = seeded.database["relationships"].find_one({"relation_type": "part_of",
                                                          "source_id": _id("risk management")})
    assert includes is not None and includes["target_id"] == _id("systems engineering")
    assert any(match["relation"] == "part_of" for match in part_of["pattern_matches"])


def test_detect_relation_patterns_and_direction() -> None:
    a, b = _name_pattern(["risk analysis"]), _name_pattern(["management activity"])
    assert detect_relation("Risk analysis is a type of management activity.", a, b) == ("is_a", True)
    assert detect_relation("Every management activity, such as risk analysis, is planned.", a, b) == ("is_a", True)
    assert detect_relation("Risk analysis is used for management activity planning.", a, b) == ("used_for", True)
    assert detect_relation("Risk analysis leads to management activity.", a, b) == ("causes", True)
    assert detect_relation("Management activity consists of risk analysis.", a, b) == ("part_of", True)
    assert detect_relation("Risk analysis and management activity are both discussed.", a, b) is None


def test_npmi_bounds() -> None:
    assert npmi(0, 3, 3, 10) == -1.0
    assert npmi(5, 5, 5, 5) == 1.0
    assert npmi(2, 2, 2, 10) > npmi(2, 8, 8, 10)


def test_rebuild_is_idempotent(seeded) -> None:
    from src.knowledge.relationships import store_relationships
    before = sorted(edge["_id"] for edge in seeded.database["relationships"].find())
    store_relationships(seeded)
    after = sorted(edge["_id"] for edge in seeded.database["relationships"].find())
    assert before == after and len(after) == len(set(after))
    assert all(re.match(r"^relationship:[0-9a-f]{64}$", edge_id) for edge_id in after)
