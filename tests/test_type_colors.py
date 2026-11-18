"""Knowledge type colours: one mapping (src/web/types.js), applied everywhere, readable, never random."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "src" / "web"
EXPECTED = {"document": "#64748B", "engineering concept": "#3B82F6", "method": "#8B5CF6",
            "organization": "#0D9488", "process": "#D97706", "system component": "#16A34A"}
GREY = "#6B7280"
CANVAS = "#FFFFFF"   # the map canvas uses var(--surface), which is #ffffff


def _evaluate() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    script = (WEB / "types.js").read_text(encoding="utf-8") + """
const T = globalThis.KnowledgeTypes;
const types = [...Object.keys(T.TYPE_COLORS), "", "concept", "principle", "Brand New Type", "  Method  ", null];
const out = {mapping: T.TYPE_COLORS, all: T.ALL_TYPES_COLOR, styles: {}, repeat: {}, known: {}};
for (const t of types) { out.styles[String(t)] = T.style(t); out.repeat[String(t)] = T.style(t); out.known[String(t)] = T.isKnown(t); }
out.canvas = Object.fromEntries(Object.keys(T.TYPE_COLORS).concat(["unknown"]).map(t =>
  [t, T.contrast(T.style(t).border, "#FFFFFF")]));
out.badge = T.badge("method");
out.legend = T.legend(["process", "something new"]);
console.log(JSON.stringify(out));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_one_fixed_mapping_with_a_neutral_fallback() -> None:
    data = _evaluate()
    assert data["mapping"] == EXPECTED and data["all"] == GREY
    for type_, color in EXPECTED.items():
        assert data["styles"][type_]["color"] == color
    for unknown in ("", "concept", "principle", "Brand New Type", "null"):
        assert data["styles"][unknown]["color"] == GREY and data["known"][unknown] is False
    assert data["styles"]["  Method  "] == data["styles"]["method"]          # same type, same colour
    assert data["repeat"] == data["styles"]                                     # deterministic


def test_badges_are_readable_and_nodes_stand_out_on_the_canvas() -> None:
    data = _evaluate()
    for type_, style in data["styles"].items():
        assert style["contrast"] >= 4.5, type_                                  # WCAG AA text on its tint
        assert style["background"] != style["color"]                          # a tint, not the solid colour
    assert all(ratio >= 3.0 for ratio in data["canvas"].values()), data["canvas"]   # WCAG non-text contrast
    assert "--type-bg:" in data["badge"] and 'data-type="method"' in data["badge"]
    assert data["legend"].count("<li>") == 2


def test_no_component_defines_its_own_type_colours() -> None:
    hexes = {color.lower() for color in [*EXPECTED.values(), GREY]}
    for path in WEB.glob("*.*"):
        if path.name == "types.js" or path.suffix not in (".js", ".css", ".html"):
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert not hexes & set(re.findall(r"#[0-9a-f]{6}", text)), path.name
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "PALETTE" not in app and "Math.random" not in app
    for use in ("Types.style(el.data.type)", "Types.paintSelect", "Types.legend", "Types.badge", "Types.dot"):
        assert use in app, use
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert page.index("/static/types.js") < page.index("/static/app.js") and 'id="type-legend"' in page
