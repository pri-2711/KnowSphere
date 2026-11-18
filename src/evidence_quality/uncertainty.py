"""Uncertainty and modality read from a statement's own wording (deterministic, no model).

Confidence is a number a component may report; uncertainty is the list of reasons a source
itself gives for doubt. They are stored separately and neither decides acceptance.

``modality_hint`` is the strongest modality the wording supports:
- "hypothesis": the source proposes or speculates ("we hypothesize", "it is possible that");
- "interpretation": the source infers or hedges ("suggests", "indicates", "likely", "may");
- "fact": a plain assertion, a requirement ("shall") or a measured value.
"""

from __future__ import annotations

import re

MODALITIES = ("fact", "interpretation", "hypothesis")

_HYPOTHESIS = re.compile(r"\b(hypothes\w*|we (propose|speculate|conjecture|postulate)|it is possible that|"
                         r"could potentially|remains to be (seen|shown|tested))\b", re.IGNORECASE)
_INTERPRETATION = re.compile(r"\b(suggest\w*|indicat\w*|appears?|seems?|likely|probabl\w*|presumabl\w*|"
                             r"may|might|could|we (believe|interpret|conclude|infer))\b", re.IGNORECASE)
_MARKERS = [
    (re.compile(r"\b(approximately|approx\.|about|around|roughly|nearly|estimated?|on the order of)\b|~|≈",
                re.IGNORECASE), "approximate value"),
    (re.compile(r"\b(TBD|TBR|TBS|to be (determined|resolved|supplied|confirmed))\b"), "open item (TBD/TBR)"),
    (re.compile(r"\b(preliminary|notional|provisional|draft|assumed|assumption)\b", re.IGNORECASE),
     "preliminary or assumed"),
    (re.compile(r"\b(unknown|uncertain\w*|unclear|not (well )?understood)\b", re.IGNORECASE), "stated as uncertain"),
]


def modality_hint(text: str) -> str:
    if _HYPOTHESIS.search(text or ""):
        return "hypothesis"
    if _INTERPRETATION.search(text or ""):
        return "interpretation"
    return "fact"


def uncertainty_markers(text: str) -> list[str]:
    """Reasons for doubt the text itself states, in a fixed order, without repeats."""
    found = [reason for pattern, reason in _MARKERS if pattern.search(text or "")]
    hint = modality_hint(text)
    if hint != "fact":
        found.append(f"hedged wording ({hint})")
    return found


def weaker(a: str, b: str) -> str:
    """The more cautious of two modalities."""
    return max(a, b, key=MODALITIES.index)
