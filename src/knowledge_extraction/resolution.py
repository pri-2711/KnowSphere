"""Canonical concept resolution: the text model decides whether two name clusters are one concept.

Runs inside Pass 2 (``enrichment.cluster``) after the deterministic merges, on the
clusters those merges left apart but that may still be one thing:

- ``abbreviation``: a short name whose letters are the initials of another name
  ("ECLSS" and "Environmental Control and Life Support System"), or an alias that
  several concepts claim ("SE" for "Systems Engineering" and "Software Engineering").
- ``synonym``: names whose embeddings are close but below the automatic merge
  threshold, and fuzzy or embedding pairs a guard kept apart (a sub-phrase, other words).

Candidates are found by general rules, never by a fixed list of terms. Similar names
only make a pair a candidate; the model decides from the passages where each name
occurs, and answers "same", "different" or "unsure". Only "same" merges. "Unsure"
leaves both concepts in place and marks them for review. An abbreviation the model
matches with several clusters is ambiguous: nothing merges and all are marked.
Differences in numbers or a negation are never merged, whatever the model says.

Every decision is stored per project (``kg_resolutions``) with the context it was
made on, so a rebuild reuses it: no repeated calls and the same graph. Without a
text model (a rebuild), stored decisions still apply and nothing new is asked.

Reviewer decisions (``source: "reviewer"``) override everything and make merges
reversible: "different" keeps two names in separate concepts whichever rule would
join them (a split), "same" joins them (a manual merge). Removing the reviewer
decision restores the automatic result on the next build.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

from src.contracts.ids import scoped_id
from src.contracts.interfaces import TextModel
from src.contracts.vocabulary import DOMAIN_SUBJECTS, looks_like_abbreviation
from src.knowledge_extraction.recovery import run_with_recovery

if TYPE_CHECKING:
    from src.knowledge_extraction.enrichment import Group

PROMPT_VERSION = "resolve-1"
DECISIONS = ("same", "different", "unsure")
MAX_MENTIONS = 3            # passages shown to the model for each side
_CONNECTORS = frozenset({"and", "of", "the", "for", "in", "on", "to", "a", "an", "with", "by"})

SYSTEM_PROMPT = """You decide whether two terms extracted from technical documents name the same thing.

The documents are about:
""" + "\n".join(f"- {subject}" for subject in DOMAIN_SUBJECTS) + """

Each term comes with its type, the other names it was written as, and passages where it occurs.

- "same": the passages show that both terms name the same entity or concept: an abbreviation and
  its full form, a synonym, an alternative name or a spelling variant.
- "different": they are distinct, even if related or similar looking: a part and its whole, a
  broader and a narrower term, two different components, a process and its result, an
  organisation and its programme.
- "unsure": the passages do not settle it, for example an abbreviation that can mean several things.
Never guess. Use only the passages and general knowledge of the domain that the passages support.

"canonical_name": when "same", the clearest complete name, copied exactly from the names given
(prefer a full form over an abbreviation). Otherwise repeat the first term's name.
"reason": one short sentence citing the passage that decided it.
Reply with JSON only."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": list(DECISIONS)},
                   "canonical_name": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["decision", "canonical_name", "reason"],
}


def _shape_errors(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["the answer must be a JSON object with decision, canonical_name and reason"]
    errors = []
    if data.get("decision") not in DECISIONS:
        errors.append("'decision' must be one of " + ", ".join(DECISIONS))
    for name in ("canonical_name", "reason"):
        if not isinstance(data.get(name), str):
            errors.append(f"'{name}' must be a string")
    return errors


def is_initialism(short: str, long: str) -> bool:
    """True when ``short`` is the initials of ``long`` (with or without connecting words)."""
    if not looks_like_abbreviation(short):
        return False
    letters = re.sub(r"[^A-Za-z0-9]", "", short).upper()
    words = re.findall(r"[A-Za-z0-9]+", long)
    if len(letters) < 2 or len(words) < 2:
        return False
    every = "".join(word[0] for word in words).upper()
    content = "".join(word[0] for word in words if word.lower() not in _CONNECTORS).upper()
    return letters in (every, content)


# ---------------------------------------------------------------- the model and its stored decisions
class DecisionStore(Protocol):
    def resolutions(self, project_id: str) -> dict[str, dict[str, Any]]: ...
    def save_resolution(self, record: Mapping[str, Any]) -> None: ...
    def reviewer_resolutions(self, project_id: str) -> list[dict[str, Any]]: ...


def reviewer_record_id(project_id: str, key_a: str, key_b: str) -> str:
    return scoped_id("kg_resolution_review", project_id, *sorted((key_a, key_b)))


@dataclass
class Side:
    """One cluster as the model sees it."""
    key: str
    name: str
    concept_type: str
    names: list[str]
    passages: list[dict[str, Any]]

    def describe(self, label: str) -> str:
        lines = [f"{label}: {self.name}", f"type: {self.concept_type}"]
        others = [name for name in self.names if name != self.name]
        if others:
            lines.append("also written as: " + ", ".join(others))
        for passage in self.passages:
            lines.append(f"- [{passage['document_id']} p.{passage['page_number']}] "
                         f"{passage['quote'] or ''} ({passage['description'] or 'no description'})")
        return "\n".join(lines)


def side(key: str, members: Iterable[Group]) -> Side:
    members = list(members)
    root = next(group for group in members if group.key == key)
    names = sorted({surface for group in members for surface in (*group.surfaces, *group.alias_surfaces)},
                   key=str.lower)
    mentions = sorted((m for group in members for m in group.mentions),
                      key=lambda m: (m["status"] != "supported", m["document_id"], m["page_number"], m["chunk_id"]))
    passages, seen = [], set()
    for mention in mentions:
        quote = (mention.get("evidence") or {}).get("quote")
        if (mention["document_id"], quote) in seen:
            continue
        seen.add((mention["document_id"], quote))
        passages.append({"document_id": mention["document_id"], "page_number": mention["page_number"],
                         "quote": quote, "description": mention["description"]})
        if len(passages) == MAX_MENTIONS:
            break
    return Side(key, root.name, root.concept_type, names, passages)


@dataclass
class ConceptResolver:
    """Asks the text model about candidate pairs, reusing stored decisions. ``llm`` None: stored only."""
    llm: TextModel | None
    decisions: DecisionStore
    project_id: str
    max_retries: int = 1
    max_new_pairs: int = 50
    enabled: bool = True             # False: only reviewer decisions apply, the model pass does not run
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _stored: dict[str, dict[str, Any]] | None = None
    _reviewed: dict[frozenset, dict[str, Any]] | None = None

    def reviewed(self) -> dict[frozenset, dict[str, Any]]:
        """Reviewer decisions by pair of name keys."""
        if self._reviewed is None:
            self._reviewed = {frozenset(record["keys"]): record
                              for record in self.decisions.reviewer_resolutions(self.project_id)}
        return self._reviewed

    def pairs(self, decision: str) -> list[tuple[str, str]]:
        return sorted(tuple(sorted(pair)) for pair, record in self.reviewed().items()
                      if record["decision"] == decision and len(pair) == 2)

    def judge(self, a: Side, b: Side) -> dict[str, Any] | None:
        """The decision for a pair: {"decision", "canonical_name", "reason"}, or None when none is available."""
        first, second = sorted((a, b), key=lambda s: s.key)
        record_id = scoped_id("kg_resolution", self.project_id, first.key, second.key)
        message = first.describe("Term 1") + "\n\n" + second.describe("Term 2")
        context = hashlib.sha256(message.encode("utf-8")).hexdigest()
        reviewed = self.reviewed().get(frozenset((first.key, second.key)))
        if reviewed is not None:
            self.stats["reviewer"] += 1
            return reviewed
        if self._stored is None:
            self._stored = self.decisions.resolutions(self.project_id)
        stored = self._stored.get(record_id)
        if stored and stored.get("prompt_version") == PROMPT_VERSION and \
                (self.llm is None or stored.get("context_sha256") == context):
            self.stats["reused"] += 1
            return stored
        if self.llm is None:
            self.stats["no_model"] += 1
            return None
        if self.stats["model_pairs"] >= self.max_new_pairs:
            self.stats["skipped_over_limit"] += 1
            return None
        self.stats["model_pairs"] += 1
        llm = self.llm
        result = run_with_recovery(lambda messages: llm.complete(SYSTEM_PROMPT, messages, SCHEMA),
                                   message, self.max_retries, _shape_errors)
        self.stats["model_calls"] += len(result.attempts)
        if not result.ok or result.data is None:
            self.stats["failed"] += 1        # not stored: asked again on the next run
            return None
        record = {"_id": record_id, "project_id": self.project_id, "keys": [first.key, second.key],
                  "names": [first.name, second.name], "decision": result.data["decision"],
                  "canonical_name": result.data["canonical_name"].strip(), "reason": result.data["reason"].strip(),
                  "context_sha256": context, "model": llm.model, "prompt_version": PROMPT_VERSION,
                  "decided_at": datetime.now(timezone.utc)}
        self.decisions.save_resolution(record)
        self._stored[record_id] = record
        return record


# ---------------------------------------------------------------- candidates and the resolution pass
def abbreviation_candidates(groups: Mapping[str, Group]) -> dict[str, set[str]]:
    """Abbreviation-like group key -> keys of groups it may stand for (initials, or a shared alias claim)."""
    partners: dict[str, set[str]] = defaultdict(set)
    long_names = [(key, group.name) for key, group in groups.items() if len(group.name.split()) >= 2
                  or "-" in group.name]
    for key, group in groups.items():
        if not looks_like_abbreviation(group.name):
            continue
        for other, name in long_names:
            if other != key and is_initialism(group.name, name):
                partners[key].add(other)
    claims: dict[str, set[str]] = defaultdict(set)
    for group in groups.values():
        for alias_key in group.alias_keys:
            if alias_key in groups and alias_key != group.key:
                claims[alias_key].add(group.key)
    for alias_key, owners in claims.items():
        if len(owners) > 1:                          # the deterministic alias rule refused these
            partners[alias_key] |= owners
    return partners


def resolve(groups: Mapping[str, Group], union: Any, resolver: ConceptResolver,
            synonym_pairs: list[tuple[float, str, str]], hard_guard: Any,
            kept_apart: Any = lambda a, b: False) -> list[dict[str, Any]]:
    """Merge clusters the model confirms; return log entries (merged, rejected or needs_review)."""
    log: list[dict[str, Any]] = []

    def members(root: str) -> list[Group]:
        return [groups[key] for key in groups if union.find(key) == root]

    def ask(a: str, b: str, rule: str) -> tuple[str | None, dict[str, Any] | None, str, str]:
        root_a, root_b = union.find(a), union.find(b)
        if root_a == root_b:
            return None, None, root_a, root_b
        reason = hard_guard(groups[root_a], groups[root_b]) or (
            "a reviewer kept these apart" if kept_apart(root_a, root_b) else None)
        if reason:
            log.append(_entry(rule, groups, root_a, root_b, "rejected", reason))
            return None, None, root_a, root_b
        verdict = resolver.judge(side(root_a, members(root_a)), side(root_b, members(root_b)))
        return (verdict or {}).get("decision"), verdict, root_a, root_b

    def merge(root_a: str, root_b: str, verdict: Mapping[str, Any], rule: str) -> None:
        wanted = verdict.get("canonical_name", "").lower()
        names_b = {surface.lower() for group in members(root_b) for surface in group.surfaces}
        names_a = {surface.lower() for group in members(root_a) for surface in group.surfaces}
        keep = root_b if wanted in names_b and wanted not in names_a else root_a
        if keep == root_a and looks_like_abbreviation(groups[root_a].name) \
                and not looks_like_abbreviation(groups[root_b].name) and wanted not in names_a:
            keep = root_b                             # no usable choice: the full form names the concept
        union.union(root_a, root_b, keep=keep)
        log.append(_entry(rule, groups, root_a, root_b, "merged", verdict.get("reason", ""), model=True))

    for short, partners in sorted(abbreviation_candidates(groups).items()):
        verdicts = []
        for partner in sorted(partners):
            decision, verdict, root_a, root_b = ask(short, partner, "abbreviation")
            if decision is not None:
                verdicts.append((decision, verdict, root_a, root_b))
        same = {root_b: (verdict, root_a) for decision, verdict, root_a, root_b in verdicts if decision == "same"}
        if len(same) == 1:
            (root_b, (verdict, root_a)), = same.items()
            merge(root_a, root_b, verdict, "abbreviation")
        elif len(same) > 1:
            meanings = ", ".join(groups[root].name for root in sorted(same))
            for root_b, (verdict, root_a) in sorted(same.items()):
                log.append(_entry("abbreviation", groups, root_a, root_b, "needs_review",
                                  f"ambiguous: '{groups[root_a].name}' could mean {meanings}", model=True))
        for decision, verdict, root_a, root_b in verdicts:
            if decision != "same":
                log.append(_entry("abbreviation", groups, root_a, root_b,
                                  "needs_review" if decision == "unsure" else "rejected", verdict["reason"],
                                  model=True))
    for _, a, b in sorted(synonym_pairs, key=lambda pair: (-pair[0], pair[1], pair[2])):
        decision, verdict, root_a, root_b = ask(a, b, "synonym")
        if decision == "same":
            merge(root_a, root_b, verdict, "synonym")
        elif decision is not None:
            log.append(_entry("synonym", groups, root_a, root_b,
                              "needs_review" if decision == "unsure" else "rejected", verdict["reason"], model=True))
    return log


def _entry(rule: str, groups: Mapping[str, Group], a: str, b: str, decision: str, reason: str,
           model: bool = False) -> dict[str, Any]:
    return {"rule": "llm_" + rule if model else rule, "score": 1.0, "a": groups[a].name, "b": groups[b].name,
            "keys": [a, b], "decision": decision, "reason": reason}
