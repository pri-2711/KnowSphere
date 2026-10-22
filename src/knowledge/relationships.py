"""Build context-based relationships between concept nodes, each with verbatim evidence.

Two concepts are related when they appear in the same sentence or in the same
page window (a page and the next page of the same document). Strength is
measured with normalized PMI (NPMI) over page windows plus the number of shared
sentences. A pair is kept when it shares at least two sentences, or one sentence
with NPMI >= 0, or (with no shared sentence) at least three windows with a high
NPMI. NPMI alone is not enough because a concept that appears on almost every
page (the book's main topic) has NPMI near 0 with everything.

The relation type is ``related_to`` unless an evidence sentence contains a clear
lexical pattern between the two names ("A is a type of B", "B consists of A",
"A is used for B", "A leads to B"). Every edge keeps at least one evidence
entry with document, page and a quote that is an exact slice of the page text.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Iterable, Mapping

from src.knowledge.merging import upsert_all
from src.storage.mongodb_store import MongoKnowledgeStore, stable_id

RELATION_VERSION = "relations-1"
MIN_NPMI_SENTENCE = 0.0     # pairs sharing one sentence must not be negatively associated
MIN_SHARED_SENTENCES = 2    # pairs sharing this many sentences are kept regardless of NPMI
MIN_NPMI_WINDOW = 0.35      # pairs that only share page windows
MIN_WINDOW_ONLY = 3         # windows needed for a window-only edge
MAX_EVIDENCE = 8

_ART = r"(?:\s+(?:the|an?))?"
# (relation type, pattern on the text between the two names, True when the first name is the source)
PATTERNS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("is_a", re.compile(r"^(?:is|are)\s+(?:an?\s+)?(?:(?:type|kind|form|class|category|subset)s?\s+of" + _ART + r")?$"),
     True),
    ("is_a", re.compile(r"^,?\s*such\s+as$"), False),
    ("part_of", re.compile(r"^(?:is|are)" + _ART + r"\s+(?:(?:integral|key|major|main|important)\s+)?"
                           r"(?:part|component|element|phase|stage|step)s?\s+of" + _ART + r"$"), True),
    ("part_of", re.compile(r"^(?:consists?\s+of|comprises?|is\s+composed\s+of|contains?|includes?)" + _ART + r"$"),
     False),
    ("used_for", re.compile(r"^(?:is|are|can\s+be)?\s*(?:used|applied)\s+(?:for|to|in)" + _ART + r"$"), True),
    ("causes", re.compile(r"^(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in|gives?\s+rise\s+to)" + _ART + r"$"),
     True),
)

logger = logging.getLogger("knowledge_map.relations")


def _name_pattern(names: Iterable[str]) -> re.Pattern[str]:
    parts = sorted({name for name in names if name}, key=len, reverse=True)
    escaped = [r"\s+".join(re.escape(word) for word in name.split()) for name in parts]
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")s?(?![A-Za-z0-9])", re.IGNORECASE)


def detect_relation(quote: str, pattern_a: re.Pattern[str], pattern_b: re.Pattern[str]) -> tuple[str, bool] | None:
    """Return (relation type, a_is_source) when a lexical pattern links the names in the quote."""
    match_a, match_b = pattern_a.search(quote), pattern_b.search(quote)
    if not match_a or not match_b:
        return None
    if match_a.start() < match_b.start():
        first_is_a, between = True, quote[match_a.end():match_b.start()]
    else:
        first_is_a, between = False, quote[match_b.end():match_a.start()]
    between = re.sub(r"\s+", " ", between).strip().lower()
    if len(between) > 40:
        return None
    for relation, pattern, first_is_source in PATTERNS:
        if pattern.match(between):
            return relation, first_is_a == first_is_source
    return None


def npmi(joint: int, count_a: int, count_b: int, total: int) -> float:
    """Normalized pointwise mutual information in [-1, 1]."""
    if joint == 0 or total == 0:
        return -1.0
    p_joint = joint / total
    if p_joint >= 1.0:
        return 1.0
    pmi = math.log(p_joint / ((count_a / total) * (count_b / total)))
    return pmi / -math.log(p_joint)


def build_relationships(
    concepts: list[Mapping[str, Any]],
    page_texts: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute edges from concept mentions and page texts. Pure function, so it is easy to test."""
    by_id = {concept["_id"]: concept for concept in concepts}
    unit_pages: dict[str, tuple[str, int]] = {}
    unit_concepts: dict[str, set[str]] = defaultdict(set)
    sentence_concepts: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    sentence_info: dict[tuple[str, int, int], dict[str, Any]] = {}
    first_quote: dict[tuple[str, str], dict[str, Any]] = {}
    for concept in concepts:
        for mention in concept["mentions"]:
            unit = mention["content_unit_id"]
            unit_pages[unit] = (mention["document_id"], mention["page_number"])
            unit_concepts[unit].add(concept["_id"])
            text = page_texts.get(unit, "")
            for start, end in mention.get("sentence_spans", []):
                key = (unit, start, end)
                sentence_concepts[key].add(concept["_id"])
                if end <= len(text):
                    sentence_info[key] = {"document_id": mention["document_id"], "content_unit_id": unit,
                                          "page_number": mention["page_number"], "quote": text[start:end],
                                          "start": start, "end": end}
            if mention.get("quotes"):
                first_quote[(concept["_id"], unit)] = {"document_id": mention["document_id"], "content_unit_id": unit,
                                                       "page_number": mention["page_number"], **mention["quotes"][0]}

    # Page windows: page p together with page p + 1 of the same document.
    pages_by_doc: dict[str, dict[int, str]] = defaultdict(dict)
    for unit, (document_id, page_number) in unit_pages.items():
        pages_by_doc[document_id][page_number] = unit
    windows: list[tuple[str, ...]] = []
    for document_id, pages in pages_by_doc.items():
        numbers = sorted(pages)
        if len(numbers) == 1:
            windows.append((pages[numbers[0]],))
        for number in numbers:
            if number + 1 in pages:
                windows.append((pages[number], pages[number + 1]))
            elif number - 1 not in pages:
                windows.append((pages[number],))
    window_count: Counter = Counter()
    pair_windows: Counter = Counter()
    pair_window_example: dict[tuple[str, str], tuple[str, ...]] = {}
    for window in windows:
        members = sorted(set().union(*(unit_concepts[unit] for unit in window)))
        window_count.update(members)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair_windows[(a, b)] += 1
                pair_window_example.setdefault((a, b), window)

    pair_sentences: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for key, members in sentence_concepts.items():
        ordered = sorted(members)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pair_sentences[(a, b)].append(key)

    @lru_cache(maxsize=None)
    def pattern_for(concept_id: str) -> re.Pattern[str]:
        concept = by_id[concept_id]
        return _name_pattern([concept["canonical_name"], *concept.get("aliases", [])])

    total = len(windows)
    edges: list[dict[str, Any]] = []
    for pair in sorted(set(pair_windows) | set(pair_sentences)):
        a, b = pair
        sentences = sorted(pair_sentences.get(pair, []))
        joint = pair_windows.get(pair, 0)
        score = npmi(joint, window_count[a], window_count[b], total)
        if sentences:
            if len(sentences) < MIN_SHARED_SENTENCES and score < MIN_NPMI_SENTENCE:
                continue
        elif joint < MIN_WINDOW_ONLY or score < MIN_NPMI_WINDOW:
            continue

        evidence: list[dict[str, Any]] = []
        relation_votes: Counter = Counter()
        pattern_matches: list[dict[str, Any]] = []
        for key in sentences:
            info = sentence_info.get(key)
            if info is None:
                continue
            entry = {**info, "kind": "same_sentence"}
            found = detect_relation(info["quote"], pattern_for(a), pattern_for(b))
            if found:
                relation_votes[found] += 1
                pattern_matches.append({"relation": found[0], "quote": info["quote"],
                                        "content_unit_id": info["content_unit_id"]})
                evidence.insert(0, entry)
            else:
                evidence.append(entry)
        if not evidence:
            window = pair_window_example.get(pair, ())
            for concept_id in (a, b):
                for unit in window:
                    quote = first_quote.get((concept_id, unit))
                    if quote:
                        evidence.append({**quote, "kind": "page_window"})
                        break
        if not evidence:
            continue

        relation, source, target = "related_to", a, b
        if relation_votes:
            (relation, a_is_source), _ = sorted(relation_votes.items(), key=lambda item: (-item[1], item[0]))[0]
            source, target = (a, b) if a_is_source else (b, a)
        sentence_count = len(sentences)
        confidence = 0.5 * max(score, 0.0) + 0.5 * min(1.0, sentence_count / 3)
        edges.append({
            "_id": stable_id("relationship", a, b),
            "source_id": source,
            "target_id": target,
            "relation_type": relation,
            "confidence": round(min(1.0, confidence), 3),
            "weight": round(sentence_count + 0.25 * joint, 3),
            "npmi": round(score, 4),
            "sentence_count": sentence_count,
            "window_count": joint,
            "evidence": evidence[:MAX_EVIDENCE],
            "pattern_matches": pattern_matches[:3],
            "version": RELATION_VERSION,
        })
    stats = {"windows": total, "candidate_pairs": len(set(pair_windows) | set(pair_sentences)),
             "edges": len(edges), "types": dict(Counter(edge["relation_type"] for edge in edges))}
    return edges, stats


def store_relationships(store: MongoKnowledgeStore) -> dict[str, Any]:
    """Rebuild the ``relationships`` collection from the current concept nodes."""
    started = time.perf_counter()
    concepts = list(store.concepts.find({}, {"canonical_name": 1, "aliases": 1, "mentions": 1}))
    unit_ids = sorted({mention["content_unit_id"] for concept in concepts for mention in concept["mentions"]})
    page_texts = {unit["_id"]: unit.get("text") or "" for unit in
                  store.content_units.find({"_id": {"$in": unit_ids}}, {"text": 1})}
    edges, stats = build_relationships(concepts, page_texts)
    collection = store.database["relationships"]
    stale = upsert_all(collection, edges)
    collection.create_index([("source_id", 1)], name="ix_rel_source")
    collection.create_index([("target_id", 1)], name="ix_rel_target")
    stats.update({"stale_removed": len(stale), "seconds": round(time.perf_counter() - started, 1)})
    logger.info("relationship summary: %s", stats)
    return stats
