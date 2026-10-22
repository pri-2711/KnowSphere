"""Merge concept mentions into concept nodes so names with the same meaning become ONE node.

Rules, applied in this order:
1. alias hash: names with the same ``concept_key`` (case, plural, hyphen and
   spacing differences only) always merge;
2. fuzzy string similarity (rapidfuzz ratio) at or above ``FUZZY_THRESHOLD``;
3. embedding cosine similarity (fastembed, bge-small) at or above
   ``EMBEDDING_THRESHOLD``.
Rules 2 and 3 are blocked when the concept types are incompatible, the names
contain different numbers, one name negates the other ("non-functional"), one
name is a strict sub-phrase of the other ("design review" vs "preliminary design
review"), or (embeddings only) the names share no word. A merge into an existing
cluster must also pass against that cluster's canonical name, so chains cannot
drift. Every decision is written to the merge report.
"""

from __future__ import annotations

import csv
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.storage.mongodb_store import MongoKnowledgeStore, stable_id
from src.textutil import concept_key, singularize, stem

MERGE_VERSION = "merge-1"
MIN_MENTIONS = 2
FUZZY_THRESHOLD = 94.0
FUZZY_MIN_LENGTH = 8
EMBEDDING_THRESHOLD = 0.92

# Types that may describe the same thing; anything else must match exactly.
TYPE_GROUPS = {
    "engineering concept": "abstract",
    "process": "abstract",
    "method": "abstract",
    "system component": "component",
    "document": "document",
    "organization": "organization",
}
_NEGATION_PREFIXES = ("non", "anti", "un", "dis", "in", "im")
_NEGATION_WORDS = frozenset({"non", "not", "no", "anti", "without"})
_DEFINITION = r"\b{name}s?\b[^.]{{0,40}}?\b(is|are|refers to|means|can be defined as|is defined as)\b"

logger = logging.getLogger("knowledge_map.merge")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Any:
        """Return an (n, d) array of L2-normalized vectors."""


class FastEmbedder:
    """fastembed ONNX model, used only for merging concept names."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> Any:
        import numpy as np
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        vectors = np.array(list(self._model.embed(list(texts), batch_size=256)), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


@dataclass
class Candidate:
    name: str
    count: int = 0
    pages: set = field(default_factory=set)
    forms: Counter = field(default_factory=Counter)
    labels: dict = field(default_factory=lambda: defaultdict(float))
    mentions: list = field(default_factory=list)

    @property
    def concept_type(self) -> str:
        return sorted(self.labels.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @property
    def words(self) -> set[str]:
        return {singularize(word) for word in re.findall(r"[a-z0-9]+", self.name.replace("-", " "))}


def load_candidates(store: MongoKnowledgeStore, min_mentions: int = MIN_MENTIONS) -> tuple[dict[str, Candidate], int]:
    """Aggregate mention records by normalized name; returns (kept candidates, raw name count)."""
    candidates: dict[str, Candidate] = {}
    for record in store.database["concept_mentions"].find():
        candidate = candidates.setdefault(record["name"], Candidate(record["name"]))
        candidate.count += record["count"]
        candidate.pages.add(record["content_unit_id"])
        for form, count in record["surface_forms"]:
            candidate.forms[form] += count
        for label, score in record["label_scores"]:
            candidate.labels[label] += score
        candidate.mentions.append(record)
    raw = len(candidates)
    kept = {name: candidate for name, candidate in candidates.items() if candidate.count >= min_mentions}
    return kept, raw


def _digits(name: str) -> set[str]:
    return set(re.findall(r"\d+", name))


def _negates(a: set[str], b: set[str]) -> bool:
    if (a ^ b) & _NEGATION_WORDS:          # "non functional" vs "functional"
        return True
    for word in a - b:                     # "nonfunctional" vs "functional"
        for prefix in _NEGATION_PREFIXES:
            if word.startswith(prefix) and word[len(prefix):].lstrip("-") in b:
                return True
    return False


def block_reason(a: Candidate, b: Candidate, rule: str) -> str | None:
    """Return why two candidates must not merge, or None when allowed."""
    if TYPE_GROUPS.get(a.concept_type, a.concept_type) != TYPE_GROUPS.get(b.concept_type, b.concept_type):
        return f"incompatible types ({a.concept_type} vs {b.concept_type})"
    if _digits(a.name) != _digits(b.name):
        return "different numbers"
    words_a, words_b = a.words, b.words
    if _negates(words_a, words_b) or _negates(words_b, words_a):
        return "negation"
    if words_a != words_b and (words_a < words_b or words_b < words_a):
        return "one name is a sub-phrase of the other"
    if rule == "embedding" and not ({stem(w) for w in words_a} & {stem(w) for w in words_b}):
        return "no shared word"
    return None


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a: int, b: int) -> int:
        root_a, root_b = self.find(a), self.find(b)
        # The smaller index has more mentions (names are sorted), so it stays canonical.
        keep, drop = min(root_a, root_b), max(root_a, root_b)
        self.parent[drop] = keep
        return keep


def _fuzzy_pairs(names: Sequence[str]) -> list[tuple[float, int, int]]:
    from rapidfuzz import fuzz, process
    eligible = [index for index, name in enumerate(names) if len(name) >= FUZZY_MIN_LENGTH]
    if len(eligible) < 2:
        return []
    texts = [names[index] for index in eligible]
    matrix = process.cdist(texts, texts, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD, workers=-1)
    pairs = []
    rows, cols = matrix.nonzero()
    for row, col in zip(rows.tolist(), cols.tolist()):
        if row < col:
            pairs.append((float(matrix[row, col]), eligible[row], eligible[col]))
    return pairs


def _embedding_pairs(names: Sequence[str], embedder: Embedder, block: int = 1024) -> list[tuple[float, int, int]]:
    import numpy as np
    if len(names) < 2:
        return []
    vectors = embedder.embed(list(names))
    pairs = []
    for start in range(0, len(names), block):
        scores = vectors[start:start + block] @ vectors.T
        rows, cols = np.nonzero(scores >= EMBEDDING_THRESHOLD)
        for row, col in zip(rows.tolist(), cols.tolist()):
            i = start + row
            if i < col:
                pairs.append((float(scores[row, col]), i, col))
    return pairs


def _pair_score(rule: str, a: str, b: str, vectors: Mapping[str, Any]) -> float:
    if rule == "fuzzy":
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a, b))
    return float(vectors[a] @ vectors[b])


def cluster_candidates(
    candidates: Mapping[str, Candidate],
    embedder: Embedder | None,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Return clusters of names (canonical first) and the merge decision log."""
    names = sorted(candidates, key=lambda name: (-candidates[name].count, name))
    index_of = {name: index for index, name in enumerate(names)}
    union = _UnionFind(len(names))
    log: list[dict[str, Any]] = []

    def record(decision: str, rule: str, score: float, a: str, b: str, reason: str = "") -> None:
        log.append({"decision": decision, "rule": rule, "score": round(score, 4), "name_a": a, "name_b": b,
                    "type_a": candidates[a].concept_type, "type_b": candidates[b].concept_type, "reason": reason})

    by_key: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_key[concept_key(name)].append(name)
    for members in by_key.values():
        for other in members[1:]:
            union.union(index_of[members[0]], index_of[other])
            record("merged", "alias_hash", 1.0, members[0], other, "same normalized key")

    vectors: dict[str, Any] = {}
    pairs: list[tuple[str, float, int, int]] = [("fuzzy", s, i, j) for s, i, j in _fuzzy_pairs(names)]
    if embedder is not None and names:
        embedded = embedder.embed(names)
        vectors = {name: embedded[index] for index, name in enumerate(names)}
        pairs += [("embedding", s, i, j) for s, i, j in _embedding_pairs(names, _Cached(embedded))]
    pairs.sort(key=lambda item: (item[0] != "fuzzy", -item[1], item[2], item[3]))

    for rule, score, i, j in pairs:
        root_i, root_j = union.find(i), union.find(j)
        if root_i == root_j:
            continue
        a, b = candidates[names[i]], candidates[names[j]]
        reason = block_reason(a, b, rule)
        if reason is None:
            canon_a, canon_b = candidates[names[root_i]], candidates[names[root_j]]
            if (canon_a.name, canon_b.name) != (a.name, b.name):
                chain_reason = block_reason(canon_a, canon_b, rule)
                threshold = FUZZY_THRESHOLD if rule == "fuzzy" else EMBEDDING_THRESHOLD
                if chain_reason is None and _pair_score(rule, canon_a.name, canon_b.name, vectors) < threshold:
                    chain_reason = "canonical names are not similar enough"
                if chain_reason:
                    reason = f"cluster check: {chain_reason}"
        if reason:
            record("rejected", rule, score, a.name, b.name, reason)
            continue
        union.union(i, j)
        record("merged", rule, score, a.name, b.name)

    clusters: dict[int, list[str]] = defaultdict(list)
    for index, name in enumerate(names):
        clusters[union.find(index)].append(name)
    return [clusters[root] for root in sorted(clusters)], log


class _Cached:
    """Wrap precomputed vectors so pair search does not embed twice."""

    def __init__(self, vectors: Any) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str]) -> Any:
        return self.vectors


def _choose_description(name_forms: Iterable[str], mentions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer a definitional sentence ("X is ..."); otherwise the first mention sentence."""
    patterns = [re.compile(_DEFINITION.format(name=re.escape(form)), re.IGNORECASE) for form in name_forms if form]
    for mention in mentions:
        for quote in mention["quotes"]:
            if any(pattern.search(quote["quote"]) for pattern in patterns):
                return {**quote, "document_id": mention["document_id"],
                        "content_unit_id": mention["content_unit_id"], "page_number": mention["page_number"]}
    for mention in mentions:
        if mention["quotes"]:
            return {**mention["quotes"][0], "document_id": mention["document_id"],
                    "content_unit_id": mention["content_unit_id"], "page_number": mention["page_number"]}
    return None


def build_nodes(
    clusters: list[list[str]],
    candidates: Mapping[str, Candidate],
    page_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Turn clusters into concept records with all aliases and all mentions."""
    nodes = []
    for members in clusters:
        canonical = candidates[members[0]]
        forms: Counter = Counter()
        labels: dict[str, float] = defaultdict(float)
        by_unit: dict[str, dict[str, Any]] = {}
        for name in members:
            candidate = candidates[name]
            forms.update(candidate.forms)
            for label, score in candidate.labels.items():
                labels[label] += score
            for record in candidate.mentions:
                unit = by_unit.setdefault(record["content_unit_id"], {
                    "document_id": record["document_id"], "content_unit_id": record["content_unit_id"],
                    "page_number": record["page_number"], "count": 0, "spans": set(),
                })
                unit["count"] += record["count"]
                unit["spans"].update(tuple(span) for span in record["sentences"])
        mentions = []
        for unit in sorted(by_unit.values(), key=lambda item: (item["document_id"], item["page_number"])):
            text = page_texts.get(unit["content_unit_id"], "")
            quotes = [{"quote": text[start:end], "start": start, "end": end}
                      for start, end in sorted(unit["spans"]) if end <= len(text)]
            mentions.append({"document_id": unit["document_id"], "content_unit_id": unit["content_unit_id"],
                             "page_number": unit["page_number"], "count": unit["count"], "quotes": quotes[:5],
                             "sentence_spans": sorted([start, end] for start, end in unit["spans"])})
        display = sorted(canonical.forms.items(), key=lambda item: (-item[1], item[0]))[0][0]
        aliases = sorted((set(forms) | set(members)) - {display})
        concept_type = sorted(labels.items(), key=lambda item: (-item[1], item[0]))[0][0]
        nodes.append({
            "_id": stable_id("concept", canonical.name),
            "canonical_name": display,
            "normalized_name": canonical.name,
            "aliases": aliases,
            "member_names": members,
            "concept_type": concept_type,
            "description": _choose_description([display, *members], mentions),
            "mentions": mentions,
            "mention_count": sum(mention["count"] for mention in mentions),
            "page_count": len(mentions),
            "document_ids": sorted({mention["document_id"] for mention in mentions}),
            "merge_version": MERGE_VERSION,
        })
    return nodes


def write_report(log: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["decision", "rule", "score", "name_a", "name_b", "type_a", "type_b", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(log)


def upsert_all(collection: Any, records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Upsert records by _id and delete records that are no longer produced. Returns removed IDs."""
    now = datetime.now(timezone.utc)
    keep = set()
    for record in records:
        payload = dict(record)
        record_id = payload.pop("_id")
        keep.add(record_id)
        collection.update_one({"_id": record_id}, {"$set": payload, "$setOnInsert": {"created_at": now}},
                              upsert=True)
    stale = [item["_id"] for item in collection.find({}, {"_id": 1}) if item["_id"] not in keep]
    if stale:
        collection.delete_many({"_id": {"$in": stale}})
    return stale


def merge_concepts(
    store: MongoKnowledgeStore,
    embedder: Embedder | None,
    report_path: Path | None = None,
    min_mentions: int = MIN_MENTIONS,
) -> dict[str, Any]:
    """Rebuild the ``concepts`` and ``concept_aliases`` collections from mentions."""
    started = time.perf_counter()
    candidates, raw_names = load_candidates(store, min_mentions)
    clusters, log = cluster_candidates(candidates, embedder)
    unit_ids = {record["content_unit_id"] for candidate in candidates.values() for record in candidate.mentions}
    page_texts = {unit["_id"]: unit.get("text") or "" for unit in
                  store.content_units.find({"_id": {"$in": sorted(unit_ids)}}, {"text": 1})}
    nodes = build_nodes(clusters, candidates, page_texts)

    stale = upsert_all(store.concepts, nodes)

    aliases: dict[str, tuple[int, str, str]] = {}
    for node in nodes:
        for alias in [node["canonical_name"], *node["aliases"]]:
            key = concept_key(alias)
            if key and (key not in aliases or aliases[key][0] < node["mention_count"]):
                aliases[key] = (node["mention_count"], node["_id"], alias)
    alias_collection = store.database["concept_aliases"]
    upsert_all(alias_collection, [{"_id": key, "concept_id": concept_id, "alias": alias}
                                  for key, (_, concept_id, alias) in sorted(aliases.items())])
    alias_collection.create_index([("concept_id", 1)], name="ix_alias_concept")

    if report_path is not None:
        write_report(log, report_path)
    rules = Counter(entry["rule"] for entry in log if entry["decision"] == "merged")
    summary = {
        "raw_names": raw_names,
        "names_after_min_mentions": len(candidates),
        "concepts": len(nodes),
        "merges": dict(rules),
        "rejected_merges": sum(1 for entry in log if entry["decision"] == "rejected"),
        "stale_removed": len(stale),
        "aliases": len(aliases),
        "seconds": round(time.perf_counter() - started, 1),
    }
    logger.info("merge summary: %s", summary)
    return summary
