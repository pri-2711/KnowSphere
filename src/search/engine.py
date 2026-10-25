"""Keyword search and extractive question answering with mandatory citations.

Nothing here generates text. Answers are sentences copied from stored page text
(``text[start:end]``), and every returned quote passes ``validate_quote`` before
it leaves this module. When the evidence is weak the answer is exactly
``NOT_FOUND`` with no citations.

Lookups are hash based: postings by term, concepts by alias key, pages by ID
and by (document_id, page_number). Autocomplete uses binary search over a
sorted alias list.
"""

from __future__ import annotations

import bisect
import math
import re
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

from src.graph.graph_builder import analyze, build_graph
from src.metadata import MetadataStore, format_citation
from src.storage.mongodb_store import MongoKnowledgeStore, stable_id
from src.textutil import collapse_whitespace, concept_key, split_sentences, tokenize, word_spans

NOT_FOUND = "Not found in the uploaded documents."
K1, B = 1.5, 0.75
CONCEPT_BOOST = 2.0          # added to a page score per query concept mentioned on the page
PHRASE_BOOST = 3.0           # added when the query words appear in the same order (scaled by phrase length)
PHRASE_RERANK = 50           # how many BM25 pages are checked for the phrase
TOP_PAGES = 10
MAX_ANSWER_SENTENCES = 5
MIN_COVERAGE = 0.6           # share of the query's IDF weight a sentence must contain
MIN_SENTENCE_SCORE = 1.0
MIN_SENTENCE_WORDS, MAX_SENTENCE_WORDS = 5, 120


def phrase_match(token_text: str, phrase: list[str]) -> float:
    """Score for the longest run of query terms (2 or more, in order) found in a token string.

    ``token_text`` is " term term ... " (space padded). Returns 0 when no run of two terms matches,
    otherwise (run length / phrase length) * min(occurrences, 3).
    """
    for size in range(len(phrase), 1, -1):
        best = 0
        for start in range(0, len(phrase) - size + 1):
            best = max(best, token_text.count(" " + " ".join(phrase[start:start + size]) + " "))
        if best:
            return size / len(phrase) * min(best, 3)
    return 0.0


class CitationError(ValueError):
    """Raised when a quote is not an exact slice of the cited page."""


class KnowledgeService:
    """In-memory caches over MongoDB used by the API, loaded at server start."""

    def __init__(self, store: MongoKnowledgeStore, metadata: MetadataStore) -> None:
        self.store = store
        self.metadata = metadata
        self._lock = threading.RLock()
        self.loaded_at: float | None = None
        self.load()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        with self._lock:
            started = time.perf_counter()
            database = self.store.database
            self.pages: dict[str, dict[str, Any]] = {}
            self._page_tokens: dict[str, str] = {}
            self.page_by_key: dict[tuple[str, int], str] = {}
            for unit in self.store.content_units.find({"unit_type": "page"},
                                                      {"text": 1, "document_id": 1, "page_number": 1}):
                self.pages[unit["_id"]] = {"text": unit.get("text") or "", "document_id": unit["document_id"],
                                           "page_number": unit["page_number"]}
                self.page_by_key[(unit["document_id"], unit["page_number"])] = unit["_id"]
                # Stemmed tokens per page, used for phrase matching without tokenizing at query time.
                self._page_tokens[unit["_id"]] = " " + " ".join(tokenize(unit.get("text") or "")) + " "

            self.postings: dict[str, list[tuple[str, int]]] = {
                record["_id"]: [(unit, tf) for unit, tf in record["postings"]]
                for record in database["search_index"].find()}
            self.doc_len = {record["_id"]: record["length"] for record in database["page_terms"].find({}, {"length": 1})}
            meta = database["search_meta"].find_one({"_id": "bm25"}) or {}
            self.page_count = meta.get("pages", len(self.doc_len))
            self.avg_len = meta.get("avg_length") or 1.0

            self.concepts: dict[str, dict[str, Any]] = {}
            self.unit_concepts: dict[str, set[str]] = defaultdict(set)
            self.sentence_concepts: dict[tuple[str, int, int], set[str]] = defaultdict(set)
            for concept in self.store.concepts.find({}, {"mentions.quotes": 0}):
                mentions = concept.pop("mentions", [])
                concept["mention_units"] = [mention["content_unit_id"] for mention in mentions]
                self.concepts[concept["_id"]] = concept
                for mention in mentions:
                    self.unit_concepts[mention["content_unit_id"]].add(concept["_id"])
                    for start, end in mention.get("sentence_spans", []):
                        self.sentence_concepts[(mention["content_unit_id"], start, end)].add(concept["_id"])
            self.alias_to_concept = {record["_id"]: record["concept_id"]
                                     for record in database["concept_aliases"].find()
                                     if record["concept_id"] in self.concepts}
            entries = set()
            for concept_id, concept in self.concepts.items():
                for name in [concept["canonical_name"], *concept.get("aliases", [])]:
                    entries.add((name.lower(), concept_id))
            self.autocomplete_entries = sorted(entries)
            self.autocomplete_keys = [name for name, _ in self.autocomplete_entries]

            self.graph = analyze(build_graph(
                self.store.concepts.find({}, {"mentions": 0}), database["relationships"].find()))
            self.load_seconds = round(time.perf_counter() - started, 3)
            self.loaded_at = time.time()

    # ---------------------------------------------------------------- helpers
    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.page_count - df + 0.5) / (df + 0.5))

    def page_id(self, document_id: str, page_number: int) -> str | None:
        """O(1) page lookup by (document, page); the stored ID is the same hash."""
        unit_id = self.page_by_key.get((document_id, page_number))
        if unit_id is None:
            candidate = stable_id("content_unit", document_id, "page", page_number)
            unit_id = candidate if candidate in self.pages else None
        return unit_id

    def citation(self, document_id: str, page_number: int) -> str:
        return format_citation(self.metadata.get(document_id), page_number)

    def validate_quote(self, unit_id: str, start: int, end: int, quote: str,
                       document_id: str, page_number: int) -> None:
        """Raise CitationError unless the quote is an exact slice of the cited page."""
        page = self.pages.get(unit_id)
        if page is None:
            raise CitationError(f"Unknown page {unit_id}")
        if page["document_id"] != document_id or page["page_number"] != page_number:
            raise CitationError("Citation document or page does not match the stored page")
        if not quote or page["text"][start:end] != quote or quote not in page["text"]:
            raise CitationError("Quote is not an exact substring of the cited page")

    def match_concepts(self, text: str, max_words: int = 5) -> list[str]:
        """Find concepts whose name or alias appears in the text (alias hash lookup over n-grams)."""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text)
        found: list[str] = []
        for size in range(min(max_words, len(words)), 0, -1):
            for start in range(0, len(words) - size + 1):
                concept_id = self.alias_to_concept.get(concept_key(" ".join(words[start:start + size])))
                if concept_id and concept_id not in found:
                    found.append(concept_id)
        return found

    def concept_summary(self, concept_id: str) -> dict[str, Any]:
        concept = self.concepts[concept_id]
        node = self.graph.nodes.get(concept_id, {})
        return {"id": concept_id, "name": concept["canonical_name"], "type": concept.get("concept_type"),
                "mention_count": concept.get("mention_count", 0), "degree": node.get("degree", 0)}

    # ----------------------------------------------------------------- ranking
    def bm25(self, terms: Iterable[str]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for unit_id, tf in postings:
                length = self.doc_len.get(unit_id, self.avg_len)
                scores[unit_id] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_len))
        return scores

    def page_token_text(self, unit_id: str) -> str:
        cached = self._page_tokens.get(unit_id)
        if cached is None:
            cached = " " + " ".join(tokenize(self.pages[unit_id]["text"])) + " "
            self._page_tokens[unit_id] = cached
        return cached

    def rank_pages(self, query: str, document_id: str | None = None) -> tuple[list[tuple[str, float]], list[str], list[str]]:
        phrase = tokenize(query)
        terms = list(dict.fromkeys(phrase))
        scores = self.bm25(terms)
        if len(phrase) >= 2:
            for unit_id, _ in sorted(scores.items(), key=lambda item: -item[1])[:PHRASE_RERANK]:
                scores[unit_id] += PHRASE_BOOST * phrase_match(self.page_token_text(unit_id), phrase)
        concept_ids = self.match_concepts(query)
        for concept_id in concept_ids:
            for unit_id in self.concepts[concept_id]["mention_units"]:
                if unit_id in scores:
                    scores[unit_id] += CONCEPT_BOOST
        ranked = sorted(((unit, score) for unit, score in scores.items()
                         if unit in self.pages and (document_id is None or self.pages[unit]["document_id"] == document_id)),
                        key=lambda item: (-item[1], item[0]))
        return ranked, terms, concept_ids

    def _highlights(self, quote: str, terms: set[str]) -> list[list[int]]:
        return [[start, end] for start, end, stemmed in word_spans(quote) if stemmed in terms]

    def _quote_payload(self, unit_id: str, start: int, end: int, terms: set[str]) -> dict[str, Any]:
        page = self.pages[unit_id]
        quote = page["text"][start:end]
        self.validate_quote(unit_id, start, end, quote, page["document_id"], page["page_number"])
        return {"quote": quote, "start": start, "end": end, "highlights": self._highlights(quote, terms),
                "content_unit_id": unit_id, "document_id": page["document_id"], "page_number": page["page_number"],
                "citation": self.citation(page["document_id"], page["page_number"])}

    # ------------------------------------------------------------------ search
    def search(self, query: str, limit: int = 10, document_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            ranked, terms, concept_ids = self.rank_pages(query, document_id)
            term_set = set(terms)
            results = []
            for unit_id, score in ranked[:limit]:
                text = self.pages[unit_id]["text"]
                best, best_hits = None, -1
                for start, end in split_sentences(text):
                    hits = sum(1 for _, _, stemmed in word_spans(text[start:end]) if stemmed in term_set)
                    if hits > best_hits:
                        best, best_hits = (start, end), hits
                if best is None:
                    continue
                try:
                    payload = self._quote_payload(unit_id, best[0], best[1], term_set)
                except CitationError:
                    continue
                results.append({**payload, "score": round(score, 3)})
            return {"query": query, "terms": terms, "results": results,
                    "concepts": [self.concept_summary(concept_id) for concept_id in concept_ids[:10]],
                    "took_ms": round((time.perf_counter() - started) * 1000, 2)}

    def autocomplete(self, prefix: str, limit: int = 10) -> list[dict[str, Any]]:
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        position = bisect.bisect_left(self.autocomplete_keys, prefix)
        seen: dict[str, str] = {}
        while position < len(self.autocomplete_entries) and len(seen) < limit * 3:
            name, concept_id = self.autocomplete_entries[position]
            if not name.startswith(prefix):
                break
            seen.setdefault(concept_id, name)
            position += 1
        ranked = sorted(seen, key=lambda concept_id: -self.concepts[concept_id].get("mention_count", 0))[:limit]
        return [{**self.concept_summary(concept_id), "matched": seen[concept_id]} for concept_id in ranked]

    # --------------------------------------------------------------------- ask
    def ask(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            ranked, terms, query_concepts = self.rank_pages(question)
            result = self._answer(ranked, terms, query_concepts, tokenize(question))
        result["question"] = question
        result["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _not_found(self, terms: list[str]) -> dict[str, Any]:
        return {"answer_found": False, "message": NOT_FOUND, "groups": [], "sentences": [], "concepts": [],
                "terms": terms}

    def _answer(self, ranked: list[tuple[str, float]], terms: list[str], query_concepts: list[str],
                phrase: list[str] | None = None) -> dict[str, Any]:
        if not terms or not ranked:
            return self._not_found(terms)
        weights = {term: self.idf(term) for term in terms}
        total_weight = sum(weights.values()) or 1.0
        top_score = ranked[0][1] or 1.0
        query_concept_set = set(query_concepts)
        candidates = []
        for unit_id, page_score in ranked[:TOP_PAGES]:
            text = self.pages[unit_id]["text"]
            for start, end in split_sentences(text):
                sentence = text[start:end]
                if ".." in sentence.replace(". .", ".."):  # table-of-contents lines
                    continue
                tokens = tokenize(sentence)
                words = len(sentence.split())
                if not MIN_SENTENCE_WORDS <= words <= MAX_SENTENCE_WORDS:
                    continue
                counts = Counter(tokens)
                matched = [term for term in terms if counts.get(term)]
                if not matched:
                    continue
                coverage = sum(weights[term] for term in matched) / total_weight
                if coverage < MIN_COVERAGE:
                    continue
                score = sum(weights[term] * counts[term] * (K1 + 1) / (counts[term] + K1) for term in matched)
                concepts_here = self.sentence_concepts.get((unit_id, start, end), set())
                score += 0.5 * len(concepts_here & query_concept_set)
                score += 0.5 * page_score / top_score
                if phrase and len(phrase) >= 2:
                    score += PHRASE_BOOST * phrase_match(" " + " ".join(tokens) + " ", phrase)
                if score < MIN_SENTENCE_SCORE:
                    continue
                candidates.append((score, coverage, unit_id, start, end))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

        selected, seen_text = [], set()
        for score, coverage, unit_id, start, end in candidates:
            key = collapse_whitespace(self.pages[unit_id]["text"][start:end]).lower()
            if key in seen_text:
                continue
            try:
                payload = self._quote_payload(unit_id, start, end, set(terms))
            except CitationError:
                continue  # never return a quote that fails validation
            seen_text.add(key)
            selected.append({**payload, "score": round(score, 3), "coverage": round(coverage, 3)})
            if len(selected) >= MAX_ANSWER_SENTENCES:
                break
        if not selected:
            return self._not_found(terms)

        groups: dict[tuple[str, int], dict[str, Any]] = {}
        for item in selected:
            key = (item["document_id"], item["page_number"])
            group = groups.setdefault(key, {"document_id": item["document_id"], "page_number": item["page_number"],
                                            "content_unit_id": item["content_unit_id"], "citation": item["citation"],
                                            "sentences": []})
            group["sentences"].append(item)
        related: Counter = Counter({concept_id: 3 for concept_id in query_concepts})
        for item in selected:
            related.update(self.sentence_concepts.get((item["content_unit_id"], item["start"], item["end"]), set()))
        concepts = [self.concept_summary(concept_id) for concept_id, _ in
                    sorted(related.items(), key=lambda pair: (-pair[1], pair[0]))[:8] if concept_id in self.concepts]
        return {"answer_found": True, "message": None, "groups": list(groups.values()), "sentences": selected,
                "concepts": concepts, "terms": terms}

    # ------------------------------------------------------------------ concepts
    def concept_detail(self, concept_id: str) -> dict[str, Any] | None:
        record = self.store.concepts.find_one({"_id": concept_id})
        if record is None:
            return None
        description = record.get("description")
        if description:
            page = self.pages.get(description["content_unit_id"])
            if page is None or page["text"][description["start"]:description["end"]] != description["quote"]:
                description = None
            else:
                description = {**description, "citation": self.citation(description["document_id"],
                                                                        description["page_number"])}
        related = []
        if concept_id in self.graph:
            for neighbor in self.graph.neighbors(concept_id):
                data = self.graph.edges[concept_id, neighbor]
                evidence = [{**item, "citation": self.citation(item["document_id"], item["page_number"])}
                            for item in data["evidence"][:3]]
                related.append({**self.concept_summary(neighbor), "relation_type": data["relation_type"],
                                "direction": "out" if data["source_id"] == concept_id else "in",
                                "weight": data["weight"], "confidence": data["confidence"], "evidence": evidence})
        related.sort(key=lambda item: (-item["weight"], item["name"]))
        sources = []
        for mention in record.get("mentions", []):
            quote = mention["quotes"][0] if mention.get("quotes") else None
            sources.append({"document_id": mention["document_id"], "page_number": mention["page_number"],
                            "content_unit_id": mention["content_unit_id"], "count": mention["count"],
                            "quote": quote, "citation": self.citation(mention["document_id"], mention["page_number"])})
        images = []
        for unit in self.store.content_units.find({"_id": {"$in": record.get("image_ids", [])}},
                                                  {"caption": 1, "page_number": 1, "document_id": 1}):
            images.append({"id": unit["_id"], "caption": unit.get("caption"), "page_number": unit["page_number"],
                           "citation": self.citation(unit["document_id"], unit["page_number"])})
        node = self.graph.nodes.get(concept_id, {})
        return {"id": concept_id, "name": record["canonical_name"], "type": record.get("concept_type"),
                "aliases": record.get("aliases", []), "description": description,
                "mention_count": record.get("mention_count", 0), "page_count": record.get("page_count", 0),
                "degree": node.get("degree", 0), "community": node.get("community"),
                "related": related[:30], "sources": sources, "images": images}

    # -------------------------------------------------------------------- graph
    def graph_payload(self, document_id: str | None = None, concept_type: str | None = None,
                      min_weight: float = 0.0, limit: int = 250, focus: str | None = None) -> dict[str, Any]:
        with self._lock:
            graph = self.graph
            nodes = [node for node, data in graph.nodes(data=True)
                     if (document_id is None or document_id in data.get("document_ids", []))
                     and (concept_type is None or data.get("concept_type") == concept_type)]
            nodes.sort(key=lambda node: (-graph.nodes[node].get("importance", 0), node))
            chosen = set(nodes[:limit])
            if focus and focus in graph:
                chosen.add(focus)
                chosen.update(list(graph.neighbors(focus))[:40])
            elements = []
            for node in chosen:
                data = graph.nodes[node]
                elements.append({"data": {"id": node, "label": data["name"], "type": data.get("concept_type"),
                                          "mentions": data.get("mention_count", 0), "degree": data.get("degree", 0),
                                          "importance": data.get("importance", 0.0),
                                          "community": data.get("community", 0)}})
            edge_count = 0
            for source, target, data in graph.edges(chosen, data=True):
                if source in chosen and target in chosen and data["weight"] >= min_weight:
                    elements.append({"data": {"id": data["id"], "source": data["source_id"], "target": data["target_id"],
                                              "relation": data["relation_type"], "weight": data["weight"],
                                              "confidence": data["confidence"]}})
                    edge_count += 1
            types = sorted({data.get("concept_type") for _, data in graph.nodes(data=True) if data.get("concept_type")})
            return {"elements": elements, "node_count": len(chosen), "edge_count": edge_count,
                    "total_nodes": graph.number_of_nodes(), "total_edges": graph.number_of_edges(), "types": types}

    def stats(self) -> dict[str, Any]:
        return {"documents": self.store.documents.count_documents({}),
                "pages": len(self.pages),
                "images": self.store.content_units.count_documents({"unit_type": "image"}),
                "concepts": len(self.concepts), "relationships": self.graph.number_of_edges(),
                "index_terms": len(self.postings),
                "index_postings": sum(len(items) for items in self.postings.values()),
                "communities": len({data.get("community") for _, data in self.graph.nodes(data=True)}),
                "cache_load_seconds": self.load_seconds}
