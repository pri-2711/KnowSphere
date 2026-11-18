"""Hybrid retrieval and evidence-first research answers.

Three methods rank pages for a query, and reciprocal rank fusion (RRF) combines them:

- lexical: BM25 with phrase and concept boosts (``engine.KnowledgeService``), exact names and numbers;
- semantic: passage embeddings (``semantic.SemanticIndex``), related meaning in other words;
- graph: pages that evidence the concepts named in the query and the relationships around them
  (``knowledge_graph.GovernedGraph``).

Every returned quote is an exact slice of its page and passes ``validate_quote`` before it is
returned. ``research`` adds what the knowledge base says about the question: accepted claims as
facts, interpretations and hypotheses (as the source words them), unconfirmed claims apart,
conflicts governance found, and unit conversions computed deterministically. Nothing is generated:
when no passage supports the question, the answer is an abstention.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from src.contracts.interfaces import Embedder
from src.contracts.schemas import Quantity
from src.evidence_quality.units import convert, requested_unit
from src.retrieval.engine import NOT_FOUND, CitationError, KnowledgeService
from src.retrieval.knowledge_graph import GovernedGraph, group_claims
from src.retrieval.semantic import SemanticIndex
from src.textutil import split_sentences, word_spans

RRF_K = 60
DEPTH = 50                 # results taken from each method before fusion


def fuse(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, list[str]]]:
    """Reciprocal rank fusion: (id, score, methods that found it), best first, ties by id."""
    scores: dict[str, float] = defaultdict(float)
    methods: dict[str, list[str]] = defaultdict(list)
    for method, ranked in rankings.items():
        for rank, item in enumerate(ranked):
            scores[item] += 1.0 / (k + rank + 1)
            methods[item].append(method)
    return sorted(((item, round(score, 6), methods[item]) for item, score in scores.items()),
                  key=lambda row: (-row[1], row[0]))


class HybridRetriever:
    def __init__(self, lexical: KnowledgeService, graph: GovernedGraph | None = None,
                 semantic: SemanticIndex | None = None, embedder: Embedder | None = None) -> None:
        self.lexical = lexical
        self.graph = graph
        self.semantic = semantic if semantic is not None and semantic.ready and embedder is not None else None
        self.embedder = embedder

    @property
    def methods(self) -> list[str]:
        return ["lexical"] + (["semantic"] if self.semantic else []) + (["graph"] if self.graph else [])

    def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        started = time.perf_counter()
        pages = self.lexical.pages
        ranked, terms, _ = self.lexical.rank_pages(query)
        rankings: dict[str, list[str]] = {"lexical": [unit for unit, _ in ranked[:DEPTH]]}
        best_passage: dict[str, tuple[int, int]] = {}
        if self.semantic:
            semantic_units: list[str] = []
            for passage, _ in self.semantic.query(query, self.embedder, DEPTH * 2):
                unit = passage["unit_id"]
                if unit in pages and unit not in best_passage:
                    best_passage[unit] = (passage["start"], passage["end"])
                    semantic_units.append(unit)
            rankings["semantic"] = semantic_units[:DEPTH]
        concepts: list[str] = []
        if self.graph:
            concepts = self.graph.match(query)
            units = self.graph.evidence_units(concepts) + self.graph.related_units(concepts)
            rankings["graph"] = list(dict.fromkeys(u for u in units if u in pages))[:DEPTH]
        results = []
        for unit, score, methods in fuse(rankings):
            if len(results) == limit:
                break
            span = best_passage.get(unit) or self._best_sentence(unit, set(terms)) or self._concept_quote(unit, concepts)
            if span is None:
                continue
            try:
                payload = self.lexical._quote_payload(unit, span[0], span[1], set(terms))
            except CitationError:
                continue                                       # never return a quote that fails the check
            results.append({**payload, "score": score, "methods": methods})
        return {"query": query, "methods": self.methods, "results": results,
                "concepts": [self.graph.summary(c) for c in concepts[:10]] if self.graph else [],
                "took_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _best_sentence(self, unit: str, terms: set[str]) -> tuple[int, int] | None:
        text = self.lexical.pages[unit]["text"]
        best, hits = None, 0
        for start, end in split_sentences(text):
            found = sum(1 for _, _, stem in word_spans(text[start:end]) if stem in terms)
            if found > hits:
                best, hits = (start, end), found
        return best

    def _concept_quote(self, unit: str, concepts: list[str]) -> tuple[int, int] | None:
        if not self.graph:
            return None
        text = self.lexical.pages[unit]["text"]
        for concept_id in concepts:
            for evidence in self.graph.concepts[concept_id].evidence:
                if evidence.source.content_unit_id == unit:
                    start = text.find(evidence.quote)
                    if start >= 0:
                        return start, start + len(evidence.quote)
        return None

    # ------------------------------------------------------------ research answers
    def research(self, question: str, limit: int = 8) -> dict[str, Any]:
        started = time.perf_counter()
        found = self.search(question, limit)
        extract = self.lexical.ask(question)
        if not found["results"] and not extract["answer_found"]:
            return {"question": question, "abstained": True, "message": NOT_FOUND, "methods": found["methods"],
                    "sentences": [],
                    "passages": [], "claims": group_claims([]), "conflicts": [], "calculations": [],
                    "took_ms": round((time.perf_counter() - started) * 1000, 2)}
        claims = self._claims(found)
        unit = requested_unit(question)
        return {"question": question, "abstained": False, "message": None,
                "sentences": extract["sentences"] if extract["answer_found"] else [],   # extractive, validated
                "methods": found["methods"], "passages": found["results"],
                "concepts": found["concepts"], "claims": group_claims(claims),
                "conflicts": [c for c in claims if c["conflicts"]],
                "calculations": self._calculations(claims, unit) if unit else [],
                "took_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _claims(self, found: dict[str, Any]) -> list[dict[str, Any]]:
        """Claims about the matched concepts or quoted on the returned pages, with checked citations."""
        if not self.graph:
            return []
        pages = {result["content_unit_id"] for result in found["results"]}
        wanted: list[str] = []
        for concept in found["concepts"]:
            wanted += self.graph.claims_by_concept.get(concept["id"], [])
        wanted += [claim.id for claim in self.graph.claims.values()
                   if any(e.source.content_unit_id in pages for e in claim.evidence)]
        rows = []
        for claim_id in dict.fromkeys(wanted):
            row = self.graph.claim(self.graph.claims[claim_id])
            row["citation_valid"] = all(self._quote_holds(e) for e in row["evidence"]) and bool(row["evidence"])
            if not row["citation_valid"]:
                row["standing"] = "unconfirmed"      # a claim whose quote no longer matches is never a fact
            rows.append(row)
        return rows

    def _quote_holds(self, evidence: dict[str, Any]) -> bool:
        page = self.lexical.pages.get(evidence["content_unit_id"])
        if page is None:
            return False
        text = page["text"]
        if evidence["start"] is not None and text[evidence["start"]:evidence["end"]] == evidence["quote"]:
            return True
        return evidence["quote"] in text

    @staticmethod
    def _calculations(claims: list[dict[str, Any]], unit: str) -> list[dict[str, Any]]:
        rows = []
        for claim in claims:
            for raw in claim["quantities"]:
                quantity = Quantity.model_validate(raw)
                value = convert(quantity, unit)
                if value is not None:
                    rows.append({"claim_id": claim["id"], "statement": claim["statement"], "stated": quantity.raw,
                                 "value": round(value, 6), "unit": unit, "standing": claim["standing"]})
        return rows
