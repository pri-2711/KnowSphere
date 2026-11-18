"""Evaluation against a labelled dataset: retrieval Recall@10, citation correctness, extraction precision
and recall. Reads only; runs no model (retrieval is lexical, graph and, when indexed, semantic).

Dataset (JSON)::

    {"name": "...", "notes": "who labelled it and how",
     "retrieval": [{"query": "...", "relevant": [{"document": "file.pdf", "page": 12}]}],
     "extraction": [{"document": "file.pdf", "page": 3,
                     "concepts": ["Oxygen Generation System", "electrolysis"],
                     "relationships": [["Oxygen Generation System", "uses", "electrolysis"]]}]}

- Recall@k: share of a query's relevant pages among the top k results, averaged over queries;
  also hit rate (at least one relevant page in the top k) and MRR.
- Citation correctness: share of returned quotes that are exact slices of the cited page at the
  cited offsets (retrieval), and share of stored evidence quotes found in the unit they cite
  (knowledge base, every item of the graph).
- Extraction: per labelled page, the graph's concepts that cite that page against the expected
  names (matched by normalised name or alias); relationships by (source, type, target).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.knowledge_repository.projects import project_filter
from src.textutil import concept_key

K = 10


def _page_lookup(app: Any, project_id: str) -> dict[tuple[str, int], str]:
    """(document filename or ID, page) -> page unit ID."""
    names = {d["_id"]: d["filename"] for d in app.store.documents.find(project_filter(project_id), {"filename": 1})}
    lookup = {}
    for unit in app.store.content_units.find({"unit_type": "page", **project_filter(project_id)},
                                             {"document_id": 1, "page_number": 1}):
        lookup[(unit["document_id"], unit["page_number"])] = unit["_id"]
        if unit["document_id"] in names:
            lookup[(names[unit["document_id"]], unit["page_number"])] = unit["_id"]
    return lookup


def evaluate_retrieval(app: Any, project_id: str, queries: list[dict[str, Any]], method: str = "hybrid",
                       k: int = K) -> dict[str, Any]:
    lookup = _page_lookup(app, project_id)
    lexical = app.retriever(project_id)
    rows, recalls, reciprocal, hits, quotes, valid = [], [], [], 0, 0, 0
    for item in queries:
        relevant = {lookup.get((r["document"], r["page"])) for r in item["relevant"]} - {None}
        if not relevant:
            rows.append({"query": item["query"], "error": "no labelled page exists in this project"})
            continue
        found = app.search_hybrid(project_id, item["query"], k) if method == "hybrid" else \
            lexical.search(item["query"], k)
        ranked = [r["content_unit_id"] for r in found["results"]]
        for result in found["results"]:
            quotes += 1
            page = lexical.pages.get(result["content_unit_id"], {})
            valid += int(page.get("text", "")[result["start"]:result["end"]] == result["quote"])
        recall = len(relevant & set(ranked[:k])) / len(relevant)
        first = next((rank for rank, unit in enumerate(ranked, start=1) if unit in relevant), None)
        recalls.append(recall)
        reciprocal.append(1 / first if first else 0.0)
        hits += int(first is not None and first <= k)
        rows.append({"query": item["query"], "recall": round(recall, 3), "first_relevant_rank": first})
    scored = len(recalls)
    return {"method": method, "k": k, "queries": scored,
            f"recall_at_{k}": round(sum(recalls) / scored, 3) if scored else None,
            f"hit_rate_at_{k}": round(hits / scored, 3) if scored else None,
            "mrr": round(sum(reciprocal) / scored, 3) if scored else None,
            "returned_quotes": quotes, "citation_correctness": round(valid / quotes, 4) if quotes else None,
            "per_query": rows}


def evaluate_knowledge_citations(app: Any, project_id: str, source: str | None = None) -> dict[str, Any]:
    """Share of the graph's evidence quotes that occur in the unit they cite."""
    graph = app.knowledge_graph(project_id, source)
    items = [*graph.concepts.values(), *graph.relationships.values(), *graph.claims.values()]
    evidence = [e for item in items for e in item.evidence]
    units = app.repository.units(project_id, {e.source.content_unit_id for e in evidence})
    found = sum(1 for e in evidence if e.source.content_unit_id in units and e.quote in units[e.source.content_unit_id].text)
    return {"source": graph.source, "items": len(items), "evidence_quotes": len(evidence),
            "citation_correctness": round(found / len(evidence), 4) if evidence else None}


def evaluate_extraction(app: Any, project_id: str, pages: list[dict[str, Any]], source: str | None = None) -> dict[str, Any]:
    graph = app.knowledge_graph(project_id, source)
    lookup = _page_lookup(app, project_id)
    concept_keys = {c.id: {concept_key(n) for n in (c.name, *c.aliases)} for c in graph.concepts.values()}
    tp = fp = fn = rtp = rfp = rfn = 0
    rows = []
    for item in pages:
        unit = lookup.get((item["document"], item["page"]))
        if unit is None:
            rows.append({**item, "error": "page not found"})
            continue
        predicted = {c.id for c in graph.concepts.values() if any(e.source.content_unit_id == unit for e in c.evidence)}
        expected = {concept_key(name) for name in item.get("concepts", [])}
        matched = {key for key in expected if any(key in concept_keys[c] for c in predicted)}
        correct = {c for c in predicted if concept_keys[c] & expected}
        tp, fp, fn = tp + len(correct), fp + len(predicted - correct), fn + len(expected - matched)
        edges = [(concept_keys[r.source_id], r.relation_type, concept_keys[r.target_id])
                 for r in graph.relationships.values() if any(e.source.content_unit_id == unit for e in r.evidence)]
        gold = {(concept_key(s), t, concept_key(o)) for s, t, o in item.get("relationships", [])}
        hit_gold = {g for g in gold if any(g[0] in s and g[1] == t and g[2] in o for s, t, o in edges)}
        hit_pred = sum(1 for s, t, o in edges if any(g[0] in s and g[1] == t and g[2] in o for g in gold))
        rtp, rfp, rfn = rtp + hit_pred, rfp + len(edges) - hit_pred, rfn + len(gold - hit_gold)
        rows.append({"document": item["document"], "page": item["page"], "predicted": len(predicted),
                     "expected": len(expected), "correct": len(correct), "missed": sorted(expected - matched)})

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None
    return {"source": graph.source, "pages": len(pages),
            "concept_precision": ratio(tp, tp + fp), "concept_recall": ratio(tp, tp + fn),
            "relationship_precision": ratio(rtp, rtp + rfp), "relationship_recall": ratio(rtp, rtp + rfn),
            "per_page": rows}


def evaluate_dataset(app: Any, project_id: str, path: Path, method: str = "hybrid") -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"dataset": dataset.get("name", path.name), "notes": dataset.get("notes"),
                              "project_id": project_id}
    if dataset.get("retrieval"):
        report["retrieval"] = evaluate_retrieval(app, project_id, dataset["retrieval"], method)
    if dataset.get("extraction"):
        report["extraction"] = evaluate_extraction(app, project_id, dataset["extraction"])
    report["knowledge_citations"] = evaluate_knowledge_citations(app, project_id)
    return report
