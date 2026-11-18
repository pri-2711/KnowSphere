"""Benchmark a text model on real chunks: latency, failures, retries, peak RAM and extraction quality.

Runs Pass 1 (the production prompt and validation) on a sample of stored chunks, one call at a
time as in production, and stores nothing: the results are returned and can be written to
``reports/``. RAM is the resident memory of the model server process (``ollama`` by default),
sampled before and after each call, which is what limits a 16 GB machine.

This module runs real inference when given a real model; it is only started by an explicit
command (``python -m src.api.cli benchmark --model ...``).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Iterable, Mapping

from src.contracts.interfaces import TextModel
from src.knowledge_extraction.llm_extraction import EXTRACTED, extract_chunk
from src.textutil import concept_key

Sampler = Callable[[], float | None]


def process_rss_mb(name: str = "ollama") -> Sampler:
    """Resident memory (MB) of all processes whose name starts with ``name``; None when none runs."""
    import psutil

    def sample() -> float | None:
        total = 0
        for process in psutil.process_iter(["name", "memory_info"]):
            if (process.info.get("name") or "").lower().startswith(name) and process.info.get("memory_info"):
                total += process.info["memory_info"].rss
        return round(total / 2**20, 1) if total else None
    return sample


def _percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))], 2)


def benchmark_text_model(model: TextModel, chunks: Iterable[Mapping[str, Any]], max_retries: int = 1,
                         sampler: Sampler | None = None,
                         gold: Mapping[tuple[str, int], list[str]] | None = None) -> dict[str, Any]:
    """``gold`` maps (document id, page) to expected concept names for quality on labelled pages."""
    latencies, retries, rows, peak = [], [], [], None
    failures = concepts = tp = fp = fn = labelled = 0
    for chunk in chunks:
        before = sampler() if sampler else None
        started = time.perf_counter()
        record = extract_chunk(chunk, model, max_retries)
        seconds = time.perf_counter() - started
        after = sampler() if sampler else None
        peak = max([v for v in (peak, before, after) if v is not None], default=None)
        latencies.append(seconds)
        retries.append(record["retries"])
        ok = record["status"] == EXTRACTED
        failures += int(not ok)
        accepted = [c for c in record["concepts"] if c["status"] != "rejected"]
        concepts += len(accepted)
        expected = (gold or {}).get((chunk["document_id"], chunk["page_number"]))
        if expected is not None:
            labelled += 1
            want = {concept_key(name) for name in expected}
            got = {c["key"] for c in accepted} | {concept_key(a) for c in accepted for a in c["aliases"]}
            correct = {c["key"] for c in accepted if c["key"] in want or {concept_key(a) for a in c["aliases"]} & want}
            tp, fp, fn = tp + len(correct), fp + len(accepted) - len(correct), fn + len(want - got)
        rows.append({"chunk_id": chunk["_id"], "seconds": round(seconds, 2), "status": record["status"],
                     "retries": record["retries"], "concepts": len(accepted), "error": record["error"],
                     "rss_mb": after})
    calls = len(latencies)
    return {
        "model": model.model, "chunks": calls,
        "latency_seconds": {"median": round(statistics.median(latencies), 2) if latencies else None,
                            "p95": _percentile(latencies, 0.95), "max": round(max(latencies), 2) if latencies else None},
        "failure_rate": round(failures / calls, 3) if calls else None,
        "mean_retries": round(sum(retries) / calls, 2) if calls else None,
        "concepts_per_chunk": round(concepts / calls, 2) if calls else None,
        "peak_server_rss_mb": peak,
        "quality": {"concept_precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "concept_recall": round(tp / (tp + fn), 3) if tp + fn else None,
                    "labelled_chunks": labelled} if gold else None,
        "per_chunk": rows,
    }


def run_benchmark(store: Any, project_id: str, model_name: str, chunk_count: int,
                  labels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Benchmark a local Ollama model on the project's first ``chunk_count`` chunks (never a cloud model)."""
    from dataclasses import replace

    from src.config import KGConfig
    from src.knowledge_repository.projects import project_filter
    from src.model_runtime.router import ModelRouter
    config = KGConfig()
    router = ModelRouter.from_llm_config(replace(config.llm, provider="ollama", model=model_name), config.runtime)
    scope = project_filter(project_id)
    chunks = list(store.content_units.find({"unit_type": "chunk", **scope}).sort("_id", 1).limit(chunk_count))
    gold = None
    if labels:
        names = {d["filename"]: d["_id"] for d in store.documents.find(scope, {"filename": 1})}
        gold = {(names.get(item["document"], item["document"]), item["page"]): item["concepts"] for item in labels}
    return benchmark_text_model(router.text_model(), chunks, config.max_retries, process_rss_mb(), gold)
