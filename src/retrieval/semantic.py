"""Semantic retrieval: passages of each page embedded with the router's embedder, searched by cosine.

A passage is a run of whole sentences (about ``PASSAGE_CHARS`` characters) and an exact slice of
its page, so a semantic hit is a quote that passes the same citation check as a lexical hit.

Vectors are stored per project in ``semantic_index`` with the passage text hash and the embedder
name; ``build`` embeds only passages that are new or changed and removes stale ones, so rebuilding
is incremental and idempotent. Building and querying need an embedder (a local model, fastembed by
default); without stored vectors the hybrid retriever simply runs without this method.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np

from src.contracts.ids import scoped_id
from src.contracts.interfaces import Embedder
from src.knowledge_repository.projects import project_filter
from src.textutil import split_sentences

COLLECTION = "semantic_index"
PASSAGE_CHARS = 700
BATCH = 64


def passages(unit_id: str, text: str, size: int = PASSAGE_CHARS) -> list[dict[str, Any]]:
    """Sentence-aligned windows of a page: {unit_id, start, end}."""
    result, start, end = [], None, None
    for sentence_start, sentence_end in split_sentences(text):
        if start is None:
            start = sentence_start
        elif sentence_end - start > size:
            result.append({"unit_id": unit_id, "start": start, "end": end})
            start = sentence_start
        end = sentence_end
    if start is not None and end is not None and text[start:end].strip():
        result.append({"unit_id": unit_id, "start": start, "end": end})
    return result


class SemanticIndex:
    def __init__(self, database: Any, project_id: str, model_name: str) -> None:
        self.collection = database[COLLECTION]
        self.project_id = project_id
        self.model_name = model_name
        self.entries: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None

    def build(self, pages: Mapping[str, Mapping[str, Any]], embedder: Embedder) -> dict[str, int]:
        """Embed new or changed passages of the given pages (unit id -> {"text"}); drop the rest."""
        wanted: dict[str, dict[str, Any]] = {}
        for unit_id, page in pages.items():
            text = page.get("text") or ""
            for passage in passages(unit_id, text):
                quote = text[passage["start"]:passage["end"]]
                record_id = scoped_id("semantic", self.project_id, unit_id, passage["start"], passage["end"])
                wanted[record_id] = {**passage, "sha": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                                     "quote": quote}
        existing = {r["_id"]: r for r in self.collection.find({**project_filter(self.project_id)},
                                                              {"sha": 1, "model": 1})}
        todo = [rid for rid, item in wanted.items()
                if existing.get(rid, {}).get("sha") != item["sha"] or existing.get(rid, {}).get("model") != self.model_name]
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            vectors = np.asarray(embedder.embed([wanted[rid]["quote"] for rid in batch]), dtype="float32")
            for rid, vector in zip(batch, vectors):
                item = wanted[rid]
                norm = float(np.linalg.norm(vector)) or 1.0
                self.collection.replace_one({"_id": rid}, {
                    "project_id": self.project_id, "unit_id": item["unit_id"], "start": item["start"],
                    "end": item["end"], "sha": item["sha"], "model": self.model_name,
                    "vector": (vector / norm).astype(float).tolist()}, upsert=True)
        stale = [rid for rid in existing if rid not in wanted]
        if stale:
            self.collection.delete_many({"_id": {"$in": stale}})
        self.matrix = None
        return {"passages": len(wanted), "embedded": len(todo), "reused": len(wanted) - len(todo),
                "removed": len(stale)}

    def load(self) -> int:
        self.entries = list(self.collection.find({"model": self.model_name, **project_filter(self.project_id)},
                                                 {"unit_id": 1, "start": 1, "end": 1, "vector": 1}))
        self.matrix = np.asarray([e.pop("vector") for e in self.entries], dtype="float32") if self.entries else None
        return len(self.entries)

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.entries) > 0

    def query(self, text: str, embedder: Embedder, limit: int = 50) -> list[tuple[dict[str, Any], float]]:
        """Passages most similar to the text, best first: (passage {unit_id, start, end}, cosine)."""
        if not self.ready:
            return []
        vector = np.asarray(embedder.embed([text]), dtype="float32")[0]
        vector = vector / (float(np.linalg.norm(vector)) or 1.0)
        scores = self.matrix @ vector
        order = np.argsort(-scores)[:limit]
        return [(self.entries[i], float(scores[i])) for i in order.tolist()]


def index_pages(pages: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {page["_id"]: {"text": page.get("text") or ""} for page in pages}
