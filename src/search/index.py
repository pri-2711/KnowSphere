"""Persistent inverted index over page text, stored in MongoDB.

``page_terms``   one record per page: text SHA-256, token count, and term counts.
                 A page is only tokenized again when its text hash changes.
``search_index`` one record per term: document frequency and postings
                 [content_unit_id, term frequency].
``search_meta``  corpus statistics used by BM25 (page count, average length).
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from src.storage.mongodb_store import MongoKnowledgeStore
from src.textutil import tokenize

INDEX_VERSION = "bm25-1"


def build_index(store: MongoKnowledgeStore) -> dict[str, Any]:
    """Update page_terms for changed pages, then rewrite postings and corpus statistics."""
    started = time.perf_counter()
    database = store.database
    page_terms = database["page_terms"]
    existing = {record["_id"]: record for record in page_terms.find({}, {"text_sha256": 1, "version": 1})}
    seen: set[str] = set()
    tokenized = unchanged = 0
    for page in store.content_units.find({"unit_type": "page"}, {"text": 1, "document_id": 1, "page_number": 1}):
        text = page.get("text") or ""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seen.add(page["_id"])
        record = existing.get(page["_id"])
        if record and record.get("text_sha256") == digest and record.get("version") == INDEX_VERSION:
            unchanged += 1
            continue
        tokens = tokenize(text)
        page_terms.replace_one({"_id": page["_id"]}, {
            "document_id": page["document_id"], "page_number": page["page_number"], "text_sha256": digest,
            "length": len(tokens), "terms": sorted([term, count] for term, count in Counter(tokens).items()),
            "version": INDEX_VERSION,
        }, upsert=True)
        tokenized += 1
    removed = [unit_id for unit_id in existing if unit_id not in seen]
    if removed:
        page_terms.delete_many({"_id": {"$in": removed}})

    postings: dict[str, list[list[Any]]] = defaultdict(list)
    lengths: list[int] = []
    for record in page_terms.find({}, {"terms": 1, "length": 1}):
        lengths.append(record["length"])
        for term, count in record["terms"]:
            postings[term].append([record["_id"], count])
    index = database["search_index"]
    index.delete_many({})
    documents = [{"_id": term, "df": len(items), "postings": sorted(items)} for term, items in sorted(postings.items())]
    for start in range(0, len(documents), 2000):
        index.insert_many(documents[start:start + 2000])
    page_count = len(lengths)
    database["search_meta"].replace_one({"_id": "bm25"}, {
        "pages": page_count,
        "avg_length": (sum(lengths) / page_count) if page_count else 0.0,
        "terms": len(documents),
        "postings": sum(item["df"] for item in documents),
        "version": INDEX_VERSION,
        "built_at": datetime.now(timezone.utc),
    }, upsert=True)
    return {"pages": page_count, "tokenized": tokenized, "unchanged": unchanged, "terms": len(documents),
            "postings": sum(item["df"] for item in documents), "seconds": round(time.perf_counter() - started, 2)}
