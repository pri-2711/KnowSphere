"""Shared fixtures: in-memory MongoDB (mongomock), a fake GLiNER model, a fake embedder, a seeded corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mongomock
import numpy as np
import pytest

from src.knowledge.concept_extraction import GlinerExtractor, extract_concepts
from src.knowledge.merging import merge_concepts
from src.knowledge.relationships import store_relationships
from src.metadata import MetadataStore
from src.search.engine import KnowledgeService
from src.search.index import build_index
from src.storage.mongodb_store import MongoKnowledgeStore, stable_id

PHRASES = {
    "systems engineering": ("engineering concept", 0.9),
    "requirements analysis": ("process", 0.85),
    "preliminary design review": ("process", 0.8),
    "design review": ("process", 0.8),
    "risk management": ("process", 0.8),
    "verification and validation": ("process", 0.8),
    "system design": ("engineering concept", 0.7),
    "stakeholder needs": ("engineering concept", 0.7),
    "figure": ("engineering concept", 0.9),      # generic word, must be filtered
    "and the": ("method", 0.9),                  # fragment, must be filtered
    "weak signal": ("method", 0.2),              # below the score threshold
}

PAGES = {
    ("doc-a", 1): ("Systems engineering is an interdisciplinary approach to building complex systems. "
                   "Requirements analysis is a part of systems engineering. "
                   "The design review checks the system design against stakeholder needs. See figure 2."),
    ("doc-a", 2): ("Requirements analysis is used for defining stakeholder needs in a project.\n\n"
                   "A preliminary design review is held early in the program. "
                   "Requirements analysis and the design review are linked by traceability. A weak signal here."),
    ("doc-a", 3): ("Contents\nChapter 1 Introduction ........................ 1\n"
                   "Chapter 2 Requirements analysis .............. 5\nChapter 3 Design review .................. 9\n"
                   "Chapter 4 Risk management ................ 12\n"),
    ("doc-a", 4): ("Systems engineering includes risk management for the whole life cycle. "
                   "Risk management reduces schedule changes when risks occur and supports the design review."),
    ("doc-b", 1): ("Systems engineering supports verification and validation of the product. "
                   "Verification and validation confirms that requirements analysis was complete and correct. "
                   "Risk management and systems engineering share the same reviews."),
}


class FakeGliner:
    """Stands in for gliner.GLiNER.inference: finds fixed phrases (case-insensitive)."""

    def __init__(self, phrases: dict[str, tuple[str, float]] = PHRASES) -> None:
        self.phrases = phrases
        self.calls = 0
        self.texts: list[str] = []

    def inference(self, texts: list[str], labels: list[str], threshold: float = 0.5,
                  batch_size: int = 8, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls += 1
        self.texts.extend(texts)
        output = []
        for text in texts:
            entities = []
            for phrase, (label, score) in self.phrases.items():
                if score < threshold:
                    continue
                for match in re.finditer(re.escape(phrase), text, re.IGNORECASE):
                    entities.append({"start": match.start(), "end": match.end(), "text": match.group(0),
                                     "label": label, "score": score})
            output.append(entities)
        return output


class FakeEmbedder:
    """Maps chosen names to fixed vectors; every other name gets its own orthogonal direction."""

    def __init__(self, similar: dict[str, str] | None = None, dim: int = 256) -> None:
        self.similar = similar or {}
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for index, text in enumerate(texts):
            anchor = self.similar.get(text, text)
            slot = sum(ord(char) * (position + 1) for position, char in enumerate(anchor)) % (self.dim - 1)
            vectors[index, slot] = 1.0
            if anchor != text:  # near-duplicate of the anchor: cosine about 0.995
                vectors[index, self.dim - 1] = 0.1
            vectors[index] /= np.linalg.norm(vectors[index])
        return vectors


def page_id(document_id: str, page_number: int) -> str:
    return stable_id("content_unit", document_id, "page", page_number)


def insert_pages(store: MongoKnowledgeStore, pages: dict[tuple[str, int], str] = PAGES) -> None:
    for (document_id, page_number), text in pages.items():
        store.content_units.insert_one({
            "_id": page_id(document_id, page_number), "document_id": document_id, "unit_type": "page",
            "unit_index": page_number, "page_number": page_number, "text": text, "quality_status": "usable",
        })
    for document_id in sorted({key[0] for key in pages}):
        count = sum(1 for key in pages if key[0] == document_id)
        store.documents.insert_one({"_id": document_id, "source_sha256": document_id,
                                    "filename": f"{document_id}.pdf", "page_count": count,
                                    "file_storage": {"uri": f"data/raw_pdfs/{document_id}.pdf"}})


@pytest.fixture
def store() -> MongoKnowledgeStore:
    database = mongomock.MongoClient()["knowledge_map_test"]
    return MongoKnowledgeStore(database, Path.cwd())


@pytest.fixture
def metadata(tmp_path: Path) -> MetadataStore:
    meta = MetadataStore(tmp_path / "document_metadata.json")
    meta.record_detection("doc-a", "doc-a.pdf", {"title": "Systems Handbook", "author": "Jane Doe", "year": "2012",
                                                 "detected_from": {"title": "test"}})
    meta.record_detection("doc-b", "doc-b.pdf", {"title": "Engineering Fundamentals", "author": None,
                                                 "year": "2001", "detected_from": {}})
    return meta


def run_pipeline(store: MongoKnowledgeStore, model: FakeGliner | None = None) -> dict[str, Any]:
    extractor = GlinerExtractor(model=model or FakeGliner(), threshold=0.5)
    return {
        "concepts": extract_concepts(store, extractor),
        "merge": merge_concepts(store, FakeEmbedder()),
        "relations": store_relationships(store),
        "index": build_index(store),
    }


@pytest.fixture
def seeded(store: MongoKnowledgeStore) -> MongoKnowledgeStore:
    insert_pages(store)
    run_pipeline(store)
    return store


@pytest.fixture
def service(seeded: MongoKnowledgeStore, metadata: MetadataStore) -> KnowledgeService:
    return KnowledgeService(seeded, metadata)
