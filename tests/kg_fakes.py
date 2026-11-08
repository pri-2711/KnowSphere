"""Shared fakes for the concept-first pipeline tests: a scripted LLM and a small two-document corpus."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from src.kg.llm import LLMResponse
from src.storage.mongodb_store import MongoKnowledgeStore, stable_id

DOC_X, DOC_Y = "doc-x", "doc-y"

PAGES = {
    (DOC_X, 1): ("Machine Learning Basics\n\n"
                 "Supervised learning is a type of machine learning that uses labeled data. Labeled data consists "
                 "of input examples paired with the correct output, prepared by people who know the answers.\n\n"
                 "Unsupervised learning finds structure in unlabeled data without any labels at all."),
    (DOC_X, 2): ("Contents\nChapter 1 Introduction ........ 1\nChapter 2 Methods ........ 5\n"
                 "Chapter 3 Results ....... 9\nChapter 4 Discussion ....... 12\n"),
    (DOC_X, 3): "",
    (DOC_Y, 1): ("Deep Learning Overview\n\n"
                 "Supervised Learning (SL) uses labeled data to train a model. A neural network is an example of "
                 "a model that is trained with SL in most practical systems today."),
}


def page_id(document_id: str, page_number: int) -> str:
    return stable_id("content_unit", document_id, "page", page_number)


def insert_corpus(store: MongoKnowledgeStore, pages: dict[tuple[str, int], str] = PAGES) -> None:
    for (document_id, page_number), text in pages.items():
        store.content_units.insert_one({"_id": page_id(document_id, page_number), "document_id": document_id,
                                        "unit_type": "page", "unit_index": page_number, "page_number": page_number,
                                        "text": text, "quality_status": "usable" if text else "no_text_detected"})
    for document_id in sorted({key[0] for key in pages}):
        store.documents.insert_one({"_id": document_id, "source_sha256": document_id,
                                    "filename": f"{document_id}.pdf",
                                    "page_count": sum(1 for key in pages if key[0] == document_id)})


def concept(name: str, type_: str, description: str, evidence: str, aliases: Sequence[str] = ()) -> dict[str, Any]:
    return {"name": name, "type": type_, "description": description, "aliases": list(aliases), "evidence": evidence}


def relation(source: str, type_: str, target: str, evidence: str) -> dict[str, Any]:
    return {"source": source, "type": type_, "target": target, "evidence": evidence}


ANSWERS = {
    "Supervised learning is a type": {
        "concepts": [
            concept("Supervised learning", "method", "Machine learning that learns from labeled data.",
                    "Supervised learning is a type of machine learning that uses labeled data."),
            concept("Machine learning", "concept", "The field supervised learning belongs to.",
                    "Supervised learning is a type of machine learning"),
            concept("Labeled data", "concept", "Input examples paired with the correct output.",
                    "Labeled data consists of input examples paired with the correct output"),
            concept("Unsupervised learning", "method", "Learning that finds structure in unlabeled data.",
                    "Unsupervised learning finds structure in unlabeled data"),
        ],
        "relationships": [
            relation("Supervised learning", "is_a", "Machine learning",
                     "Supervised learning is a type of machine learning"),
            relation("Supervised learning", "uses", "Labeled data",
                     "Supervised learning is a type of machine learning that uses labeled data."),
        ],
    },
    "Supervised Learning (SL) uses": {
        "concepts": [
            concept("Supervised Learning", "method", "Training a model with labeled data.",
                    "Supervised Learning (SL) uses labeled data to train a model.", aliases=["SL"]),
            concept("Labeled data", "concept", "Data used to train a model.",
                    "uses labeled data to train a model"),
            concept("Neural network", "artifact", "A model that is trained with supervised learning.",
                    "A neural network is an example of a model that is trained with SL"),
        ],
        "relationships": [
            relation("SL", "uses", "Labeled data", "Supervised Learning (SL) uses labeled data to train a model."),
            relation("Neural network", "example_of", "Supervised Learning",
                     "A neural network is an example of a model that is trained with SL"),
        ],
    },
}


class ScriptedLLM:
    """Answers by looking up a marker string in the passage; records every call."""

    model = "fake-model"

    def __init__(self, answers: dict[str, Any] | None = None,
                 responder: Callable[[Sequence[dict[str, Any]]], LLMResponse] | None = None) -> None:
        self.answers = ANSWERS if answers is None else answers
        self.responder = responder
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        self.calls.append(list(messages))
        if self.responder is not None:
            return self.responder(messages)
        passage = messages[0]["content"]
        for marker, answer in self.answers.items():
            if marker in passage:
                return LLMResponse(json.dumps(answer), stop_reason="end_turn", model=self.model)
        return LLMResponse(json.dumps({"concepts": [], "relationships": []}), stop_reason="end_turn", model=self.model)
