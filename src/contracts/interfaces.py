"""Interfaces between components. Components depend on these protocols, not on each other's code.

- ``TextModel``, ``VisionModel``, ``SpanModel``, ``Embedder``: implemented in ``model_runtime``
  (Ollama/Qwen, Claude, GLiNER, fastembed) and handed out by ``model_runtime.router.ModelRouter``.
- ``DocumentParser``: implemented in ``document_intelligence.parsers`` (PyMuPDF, Docling).
- ``KnowledgeRepository``: implemented in ``knowledge_repository.repository`` (MongoDB).
- ``Retriever``: implemented in ``retrieval.engine.KnowledgeService``.

Every repository and retrieval call takes a ``project_id``: there is no cross-project operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from src.contracts.schemas import (Adjudication, Claim, Concept, ContentUnit, Document, DocumentVersion, KnowledgeStatus,
                                   ParsedDocument, Relationship, ValidationRecord)


# ------------------------------------------------------------------ model runtime
@dataclass
class LLMResponse:
    text: str
    stop_reason: str | None = None        # "end_turn", "max_tokens" or "refusal" (providers map theirs to these)
    model: str | None = None
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """A failed call for one unit of work. ``retryable`` is False when the same request would fail again."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMFatalError(LLMError):
    """A failure that affects every call (bad credentials, unknown model, server not running): stop the run."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


@runtime_checkable
class TextModel(Protocol):
    """Structured text generation: return JSON that follows ``schema``."""
    model: str

    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        ...


@runtime_checkable
class VisionModel(Protocol):
    """Structured image understanding: describe one image as JSON that follows ``schema``."""
    model: str

    def describe(self, image_path: Path, system: str, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        ...


@runtime_checkable
class SpanModel(Protocol):
    """Entity span detection (GLiNER): one list of {label, score, start, end} per input text."""
    model_name: str

    def extract_many(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Any:
        """Return an (n, d) array of L2-normalized vectors."""


@runtime_checkable
class Captioner(Protocol):
    model_name: str

    def caption(self, image_path: Path) -> str:
        ...


# ------------------------------------------------------------------ document intelligence
@runtime_checkable
class DocumentParser(Protocol):
    """Finds structure (sections, tables, figures, equations) in a PDF, tied to pages and locations."""
    name: str

    def available(self) -> bool:
        ...

    def parse(self, pdf_path: Path, project_id: str, document_id: str,
              document_version_id: str | None = None) -> ParsedDocument:
        ...


# ------------------------------------------------------------------ repository and retrieval
@dataclass
class KnowledgeSet:
    """One project's knowledge from one graph source, as canonical models.

    ``invalid`` lists stored records that could not be read as canonical models:
    (item id, item type, error). Governance rejects them on the schema check.
    """
    project_id: str
    source: str                                   # "gliner" or "concept_first"
    concepts: list[Concept] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    invalid: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class UnitRecord:
    document_id: str
    page_number: int | None
    text: str


class KnowledgeRepository(Protocol):
    """Canonical persistence. Every call is scoped to one project; writes are idempotent upserts by ID."""

    def documents(self, project_id: str) -> list[Document]: ...
    def document_versions(self, project_id: str, lineage_id: str | None = None) -> list[DocumentVersion]: ...
    def record_version(self, project_id: str, document_id: str, sha256: str, filename: str,
                       previous_document_id: str | None = None, parser: str | None = None) -> DocumentVersion: ...
    def content_units(self, project_id: str, unit_types: Iterable[str] | None = None,
                      document_id: str | None = None) -> list[ContentUnit]: ...
    def save_structure(self, parsed: ParsedDocument) -> dict[str, int]: ...
    def knowledge(self, project_id: str, source: str) -> KnowledgeSet: ...
    def units(self, project_id: str, unit_ids: Iterable[str]) -> dict[str, UnitRecord]: ...
    def page_counts(self, project_id: str) -> dict[str, int]: ...
    def adjudications(self, project_id: str) -> dict[str, Adjudication]: ...
    def adjudicate(self, project_id: str, item_id: str, adjudication: Adjudication) -> None: ...
    def save_validation(self, project_id: str, source: str, records: list[ValidationRecord],
                        statuses: dict[str, KnowledgeStatus]) -> dict[str, int]: ...


class Retriever(Protocol):
    """One project's lexical, graph and extractive question answering."""
    project_id: str

    def search(self, query: str, limit: int = 10, document_id: str | None = None) -> dict[str, Any]: ...
    def ask(self, question: str) -> dict[str, Any]: ...
    def concept_detail(self, concept_id: str) -> dict[str, Any] | None: ...
    def graph_payload(self, document_id: str | None = None, concept_type: str | None = None,
                      min_weight: float = 0.0, limit: int = 250, focus: str | None = None) -> dict[str, Any]: ...
