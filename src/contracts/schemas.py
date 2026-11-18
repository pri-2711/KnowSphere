"""Canonical knowledge model, version ``SCHEMA_VERSION``.

These Pydantic models are the shared contract between components. They describe
knowledge, not storage or models: there is no MongoDB, LLM or parser specific
field here. Storage adapters (``knowledge_repository.canonical``) map stored
records to and from these models; extractors return them; governance checks them.

Rules every record follows:
- it belongs to exactly one project (``project_id``); nothing links across projects;
- knowledge items (concepts, claims, relationships) carry ``evidence``: each
  ``Evidence`` points at a ``SourceRef`` (document, version, content unit, page,
  character offsets, optional bounding box) and records how it was produced
  (``Provenance``);
- acceptance is a ``KnowledgeStatus`` decided by governance checks, never by a
  model's confidence score.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "knowpal-1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    """Base for every contract: unknown fields are rejected, every record states its schema version."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: str = SCHEMA_VERSION


# ------------------------------------------------------------------ statuses
class KnowledgeStatus(str, Enum):
    """Lifecycle of a knowledge item. Only governance changes it; ``adjudicated`` only a person sets."""
    CANDIDATE = "candidate"          # proposed by an extractor, not checked yet
    NEEDS_REVIEW = "needs_review"    # checked: usable but doubtful (weak evidence, contested)
    VERIFIED = "verified"            # every blocking check passed, evidence traced to the source text
    CORROBORATED = "corroborated"    # verified and supported by independent sources (two or more documents)
    ADJUDICATED = "adjudicated"      # a reviewer decided (accepted or rejected, see ``Adjudication``)
    REJECTED = "rejected"            # a blocking check failed; kept for audit, never shown as knowledge

    @property
    def accepted(self) -> bool:
        return self in (KnowledgeStatus.VERIFIED, KnowledgeStatus.CORROBORATED)


class UnitType(str, Enum):
    PAGE = "page"
    CHUNK = "chunk"          # semantic segment spanning one or more pages
    SECTION = "section"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    EQUATION = "equation"


STRUCTURE_TYPES = (UnitType.SECTION, UnitType.TABLE, UnitType.FIGURE, UnitType.EQUATION)


# ------------------------------------------------------------------ provenance and evidence
class Provenance(Contract):
    """How a record was produced: which component, method and (when used) which model and prompt."""
    component: str                       # e.g. "document_intelligence", "knowledge_extraction"
    method: str                          # e.g. "pymupdf", "tesseract", "docling", "gliner", "llm", "vision", "cooccurrence"
    model: str | None = None             # model identifier when a model was involved
    prompt_version: str | None = None
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class SourceRef(Contract):
    """An exact place in a source document. Offsets index the stored text of ``content_unit_id``."""
    project_id: str
    document_id: str
    document_version_id: str | None = None
    content_unit_id: str
    unit_type: UnitType = UnitType.PAGE
    page_number: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    bbox: list[float] | None = None       # [x0, y0, x1, y1] in PDF points, for figures and tables

    @model_validator(mode="after")
    def _offsets(self) -> "SourceRef":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be given together")
        if self.char_start is not None and self.char_end is not None and self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must be [x0, y0, x1, y1]")
        return self


class Evidence(Contract):
    """A verbatim quote at a source location, and how it was found."""
    source: SourceRef
    quote: str = Field(min_length=1)
    provenance: Provenance
    traced: bool | None = None            # set by governance: the quote was found at ``source``


# ------------------------------------------------------------------ documents and content
class DocumentVersion(Contract):
    """One version of a document's file. A replaced file becomes a new version of the same lineage."""
    id: str
    project_id: str
    document_id: str
    lineage_id: str                       # stable across versions (the first version's document ID)
    version: int = Field(ge=1)
    sha256: str
    filename: str
    parser: str | None = None
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Document(Contract):
    id: str
    project_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=0)
    title: str | None = None
    author: str | None = None
    year: str | None = None
    current_version_id: str | None = None
    sources: list[str] = Field(default_factory=list)


class ContentUnit(Contract):
    """A unit of document content: a page, a semantic chunk, a section, table, figure, image or equation."""
    id: str
    project_id: str
    document_id: str
    document_version_id: str | None = None
    unit_type: UnitType
    page_number: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: list[str] = Field(default_factory=list)   # heading hierarchy, outermost first
    text: str = ""
    bbox: list[float] | None = None
    parent_id: str | None = None          # e.g. the page a table sits on
    provenance: Provenance
    quality_status: str = "usable"


# ------------------------------------------------------------------ knowledge items
class Adjudication(Contract):
    reviewer: str
    decision: Literal["accept", "reject"]
    note: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class KnowledgeItem(Contract):
    id: str
    project_id: str
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)   # informational only, never decides acceptance
    uncertainty: list[str] = Field(default_factory=list)   # reasons for doubt the source states; not a score
    adjudication: Adjudication | None = None

    @model_validator(mode="after")
    def _one_project(self) -> "KnowledgeItem":
        for item in self.evidence:
            if item.source.project_id != self.project_id:
                raise ValueError("evidence must come from the item's own project")
        return self

    @property
    def document_ids(self) -> set[str]:
        return {item.source.document_id for item in self.evidence}


class Concept(KnowledgeItem):
    """A concept or named entity (``kind``) with its aliases."""
    kind: Literal["concept", "entity"] = "concept"
    name: str = Field(min_length=1)
    normalized_name: str
    concept_type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    possible_duplicates: list[str] = Field(default_factory=list)   # concept ids resolution left for review


class Quantity(Contract):
    value: float
    unit: str | None = None
    raw: str                              # the text the number was read from, e.g. "4.5 kg/day"


class Claim(KnowledgeItem):
    """An assertion made by a source: subject, predicate, object, the full statement and its quantities."""
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = ""
    statement: str = Field(min_length=1)
    polarity: Literal["affirmed", "negated"] = "affirmed"
    kind: str = "statement"      # requirement, specification, measurement, definition, cause, comparison,
                                 # method, limitation, result, statement
    modality: Literal["fact", "interpretation", "hypothesis"] = "fact"
    quantities: list[Quantity] = Field(default_factory=list)
    subject_concept_id: str | None = None
    object_concept_id: str | None = None


class Relationship(KnowledgeItem):
    source_id: str
    target_id: str
    relation_type: str
    weight: float | None = None

    @field_validator("target_id")
    @classmethod
    def _no_self_loop(cls, target_id: str, info: Any) -> str:
        if info.data.get("source_id") == target_id:
            raise ValueError("a relationship needs two different endpoints")
        return target_id


# ------------------------------------------------------------------ extraction and validation records
class ExtractionResult(Contract):
    """What one extractor proposes for one content unit. Everything in it is a candidate."""
    project_id: str
    content_unit_id: str
    extractor: Provenance
    concepts: list[Concept] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    status: Literal["extracted", "failed"] = "extracted"
    error: str | None = None
    attempts: int = 1


class CheckResult(Contract):
    check: str
    passed: bool
    blocking: bool = True                 # a failed blocking check rejects the item
    message: str = ""


class ValidationRecord(Contract):
    """The outcome of governance for one item: every check that ran and the status it led to."""
    id: str
    project_id: str
    item_id: str
    item_type: Literal["concept", "claim", "relationship"]
    checks: list[CheckResult]
    status_before: KnowledgeStatus
    status_after: KnowledgeStatus
    policy_version: str
    created_at: datetime = Field(default_factory=utcnow)


class ParsedDocument(Contract):
    """Parser output: structure units found in a document, each tied to its page."""
    project_id: str
    document_id: str
    parser: Provenance
    page_count: int = Field(ge=0)
    units: list[ContentUnit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ decisions and contributions
class HistoryEntry(Contract):
    """One append-only record of a decision or contribution: who did what to which item, and what changed.

    Kept forever (also when a decision is later revoked or reverted), so every state of the
    knowledge base can be explained. ``actor`` is "reviewer:<name>", "agent:<name>" or "system".
    """
    id: str
    project_id: str
    actor: str
    action: str                      # adjudicate, revoke_adjudication, merge, keep_apart, revert_merge,
                                     # propose, ontology_extend
    item_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)


class Proposal(Contract):
    """Knowledge an agent or person proposes. It is a candidate like any extraction: it enters the graph
    through Pass 2 and governance, never directly."""
    id: str
    project_id: str
    agent: str
    kind: Literal["concept", "relationship", "claim"]
    unit_id: str                     # the content unit (page or chunk) the evidence quote comes from
    data: dict[str, Any]             # same fields an extractor returns for this kind, including "evidence"
    status: Literal["submitted", "rejected", "withdrawn"] = "submitted"
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
