"""Types, controlled vocabularies and the JSON schema the LLM must follow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

CONCEPT_TYPES = ("concept", "method", "process", "principle", "entity", "artifact", "standard")
RELATION_TYPES = ("is_a", "includes", "uses", "defines", "prerequisite_for", "example_of", "related_to")
STATUSES = ("supported", "needs_review", "rejected")
Status = Literal["supported", "needs_review", "rejected"]

# Output contract for Pass 1. Item-level checks (vocabulary, evidence, endpoints) live in validation.py.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(CONCEPT_TYPES)},
                    "description": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                },
                "required": ["name", "type", "description", "aliases", "evidence"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "type": {"type": "string", "enum": list(RELATION_TYPES)},
                    "target": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["source", "type", "target", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["concepts", "relationships"],
    "additionalProperties": False,
}


class Evidence(TypedDict, total=False):
    """A verbatim quote and where it is: chunk offsets and the page it starts on."""
    chunk_id: str
    document_id: str
    content_unit_id: str      # the page unit the quote starts on
    page_number: int
    page_start: int           # offsets into that page's stored text
    page_end: int
    chunk_start: int
    chunk_end: int
    quote: str


class LocalConcept(TypedDict, total=False):
    name: str
    key: str                  # concept_key(name), the hash key used for matching
    type: str
    description: str
    aliases: list[str]        # aliases that appear in the chunk text
    unsupported_aliases: list[str]
    evidence: Evidence | None
    status: Status
    issues: list[str]


class LocalRelationship(TypedDict, total=False):
    source: str
    target: str
    source_key: str
    target_key: str
    type: str
    evidence: Evidence | None
    status: Status
    issues: list[str]


def worst(*statuses: Status) -> Status:
    """The least trusted of several statuses."""
    return max(statuses, key=STATUSES.index)


def best(*statuses: Status) -> Status:
    return min(statuses, key=STATUSES.index)
