"""Types, controlled vocabularies and the JSON schema the LLM must follow."""

from __future__ import annotations

from typing import Any

# The vocabulary is shared (contracts); these names stay importable from here.
from src.contracts.vocabulary import (CONCEPT_TYPES, RELATION_TYPES, STATUSES, Evidence,  # noqa: F401
                                      LocalConcept, LocalRelationship, Status, best, worst)

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
