"""Shared vocabulary for candidate extraction: concept and relation types, local statuses, and the
shape of local (per-chunk) candidates. Independent of any model, parser or database.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, TypedDict

CONCEPT_TYPES = ("concept", "method", "process", "principle", "entity", "artifact", "standard")
RELATION_TYPES = ("is_a", "includes", "uses", "defines", "prerequisite_for", "example_of", "related_to")
STATUSES = ("supported", "needs_review", "rejected")
Status = Literal["supported", "needs_review", "rejected"]

# Domain of the documents (space technology, aerospace and systems engineering), told to the model.
DOMAIN_SUBJECTS = (
    "space technology and space exploration",
    "aerospace engineering",
    "systems engineering",
    "spacecraft and satellite systems",
    "NASA technical documentation",
    "systems architecture, requirements engineering, verification and validation, integration and testing",
)

# Known expansions in this domain: abbreviation -> (full form, needs_context). With needs_context the
# abbreviation has other common meanings ("SE" is also "software engineering"), so the expansion is only
# accepted when the passage itself states it. Without it the expansion is standard in this domain.
DOMAIN_ABBREVIATIONS: dict[str, tuple[str, bool]] = {
    "NASA": ("National Aeronautics and Space Administration", False),
    "SE": ("Systems Engineering", True),
    "MBSE": ("Model-Based Systems Engineering", False),
    "V&V": ("Verification and Validation", False),
    "ICD": ("Interface Control Document", False),
}


def looks_like_abbreviation(name: str) -> bool:
    """'NASA', 'V&V', 'ECLSS', 'CO2', 'SysML': short, no spaces, at least two capitals."""
    compact = name.strip()
    return (2 <= len(compact) <= 10 and " " not in compact and sum(char.isupper() for char in compact) >= 2
            and all(char.isalnum() or char in "&/-." for char in compact))


def domain_expansion(abbreviation: str) -> tuple[str, bool] | None:
    return DOMAIN_ABBREVIATIONS.get(abbreviation.strip())


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


def chunk_offset_to_page(chunk: Mapping[str, Any], start: int, end: int) -> dict[str, Any] | None:
    """Map a chunk slice to the page it starts on (unit, page number, page offsets)."""
    for span in chunk["spans"]:
        if span["chunk_start"] <= start < span["chunk_end"]:
            offset = span["start"] + (start - span["chunk_start"])
            page_end = span["start"] + (min(end, span["chunk_end"]) - span["chunk_start"])
            return {"content_unit_id": span["content_unit_id"], "page_number": span["page_number"],
                    "page_start": offset, "page_end": page_end}
    return None
