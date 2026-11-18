"""Deterministic identifiers shared by every component (same input, same ID, on every run)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_PROJECT = "default"


def stable_id(entity_type: str, *parts: Any) -> str:
    """Create a deterministic, storage-safe ID from an entity type and key."""
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{entity_type}:{digest}"


# Project-scoped IDs: the same name or file in two projects never becomes the same record.
def is_default(project_id: str | None) -> bool:
    return project_id in (None, DEFAULT_PROJECT)


def scoped_id(entity: str, project_id: str | None, *parts: Any) -> str:
    """stable_id that differs between projects (unchanged for the default project)."""
    return stable_id(entity, *parts) if is_default(project_id) else stable_id(entity, "project", project_id, *parts)


def scoped_key(project_id: str | None, key: str) -> str:
    """Record key for a word-like key (alias, index term) inside one project."""
    return key if is_default(project_id) else f"{project_id}|{key}"


def document_id_for(project_id: str | None, sha256: str) -> str:
    """Document ID: the PDF hash in the default project, a project-specific hash elsewhere."""
    if is_default(project_id):
        return sha256
    return hashlib.sha256(f"project:{project_id}:{sha256}".encode("utf-8")).hexdigest()
