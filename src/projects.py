"""Projects: independent workspaces, each with its own documents, concepts, relationships and map.

Isolation rules used everywhere:
- every project-owned record carries ``project_id``;
- queries go through ``project_filter`` and rebuilds only replace records inside that filter;
- IDs that are derived from names (concepts, aliases, index terms, document hashes) are scoped
  with ``scoped_id`` / ``scoped_key`` / ``document_id_for``, so the same name or the same PDF in
  two projects never becomes the same record.

The default project keeps the IDs the app used before projects existed, so existing data and
references stay valid. ``migrate`` only adds ``project_id: "default"`` where it is missing.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from src.storage.mongodb_store import stable_id

DEFAULT_PROJECT = "default"
DEFAULT_NAME = "Default project"
COLLECTION = "projects"

# Collections whose records belong to exactly one project.
PROJECT_SCOPED = (
    "documents", "content_units", "assets", "image_state", "extraction_state", "concept_mentions",
    "concepts", "concept_aliases", "relationships", "page_terms", "search_index", "search_meta",
    "kg_concepts", "kg_relationships", "deleted_documents", "watched_files",
)


class ProjectError(ValueError):
    """Invalid project request (unknown project, duplicate name, protected project)."""


def is_default(project_id: str | None) -> bool:
    return project_id in (None, DEFAULT_PROJECT)


def project_filter(project_id: str | None) -> dict[str, Any]:
    """Query fragment selecting one project's records (default also matches records not yet migrated)."""
    if is_default(project_id):
        return {"project_id": {"$in": [DEFAULT_PROJECT, None]}}
    return {"project_id": project_id}


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


def migrate(database: Any) -> dict[str, int]:
    """Give every record without a project to the default project. Adds a field; changes nothing else."""
    updated = {}
    for name in PROJECT_SCOPED:
        result = database[name].update_many({"project_id": {"$exists": False}},
                                            {"$set": {"project_id": DEFAULT_PROJECT}})
        updated[name] = result.modified_count
    return updated


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name:
        raise ProjectError("Project name is empty")
    if len(name) > 80:
        raise ProjectError("Project name is longer than 80 characters")
    return name


class ProjectStore:
    """CRUD for project records. Deleting a project's data is done by ``ingest.documents.delete_project``."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self.projects = database[COLLECTION]
        self.ensure_default()

    def ensure_default(self) -> None:
        now = datetime.now(timezone.utc)
        self.projects.update_one({"_id": DEFAULT_PROJECT},
                                 {"$setOnInsert": {"name": DEFAULT_NAME, "created_at": now, "updated_at": now}},
                                 upsert=True)

    def list(self) -> list[dict[str, Any]]:
        projects = list(self.projects.find())
        return sorted(projects, key=lambda p: (p["_id"] != DEFAULT_PROJECT, p["name"].lower()))

    def get(self, project_id: str) -> dict[str, Any]:
        project = self.projects.find_one({"_id": project_id})
        if project is None:
            raise ProjectError(f"Unknown project '{project_id}'")
        return project

    def _check_unique(self, name: str, exclude: str | None = None) -> None:
        for project in self.projects.find({}, {"name": 1}):
            if project["_id"] != exclude and project["name"].lower() == name.lower():
                raise ProjectError(f"A project named '{name}' already exists")

    def create(self, name: str) -> dict[str, Any]:
        name = _clean_name(name)
        self._check_unique(name)
        now = datetime.now(timezone.utc)
        project = {"_id": uuid.uuid4().hex[:12], "name": name, "created_at": now, "updated_at": now}
        self.projects.insert_one(project)
        return project

    def rename(self, project_id: str, name: str) -> dict[str, Any]:
        self.get(project_id)
        name = _clean_name(name)
        self._check_unique(name, exclude=project_id)
        self.projects.update_one({"_id": project_id},
                                 {"$set": {"name": name, "updated_at": datetime.now(timezone.utc)}})
        return self.get(project_id)

    def remove_record(self, project_id: str) -> None:
        if is_default(project_id):
            raise ProjectError("The default project cannot be deleted")
        self.projects.delete_one({"_id": project_id})
