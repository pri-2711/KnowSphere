"""Backup and restore of one project (or all projects) as MongoDB Extended JSON files.

Backup writes one ``<collection>.jsonl`` per collection plus ``manifest.json`` (counts and SHA-256
of every file) into a new folder; it only reads the database. Restore verifies the checksums first
and refuses to write into any collection that already holds records of the restored project, so
it can never overwrite or mix data: restore into a new or emptied database (``MONGODB_DATABASE``),
compare, then switch. Uploaded files (``data/uploads``) and extracted images are files on disk and
are backed up by copying those folders; the manifest lists them.

For a full-server backup with indexes, ``mongodump``/``mongorestore`` remain the standard tools;
this module exists so a single project can be saved and restored without them and without
touching other projects.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import json_util

from src.knowledge_repository.projects import COLLECTION as PROJECTS, PROJECT_SCOPED, project_filter

FORMAT = "knowpal-backup-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def backup(database: Any, target: Path, project_id: str | None = None) -> dict[str, Any]:
    """Write a backup folder; ``project_id`` None backs up every project."""
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Backup folder is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    scopes = {name: (project_filter(project_id) if project_id else {}) for name in PROJECT_SCOPED}
    scopes[PROJECTS] = {"_id": project_id} if project_id else {}
    for name, query in sorted(scopes.items()):
        path = target / f"{name}.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for record in database[name].find(query):
                handle.write(json_util.dumps(record, json_options=json_util.CANONICAL_JSON_OPTIONS) + "\n")
                count += 1
        files[name] = {"file": path.name, "records": count, "sha256": _sha256(path)}
    uploads = sorted({r.get("file_storage", {}).get("uri") for r in database["documents"].find(
        scopes["documents"], {"file_storage.uri": 1}) if r.get("file_storage")})
    manifest = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
                "database": database.name, "project_id": project_id, "collections": files,
                "files_on_disk": uploads}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def restore(database: Any, source: Path) -> dict[str, int]:
    """Restore a backup folder into ``database``. Refuses on a checksum mismatch or any existing record."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Not a KnowPal backup: {source}")
    project_id = manifest.get("project_id")
    for name, entry in manifest["collections"].items():
        if _sha256(source / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch in {entry['file']}: the backup is damaged")
        query = ({"_id": project_id} if name == PROJECTS else project_filter(project_id)) if project_id else {}
        if entry["records"] and database[name].find_one(query, {"_id": 1}) is not None:
            raise FileExistsError(f"'{name}' already holds records of this project; restore into an empty "
                                  "database, it never overwrites")
    restored: dict[str, int] = {}
    for name, entry in sorted(manifest["collections"].items()):
        records = [json_util.loads(line) for line in (source / entry["file"]).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        if records:
            database[name].insert_many(records, ordered=True)
        restored[name] = len(records)
    return restored
