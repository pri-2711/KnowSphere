"""Watch a folder (default ``data/inbox/``) and send added, changed or removed PDFs through the pipeline.

Polling is used instead of OS file events: it works the same on Windows, Linux
and macOS and needs no extra dependency. Each scan compares the folder with the
``watched_files`` collection (path, size, modification time, SHA-256, document ID):

- a new PDF is ingested and every knowledge stage runs;
- a PDF whose content changed is ingested as a new version; the old version
  loses this source and is deleted when no other source remains, and the
  user's metadata edits carry over;
- a removed PDF removes its source; the document is deleted when no other
  source (an upload, data/raw_pdfs, another watched path) still provides it.

A file is only processed after its size and modification time stayed the same
for one full interval, so a file that is still being copied is not read.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.ingest.documents import is_tombstoned
from src.storage.mongodb_store import MongoKnowledgeStore, sha256_file

logger = logging.getLogger("knowledge_map.watch")
Progress = Callable[[str, float], None]
STATE = "watched_files"


@dataclass
class WatchEvent:
    kind: str                       # "added", "updated" or "deleted"
    relative_path: str
    path: Path | None = None
    sha256: str | None = None
    old_document_id: str | None = None
    size: int | None = None
    mtime_ns: int | None = None

    @property
    def source(self) -> str:
        return f"watch:{self.relative_path}"


class FolderWatcher:
    """Finds changes in the watched folder; ``process_events`` applies them."""

    def __init__(self, store: MongoKnowledgeStore, folder: Path) -> None:
        self.store = store
        self.folder = folder
        self.state = store.database[STATE]
        self._seen: dict[str, tuple[int, int]] = {}   # last observed (size, mtime) of changed files

    def _current_files(self) -> dict[str, tuple[Path, int, int]]:
        files = {}
        if not self.folder.is_dir():
            return files
        for path in self.folder.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                stat = path.stat()
                files[path.relative_to(self.folder).as_posix()] = (path, stat.st_size, stat.st_mtime_ns)
        return files

    def scan(self) -> list[WatchEvent]:
        """Compare the folder with the stored state and return stable changes."""
        self.folder.mkdir(parents=True, exist_ok=True)
        current = self._current_files()
        known = {record["_id"]: record for record in self.state.find()}
        events: list[WatchEvent] = []
        for relative, (path, size, mtime) in sorted(current.items()):
            record = known.get(relative)
            if record and record.get("size") == size and record.get("mtime_ns") == mtime:
                self._seen.pop(relative, None)
                continue
            if self._seen.get(relative) != (size, mtime):
                self._seen[relative] = (size, mtime)       # wait one more scan: the copy may be in progress
                continue
            digest = sha256_file(path)
            if record and record.get("sha256") == digest:   # touched but not changed
                self.state.update_one({"_id": relative}, {"$set": {"size": size, "mtime_ns": mtime}})
                self._seen.pop(relative, None)
                continue
            events.append(WatchEvent("updated" if record else "added", relative, path, digest,
                                     record.get("document_id") if record else None, size, mtime))
        for relative, record in sorted(known.items()):
            if relative not in current:
                events.append(WatchEvent("deleted", relative, old_document_id=record.get("document_id")))
        return events

    def mark_done(self, event: WatchEvent) -> None:
        self._seen.pop(event.relative_path, None)
        if event.kind == "deleted":
            self.state.delete_one({"_id": event.relative_path})
            return
        self.state.update_one({"_id": event.relative_path}, {"$set": {
            "sha256": event.sha256, "document_id": event.sha256, "size": event.size, "mtime_ns": event.mtime_ns,
            "processed_at": datetime.now(timezone.utc)}}, upsert=True)


def process_events(pipeline: Any, watcher: FolderWatcher, events: list[WatchEvent],
                   progress: Progress = lambda stage, fraction: None) -> list[dict[str, Any]]:
    """Apply watch events through the pipeline, then run the knowledge stages once."""
    results = []
    for event in events:
        name = Path(event.relative_path).name
        logger.info("watch %s: %s", event.kind, event.relative_path)
        result: dict[str, Any] = {"event": event.kind, "path": event.relative_path}
        if event.kind in ("added", "updated") and is_tombstoned(pipeline.store, event.sha256):
            result["skipped"] = "this file version was deleted in the app"
        elif event.kind == "added":
            result.update(pipeline.ingest_file(event.path, name, progress, source_label=event.source))
        elif event.kind == "updated":
            result.update(pipeline.ingest_file(event.path, name, progress, source_label=event.source))
            if event.old_document_id and event.old_document_id != event.sha256:
                pipeline.metadata.carry_over_edits(event.old_document_id, event.sha256)
                result["old_version"] = pipeline.remove_source(event.old_document_id, event.source, progress)
        elif event.kind == "deleted" and event.old_document_id:
            result.update(pipeline.remove_source(event.old_document_id, event.source, progress))
        watcher.mark_done(event)
        results.append(result)
    if events:
        pipeline.knowledge_stages(progress)
    return results


class WatchService:
    """Background thread for the API: scans every ``interval`` seconds and queues one job per batch."""

    def __init__(self, watcher: FolderWatcher, submit: Callable[[list[WatchEvent]], Any], interval: float) -> None:
        self.watcher = watcher
        self.submit = submit
        self.interval = interval
        self.busy = threading.Event()        # set while a submitted batch is queued or running
        self.last_scan: float | None = None
        self.last_events: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="km-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if self.busy.is_set():
                continue
            try:
                events = self.watcher.scan()
                self.last_scan, self.last_error = time.time(), None
                if events:
                    self.last_events = [{"event": event.kind, "path": event.relative_path} for event in events]
                    self.busy.set()
                    self.submit(events)
            except Exception as exc:  # keep watching after a failure
                logger.exception("watch scan failed")
                self.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {"enabled": self._thread is not None and not self._stop.is_set(), "folder": str(self.watcher.folder),
                "interval_seconds": self.interval, "last_scan": self.last_scan, "busy": self.busy.is_set(),
                "last_events": self.last_events, "last_error": self.last_error,
                "files": self.watcher.state.count_documents({})}
