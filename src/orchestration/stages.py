"""Stage tracking for recovery: what ran, for which project (and document), and how it ended.

Every stage is idempotent and resumes at its own granularity (GLiNER per page,
Pass 1 and claims per chunk, visual extraction per image, images per document),
so a failed run is resumed by running it again: finished work is skipped. The
``stage_runs`` collection makes that state visible (``pipeline status``) and tells
``pipeline resume`` which projects need a rerun.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.contracts.ids import stable_id
from src.knowledge_repository.projects import project_filter

STAGE_RUNS = "stage_runs"
RUNNING, DONE, FAILED = "running", "done", "failed"


class StageTracker:
    def __init__(self, database: Any) -> None:
        self.collection = database[STAGE_RUNS]

    def run(self, project_id: str, stage: str, action: Callable[[], Any], document_id: str | None = None) -> Any:
        """Run one stage and record its outcome; the exception is re-raised after it is recorded."""
        key = stable_id("stage_run", project_id, stage, document_id)
        started = time.perf_counter()
        self.collection.update_one({"_id": key}, {"$set": {
            "project_id": project_id, "stage": stage, "document_id": document_id, "status": RUNNING,
            "started_at": datetime.now(timezone.utc), "error": None}}, upsert=True)
        try:
            result = action()
        except Exception as exc:
            self.collection.update_one({"_id": key}, {"$set": {
                "status": FAILED, "error": f"{type(exc).__name__}: {exc}"[:500],
                "seconds": round(time.perf_counter() - started, 1), "finished_at": datetime.now(timezone.utc)}})
            raise
        self.collection.update_one({"_id": key}, {"$set": {
            "status": DONE, "seconds": round(time.perf_counter() - started, 1),
            "finished_at": datetime.now(timezone.utc)}})
        return result

    def status(self, project_id: str) -> list[dict[str, Any]]:
        return [{key: value for key, value in record.items() if key != "_id"}
                for record in self.collection.find(project_filter(project_id)).sort("started_at", 1)]

    def unfinished(self, project_id: str) -> list[dict[str, Any]]:
        """Stages whose last run failed or never finished (for example the process was killed)."""
        return [record for record in self.status(project_id) if record["status"] != DONE]
