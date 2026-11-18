"""Project locks: one writing operation per project at a time, across threads and processes.

A lock is a record in ``project_locks`` keyed by the project ID (MongoDB guarantees one record per
key, so two processes cannot both insert it). It names the operation and owner and expires after
``ttl`` seconds, so a crashed process cannot block a project for ever. The same thread may take a
lock it already holds (an ingest that then rebuilds), other threads and processes get
``ProjectBusy``.
"""

from __future__ import annotations

import os
import socket
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from pymongo.errors import DuplicateKeyError

COLLECTION = "project_locks"


class ProjectBusy(RuntimeError):
    """Another operation is writing to this project."""


class ProjectLocks:
    def __init__(self, database: Any, ttl_seconds: int = 6 * 3600) -> None:
        self.collection = database[COLLECTION]
        self.ttl = timedelta(seconds=ttl_seconds)
        self._held = threading.local()
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def _mine(self) -> dict[str, str]:
        if not hasattr(self._held, "tokens"):
            self._held.tokens = {}
        return self._held.tokens

    @contextmanager
    def hold(self, project_id: str, operation: str) -> Iterator[None]:
        mine = self._mine()
        if project_id in mine:                      # re-entrant within one thread
            yield
            return
        token, now = uuid.uuid4().hex, datetime.now(timezone.utc)
        record = {"_id": project_id, "token": token, "operation": operation, "owner": self.owner,
                  "acquired_at": now, "expires_at": now + self.ttl}
        try:
            self.collection.insert_one(record)
        except DuplicateKeyError:
            taken = self.collection.find_one_and_replace(
                {"_id": project_id, "expires_at": {"$lt": now}}, record)   # only an expired lock is taken over
            if taken is None:
                current = self.collection.find_one({"_id": project_id}) or {}
                raise ProjectBusy(f"Project '{project_id}' is busy: {current.get('operation', 'another operation')} "
                                  f"by {current.get('owner', '?')} since {current.get('acquired_at')}") from None
        mine[project_id] = token
        try:
            yield
        finally:
            mine.pop(project_id, None)
            self.collection.delete_one({"_id": project_id, "token": token})

    def current(self, project_id: str) -> dict[str, Any] | None:
        record = self.collection.find_one({"_id": project_id})
        return {k: v for k, v in record.items() if k != "token"} if record else None
