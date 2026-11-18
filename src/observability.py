"""Logging setup shared by the API, the command lines and the MCP server.

``KM_LOG_FORMAT=json`` writes one JSON object per line (time, level, logger, message, and any
``extra`` fields such as project, job, agent, stage, ms), which log collectors can index. The
default is plain text. Logs go to stderr (the MCP server's stdout carries protocol messages) and,
when ``file`` is given, also to that file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        entry.update({key: value for key, value in vars(record).items() if key not in _STANDARD})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure(level: str | None = None, file: Path | None = None, json_format: bool | None = None,
              stderr: bool = True) -> None:
    """Replace the root handlers; safe to call more than once."""
    json_format = os.environ.get("KM_LOG_FORMAT", "text").lower() == "json" if json_format is None else json_format
    formatter: logging.Formatter = JsonFormatter() if json_format else \
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)] if stderr else []
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level or os.environ.get("KM_LOG_LEVEL", "INFO"))
