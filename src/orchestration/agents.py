"""Agent access: who may read or propose in which project, and an audit record of every call.

Agents (Claude Code, Codex and other MCP clients) never reach the database: every call goes
through ``AgentGateway``, which checks the agent's grant, runs the shared application service, and
records the call in ``agent_activity`` (project, agent, tool, arguments, outcome, duration).

Policy file (``KM_AGENT_POLICY``, default ``config/agents.json``)::

    {"default": {"projects": ["default"], "scopes": ["read"]},
     "agents": {"claude-code": {"projects": ["*"], "scopes": ["read", "propose"]},
                "codex": {"projects": ["default"], "scopes": ["read"]}}}

Scopes: ``read`` (search, documents, knowledge, evidence, graph) and ``propose`` (submit
knowledge proposals, which governance then decides). Without a policy file every agent may read
the default project and nothing else; no agent may propose.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.config import PROJECT_ROOT

SCOPES = ("read", "propose")
POLICY_FILE = Path(os.environ.get("KM_AGENT_POLICY", PROJECT_ROOT / "config" / "agents.json"))
_FALLBACK = {"projects": ["default"], "scopes": ["read"]}


class PermissionDenied(PermissionError):
    """The agent's grant does not cover this project or scope."""


@dataclass(frozen=True)
class Grant:
    projects: frozenset[str]
    scopes: frozenset[str]

    def allows(self, project_id: str, scope: str) -> bool:
        return scope in self.scopes and ("*" in self.projects or project_id in self.projects)


def _grant(entry: dict[str, Any]) -> Grant:
    scopes = set(entry.get("scopes", []))
    unknown = scopes - set(SCOPES)
    if unknown:
        raise ValueError(f"unknown scope(s) in agent policy: {', '.join(sorted(unknown))}")
    return Grant(frozenset(entry.get("projects", [])), frozenset(scopes))


class AgentPolicy:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.default = _grant(data.get("default", _FALLBACK))
        self.agents = {name: _grant(entry) for name, entry in data.get("agents", {}).items()}

    @classmethod
    def load(cls, path: Path = POLICY_FILE) -> "AgentPolicy":
        return cls(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)

    def grant(self, agent: str) -> Grant:
        return self.agents.get(agent, self.default)


class AgentGateway:
    """One agent's access to the application services, checked and audited."""

    def __init__(self, app: Any, policy: AgentPolicy, agent: str) -> None:
        if not agent or not agent.replace("-", "").replace("_", "").isalnum():
            raise ValueError("an agent name is required (letters, digits, '-' and '_')")
        self.app = app
        self.policy = policy
        self.agent = agent

    @property
    def grant(self) -> Grant:
        return self.policy.grant(self.agent)

    def projects(self) -> list[dict[str, Any]]:
        return [p for p in self.app.list_projects() if self.grant.allows(p["id"], "read")]

    def call(self, tool: str, project_id: str, scope: str, action: Callable[[], Any],
             arguments: dict[str, Any] | None = None) -> Any:
        started = time.perf_counter()
        outcome, error = "ok", None
        try:
            if not self.grant.allows(project_id, scope):
                raise PermissionDenied(f"agent '{self.agent}' may not {scope} in project '{project_id}'")
            return action()
        except Exception as exc:
            outcome, error = ("denied" if isinstance(exc, PermissionDenied) else "error"), f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.app.repository.log_activity({
                "project_id": project_id, "agent": self.agent, "tool": tool, "scope": scope,
                "arguments": {k: (v if len(str(v)) <= 300 else str(v)[:300] + "...") for k, v in (arguments or {}).items()},
                "outcome": outcome, "error": error[:300] if error else None,
                "ms": round((time.perf_counter() - started) * 1000, 1)})
