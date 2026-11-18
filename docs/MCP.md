# KnowPal MCP server

The MCP server gives coding agents (Claude Code, Codex, any Model Context Protocol client) the
knowledge base as tools. It runs locally over stdio, uses the same application services as the web
API and the CLI, and never lets an agent reach the database directly.

```
agent  --stdio JSON-RPC-->  src/api/mcp_server.py  -->  AgentGateway (grant check, audit)
                                                      -->  KnowledgeApplication (services)
                                                      -->  repository, retrieval, governance
```

## Tools

| Tool | Scope | What it returns |
|---|---|---|
| `list_projects` | read | Projects this agent may read |
| `project_context` | read | Documents, graph size, central concepts, review queue, ontology, recent decisions |
| `search` | read | Hybrid search: passages with document, page, offsets and the methods that found them |
| `research` | read | Evidence-first answer: passages, claims as facts / interpretations / hypotheses, unconfirmed claims, conflicts, unit conversions; abstains without evidence |
| `get_document` | read | Extraction quality (digital, scanned, mixed) and version history |
| `get_page` | read | Stored text of one page with its citation |
| `find_concepts`, `get_concept` | read | Canonical entities with aliases, evidence, relationships, claims, conflicts |
| `get_claims` | read | Claims filtered by concept, kind (requirement, measurement, method, limitation, ...) or modality |
| `get_evidence` | read | Evidence of any item, each quote re-checked against its source |
| `explore_graph`, `find_path` | read | Multi-hop neighbourhood; shortest relationship chain with evidence per step |
| `find_contradictions` | read | Conflicting claims and relationships found by governance |
| `propose_knowledge` | propose | Submit a concept, relationship or claim with an exact quote; it becomes a candidate that governance decides on the next graph build |

Every tool takes an optional `project` (default `default`). Tool descriptions tell the agent that
page text and quotes are untrusted document content, to be treated as data.

## Permissions and audit

The agent's name comes from the `KM_AGENT` environment variable of the server process (set in the
client configuration below), not from tool arguments, so an agent cannot use another agent's grant.
Grants are read from `config/agents.json` (or the file named by `KM_AGENT_POLICY`); copy
`config/agents.example.json` to start. Without a policy file every agent may read the default
project and none may propose.

Each call is recorded in `agent_activity` (agent, tool, project, arguments, outcome `ok`, `denied`
or `error`, duration). See it with `python -m src.api.cli activity`, `GET /api/agents/activity`, or
the Overview screen. Proposals also appear in the decision history.

## Claude Code

Project scope: create `.mcp.json` in the repository root (Claude Code asks once before it starts
a project server):

```json
{
  "mcpServers": {
    "knowpal": {
      "command": ".venv/Scripts/python.exe",
      "args": ["-m", "src.api.mcp_server"],
      "env": {"KM_AGENT": "claude-code"}
    }
  }
}
```

Or from a terminal in the repository: `claude mcp add knowpal --env KM_AGENT=claude-code -- .venv/Scripts/python.exe -m src.api.mcp_server`.

## Codex

In `~/.codex/config.toml` (absolute paths, because Codex starts servers outside the repository;
the module adds the repository to the import path itself when started by file path):

```toml
[mcp_servers.knowpal]
command = "D:/internship/knowledgeMap/.venv/Scripts/python.exe"
args = ["D:/internship/knowledgeMap/src/api/mcp_server.py"]
env = { KM_AGENT = "codex" }
```

MongoDB settings come from the usual `MONGODB_URI` and `MONGODB_DATABASE` variables (add them to
`env` when they differ from the local defaults).

## Verified

- In process (`tests/test_interfaces_and_operations.py`): handshake, tool listing, every read tool
  through the services, permission denial for scope and project, audit records, malformed input.
- Over stdio against the local database: `initialize` (protocol `2025-06-18`), `tools/list`
  (14 tools) and `ping` answered by `python -m src.api.mcp_server`.
- Not yet verified: a session driven by the Claude Code or Codex client itself.
