# Implementation status

Living checklist of KnowPal against its specification. Updated with every milestone.
Last update: 2026-11-17, 170 tests passing (`python -m pytest`).

**Specification used.** No single "KnowPal requirements document" exists in the repository. The
specification is the finalisation request (November 2026) together with the documents in the
repository root: `KnowledgeMap_doc.docx` (planning document: pipeline stages, outputs, stress-test
targets, phases), `Knowledge_Map_1pager.docx` (concept note: two-pass discovery, evidence-first
retrieval, evaluation criteria) and `Knowledge_Map_Project_Concept.docx` (image understanding,
abstention). Quality targets quoted below come from the planning document's stress-test matrix.

Legend: **Done** (implemented and exercised by a test), **Partial**, **Missing**, **Blocked**
(needs a download, an install or an approval), **Decision** (evaluated; outcome recorded).
"Measured" numbers come from runs on the real database, read-only, with no model inference.

## Current status

### 1. Knowledge engine

| Item | Status | Where | Evidence |
|---|---|---|---|
| Universal schemas: concept/entity, claim, relationship, evidence, provenance | Done | `contracts/schemas.py` | `test_contracts_runtime.py` |
| Canonical resolution, abbreviations, synonyms (general rules + model judgement) | Done | `knowledge_extraction/resolution.py`, `enrichment.py` | `test_canonical_resolution.py` |
| Alias management | Done | Pass 2 aliases, reviewer merges | `test_knowledge_engine.py` |
| Extensible ontology with domain extensions | Partial | `contracts/ontology.py`, repository, `/api/ontology` | extensions validated and stored; extraction prompts still use the core types, extensions are used by proposals (mapped to their core type) |
| Evidence-backed claims with kind (requirement, measurement, method, limitation, result, ...) | Done | `knowledge_extraction/claims.py` (`claims-2`) | `test_knowledge_engine.py` |
| Fact / interpretation / hypothesis | Done | claim `modality`; the source's own hedging always wins over the model's label | `test_claim_modality_follows_the_sources_own_hedging` |
| Confidence and uncertainty as separate properties | Done | `confidence` (number, never decides) and `uncertainty` (reasons the source states) | same |
| Contradiction detection | Done | `governance.find_contradictions`, served by retrieval | `test_research_separates_facts_...` |
| Status lifecycle candidate to adjudicated | Done | `governance-1` | `test_governance.py` |
| Reversible merges | Done | reviewer "same" / "different" decisions override every rule; revert restores the automatic result | `test_reviewer_split_and_merge_are_reversible_and_recorded` |
| Reversible governance decisions | Done | `revoke_adjudication` | `test_adjudication_can_be_revoked_...` |
| Historical decision and contribution records | Done | `decision_history` (append only) | same |
| Extraction proposes, governance decides | Done | model output, figures and agent proposals all enter Pass 2 as candidates | `test_agent_proposals_enter_the_graph_only_through_governance` |

### 2. Document intelligence

| Item | Status | Notes |
|---|---|---|
| PDF ingestion, SHA-256 fingerprint, duplicate detection | Done | unchanged |
| Digital / scanned / mixed classification and quality report | Done | `document_intelligence/quality.py`; measured: 3 digital documents, NASA-STD-3000B scanned (503 OCR pages, 98.2 % usable) |
| PyMuPDF + Tesseract | Done | unchanged |
| Structure: headings, tables, figures, captions, equations | Done | `parsers.py` (PyMuPDF) |
| Docling evaluation | **Blocked** | not installed: its PyTorch/Transformers requirements conflict with the pinned GLiNER stack. Needs approval to install into a copy of the `.venv`; the adapter is ready |
| Semantic chunking with section references | Done | unchanged |
| Incremental ingestion, retries, recovery | Done | stage runs, `resume` |
| Original preservation, version history | Done | unchanged |
| DOCX, HTML, Markdown | Done | `document_intelligence/text_formats.py`: logical pages through the PDF sidecar path; upload, CLI and watcher accept them. No page images (the viewer falls back to text) |

### 3. Local AI

| Item | Status | Notes |
|---|---|---|
| Provider-neutral router, Ollama, sequential gate, RAM budget | Done | unchanged; no silent cloud fallback (the benchmark forces the Ollama provider) |
| Concept, relationship, claim, figure extraction; abbreviation and synonym resolution | Done | all through `TextModel` / `VisionModel` |
| Methodology and limitation extraction | Done | claim kinds `method`, `limitation`, `result` |
| Table and equation interpretation by a model | Missing | tables and equations are located (PyMuPDF) and their text is chunked; no dedicated prompt |
| Document classification by a model | Missing | deterministic digital/scanned classification only |
| Contradiction analysis | Done | deterministic (negation, values with unit conversion, inverse edges); no model step |
| Benchmarks: RAM, latency, failures, quality | **Blocked** | harness done (`orchestration/benchmark.py`, `cli benchmark --model ... --chunks N [--dataset ...]`, tested with a fake). No model is pulled (Ollama 0.40.13 installed). Needs approval to pull `qwen2.5:7b` (about 4.7 GB) and run real inference |
| Prompt-injection protection | Done | untrusted-text rule in every extraction prompt, delimiter fencing, instruction detection sends candidates to review (`evidence_quality/injection.py`) |

### 4. Repository

| Item | Status | Notes |
|---|---|---|
| Shared repository interface | Partial | services, governance, retrieval of the governed graph, MCP and API use it; the GLiNER stages and the lexical engine still read MongoDB directly (listed in `tests/test_architecture.py`) |
| Entities, aliases, claims, relationships, evidence, provenance, versions | Done | |
| Ontology records, governance decisions, agent contributions, merge history | Done | `ontologies`, `adjudications`, `decision_history`, `proposals`, `kg_resolutions` |
| Idempotent writes | Done | stable IDs, upserts |
| Concurrency protection | Done | `knowledge_repository/locks.py`: one writer per project across threads and processes, expiring locks; API answers 409 |
| Transactions | Decision | the local MongoDB is a standalone server, which has no multi-document transactions. Writes are idempotent and rebuildable instead; enabling a single-node replica set would allow transactions |
| PostgreSQL + pgvector | Decision | not migrated. Measured: the largest graph has 545 concepts and 813 relationships, hybrid search answers in a median of 78 ms, and semantic vectors (about 4 000 passages at 384 dimensions, about 6 MB) fit in memory. Reconsider when one project exceeds about 1 million passages or needs relational constraints across services. The repository interface keeps the migration path described in `ARCHITECTURE.md` section 8 |
| Backup and restore | Done | `knowledge_repository/backup.py`, `cli backup|restore`: per project, checksummed, never overwrites |

### 5. Retrieval and reasoning

| Item | Status | Notes |
|---|---|---|
| Lexical retrieval | Done | BM25 with phrase and concept boosts |
| Semantic retrieval | Partial | `retrieval/semantic.py`: passages, incremental index, tested with a fake embedder. The real index is not built: building runs the fastembed model (`cli semantic-index`), which needs approval |
| Graph retrieval, multi-hop, paths | Done | `retrieval/knowledge_graph.py` over the governed graph (either source) |
| Hybrid retrieval (reciprocal rank fusion) | Done | `retrieval/hybrid.py` |
| Answers: accepted knowledge, conflicts, facts vs interpretations vs hypotheses, uncertainty | Done | `research`; contested claims are never shown as facts |
| Citations preserved and validated | Done | every returned quote is re-checked against its page; claims with a failing quote are demoted |
| Deterministic calculations and unit validation | Done | conversions to the unit a question asks for (`units.convert`) |
| Governed concept-first graph served to users | Done | map source selector, `/api/knowledge/*`, MCP |

### 6. Interfaces

| Item | Status | Notes |
|---|---|---|
| REST: projects, documents, jobs, search, ask, graph, governance | Done | |
| REST: hybrid search, research, ontology, claims, evidence, conflicts, exploration, review, merges, proposals, history, agent activity, diagnostics, quality, semantic index | Done | `api/app.py`, `test_rest_endpoints_reach_the_same_services` |
| CLI: search, research, diagnostics, review, history, activity, semantic-index, backup, restore, evaluate, benchmark | Done | `api/cli.py` (commands that need MongoDB are exercised through the services they call) |
| MCP server | Done | `api/mcp_server.py`, 14 tools, see `docs/MCP.md`; verified in process and over stdio against the local database (handshake, tool list, ping). A session driven by Claude Code or Codex itself has not been run |
| Scoped agent permissions, audit | Done | `orchestration/agents.py`, `agent_activity` |

### 7. Knowledge Sphere (web)

| Item | Status | Notes |
|---|---|---|
| Overview: counts, statuses, central concepts, problems, decisions, agent activity | Done | `sphere.js` |
| Concept and relationship exploration of the governed graph | Done | map source "Governed concepts"; panel with claims by modality, relationships, evidence |
| Evidence explorer | Partial | evidence in the concept panel and the review queue, each quote re-checked; no standalone evidence screen |
| Document library | Done | upload of PDF, DOCX, HTML, Markdown |
| Research assistant | Done | the Research screen uses `/api/research` |
| Governance review | Done | queue, accept/reject, merge or keep apart, revert merges, proposals, history |
| Semantic colours | Done | unchanged; concept-first core types (artifact, process, ...) are not among the six coloured types and show in the neutral grey, as the colour specification requires. Mapping them is a product decision |
| Browser check | Partial | renderers tested with Node on real API responses; pages and endpoints served by a running app checked over HTTP. Not checked visually in a browser |

### 8. Quality, security, reliability

| Item | Status | Measured / notes |
|---|---|---|
| Unit, integration, architecture tests | Done | 170 tests |
| Evaluation datasets | Partial | `evaluation/datasets/eclss_seed.json`: 12 retrieval queries, a DRAFT labelled by the assistant, needs human review. No extraction gold set yet |
| Retrieval Recall@10 | Done (measured) | seed set: lexical Recall@10 0.917, MRR 0.75; hybrid (lexical + GLiNER graph, no semantic index) Recall@10 0.917, MRR 0.683. The paraphrased query about a fire is missed by both; the graph signal moved one exact-term query from rank 1 to 5 |
| Citation correctness | Done (measured) | returned quotes 1.0 (both methods); stored graph evidence 3 606 of 3 606 quotes found in their source (1.0) |
| Extraction precision and recall | Blocked | metric implemented (`evaluate_extraction`); needs a human-labelled page set and extractions from a real model |
| Prompt-injection protection | Done | see section 3 |
| Scoped agent permissions, audit logs | Done | |
| Structured logging | Done | `src/observability.py`, `KM_LOG_FORMAT=json` (MCP server; the API and CLI keep their handlers) |
| Backup and restore | Done | |
| Recovery from failed jobs | Done | stage runs and `resume`; expiring project locks |
| Concurrency protection | Done | project locks |
| Performance | Done (measured) | hybrid search median 78 ms, max 101 ms; research answers median 139 ms (10 queries, real database). Target from the planning document: search under 500 ms |

## Quality targets from the planning document

| Target | Status |
|---|---|
| Search API under 500 ms | Met on the current corpus (median 78 ms) |
| 0 duplicate canonical nodes | Not measured: needs a labelled duplicate set; resolution and its guards are tested |
| 0 % hallucination, 100 % abstention without evidence | Answers are extractive by construction; abstention tested on fixtures. No 100-question benchmark exists yet |
| > 95 % text reading accuracy | Not measured: needs ground-truth text for scanned pages |
| Zero cross-tenant access | Project isolation tested throughout; agent grants per project |

## Blockers that need a decision or approval

1. Pull and benchmark local models (`ollama pull qwen2.5:7b`, `qwen2.5vl:3b`), then
   `cli benchmark --model qwen2.5:7b --chunks 20`: real inference.
2. Build the semantic index (`cli semantic-index`): runs the fastembed model over about 4 000 passages.
3. Install Docling in a separate environment to evaluate it on the scanned standard.
4. Human review of `evaluation/datasets/eclss_seed.json` and a labelled extraction set.
5. Decide whether concept-first core types get colours (they are grey under the current colour specification).

## Audit at the start of the finalisation (before any change)

| Area | Item | Status at audit |
|---|---|---|
| Contracts | Canonical models | Done |
| Contracts | Confidence vs uncertainty | Partial |
| Contracts | Fact / interpretation / hypothesis | Missing |
| Contracts | Extensible ontology | Missing |
| Knowledge | Canonical resolution | Done |
| Knowledge | Claims, contradictions, lifecycle | Done |
| Knowledge | Reversible merges and decisions, history, agent contributions | Missing |
| Knowledge | Methodology and limitation extraction | Missing |
| Documents | PDF, OCR, structure, chunking, versions, recovery | Done |
| Documents | Digital / scanned / mixed classification, quality report | Partial |
| Documents | Docling | Blocked |
| Documents | DOCX, HTML, Markdown | Missing |
| Models | Router and Ollama | Done |
| Models | Benchmarks | Blocked, no harness |
| Repository | Shared interface | Partial |
| Repository | Ontology, history, contributions, merge history, transactions | Missing |
| Retrieval | Lexical | Done |
| Retrieval | Semantic, hybrid | Missing |
| Retrieval | Graph retrieval | Partial (GLiNER neighbours only) |
| Retrieval | Governed concept-first graph served to users | Disconnected |
| Interfaces | REST and CLI basics | Done |
| Interfaces | MCP, agent permissions | Missing |
| Frontend | Overview, review, evidence, agent activity | Missing |
| Quality | Evaluation, injection protection, audit, logging, backup, locks, performance | Missing or partial |
