# Knowledge Map

Knowledge Map turns PDF books into a knowledge map you can explore and question. It extracts text and images from each page, finds concepts, merges concepts with the same meaning into one node, connects concepts that appear together, and answers questions by quoting sentences from the documents. Every answer sentence, concept description and relationship carries a citation:

```
[Document Title | Author/Editor | Year of Publication, p. N]
```

Answers are extractive. The web app never writes its own text and calls no LLM. If the documents do not contain enough evidence, the answer is exactly: `Not found in the uploaded documents.`

A separate, optional **concept-first pipeline** (`python -m src.api.kg_cli`) uses Claude to extract concepts and stated relationships from semantic chunks. It runs only when you start it, needs an API key, and writes to its own collections (`kg_concepts`, `kg_relationships`). Every item it stores is checked against a verbatim quote from the source.

## Features

- **Projects**: independent workspaces. Each project has its own documents, concepts, relationships, map, search and answers; nothing is shared or connected between projects. Create, switch, rename and delete projects in the sidebar.
- **Add, edit, delete documents**: upload a PDF, replace it with a new version, edit its citation metadata, or delete it with everything derived from it. Concepts, relationships and the index are rebuilt after every change.
- **Watched folder**: PDFs added to, changed in or removed from `data/inbox/` go through the whole pipeline automatically while the app (or `python -m src.api.cli watch`) runs.
- **Ingestion**: PDF upload from the web app or the command line. Page text comes from PyMuPDF, with Tesseract OCR for scanned pages. Embedded images are saved with page, bounding box, nearby caption and surrounding text. Optional image captioning (ViT-GPT2) is off by default and is stored apart from the source text.
- **Citation metadata**: title, author/editor and year are detected from PDF metadata and the first pages, written to `data/document_metadata.json`, and shown as "Unknown" when not found. You can edit them in the file or on the Documents screen; citations everywhere update.
- **Concepts**: GLiNER finds concept spans, batched and resumable. Table-of-contents pages, generic words and fragments are filtered out.
- **Merging**: names with the same meaning become one node, using alias hashing, fuzzy matching (rapidfuzz) and embeddings (fastembed, bge-small). The rules are conservative, and every decision is written to `output/reports/merge_report.csv`.
- **Relationships**: edges come from co-occurrence in a sentence or a page window, scored with NPMI. The type is `related_to` unless a clear phrase supports `is_a`, `part_of`, `used_for` or `causes`. Every edge stores verbatim evidence quotes.
- **Graph**: built in memory with NetworkX, with degree, PageRank importance and Louvain communities. Colours show the concept type and come from one mapping (`src/web/types.js`), used by graph nodes, the type filter, the legend, badges and the concept panel; unknown types are neutral grey.
- **Search and Ask**: BM25 over a persistent inverted index, with concept-name boosting and prefix autocomplete. `/ask` returns verbatim sentences grouped by source. A validator checks each quote against the stored page before it is returned.
- **Concept-first graph (optional, model-based)**: semantic chunking, Pass 1 concept and relationship extraction per chunk with a local Qwen model through Ollama (or Claude), claim extraction, figure understanding with Qwen2.5-VL grounded on caption and OCR text, Pass 2 cross-document merging, recovery of malformed answers, and storage with full provenance. See "Concept-first knowledge graph" below.
- **Evidence governance**: extraction only proposes candidates. Independent checks (evidence traceability, source and page validity, relationship endpoints, duplicate aliases, numbers and units, contradictions) decide each item's status: `candidate`, `verified`, `corroborated` (two or more documents), `needs_review`, `rejected` or `adjudicated` (a reviewer's decision, kept across rebuilds). A model's confidence score never decides. Rejected items are kept for audit and hidden from the map and answers.
- **Document structure**: a parser adapter (Docling when installed, PyMuPDF otherwise) adds sections with their hierarchy, tables, figure captions and equations as content units tied to page and bounding box. Stored page text is never replaced.
- **Versions**: replacing a document's PDF records a new version in the same lineage; history is kept.
- **Architecture**: one folder per component with typed contracts between them; see `docs/ARCHITECTURE.md`.
- **Obsidian export**: one note per concept and one index note per document.
- **Knowledge engine**: claims carry a kind (requirement, specification, measurement, definition, cause, comparison, method, limitation, result) and a modality (fact, interpretation, hypothesis) that follows the source's own hedging; uncertainty (approximate values, open items, hedged wording) is stored apart from confidence. Projects can extend the ontology with domain types that specialise core types. Reviewers can merge two concepts or keep two names apart, revoke a decision or revert a merge; every decision and contribution is kept in an append-only history.
- **Hybrid retrieval and research answers**: lexical (BM25), semantic (passage embeddings, once the index is built) and graph retrieval over the governed graph, combined by reciprocal rank fusion. Research answers separate accepted facts, interpretations, hypotheses and unconfirmed claims, list conflicts, convert units deterministically and abstain without evidence. Every quote is re-checked against its page.
- **Other formats**: DOCX, HTML and Markdown are ingested as logical pages through the same path as PDFs; each document gets a quality report (digital, scanned or mixed).
- **Agents (MCP)**: a local MCP server gives Claude Code, Codex and other clients 14 tools (search, research, documents, concepts, claims, evidence, graph exploration, contradictions, project context, proposals) with per-agent project and scope grants and an audit log. See `docs/MCP.md`.
- **Operations**: one writer per project (locks across processes), backup and restore of a project, diagnostics, JSON logs (`KM_LOG_FORMAT=json`), evaluation of retrieval and citations against labelled datasets, and a model benchmark harness. Status of every requirement: `docs/IMPLEMENTATION_STATUS.md`.
- **Web app** at `http://localhost:8000` with five screens: Overview, Documents, Knowledge Map (extracted mentions or governed concepts), Research, Review.

## Tech stack

| Area | Tool | Why it is lightweight |
|---|---|---|
| Language | Python 3.12 | One language for the whole system. |
| PDF text and images | PyMuPDF | A single C-backed library; no external service. |
| OCR fallback | Tesseract via pytesseract | Runs only on pages without a usable text layer. |
| Storage (source of truth) | MongoDB Community Server, PyMongo | One local server already in use; no extra database for graphs or vectors. |
| Concept extraction | GLiNER `urchade/gliner_medium-v2.1` (PyTorch, CPU) | A small BERT-family model that runs locally; no API calls. |
| Contracts | Pydantic 2 | Shared, versioned models for every knowledge record; no ORM. |
| Local models | Ollama running Qwen (`qwen2.5:7b`) and Qwen2.5-VL (`qwen2.5vl:3b`) | One local server, one model in memory at a time (16 GB RAM laptop); no Python model code for Qwen. |
| Hosted model (optional) | Claude via the official `anthropic` SDK, JSON-schema structured output | Only when `KM_LLM_PROVIDER=anthropic`; nothing else needs an API key. |
| Structure parser (optional) | Docling | Used when installed (`KM_PARSER=docling` or `auto`); PyMuPDF is the fallback. |
| Concept merging | fastembed `BAAI/bge-small-en-v1.5` (ONNX), rapidfuzz | A small ONNX model, used only during merging; not at query time. |
| Keyword search | Own BM25 code, Snowball stemmer (`snowballstemmer`) | Pure Python dictionaries; no search server or vector store. |
| Graph | NetworkX (PageRank computed with NumPy) | In-memory graph built from MongoDB; no Neo4j. |
| API | FastAPI, uvicorn | One process serves the API and the web page. |
| Frontend | Plain HTML/CSS/JavaScript, Cytoscape.js 3.30.2 (vendored in `src/web/vendor/`) | No Node.js, no build step; works offline. |
| Tests | pytest, mongomock, mocked models | No database server or model download needed for tests. |

All versions are pinned in `requirements.txt`.

## Install (virtual environment)

You need Python 3.12, a local MongoDB Community Server, and (for scanned pages) the Tesseract executable.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows (cmd): use `.venv\Scripts\activate.bat` instead of `Activate.ps1`.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All packages go into `.venv`; your global Python is not changed.

On Linux, `pip` installs the large CUDA build of torch by default. For CPU only, install torch first from the CPU index, then the rest:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
``` The first pipeline run downloads the GLiNER model (about 0.8 GB) and the bge-small ONNX model (about 0.1 GB) into the local model cache.

### Version conflicts with other projects

This project pins exact versions of torch, transformers, Pillow, numpy and others. Other projects on the same machine may need different versions (for example `facenet-pytorch` needs torch below 2.3 and Pillow below 10.3; `streamlit` 1.45 needs Pillow below 12). Do not install `requirements.txt` into your global Python. Keep this project in its own `.venv`, and other projects in their own environments.

Check which interpreter and versions you are using:

```bash
python -m src.envcheck
```

It prints `OK` inside a correct `.venv`. Otherwise it lists every package whose version differs from `requirements.txt`. The pipeline and the server print the same warning at start. In VS Code, choose the interpreter `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/macOS) with "Python: Select Interpreter".

## MongoDB setup

Install MongoDB Community Server and start it locally. The default connection is `mongodb://127.0.0.1:27017`, database `knowledge_map`. To change it, create a `.env` file in the project root (it is gitignored):

```
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DATABASE=knowledge_map
```

Put credentials only in `.env`, never in code. More detail: `docs/mongodb_storage.md`.

If Tesseract is not on `PATH`, set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`.

## Run

All commands run from the project root with the `.venv` activated.

```bash
# 1. Process the books in data/raw_pdfs (read only) and run every stage
python -m src.api.cli run

# 2. Start the web app, then open http://localhost:8000
uvicorn src.api.app:app --port 8000

# 3. Optional: write the Obsidian vault to output/obsidian_vault/
python -m src.api.cli export
```

`python -m src.api.cli run` is safe to repeat. Unchanged documents and pages are skipped by their SHA-256 hash, so a second run only redoes the fast stages (merging, relationships, index).

## Projects

Every document belongs to exactly one project, and everything derived from it (pages, images, concept mentions, concepts, relationships, index entries, the concept-first graph) belongs to the same project. The project selector in the sidebar decides which project the Documents, Knowledge Map and Ask screens show, and where uploads go.

- **Isolation is enforced by the server, not only the page.** Every API request names its project (`?project=ID`; default `default`). Every query is filtered by project. Concept, page, image and document lookups return 404 for another project's records. Merging, relationships, image links and the keyword index are rebuilt per project and only replace that project's records.
- **The same name or PDF in two projects never becomes one record.** Concept, alias, index-term and document IDs include the project, so identical concepts in two projects stay two separate nodes and are never merged or connected. The same PDF can be added to several projects.
- **Existing data** belongs to the project `default` ("Default project"). On first start the app adds `project_id: "default"` to records that have none; nothing is deleted or overwritten, and default-project IDs are unchanged. The default project can be renamed but not deleted.
- **Deleting a project** removes its documents (with their copies in `data/uploads/` and images in `data/images/`) and its whole graph. Other projects are not touched. Files in `data/raw_pdfs/` are never deleted.
- **Command line:** add `--project ID` to any pipeline command, for example `python -m src.api.cli --project 3f2a9c1b7d4e ingest notes.pdf`. List or create projects with `python -m src.api.cli projects [--create NAME]`. `python -m src.api.cli run` (the books in `data/raw_pdfs`) always uses the default project. `python -m src.api.kg_cli` and `python -m src.retrieval.graph` also take `--project`.
- **Watched folder:** a PDF in `data/inbox/<project id>/` goes to that project; a PDF directly in `data/inbox/` goes to the default project.

## Adding, changing and deleting documents

| What you want | In the web app (Documents screen) | Watched folder `data/inbox/` | Command line |
|---|---|---|---|
| Add a PDF | Drag and drop, or click to upload | Copy the PDF into the folder | `python -m src.api.cli ingest FILE.pdf` |
| New version of a PDF | "Replace file" on its row | Overwrite the file with the new version | Delete, then ingest |
| Fix title, author/editor, year | Edit the fields, then "Save" | | Edit `data/document_metadata.json` |
| Delete | "Delete", then "Confirm delete" | Remove the file from the folder | `python -m src.api.cli delete ID_OR_FILENAME` |

What happens:

- **Add**: text, images and metadata are extracted, then concepts, merging, image links, relationships and the index are updated. Pages already processed (same text hash) are not processed again.
- **Replace**: the new version is added, metadata fields you edited are copied to it, and the old version is deleted.
- **Delete**: the document's pages, images, image files, concept mentions, index entries and metadata entry are removed, then concepts, relationships and the index are rebuilt. Concepts that only came from this document disappear. Files in `data/raw_pdfs/` are never deleted. A document deleted in the app is remembered (`deleted_documents`), so `pipeline run` and the folder watcher do not add the same file back. Uploading it again on purpose adds it back.
- **Watched folder**: the app scans `data/inbox/` every 5 seconds (change with `KM_WATCH_DIR`, `KM_WATCH_INTERVAL`; turn off with `KM_WATCH_ENABLED=0` in `.env`). A file is processed once its size and time have not changed for one scan, so files still being copied are not read. If the same PDF is also uploaded or in `data/raw_pdfs/`, removing it from the folder keeps the document.

Each change runs as one background job. Jobs run one at a time, and the Documents screen shows their progress.

## Concept-first knowledge graph (optional, model-based)

```
Document -> semantic segmentation -> Pass 1: concepts, relationships (text model) -> local validation / recovery
         -> claims (text model)    -> figures (vision model, grounded on caption + OCR)
         -> Pass 2: cross-document enrichment -> graph validation -> governance
         -> MongoDB (kg_concepts, kg_relationships, kg_claims, validation_records)
```

**Setup (local, default).** Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b` (text) and, for figures, `ollama pull qwen2.5vl:3b`, and keep `ollama serve` running. Only one model is in memory at a time; a model whose estimated memory is above `KM_MODEL_RAM_GB` (default 10) is refused before loading. Settings in `.env`: `KM_LLM_MODEL`, `KM_VISION_MODEL`, `KM_OLLAMA_URL` (default `http://localhost:11434`), `KM_OLLAMA_NUM_CTX` (default 8192), `KM_OLLAMA_KEEP_ALIVE` (default `5m`), `KM_USE_GLINER=0` to turn off the GLiNER stage.

**Setup (Claude, optional).** `KM_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` in `.env` (gitignored); `KM_LLM_MODEL` (default `claude-opus-5-5`), `KM_LLM_EFFORT` (default `medium`), `KM_LLM_FALLBACKS=0` to turn off the server-side refusal fallback. Shared settings: `KM_LLM_MAX_RETRIES` (default 2), `KM_CHUNK_MIN_CHARS` / `KM_CHUNK_TARGET_CHARS` / `KM_CHUNK_MAX_CHARS` / `KM_CHUNK_COHESION`, `KM_KG_EMBEDDINGS=0` to merge without embeddings. `python -m src.api.cli models` shows what each capability is routed to.

**Run on one document** (filename, document ID or ID prefix):

```bash
python -m src.api.kg_cli --document ICES-2023-74.pdf
python -m src.api.kg_cli --document ICES-2023-74.pdf --limit-chunks 5   # small trial first
python -m src.api.kg_cli --all
python -m src.api.kg_cli --all --claims                               # also claim candidates
python -m src.api.kg_cli --all --visual --no-concepts                 # figures only (Qwen2.5-VL + OCR)
python -m src.api.kg_cli --rebuild-only                               # Pass 2 + governance, no model calls
```

**Domain and abbreviations.** The extraction prompt tells the model the domain (space technology, aerospace and systems engineering, spacecraft and satellites, NASA documentation, requirements, V&V, integration and testing) and to extract one concept per term: the full form as the name and the abbreviation as an alias. Full forms are used only when the passage states them or when they are in the standard list in `src/contracts/vocabulary.py` (`DOMAIN_ABBREVIATIONS`: NASA, MBSE, V&V, ICD; SE only when the passage supports it); anything else stays unresolved. Validation sends an expansion the passage does not state to review, and Pass 2 merges an abbreviation used alone in another document into its full form unless two concepts claim it.

**Canonical concept resolution.** After the rule-based merges, Pass 2 asks the same text model about pairs the rules left apart that may still be one thing: a short name whose letters are another name's initials ("ECLSS"), an alias claimed by several concepts, and names that are close in meaning but not in spelling ("spacecraft bus" and "satellite platform"). Candidates come from general rules, not a list of terms; the model decides from the passages each name occurs in and answers same, different or unsure. Only "same" merges (the variants become aliases, relationships move to the canonical node, every mention keeps its document, page and quote). "Unsure", or an abbreviation that matches several concepts, merges nothing and sends the concepts to review through governance (`possible_duplicate` check). Names that differ in a number or a negation never merge. Decisions are stored per project in `kg_resolutions` and reused, so reruns and rebuilds without a model give the same graph. At most `KM_KG_RESOLVE_PAIRS` (default 50) new questions are asked per build; `KM_KG_RESOLVE=0` turns resolution off.

With Ollama nothing leaves the machine, but CPU inference is slow; start with `--limit-chunks` to check speed and quality. With Claude, chunks are sent to the API, which costs money per token and shares that text with Anthropic. Chunk counts on the current four documents: ICES-2023-74 33, eclss-technical-brief 54, BVAD 777, NASA-STD-3000B 607.

**What each step does**

| Step | Module | What it does |
|---|---|---|
| Segmentation | `src/document_intelligence/segmentation.py` | Splits stored page text into chunks at paragraphs and headings, closes a chunk at a size limit or when word overlap with the next paragraph drops (topic shift), keeps section titles, skips running page headers, marks table-of-contents pages, never drops text. |
| Pass 1 | `src/knowledge_extraction/llm_extraction.py`, `prompts.py`, `llm.py` | Asks Claude for concepts (name, type, description, aliases, evidence quote) and relationships from a fixed vocabulary (`is_a`, `includes`, `uses`, `defines`, `prerequisite_for`, `example_of`, `related_to`), as JSON following `schema.py`. |
| Validation | `src/evidence_quality/validation.py` | Marks every item `supported`, `needs_review` or `rejected`: schema, vocabulary, endpoints, quotes found in the passage, names inside quotes, duplicates, self-loops; after Pass 2, endpoints and source chunks exist. Rejected items never reach the graph. |
| Recovery | `src/knowledge_extraction/recovery.py` | Fixes harmless syntax (code fences, trailing commas), otherwise sends the error back to Claude for that chunk only, at most `KM_LLM_MAX_RETRIES` times. Every raw answer and error is stored on the chunk. A chunk that still fails is marked for review and retried on the next run; nothing is invented. |
| Pass 2 | `src/knowledge_extraction/enrichment.py` | Merges the same concept across chunks and documents (same normalized name, stated aliases, spelling variants, name embeddings) with the same conservative guards as the GLiNER merge; an alias claimed for two concepts merges nothing. Relationships are only re-pointed and combined, never created. |
| Storage | `src/knowledge_repository/kg_store.py` | Chunks in `content_units` (`unit_type: "chunk"`), graph in `kg_concepts` and `kg_relationships`, stable IDs, upserts; see `docs/mongodb_storage.md`. |
| Orchestration | `src/orchestration/kg_pipeline.py` | Runs the steps and prints a summary (chunks, concepts extracted and merged, relationships, validation counts, recovery attempts and failures). Merge decisions go to `output/reports/kg_merge_report.json`. |

Re-running is safe: unchanged chunks reuse their stored result (same text, prompt version and model), so a second run makes no LLM calls and creates no duplicates. When a document is added, replaced or deleted in the app, the concept-first graph is rebuilt from the stored results without LLM calls; extracting new text needs the command above.

## CLI reference

| Command | What it does |
|---|---|
| `python -m src.api.cli run` | Ingests `data/raw_pdfs/*.pdf` (text, metadata, images), then concepts, merge, image links, relationships and index. Writes timings to `output/reports/pipeline_run.json`. |
| `python -m src.api.cli ingest FILE.pdf [...]` | Copies new PDFs to `data/uploads/<hash>/`, ingests them, then runs every stage. |
| `python -m src.api.cli watch [--folder DIR] [--interval S]` | Watches a folder (default `data/inbox/`) without the web app and processes added, changed and removed PDFs. |
| `python -m src.api.cli documents` | Lists documents with ID, pages and sources. |
| `python -m src.api.cli delete ID_OR_FILENAME [...]` | Deletes documents (ID, ID prefix or filename), then rebuilds. |
| `python -m src.api.kg_cli --document NAME` | Concept-first pipeline for one document (see above). Also `--all`, `--limit-chunks N`, `--claims`, `--visual`, `--no-concepts`, `--rebuild-only`. |
| `python -m src.api.cli structure [--document ID] [--parser auto/docling/pymupdf]` | Adds sections, tables, figure captions and equations as content units (no model; page text unchanged). |
| `python -m src.api.cli validate [--source gliner/concept_first]` | Runs evidence governance and stores statuses and validation records. |
| `python -m src.api.cli rebuild` | Rebuilds the project's graphs, index and governance from stored results (no ingestion, no model calls). |
| `python -m src.api.cli status` / `resume` | Shows stage runs and knowledge statuses; reruns the knowledge stages if a run did not finish (finished work is skipped). |
| `python -m src.api.cli versions DOCUMENT_ID` | Version history of a document. |
| `python -m src.api.cli models` | Which model serves each capability (loads nothing). |
| `python -m src.envcheck` | Checks the interpreter and package versions against `requirements.txt`. |
| `python -m src.api.cli concepts` | Concept mention extraction only (resumable, progress bar). |
| `python -m src.api.cli merge` | Rebuilds concept nodes and aliases, writes `output/reports/merge_report.csv`, links images to concepts. |
| `python -m src.api.cli relations` | Rebuilds relationships. |
| `python -m src.api.cli index` | Updates the keyword index. |
| `python -m src.api.cli export` | Writes the Obsidian vault. |
| `python -m src.api.cli metadata` | Detects citation metadata for documents that have none yet. |
| `python -m src.api.cli stats` | Prints collection counts and graph statistics. |
| `python -m src.api.cli search "QUERY"` / `research "QUESTION"` | Hybrid search; evidence-first answer with claims, conflicts and unit conversions. |
| `python -m src.api.cli diagnostics [--probe-models]` | Project and platform health; `--probe-models` asks Ollama which models are pulled (nothing is loaded). |
| `python -m src.api.cli review` / `history` / `activity` | Items waiting for review; decisions and contributions; agent calls. |
| `python -m src.api.cli semantic-index` | Embeds page passages for semantic retrieval (runs the local embedding model; incremental). |
| `python -m src.api.cli backup FOLDER [--all-projects]` / `restore FOLDER` | Back up a project as checksummed Extended JSON; restore refuses to overwrite existing records. |
| `python -m src.api.cli evaluate DATASET.json [--method lexical/hybrid]` | Recall@10, MRR, citation correctness and extraction precision/recall against a labelled dataset (no model). |
| `python -m src.api.cli benchmark --model qwen2.5:7b --chunks 20 [--dataset ...]` | Runs Pass 1 with a local Ollama model on stored chunks and measures latency, failures, retries, peak RAM and quality (real inference; stores nothing). |
| `python -m src.api.mcp_server` | MCP server over stdio for coding agents (see `docs/MCP.md`). |
| `python -m src.api.cli --caption-images run` | Same as `run`, also captions images with ViT-GPT2 (off by default). |
| `python -m src.retrieval.graph` | Prints node count, edge count and top connected concepts. |
| `python -m src.knowledge_extraction.gliner_concepts` | Concept extraction with options `--document-id`, `--limit-pages`, `--threshold`, `--batch-size`, `--model`. |
| `python src/document_intelligence/pdf_extraction.py` | Original extraction step: `data/raw_pdfs` to `data/extracted`. |
| `python -m src.knowledge_repository.mongodb_store` | Original ingestion step: `data/extracted` to MongoDB. |
| `python -m pytest` | Runs all tests. |

Logs are written to `output/logs/pipeline.log`.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/projects`, `POST /api/projects`, `PUT /api/projects/{id}`, `DELETE /api/projects/{id}` | List, create, rename and delete projects (delete runs as a background job; the default project cannot be deleted). |
| `GET /api/health`, `GET /api/stats` | Server and database status, counts. All endpoints below take `?project=ID` (default `default`). |
| `POST /api/documents/upload` | Upload a PDF; processing runs as a background job. |
| `PUT /api/documents/{id}/file` | Replace a document with a new version of its PDF (background job). |
| `DELETE /api/documents/{id}` | Delete a document and everything derived from it (background job). |
| `GET /api/documents`, `GET /api/jobs`, `GET /api/jobs/{id}` | Documents with sources, status and progress; jobs. |
| `GET /api/watch` | Watched folder, scan interval, last scan, last changes. |
| `PUT /api/documents/{id}/metadata` | Edit title, author/editor, year. |
| `GET /api/concepts/autocomplete?q=` | Prefix autocomplete over names and aliases. |
| `GET /api/concepts/search?q=` | Concepts named in the text, then prefix matches. |
| `GET /api/concepts/{id}` | Description, aliases, related concepts with evidence, sources, images. |
| `GET /api/graph?document_id=&type=&min_weight=&limit=&focus=` | Graph elements for Cytoscape. |
| `GET /api/search?q=` | BM25 page search with highlighted snippets and citations. |
| `POST /api/ask` `{"question": "..."}` | Extractive answer with citations, or "Not found in the uploaded documents." |
| `GET /api/pages/{id}`, `GET /api/pages/{id}/image?quote=` | Page text, and the rendered page with the quote highlighted. |
| `GET /api/images/{id}` | An extracted image file. |
| `GET /api/documents/{id}/versions` | Version history of a document. |
| `POST /api/documents/{id}/structure?parser=` | Parse sections, tables, figures and equations (background job). |
| `POST /api/projects/{id}/rebuild`, `POST /api/projects/{id}/validate`, `GET /api/projects/{id}/stages` | Rebuild a project (job), run governance, show stage runs. |
| `GET /api/knowledge/summary`, `GET /api/knowledge/{item}/validation`, `POST /api/knowledge/{item}/adjudicate` | Status counts per graph, the checks behind one item's status, a reviewer decision (body: reviewer, decision `accept` or `reject`, note). |
| `GET /api/models` | Capability routing (text, vision, spans, embeddings). |
| `GET /api/search/hybrid?q=&source=`, `POST /api/research` | Hybrid search; evidence-first research answer. |
| `GET /api/graph?source=concept_first` | The governed concept-first graph for the map (default `gliner`, as before). |
| `GET /api/knowledge/concepts?q=`, `GET /api/knowledge/concepts/{id}` | Governed concepts: aliases, evidence, relationships, claims by modality, conflicts. |
| `GET /api/knowledge/graph/explore?concept=&hops=`, `GET /api/knowledge/graph/path?a=&b=` | Multi-hop neighbourhood; shortest relationship chain with evidence. |
| `GET /api/knowledge/claims?concept=&kind=&modality=`, `GET /api/knowledge/conflicts`, `GET /api/knowledge/items/{id}/evidence` | Claims, conflicts, evidence re-checked against the source. |
| `GET /api/knowledge/review`, `POST /api/knowledge/{id}/revoke` | Review queue; revoke a reviewer decision. |
| `GET/POST /api/knowledge/merges`, `POST /api/knowledge/merges/{id}/revert` | Reviewer merge or keep-apart decisions and their reversal. |
| `GET/POST /api/ontology`, `GET/POST /api/proposals`, `POST /api/proposals/{id}/withdraw` | Ontology extensions; knowledge proposals. |
| `GET /api/history`, `GET /api/agents/activity` | Decision history; agent calls. |
| `GET /api/projects/{id}/context`, `GET /api/projects/{id}/diagnostics`, `GET /api/documents/{id}/quality` | Project overview, health, document quality. |
| `POST /api/projects/{id}/semantic-index`, `POST /api/projects/{id}/concept-graph` | Build the semantic index (job, runs the embedding model); rebuild the concept-first graph from stored results (job, no model). |

## Project structure

```
data/raw_pdfs/              original books (never modified)
data/extracted/             text and .pages.jsonl from the original extraction step
data/inbox/                 watched folder: add, change or remove PDFs here
data/uploads/               copies of uploaded and watched PDFs, one folder per file hash (gitignored)
data/images/                extracted images (gitignored)
data/document_metadata.json citation metadata, editable
docs/                       ARCHITECTURE.md, IMPLEMENTATION_STATUS.md, MCP.md, DATA_FLOW.md, REPORT.md, mongodb_storage.md
config/                     agents.example.json (MCP agent grants; copy to agents.json)
evaluation/datasets/        labelled evaluation datasets (eclss_seed.json is a draft)
output/                     obsidian_vault/, reports/, logs/ (gitignored)
src/contracts/              canonical Pydantic schemas, interfaces (protocols), IDs, shared vocabulary
src/document_intelligence/  PyMuPDF + Tesseract page extraction (unchanged), images, OCR, segmentation, parser adapters
src/model_runtime/          model router, Ollama (Qwen, Qwen2.5-VL), Claude, GLiNER, embeddings, captioning
src/knowledge_extraction/   candidates: GLiNER concepts, merging, relationships, LLM concepts, claims, figures, enrichment
src/evidence_quality/       local validation, governance checks, units, status lifecycle
src/knowledge_repository/   MongoDB store, canonical mapping, repository, projects, metadata, concept-first store
src/retrieval/              inverted index, BM25, extractive answers, citation validator, NetworkX graph
src/orchestration/          pipeline stages, concept-first pipeline, document lifecycle, watcher, stage tracking, services
src/api/                    interfaces: FastAPI app, CLI (cli.py, kg_cli.py), Obsidian exporter
src/web/                    HTML, CSS, JavaScript, vendored Cytoscape.js
src/config.py, envcheck.py, textutil.py   settings, version check, text helpers
tests/                      pytest suite (mongomock, mocked models)
```

## Limitations

- Concept types are GLiNER's guesses from a fixed label list. Some are wrong, and some extracted names are still generic or partial.
- Relationship types beyond `related_to` come only from a few English phrase patterns. Most edges are `related_to`, meaning "discussed together", not a specific semantic relation.
- Answers are sentence extracts. They can be partial, and two quoted sentences can come from different contexts. The citation shows where each one comes from.
- Sentence splitting is rule-based. Multi-column pages, tables and broken lines can produce awkward sentence boundaries. Quotes keep the original line breaks and hyphenation of the extracted text.
- Page numbers in citations are PDF page numbers (1-based), not the numbers printed on the pages.
- Metadata detection only reads PDF metadata and the first 8 pages. Check `data/document_metadata.json` and correct it if needed.
- Figure understanding (Qwen2.5-VL) only accepts concepts whose quote is found in the caption, the nearby text or the Tesseract OCR of the image; a concept drawn but not labelled in text is rejected. Model descriptions and captions are stored apart and never used as evidence.
- Concept extraction runs on CPU: 1.4 to 3.5 seconds per page on the development machine depending on other load, so about 16 minutes for the two books without other load (see `docs/REPORT.md`).
- One processing job runs at a time. Job progress is kept in memory and is lost when the server restarts. The documents themselves and the watch state are stored in MongoDB.
- Every add, replace or delete rebuilds concept merging, relationships and the index for the whole collection. With the two books this took 8 to 31 seconds per change after the models were loaded (see `docs/REPORT.md`), and it grows with the collection.
- The folder watcher polls; changes are picked up within about two scan intervals (10 seconds by default), not instantly.
- On two-column pages the original extraction (PyMuPDF `sort=True`) can interleave the columns, so some quoted sentences mix text from both columns. Quotes are still verbatim and correctly cited.
- There is no user authentication. The app is meant to run on your own machine.
- The concept-first pipeline, claims and figure extraction have only been tested with mocked models. They have not run against Ollama (no Qwen model is pulled on the development machine) or the Claude API, so their quality and speed are not measured yet.
- The PyMuPDF structure parser finds headings by font size, bold text and numbering, tables with ruling lines, and equations by a symbol heuristic. On the scanned NASA-STD-3000B it finds nothing (there is no text layer); Docling is not installed and its adapter is tested only with a fake converter.
- Concept-first descriptions are written by the LLM (stored with `generated: true` and the quote they came from). The map shows the GLiNER graph by default; "Governed concepts" shows the concept-first graph, whose core types (artifact, process, ...) appear in neutral grey because they are not among the six coloured types.
- The semantic retrieval index is not built on the development machine (building it runs the embedding model). Without it, hybrid retrieval is lexical plus graph, which on the draft ECLSS seed set did not beat lexical alone (MRR 0.683 vs 0.75); see `docs/IMPLEMENTATION_STATUS.md`.
- DOCX, HTML and Markdown documents have logical pages (about 3 000 characters), not printed pages, and no page images.
