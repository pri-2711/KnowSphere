# Knowledge Map

Knowledge Map turns PDF books into a knowledge map you can explore and question. It extracts text and images from each page, finds concepts, merges concepts with the same meaning into one node, connects concepts that appear together, and answers questions by quoting sentences from the documents. Every answer sentence, concept description and relationship carries a citation:

```
[Document Title | Author/Editor | Year of Publication, p. N]
```

Answers are extractive. The system never writes its own text and calls no external LLM. If the documents do not contain enough evidence, the answer is exactly: `Not found in the uploaded documents.`

## Features

- **Ingestion**: PDF upload from the web app or the command line. Page text comes from PyMuPDF, with Tesseract OCR for scanned pages. Embedded images are saved with page, bounding box, nearby caption and surrounding text. Optional image captioning (ViT-GPT2) is off by default and is stored apart from the source text.
- **Citation metadata**: title, author/editor and year are detected from PDF metadata and the first pages, written to `data/document_metadata.json`, and shown as "Unknown" when not found. You can edit them in the file or on the Documents screen; citations everywhere update.
- **Concepts**: GLiNER finds concept spans, batched and resumable. Table-of-contents pages, generic words and fragments are filtered out.
- **Merging**: names with the same meaning become one node, using alias hashing, fuzzy matching (rapidfuzz) and embeddings (fastembed, bge-small). The rules are conservative, and every decision is written to `output/reports/merge_report.csv`.
- **Relationships**: edges come from co-occurrence in a sentence or a page window, scored with NPMI. The type is `related_to` unless a clear phrase supports `is_a`, `part_of`, `used_for` or `causes`. Every edge stores verbatim evidence quotes.
- **Graph**: built in memory with NetworkX, with degree, PageRank importance and Louvain communities.
- **Search and Ask**: BM25 over a persistent inverted index, with concept-name boosting and prefix autocomplete. `/ask` returns verbatim sentences grouped by source. A validator checks each quote against the stored page before it is returned.
- **Obsidian export**: one note per concept and one index note per document.
- **Web app** at `http://localhost:8000` with three screens: Documents, Knowledge Map, Ask.

## Tech stack

| Area | Tool | Why it is lightweight |
|---|---|---|
| Language | Python 3.12 | One language for the whole system. |
| PDF text and images | PyMuPDF | A single C-backed library; no external service. |
| OCR fallback | Tesseract via pytesseract | Runs only on pages without a usable text layer. |
| Storage (source of truth) | MongoDB Community Server, PyMongo | One local server already in use; no extra database for graphs or vectors. |
| Concept extraction | GLiNER `urchade/gliner_medium-v2.1` (PyTorch, CPU) | A small BERT-family model that runs locally; no API calls. |
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

All packages go into `.venv`; your global Python is not changed. The first pipeline run downloads the GLiNER model (about 0.8 GB) and the bge-small ONNX model (about 0.1 GB) into the local model cache.

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
python -m src.pipeline run

# 2. Start the web app, then open http://localhost:8000
uvicorn src.api.app:app --port 8000

# 3. Optional: write the Obsidian vault to output/obsidian_vault/
python -m src.pipeline export
```

`python -m src.pipeline run` is safe to repeat. Unchanged documents and pages are skipped by their SHA-256 hash, so a second run only redoes the fast stages (merging, relationships, index).

## CLI reference

| Command | What it does |
|---|---|
| `python -m src.pipeline run` | Ingests `data/raw_pdfs/*.pdf` (text, metadata, images), then concepts, merge, image links, relationships and index. Writes timings to `output/reports/pipeline_run.json`. |
| `python -m src.pipeline ingest FILE.pdf [...]` | Copies new PDFs to `data/uploads/<hash>/`, ingests them, then runs every stage. |
| `python -m src.pipeline concepts` | Concept mention extraction only (resumable, progress bar). |
| `python -m src.pipeline merge` | Rebuilds concept nodes and aliases, writes `output/reports/merge_report.csv`, links images to concepts. |
| `python -m src.pipeline relations` | Rebuilds relationships. |
| `python -m src.pipeline index` | Updates the keyword index. |
| `python -m src.pipeline export` | Writes the Obsidian vault. |
| `python -m src.pipeline metadata` | Detects citation metadata for documents that have none yet. |
| `python -m src.pipeline stats` | Prints collection counts and graph statistics. |
| `python -m src.pipeline --caption-images run` | Same as `run`, also captions images with ViT-GPT2 (off by default). |
| `python -m src.graph.graph_builder` | Prints node count, edge count and top connected concepts. |
| `python -m src.knowledge.concept_extraction` | Concept extraction with options `--document-id`, `--limit-pages`, `--threshold`, `--batch-size`, `--model`. |
| `python src/pdf_extraction.py` | Original extraction step: `data/raw_pdfs` to `data/extracted`. |
| `python -m src.storage.mongodb_store` | Original ingestion step: `data/extracted` to MongoDB. |
| `python -m pytest` | Runs all tests. |

Logs are written to `output/logs/pipeline.log`.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/health`, `GET /api/stats` | Server and database status, counts. |
| `POST /api/documents/upload` | Upload a PDF; processing runs as a background job. |
| `GET /api/documents`, `GET /api/jobs/{id}` | Documents with status and progress. |
| `PUT /api/documents/{id}/metadata` | Edit title, author/editor, year. |
| `GET /api/concepts/autocomplete?q=` | Prefix autocomplete over names and aliases. |
| `GET /api/concepts/search?q=` | Concepts named in the text, then prefix matches. |
| `GET /api/concepts/{id}` | Description, aliases, related concepts with evidence, sources, images. |
| `GET /api/graph?document_id=&type=&min_weight=&limit=&focus=` | Graph elements for Cytoscape. |
| `GET /api/search?q=` | BM25 page search with highlighted snippets and citations. |
| `POST /api/ask` `{"question": "..."}` | Extractive answer with citations, or "Not found in the uploaded documents." |
| `GET /api/pages/{id}`, `GET /api/pages/{id}/image?quote=` | Page text, and the rendered page with the quote highlighted. |
| `GET /api/images/{id}` | An extracted image file. |

## Project structure

```
data/raw_pdfs/              original books (never modified)
data/extracted/             text and .pages.jsonl from the original extraction step
data/uploads/               uploaded PDFs, one folder per file hash (gitignored)
data/images/                extracted images (gitignored)
data/document_metadata.json citation metadata, editable
docs/                       DATA_FLOW.md, REPORT.md, mongodb_storage.md
output/                     obsidian_vault/, reports/, logs/ (gitignored)
src/pdf_extraction.py       PyMuPDF + Tesseract page extraction (unchanged)
src/storage/                MongoDB store and ingestion (unchanged behaviour)
src/ingest/                 upload handling, image extraction
src/knowledge/              concept extraction, merging, relationships
src/search/                 inverted index, BM25, extractive answers, citation validator
src/graph/                  NetworkX graph builder
src/export/                 Obsidian exporter
src/api/                    FastAPI app
src/web/                    HTML, CSS, JavaScript, vendored Cytoscape.js
src/metadata.py             metadata detection and citation format
src/textutil.py             tokenizer, stemmer, sentence spans, name keys
src/pipeline.py             command line pipeline
tests/                      pytest suite (mongomock, mocked models)
```

## Limitations

- Concept types are GLiNER's guesses from a fixed label list. Some are wrong, and some extracted names are still generic or partial.
- Relationship types beyond `related_to` come only from a few English phrase patterns. Most edges are `related_to`, meaning "discussed together", not a specific semantic relation.
- Answers are sentence extracts. They can be partial, and two quoted sentences can come from different contexts. The citation shows where each one comes from.
- Sentence splitting is rule-based. Multi-column pages, tables and broken lines can produce awkward sentence boundaries. Quotes keep the original line breaks and hyphenation of the extracted text.
- Page numbers in citations are PDF page numbers (1-based), not the numbers printed on the pages.
- Metadata detection only reads PDF metadata and the first 8 pages. Check `data/document_metadata.json` and correct it if needed.
- Image understanding is limited to captions and nearby text. Diagram content is not interpreted unless the optional captioner is enabled, and model captions are never used as evidence.
- Concept extraction runs on CPU. It takes about 1 second per page on the development machine (see `docs/REPORT.md`).
- One processing job runs at a time. Job progress is kept in memory and is lost when the server restarts.
- There is no user authentication. The app is meant to run on your own machine.
