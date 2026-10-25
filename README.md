# Knowledge Map

Knowledge Map turns PDF books into a knowledge map you can explore and question. It extracts text and images from each page, finds concepts, merges concepts with the same meaning into one node, connects concepts that appear together, and answers questions by quoting sentences from the documents. Every answer sentence, concept description and relationship carries a citation:

```
[Document Title | Author/Editor | Year of Publication, p. N]
```

Answers are extractive. The system never writes its own text and calls no external LLM. If the documents do not contain enough evidence, the answer is exactly: `Not found in the uploaded documents.`

## Features

- **Add, edit, delete documents**: upload a PDF, replace it with a new version, edit its citation metadata, or delete it with everything derived from it. Concepts, relationships and the index are rebuilt after every change.
- **Watched folder**: PDFs added to, changed in or removed from `data/inbox/` go through the whole pipeline automatically while the app (or `python -m src.pipeline watch`) runs.
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
python -m src.pipeline run

# 2. Start the web app, then open http://localhost:8000
uvicorn src.api.app:app --port 8000

# 3. Optional: write the Obsidian vault to output/obsidian_vault/
python -m src.pipeline export
```

`python -m src.pipeline run` is safe to repeat. Unchanged documents and pages are skipped by their SHA-256 hash, so a second run only redoes the fast stages (merging, relationships, index).

## Adding, changing and deleting documents

| What you want | In the web app (Documents screen) | Watched folder `data/inbox/` | Command line |
|---|---|---|---|
| Add a PDF | Drag and drop, or click to upload | Copy the PDF into the folder | `python -m src.pipeline ingest FILE.pdf` |
| New version of a PDF | "Replace file" on its row | Overwrite the file with the new version | Delete, then ingest |
| Fix title, author/editor, year | Edit the fields, then "Save" | | Edit `data/document_metadata.json` |
| Delete | "Delete", then "Confirm delete" | Remove the file from the folder | `python -m src.pipeline delete ID_OR_FILENAME` |

What happens:

- **Add**: text, images and metadata are extracted, then concepts, merging, image links, relationships and the index are updated. Pages already processed (same text hash) are not processed again.
- **Replace**: the new version is added, metadata fields you edited are copied to it, and the old version is deleted.
- **Delete**: the document's pages, images, image files, concept mentions, index entries and metadata entry are removed, then concepts, relationships and the index are rebuilt. Concepts that only came from this document disappear. Files in `data/raw_pdfs/` are never deleted. A document deleted in the app is remembered (`deleted_documents`), so `pipeline run` and the folder watcher do not add the same file back. Uploading it again on purpose adds it back.
- **Watched folder**: the app scans `data/inbox/` every 5 seconds (change with `KM_WATCH_DIR`, `KM_WATCH_INTERVAL`; turn off with `KM_WATCH_ENABLED=0` in `.env`). A file is processed once its size and time have not changed for one scan, so files still being copied are not read. If the same PDF is also uploaded or in `data/raw_pdfs/`, removing it from the folder keeps the document.

Each change runs as one background job. Jobs run one at a time, and the Documents screen shows their progress.

## CLI reference

| Command | What it does |
|---|---|
| `python -m src.pipeline run` | Ingests `data/raw_pdfs/*.pdf` (text, metadata, images), then concepts, merge, image links, relationships and index. Writes timings to `output/reports/pipeline_run.json`. |
| `python -m src.pipeline ingest FILE.pdf [...]` | Copies new PDFs to `data/uploads/<hash>/`, ingests them, then runs every stage. |
| `python -m src.pipeline watch [--folder DIR] [--interval S]` | Watches a folder (default `data/inbox/`) without the web app and processes added, changed and removed PDFs. |
| `python -m src.pipeline documents` | Lists documents with ID, pages and sources. |
| `python -m src.pipeline delete ID_OR_FILENAME [...]` | Deletes documents (ID, ID prefix or filename), then rebuilds. |
| `python -m src.envcheck` | Checks the interpreter and package versions against `requirements.txt`. |
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

## Project structure

```
data/raw_pdfs/              original books (never modified)
data/extracted/             text and .pages.jsonl from the original extraction step
data/inbox/                 watched folder: add, change or remove PDFs here
data/uploads/               copies of uploaded and watched PDFs, one folder per file hash (gitignored)
data/images/                extracted images (gitignored)
data/document_metadata.json citation metadata, editable
docs/                       DATA_FLOW.md, REPORT.md, mongodb_storage.md
output/                     obsidian_vault/, reports/, logs/ (gitignored)
src/pdf_extraction.py       PyMuPDF + Tesseract page extraction (unchanged)
src/storage/                MongoDB store and ingestion (unchanged behaviour)
src/ingest/                 add/replace/delete documents, folder watcher, image extraction
src/envcheck.py             interpreter and package version check
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
- Concept extraction runs on CPU: 1.4 to 3.5 seconds per page on the development machine depending on other load, so about 16 minutes for the two books without other load (see `docs/REPORT.md`).
- One processing job runs at a time. Job progress is kept in memory and is lost when the server restarts. The documents themselves and the watch state are stored in MongoDB.
- Every add, replace or delete rebuilds concept merging, relationships and the index for the whole collection. With the two books this took 8 to 31 seconds per change after the models were loaded (see `docs/REPORT.md`), and it grows with the collection.
- The folder watcher polls; changes are picked up within about two scan intervals (10 seconds by default), not instantly.
- On two-column pages the original extraction (PyMuPDF `sort=True`) can interleave the columns, so some quoted sentences mix text from both columns. Quotes are still verbatim and correctly cited.
- There is no user authentication. The app is meant to run on your own machine.
