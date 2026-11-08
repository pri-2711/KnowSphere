# MongoDB-first storage

## Development deployment

The development setup uses local MongoDB Community Server and the synchronous PyMongo driver. It requires no cloud cluster, subscription, container, or API calls. The current development machine has MongoDB Server 8.2.6 running locally; the project `.venv` pins PyMongo 4.19.0 in `requirements.txt`. MongoDB describes Community Edition as free for local self-managed development; its server is licensed under SSPL, so review that license before offering a modified deployment as a service.

The default target is `mongodb://127.0.0.1:27017`, database `knowledge_map`. Override the connection through environment variables or a `.env` file in the project root (gitignored); never put credentials in source code or commit them:

```powershell
$env:MONGODB_URI = 'mongodb://username:password@host:27017/?authSource=admin'
$env:MONGODB_DATABASE = 'knowledge_map'
```

For the local no-auth service, no environment variables are required. PyMongo reads the connection URI and creates a client as documented in the [official driver guide](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient/).

All dependencies, including `mongomock` for the tests, are installed into the project virtual environment with `python -m pip install -r requirements.txt` (see `README.md`). The older `requirements-storage.txt` and `requirements-storage-test.txt` still work for the storage layer alone.

## Collections

| Collection | Stored fields and purpose | Written by |
|---|---|---|
| `documents` | `_id` and `source_sha256` are the SHA-256 of the original PDF bytes. Stores filename, PDF metadata title/page count, extraction document ID, local file references to the untouched PDF, TXT, and JSONL, and `sources` (`upload`, `raw_pdfs/<file>`, `watch:<path>`). PDF bytes stay on disk (`data/raw_pdfs/` or `data/uploads/`). Deleting a document removes every record that carries its ID. | `mongodb_store.py` |
| `deleted_documents` | Tombstones: `_id` is the SHA-256 of a document deleted in the app, with filename and time. `pipeline run` and the folder watcher skip these files; an explicit upload removes the tombstone. | `ingest/documents.py` |
| `watched_files` | One record per PDF in the watched folder: `_id` is its path relative to the folder, with `sha256`, `size`, `mtime_ns`, `document_id`. Used to detect added, changed and removed files. | `ingest/watcher.py` |
| `content_units` | Pages (`unit_type: "page"`): one per JSONL row with the extracted text (never modified), original PyMuPDF text, content type, extraction method, quality/warnings/errors, PDF page number, and asset IDs. Images (`unit_type: "image"`): page, `bbox`, `caption`, `caption_source`, `context_text`, `image_uri`, optional `model_caption`, and `concept_ids` linked through the caption/context text. Semantic chunks (`unit_type: "chunk"`, concept-first pipeline): `_id` is `stable_id("content_unit", document_id, "chunk", first page unit, start offset)`, `unit_index` is the start offset on the first page, plus `chunk_index`, `page_number`/`page_end`, `text`, `text_sha256`, `section`, `kind` (`text` or `toc`), `spans` (chunk offsets mapped to page unit and page offsets), `chunker_version`, and `extraction` (Pass 1 result: status, prompt version, model, every attempt with its raw answer and error, validated local concepts and relationships). | `mongodb_store.py`, `ingest/images.py`, `kg/store.py` |
| `assets` | File references and metadata only (no bytes): kind (`source_page` or `image`), caption, page, URI, media type; images also store `bbox`, pixel size, SHA-256 and optional `model_caption` with its model name. | `mongodb_store.py`, `ingest/images.py` |
| `image_state` | One record per document: image extraction version, image count, whether captioning ran. Lets image extraction skip unchanged documents. | `ingest/images.py` |
| `extraction_state` | One record per page: SHA-256 of the page text, extractor version, status (`done`, `skipped_toc`, `skipped_empty`), number of names found. Pages with an unchanged hash are not sent to GLiNER again. | `knowledge/concept_extraction.py` |
| `concept_mentions` | One record per (normalized name, page): `name`, `document_id`, `content_unit_id`, `page_number`, `surface_forms`, `label_scores`, `count`, `max_score`, and `sentences` (start/end offsets into the page text). | `knowledge/concept_extraction.py` |
| `concepts` | Merged concept nodes. `_id` is `stable_id("concept", normalized name of the canonical member)`. Stores `canonical_name`, `normalized_name`, `aliases`, `member_names`, `concept_type`, `description` (verbatim quote with document, page, content unit and offsets), `mentions` (one per page with count, up to 5 verbatim quotes and all `sentence_spans`), `mention_count`, `page_count`, `document_ids`, `image_ids`. Rebuilt from `concept_mentions` on every merge; nodes that are no longer produced are deleted. | `knowledge/merging.py` |
| `concept_aliases` | `_id` is the alias hash key (`concept_key`: lowercase, singular, no spaces or hyphens); stores `concept_id` and the alias text. Used for O(1) concept lookups. | `knowledge/merging.py` |
| `kg_concepts` | Concept-first pipeline: one canonical concept. `_id` is `stable_id("kg_concept", normalized name)`. Stores `canonical_name`, `normalized_name`, `concept_type`, `aliases`, `description` (LLM-written, marked `generated: true`, with the chunk and evidence it came from), `mentions` (every contributing passage: chunk, document, page, name as written, description, evidence quote with page offsets, status), `mention_count`, `document_ids`, `status`, `merged_from`. Rebuilt from the stored chunk extractions. | `kg/store.py` |
| `kg_relationships` | Concept-first pipeline: `_id` is `stable_id("kg_relationship", source_id, relation_type, target_id)`. Stores `source_id`, `target_id`, `relation_type` (`is_a`, `includes`, `uses`, `defines`, `prerequisite_for`, `example_of`, `related_to`), `status`, `evidence` (every supporting passage: chunk, document, page, quote with offsets, status), `evidence_count`, `document_ids`, `cross_document`. | `kg/store.py` |
| `relationships` | One edge per concept pair. `_id` is `stable_id("relationship", id_a, id_b)`. Stores `source_id`, `target_id`, `relation_type` (`related_to`, `is_a`, `part_of`, `used_for`, `causes`), `confidence`, `weight`, `npmi`, `sentence_count`, `window_count`, `evidence` (document, page, content unit, verbatim quote, offsets, kind) and `pattern_matches`. | `knowledge/relationships.py` |
| `page_terms` | One record per page: text SHA-256, token count and stemmed term counts. Only pages with a changed hash are tokenized again. | `search/index.py` |
| `search_index` | Inverted index: `_id` is a stemmed term, with `df` and `postings` (`[content_unit_id, term frequency]`). | `search/index.py` |
| `search_meta` | BM25 corpus statistics: page count, average page length, term and posting counts. | `search/index.py` |
| `semantic_groups` | Group ID, name/description, member concept IDs. Part of the original storage layer; not filled by the pipeline. | `mongodb_store.py` |
| `relationship_evidence` | One evidence-backed assertion with provenance. Part of the original storage layer; the pipeline stores edges with their evidence list in `relationships` instead. | `mongodb_store.py` |

### Projects

| Collection | Stored fields and purpose | Written by |
|---|---|---|
| `projects` | `_id` (`"default"` or a 12-character hex ID), `name` (unique, case-insensitive), `created_at`, `updated_at`. The default project is created automatically and cannot be deleted. | `projects.py` |

Every collection above except `projects` and the unused `semantic_groups` / `relationship_evidence` stores `project_id`. Records that are derived from a name or a file hash use project-scoped keys so that two projects never share a record: document `_id` (`document_id_for`: the PDF SHA-256 in the default project, a hash of project and PDF hash elsewhere; `source_sha256` always holds the plain file hash), concept, KG concept and KG relationship IDs (`scoped_id`), and `concept_aliases`, `search_index` and `search_meta` keys (`scoped_key`: `"<project>|<key>"`; plain key in the default project). Alias and index records also store the plain `key` / `term`. The default project keeps exactly the IDs used before projects existed.

Migration (`projects.migrate`, run when the API or a pipeline command starts) only adds `project_id: "default"` to records that have no `project_id`. Queries for the default project also match records without the field, so nothing is hidden before the migration runs.

`stable_id(type, *parts)` produces deterministic IDs (SHA-256 of the type and key). Document identity uses the source PDF hash. If the PDF bytes change, the new hash creates a new document version; the previous version is retained rather than silently overwritten.

Citation metadata (title, author/editor, year) is not stored in MongoDB. It lives in `data/document_metadata.json`, keyed by document ID, so it can be edited by hand or from the web app.

## Indexes and idempotency

- Documents have a unique `(project_id, source_sha256)` index (`uq_project_document_hash`; it replaces the earlier global `uq_document_hash`, which is dropped on start so the same PDF can be in several projects) and a filename lookup index.
- Content units have a unique `(document_id, page_number, unit_type, unit_index)` index and a document/page lookup index.
- Assets have a unique `(document_id, asset_id)` index and a document/page lookup index.
- Semantic groups index member concept IDs; concepts index canonical name and mention content-unit IDs.
- Relationship evidence indexes source/target/type and provenance content-unit IDs.
- `concept_mentions` indexes `content_unit_id` and `name`; `extraction_state` indexes `document_id`.
- `concept_aliases` indexes `concept_id`; `relationships` indexes `source_id` and `target_id`.
- `content_units` also indexes `(unit_type, document_id)` for chunk lookups; `kg_concepts` indexes `normalized_name`, `document_ids` and `mentions.chunk_id`; `kg_relationships` indexes `source_id`, `target_id` and `relation_type`.

`kg_concepts` and `kg_relationships` are separate from `concepts` and `relationships` on purpose: the GLiNER stage rebuilds its two collections as a whole and deletes records it did not produce, so sharing them would make each pipeline erase the other.

Every collection is written with deterministic `_id` values and upserts, or rebuilt as a whole from its inputs (concepts, aliases, relationships, search index). Running any stage twice gives the same records and no duplicates. Files referenced by the extracted-page JSONL are resolved from the sidecar location; source-page references are checked against the original PDF filename.

## Run ingestion only

From the project root, inside the `.venv`:

```bash
python -m src.storage.mongodb_store --input data/extracted --raw-pdfs data/raw_pdfs --project-root .
```

Set `MONGODB_URI` to an authenticated local or hosted connection string and `MONGODB_DATABASE` to select a database. The default command connects only to local MongoDB. No Atlas cluster or cloud resource is created by this tool. The full pipeline (`python -m src.pipeline run`) calls the same importer and then fills the other collections; see `README.md`.
