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
| `documents` | `_id` and `source_sha256` are the SHA-256 of the original PDF bytes. Stores filename, PDF metadata title/page count, extraction document ID, and local file references to the untouched PDF, TXT, and JSONL. PDF bytes stay on disk (`data/raw_pdfs/` or `data/uploads/`). | `mongodb_store.py` |
| `content_units` | Pages (`unit_type: "page"`): one per JSONL row with the extracted text (never modified), original PyMuPDF text, content type, extraction method, quality/warnings/errors, PDF page number, and asset IDs. Images (`unit_type: "image"`): page, `bbox`, `caption`, `caption_source`, `context_text`, `image_uri`, optional `model_caption`, and `concept_ids` linked through the caption/context text. | `mongodb_store.py`, `ingest/images.py` |
| `assets` | File references and metadata only (no bytes): kind (`source_page` or `image`), caption, page, URI, media type; images also store `bbox`, pixel size, SHA-256 and optional `model_caption` with its model name. | `mongodb_store.py`, `ingest/images.py` |
| `image_state` | One record per document: image extraction version, image count, whether captioning ran. Lets image extraction skip unchanged documents. | `ingest/images.py` |
| `extraction_state` | One record per page: SHA-256 of the page text, extractor version, status (`done`, `skipped_toc`, `skipped_empty`), number of names found. Pages with an unchanged hash are not sent to GLiNER again. | `knowledge/concept_extraction.py` |
| `concept_mentions` | One record per (normalized name, page): `name`, `document_id`, `content_unit_id`, `page_number`, `surface_forms`, `label_scores`, `count`, `max_score`, and `sentences` (start/end offsets into the page text). | `knowledge/concept_extraction.py` |
| `concepts` | Merged concept nodes. `_id` is `stable_id("concept", normalized name of the canonical member)`. Stores `canonical_name`, `normalized_name`, `aliases`, `member_names`, `concept_type`, `description` (verbatim quote with document, page, content unit and offsets), `mentions` (one per page with count, up to 5 verbatim quotes and all `sentence_spans`), `mention_count`, `page_count`, `document_ids`, `image_ids`. Rebuilt from `concept_mentions` on every merge; nodes that are no longer produced are deleted. | `knowledge/merging.py` |
| `concept_aliases` | `_id` is the alias hash key (`concept_key`: lowercase, singular, no spaces or hyphens); stores `concept_id` and the alias text. Used for O(1) concept lookups. | `knowledge/merging.py` |
| `relationships` | One edge per concept pair. `_id` is `stable_id("relationship", id_a, id_b)`. Stores `source_id`, `target_id`, `relation_type` (`related_to`, `is_a`, `part_of`, `used_for`, `causes`), `confidence`, `weight`, `npmi`, `sentence_count`, `window_count`, `evidence` (document, page, content unit, verbatim quote, offsets, kind) and `pattern_matches`. | `knowledge/relationships.py` |
| `page_terms` | One record per page: text SHA-256, token count and stemmed term counts. Only pages with a changed hash are tokenized again. | `search/index.py` |
| `search_index` | Inverted index: `_id` is a stemmed term, with `df` and `postings` (`[content_unit_id, term frequency]`). | `search/index.py` |
| `search_meta` | BM25 corpus statistics: page count, average page length, term and posting counts. | `search/index.py` |
| `semantic_groups` | Group ID, name/description, member concept IDs. Part of the original storage layer; not filled by the pipeline. | `mongodb_store.py` |
| `relationship_evidence` | One evidence-backed assertion with provenance. Part of the original storage layer; the pipeline stores edges with their evidence list in `relationships` instead. | `mongodb_store.py` |

`stable_id(type, *parts)` produces deterministic IDs (SHA-256 of the type and key). Document identity uses the source PDF hash. If the PDF bytes change, the new hash creates a new document version; the previous version is retained rather than silently overwritten.

Citation metadata (title, author/editor, year) is not stored in MongoDB. It lives in `data/document_metadata.json`, keyed by document ID, so it can be edited by hand or from the web app.

## Indexes and idempotency

- Documents have a unique source-hash index and a filename lookup index.
- Content units have a unique `(document_id, page_number, unit_type, unit_index)` index and a document/page lookup index.
- Assets have a unique `(document_id, asset_id)` index and a document/page lookup index.
- Semantic groups index member concept IDs; concepts index canonical name and mention content-unit IDs.
- Relationship evidence indexes source/target/type and provenance content-unit IDs.
- `concept_mentions` indexes `content_unit_id` and `name`; `extraction_state` indexes `document_id`.
- `concept_aliases` indexes `concept_id`; `relationships` indexes `source_id` and `target_id`.

Every collection is written with deterministic `_id` values and upserts, or rebuilt as a whole from its inputs (concepts, aliases, relationships, search index). Running any stage twice gives the same records and no duplicates. Files referenced by the extracted-page JSONL are resolved from the sidecar location; source-page references are checked against the original PDF filename.

## Run ingestion only

From the project root, inside the `.venv`:

```bash
python -m src.storage.mongodb_store --input data/extracted --raw-pdfs data/raw_pdfs --project-root .
```

Set `MONGODB_URI` to an authenticated local or hosted connection string and `MONGODB_DATABASE` to select a database. The default command connects only to local MongoDB. No Atlas cluster or cloud resource is created by this tool. The full pipeline (`python -m src.pipeline run`) calls the same importer and then fills the other collections; see `README.md`.
