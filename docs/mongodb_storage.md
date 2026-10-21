# MongoDB-first storage

## Development deployment

The development setup uses local MongoDB Community Server and the synchronous PyMongo driver. It requires no cloud cluster, subscription, container, or API calls. The current development machine has MongoDB Server 8.2.6 running locally and PyMongo 4.17.0 installed. MongoDB describes Community Edition as free for local self-managed development; its server is licensed under SSPL, so review that license before offering a modified deployment as a service.

The default target is `mongodb://127.0.0.1:27017`, database `knowledge_map`. Override the connection through environment variables; never put credentials in source code or commit them:

```powershell
$env:MONGODB_URI = 'mongodb://username:password@host:27017/?authSource=admin'
$env:MONGODB_DATABASE = 'knowledge_map'
```

For the local no-auth service, no environment variables are required. PyMongo reads the connection URI and creates a client as documented in the [official driver guide](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient/).

Install the focused runtime dependency when preparing another environment:

```bash
python -m pip install -r requirements-storage.txt
```

The unit tests use `mongomock` and can be prepared with:

```bash
python -m pip install -r requirements-storage-test.txt
```

## Collections

| Collection | Stored fields and purpose |
|---|---|
| `documents` | `_id` and `source_sha256` are the SHA-256 of the original PDF bytes. Stores filename, PDF metadata title/page count, extraction document ID, and local file references to the untouched PDF, TXT, and JSONL. PDF bytes stay in `data/raw_pdfs/`. |
| `content_units` | One page unit per JSONL row. Stores the selected extracted text, original PyMuPDF text, page content type, extraction method, quality/warnings/errors, stable document ID, PDF page number, and associated asset IDs. |
| `assets` | Stable file references and metadata only: kind, caption, document/page provenance, URI, and media type. Image/PDF bytes are not embedded in MongoDB. The current extraction run has source-page references; separate figure files can be added later. |
| `semantic_groups` | Future group ID, name/description, member concept IDs, and caller-supplied properties. No groups are generated during ingestion. |
| `concepts` | Future stable concept ID, canonical name, aliases, description, evidence IDs, and caller-supplied properties. No concepts are extracted during ingestion. |
| `relationship_evidence` | One evidence-backed assertion with source/target concept IDs, relation type, and provenance (`document_id`, `content_unit_id`, page number, quote). This stores evidence, not a Neo4j edge. |

`stable_id(type, *parts)` produces deterministic IDs for concepts, groups, evidence, and content units. Document identity uses the source PDF hash. If the PDF bytes change, the new hash creates a new document version; the previous version is retained rather than silently overwritten.

## Indexes and idempotency

- Documents have a unique source-hash index and a filename lookup index.
- Content units have a unique `(document_id, page_number, unit_type, unit_index)` index and a document/page lookup index.
- Assets have a unique `(document_id, asset_id)` index and a document/page lookup index.
- Semantic groups index member concept IDs; concepts index canonical name.
- Relationship evidence indexes source/target/type and provenance content-unit IDs.

Documents, page units, and assets are written with deterministic `_id` values and upserts. Re-running ingestion updates the same records instead of inserting copies. Files referenced by the extracted-page JSONL are resolved from the sidecar location; source-page references are checked against the original PDF filename. Only the PDF file hash is read from disk; PDF pages are not re-extracted.

## Run ingestion

From the project root:

```bash
python -m src.storage.mongodb_store --input data/extracted --raw-pdfs data/raw_pdfs --project-root .
```

Set `MONGODB_URI` to an authenticated local or hosted connection string and `MONGODB_DATABASE` to select a database. The default command connects only to local MongoDB. No Atlas cluster or cloud resource is created by this tool.

## Current boundary

The importer consumes existing `*.pages.jsonl`, `extraction_metadata.json`, and the referenced PDF/TXT files. It does not run PDF extraction, embeddings, semantic search, AI concept extraction, or graph construction. Those later stages can call the store's explicit upsert methods and attach their outputs to the existing content-unit and evidence IDs.
