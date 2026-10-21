"""MongoDB-first storage for extracted Knowledge Map content.

This module stores document metadata, page text, and references to files. It
never stores PDF/image bytes and does not perform extraction or AI inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

try:
    from pymongo import ASCENDING, MongoClient
    from pymongo.database import Database
except ImportError:  # pragma: no cover - depends on user's storage setup
    ASCENDING = 1
    MongoClient = None
    Database = Any


COLLECTIONS = (
    "documents",
    "content_units",
    "assets",
    "semantic_groups",
    "concepts",
    "relationship_evidence",
)


def stable_id(entity_type: str, *parts: Any) -> str:
    """Create a deterministic MongoDB-safe ID from an entity type and key."""
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{entity_type}:{digest}"


def sha256_file(path: Path) -> str:
    """Hash an original file without opening or parsing its PDF pages."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file_uri(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_uri()


class MongoKnowledgeStore:
    """Idempotent document/page ingestion and future concept storage APIs."""

    def __init__(self, database: Database, project_root: Path | None = None) -> None:
        self.database = database
        self.project_root = (project_root or Path.cwd()).resolve()
        self.documents = database["documents"]
        self.content_units = database["content_units"]
        self.assets = database["assets"]
        self.semantic_groups = database["semantic_groups"]
        self.concepts = database["concepts"]
        self.relationship_evidence = database["relationship_evidence"]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        """Create unique identity and common provenance/relationship indexes."""
        self.documents.create_index([("source_sha256", ASCENDING)], unique=True, name="uq_document_hash")
        self.documents.create_index([("filename", ASCENDING)], name="ix_document_filename")

        self.content_units.create_index(
            [("document_id", ASCENDING), ("page_number", ASCENDING),
             ("unit_type", ASCENDING), ("unit_index", ASCENDING)],
            unique=True,
            name="uq_content_provenance_unit",
        )
        self.content_units.create_index(
            [("document_id", ASCENDING), ("page_number", ASCENDING)], name="ix_content_page"
        )

        self.assets.create_index(
            [("document_id", ASCENDING), ("asset_id", ASCENDING)],
            unique=True,
            name="uq_asset_document_key",
        )
        self.assets.create_index(
            [("document_id", ASCENDING), ("page_number", ASCENDING)], name="ix_asset_page"
        )

        self.semantic_groups.create_index([("concept_ids", ASCENDING)], name="ix_group_concepts")
        self.concepts.create_index([("canonical_name", ASCENDING)], name="ix_concept_name")
        self.relationship_evidence.create_index(
            [("source_concept_id", ASCENDING), ("target_concept_id", ASCENDING),
             ("relation_type", ASCENDING)],
            name="ix_relationship_endpoints_type",
        )
        self.relationship_evidence.create_index(
            [("provenance.content_unit_id", ASCENDING)], name="ix_relationship_evidence_unit"
        )

    @staticmethod
    def _upsert(collection: Any, record: Mapping[str, Any]) -> bool:
        """Write by stable _id; return True only when a record was inserted."""
        if not record.get("_id"):
            raise ValueError("Every stored entity must have a stable _id.")
        payload = dict(record)
        entity_id = payload.pop("_id")
        result = collection.update_one(
            {"_id": entity_id},
            {
                "$set": payload,
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
        return result.upserted_id is not None

    def upsert_semantic_group(self, group_id: str, fields: Mapping[str, Any]) -> bool:
        """Upsert a manually or later-computed semantic group using a stable ID."""
        if not group_id:
            raise ValueError("group_id is required; create it with stable_id().")
        record = dict(fields)
        record["_id"] = group_id
        record.setdefault("concept_ids", [])
        return self._upsert(self.semantic_groups, record)

    def upsert_concept(self, concept_id: str, fields: Mapping[str, Any]) -> bool:
        """Upsert a concept; this method does not generate or infer concepts."""
        if not concept_id:
            raise ValueError("concept_id is required; create it with stable_id().")
        record = dict(fields)
        if not record.get("canonical_name"):
            raise ValueError("A concept requires canonical_name.")
        record["_id"] = concept_id
        record.setdefault("aliases", [])
        record.setdefault("evidence_ids", [])
        return self._upsert(self.concepts, record)

    def upsert_relationship_evidence(self, evidence_id: str, fields: Mapping[str, Any]) -> bool:
        """Store one sourced relationship assertion, not an inferred graph edge."""
        if not evidence_id:
            raise ValueError("evidence_id is required; create it with stable_id().")
        required = ("source_concept_id", "target_concept_id", "relation_type", "provenance")
        missing = [key for key in required if not fields.get(key)]
        if missing:
            raise ValueError(f"Relationship evidence is missing: {', '.join(missing)}")
        provenance = fields["provenance"]
        for key in ("document_id", "content_unit_id", "page_number", "quote"):
            if key not in provenance:
                raise ValueError(f"Relationship provenance is missing {key}.")
        record = dict(fields)
        record["_id"] = evidence_id
        return self._upsert(self.relationship_evidence, record)

    def ingest_extracted_directory(
        self,
        extracted_dir: Path,
        raw_pdf_dir: Path,
        metadata_path: Path | None = None,
    ) -> dict[str, Any]:
        """Ingest existing ``*.pages.jsonl`` sidecars without reprocessing PDFs."""
        extracted_dir = extracted_dir.resolve()
        raw_pdf_dir = raw_pdf_dir.resolve()
        if not extracted_dir.is_dir():
            raise NotADirectoryError(f"Extraction directory does not exist: {extracted_dir}")
        if not raw_pdf_dir.is_dir():
            raise NotADirectoryError(f"Raw PDF directory does not exist: {raw_pdf_dir}")

        metadata_file = metadata_path.resolve() if metadata_path else extracted_dir / "extraction_metadata.json"
        books_by_filename: dict[str, dict[str, Any]] = {}
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            books_by_filename = {book["filename"]: book for book in metadata.get("books", [])}

        summary: dict[str, Any] = {
            "documents": {"inserted": 0, "total": 0},
            "content_units": {"inserted": 0, "total": 0},
            "assets": {"inserted": 0, "total": 0},
            "sidecars": 0,
        }
        sidecars = sorted(extracted_dir.glob("*.pages.jsonl"), key=lambda path: path.name.casefold())
        for sidecar in sidecars:
            records = self._read_sidecar(sidecar)
            if not records:
                continue
            filename = records[0]["filename"]
            if any(record.get("filename") != filename for record in records):
                raise ValueError(f"Sidecar {sidecar.name} contains mixed filenames.")
            source_pdf = raw_pdf_dir / filename
            if not source_pdf.is_file():
                raise FileNotFoundError(f"Source PDF for {sidecar.name} is missing: {source_pdf}")
            sidecar_stem = sidecar.name[:-len(".pages.jsonl")]
            if sidecar_stem.casefold() != Path(filename).stem.casefold():
                raise ValueError(f"Sidecar {sidecar.name} does not match source filename {filename!r}.")
            document_id = sha256_file(source_pdf)
            extraction_doc_id = records[0].get("document_id")
            if any(record.get("document_id") != extraction_doc_id for record in records):
                raise ValueError(f"Sidecar {sidecar.name} contains mixed document IDs.")

            book_meta = books_by_filename.get(filename, {})
            document_record = {
                "_id": document_id,
                "document_id": document_id,
                "extraction_document_id": extraction_doc_id,
                "source_sha256": document_id,
                "filename": filename,
                "title": book_meta.get("title"),
                "page_count": book_meta.get("total_pages") or len(records),
                "file_storage": {
                    "provider": "local",
                    "uri": _relative_file_uri(source_pdf, self.project_root),
                    "media_type": "application/pdf",
                },
                "extraction_outputs": {
                    "txt_uri": _relative_file_uri(extracted_dir / f"{Path(filename).stem}.txt", self.project_root),
                    "pages_jsonl_uri": _relative_file_uri(sidecar, self.project_root),
                    "schema_version": records[0].get("schema_version"),
                },
            }
            summary["documents"]["inserted"] += int(self._upsert(self.documents, document_record))
            summary["documents"]["total"] += 1

            page_numbers: set[int] = set()
            for page_record in records:
                page_number = page_record.get("page_number")
                if not isinstance(page_number, int) or page_number < 1:
                    raise ValueError(f"Invalid page_number in {sidecar.name}: {page_number!r}")
                if page_number in page_numbers:
                    raise ValueError(f"Duplicate page {page_number} in {sidecar.name}.")
                page_numbers.add(page_number)
                if not isinstance(page_record.get("extracted_text"), str):
                    raise ValueError(f"Missing extracted_text for page {page_number} in {sidecar.name}.")

                asset_ids: list[str] = []
                for asset_ref in page_record.get("associated_assets", []):
                    asset_record = self._asset_record(asset_ref, page_record, document_id, sidecar)
                    asset_ids.append(asset_record["_id"])
                    summary["assets"]["inserted"] += int(self._upsert(self.assets, asset_record))
                    summary["assets"]["total"] += 1

                unit_id = stable_id("content_unit", document_id, "page", page_number)
                unit_record = {
                    "_id": unit_id,
                    "document_id": document_id,
                    "source_document_id": extraction_doc_id,
                    "unit_type": "page",
                    "unit_index": page_number,
                    "page_number": page_number,
                    "text": page_record["extracted_text"],
                    "original_text": page_record.get("pymupdf_text"),
                    "content_type": page_record.get("content_type", "unknown"),
                    "extraction_method": page_record.get("extraction_method", "none"),
                    "quality_status": page_record.get("quality_status", "review"),
                    "warnings": page_record.get("warnings", []),
                    "errors": page_record.get("errors", []),
                    "asset_ids": asset_ids,
                    "source": {
                        "filename": filename,
                        "pdf_page": page_number,
                        "pdf_uri": document_record["file_storage"]["uri"] + f"#page={page_number}",
                    },
                }
                summary["content_units"]["inserted"] += int(
                    self._upsert(self.content_units, unit_record)
                )
                summary["content_units"]["total"] += 1
            summary["sidecars"] += 1
        return summary

    @staticmethod
    def _read_sidecar(sidecar: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with sidecar.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {sidecar.name}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {sidecar.name}:{line_number}.")
                for key in ("filename", "document_id", "page_number", "extracted_text"):
                    if key not in record:
                        raise ValueError(f"Missing {key} in {sidecar.name}:{line_number}.")
                records.append(record)
        return records

    def _asset_record(
        self,
        asset_ref: Mapping[str, Any],
        page_record: Mapping[str, Any],
        document_id: str,
        sidecar: Path,
    ) -> dict[str, Any]:
        for key in ("asset_id", "kind", "uri"):
            if not asset_ref.get(key):
                raise ValueError(f"Asset reference is missing {key}.")
        parsed = urlsplit(str(asset_ref["uri"]))
        asset_path: Path | None = None
        storage_uri = str(asset_ref["uri"])
        if not parsed.scheme:
            asset_path = (sidecar.parent / unquote(parsed.path)).resolve()
            storage_uri = _relative_file_uri(asset_path, self.project_root)
        elif parsed.scheme == "file":
            asset_path = Path(unquote(parsed.path)).resolve()
            storage_uri = _relative_file_uri(asset_path, self.project_root)
        if asset_ref["kind"] == "source_page":
            if asset_path is None or not asset_path.is_file():
                raise FileNotFoundError(f"Source-page asset is missing: {asset_ref['uri']}")
            if asset_path.name != page_record["filename"]:
                raise ValueError(f"Source-page asset does not match {page_record['filename']!r}.")
        asset_id = stable_id("asset", document_id, asset_ref["asset_id"])
        return {
            "_id": asset_id,
            "asset_id": asset_ref["asset_id"],
            "document_id": document_id,
            "source_document_id": asset_ref.get("document_id", page_record["document_id"]),
            "page_number": asset_ref.get("page_number", page_record["page_number"]),
            "kind": asset_ref["kind"],
            "caption": asset_ref.get("caption"),
            "source_uri": str(asset_ref["uri"]),
            "storage": {
                "provider": "local" if asset_path is not None else "external_reference",
                "uri": storage_uri,
                "media_type": mimetypes.guess_type(storage_uri)[0],
            },
        }


def connect_from_environment() -> tuple[Any, Any]:
    """Connect with env-provided settings; defaults target local development."""
    if MongoClient is None:
        raise RuntimeError("PyMongo is missing. Install the MongoDB storage requirements first.")
    uri = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017")
    database_name = os.environ.get("MONGODB_DATABASE", "knowledge_map")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        raise
    return client, client[database_name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest existing extraction JSONL into MongoDB.")
    parser.add_argument("--input", type=Path, default=Path("data/extracted"), help="Extraction output directory")
    parser.add_argument("--raw-pdfs", type=Path, default=Path("data/raw_pdfs"), help="Original PDF directory")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Root for stored file references")
    args = parser.parse_args(argv)
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database, args.project_root)
        summary = store.ingest_extracted_directory(args.input, args.raw_pdfs)
        print(f"MongoDB database: {database.name}")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"MongoDB ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
