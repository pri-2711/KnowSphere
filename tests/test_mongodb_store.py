"""MongoDB ingestion tests using mongomock (no database server required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import mongomock
except ImportError:  # pragma: no cover - actionable test dependency message
    mongomock = None

from src.storage.mongodb_store import MongoKnowledgeStore, stable_id


@unittest.skipIf(mongomock is None, "Install requirements-storage-test.txt to run MongoDB storage tests")
class MongoKnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        self.raw = self.root / "data" / "raw_pdfs"
        self.extracted = self.root / "data" / "extracted"
        self.raw.mkdir(parents=True)
        self.extracted.mkdir(parents=True)
        self.pdf_path = self.raw / "sample.pdf"
        self.pdf_path.write_bytes(b"mock original PDF bytes; importer hashes but never parses them")
        self.database = mongomock.MongoClient()["knowledge_map_test"]
        self.store = MongoKnowledgeStore(self.database, self.root)
        self._write_extraction_outputs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_extraction_outputs(self) -> None:
        self.records = [
            {
                "schema_version": "1.0.0",
                "document_id": "sample",
                "filename": "sample.pdf",
                "page_number": 1,
                "extracted_text": "Page one extracted text.",
                "pymupdf_text": "Page one original text.",
                "content_type": "normal_text",
                "extraction_method": "pymupdf",
                "quality_status": "usable",
                "warnings": [],
                "errors": [],
                "associated_assets": [{
                    "asset_id": "sample-page-1-source",
                    "kind": "source_page",
                    "uri": "../raw_pdfs/sample.pdf#page=1",
                    "document_id": "sample",
                    "page_number": 1,
                }],
                "interpretations": [],
            },
            {
                "schema_version": "1.0.0",
                "document_id": "sample",
                "filename": "sample.pdf",
                "page_number": 2,
                "extracted_text": "Page two extracted text.",
                "pymupdf_text": None,
                "content_type": "unknown",
                "extraction_method": "tesseract",
                "quality_status": "usable",
                "warnings": [],
                "errors": [],
                "associated_assets": [{
                    "asset_id": "sample-page-2-source",
                    "kind": "source_page",
                    "uri": "../raw_pdfs/sample.pdf#page=2",
                    "document_id": "sample",
                    "page_number": 2,
                }],
                "interpretations": [],
            },
        ]
        sidecar = self.extracted / "sample.pages.jsonl"
        sidecar.write_text("".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8")
        (self.extracted / "sample.txt").write_text(
            "Source PDF: sample.pdf\n======================== PAGE 1 ========================\nPage one extracted text.\n"
            "======================== PAGE 2 ========================\nPage two extracted text.\n",
            encoding="utf-8",
        )
        (self.extracted / "extraction_metadata.json").write_text(json.dumps({
            "books": [{"filename": "sample.pdf", "title": "Sample Systems Book", "total_pages": 2}]
        }), encoding="utf-8")

    def test_ingests_text_and_file_references_with_provenance(self) -> None:
        result = self.store.ingest_extracted_directory(self.extracted, self.raw)

        self.assertEqual(result["documents"], {"inserted": 1, "total": 1})
        self.assertEqual(result["content_units"], {"inserted": 2, "total": 2})
        self.assertEqual(result["assets"], {"inserted": 2, "total": 2})
        self.assertEqual(result["sidecars"], 1)

        document = self.database.documents.find_one()
        self.assertEqual(document["title"], "Sample Systems Book")
        self.assertEqual(document["file_storage"]["uri"], "data/raw_pdfs/sample.pdf")
        self.assertEqual(document["extraction_outputs"]["txt_uri"], "data/extracted/sample.txt")

        page_one = self.database.content_units.find_one({"page_number": 1})
        self.assertEqual(page_one["text"], "Page one extracted text.")
        self.assertEqual(page_one["original_text"], "Page one original text.")
        self.assertEqual(page_one["source"]["pdf_page"], 1)
        self.assertEqual(len(page_one["asset_ids"]), 1)
        asset = self.database.assets.find_one({"page_number": 1})
        self.assertEqual(asset["storage"]["uri"], "data/raw_pdfs/sample.pdf")
        self.assertNotIn("bytes", asset)

    def test_repeated_ingestion_is_idempotent(self) -> None:
        self.store.ingest_extracted_directory(self.extracted, self.raw)
        second = self.store.ingest_extracted_directory(self.extracted, self.raw)

        self.assertEqual(second["documents"]["inserted"], 0)
        self.assertEqual(second["content_units"]["inserted"], 0)
        self.assertEqual(second["assets"]["inserted"], 0)
        self.assertEqual(self.database.documents.count_documents({}), 1)
        self.assertEqual(self.database.content_units.count_documents({}), 2)
        self.assertEqual(self.database.assets.count_documents({}), 2)

    def test_future_concepts_and_relationships_keep_evidence_references(self) -> None:
        concept_a = stable_id("concept", "system requirement")
        concept_b = stable_id("concept", "system architecture")
        self.store.upsert_concept(concept_a, {"canonical_name": "system requirement"})
        self.store.upsert_concept(concept_b, {"canonical_name": "system architecture"})
        group_id = stable_id("semantic_group", "systems engineering")
        self.store.upsert_semantic_group(group_id, {
            "name": "Systems Engineering",
            "concept_ids": [concept_a, concept_b],
        })
        evidence_fields = {
            "source_concept_id": concept_a,
            "target_concept_id": concept_b,
            "relation_type": "informs",
            "provenance": {
                "document_id": "source-hash",
                "content_unit_id": "page-unit-id",
                "page_number": 4,
                "quote": "Requirements inform the architecture.",
            },
        }
        evidence_id = stable_id("relationship_evidence", concept_a, concept_b, "page-unit-id", "quote")
        self.assertTrue(self.store.upsert_relationship_evidence(evidence_id, evidence_fields))
        self.assertFalse(self.store.upsert_relationship_evidence(evidence_id, evidence_fields))
        evidence = self.database.relationship_evidence.find_one({"_id": evidence_id})
        self.assertEqual(evidence["provenance"]["page_number"], 4)
        self.assertEqual(self.database.semantic_groups.count_documents({"_id": group_id}), 1)
        self.assertEqual(self.database.concepts.count_documents({}), 2)


if __name__ == "__main__":
    unittest.main()
