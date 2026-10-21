"""Focused regression tests for the page-aware PDF extraction sidecar."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import pdf_extraction


@unittest.skipIf(pdf_extraction.fitz is None, "PyMuPDF is required by the extractor")
class PdfExtractionSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "raw_pdfs"
        self.output_dir = self.root / "extracted"
        self.input_dir.mkdir()
        self.pdf_path = self.input_dir / "sample_book.pdf"
        self._create_sample_pdf()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_sample_pdf(self) -> None:
        fitz = pdf_extraction.fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Part 1")
        page.insert_text((72, 100), "Systems Engineering Practice")

        scanned_page = doc.new_page()
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
        pixmap.clear_with(255)
        scanned_page.insert_image(scanned_page.rect, pixmap=pixmap)

        doc.new_page()  # A page with no embedded text or raster image.
        doc.set_metadata({"title": "Sample Book"})
        doc.save(self.pdf_path)
        doc.close()

    @staticmethod
    def _read_txt_page(text: str, number: int) -> str:
        match = re.search(
            rf"^={{24}} PAGE {number} ={{24}}\n(.*?)(?=^={{24}} PAGE |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"TXT page separator for page {number} was not found")
        return match.group(1).strip()

    def test_adds_page_jsonl_and_keeps_legacy_outputs(self) -> None:
        fake_tesseract = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=""))
        ocr_text = "Recovered scanned text with enough meaningful words to pass quality review."
        with (
            patch.object(pdf_extraction, "pytesseract", fake_tesseract),
            patch.object(pdf_extraction, "Image", object()),
            patch.object(pdf_extraction, "_resolve_tesseract", return_value="mock-tesseract"),
            patch.object(pdf_extraction, "_ocr_page", return_value=ocr_text) as ocr_mock,
        ):
            result = pdf_extraction.run(self.input_dir, self.output_dir)

        txt_path = self.output_dir / "sample_book.txt"
        metadata_path = self.output_dir / "extraction_metadata.json"
        report_path = self.output_dir / "extraction_report.txt"
        jsonl_path = self.output_dir / "sample_book.pages.jsonl"
        for path in (txt_path, metadata_path, report_path, jsonl_path):
            self.assertTrue(path.is_file(), f"Expected output file: {path.name}")

        legacy_txt = txt_path.read_text(encoding="utf-8")
        self.assertEqual(legacy_txt.count("PAGE "), 3)
        self.assertEqual(self._read_txt_page(legacy_txt, 1), "Part 1\n\nSystems Engineering Practice")
        self.assertEqual(self._read_txt_page(legacy_txt, 2), ocr_text)
        self.assertEqual(self._read_txt_page(legacy_txt, 3), "")

        legacy_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        legacy_page = legacy_metadata["books"][0]["pages"][0]
        self.assertEqual(
            set(legacy_page), {"page", "method", "character_count", "warnings", "errors"}
        )
        self.assertIn("Pages: 3", report_path.read_text(encoding="utf-8"))
        self.assertEqual(result["books"][0]["pages_processed"], 3)

        page_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(page_records), 3)
        sparse, scanned, blank = page_records
        self.assertEqual(sparse["document_id"], "sample_book")
        self.assertEqual(sparse["page_number"], 1)
        self.assertEqual(sparse["content_type"], "unknown")
        self.assertEqual(sparse["extraction_method"], "pymupdf")
        self.assertEqual(sparse["quality_status"], "review")
        self.assertEqual(sparse["pymupdf_text"], self._read_txt_page(legacy_txt, 1))
        self.assertEqual(sparse["extracted_text"], self._read_txt_page(legacy_txt, 1))
        self.assertEqual(sparse["interpretations"], [])
        self.assertEqual(sparse["associated_assets"][0]["page_number"], 1)
        self.assertTrue(sparse["associated_assets"][0]["uri"].endswith("sample_book.pdf#page=1"))

        self.assertEqual(scanned["pymupdf_text"], "")
        self.assertEqual(scanned["extracted_text"], ocr_text)
        self.assertEqual(scanned["extraction_method"], "tesseract")
        self.assertEqual(scanned["quality_status"], "usable")
        self.assertEqual(blank["quality_status"], "no_text_detected")
        self.assertEqual(blank["extraction_method"], "none")
        self.assertEqual(ocr_mock.call_count, 1, "OCR should run only for the image-only page")

    def test_ocr_failure_does_not_discard_embedded_text(self) -> None:
        fitz = pdf_extraction.fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Short but retained header")
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
        pixmap.clear_with(255)
        page.insert_image(page.rect, pixmap=pixmap)
        doc.save(self.pdf_path)
        doc.close()

        fake_tesseract = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=""))
        with (
            patch.object(pdf_extraction, "pytesseract", fake_tesseract),
            patch.object(pdf_extraction, "Image", object()),
            patch.object(pdf_extraction, "_resolve_tesseract", return_value="mock-tesseract"),
            patch.object(pdf_extraction, "_ocr_page", side_effect=RuntimeError("mock OCR error")),
        ):
            pdf_extraction.run(self.input_dir, self.output_dir)

        legacy_txt = (self.output_dir / "sample_book.txt").read_text(encoding="utf-8")
        record = json.loads((self.output_dir / "sample_book.pages.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(self._read_txt_page(legacy_txt, 1), "Short but retained header")
        self.assertEqual(record["pymupdf_text"], "Short but retained header")
        self.assertEqual(record["extracted_text"], "Short but retained header")
        self.assertEqual(record["quality_status"], "failed")
        self.assertTrue(any("mock OCR error" in error for error in record["errors"]))


if __name__ == "__main__":
    unittest.main()
