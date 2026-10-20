#!/usr/bin/env python3
"""Extract page-aware text from PDF books, using Tesseract for scan-like pages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - reported at runtime
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover - reported at runtime
    pytesseract = None
    Image = None


def _has_usable_text(text: str, minimum: int) -> bool:
    """Reject blank/noisy text layers while retaining short meaningful pages."""
    normalized = "".join(ch for ch in text if ch.isprintable() and not ch.isspace())
    alphanumeric = sum(ch.isalnum() for ch in normalized)
    return len(normalized) >= minimum and alphanumeric >= min(20, minimum // 2)


def _resolve_tesseract(command: str | None) -> str | None:
    candidate = command or os.environ.get("TESSERACT_CMD")
    if candidate:
        path = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
    else:
        path = shutil.which("tesseract")
        if not path and os.name == "nt":
            for location in (
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
                Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
            ):
                if location.is_file():
                    path = str(location)
                    break
    return str(path) if path else None


def _page_text(page: Any) -> str:
    # sort=True reconstructs visual reading order from positioned text spans.
    return page.get_text("text", sort=True).strip()


def _ocr_page(page: Any, dpi: int, language: str) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("Python OCR support is unavailable (install pytesseract and Pillow).")
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image, lang=language).strip()


def _extract_pdf(pdf_path: Path, output_dir: Path, minimum: int, dpi: int,
                 language: str, tesseract_command: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": pdf_path.name,
        "title": None,
        "total_pages": None,
        "pages_processed": 0,
        "pages": [],
        "warnings": [],
        "errors": [],
        "text_file": None,
    }
    target = output_dir / f"{pdf_path.stem}.txt"
    if fitz is None:
        record["errors"].append("PyMuPDF (fitz) is unavailable; cannot open PDF.")
        return record
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        record["errors"].append(f"Could not open PDF: {type(exc).__name__}: {exc}")
        return record

    try:
        record["title"] = doc.metadata.get("title") or None
        record["total_pages"] = len(doc)
        ocr_executable = _resolve_tesseract(tesseract_command)
        ocr_ready = pytesseract is not None and Image is not None and ocr_executable is not None
        if pytesseract is not None and ocr_executable:
            pytesseract.pytesseract.tesseract_cmd = ocr_executable
        needs_ocr = False
        sections: list[str] = []
        for index in range(len(doc)):
            page_number = index + 1
            page_record: dict[str, Any] = {
                "page": page_number,
                "method": None,
                "character_count": 0,
                "warnings": [],
                "errors": [],
            }
            try:
                page = doc.load_page(index)
                text = _page_text(page)
                if _has_usable_text(text, minimum):
                    method = "text"
                else:
                    needs_ocr = True
                    page_record["warnings"].append("Little or no usable embedded text; potential scanned page.")
                    if not ocr_ready:
                        method = "unavailable"
                        page_record["errors"].append(
                            "OCR unavailable. Install the Tesseract executable and Python packages, "
                            "put tesseract on PATH, set TESSERACT_CMD, or pass --tesseract-cmd."
                        )
                        text = ""
                    else:
                        try:
                            text = _ocr_page(page, dpi, language)
                            if _has_usable_text(text, minimum):
                                method = "ocr"
                            elif text.strip():
                                method = "low_text"
                                page_record["warnings"].append("OCR returned very little usable text.")
                            else:
                                method = "empty"
                                page_record["warnings"].append("OCR found no text; page may be intentionally blank.")
                        except Exception as exc:
                            method = "failed"
                            text = ""
                            page_record["errors"].append(f"OCR failed: {type(exc).__name__}: {exc}")
                page_record["method"] = method
                page_record["character_count"] = len(text)
                sections.append(f"{'=' * 24} PAGE {page_number} {'=' * 24}\n{text}\n")
            except Exception as exc:
                page_record["method"] = "failed"
                page_record["errors"].append(f"Page extraction failed: {type(exc).__name__}: {exc}")
                sections.append(f"{'=' * 24} PAGE {page_number} {'=' * 24}\n[Page extraction failed]\n")
            record["pages"].append(page_record)
        record["pages_processed"] = len(record["pages"])
        if needs_ocr and not ocr_ready:
            record["warnings"].append(
                "Some pages may be scans, but OCR was unavailable. Install/configure Tesseract and rerun."
            )
        if record["total_pages"] == 0:
            record["warnings"].append("PDF contains no pages.")
        target.write_text(
            f"Source PDF: {pdf_path.name}\nTitle: {record['title'] or '(not provided)'}\n"
            f"Pages: {record['total_pages']}\n\n" + "\n".join(sections),
            encoding="utf-8",
        )
        record["text_file"] = target.name
    finally:
        doc.close()
    return record


def run(input_dir: Path, output_dir: Path, minimum: int = 40, dpi: int = 300,
        language: str = "eng", tesseract_command: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    pdfs = sorted((path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
                  key=lambda path: path.name.casefold())
    books = [_extract_pdf(pdf, output_dir, minimum, dpi, language, tesseract_command) for pdf in pdfs]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "books": books,
    }
    (output_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: list[str] = ["PDF Extraction Report", "====================", f"Generated: {metadata['generated_at_utc']}",
                         f"Input: {input_dir.resolve()}", f"PDF files found: {len(pdfs)}", ""]
    for book in books:
        counts: dict[str, int] = {}
        for page in book["pages"]:
            counts[page["method"]] = counts.get(page["method"], 0) + 1
        report.extend([
            f"{book['filename']} — {book['title'] or '(title unavailable)'}",
            f"  Pages: {book['total_pages']}; processed: {book['pages_processed']}; methods: "
            + (", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "none"),
            f"  Text file: {book['text_file'] or '(not created)' }",
        ])
        for warning in book["warnings"]:
            report.append(f"  WARNING: {warning}")
        for error in book["errors"]:
            report.append(f"  ERROR: {error}")
        for page in book["pages"]:
            for warning in page["warnings"]:
                report.append(f"  Page {page['page']} WARNING: {warning}")
            for error in page["errors"]:
                report.append(f"  Page {page['page']} ERROR: {error}")
        report.append("")
    if not pdfs:
        report.append("No PDF files found in the input directory.")
    (output_dir / "extraction_report.txt").write_text("\n".join(report), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw_pdfs"), help="Folder containing PDF files")
    parser.add_argument("--output", type=Path, default=Path("data/extracted"), help="Folder for text and reports")
    parser.add_argument("--min-text-chars", type=int, default=40,
                        help="Minimum compact embedded/OCR text characters to count as usable (default: 40)")
    parser.add_argument("--dpi", type=int, default=300, help="Render resolution for OCR (default: 300)")
    parser.add_argument("--language", default="eng", help="Tesseract language code (default: eng)")
    parser.add_argument("--tesseract-cmd", help="Path or command name for the Tesseract executable")
    args = parser.parse_args(argv)
    if args.min_text_chars < 1 or args.dpi < 72:
        parser.error("--min-text-chars must be positive and --dpi must be at least 72")
    try:
        result = run(args.input, args.output, args.min_text_chars, args.dpi, args.language, args.tesseract_cmd)
    except Exception as exc:
        print(f"Extraction could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"PDFs: {len(result['books'])}")
    for book in result["books"]:
        methods: dict[str, int] = {}
        for page in book["pages"]:
            methods[page["method"]] = methods.get(page["method"], 0) + 1
        print(f"{book['filename']}: {book['pages_processed']}/{book['total_pages'] or 0} pages; "
              + (", ".join(f"{key}={value}" for key, value in sorted(methods.items())) or "no pages processed"))
    print(f"Output: {args.output.resolve()}")
    print("Files: one .txt per PDF, extraction_metadata.json, extraction_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
