"""Document-level extraction quality: digital, scanned or mixed, and how trustworthy the text is.

Built from the per-page records ingestion already stores (``extraction_method``, ``quality_status``,
text length), so it costs no reprocessing and works for every stored document.

    digital   every page with text came from the embedded text layer (or a DOCX/HTML/Markdown file)
    scanned   every page with text came from OCR
    mixed     both kinds of page
    empty     no page has text
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

OCR_METHODS = {"tesseract", "ocr"}
EMPTY = {"none", "failed", "empty", None}
MIN_USABLE_SHARE = 0.9      # below this share of usable pages the document is flagged for review


def _has_text(page: Mapping[str, Any]) -> bool:
    """From the text when it was loaded, else from the extraction method (cheap listing)."""
    if "text" in page:
        return bool((page.get("text") or "").strip())
    method = page.get("extraction_method")
    return method not in EMPTY or (method is None and page.get("quality_status") == "usable")


def assess(pages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    pages = list(pages)
    methods = Counter(page.get("extraction_method") or "none" for page in pages)
    statuses = Counter(page.get("quality_status") or "review" for page in pages)
    with_text = [page for page in pages if _has_text(page)]
    ocr = sum(1 for page in with_text if page.get("extraction_method") in OCR_METHODS)
    digital = len(with_text) - ocr
    kind = "empty" if not with_text else "scanned" if digital == 0 else "digital" if ocr == 0 else "mixed"
    usable = statuses.get("usable", 0)
    share = usable / len(pages) if pages else 0.0
    warnings = []
    if kind in ("scanned", "mixed"):
        warnings.append(f"{ocr} page(s) read by OCR: check numbers and symbols against the page image")
    blank = len(pages) - len(with_text)
    if blank:
        warnings.append(f"{blank} page(s) without text (blank, image-only or failed)")
    if statuses.get("review"):
        warnings.append(f"{statuses['review']} page(s) marked for review (low or sparse text)")
    if pages and share < MIN_USABLE_SHARE:
        warnings.append(f"only {share:.0%} of pages are usable")
    characters = [len(page.get("text") or "") for page in with_text]
    return {"kind": kind, "pages": len(pages), "pages_with_text": len(with_text), "ocr_pages": ocr,
            "digital_pages": digital, "usable_share": round(share, 3), "methods": dict(methods),
            "statuses": dict(statuses), "mean_characters_per_page": round(sum(characters) / len(characters))
            if characters else 0, "needs_review": share < MIN_USABLE_SHARE or kind != "digital",
            "warnings": warnings}
