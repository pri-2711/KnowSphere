"""Document metadata for citations: auto-detection, the editable JSON file, and formatting.

Metadata lives in ``data/document_metadata.json`` keyed by document ID (the PDF
SHA-256). Detection only records values that are literally present in the PDF
metadata or the first pages, together with where they were found. Anything not
found is stored as null and shown as "Unknown". Fields the user edited are never
overwritten by detection.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Mapping

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - reported at runtime
    fitz = None

UNKNOWN = "Unknown"
FIELDS = ("title", "author", "year")
FIRST_PAGES = 8

# PDF metadata values that are clearly not a real title or author.
_JUNK_TITLES = {"intro", "introduction", "untitled", "title", "document", "cover", "contents", "preface"}
_YEAR = r"(1[5-9]\d\d|20\d\d)"
_COPYRIGHT = re.compile(r"(?:©|\(c\)|copyright|�)[^\n]{0,80}?\b" + _YEAR + r"\b", re.IGNORECASE)
_PUBLISHED = re.compile(r"\b(?:first\s+)?published(?:\s+in)?[^\n]{0,40}?\b" + _YEAR + r"\b", re.IGNORECASE)
_MONTH_YEAR = re.compile(
    r"^\s*(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+)?"
    + _YEAR + r"\s*$", re.MULTILINE)
_AUTHOR_PATTERNS = (
    ("'Edited by' line", re.compile(r"^\s*edited\s+by\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("'Editor:' line", re.compile(r"^\s*editors?\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("'By' line", re.compile(r"^\s*by\s+([A-Z][^\n]{2,80}?)\s*$", re.MULTILINE)),
    ("'Prepared by' line",
     re.compile(r"prepared\s+by\s+(?:the\s+)?\s*([A-Z][A-Za-z.&,' ]{3,80}?)\s*(?:\n|$)", re.IGNORECASE)),
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", value.replace("�", " ")).strip(" -:;,")
    return value or None


def _title_case_if_shouting(value: str) -> str:
    return value.title() if value.isupper() else value


def _lines(page: Any) -> list[tuple[float, str]]:
    """(font size, text) for every text line on a page."""
    lines: list[tuple[float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            sizes = [span.get("size", 0) for span in line.get("spans", []) if span.get("text", "").strip()]
            if text and sizes:
                lines.append((max(sizes), text))
    return lines


def _body_size(pages: list[list[tuple[float, str]]]) -> float:
    """The font size carrying the most characters over the first pages (the running text)."""
    chars_by_size: dict[float, int] = {}
    for lines in pages:
        for size, text in lines:
            chars_by_size[round(size, 1)] = chars_by_size.get(round(size, 1), 0) + len(text)
    return max(chars_by_size.items(), key=lambda item: item[1])[0] if chars_by_size else 0.0


def _largest_text(lines: list[tuple[float, str]], body: float) -> str | None:
    """Return the lines of page 1 set in the largest font size, if that size stands out from the body text."""
    if not lines:
        return None
    largest = max(size for size, _ in lines)
    # A page set in one size has no detectable title, so nothing is guessed.
    if largest < body * 1.25:
        return None
    # Lines within 15 percent of the largest size form a multi-line title.
    title = _clean(" ".join(text for size, text in lines if size >= largest * 0.85))
    if not title or len(title) > 150 or len(title.split()) > 20:
        return None
    return title


def detect_metadata(pdf_path: Path) -> dict[str, Any]:
    """Detect title, author/editor and publication year, recording the source of each value."""
    result: dict[str, Any] = {"title": None, "author": None, "year": None, "detected_from": {}}
    if fitz is None:
        return result
    with fitz.open(pdf_path) as doc:
        info = doc.metadata or {}
        pages = [doc[index].get_text("text", sort=True) for index in range(min(FIRST_PAGES, len(doc)))]
        page_lines = [_lines(doc[index]) for index in range(min(FIRST_PAGES, len(doc)))]
        first_page_title = _largest_text(page_lines[0], _body_size(page_lines)) if page_lines else None

    title = _clean(info.get("title"))
    if title and len(title.split()) >= 2 and title.lower() not in _JUNK_TITLES:
        result["title"] = title
        result["detected_from"]["title"] = "PDF metadata title"
    elif first_page_title and len(first_page_title) >= 4:
        result["title"] = _title_case_if_shouting(first_page_title)
        result["detected_from"]["title"] = "largest text on PDF page 1"
        # A title page often shows only the main title; a later line may give "Title - Subtitle".
        fuller = re.compile(r"^\s*(" + re.escape(first_page_title) + r"\s*[-:–]\s*\S[^\n]{2,120}?)\s*$",
                            re.IGNORECASE | re.MULTILINE)
        found = [(page_index, _clean(match.group(1)).replace("–", "-")) for page_index, text in enumerate(pages)
                 for match in [fuller.search(text)] if match]
        if found:
            # Prefer a line in normal case over an all-caps one.
            page_index, line = next((item for item in found if not item[1].isupper()), found[0])
            result["title"] = _title_case_if_shouting(line)
            result["detected_from"]["title"] = f"title line on PDF page {page_index + 1}"

    author = _clean(info.get("author"))
    if author and len(author.split()) >= 2:
        result["author"] = author
        result["detected_from"]["author"] = "PDF metadata author"
    else:
        for page_index, text in enumerate(pages):
            for label, pattern in _AUTHOR_PATTERNS:
                match = pattern.search(text)
                if match and _clean(match.group(1)):
                    result["author"] = _title_case_if_shouting(_clean(match.group(1)))
                    result["detected_from"]["author"] = f"{label} on PDF page {page_index + 1}"
                    break
            if result["author"]:
                break

    for page_index, text in enumerate(pages):
        match = _COPYRIGHT.search(text) or _PUBLISHED.search(text)
        if match:
            result["year"] = match.group(1)
            result["detected_from"]["year"] = f"copyright/published line on PDF page {page_index + 1}"
            break
    if not result["year"] and pages:
        match = _MONTH_YEAR.search(pages[0])
        if match:
            result["year"] = match.group(1)
            result["detected_from"]["year"] = "date line on PDF page 1"
    return result


class MetadataStore:
    """Thread-safe access to data/document_metadata.json, reloaded when the file changes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}
        self._mtime: float | None = None

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                self._cache, self._mtime = {}, None
                return {}
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = mtime
            return self._cache

    def _write(self, data: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def get(self, document_id: str) -> dict[str, Any]:
        return dict(self.load().get(document_id, {}))

    def record_detection(self, document_id: str, filename: str, detected: Mapping[str, Any]) -> dict[str, Any]:
        """Store detected values without overwriting fields the user has edited."""
        data = dict(self.load())
        entry = dict(data.get(document_id, {}))
        edited = set(entry.get("user_edited", []))
        entry["filename"] = filename
        for field in FIELDS:
            if field not in edited:
                entry[field] = detected.get(field)
        entry["detected_from"] = {key: value for key, value in detected.get("detected_from", {}).items()
                                  if key not in edited}
        entry.setdefault("user_edited", [])
        data[document_id] = entry
        with self._lock:
            self._write(data)
            self._mtime = None
        return entry

    def update(self, document_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a user edit; an empty value resets the field to Unknown."""
        data = dict(self.load())
        entry = dict(data.get(document_id, {}))
        edited = set(entry.get("user_edited", []))
        for field in FIELDS:
            if field in changes:
                value = _clean(changes[field]) if changes[field] is not None else None
                entry[field] = None if value in (None, "", UNKNOWN) else value
                edited.add(field)
        entry["user_edited"] = sorted(edited)
        data[document_id] = entry
        with self._lock:
            self._write(data)
            self._mtime = None
        return entry


    def remove(self, document_id: str) -> None:
        data = dict(self.load())
        if data.pop(document_id, None) is not None:
            with self._lock:
                self._write(data)
                self._mtime = None

    def carry_over_edits(self, old_id: str, new_id: str) -> dict[str, Any]:
        """When a document is replaced by a new version, keep the fields the user edited."""
        old = self.get(old_id)
        edited = {field: old.get(field) for field in old.get("user_edited", [])}
        return self.update(new_id, edited) if edited else self.get(new_id)


def format_citation(metadata: Mapping[str, Any], page_number: int) -> str:
    """[Document Title | Author/Editor | Year of Publication, p. N]"""
    title = metadata.get("title") or UNKNOWN
    author = metadata.get("author") or UNKNOWN
    year = metadata.get("year") or UNKNOWN
    return f"[{title} | {author} | {year}, p. {page_number}]"
