"""Parser adapters. Each implements ``contracts.interfaces.DocumentParser`` and returns a ``ParsedDocument``.

Page text is still produced by the existing PyMuPDF + Tesseract extraction
(``pdf_extraction``); that stored text is never replaced. A parser adds document
*structure* as extra content units tied to their page and location:

- ``section``  headings with their place in the section hierarchy (``section_path``)
- ``table``    detected tables, text as pipe-separated rows
- ``figure``   figure and table captions with their bounding box
- ``equation`` display equations (Docling labels them; PyMuPDF uses a symbol heuristic)

``DoclingParser`` is the structured parser of choice when the optional ``docling``
package is installed; ``PyMuPDFStructureParser`` always works and is the fallback.
``select_parser`` picks one (``KM_PARSER=auto|docling|pymupdf``).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.contracts.schemas import STRUCTURE_TYPES, ContentUnit, ParsedDocument, Provenance, UnitType  # noqa: F401
from src.contracts.ids import stable_id

COMPONENT = "document_intelligence"
_CAPTION = re.compile(r"^\s*(?:figure|fig\.|table|exhibit|chart|diagram)\s*[\dA-Z]+[\-.:\d]*\s+\S", re.IGNORECASE)
_MATH = set("=+-±×÷∑∏∫√∂∆∇≤≥≈≠∞αβγδεθλμπρστφωΩ^")
_SENTENCE_END = re.compile(r"[.;:,]$")


def unit_id(document_id: str, unit_type: UnitType, page_number: int, index: int) -> str:
    return stable_id("content_unit", document_id, unit_type.value, page_number, index)


def _bbox(values: Any) -> list[float]:
    return [round(float(value), 1) for value in values]


class PyMuPDFStructureParser:
    """Headings by font size, tables with ``page.find_tables``, captions and equation-like lines."""

    name = "pymupdf"

    def __init__(self, heading_ratio: float = 1.15, max_heading_words: int = 14) -> None:
        self.heading_ratio = heading_ratio
        self.max_heading_words = max_heading_words

    def available(self) -> bool:
        try:
            import fitz  # noqa: F401
        except ImportError:
            return False
        return True

    def parse(self, pdf_path: Path, project_id: str, document_id: str,
              document_version_id: str | None = None) -> ParsedDocument:
        import fitz

        provenance = Provenance(component=COMPONENT, method="pymupdf", model=f"pymupdf-{fitz.VersionBind}")
        units: list[ContentUnit] = []
        warnings: list[str] = []
        with fitz.open(pdf_path) as doc:
            pages = [self._lines(page) for page in doc]
            body = self._body_size(pages)
            heading_sizes = sorted({round(size, 1) for lines in pages for _, size, bold, text in lines
                                    if self._is_heading(text, size, False, body)}, reverse=True)
            level_of = {size: min(index + 1, 4) for index, size in enumerate(heading_sizes)}
            bold_level = min(len(heading_sizes) + 1, 4)       # bold headings at body size: the deepest level
            path: list[str] = []
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                counters: Counter = Counter()

                def add(kind: UnitType, text: str, bbox: Any, section: list[str], method: str = "pymupdf") -> None:
                    counters[kind] += 1
                    units.append(ContentUnit(
                        id=unit_id(document_id, kind, page_number, counters[kind]), project_id=project_id,
                        document_id=document_id, document_version_id=document_version_id, unit_type=kind,
                        page_number=page_number, section_path=list(section), text=text, bbox=_bbox(bbox),
                        provenance=provenance.model_copy(update={"method": method})))

                table_boxes: list[Any] = []
                try:
                    for table in page.find_tables().tables:
                        cells = [[(cell or "").strip() for cell in row] for row in table.extract()]
                        width = max((len(row) for row in cells), default=0)
                        keep = [i for i in range(width) if any(i < len(row) and row[i] for row in cells)]
                        rows = [" | ".join(row[i] if i < len(row) else "" for i in keep) for row in cells]
                        text = "\n".join(row for row in rows if row.strip(" |"))
                        if text:
                            table_boxes.append(fitz.Rect(table.bbox))
                            add(UnitType.TABLE, text, table.bbox, path)
                except Exception as exc:  # table detection is best effort per page
                    warnings.append(f"page {page_number}: table detection failed: {exc}")
                previous: tuple[Any, float] | None = None       # (bbox, size) of the last heading line
                for bbox, size, bold, text in pages[page_index]:
                    if any(fitz.Rect(bbox).intersects(box) for box in table_boxes):
                        previous = None
                        continue
                    if self._is_heading(text, size, bold, body):
                        last = units[-1] if units else None
                        wrapped = (previous is not None and last is not None and last.unit_type is UnitType.SECTION
                                   and last.page_number == page_number and abs(previous[1] - size) < 0.2
                                   and 0 <= bbox[1] - previous[0][3] < size * 0.8)
                        if wrapped:
                            # A heading wrapped over two lines: one section.
                            last.text = f"{last.text} {text}"
                            last.bbox = _bbox((min(last.bbox[0], bbox[0]), last.bbox[1], max(last.bbox[2], bbox[2]),
                                               bbox[3]))
                            path[-1] = last.text
                            last.section_path = list(path)
                            previous = (bbox, size)
                            continue
                        level = self._numbered_level(text) or level_of.get(round(size, 1), bold_level)
                        path = path[:level - 1] + [text]
                        add(UnitType.SECTION, text, bbox, path)
                        previous = (bbox, size)
                        continue
                    previous = None
                    if _CAPTION.match(text):
                        add(UnitType.FIGURE, text, bbox, path)
                    elif self._is_equation(text):
                        add(UnitType.EQUATION, text, bbox, path, method="pymupdf-heuristic")
            page_count = len(doc)
        return ParsedDocument(project_id=project_id, document_id=document_id, parser=provenance,
                              page_count=page_count, units=units, warnings=warnings)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _lines(page: Any) -> list[tuple[Any, float, bool, str]]:
        """(bbox, largest font size, bold, text) for each text line of a page."""
        lines = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                if not spans:
                    continue
                text = re.sub(r"\s+", " ", "".join(span["text"] for span in spans)).strip()
                size = max(float(span.get("size", 0)) for span in spans)
                bold = all(int(span.get("flags", 0)) & 16 for span in spans)
                lines.append((line["bbox"], size, bold, text))
        return lines

    @staticmethod
    def _body_size(pages: list[list[tuple[Any, float, bool, str]]]) -> float:
        sizes: Counter = Counter()
        for lines in pages:
            for _, size, _, text in lines:
                sizes[round(size, 1)] += len(text)
        return sizes.most_common(1)[0][0] if sizes else 10.0

    def _is_heading(self, text: str, size: float, bold: bool, body: float) -> bool:
        words = text.split()
        if not 1 <= len(words) <= self.max_heading_words or sum(char.isalpha() for char in text) < 3                 or _SENTENCE_END.search(text) or not (text[0].isupper() or text[0].isdigit()):
            return False
        if size >= body * self.heading_ratio:
            return True
        return bold and size >= body * 0.95 and len(words) <= 10 and not _CAPTION.match(text)

    @staticmethod
    def _numbered_level(text: str) -> int | None:
        """'2.1 Water' is level 2, 'IV. Results' and '3 Methods' are level 1."""
        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+\S", text)
        if match:
            return min(match.group(1).count(".") + 1, 4)
        return 1 if re.match(r"^[IVX]+\.\s+[A-Z]", text) else None

    @staticmethod
    def _is_equation(text: str) -> bool:
        compact = text.replace(" ", "")
        if not 3 <= len(compact) <= 120 or "=" not in compact:
            return False
        symbols = sum(1 for char in compact if char in _MATH or char.isdigit())
        return symbols / len(compact) >= 0.3 and len(re.findall(r"[A-Za-z]{4,}", text)) <= 3


class DoclingParser:
    """Docling (optional dependency). Maps Docling items to structure units with their page and box.

    ``converter`` is injectable; by default ``docling.document_converter.DocumentConverter`` is used.
    """

    name = "docling"
    _LABELS = {"section_header": UnitType.SECTION, "title": UnitType.SECTION, "table": UnitType.TABLE,
               "picture": UnitType.FIGURE, "chart": UnitType.FIGURE, "formula": UnitType.EQUATION}

    def __init__(self, converter_factory: Callable[[], Any] | None = None) -> None:
        self.converter_factory = converter_factory

    def available(self) -> bool:
        if self.converter_factory is not None:
            return True
        try:
            import docling.document_converter  # noqa: F401
        except ImportError:
            return False
        return True

    def parse(self, pdf_path: Path, project_id: str, document_id: str,
              document_version_id: str | None = None) -> ParsedDocument:
        if self.converter_factory is not None:
            converter = self.converter_factory()
        else:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
        document = converter.convert(str(pdf_path)).document
        version = getattr(document, "version", None)
        provenance = Provenance(component=COMPONENT, method="docling", model=f"docling-{version}" if version else "docling")
        units: list[ContentUnit] = []
        counters: Counter = Counter()
        path: list[str] = []
        pages = getattr(document, "pages", {}) or {}
        for item, level in document.iterate_items():
            label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", ""))).lower()
            kind = self._LABELS.get(label)
            prov = (getattr(item, "prov", None) or [None])[0]
            if kind is None or prov is None:
                continue
            page_number = int(prov.page_no)
            text = self._text(item, kind, document)
            if not text:
                continue
            if kind is UnitType.SECTION:
                depth = int(getattr(item, "level", None) or level or 1)
                path = path[:max(depth - 1, 0)] + [text]
            counters[(kind, page_number)] += 1
            units.append(ContentUnit(
                id=unit_id(document_id, kind, page_number, counters[(kind, page_number)]), project_id=project_id,
                document_id=document_id, document_version_id=document_version_id, unit_type=kind,
                page_number=page_number, section_path=list(path), text=text,
                bbox=self._top_left_bbox(prov.bbox, pages.get(page_number)), provenance=provenance))
        return ParsedDocument(project_id=project_id, document_id=document_id, parser=provenance,
                              page_count=len(pages), units=units)

    @staticmethod
    def _text(item: Any, kind: UnitType, document: Any) -> str:
        if kind is UnitType.TABLE and hasattr(item, "export_to_markdown"):
            return item.export_to_markdown(doc=document).strip()
        if kind is UnitType.FIGURE:
            caption = item.caption_text(document) if hasattr(item, "caption_text") else ""
            return (caption or "").strip() or "(figure without caption)"
        return re.sub(r"\s+", " ", getattr(item, "text", "") or "").strip()

    @staticmethod
    def _top_left_bbox(bbox: Any, page: Any) -> list[float] | None:
        if bbox is None:
            return None
        if page is not None and hasattr(bbox, "to_top_left_origin"):
            bbox = bbox.to_top_left_origin(page_height=page.size.height)
        return _bbox((bbox.l, bbox.t, bbox.r, bbox.b))


PARSERS: dict[str, Callable[[], Any]] = {"docling": DoclingParser, "pymupdf": PyMuPDFStructureParser}


def select_parser(choice: str | None = None) -> Any:
    """``auto`` prefers Docling when installed, else PyMuPDF. An explicit choice that is unavailable fails."""
    choice = (choice or os.environ.get("KM_PARSER", "auto")).lower()
    if choice == "auto":
        docling = DoclingParser()
        return docling if docling.available() else PyMuPDFStructureParser()
    if choice not in PARSERS:
        raise ValueError(f"Unknown parser '{choice}'. Choose auto, docling or pymupdf.")
    parser = PARSERS[choice]()
    if not parser.available():
        raise RuntimeError(f"Parser '{choice}' is not installed in this environment")
    return parser
