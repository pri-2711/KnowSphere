"""Adapters for DOCX, HTML and Markdown documents (standard library only, no model).

Each adapter reads a file into sections (heading path + text), then ``to_pages`` groups the
sections into logical pages of bounded size. The pages are written as the same ``*.pages.jsonl``
sidecar the PDF extractor writes, so storage, chunking, extraction, governance and retrieval treat
them exactly like PDF pages. A logical page number is a stable citation target ("section page 3"),
not a printed page; ``source.kind`` records which it is.

PDF ingestion is not changed by this module.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

SUFFIXES = {".docx": "docx", ".html": "html", ".htm": "html", ".md": "markdown", ".markdown": "markdown"}
MEDIA_TYPES = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
               "html": "text/html", "markdown": "text/markdown"}
PAGE_CHARS = 3000          # target size of one logical page
SCHEMA_VERSION = "1.0.0"   # the sidecar schema the PDF extractor writes


@dataclass
class Section:
    heading_path: list[str]
    text: str


@dataclass
class TextDocument:
    filename: str
    format: str
    title: str | None
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def supported(path: Path) -> bool:
    return path.suffix.lower() in SUFFIXES


def _collapse(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


class _Sections:
    """Collects (heading level, text) blocks into sections with a heading path."""

    def __init__(self) -> None:
        self.path: list[tuple[int, str]] = []
        self.sections: list[Section] = []
        self.buffer: list[str] = []

    def heading(self, level: int, text: str) -> None:
        self.flush()
        self.path = [item for item in self.path if item[0] < level] + [(level, text)]
        self.buffer.append(text)

    def block(self, text: str) -> None:
        if text:
            self.buffer.append(text)

    def flush(self) -> None:
        body = "\n".join(line for line in self.buffer if line).strip()
        if body:
            self.sections.append(Section([title for _, title in self.path], body))
        self.buffer = []

    def result(self) -> list[Section]:
        self.flush()
        return self.sections


# ---------------------------------------------------------------- DOCX
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(path: Path) -> TextDocument:
    document = TextDocument(path.name, "docx", None)
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        if "docProps/core.xml" in archive.namelist():
            core = ElementTree.fromstring(archive.read("docProps/core.xml"))
            title = core.find("{http://purl.org/dc/elements/1.1/}title")
            document.title = (title.text or "").strip() or None if title is not None else None
    collector = _Sections()
    body = root.find(f"{_W}body")
    for element in (body if body is not None else []):
        if element.tag == f"{_W}p":
            text = _collapse("".join(node.text or "" for node in element.iter(f"{_W}t")))
            style = element.find(f"{_W}pPr/{_W}pStyle")
            name = (style.get(f"{_W}val") or "") if style is not None else ""
            heading = re.match(r"(?i)heading\s?(\d)", name)
            level = int(heading.group(1)) if heading else 1 if name.lower() == "title" else None
            if text and level:
                collector.heading(level, text)
            else:
                collector.block(text)
        elif element.tag == f"{_W}tbl":
            rows = [" | ".join(_collapse("".join(t.text or "" for t in cell.iter(f"{_W}t")))
                               for cell in row.iter(f"{_W}tc")) for row in element.iter(f"{_W}tr")]
            collector.block("\n".join(row for row in rows if row.strip(" |")))
    document.sections = collector.result()
    return document


# ---------------------------------------------------------------- HTML
class _HTML(HTMLParser):
    BLOCKS = {"p", "li", "td", "th", "pre", "blockquote", "dd", "dt", "caption", "figcaption", "div", "br", "tr"}
    SKIP = {"script", "style", "noscript", "template", "head", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.collector = _Sections()
        self.text: list[str] = []
        self.heading_level: int | None = None
        self.skip = 0
        self.title: str | None = None
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCKS:
            self._end_block()
            if re.fullmatch(r"h[1-6]", tag):
                self.heading_level = int(tag[1])

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCKS:
            self._end_block()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title = _collapse((self.title or "") + " " + data) or None
        elif not self.skip:
            self.text.append(data)

    def _end_block(self) -> None:
        text = _collapse("".join(self.text))     # data pieces already carry their spaces
        self.text = []
        if text and self.heading_level is not None:
            self.collector.heading(self.heading_level, text)
        elif text:
            self.collector.block(text)
        self.heading_level = None


def read_html(path: Path) -> TextDocument:
    parser = _HTML()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    parser._end_block()
    return TextDocument(path.name, "html", parser.title, parser.collector.result())


# ---------------------------------------------------------------- Markdown
def read_markdown(path: Path) -> TextDocument:
    collector = _Sections()
    title = None
    paragraph: list[str] = []
    in_code = False

    def end_paragraph() -> None:
        if paragraph:
            collector.block(_collapse(" ".join(paragraph)))
            paragraph.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("```"):
            end_paragraph()
            in_code = not in_code
            continue
        if in_code:
            collector.block(line.rstrip())
            continue
        heading = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if heading:
            end_paragraph()
            text = heading.group(2).strip()
            title = title or (text if len(heading.group(1)) == 1 else None)
            collector.heading(len(heading.group(1)), text)
        elif not line.strip():
            end_paragraph()
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line) or line.lstrip().startswith("|"):
            end_paragraph()
            collector.block(_collapse(re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line)))
        else:
            paragraph.append(line.strip())
    end_paragraph()
    return TextDocument(path.name, "markdown", title, collector.result())


READERS = {"docx": read_docx, "html": read_html, "markdown": read_markdown}


def read(path: Path) -> TextDocument:
    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"Unsupported format: {path.suffix}")
    document = READERS[kind](path)
    if not document.sections:
        document.warnings.append("no text found")
    return document


def to_pages(sections: Iterable[Section], page_chars: int = PAGE_CHARS) -> list[dict]:
    """Pack paragraphs into logical pages of at most about ``page_chars`` characters, in reading order.
    A page records the heading path of its first paragraph."""
    pages: list[dict] = []
    current: list[str] = []
    size = 0
    path: list[str] = []
    for section in sections:
        for paragraph in section.text.split("\n"):
            if current and size + len(paragraph) > page_chars:
                pages.append({"text": "\n".join(current), "section_path": path})
                current, size = [], 0
            if not current:
                path = section.heading_path
            current.append(paragraph)
            size += len(paragraph) + 1
    if current:
        pages.append({"text": "\n".join(current), "section_path": path})
    return [{**page, "page_number": number} for number, page in enumerate(pages, start=1)]


def write_sidecar(document: TextDocument, extracted_dir: Path, document_label: str) -> Path:
    """Write ``<stem>.pages.jsonl`` in the PDF extractor's format; returns its path."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    target = extracted_dir / f"{Path(document.filename).stem}.pages.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for page in to_pages(document.sections):
            handle.write(json.dumps({
                "schema_version": SCHEMA_VERSION, "document_id": document_label, "filename": document.filename,
                "page_number": page["page_number"], "extracted_text": page["text"], "pymupdf_text": None,
                "content_type": "text", "extraction_method": document.format, "quality_status": "usable",
                "section_path": page["section_path"], "warnings": document.warnings, "errors": [],
                "associated_assets": [], "interpretations": [],
            }, ensure_ascii=False) + "\n")
    return target
