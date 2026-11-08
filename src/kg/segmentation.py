"""Semantic segmentation: turn a document's stored pages into coherent chunks.

1. Each page is split into blocks: paragraphs (separated by blank lines), with a
   heading line at the start of a paragraph split off as its own block.
2. Heading-like lines repeated at the top or bottom of several pages are running page headers, not
   section titles; they stay in the text but do not start sections.
3. Blocks are grouped in reading order, across page breaks. A chunk is closed at a
   heading (once it has some content), when it reaches the target size, when the
   next block would exceed the maximum, or when the next block shares too few word
   stems with the chunk (a topic shift). Very long paragraphs are cut at sentence
   boundaries.
4. Table-of-contents pages become their own chunks marked ``kind: "toc"`` and are
   not sent to the LLM. Empty pages produce nothing.

Nothing is dropped: every non-space character of every page lies inside exactly one
chunk span. A chunk's text is its page slices joined by a blank line, and ``spans``
maps chunk offsets back to page offsets.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.kg.config import ChunkingConfig
from src.storage.mongodb_store import stable_id
from src.textutil import is_toc_page, split_sentences, tokenize

SEPARATOR = "\n\n"
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S")


@dataclass
class Block:
    unit_id: str
    page_number: int
    start: int
    end: int
    text: str
    heading: bool = False


def _paragraphs(text: str) -> list[tuple[int, int]]:
    spans, cursor = [], 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    trimmed = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def is_heading_line(line: str, config: ChunkingConfig) -> bool:
    """Short line without closing punctuation that looks like a title or a numbered heading."""
    line = line.strip()
    words = line.split()
    if not words or len(words) > config.heading_max_words or len(line) > 100:
        return False
    if line[-1] in ".,;:!?" or not any(char.isalpha() for char in line):
        return False
    if _NUMBERED_HEADING.match(line) or line.isupper():
        return True
    capitalized = sum(1 for word in words if word[0].isupper() or not word[0].isalpha())
    return capitalized / len(words) >= 0.6


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip().lower()


def running_headers(pages: Iterable[Mapping[str, Any]], config: ChunkingConfig) -> set[str]:
    """Lines in the first or last two lines of many pages (numbers ignored): page headers and footers."""
    counts: Counter = Counter()
    for page in pages:
        lines = [line for line in (page.get("text") or "").splitlines() if line.strip()]
        edge = lines[:2] + lines[-2:]
        counts.update({_normalize_line(line) for line in edge if is_heading_line(line, config)})
    return {line for line, count in counts.items() if count >= config.running_header_pages}


def page_blocks(page: Mapping[str, Any], config: ChunkingConfig, headers: set[str]) -> list[Block]:
    text = page.get("text") or ""
    blocks: list[Block] = []
    for start, end in _paragraphs(text):
        first_line_end = text.find("\n", start, end)
        first_line = text[start:first_line_end if first_line_end != -1 else end]
        heading = is_heading_line(first_line, config) and _normalize_line(first_line) not in headers
        if heading and first_line_end != -1:
            # Split the heading line off; the rest of the paragraph is body text.
            rest = first_line_end + 1
            while rest < end and text[rest].isspace():
                rest += 1
            blocks.append(Block(page["_id"], page["page_number"], start, start + len(first_line.rstrip()),
                                first_line.rstrip(), True))
            if rest < end:
                blocks.extend(_sized(page, rest, end, text, config))
        elif heading:
            blocks.append(Block(page["_id"], page["page_number"], start, end, text[start:end], True))
        else:
            blocks.extend(_sized(page, start, end, text, config))
    return blocks


def _sized(page: Mapping[str, Any], start: int, end: int, text: str, config: ChunkingConfig) -> list[Block]:
    """A body block, cut at sentence boundaries when longer than the maximum chunk size."""
    if end - start <= config.max_chars:
        return [Block(page["_id"], page["page_number"], start, end, text[start:end])]
    blocks, piece_start, piece_end = [], start, start
    for s, e in split_sentences(text[start:end]):          # sentences of at most 500 characters
        s, e = start + s, start + e
        if e - piece_start > config.target_chars and piece_end > piece_start:
            blocks.append(Block(page["_id"], page["page_number"], piece_start, piece_end, text[piece_start:piece_end]))
            piece_start = s
        piece_end = e
    blocks.append(Block(page["_id"], page["page_number"], piece_start, end, text[piece_start:end]))
    return blocks


def cohesion(left: Counter, right: Counter) -> float:
    """Cosine similarity of two stem-count vectors."""
    if not left or not right:
        return 1.0
    dot = sum(count * right.get(term, 0) for term, count in left.items())
    norm = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(sum(v * v for v in right.values()))
    return dot / norm if norm else 0.0


def _build_chunk(document_id: str, blocks: list[Block], pages: Mapping[str, str], section: str | None,
                 kind: str, config: ChunkingConfig) -> dict[str, Any]:
    """One chunk: contiguous blocks of the same page become one verbatim page slice."""
    segments: list[list[Any]] = []          # [unit_id, page_number, start, end]
    for block in blocks:
        if segments and segments[-1][0] == block.unit_id:
            segments[-1][3] = block.end
        else:
            segments.append([block.unit_id, block.page_number, block.start, block.end])
    parts, spans, cursor = [], [], 0
    for unit_id, page_number, start, end in segments:
        if parts:
            cursor += len(SEPARATOR)
        piece = pages[unit_id][start:end]
        spans.append({"content_unit_id": unit_id, "page_number": page_number, "start": start, "end": end,
                      "chunk_start": cursor, "chunk_end": cursor + len(piece)})
        parts.append(piece)
        cursor += len(piece)
    text = SEPARATOR.join(parts)
    first = segments[0]
    return {
        "_id": stable_id("content_unit", document_id, "chunk", first[0], first[2]),
        "document_id": document_id,
        "unit_type": "chunk",
        "unit_index": first[2],             # start offset on the first page: unique per page, stable
        "page_number": first[1],
        "page_end": segments[-1][1],
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "section": section,
        "kind": kind,
        "spans": spans,
        "chunker_version": config.version,
    }


def segment_document(document_id: str, pages: list[Mapping[str, Any]], config: ChunkingConfig,
                     project_id: str = "default") -> list[dict[str, Any]]:
    """Semantic chunks for one document, in reading order (``chunk_index`` = position)."""
    pages = sorted(pages, key=lambda page: page["page_number"])
    page_text = {page["_id"]: page.get("text") or "" for page in pages}
    headers = running_headers(pages, config)
    chunks: list[dict[str, Any]] = []
    current: list[Block] = []
    current_terms: Counter = Counter()
    current_size = 0
    section: str | None = None
    chunk_section: str | None = None

    def close(kind: str = "text") -> None:
        nonlocal current, current_terms, current_size
        if current:
            chunks.append(_build_chunk(document_id, current, page_text, chunk_section, kind, config))
        current, current_terms, current_size = [], Counter(), 0

    for page in pages:
        text = page_text[page["_id"]]
        if not text.strip():
            continue
        blocks = page_blocks(page, config, headers)
        if is_toc_page(text):
            close()
            current = blocks
            close("toc")
            continue
        for block in blocks:
            terms = Counter(tokenize(block.text))
            # What the chunk text grows by: the gap from the previous block on the same page, or a separator.
            previous = current[-1] if current else None
            size = (block.end - previous.end) if previous and previous.unit_id == block.unit_id \
                else len(block.text) + (len(SEPARATOR) if previous else 0)
            if block.heading:
                if current_size >= config.min_chars:
                    close()
                section = re.sub(r"\s+", " ", block.text).strip()
                if not current:
                    chunk_section = section
            else:
                if current and current_size + size > config.max_chars:
                    close()
                elif current_size >= config.target_chars:
                    close()
                elif (current_size >= config.min_chars and sum(terms.values()) >= 5
                      and cohesion(current_terms, terms) < config.cohesion_threshold):
                    close()
                if not current:
                    chunk_section = section
            if not current:
                size = len(block.text)   # a new chunk starts at this block
            current.append(block)
            current_terms.update(terms)
            current_size += size
    close()

    # A very small text chunk (a stray line, a lone heading) joins its neighbour when they fit together.
    merged_chunks: list[dict[str, Any]] = []
    tiny = config.min_chars // 4
    for chunk in chunks:
        last = merged_chunks[-1] if merged_chunks else None
        if last is not None and last["kind"] == chunk["kind"] == "text" \
                and min(len(last["text"]), len(chunk["text"])) < tiny \
                and len(last["text"]) + len(SEPARATOR) + len(chunk["text"]) <= config.max_chars:
            section = last["section"] if len(last["text"]) >= tiny else chunk["section"]
            merged_chunks[-1] = _build_chunk(document_id, _blocks_from_spans(last) + _blocks_from_spans(chunk),
                                             page_text, section, "text", config)
        else:
            merged_chunks.append(chunk)
    chunks = merged_chunks
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["project_id"] = project_id
    return chunks


def _blocks_from_spans(chunk: Mapping[str, Any]) -> list[Block]:
    return [Block(span["content_unit_id"], span["page_number"], span["start"], span["end"], "")
            for span in chunk["spans"]]


def chunk_offset_to_page(chunk: Mapping[str, Any], start: int, end: int) -> dict[str, Any] | None:
    """Map a chunk slice to the page it starts on (unit, page number, page offsets)."""
    for span in chunk["spans"]:
        if span["chunk_start"] <= start < span["chunk_end"]:
            offset = span["start"] + (start - span["chunk_start"])
            page_end = span["start"] + (min(end, span["chunk_end"]) - span["chunk_start"])
            return {"content_unit_id": span["content_unit_id"], "page_number": span["page_number"],
                    "page_start": offset, "page_end": page_end}
    return None
