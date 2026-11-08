"""Semantic segmentation: coverage, provenance, headings, running headers, TOC, empty pages, sizes, stable IDs."""

from __future__ import annotations

from collections import Counter

from kg_fakes import DOC_X, PAGES, page_id

from src.kg.config import ChunkingConfig
from src.kg.segmentation import chunk_offset_to_page, cohesion, is_heading_line, segment_document

SMALL = ChunkingConfig(min_chars=60, target_chars=220, max_chars=400, cohesion_threshold=0.08)


def _pages(texts: dict[int, str], document_id: str = "doc") -> list[dict]:
    return [{"_id": page_id(document_id, number), "page_number": number, "text": text}
            for number, text in texts.items()]


def _assert_complete_and_verbatim(pages: list[dict], chunks: list[dict]) -> None:
    text_of = {page["_id"]: page["text"] for page in pages}
    covered = {page["_id"]: bytearray(len(page["text"])) for page in pages}
    for chunk in chunks:
        for span in chunk["spans"]:
            # Every chunk slice is the page text, character for character.
            assert chunk["text"][span["chunk_start"]:span["chunk_end"]] == text_of[span["content_unit_id"]][span["start"]:span["end"]]
            covered[span["content_unit_id"]][span["start"]:span["end"]] = b"\x01" * (span["end"] - span["start"])
    for page in pages:
        lost = [char for i, char in enumerate(page["text"]) if not char.isspace() and not covered[page["_id"]][i]]
        assert lost == [], f"content lost on page {page['page_number']}"


def test_no_content_is_lost_and_provenance_is_kept() -> None:
    pages = [{"_id": page_id(doc, number), "page_number": number, "text": text}
             for (doc, number), text in PAGES.items() if doc == DOC_X]
    chunks = segment_document(DOC_X, pages, SMALL)
    _assert_complete_and_verbatim(pages, chunks)
    assert all(chunk["document_id"] == DOC_X and chunk["unit_type"] == "chunk" for chunk in chunks)
    first = chunks[0]
    assert first["page_number"] == 1 and first["section"] == "Machine Learning Basics"
    located = chunk_offset_to_page(first, first["text"].index("Labeled data"), first["text"].index("Labeled data") + 12)
    assert located["content_unit_id"] == page_id(DOC_X, 1)
    assert PAGES[(DOC_X, 1)][located["page_start"]:located["page_end"]] == "Labeled data"


def test_toc_pages_are_marked_and_empty_pages_make_no_chunk() -> None:
    pages = [{"_id": page_id(doc, number), "page_number": number, "text": text}
             for (doc, number), text in PAGES.items() if doc == DOC_X]
    chunks = segment_document(DOC_X, pages, SMALL)
    kinds = {chunk["page_number"]: chunk["kind"] for chunk in chunks}
    assert kinds[2] == "toc"
    assert 3 not in kinds


def test_headings_start_sections_and_chunks_respect_the_maximum() -> None:
    body = "Thermal control keeps every component within its temperature limits during the mission. " * 8
    pages = _pages({1: f"1.2 Thermal Control\n{body}\n\n2.1 Power Systems\n" + "Batteries store energy for eclipse periods. " * 6})
    chunks = segment_document("doc", pages, SMALL)
    _assert_complete_and_verbatim(pages, chunks)
    assert {chunk["section"] for chunk in chunks} == {"1.2 Thermal Control", "2.1 Power Systems"}
    assert all(len(chunk["text"]) <= SMALL.max_chars for chunk in chunks)
    assert chunks[-1]["text"].startswith("2.1 Power Systems")


def test_running_page_headers_do_not_become_sections() -> None:
    texts = {n: f"NASA Design Standard\n\nSection text {n} explains crew interfaces and displays for operators." * 1
             for n in range(1, 5)}
    chunks = segment_document("doc", _pages(texts), SMALL)
    assert all(chunk["section"] is None for chunk in chunks)


def test_topic_shift_closes_a_chunk() -> None:
    a = "Oxygen generation splits water by electrolysis to supply oxygen for the crew cabin atmosphere. " * 2
    b = "Spacesuit gloves limit finger dexterity, so tools need large handles and simple latches. " * 2
    pages = _pages({1: f"{a}\n\n{b}"})
    chunks = segment_document("doc", pages, SMALL)
    assert len(chunks) == 2 and chunks[1]["text"].startswith("Spacesuit")
    assert cohesion(Counter({"oxygen": 2}), Counter({"glove": 1})) == 0


def test_ids_are_stable_and_heading_detection() -> None:
    pages = [{"_id": page_id(doc, number), "page_number": number, "text": text}
             for (doc, number), text in PAGES.items() if doc == DOC_X]
    first = [chunk["_id"] for chunk in segment_document(DOC_X, pages, SMALL)]
    second = [chunk["_id"] for chunk in segment_document(DOC_X, pages, SMALL)]
    assert first == second and len(set(first)) == len(first)
    assert is_heading_line("5.7.2 Alarm Classification", SMALL)
    assert is_heading_line("FOREWORD", SMALL)
    assert not is_heading_line("This sentence ends with a period.", SMALL)
    assert not is_heading_line("the lower case start of a sentence fragment", SMALL)
