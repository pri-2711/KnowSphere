"""Inverted index, BM25, hash lookups, citations, the citation validator, and abstention."""

from __future__ import annotations

import re

import pytest
from conftest import page_id

from src.metadata import format_citation
from src.search.engine import NOT_FOUND, CitationError
from src.search.index import build_index

CITATION = re.compile(r"^\[[^|\]]+ \| [^|\]]+ \| [^|\]]+, p\. \d+\]$")
SAMPLE_QUERIES = [
    "What is systems engineering?",
    "requirements analysis",
    "Which review is held early in the program?",
    "risk management schedule changes",
    "verification and validation",
    "stakeholder needs",
]


def test_inverted_index_postings_and_unchanged_pages(seeded) -> None:
    term = seeded.database["search_index"].find_one({"_id": "requir"})
    units = {unit for unit, _ in term["postings"]}
    assert page_id("doc-a", 1) in units and page_id("doc-b", 1) in units
    assert term["df"] == len(term["postings"])
    tf = dict(term["postings"])[page_id("doc-a", 2)]
    assert tf == 2
    again = build_index(seeded)
    assert again["tokenized"] == 0 and again["unchanged"] == again["pages"]


def test_bm25_ranks_the_most_relevant_page_first(service) -> None:
    ranked, terms, _ = service.rank_pages("preliminary design review")
    assert ranked[0][0] == page_id("doc-a", 2)
    assert terms == ["preliminari", "design", "review"]


def test_hash_lookups_and_autocomplete(service) -> None:
    assert service.page_id("doc-a", 2) == page_id("doc-a", 2)
    concept_ids = service.match_concepts("Tell me about Requirements Analysis")
    assert service.concepts[concept_ids[0]]["normalized_name"] == "requirements analysis"
    suggestions = service.autocomplete("requ")
    assert suggestions and suggestions[0]["name"].lower().startswith("requirements analysis")
    assert service.autocomplete("zzz") == []


def test_concept_keyword_matches_boost_results(service) -> None:
    plain = service.bm25(["risk", "manag"])
    ranked, _, concepts = service.rank_pages("risk management")
    assert concepts
    top_unit, top_score = ranked[0]
    assert top_score > plain[top_unit]


def test_citation_format_uses_metadata_and_unknown() -> None:
    assert format_citation({"title": "Systems Handbook", "author": "Jane Doe", "year": "2012"}, 7) == \
        "[Systems Handbook | Jane Doe | 2012, p. 7]"
    assert format_citation({}, 3) == "[Unknown | Unknown | Unknown, p. 3]"


def test_citation_validator_rejects_anything_not_verbatim(service) -> None:
    unit = page_id("doc-a", 1)
    text = service.pages[unit]["text"]
    start = text.index("Requirements analysis")
    end = start + len("Requirements analysis is a part of systems engineering.")
    service.validate_quote(unit, start, end, text[start:end], "doc-a", 1)
    with pytest.raises(CitationError):
        service.validate_quote(unit, start, end, text[start:end].replace("part", "core"), "doc-a", 1)
    with pytest.raises(CitationError):
        service.validate_quote(unit, start, end, text[start:end], "doc-a", 2)
    with pytest.raises(CitationError):
        service.validate_quote(unit, start, end, text[start:end], "doc-b", 1)


@pytest.mark.parametrize("question", ["What is the capital of France?", "quantum chromodynamics lattice",
                                      "the of and", ""])
def test_abstains_when_evidence_is_weak(service, question) -> None:
    result = service.ask(question)
    assert result["answer_found"] is False
    assert result["message"] == NOT_FOUND
    assert result["groups"] == [] and result["sentences"] == []


def test_every_returned_quote_exists_verbatim_in_the_cited_page(service, seeded) -> None:
    answered = 0
    for question in SAMPLE_QUERIES:
        result = service.ask(question)
        if not result["answer_found"]:
            continue
        answered += 1
        for sentence in result["sentences"]:
            page = seeded.content_units.find_one({"_id": sentence["content_unit_id"]})
            assert page["document_id"] == sentence["document_id"]
            assert page["page_number"] == sentence["page_number"]
            assert page["text"][sentence["start"]:sentence["end"]] == sentence["quote"]
            assert sentence["quote"] in page["text"]
            assert CITATION.match(sentence["citation"])
        for group in result["groups"]:
            assert CITATION.match(group["citation"])
        search = service.search(question)
        for item in search["results"]:
            page = seeded.content_units.find_one({"_id": item["content_unit_id"]})
            assert page["text"][item["start"]:item["end"]] == item["quote"]
    assert answered >= 4


def test_answer_contains_cited_sentences_and_related_concepts(service) -> None:
    result = service.ask("Which review is held early in the program?")
    assert result["answer_found"]
    assert result["sentences"][0]["quote"] == "A preliminary design review is held early in the program."
    assert result["sentences"][0]["citation"] == "[Systems Handbook | Jane Doe | 2012, p. 2]"
    assert any(concept["name"].lower() == "design review" for concept in result["concepts"])


def test_metadata_edit_changes_citations(service, metadata) -> None:
    metadata.update("doc-b", {"author": "Defense Press"})
    result = service.ask("verification and validation")
    citations = {sentence["citation"] for sentence in result["sentences"] if sentence["document_id"] == "doc-b"}
    assert citations == {"[Engineering Fundamentals | Defense Press | 2001, p. 1]"}
