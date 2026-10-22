"""Concept extraction with a mocked GLiNER model: offsets, evidence, noise filtering, idempotency, resume."""

from __future__ import annotations

from conftest import FakeGliner, insert_pages, page_id

from src.knowledge.concept_extraction import GlinerExtractor, chunk_text_by_words, extract_concepts, is_noise


def _extract(store, model=None):
    model = model or FakeGliner()
    return extract_concepts(store, GlinerExtractor(model=model, threshold=0.5)), model


def test_chunks_keep_page_offsets() -> None:
    text = "one two  three four five"
    chunks = chunk_text_by_words(text, 2)
    assert [chunk for _, chunk in chunks] == ["one two", "three four", "five"]
    assert all(text[start:start + len(chunk)] == chunk for start, chunk in chunks)


def test_extractor_maps_chunk_offsets_back_to_the_page() -> None:
    text = "alpha gamma risk management delta " * 3
    extractor = GlinerExtractor(model=FakeGliner(), chunk_words=4, batch_size=2)
    spans = extractor.extract_many([text])[0]
    assert spans
    assert all(text[span["start"]:span["end"]].lower() == "risk management" for span in spans)


def test_mentions_carry_verbatim_sentence_evidence(store) -> None:
    insert_pages(store)
    _extract(store)
    record = store.database["concept_mentions"].find_one({"name": "requirements analysis",
                                                          "content_unit_id": page_id("doc-a", 1)})
    page = store.content_units.find_one({"_id": page_id("doc-a", 1)})
    assert record["document_id"] == "doc-a" and record["page_number"] == 1
    start, end = record["sentences"][0]
    assert page["text"][start:end] == "Requirements analysis is a part of systems engineering."


def test_noise_is_filtered(store) -> None:
    insert_pages(store)
    summary, model = _extract(store)
    names = set(store.database["concept_mentions"].distinct("name"))
    assert "figure" not in names            # generic stoplist word
    assert "and the" not in names           # fragment
    assert "weak signal" not in names       # below threshold
    assert summary["skipped_toc"] == 1      # table-of-contents page is not sent to the model
    assert not any("....." in text for text in model.texts)
    state = store.database["extraction_state"].find_one({"_id": page_id("doc-a", 3)})
    assert state["status"] == "skipped_toc"
    assert is_noise("system") and is_noise("of the system") and is_noise("12 34") and is_noise("ab")
    assert not is_noise("requirements analysis")


def test_second_run_skips_unchanged_pages_and_creates_no_duplicates(store) -> None:
    insert_pages(store)
    _extract(store)
    before = sorted(record["_id"] for record in store.database["concept_mentions"].find())
    summary, model = _extract(store)
    after = sorted(record["_id"] for record in store.database["concept_mentions"].find())
    assert summary["unchanged"] == summary["pages"] and summary["processed"] == 0
    assert model.calls == 0
    assert before == after and len(after) == len(set(after))


def test_changed_page_is_reprocessed_alone(store) -> None:
    insert_pages(store)
    _extract(store)
    store.content_units.update_one({"_id": page_id("doc-b", 1)},
                                   {"$set": {"text": "Risk management is planned early in every program. It covers cost, "
                                                     "schedule and technical risks for the whole team and all "
                                                     "suppliers involved."}})
    summary, _ = _extract(store)
    assert summary["processed"] == 1 and summary["unchanged"] == summary["pages"] - 1
    names = set(store.database["concept_mentions"].distinct("name", {"content_unit_id": page_id("doc-b", 1)}))
    assert names == {"risk management"}
