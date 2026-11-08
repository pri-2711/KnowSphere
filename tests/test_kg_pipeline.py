"""End-to-end concept-first pipeline on mongomock with a scripted LLM: persistence, idempotency, recovery, lifecycle."""

from __future__ import annotations

import json

import pytest
from conftest import FakeEmbedder
from kg_fakes import DOC_X, DOC_Y, PAGES, ScriptedLLM, insert_corpus, page_id

from src.ingest.documents import delete_document
from src.kg.config import ChunkingConfig, KGConfig
from src.kg.llm import LLMResponse
from src.kg.orchestrator import KGPipeline
from src.kg.store import CONCEPTS, RELATIONSHIPS

CONFIG = KGConfig(chunking=ChunkingConfig(min_chars=60, target_chars=1000, max_chars=2000), max_retries=1)


@pytest.fixture
def corpus(store):
    insert_corpus(store)
    return store


def _pipeline(store, llm=None):
    return KGPipeline(store, CONFIG, llm or ScriptedLLM(), FakeEmbedder())


def _snapshot(store):
    concepts = sorted((c["_id"], c["canonical_name"], c["mention_count"]) for c in store.database[CONCEPTS].find())
    relationships = sorted((r["_id"], r["relation_type"], r["evidence_count"]) for r in store.database[RELATIONSHIPS].find())
    chunks = sorted(u["_id"] for u in store.content_units.find({"unit_type": "chunk"}))
    return concepts, relationships, chunks


def test_end_to_end_builds_a_traceable_cross_document_graph(corpus) -> None:
    summary = _pipeline(corpus).run([DOC_X, DOC_Y])
    assert summary["segmentation"]["toc_chunks"] == 1
    assert summary["pass1"]["processed"] == 2 and summary["pass1"]["failed"] == 0
    assert summary["pass2"]["concepts_in_several_documents"] >= 2     # supervised learning, labeled data
    names = {c["canonical_name"]: c for c in corpus.database[CONCEPTS].find()}
    assert set(names) == {"Supervised Learning", "Machine learning", "Labeled data", "Unsupervised learning",
                          "Neural network"}
    supervised = names["Supervised Learning"]
    assert supervised["document_ids"] == [DOC_X, DOC_Y] and "SL" in supervised["aliases"]
    assert supervised["description"]["generated"] is True and supervised["description"]["evidence"]["quote"]
    # Every piece of evidence leads back to the exact page text.
    pages = {u["_id"]: u["text"] for u in corpus.content_units.find({"unit_type": "page"})}
    for concept in corpus.database[CONCEPTS].find():
        for mention in concept["mentions"]:
            evidence = mention["evidence"]
            assert pages[evidence["content_unit_id"]][evidence["page_start"]:evidence["page_end"]] == evidence["quote"]
    uses = corpus.database[RELATIONSHIPS].find_one({"relation_type": "uses"})
    assert uses["source_id"] == supervised["_id"] and uses["cross_document"] and uses["evidence_count"] == 2
    assert corpus.database[RELATIONSHIPS].find_one({"relation_type": "example_of"})["target_id"] == supervised["_id"]


def test_rerun_creates_no_duplicates_and_calls_no_llm(corpus) -> None:
    _pipeline(corpus).run([DOC_X, DOC_Y])
    before = _snapshot(corpus)
    llm = ScriptedLLM()
    summary = _pipeline(corpus, llm).run([DOC_X, DOC_Y])
    assert llm.calls == [] and summary["pass1"]["reused"] == 2
    assert summary["segmentation"]["unchanged"] == summary["segmentation"]["chunks"]
    assert _snapshot(corpus) == before


def test_failed_chunk_is_kept_for_review_and_only_it_is_retried(corpus) -> None:
    def broken_for_y(messages):
        if "Supervised Learning (SL)" in messages[0]["content"]:
            return LLMResponse("not json at all", stop_reason="end_turn")
        return ScriptedLLM().complete("", messages, {})

    summary = _pipeline(corpus, ScriptedLLM(responder=broken_for_y)).run([DOC_X, DOC_Y])
    assert summary["pass1"]["failed"] == 1 and summary["pass1"]["recovery_attempts"] == 1
    failed = corpus.content_units.find_one({"unit_type": "chunk", "document_id": DOC_Y})["extraction"]
    assert failed["status"] == "failed" and failed["status_detail"] == "needs_review"
    assert [a["raw"] for a in failed["attempts"]] == ["not json at all", "not json at all"]
    assert corpus.database[CONCEPTS].find_one({"canonical_name": "Neural network"}) is None   # nothing invented

    llm = ScriptedLLM()
    summary = _pipeline(corpus, llm).run([DOC_X, DOC_Y])
    assert len(llm.calls) == 1 and "Supervised Learning (SL)" in llm.calls[0][0]["content"]
    assert summary["pass1"]["failed"] == 0
    assert corpus.database[CONCEPTS].find_one({"canonical_name": "Neural network"}) is not None


def test_changed_page_text_re_extracts_only_its_chunk(corpus) -> None:
    _pipeline(corpus).run([DOC_X, DOC_Y])
    corpus.content_units.update_one({"_id": page_id(DOC_Y, 1)},
                                    {"$set": {"text": PAGES[(DOC_Y, 1)] + " Transfer learning reuses a model."}})
    llm = ScriptedLLM()
    summary = _pipeline(corpus, llm).run([DOC_X, DOC_Y])
    assert len(llm.calls) == 1 and summary["segmentation"]["updated"] == 1
    assert corpus.content_units.count_documents({"unit_type": "chunk", "document_id": DOC_Y}) == 1


def test_existing_records_are_untouched_and_limit_controls_cost(corpus) -> None:
    pages_before = list(corpus.content_units.find({"unit_type": "page"}))
    llm = ScriptedLLM()
    summary = _pipeline(corpus, llm).run([DOC_X, DOC_Y], limit=1)
    assert len(llm.calls) == 1 and summary["pass1"]["processed"] == 1
    assert list(corpus.content_units.find({"unit_type": "page"})) == pages_before
    assert corpus.database["concepts"].count_documents({}) == 0      # the GLiNER collections are not used


def test_deleting_a_document_removes_its_part_of_the_graph(corpus, tmp_path) -> None:
    pipeline = _pipeline(corpus)
    pipeline.run([DOC_X, DOC_Y])
    delete_document(corpus, None, DOC_Y, tmp_path / "uploads", tmp_path / "images")
    stats = pipeline.build_graph()
    assert corpus.content_units.count_documents({"document_id": DOC_Y}) == 0
    assert corpus.database[CONCEPTS].find_one({"canonical_name": "Neural network"}) is None
    supervised = corpus.database[CONCEPTS].find_one({"normalized_name": "supervisedlearning"})
    assert supervised["document_ids"] == [DOC_X]
    assert all(DOC_Y not in edge["document_ids"] for edge in corpus.database[RELATIONSHIPS].find())
    assert stats["stale_concepts_removed"] >= 1


def test_graph_validation_and_merge_report(corpus, tmp_path) -> None:
    pipeline = KGPipeline(corpus, CONFIG, ScriptedLLM(), FakeEmbedder(), report_dir=tmp_path)
    summary = pipeline.run([DOC_X, DOC_Y])
    validation = summary["pass2"]["validation"]
    assert validation["relationships_rejected"] == 0
    assert validation["concepts_supported"] == summary["pass2"]["canonical_concepts"]
    log = json.loads((tmp_path / "kg_merge_report.json").read_text(encoding="utf-8"))
    assert isinstance(log, list)


def test_app_lifecycle_delete_also_updates_the_concept_first_graph(corpus, metadata, tmp_path) -> None:
    from conftest import FakeGliner

    from src.knowledge.concept_extraction import GlinerExtractor
    from src.pipeline import Pipeline

    _pipeline(corpus).run([DOC_X, DOC_Y])
    app_pipeline = Pipeline(corpus, metadata, extractor=GlinerExtractor(model=FakeGliner(), threshold=0.5),
                            embedder=FakeEmbedder(), upload_dir=tmp_path / "uploads", image_dir=tmp_path / "images",
                            report_dir=tmp_path / "reports")
    app_pipeline.delete(DOC_Y)
    stages = app_pipeline.knowledge_stages()
    assert stages["kg_graph"]["result"]["stale_concepts_removed"] >= 1
    assert corpus.database[CONCEPTS].find_one({"canonical_name": "Neural network"}) is None
    assert all(concept["document_ids"] == [DOC_X] for concept in corpus.database[CONCEPTS].find())
