"""Add, replace (edit) and delete documents, and the watched folder, through the real Pipeline.

GLiNER and the embedding model are mocked; PDFs are generated with PyMuPDF.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from conftest import FakeEmbedder, FakeGliner

from src.ingest.watcher import FolderWatcher, process_events
from src.knowledge.concept_extraction import GlinerExtractor
from src.pipeline import Pipeline
from src.storage.mongodb_store import sha256_file

TEXT_A = ("Systems engineering includes risk management for the whole life cycle of the product. "
          "Requirements analysis is a part of systems engineering and starts early in every program. "
          "Risk management and requirements analysis are reviewed together at each design review.")
TEXT_B = ("Verification and validation confirms that requirements analysis was complete and correct. "
          "Systems engineering supports verification and validation of every delivered product and service. "
          "The design review checks verification and validation plans against stakeholder needs.")


def make_pdf(path: Path, *pages: str) -> Path:
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_textbox(fitz.Rect(72, 72, 540, 760), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def pipeline(store, metadata, tmp_path) -> Pipeline:
    return Pipeline(store, metadata, extractor=GlinerExtractor(model=FakeGliner(), threshold=0.5),
                    embedder=FakeEmbedder(), upload_dir=tmp_path / "uploads", image_dir=tmp_path / "images",
                    report_dir=tmp_path / "reports")


def _ids_everywhere(store, document_id: str) -> dict[str, int]:
    collections = ("content_units", "assets", "extraction_state", "concept_mentions", "page_terms")
    counts = {name: store.database[name].count_documents({"document_id": document_id}) for name in collections}
    counts["concepts"] = store.concepts.count_documents({"document_ids": document_id})
    counts["relationship_evidence"] = sum(
        1 for edge in store.database["relationships"].find() for item in edge["evidence"]
        if item["document_id"] == document_id)
    counts["documents"] = store.documents.count_documents({"_id": document_id})
    return counts


def test_add_then_delete_removes_every_derived_record(pipeline, store, tmp_path) -> None:
    pdf_a = make_pdf(tmp_path / "alpha.pdf", TEXT_A, TEXT_A)
    pdf_b = make_pdf(tmp_path / "beta.pdf", TEXT_B)
    a = pipeline.ingest_file(pdf_a)["document_id"]
    b = pipeline.ingest_file(pdf_b)["document_id"]
    pipeline.knowledge_stages()
    assert _ids_everywhere(store, b)["concepts"] > 0
    assert (tmp_path / "uploads" / b[:16] / "beta.pdf").is_file()

    pipeline.delete(b)
    pipeline.knowledge_stages()
    assert all(count == 0 for count in _ids_everywhere(store, b).values())
    assert not (tmp_path / "uploads" / b[:16]).exists()
    assert pipeline.metadata.get(b) == {}
    names = {concept["normalized_name"] for concept in store.concepts.find()}
    assert "verification and validation" not in names            # only document B mentioned it
    assert _ids_everywhere(store, a)["concepts"] > 0            # document A is untouched
    assert pdf_b.is_file()                                       # the user's own file is never deleted


def test_replace_keeps_metadata_edits_and_drops_the_old_version(pipeline, store, tmp_path) -> None:
    old = pipeline.ingest_file(make_pdf(tmp_path / "v1.pdf", TEXT_A))["document_id"]
    pipeline.knowledge_stages()
    pipeline.metadata.update(old, {"author": "Edited Author"})
    new_pdf = make_pdf(tmp_path / "v2.pdf", TEXT_A, TEXT_B)
    result = pipeline.replace(old, new_pdf, "v2.pdf")
    pipeline.knowledge_stages()
    new = result["document_id"]
    assert new != old and result["replaced"]["deleted"]
    assert store.documents.count_documents({"_id": old}) == 0
    assert store.documents.find_one({"_id": new})["page_count"] == 2
    assert pipeline.metadata.get(new)["author"] == "Edited Author"
    assert all(count == 0 for count in _ids_everywhere(store, old).values())


def test_deleted_document_is_not_re_added_by_the_watcher(pipeline, store, tmp_path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    watcher = FolderWatcher(store, inbox)
    make_pdf(inbox / "report.pdf", TEXT_A)
    assert watcher.scan() == []                                  # first sighting: wait until the file is stable
    process_events(pipeline, watcher, watcher.scan())
    document_id = sha256_file(inbox / "report.pdf")
    pipeline.delete(document_id)                                 # deleted in the app
    watcher.state.delete_many({})                                # even if the watch state were lost
    watcher.scan()
    results = process_events(pipeline, watcher, watcher.scan())
    assert results[0]["skipped"]
    assert store.documents.count_documents({"_id": document_id}) == 0


def test_watched_folder_add_update_delete(pipeline, store, tmp_path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    watcher = FolderWatcher(store, inbox)

    path = make_pdf(inbox / "notes.pdf", TEXT_A)
    assert watcher.scan() == []
    events = watcher.scan()
    assert [(event.kind, event.relative_path) for event in events] == [("added", "notes.pdf")]
    process_events(pipeline, watcher, events)
    first = sha256_file(path)
    assert store.documents.find_one({"_id": first})["sources"] == ["watch:notes.pdf"]
    assert store.concepts.count_documents({"document_ids": first}) > 0
    assert watcher.scan() == [] and watcher.scan() == []         # nothing changed, nothing to do

    pipeline.metadata.update(first, {"title": "My Notes"})
    make_pdf(path, TEXT_B, TEXT_A)                               # edit the file in place
    watcher.scan()
    events = watcher.scan()
    assert [event.kind for event in events] == ["updated"]
    process_events(pipeline, watcher, events)
    second = sha256_file(path)
    assert store.documents.count_documents({"_id": first}) == 0
    assert store.documents.find_one({"_id": second})["page_count"] == 2
    assert pipeline.metadata.get(second)["title"] == "My Notes"

    path.unlink()
    events = watcher.scan()
    assert [event.kind for event in events] == ["deleted"]
    process_events(pipeline, watcher, events)
    assert store.documents.count_documents({}) == 0
    assert store.concepts.count_documents({"document_ids": second}) == 0
    assert watcher.state.count_documents({}) == 0


def test_document_with_another_source_survives_watch_removal(pipeline, store, tmp_path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    watcher = FolderWatcher(store, inbox)
    path = make_pdf(inbox / "shared.pdf", TEXT_A)
    pipeline.ingest_file(path, "shared.pdf")                     # also uploaded in the app
    watcher.scan()
    process_events(pipeline, watcher, watcher.scan())
    document_id = sha256_file(path)
    path.unlink()
    process_events(pipeline, watcher, watcher.scan())
    assert store.documents.find_one({"_id": document_id})["sources"] == ["upload"]
