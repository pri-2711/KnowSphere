"""Metadata detection and editing, and image extraction with captions, context and concept links."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.ingest.images import extract_images, link_images_to_concepts
from src.metadata import MetadataStore, detect_metadata, format_citation
from src.storage.mongodb_store import stable_id


def _make_book(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 120), "Example Systems Book", fontsize=28)
    page.insert_text((72, 180), "Edited by Jane Doe", fontsize=12)
    page = doc.new_page()
    page.insert_text((72, 100), "Copyright 2015 Example Press. All rights reserved.", fontsize=10)
    doc.save(path)
    doc.close()


def test_detects_title_editor_and_year_with_sources(tmp_path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_book(pdf)
    detected = detect_metadata(pdf)
    assert detected["title"] == "Example Systems Book"
    assert detected["author"] == "Jane Doe"
    assert detected["year"] == "2015"
    assert set(detected["detected_from"]) == {"title", "author", "year"}


def test_missing_metadata_stays_unknown(tmp_path) -> None:
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "just some body text without any title cues", fontsize=11)
    doc.save(pdf)
    doc.close()
    detected = detect_metadata(pdf)
    assert detected["author"] is None and detected["year"] is None
    assert format_citation(detected, 4).endswith("| Unknown | Unknown, p. 4]")


def test_user_edits_survive_new_detection(tmp_path) -> None:
    meta = MetadataStore(tmp_path / "document_metadata.json")
    meta.record_detection("d1", "d1.pdf", {"title": "Detected", "author": None, "year": "1999", "detected_from": {}})
    meta.update("d1", {"author": "Corrected Author", "year": ""})
    meta.record_detection("d1", "d1.pdf", {"title": "Detected again", "author": "Wrong", "year": "1999",
                                           "detected_from": {}})
    entry = meta.get("d1")
    assert entry["title"] == "Detected again"
    assert entry["author"] == "Corrected Author"
    assert entry["year"] is None                      # cleared by the user, shown as Unknown
    assert format_citation(entry, 1) == "[Detected again | Corrected Author | Unknown, p. 1]"


def _make_pdf_with_image(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 150), False)
    pixmap.set_rect(pixmap.irect, (40, 120, 200))
    page.insert_text((72, 80), "The requirements analysis flow is shown below.", fontsize=11)
    page.insert_image(fitz.Rect(72, 100, 372, 325), pixmap=pixmap)
    page.insert_text((72, 345), "Figure 4. Requirements analysis flow", fontsize=10)
    small = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    page.insert_image(fitz.Rect(500, 700, 520, 720), pixmap=small)   # icon, must be skipped
    doc.save(path)
    doc.close()


def test_extracts_images_with_bbox_caption_and_context(store, tmp_path) -> None:
    pdf = tmp_path / "figures.pdf"
    _make_pdf_with_image(pdf)
    summary = extract_images(store, "doc-img", pdf, tmp_path / "images")
    assert summary["images"] == 1 and summary["too_small"] >= 1
    unit = store.content_units.find_one({"unit_type": "image"})
    assert unit["page_number"] == 1
    assert unit["caption"] == "Figure 4. Requirements analysis flow"
    assert "requirements analysis flow is shown below" in unit["context_text"]
    x0, y0, x1, y1 = unit["bbox"]
    assert (round(x0), round(y0), round(x1), round(y1)) == (72, 100, 372, 325)
    asset = store.assets.find_one({"kind": "image"})
    assert Path(asset["storage"]["uri"]).name.startswith("p0001_i01_")
    assert any((tmp_path / "images").rglob("p0001_i01_*"))
    assert extract_images(store, "doc-img", pdf, tmp_path / "images")["skipped"] is True
    assert store.content_units.count_documents({"unit_type": "image"}) == 1

    concept_id = stable_id("concept", "requirements analysis")
    store.concepts.insert_one({"_id": concept_id, "canonical_name": "Requirements analysis"})
    store.database["concept_aliases"].insert_one({"_id": "requirementanalysis", "concept_id": concept_id})
    assert link_images_to_concepts(store)["images_linked"] == 1
    assert store.concepts.find_one({"_id": concept_id})["image_ids"] == [unit["_id"]]
