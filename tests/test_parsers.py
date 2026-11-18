"""Parser adapters: PyMuPDF structure on a generated PDF, Docling through a fake converter, structure storage."""

from __future__ import annotations

from types import SimpleNamespace

import fitz
import pytest

from src.contracts.interfaces import DocumentParser
from src.contracts.schemas import UnitType
from src.document_intelligence.parsers import DoclingParser, PyMuPDFStructureParser, select_parser
from src.knowledge_repository.repository import MongoKnowledgeRepository


def _pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Water Recovery", fontsize=20)
    page.insert_text((72, 100), "Urine Processing", fontsize=15)
    page.insert_text((72, 130), "The processor distills urine and returns water to the potable bus.", fontsize=11)
    page.insert_text((72, 150), "It runs continuously during crewed operations of the station.", fontsize=11)
    page.insert_text((72, 175), "Q = 4.5 x 10^3 + 2", fontsize=11)
    page.insert_text((72, 200), "Figure 1 Water loop schematic", fontsize=11)
    x0, y0, w, h = 72, 240, 120, 24
    for row in range(3):
        for col in range(2):
            page.draw_rect(fitz.Rect(x0 + col * w, y0 + row * h, x0 + (col + 1) * w, y0 + (row + 1) * h), width=1)
            page.insert_text((x0 + col * w + 6, y0 + row * h + 16), f"r{row}c{col}", fontsize=10)
    second = doc.new_page()
    second.insert_text((72, 72), "Air Revitalization", fontsize=20)
    second.insert_text((72, 100), "Carbon dioxide is removed by a sorbent bed in the cabin loop.", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_pymupdf_parser_keeps_pages_sections_tables_figures_equations(tmp_path) -> None:
    parser = PyMuPDFStructureParser()
    assert isinstance(parser, DocumentParser) and parser.available()
    parsed = parser.parse(_pdf(tmp_path / "eclss.pdf"), "p1", "doc-1", "version-1")
    by_type = {kind: [u for u in parsed.units if u.unit_type is kind] for kind in UnitType}
    sections = by_type[UnitType.SECTION]
    assert [(u.text, u.page_number) for u in sections] == [("Water Recovery", 1), ("Urine Processing", 1),
                                                          ("Air Revitalization", 2)]
    assert sections[1].section_path == ["Water Recovery", "Urine Processing"]
    assert sections[2].section_path == ["Air Revitalization"]                       # hierarchy restarts
    assert [u.text for u in by_type[UnitType.FIGURE]] == ["Figure 1 Water loop schematic"]
    assert by_type[UnitType.EQUATION][0].text == "Q = 4.5 x 10^3 + 2"
    assert by_type[UnitType.EQUATION][0].provenance.method == "pymupdf-heuristic"
    table = by_type[UnitType.TABLE][0]
    assert "r0c0 | r0c1" in table.text and table.page_number == 1 and len(table.bbox) == 4
    assert all(u.project_id == "p1" and u.document_version_id == "version-1" for u in parsed.units)
    assert parsed.page_count == 2


class _FakeDoclingDocument:
    version = "2.0"
    pages = {1: SimpleNamespace(size=SimpleNamespace(height=800.0))}

    def iterate_items(self):
        box = SimpleNamespace(l=10, t=700, r=200, b=680, to_top_left_origin=lambda page_height: SimpleNamespace(
            l=10, t=100, r=200, b=120))
        prov = [SimpleNamespace(page_no=1, bbox=box)]
        table = SimpleNamespace(label="table", prov=prov, export_to_markdown=lambda doc: "| a | b |")
        yield SimpleNamespace(label="section_header", level=1, text="Water Recovery", prov=prov), 1
        yield SimpleNamespace(label="text", text="Body text", prov=prov), 2
        yield table, 2
        yield SimpleNamespace(label="formula", text="E = mc^2", prov=prov), 2
        yield SimpleNamespace(label="picture", prov=prov, caption_text=lambda doc: "Figure 2 Loop"), 2


def test_docling_adapter_maps_items_to_the_same_units() -> None:
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=_FakeDoclingDocument()))
    parser = DoclingParser(lambda: converter)
    parsed = parser.parse("unused.pdf", "p1", "doc-1")
    assert [(u.unit_type, u.text) for u in parsed.units] == [
        (UnitType.SECTION, "Water Recovery"), (UnitType.TABLE, "| a | b |"),
        (UnitType.EQUATION, "E = mc^2"), (UnitType.FIGURE, "Figure 2 Loop")]
    assert parsed.units[0].bbox == [10.0, 100.0, 200.0, 120.0]                       # top-left origin
    assert parsed.parser.method == "docling" and parsed.units[1].section_path == ["Water Recovery"]


def test_parser_selection_falls_back_to_pymupdf(monkeypatch) -> None:
    monkeypatch.setattr(DoclingParser, "available", lambda self: False)
    assert select_parser("auto").name == "pymupdf"
    with pytest.raises(RuntimeError, match="not installed"):
        select_parser("docling")
    with pytest.raises(ValueError):
        select_parser("word")


def test_structure_units_are_added_without_touching_pages_or_other_projects(store, tmp_path) -> None:
    store.documents.insert_one({"_id": "doc-1", "project_id": "p1", "filename": "eclss.pdf", "page_count": 2})
    store.content_units.insert_one({"_id": "page-1", "document_id": "doc-1", "project_id": "p1",
                                    "unit_type": "page", "unit_index": 1, "page_number": 1, "text": "original"})
    store.content_units.insert_one({"_id": "other", "document_id": "doc-9", "project_id": "p2",
                                    "unit_type": "section", "unit_index": 1, "page_number": 1, "text": "keep"})
    repository = MongoKnowledgeRepository(store)
    parsed = PyMuPDFStructureParser().parse(_pdf(tmp_path / "eclss.pdf"), "p1", "doc-1")
    first = repository.save_structure(parsed)
    again = repository.save_structure(parsed)                                        # idempotent
    assert first["section"] == again["section"] == 3 and again["removed"] == len(parsed.units)
    assert store.content_units.find_one({"_id": "page-1"})["text"] == "original"
    assert store.content_units.find_one({"_id": "other"})["text"] == "keep"
    units = repository.content_units("p1", ["section"], "doc-1")
    assert [u.section_path[-1] for u in units] == ["Water Recovery", "Urine Processing", "Air Revitalization"]
    assert store.documents.find_one({"_id": "doc-1"})["structure"]["parser"] == "pymupdf"
