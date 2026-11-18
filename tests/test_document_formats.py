"""DOCX, HTML and Markdown adapters, their ingestion through the PDF page path, upload checks, and the
document quality report (digital, scanned, mixed)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.document_intelligence import text_formats
from src.document_intelligence.quality import assess
from src.orchestration.documents import accepted_name, ingest_text_file, upload_problem

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(path: Path) -> Path:
    def para(text: str, style: str | None = None) -> str:
        props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r></w:p>"
    body = (para("Thermal Control", "Heading1") + para("Radiators reject heat to space.")
            + para("Heat Pipes", "Heading2") + para("Heat pipes move heat with a working fluid.")
            + '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Fluid</w:t></w:r></w:p></w:tc>'
              '<w:tc><w:p><w:r><w:t>Ammonia</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')
        archive.writestr("docProps/core.xml", '<cp:coreProperties xmlns:cp="x" xmlns:dc="http://purl.org/dc/'
                                              'elements/1.1/"><dc:title>Thermal Handbook</dc:title></cp:coreProperties>')
    return path


def test_docx_html_and_markdown_become_sections_with_heading_paths(tmp_path) -> None:
    docx = text_formats.read(make_docx(tmp_path / "thermal.docx"))
    assert docx.title == "Thermal Handbook" and docx.format == "docx"
    assert [s.heading_path for s in docx.sections] == [["Thermal Control"], ["Thermal Control", "Heat Pipes"]]
    assert "Fluid | Ammonia" in docx.sections[1].text

    html = tmp_path / "page.html"
    html.write_text("<html><head><title>Power</title><script>var x = 1;</script></head><body><nav>Menu</nav>"
                    "<h1>Power System</h1><p>Solar arrays generate <b>power</b>.</p><h2>Batteries</h2>"
                    "<ul><li>Lithium-ion cells store energy.</li></ul></body></html>", encoding="utf-8")
    page = text_formats.read(html)
    assert page.title == "Power" and "var x" not in str(page.sections) and "Menu" not in str(page.sections)
    assert page.sections[0].text == "Power System\nSolar arrays generate power."
    assert page.sections[1].heading_path == ["Power System", "Batteries"]

    markdown = tmp_path / "notes.md"
    markdown.write_text("# Life Support\n\nThe ECLSS recycles\nwater and air.\n\n## Oxygen\n\n- Electrolysis "
                        "splits water.\n\n```\ncode block\n```\n", encoding="utf-8")
    notes = text_formats.read(markdown)
    assert notes.title == "Life Support"
    assert notes.sections[0].text == "Life Support\nThe ECLSS recycles water and air."
    assert notes.sections[1].heading_path == ["Life Support", "Oxygen"] and "code block" in notes.sections[1].text


def test_long_sections_are_split_into_bounded_logical_pages() -> None:
    sections = [text_formats.Section(["A"], "\n".join(f"paragraph {i} " + "x" * 90 for i in range(50))),
                text_formats.Section(["B"], "short tail")]
    pages = text_formats.to_pages(sections, page_chars=1000)
    assert [p["page_number"] for p in pages] == list(range(1, len(pages) + 1)) and len(pages) >= 5
    assert all(len(p["text"]) <= 1000 for p in pages)
    assert "".join(p["text"].replace("\n", "") for p in pages).count("paragraph") == 50   # nothing lost
    assert pages[0]["section_path"] == ["A"]


def test_text_documents_are_stored_like_pdf_pages(store, metadata, tmp_path) -> None:
    folder = tmp_path / "upload"
    folder.mkdir()
    path = make_docx(folder / "thermal.docx")
    first = ingest_text_file(store, metadata, path, folder / "extracted", project_id="default")
    again = ingest_text_file(store, metadata, path, folder / "extracted", project_id="default")
    assert again["pages"] == "unchanged" and again["document_id"] == first["document_id"]
    document = store.documents.find_one({"_id": first["document_id"]})
    assert document["format"] == "docx" and document["file_storage"]["media_type"].endswith("document")
    pages = list(store.content_units.find({"document_id": first["document_id"], "unit_type": "page"}))
    assert len(pages) == document["page_count"] == 1
    assert "Heat pipes move heat" in pages[0]["text"] and pages[0]["extraction_method"] == "docx"
    assert metadata.get(first["document_id"])["title"] == "Thermal Handbook"


def test_uploads_must_match_their_type(tmp_path) -> None:
    fake_pdf = tmp_path / "a.pdf"
    fake_pdf.write_bytes(b"hello")
    binary_md = tmp_path / "b.md"
    binary_md.write_bytes(b"\x00\x01\xff")
    good_md = tmp_path / "c.md"
    good_md.write_text("# Title", encoding="utf-8")
    assert upload_problem(fake_pdf) == "The file is not a valid PDF file"
    assert upload_problem(binary_md) and upload_problem(good_md) is None
    assert upload_problem(make_docx(tmp_path / "d.docx")) is None
    assert accepted_name("x.HTM") and not accepted_name("x.exe")


def test_quality_report_tells_digital_scanned_and_mixed_apart() -> None:
    digital = [{"text": "Page text " * 20, "extraction_method": "pymupdf", "quality_status": "usable"}] * 3
    scanned = [{"text": "OCR text", "extraction_method": "tesseract", "quality_status": "usable"}] * 9 + \
              [{"text": "", "extraction_method": "none", "quality_status": "no_text_detected"}]
    assert assess(digital)["kind"] == "digital" and not assess(digital)["needs_review"]
    report = assess(scanned)
    assert report["kind"] == "scanned" and report["needs_review"] and report["ocr_pages"] == 9
    assert any("without text" in warning for warning in report["warnings"])
    assert assess(digital + scanned)["kind"] == "mixed"
    assert assess([])["kind"] == "empty"
    listing = [{"extraction_method": "tesseract", "quality_status": "usable"}]        # no text loaded
    assert assess(listing)["kind"] == "scanned"


def test_the_watched_folder_picks_up_text_documents(store, tmp_path) -> None:
    from src.orchestration.watcher import FolderWatcher
    (tmp_path / "notes.md").write_text("# Notes", encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"MZ")
    make_docx(tmp_path / "brief.docx")
    watcher = FolderWatcher(store, tmp_path)
    assert watcher.scan() == []                       # the first scan waits: a copy may still be in progress
    events = watcher.scan()
    assert sorted(Path(event.relative_path).name for event in events) == ["brief.docx", "notes.md"]
