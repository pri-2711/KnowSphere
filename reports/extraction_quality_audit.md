# Extraction quality audit

**Audit date:** 2026-10-20  
**Status:** completed for the representative pages listed below; not a visual review of all 590 pages.

## Scope and method

Reviewed the current extraction implementation, the existing JSON metadata and text report, and representative text output. Opened the source PDFs with PyMuPDF, compared page text to the saved TXT, and rendered selected pages and every page previously labeled `empty` or low-text for visual inspection.

The audited files are `data/raw_pdfs/book1.pdf` (368 pages) and `data/raw_pdfs/book2.pdf` (222 pages). PDF metadata gives no title for `book1.pdf` and gives `Intro` for `book2.pdf`; the visible titles are *Systems Engineering: Practice and Theory* and *Systems Engineering Fundamentals*, respectively. The second document is not visibly titled “AI Fundamentals.”

The existing report records 359 `text`, 2 `low_text`, and 7 `empty` pages for book1; book2 has 221 `text` and 1 `low_text` page. `low_text` and `empty` are extractor method labels, not content-type determinations.

## Confirmed findings

### Sparse pages can be read but still trigger OCR

- **book1 PDF page 23:** the PDF text layer contains “Part 1” and “Systems Engineering Practice.” The saved TXT contains that heading, but the page is classified `low_text` because its content is under the 40-character threshold. The rendered page confirms it is a section divider, not a scan. The same pattern occurs on **page 243**, which contains “Part 2” and “New Systems Engineering Theories.”
- **book2 PDF page 6:** the page has a running header, chapter label, and printed page number but little body content. It is a sparse chapter page, not evidence of a scanned page. The existing TXT retains the header and page number.

These are confirmed false scan suspicions. The current implementation replaces short PyMuPDF text with OCR output on those pages and does not separately save the pre-OCR text.

### Mixed visual content is not classified

- **book1 PDF page 31:** the rendered page is a table with three system rows, descriptive text, and embedded schematic drawings. Text extraction gets much of the wording, but the table-cell relationships and diagram meaning cannot be represented reliably by one linear text string.
- **book2 PDF page 32:** the page is a structured table. The extracted text contains table headings and values, but their row/column association is less explicit than the rendered grid.
- **book2 PDF page 65:** the page combines explanatory prose with a labeled concept diagram. The existing extractor labels it `text` (1,181 characters); it does not identify the diagram as separate visual content.

These are confirmed limitations of the current output model. They do not establish that every table or figure was misread.

### Column order is unreliable on at least one page

- **book2 PDF page 112:** the rendered page has two columns. The saved text starts with material from the left column and immediately appends a fragment from the right column on the same line (“…established for production. cases where system technical maturity…”). The next passage then continues across columns in a visually confusing order. This is a confirmed reading-order error on this page.
- **book1 PDF page 53:** its running title is concatenated in the saved output (`Complex-SystemsComplex-Systems Design MethodologyDesignfor Systems-EngineeringMethodologyCollaborativefor Systems-EngineeringEnvironment...`). The rendered page contains a figure amid the article text. This confirms a bad header reading order; the exact effect on the main body was not audited line by line.

### Running furniture and source references

Running book/chapter headers and printed page numbers appear in extracted body text (for example, book1 page 185 and book2 page 112). Page separators use PDF page numbers, which is useful for source lookup, but printed page numbers can differ (book2 PDF page 112 is printed page 108). The TXT format does not explicitly distinguish running headers, printed page numbers, captions, and body text.

## Blank-page review

The seven book1 pages previously labeled `empty` (PDF pages 2, 6, 8, 12, 24, 244, and 367) were rendered and appeared visually blank at the inspected preview scale. They contain no extractable text or raster images. Page 367 has a vector drawing primitive but no visible content in the rendered preview. This makes “blank” plausible for these specific pages, but the extractor itself does not perform visual blank-page detection. Its `empty` status means only that OCR returned no text.

The low-text pages 23, 243, and book2 page 6 are not blank and should remain distinct from blank-page outcomes.

## Suspected issues and unverified areas

- **Possible text/reading-order defects:** other pages with multiple columns, equations, or dense diagrams may have similar interleaving. Only the cited examples were visually compared.
- **Possible table loss:** PyMuPDF can emit the words from a table without preserving a dependable cell structure. Pages 31 and 32 demonstrate the risk; no exhaustive table-by-table comparison was done.
- **Possible character-map defects:** some extracted book1 text has odd symbols or ligature encodings. A systematic count and correction review was not performed; no text has been normalized or “fixed” in this audit.
- **Scans:** no audited low-text page showed a scanned page image. This does not prove that no other PDF page contains a scan.
- **OCR accuracy:** Tesseract was used on the three low-text pages. Their visible content is short, but no broader OCR accuracy benchmark or ground-truth transcription was performed.
- **Header/footer removal:** headers and printed page numbers are preserved. Their repetition is visible, but automatic removal is not recommended without reliable region-level detection.

## Focused improvement implemented after the audit

The extractor now writes an additive `<pdf-stem>.pages.jsonl` sidecar alongside the existing TXT and metadata/report files. Each page record stores the unmodified PyMuPDF text separately from the selected text (which may be OCR), its source PDF/page reference, method, quality status, warnings, and a separate interpretations array. Content type remains `unknown` unless classified safely; no visual classifier was added.

Sparse embedded text is now retained and marked for review when a page has no raster image, rather than automatically being treated as a scan. OCR is attempted for low-text pages containing raster content. Pages with no text or raster content are marked `no_text_detected` in JSONL, with a warning that blankness is unverified and vector-only content may require review. This is a heuristic: vector-only scanned text is not recognized by this raster check.

The documented schema and example are in [`../schemas/extracted_page.schema.json`](../schemas/extracted_page.schema.json) and [`../schemas/extracted_page.example.json`](../schemas/extracted_page.example.json). On the 2026-10-21 standard run, book1 reported 361 `text` and 7 `empty` pages; book2 reported 222 `text` pages. Both books have one JSONL record per PDF page, and all 590 records validated against the schema. The existing TXT files, metadata JSON, and report were regenerated at their original paths and retain their formats. The source PDFs were not modified.
