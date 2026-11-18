"""Visual extraction: concepts and relationships from figures and diagrams (candidates only).

Text and visual extraction produce the same record shape (``extraction`` with
local concepts and relationships, validated by ``evidence_quality.validation``),
so cross-document enrichment, governance and retrieval treat both alike.

Grounding: a vision model's answer is generated, so its evidence must be text
that can be checked. Each image unit gets a *grounding text*: its caption, the
page text around it, and Tesseract OCR of the image itself (labels inside the
diagram). The model must quote that text; a concept whose quote and name are
not found there is rejected, exactly as for a text chunk. The model's free-text
description is stored apart as ``interpretation`` and is never used as evidence.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.contracts.interfaces import VisionModel
from src.evidence_quality.injection import UNTRUSTED_RULE, fence
from src.evidence_quality.validation import validate_local
from src.knowledge_extraction.llm_schema import EXTRACTION_SCHEMA
from src.knowledge_extraction.recovery import run_with_recovery

VISUAL_PROMPT_VERSION = "visual-2"   # 2: untrusted-text rule
EXTRACTED, FAILED = "extracted", "failed"
OCR = Callable[[Path], str]

VISUAL_SCHEMA: dict[str, Any] = {
    **EXTRACTION_SCHEMA,
    "properties": {**EXTRACTION_SCHEMA["properties"], "description": {"type": "string"}},
    "required": [*EXTRACTION_SCHEMA["required"], "description"],
}

VISUAL_SYSTEM_PROMPT = """You read one figure, diagram or table image from a technical document.

You also get the figure's caption, the text printed next to it, and machine-read (OCR) text of labels
inside the image. Together that is the GROUNDING TEXT.
- "description": two or three sentences on what the image shows (kept apart from the evidence).
- "concepts": technical concepts the image shows or labels. "evidence" must be copied exactly from the
  GROUNDING TEXT (a label, a caption phrase or a sentence). Never quote text that is not in it.
- "relationships": only connections the image clearly draws (arrows, containment, flow) between two of
  your concepts; "evidence" is a quote from the GROUNDING TEXT naming both concepts.
- Use the same concept types and relationship types as the schema allows. If the image is decorative or
  unreadable, return empty lists.

""" + UNTRUSTED_RULE


def grounding_text(unit: Mapping[str, Any]) -> str:
    parts = [unit.get("caption") or "", unit.get("context_text") or "", unit.get("ocr_text") or ""]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def visual_user_prompt(unit: Mapping[str, Any], grounding: str, document_title: str | None) -> str:
    return "\n".join([f"Document: {document_title or 'Unknown'}", f"Page: {unit.get('page_number')}", "",
                      "<grounding_text>", fence(grounding), "</grounding_text>"])


def pseudo_chunk(unit: Mapping[str, Any], text: str) -> dict[str, Any]:
    """The image unit seen as a one-span chunk, so the text validation rules apply unchanged."""
    span = {"content_unit_id": unit["_id"], "page_number": unit.get("page_number"), "start": 0, "end": len(text),
            "chunk_start": 0, "chunk_end": len(text)}
    return {"_id": unit["_id"], "document_id": unit["document_id"], "text": text, "spans": [span],
            "page_number": unit.get("page_number"), "page_end": unit.get("page_number")}


def visual_is_current(unit: Mapping[str, Any], model: str) -> bool:
    record = unit.get("extraction")
    text = unit.get("grounding_text")
    return bool(record) and bool(text) and record.get("status") == EXTRACTED and record.get("method") == "vision" \
        and record.get("text_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest() \
        and record.get("prompt_version") == VISUAL_PROMPT_VERSION and record.get("model") == model


def extract_image_unit(unit: Mapping[str, Any], image_path: Path, model: VisionModel, max_retries: int,
                       ocr: OCR | None = None, document_title: str | None = None) -> dict[str, Any]:
    """Return the fields to store on the image unit: OCR text, grounding text, extraction, interpretation."""
    fields: dict[str, Any] = {}
    if unit.get("ocr_text") is None:
        fields["ocr_text"] = ocr(image_path) if ocr is not None else ""
        fields["ocr_method"] = "tesseract" if ocr is not None else None
    merged = {**unit, **fields}
    text = grounding_text(merged)
    fields["grounding_text"] = text
    record: dict[str, Any] = {"status": FAILED, "method": "vision", "kind": "image", "model": model.model,
                              "prompt_version": VISUAL_PROMPT_VERSION,
                              "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                              "extracted_at": datetime.now(timezone.utc), "concepts": [], "relationships": [],
                              "attempts": [], "retries": 0, "error": None}
    if not text:
        record.update(status=EXTRACTED, error="no caption, context or OCR text to ground the image; skipped")
        fields["extraction"] = record
        return fields
    first = visual_user_prompt(merged, text, document_title)

    def call(messages: list[dict[str, Any]]) -> Any:
        # A vision call is single-turn: a repair request is appended to the original prompt.
        prompt = first if len(messages) == 1 else f"{first}\n\n{messages[-1]['content']}"
        return model.describe(image_path, VISUAL_SYSTEM_PROMPT, prompt, VISUAL_SCHEMA)

    result = run_with_recovery(call, first, max_retries)
    record.update(attempts=result.attempts, retries=result.retries, error=result.error,
                  status=EXTRACTED if result.ok else FAILED)
    if result.ok and result.data is not None:
        record["concepts"], record["relationships"] = validate_local(result.data, pseudo_chunk(merged, text))
        fields["interpretation"] = {"description": str(result.data.get("description", "")), "generated": True,
                                    "model": model.model, "prompt_version": VISUAL_PROMPT_VERSION}
    fields["extraction"] = record
    return fields
