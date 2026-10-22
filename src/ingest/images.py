"""Extract embedded images with PyMuPDF and store them as independent content units.

For each image we store the page, bounding box, file path, a caption when a
"Figure/Table N" text block sits next to it, and the text around it on the page
(context). Images are linked to concepts by finding concept names in that text.

Optional captioning with a small ViT-GPT2 model is off by default. A model
caption is generated text, so it is stored separately as ``model_caption`` and
is never used for answers or citations.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.storage.mongodb_store import MongoKnowledgeStore, _relative_file_uri, stable_id
from src.textutil import concept_key

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - reported at runtime
    fitz = None

IMAGE_VERSION = "images-1"
MIN_PIXELS = 100            # skip icons and equation fragments smaller than this (width or height)
MIN_AREA_FRACTION = 0.01    # skip images covering less than 1 percent of the page
MAX_REPEAT_PAGES = 3        # an identical image on more pages is treated as a logo or decoration
CAPTION_DISTANCE = 80.0     # points between image edge and caption block
CONTEXT_DISTANCE = 150.0    # points around the image collected as context
MAX_CONTEXT_CHARS = 1500
_CAPTION = re.compile(r"^\s*(?:figure|fig\.?|table|exhibit|chart|diagram|graph)\s*[\dA-Z]+[\-.:\d]*", re.IGNORECASE)
_WEB_FORMATS = {"png", "jpg", "jpeg", "gif"}

logger = logging.getLogger("knowledge_map.images")


class Captioner(Protocol):
    model_name: str

    def caption(self, image_path: Path) -> str:
        ...


class VitCaptioner:
    """Optional ViT-GPT2 image captioning (transformers). Loaded only when enabled."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def caption(self, image_path: Path) -> str:
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("image-to-text", model=self.model_name)
        result = self._pipeline(str(image_path))
        return (result[0].get("generated_text") or "").strip() if result else ""


def _blocks(page: Any) -> list[tuple[Any, str]]:
    blocks = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        if len(block) > 6 and block[6] != 0:  # image block
            continue
        if text and text.strip():
            blocks.append((fitz.Rect(x0, y0, x1, y1), text.strip()))
    return blocks


def _vertical_gap(a: Any, b: Any) -> float:
    if b.y0 >= a.y1:
        return b.y0 - a.y1
    if a.y0 >= b.y1:
        return a.y0 - b.y1
    return 0.0


def _horizontal_overlap(a: Any, b: Any) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 0


def find_caption_and_context(page: Any, rect: Any) -> tuple[str | None, str]:
    """Caption: nearest 'Figure/Table N' block below or above the image. Context: nearby text blocks."""
    caption, caption_gap = None, None
    context_parts = []
    for block_rect, text in _blocks(page):
        gap = _vertical_gap(rect, block_rect)
        if _CAPTION.match(text) and gap <= CAPTION_DISTANCE and _horizontal_overlap(rect, block_rect):
            # Prefer a caption below the image when two are equally close.
            adjusted = gap - (1.0 if block_rect.y0 >= rect.y1 else 0.0)
            if caption_gap is None or adjusted < caption_gap:
                caption, caption_gap = re.sub(r"\s+", " ", text), adjusted
            continue
        if gap <= CONTEXT_DISTANCE:
            context_parts.append(text)
    context = re.sub(r"[ \t]+", " ", "\n".join(context_parts))[:MAX_CONTEXT_CHARS]
    return caption, context


def _save_image(doc: Any, xref: int, target_stem: Path) -> tuple[Path, str]:
    info = doc.extract_image(xref)
    ext = (info.get("ext") or "").lower()
    if ext in _WEB_FORMATS:
        path = target_stem.with_suffix("." + ext)
        path.write_bytes(info["image"])
        return path, "image/png" if ext == "png" else ("image/gif" if ext == "gif" else "image/jpeg")
    pixmap = fitz.Pixmap(doc, xref)
    if pixmap.n - pixmap.alpha >= 4:  # CMYK and similar
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
    path = target_stem.with_suffix(".png")
    pixmap.save(path)
    return path, "image/png"


def extract_images(
    store: MongoKnowledgeStore,
    document_id: str,
    pdf_path: Path,
    image_root: Path,
    captioner: Captioner | None = None,
) -> dict[str, Any]:
    """Extract qualifying images of one document; skipped when this PDF hash was already processed."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is missing.")
    state = store.database["image_state"].find_one({"_id": document_id})
    if state and state.get("version") == IMAGE_VERSION and (captioner is None or state.get("captioned")):
        return {"document_id": document_id, "skipped": True, "images": state.get("images", 0)}

    out_dir = image_root / document_id[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"document_id": document_id, "skipped": False, "images": 0, "too_small": 0, "repeated": 0}
    with fitz.open(pdf_path) as doc:
        pages_by_digest: dict[str, set[int]] = defaultdict(set)
        digest_by_xref: dict[int, str] = {}
        for page in doc:
            for image in page.get_images(full=True):
                xref = image[0]
                if xref not in digest_by_xref:
                    digest_by_xref[xref] = hashlib.sha256(doc.xref_stream_raw(xref) or b"").hexdigest()
                pages_by_digest[digest_by_xref[xref]].add(page.number)

        for page in doc:
            page_number = page.number + 1
            page_area = page.rect.width * page.rect.height
            index = 0
            for image in page.get_images(full=True):
                xref, width, height = image[0], image[2], image[3]
                if width < MIN_PIXELS or height < MIN_PIXELS:
                    summary["too_small"] += 1
                    continue
                digest = digest_by_xref[xref]
                if len(pages_by_digest[digest]) > MAX_REPEAT_PAGES:
                    summary["repeated"] += 1
                    continue
                rects = page.get_image_rects(xref)
                if not rects or (rects[0].width * rects[0].height) < MIN_AREA_FRACTION * page_area:
                    summary["too_small"] += 1
                    continue
                rect = rects[0]
                index += 1
                path, media_type = _save_image(doc, xref, out_dir / f"p{page_number:04d}_i{index:02d}_{digest[:8]}")
                caption, context = find_caption_and_context(page, rect)
                model_caption = captioner.caption(path) if captioner is not None else None
                asset_key = f"image-p{page_number}-{index}"
                asset_id = stable_id("asset", document_id, asset_key)
                uri = _relative_file_uri(path, store.project_root)
                bbox = [round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)]
                store._upsert(store.assets, {
                    "_id": asset_id, "asset_id": asset_key, "document_id": document_id,
                    "page_number": page_number, "kind": "image", "caption": caption,
                    "bbox": bbox, "width": width, "height": height, "sha256": digest,
                    "storage": {"provider": "local", "uri": uri, "media_type": media_type},
                    "model_caption": model_caption,
                    "model_caption_source": captioner.model_name if captioner is not None else None,
                })
                store._upsert(store.content_units, {
                    "_id": stable_id("content_unit", document_id, "image", page_number, index),
                    "document_id": document_id, "unit_type": "image", "unit_index": index,
                    "page_number": page_number,
                    "text": "\n".join(part for part in (caption, context) if part),
                    "caption": caption, "caption_source": "nearby_text" if caption else None,
                    "context_text": context, "bbox": bbox, "asset_ids": [asset_id],
                    "content_type": "image", "extraction_method": "pymupdf", "quality_status": "usable",
                    "image_uri": uri, "model_caption": model_caption,
                })
                summary["images"] += 1
    store.database["image_state"].update_one(
        {"_id": document_id},
        {"$set": {"version": IMAGE_VERSION, "images": summary["images"], "captioned": captioner is not None,
                  "processed_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return summary


def link_images_to_concepts(store: MongoKnowledgeStore, max_words: int = 5) -> dict[str, int]:
    """Attach concept IDs to image units whose caption or context contains a concept name or alias."""
    aliases = {record["_id"]: record["concept_id"] for record in store.database["concept_aliases"].find()}
    images_by_concept: dict[str, list[str]] = defaultdict(list)
    linked = 0
    for unit in store.content_units.find({"unit_type": "image"}, {"text": 1}):
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", unit.get("text") or "")
        found: set[str] = set()
        for size in range(1, max_words + 1):
            for start in range(0, len(words) - size + 1):
                concept_id = aliases.get(concept_key(" ".join(words[start:start + size])))
                if concept_id:
                    found.add(concept_id)
        store.content_units.update_one({"_id": unit["_id"]}, {"$set": {"concept_ids": sorted(found)}})
        for concept_id in found:
            images_by_concept[concept_id].append(unit["_id"])
        linked += bool(found)
    store.concepts.update_many({}, {"$set": {"image_ids": []}})
    for concept_id, unit_ids in images_by_concept.items():
        store.concepts.update_one({"_id": concept_id}, {"$set": {"image_ids": sorted(unit_ids)}})
    return {"images_linked": linked, "concepts_with_images": len(images_by_concept)}
