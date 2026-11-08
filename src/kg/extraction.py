"""Pass 1: local concept and relationship extraction for one semantic chunk.

The LLM call goes through ``recovery.run_with_recovery``; the parsed answer goes
through ``validation.validate_local``. The result is a plain record stored on the
chunk (``content_units.extraction``), so a chunk is only sent again when its text,
the prompt version or the model changes, or when its last extraction failed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from src.kg.llm import LLMClient
from src.kg.prompts import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt
from src.kg.recovery import run_with_recovery
from src.kg.schema import EXTRACTION_SCHEMA
from src.kg.validation import validate_local

EXTRACTED, FAILED = "extracted", "failed"


def extraction_is_current(chunk: Mapping[str, Any], model: str) -> bool:
    """True when the stored result can be reused for this chunk text, prompt and model."""
    record = chunk.get("extraction")
    return bool(record) and record.get("status") == EXTRACTED and record.get("text_sha256") == chunk["text_sha256"] \
        and record.get("prompt_version") == PROMPT_VERSION and record.get("model") == model


def extract_chunk(chunk: Mapping[str, Any], client: LLMClient, max_retries: int,
                  document_title: str | None = None) -> dict[str, Any]:
    """Run Pass 1 on one chunk and return its extraction record (never raises for a bad answer)."""
    result = run_with_recovery(
        lambda messages: client.complete(SYSTEM_PROMPT, messages, EXTRACTION_SCHEMA),
        user_prompt(chunk, document_title),
        max_retries,
    )
    record: dict[str, Any] = {
        "status": EXTRACTED if result.ok else FAILED,
        "text_sha256": chunk["text_sha256"],
        "prompt_version": PROMPT_VERSION,
        "model": client.model,
        "attempts": result.attempts,
        "retries": result.retries,
        "error": result.error,
        "extracted_at": datetime.now(timezone.utc),
        "concepts": [],
        "relationships": [],
    }
    if result.ok and result.data is not None:
        concepts, relationships = validate_local(result.data, chunk)
        record["concepts"], record["relationships"] = concepts, relationships
    else:
        record["status_detail"] = "needs_review"   # recovery failed: kept for review, not silently dropped
    return record
