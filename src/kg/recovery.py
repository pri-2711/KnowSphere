"""Recovery for malformed or incomplete LLM output, one chunk at a time.

``parse_output`` turns a raw answer into data or a precise error. It only repairs
syntax that cannot change meaning (a Markdown code fence around the JSON, text
before or after the JSON object, trailing commas). Anything else is sent back to
the model with the error, at most ``max_retries`` times. Every raw answer and
error is kept. When recovery fails, the chunk is marked for review; nothing is
invented to fill the gap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from src.kg.llm import LLMError, LLMFatalError, LLMResponse
from src.kg.prompts import repair_prompt

MAX_STORED_RAW = 20_000   # characters of each raw answer kept for debugging

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


@dataclass
class RecoveryResult:
    ok: bool
    data: dict[str, Any] | None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def structure_errors(data: Any) -> list[str]:
    """Shape errors that make the answer unusable as a whole (item content is checked in validation)."""
    if not isinstance(data, dict):
        return ["the answer must be a JSON object with 'concepts' and 'relationships'"]
    errors = []
    for key, fields in (("concepts", ("name", "type", "description", "aliases", "evidence")),
                        ("relationships", ("source", "type", "target", "evidence"))):
        items = data.get(key)
        if not isinstance(items, list):
            errors.append(f"'{key}' must be a list")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{key}[{index}] must be an object")
                continue
            missing = [name for name in fields if name not in item]
            if missing:
                errors.append(f"{key}[{index}] is missing {', '.join(missing)}")
            for name in fields:
                value = item.get(name)
                if name == "aliases":
                    if value is not None and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
                        errors.append(f"{key}[{index}].aliases must be a list of strings")
                elif value is not None and not isinstance(value, str):
                    errors.append(f"{key}[{index}].{name} must be a string")
    return errors[:10]


def parse_output(response: LLMResponse) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Return (data, error, local repairs applied)."""
    if response.stop_reason == "max_tokens":
        return None, "the answer was cut off before the JSON was complete; return fewer, shorter items", []
    if response.stop_reason == "refusal":
        return None, "the model declined this passage", []
    text = response.text.strip()
    repairs: list[str] = []
    fence = _FENCE.match(text)
    if fence:
        text, repairs = fence.group(1), repairs + ["removed code fence"]
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, "the answer did not contain a JSON object", repairs
        text, repairs = text[start:end + 1], repairs + ["removed text around the JSON object"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cleaned = _TRAILING_COMMA.sub(r"\1", text)
        try:
            data = json.loads(cleaned)
            repairs.append("removed trailing commas")
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}", repairs
    errors = structure_errors(data)
    if errors:
        return None, "the JSON does not follow the schema: " + "; ".join(errors), repairs
    return data, None, repairs


def run_with_recovery(
    call: Callable[[Sequence[dict[str, Any]]], LLMResponse],
    first_message: str,
    max_retries: int,
) -> RecoveryResult:
    """Ask once, then repair up to ``max_retries`` times. Raises only on fatal client errors."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": first_message}]
    attempts: list[dict[str, Any]] = []
    error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call(messages)
        except LLMFatalError:
            raise
        except LLMError as exc:
            attempts.append({"attempt": attempt + 1, "raw": None, "error": str(exc), "stop_reason": None})
            error = str(exc)
            if not exc.retryable:
                break
            continue
        data, error, repairs = parse_output(response)
        attempts.append({"attempt": attempt + 1, "raw": response.text[:MAX_STORED_RAW], "error": error,
                         "stop_reason": response.stop_reason, "repairs": repairs, "model": response.model,
                         "request_id": response.request_id, "usage": response.usage})
        if error is None:
            return RecoveryResult(True, data, attempts)
        if response.stop_reason == "refusal":
            break  # asking again with the same passage would be declined again
        messages = messages + [{"role": "assistant", "content": response.text or "(empty answer)"},
                               {"role": "user", "content": repair_prompt(error)}]
    return RecoveryResult(False, None, attempts, error)
