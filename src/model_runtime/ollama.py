"""Local inference through Ollama: Qwen text models and Qwen2.5-VL vision models.

Implements ``contracts.interfaces.TextModel`` and ``VisionModel`` with the Ollama
chat API (``POST /api/chat``). Structured output uses Ollama's ``format`` field
with the same JSON schema the Claude provider uses, so extraction code does not
know which provider answered. ``temperature`` is 0 so a rerun gives the same answer.

The HTTP transport is injectable; tests pass a fake and nothing reaches a server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import RuntimeConfig
from src.contracts.interfaces import LLMError, LLMFatalError, LLMResponse

# (url, json body, timeout seconds) -> (status code, parsed JSON body or None, raw text)
Transport = Callable[[str, dict[str, Any], float], tuple[int, Any, str]]

_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


def httpx_transport(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any, str]:
    import httpx
    try:
        response = httpx.post(url, json=body, timeout=timeout)
    except httpx.ConnectError as exc:
        raise LLMFatalError(f"Ollama is not reachable at {url}: start it with 'ollama serve'") from exc
    except httpx.TimeoutException as exc:
        raise LLMError(f"Ollama timed out after {timeout} s: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
    try:
        data = response.json()
    except ValueError:
        data = None
    return response.status_code, data, response.text


class OllamaModel:
    """One Ollama model. ``complete`` for text, ``describe`` for an image (vision models only)."""

    def __init__(self, model: str, config: RuntimeConfig | None = None, transport: Transport | None = None) -> None:
        self.model = model
        self.config = config or RuntimeConfig()
        self.transport = transport or httpx_transport

    # ------------------------------------------------------------ protocol methods
    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        chat = [{"role": "system", "content": system}] + [
            {"role": message["role"], "content": _text_of(message["content"])} for message in messages]
        return self._chat(chat, schema)

    def describe(self, image_path: Path, system: str, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        image = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        chat = [{"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [image]}]
        return self._chat(chat, schema)

    def unload(self) -> None:
        """Ask Ollama to free this model's memory now (used by the router before loading another model)."""
        try:
            self.transport(f"{self.config.ollama_url}/api/generate", {"model": self.model, "keep_alive": 0}, 30)
        except LLMError:
            pass   # nothing loaded or server gone: nothing to free

    # ------------------------------------------------------------ transport
    def _chat(self, chat: list[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        body = {"model": self.model, "messages": chat, "stream": False, "format": schema,
                "keep_alive": self.config.keep_alive,
                "options": {"temperature": 0, "num_ctx": self.config.ollama_num_ctx}}
        status, data, raw = self.transport(f"{self.config.ollama_url}/api/chat", body,
                                           self.config.ollama_timeout_seconds)
        if status == 404:
            raise LLMFatalError(f"Ollama model '{self.model}' is not installed: run 'ollama pull {self.model}'")
        if status >= 500:
            raise LLMError(f"Ollama server error {status}: {raw[:300]}")
        if status >= 400:
            raise LLMError(f"Ollama rejected the request ({status}): {raw[:300]}", retryable=False)
        if not isinstance(data, dict) or "message" not in data:
            raise LLMError(f"Unexpected Ollama answer: {raw[:300]}")
        usage = {"input_tokens": int(data.get("prompt_eval_count") or 0),
                 "output_tokens": int(data.get("eval_count") or 0)}
        return LLMResponse(text=data["message"].get("content") or "",
                           stop_reason=_STOP_REASONS.get(data.get("done_reason") or "stop", data.get("done_reason")),
                           model=data.get("model") or self.model, usage=usage)


def _text_of(content: Any) -> str:
    """Messages may carry a string or Anthropic-style content blocks; Ollama wants plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return json.dumps(content)


def ollama_status(config: RuntimeConfig | None = None, timeout: float = 3.0) -> dict[str, Any]:
    """Whether the local Ollama server answers and which models are pulled (``GET /api/tags``).
    A status read only: nothing is loaded and no inference runs."""
    import httpx
    config = config or RuntimeConfig()
    try:
        response = httpx.get(config.ollama_url.rstrip("/") + "/api/tags", timeout=timeout)
        models = [{"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 2)}
                  for m in (response.json() or {}).get("models", [])]
        return {"reachable": response.status_code == 200, "url": config.ollama_url, "models": models}
    except (httpx.HTTPError, ValueError) as exc:
        return {"reachable": False, "url": config.ollama_url, "error": f"{type(exc).__name__}: {exc}"}
