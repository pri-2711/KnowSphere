"""LLM access for Pass 1. Business logic only sees the small ``LLMClient`` protocol.

``AnthropicClient`` calls the Claude Messages API with JSON-schema structured
output. Credentials come from the environment (``ANTHROPIC_API_KEY`` in ``.env``,
or an ``ant auth login`` profile); nothing is hard-coded. Tests use a fake client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from src.kg.config import LLMConfig

# Models that accept the server-side refusal fallback ("fallbacks": "default").
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResponse:
    text: str
    stop_reason: str | None = None
    model: str | None = None
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """A failed call for one chunk. ``retryable`` is False when the same request would fail again."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMFatalError(LLMError):
    """A failure that affects every chunk (bad credentials, unknown model): stop the run."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        ...


class AnthropicClient:
    """Claude via the official SDK. The SDK itself retries 429, 5xx and connection errors twice."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model
        self._client: Any = None

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=self.config.timeout_seconds, max_retries=2)
        return self._client

    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        import anthropic

        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": list(messages),
            "output_config": {"effort": self.config.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        client = self._sdk()
        try:
            if self.config.use_fallbacks and self.config.model in _FALLBACK_MODELS:
                response = client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **request)
            else:
                response = client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as exc:
            raise LLMFatalError(f"{type(exc).__name__}: {exc}") from exc
        except TypeError as exc:
            if "authentication" in str(exc).lower():   # no API key, token or profile configured
                raise LLMFatalError(f"No Claude credentials found: {exc}") from exc
            raise
        except anthropic.BadRequestError as exc:
            raise LLMError(f"BadRequestError: {exc}", retryable=False) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"RateLimitError (after SDK retries): {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"APIStatusError {exc.status_code}: {exc}", retryable=exc.status_code >= 500) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"APIConnectionError: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return LLMResponse(text=text, stop_reason=response.stop_reason, model=response.model,
                           request_id=getattr(response, "_request_id", None), usage=usage)


def create_client(config: LLMConfig) -> LLMClient:
    if config.provider == "anthropic":
        return AnthropicClient(config)
    raise ValueError(f"Unknown LLM provider '{config.provider}'. Supported: anthropic.")
