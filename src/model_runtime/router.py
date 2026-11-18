"""Model router: the only place that decides which model serves which capability.

Capabilities:
    text     concept, relationship and claim extraction (Qwen through Ollama, or Claude)
    vision   figure and diagram understanding (Qwen2.5-VL through Ollama)
    spans    entity spans (GLiNER, optional)
    embed    text embeddings for alias resolution (fastembed)

Extraction code asks for a capability and receives an object that implements the
matching protocol in ``contracts.interfaces``; it never imports a provider.

Resource policy for a 16 GB RAM laptop: every local model call runs through one
``ResourceGate``, so at most one local model works at a time, and before a
different Ollama model is used the previous one is unloaded. A model whose
estimated memory exceeds ``RuntimeConfig.ram_budget_gb`` is refused before it is
loaded. Hosted providers (Claude) need no local memory and skip the gate.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import EMBEDDING_MODEL, GLINER_MODEL, LLMConfig, RuntimeConfig
from src.contracts.interfaces import Embedder, LLMFatalError, LLMResponse, SpanModel, TextModel, VisionModel

logger = logging.getLogger("knowledge_map.models")

# Rough resident memory in GB (4-bit quantized weights as Ollama ships them, plus context). Estimates, not benchmarks.
MEMORY_ESTIMATES_GB = {
    "qwen2.5:3b": 2.6, "qwen2.5:7b": 5.5, "qwen2.5:14b": 10.5,
    "qwen2.5vl:3b": 4.2, "qwen2.5vl:7b": 7.0, "qwen2.5vl:32b": 22.0,
    GLINER_MODEL: 1.5, EMBEDDING_MODEL: 0.3,
}
DEFAULT_ESTIMATE_GB = 6.0


class ModelUnavailable(LLMFatalError):
    """The configured model cannot be used here (unknown provider, over the memory budget, disabled)."""


class ResourceGate:
    """Serialises local model work and keeps one Ollama model resident at a time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.resident: Any = None          # the local model used last
        self.switches = 0

    def run(self, model: Any, call: Callable[[], Any]) -> Any:
        with self._lock:
            if self.resident is not None and self.resident is not model:
                unload = getattr(self.resident, "unload", None)
                if callable(unload):
                    unload()
                self.switches += 1
            self.resident = model
            return call()


@dataclass
class GatedText:
    """A text model whose calls go through the gate. Same protocol as the wrapped model."""
    inner: TextModel
    gate: ResourceGate | None

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, system: str, messages: Sequence[dict[str, Any]], schema: dict[str, Any]) -> LLMResponse:
        if self.gate is None:
            return self.inner.complete(system, messages, schema)
        return self.gate.run(self.inner, lambda: self.inner.complete(system, messages, schema))


@dataclass
class GatedVision:
    inner: VisionModel
    gate: ResourceGate

    @property
    def model(self) -> str:
        return self.inner.model

    def describe(self, image_path: Path, system: str, prompt: str, schema: dict[str, Any]) -> LLMResponse:
        return self.gate.run(self.inner, lambda: self.inner.describe(image_path, system, prompt, schema))


@dataclass
class GatedSpans:
    inner: SpanModel
    gate: ResourceGate

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def extract_many(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        return self.gate.run(self.inner, lambda: self.inner.extract_many(texts))


@dataclass
class ModelRouter:
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gate: ResourceGate = field(default_factory=ResourceGate)
    # Pre-built models by capability, used by tests and by callers that already hold one.
    overrides: dict[str, Any] = field(default_factory=dict)
    transport: Any = None                  # Ollama HTTP transport (tests pass a fake)
    _cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm_config(cls, llm: LLMConfig, runtime: RuntimeConfig | None = None) -> "ModelRouter":
        return cls(llm=llm, runtime=runtime or RuntimeConfig())

    # ------------------------------------------------------------ policy
    def check_memory(self, model_name: str) -> float:
        estimate = MEMORY_ESTIMATES_GB.get(model_name, DEFAULT_ESTIMATE_GB)
        if estimate > self.runtime.ram_budget_gb:
            raise ModelUnavailable(
                f"{model_name} needs about {estimate} GB, over the {self.runtime.ram_budget_gb} GB budget "
                "(KM_MODEL_RAM_GB); choose a smaller model")
        return estimate

    def describe(self) -> dict[str, Any]:
        """What each capability is routed to, without loading anything."""
        return {
            "text": {"provider": self.llm.provider, "model": self.llm.model},
            "vision": {"provider": self.runtime.vision_provider, "model": self.runtime.vision_model},
            "spans": {"provider": "gliner", "model": GLINER_MODEL, "enabled": self.runtime.use_gliner},
            "embed": {"provider": "fastembed", "model": EMBEDDING_MODEL},
            "ram_budget_gb": self.runtime.ram_budget_gb, "sequential_local_execution": True,
        }

    # ------------------------------------------------------------ capabilities
    def text_model(self) -> TextModel:
        if "text" in self.overrides:
            return self.overrides["text"]
        if "text" not in self._cache:
            provider = self.llm.provider
            if provider == "ollama":
                from src.model_runtime.ollama import OllamaModel
                self.check_memory(self.llm.model)
                self._cache["text"] = GatedText(OllamaModel(self.llm.model, self.runtime, self.transport), self.gate)
            elif provider == "anthropic":
                from src.model_runtime.llm import AnthropicClient
                self._cache["text"] = GatedText(AnthropicClient(self.llm), None)   # hosted: no local memory
            else:
                raise ModelUnavailable(f"Unknown text provider '{provider}'. Supported: ollama, anthropic.")
        return self._cache["text"]

    def vision_model(self) -> VisionModel:
        if "vision" in self.overrides:
            return self.overrides["vision"]
        if "vision" not in self._cache:
            if self.runtime.vision_provider != "ollama":
                raise ModelUnavailable(f"Unknown vision provider '{self.runtime.vision_provider}'. Supported: ollama.")
            from src.model_runtime.ollama import OllamaModel
            self.check_memory(self.runtime.vision_model)
            self._cache["vision"] = GatedVision(
                OllamaModel(self.runtime.vision_model, self.runtime, self.transport), self.gate)
        return self._cache["vision"]

    def span_model(self) -> SpanModel | None:
        """GLiNER, or None when the optional entity stage is turned off (KM_USE_GLINER=0)."""
        if "spans" in self.overrides:
            return self.overrides["spans"]
        if not self.runtime.use_gliner:
            return None
        if "spans" not in self._cache:
            from src.model_runtime.gliner import GlinerExtractor
            self.check_memory(GLINER_MODEL)
            self._cache["spans"] = GatedSpans(GlinerExtractor(GLINER_MODEL), self.gate)
        return self._cache["spans"]

    def embedder(self) -> Embedder:
        if "embed" in self.overrides:
            return self.overrides["embed"]
        if "embed" not in self._cache:
            from src.model_runtime.embeddings import FastEmbedder
            self._cache["embed"] = FastEmbedder(EMBEDDING_MODEL)
        return self._cache["embed"]
