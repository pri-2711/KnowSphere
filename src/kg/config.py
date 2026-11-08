"""Settings for the concept-first pipeline, kept apart from the code that uses them.

Every value can be overridden in ``.env`` (see README.md). Changing a chunking
value changes ``ChunkingConfig.version``, so stored chunks are rebuilt; changing
the prompt or model makes affected chunks be extracted again.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import src.config  # noqa: F401  (loads .env)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ChunkingConfig:
    min_chars: int = int(_env("KM_CHUNK_MIN_CHARS", "700"))        # below this a chunk is not closed at headings or topic shifts
    target_chars: int = int(_env("KM_CHUNK_TARGET_CHARS", "1400"))  # a chunk is closed once it reaches this size
    max_chars: int = int(_env("KM_CHUNK_MAX_CHARS", "2400"))        # hard limit; longer paragraphs are split by sentence
    cohesion_threshold: float = float(_env("KM_CHUNK_COHESION", "0.08"))  # stem-overlap cosine below this = topic shift
    heading_max_words: int = 12
    running_header_pages: int = 3   # a heading-like line repeated on this many pages is a page header, not a section

    @property
    def version(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"semantic-chunks-1:{digest}"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = _env("KM_LLM_PROVIDER", "anthropic")
    model: str = _env("KM_LLM_MODEL", "claude-opus-5-5")
    effort: str = _env("KM_LLM_EFFORT", "medium")
    max_tokens: int = int(_env("KM_LLM_MAX_TOKENS", "16000"))
    timeout_seconds: float = float(_env("KM_LLM_TIMEOUT", "300"))
    # Server-side refusal fallback ("fallbacks": "default"); set KM_LLM_FALLBACKS=0 to turn it off.
    use_fallbacks: bool = _env("KM_LLM_FALLBACKS", "1") != "0"


@dataclass(frozen=True)
class EnrichmentConfig:
    fuzzy_threshold: float = 94.0
    fuzzy_min_length: int = 8
    embedding_threshold: float = 0.92
    use_embeddings: bool = _env("KM_KG_EMBEDDINGS", "1") != "0"


@dataclass(frozen=True)
class KGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    max_retries: int = int(_env("KM_LLM_MAX_RETRIES", "2"))   # repair attempts after the first answer, per chunk
