"""Project paths and settings. Values can be overridden with environment variables or .env."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"          # original books, read only
EXTRACTED_DIR = DATA_DIR / "extracted"       # existing extraction output for raw_pdfs
UPLOAD_DIR = DATA_DIR / "uploads"            # one sub-folder per uploaded PDF
IMAGE_DIR = DATA_DIR / "images"              # images extracted from PDFs
METADATA_FILE = DATA_DIR / "document_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
VAULT_DIR = OUTPUT_DIR / "obsidian_vault"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
WEB_DIR = PROJECT_ROOT / "src" / "web"

WATCH_DIR = Path(os.environ.get("KM_WATCH_DIR", DATA_DIR / "inbox"))   # folder watched for PDFs
WATCH_INTERVAL = float(os.environ.get("KM_WATCH_INTERVAL", "5"))      # seconds between scans
WATCH_ENABLED = os.environ.get("KM_WATCH_ENABLED", "1") != "0"        # the API starts the watcher

GLINER_MODEL =os.environ.get("KM_GLINER_MODEL", "urchade/gliner_medium-v2.1")
EMBEDDING_MODEL = os.environ.get("KM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CAPTION_MODEL = os.environ.get("KM_CAPTION_MODEL", "nlpconnect/vit-gpt2-image-captioning")


# ---------------------------------------------------------------- concept-first pipeline settings
# Every value can be overridden in .env (see README.md). Changing a chunking value changes
# ChunkingConfig.version, so stored chunks are rebuilt; changing the prompt or model makes
# affected chunks be extracted again.
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
    """Text model for concept, relationship and claim extraction. Local Qwen through Ollama by default;
    set KM_LLM_PROVIDER=anthropic (and optionally KM_LLM_MODEL) to use Claude instead."""
    provider: str = _env("KM_LLM_PROVIDER", "ollama")
    model: str = _env("KM_LLM_MODEL", "qwen2.5:7b" if _env("KM_LLM_PROVIDER", "ollama") == "ollama"
                      else "claude-opus-5-5")
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
    # Canonical resolution: the text model judges pairs the rules above left apart (see resolution.py).
    resolve: bool = _env("KM_KG_RESOLVE", "1") != "0"
    resolution_threshold: float = 0.80        # name embeddings this close become candidates for the model
    resolution_max_pairs: int = int(_env("KM_KG_RESOLVE_PAIRS", "50"))   # new model questions per build


@dataclass(frozen=True)
class RuntimeConfig:
    """Local inference settings for a 16 GB RAM laptop: one model in memory at a time."""
    ollama_url: str = _env("KM_OLLAMA_URL", "http://localhost:11434")
    ollama_timeout_seconds: float = float(_env("KM_OLLAMA_TIMEOUT", "600"))
    ollama_num_ctx: int = int(_env("KM_OLLAMA_NUM_CTX", "8192"))
    keep_alive: str = _env("KM_OLLAMA_KEEP_ALIVE", "5m")       # how long Ollama keeps a model loaded between calls
    vision_provider: str = _env("KM_VISION_PROVIDER", "ollama")
    vision_model: str = _env("KM_VISION_MODEL", "qwen2.5vl:3b")
    use_gliner: bool = _env("KM_USE_GLINER", "1") != "0"      # GLiNER entity stage (optional component)
    ram_budget_gb: float = float(_env("KM_MODEL_RAM_GB", "10"))  # memory a model may use; larger models are refused


@dataclass(frozen=True)
class KGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    max_retries: int = int(_env("KM_LLM_MAX_RETRIES", "2"))   # repair attempts after the first answer, per chunk
