"""Project paths and settings. Values can be overridden with environment variables or .env."""

from __future__ import annotations

import os
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
