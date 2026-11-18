"""Command line interface of the concept-first pipeline (model-based candidate extraction).

    python -m src.api.kg_cli --document "ICES-2023-74.pdf"
    python -m src.api.kg_cli --all
    python -m src.api.kg_cli --rebuild-only        # Pass 2 + validation + governance, no model calls
    python -m src.api.kg_cli --all --claims        # also extract claim candidates
    python -m src.api.kg_cli --all --visual --no-concepts   # figures only (Qwen2.5-VL + OCR)
    add --project PROJECT_ID to work on a project other than the default one
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import METADATA_FILE, PROJECT_ROOT, REPORT_DIR, KGConfig
from src.contracts.interfaces import Embedder, LLMFatalError
from src.knowledge_repository.metadata import MetadataStore
from src.knowledge_repository.mongodb_store import MongoKnowledgeStore, connect_from_environment
from src.knowledge_repository.projects import DEFAULT_PROJECT, ProjectStore, migrate
from src.model_runtime.router import ModelRouter
from src.orchestration.kg_pipeline import KGPipeline, logger


def _embedder(config: KGConfig) -> Embedder | None:
    if not config.enrichment.use_embeddings:
        return None
    from src.config import EMBEDDING_MODEL
    from src.model_runtime.embeddings import FastEmbedder
    return FastEmbedder(EMBEDDING_MODEL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concept-first knowledge graph pipeline")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--document", action="append", help="Document filename, ID or ID prefix (repeatable)")
    target.add_argument("--all", action="store_true", help="Every stored document")
    target.add_argument("--rebuild-only", action="store_true",
                        help="Pass 2 + validation + claims + governance only (no model calls)")
    parser.add_argument("--limit-chunks", type=int, help="Send at most N chunks (or images) to a model in this run")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project ID (default: default)")
    parser.add_argument("--claims", action="store_true", help="Also extract claim candidates (text model)")
    parser.add_argument("--visual", action="store_true",
                        help="Also extract concepts from figures (vision model, grounded on caption and OCR)")
    parser.add_argument("--no-concepts", action="store_true", help="Skip Pass 1 concept extraction in this run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = KGConfig()
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database, PROJECT_ROOT)
        migrate(database)
        ProjectStore(database).get(args.project)
        metadata = MetadataStore(METADATA_FILE)
        if args.rebuild_only:
            pipeline = KGPipeline(store, config, embedder=_embedder(config), metadata=metadata, report_dir=REPORT_DIR)
            print(json.dumps(pipeline.build_graph(args.project), indent=2, default=str))
            return 0
        from src.document_intelligence.pdf_extraction import ocr_image_file
        router = ModelRouter(config.llm, config.runtime)
        needs_text = args.claims or not args.no_concepts
        pipeline = KGPipeline(store, config, router.text_model() if needs_text else None, _embedder(config),
                              metadata, REPORT_DIR, vision=router.vision_model() if args.visual else None,
                              ocr=ocr_image_file)
        documents = pipeline.kg.all_documents(args.project) if args.all else \
            pipeline.kg.resolve_documents(args.document, args.project)

        def progress(stage: str, done: int, total: int) -> None:
            print(f"\r{stage}: {done}/{total}", end="", file=sys.stderr, flush=True)

        summary = pipeline.run([document["_id"] for document in documents], args.limit_chunks, progress,
                               concepts=not args.no_concepts, claims=args.claims, visual=args.visual)
        print(file=sys.stderr)
        summary["documents"] = [document["filename"] for document in documents]
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except LLMFatalError as exc:
        print(f"\nModel setup problem, run stopped: {exc}\n"
              "Local Qwen: install Ollama, run 'ollama pull <model>' and keep 'ollama serve' running. "
              "Claude: set KM_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY in .env.", file=sys.stderr)
        return 3
    except Exception as exc:
        logger.exception("kg pipeline failed")
        print(f"KG pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
