"""Extract concept mentions from MongoDB page records with GLiNER.

Output goes to ``concept_mentions``: one record per (normalized concept name,
page) with the surface forms seen, label scores, and the sentence spans where
the name occurs. Spans are offsets into the stored page text, so every quote is
verbatim. Merging mentions into concept nodes happens in ``merging.py``.

Speed and resumability:
- chunks from many pages are pooled and sent to GLiNER in batches;
- table-of-contents and near-empty pages are skipped;
- ``extraction_state`` stores the SHA-256 of each processed page text, so a
  page is only processed again when its text or the extractor version changes;
- results are written after every batch, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from src.config import GLINER_MODEL, LOG_DIR
from src.storage.mongodb_store import MongoKnowledgeStore, connect_from_environment, stable_id
from src.textutil import clean_surface, is_toc_page, mask_dot_leaders, normalize_concept, split_sentences

EXTRACTOR_VERSION = "gliner-concepts-2"
DEFAULT_LABELS = (
    "engineering concept",
    "process",
    "method",
    "system component",
    "document",
    "organization",
)
DEFAULT_THRESHOLD = 0.5
CHUNK_WORDS = 200          # shorter chunks found more spans at the same speed (measured)
BATCH_SIZE = 8             # chunks per forward pass
PAGES_PER_WRITE = 24       # pages processed between MongoDB writes (resume granularity)
MIN_ALPHA_WORDS = 15       # pages with fewer real words are skipped as near-empty
SKIPPED_QUALITY = ("no_text_detected", "failed")

# Whole names that are too generic to be useful concept nodes.
GENERIC_STOPLIST = frozenset("""
system document documentation page figure fig table chapter section part example appendix reference
introduction summary conclusion overview method approach model process information data result work case
way type number value use user level time step activity element item thing area issue problem solution
product program project organization team book author editor contributor paper article text note list
group set form tool technique concept
""".split())
_FUNCTION_WORDS = frozenset("""
and or of to for with in on by as at from which that this these those is are be was were its their
our an a the into than then also such
""".split())

logger = logging.getLogger("knowledge_map.concepts")


class ChunkModel(Protocol):
    """The part of the GLiNER API this module uses."""

    def inference(self, texts: list[str], labels: list[str], threshold: float = 0.5,
                  batch_size: int = 8, **kwargs: Any) -> list[list[dict[str, Any]]]:
        ...


def chunk_text_by_words(text: str, chunk_words: int) -> list[tuple[int, str]]:
    """Split text into chunks of at most ``chunk_words`` words, keeping start offsets."""
    words = list(re.finditer(r"\S+", text))
    chunks: list[tuple[int, str]] = []
    for index in range(0, len(words), chunk_words):
        group = words[index:index + chunk_words]
        start, end = group[0].start(), group[-1].end()
        chunks.append((start, text[start:end]))
    return chunks


class GlinerExtractor:
    """Batched GLiNER inference over many pages, returning page-relative spans."""

    def __init__(
        self,
        model_name: str = GLINER_MODEL,
        labels: Iterable[str] = DEFAULT_LABELS,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_words: int = CHUNK_WORDS,
        batch_size: int = BATCH_SIZE,
        model: ChunkModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.labels = tuple(labels)
        self.threshold = threshold
        self.chunk_words = chunk_words
        self.batch_size = batch_size
        self._model = model

    def _load(self) -> ChunkModel:
        if self._model is None:
            try:
                from gliner import GLiNER
            except ImportError as exc:  # pragma: no cover - depends on setup
                raise RuntimeError("GLiNER is missing. Install requirements.txt in the .venv first.") from exc
            self._model = GLiNER.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def extract_many(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Return one list of spans (with page offsets) per input text."""
        model = self._load()
        chunks: list[tuple[int, int, str]] = []  # (text index, chunk start, chunk text)
        for text_index, text in enumerate(texts):
            for start, chunk in chunk_text_by_words(text, self.chunk_words):
                chunks.append((text_index, start, chunk))
        # Similar lengths in one batch waste less padding.
        order = sorted(range(len(chunks)), key=lambda index: len(chunks[index][2]))
        results: list[list[dict[str, Any]]] = [[] for _ in texts]
        batch = [chunks[index] for index in order]
        predictions = model.inference(
            [chunk for _, _, chunk in batch], list(self.labels), threshold=self.threshold,
            batch_size=self.batch_size,
        ) if batch else []
        for (text_index, chunk_start, _), entities in zip(batch, predictions):
            for entity in entities:
                results[text_index].append({
                    "label": entity["label"],
                    "score": float(entity["score"]),
                    "start": chunk_start + int(entity["start"]),
                    "end": chunk_start + int(entity["end"]),
                })
        return results


def is_noise(normalized: str) -> bool:
    """Reject spans that cannot be useful concept names."""
    if len(normalized) < 3 or len(normalized) > 80:
        return True
    words = normalized.split()
    if not words or len(words) > 6:
        return True
    if normalized in GENERIC_STOPLIST:
        return True
    if words[0] in _FUNCTION_WORDS or words[-1] in _FUNCTION_WORDS:
        return True
    letters = sum(char.isalpha() for char in normalized)
    if letters < 3 or letters / len(normalized.replace(" ", "")) < 0.6:
        return True
    if any(char in normalized for char in "�…|@=<>{}[]") or ".." in normalized:
        return True
    # Fragments such as "-ing" or "tion" left over from broken lines.
    if len(words) == 1 and len(words[0]) < 4 and not words[0].isupper():
        return True
    return False


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _alpha_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text))


def build_page_mentions(page: Mapping[str, Any], spans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group a page's spans by normalized name and attach sentence spans."""
    text: str = page["text"]
    sentences = split_sentences(text)
    starts = [start for start, _ in sentences]
    grouped: dict[str, dict[str, Any]] = {}
    for span in spans:
        surface = clean_surface(text[span["start"]:span["end"]])
        normalized = normalize_concept(surface)
        if is_noise(normalized):
            continue
        entry = grouped.setdefault(normalized, {
            "forms": Counter(), "labels": defaultdict(float), "count": 0, "max_score": 0.0, "sentences": set(),
        })
        entry["forms"][surface] += 1
        entry["labels"][span["label"]] += span["score"]
        entry["count"] += 1
        entry["max_score"] = max(entry["max_score"], span["score"])
        position = bisect.bisect_right(starts, span["start"]) - 1
        if position >= 0 and sentences[position][0] <= span["start"] < sentences[position][1]:
            entry["sentences"].add(sentences[position])
    records = []
    for normalized, entry in sorted(grouped.items()):
        records.append({
            "_id": stable_id("concept_mention", normalized, page["_id"]),
            "name": normalized,
            "document_id": page["document_id"],
            "content_unit_id": page["_id"],
            "page_number": page["page_number"],
            "surface_forms": sorted([form, count] for form, count in entry["forms"].items()),
            "label_scores": sorted([label, round(score, 4)] for label, score in entry["labels"].items()),
            "count": entry["count"],
            "max_score": round(entry["max_score"], 4),
            "sentences": sorted([start, end] for start, end in entry["sentences"]),
            "extractor_version": EXTRACTOR_VERSION,
        })
    return records


def _page_query(document_id: str | None) -> dict[str, Any]:
    query: dict[str, Any] = {"unit_type": "page", "quality_status": {"$nin": list(SKIPPED_QUALITY)}}
    if document_id:
        query["document_id"] = document_id
    return query


def _save_page(store: MongoKnowledgeStore, page: Mapping[str, Any], digest: str, status: str,
               mentions: list[dict[str, Any]]) -> None:
    """Replace this page's mentions and record that it is done (one page at a time keeps resume safe)."""
    store.database["concept_mentions"].delete_many({"content_unit_id": page["_id"]})
    if mentions:
        store.database["concept_mentions"].insert_many(mentions)
    store.database["extraction_state"].update_one(
        {"_id": page["_id"]},
        {"$set": {
            "document_id": page["document_id"],
            "page_number": page["page_number"],
            "text_sha256": digest,
            "extractor_version": EXTRACTOR_VERSION,
            "status": status,
            "mention_names": len(mentions),
            "processed_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


def ensure_indexes(store: MongoKnowledgeStore) -> None:
    mentions = store.database["concept_mentions"]
    mentions.create_index([("content_unit_id", 1)], name="ix_mention_unit")
    mentions.create_index([("name", 1)], name="ix_mention_name")
    store.database["extraction_state"].create_index([("document_id", 1)], name="ix_state_document")


def extract_concepts(
    store: MongoKnowledgeStore,
    extractor: GlinerExtractor,
    document_id: str | None = None,
    limit_pages: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    show_bar: bool = False,
) -> dict[str, Any]:
    """Extract mentions for every page whose text hash is not yet processed."""
    ensure_indexes(store)
    started = time.perf_counter()
    cursor = store.content_units.find(_page_query(document_id)).sort([("document_id", 1), ("page_number", 1)])
    if limit_pages:
        cursor = cursor.limit(limit_pages)
    pages = list(cursor)
    states = {state["_id"]: state for state in store.database["extraction_state"].find(
        {"_id": {"$in": [page["_id"] for page in pages]}})}

    todo: list[tuple[dict[str, Any], str]] = []
    unchanged = 0
    for page in pages:
        digest = text_sha256(page.get("text") or "")
        state = states.get(page["_id"])
        if state and state.get("text_sha256") == digest and state.get("extractor_version") == EXTRACTOR_VERSION:
            unchanged += 1
            continue
        todo.append((page, digest))

    summary = {"pages": len(pages), "unchanged": unchanged, "processed": 0, "skipped_toc": 0,
               "skipped_empty": 0, "mentions": 0}
    bar = None
    if show_bar:
        from tqdm import tqdm
        bar = tqdm(total=len(todo), unit="page", desc="concepts")
    done = 0
    for batch_start in range(0, len(todo), PAGES_PER_WRITE):
        batch = todo[batch_start:batch_start + PAGES_PER_WRITE]
        runnable: list[tuple[dict[str, Any], str]] = []
        for page, digest in batch:
            text = page.get("text") or ""
            if is_toc_page(text):
                _save_page(store, page, digest, "skipped_toc", [])
                summary["skipped_toc"] += 1
            elif _alpha_words(text) < MIN_ALPHA_WORDS:
                _save_page(store, page, digest, "skipped_empty", [])
                summary["skipped_empty"] += 1
            else:
                runnable.append((page, digest))
        if runnable:
            texts = [mask_dot_leaders(page["text"]) for page, _ in runnable]
            for (page, digest), spans in zip(runnable, extractor.extract_many(texts)):
                mentions = build_page_mentions(page, spans)
                _save_page(store, page, digest, "done", mentions)
                summary["processed"] += 1
                summary["mentions"] += len(mentions)
        done += len(batch)
        if bar is not None:
            bar.update(len(batch))
        if progress is not None:
            progress(done, len(todo))
        logger.info("concept extraction: %d/%d pages", done, len(todo))
    if bar is not None:
        bar.close()
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract concept mentions from MongoDB pages with GLiNER.")
    parser.add_argument("--model", default=GLINER_MODEL, help=f"GLiNER model (default: {GLINER_MODEL})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum span score")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Chunks per inference batch")
    parser.add_argument("--document-id", help="Only process pages of this document ID")
    parser.add_argument("--limit-pages", type=int, help="Only consider the first N pages (trial runs)")
    args = parser.parse_args(argv)
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    configure_logging()
    client = None
    try:
        client, database = connect_from_environment()
        store = MongoKnowledgeStore(database)
        extractor = GlinerExtractor(args.model, threshold=args.threshold, batch_size=args.batch_size)
        summary = extract_concepts(store, extractor, args.document_id, args.limit_pages, show_bar=True)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"Concept extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
