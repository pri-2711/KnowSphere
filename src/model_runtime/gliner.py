"""GLiNER entity span model (optional component). Implements ``contracts.interfaces.SpanModel``.

Batched GLiNER inference over many texts, returning text-relative spans. The model is
loaded on first use; tests pass a fake ``model``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Protocol, Sequence

from src.config import GLINER_MODEL

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
