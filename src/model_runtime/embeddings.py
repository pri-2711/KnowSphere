"""Text embeddings (fastembed ONNX, bge-small). Implements ``contracts.interfaces.Embedder``."""

from __future__ import annotations

from typing import Any, Sequence


class FastEmbedder:
    """fastembed ONNX model, used only for merging concept names."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> Any:
        import numpy as np
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        vectors = np.array(list(self._model.embed(list(texts), batch_size=256)), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
