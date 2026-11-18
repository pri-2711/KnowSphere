"""Image models. ``VitCaptioner`` (ViT-GPT2 captions, optional) implements ``contracts.interfaces.Captioner``.
Qwen2.5-VL image understanding is ``model_runtime.ollama.OllamaModel.describe``.
"""

from __future__ import annotations

from pathlib import Path


class VitCaptioner:
    """Optional ViT-GPT2 image captioning (transformers). Loaded only when enabled."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def caption(self, image_path: Path) -> str:
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("image-to-text", model=self.model_name)
        result = self._pipeline(str(image_path))
        return (result[0].get("generated_text") or "").strip() if result else ""
