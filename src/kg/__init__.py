"""Concept-first knowledge graph pipeline.

Document -> semantic segmentation -> Pass 1 (local concept extraction with an LLM)
-> Pass 2 (cross-document enrichment) -> validation -> recovery -> MongoDB.
Each step lives in its own module; ``orchestrator`` only coordinates them.
"""
