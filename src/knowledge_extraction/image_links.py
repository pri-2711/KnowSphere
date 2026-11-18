"""Link image content units to a project's concepts by finding concept names in their caption and context."""

from __future__ import annotations

import re
from collections import defaultdict

from src.knowledge_repository.mongodb_store import MongoKnowledgeStore
from src.knowledge_repository.projects import DEFAULT_PROJECT, project_filter
from src.textutil import concept_key


def link_images_to_concepts(store: MongoKnowledgeStore, max_words: int = 5,
                            project_id: str = DEFAULT_PROJECT) -> dict[str, int]:
    """Attach concept IDs to a project's image units whose caption or context names one of its concepts."""
    scope = project_filter(project_id)
    aliases = {record.get("key", record["_id"]): record["concept_id"]
               for record in store.database["concept_aliases"].find(scope)}
    images_by_concept: dict[str, list[str]] = defaultdict(list)
    linked = 0
    for unit in store.content_units.find({"unit_type": "image", **scope}, {"text": 1}):
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", unit.get("text") or "")
        found: set[str] = set()
        for size in range(1, max_words + 1):
            for start in range(0, len(words) - size + 1):
                concept_id = aliases.get(concept_key(" ".join(words[start:start + size])))
                if concept_id:
                    found.add(concept_id)
        store.content_units.update_one({"_id": unit["_id"]}, {"$set": {"concept_ids": sorted(found)}})
        for concept_id in found:
            images_by_concept[concept_id].append(unit["_id"])
        linked += bool(found)
    store.concepts.update_many(scope, {"$set": {"image_ids": []}})
    for concept_id, unit_ids in images_by_concept.items():
        store.concepts.update_one({"_id": concept_id}, {"$set": {"image_ids": sorted(unit_ids)}})
    return {"images_linked": linked, "concepts_with_images": len(images_by_concept)}
