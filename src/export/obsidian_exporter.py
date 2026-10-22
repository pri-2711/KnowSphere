"""Write an Obsidian vault: one note per concept and one index note per document.

Notes are regenerated on every export. Descriptions and source lines are
verbatim quotes with citations in the form [Title | Author/Editor | Year, p. N].
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.metadata import UNKNOWN, MetadataStore, format_citation
from src.storage.mongodb_store import MongoKnowledgeStore
from src.textutil import collapse_whitespace

MAX_RELATED = 25
MAX_SOURCES = 30
_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]]')


def note_name(name: str) -> str:
    """File-safe note name that Obsidian wikilinks can target."""
    cleaned = _UNSAFE.sub(" ", name)
    return re.sub(r"\s+", " ", cleaned).strip()[:120] or "Untitled"


def _yaml(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def export_vault(store: MongoKnowledgeStore, metadata: MetadataStore, vault_dir: Path) -> dict[str, Any]:
    concepts = list(store.concepts.find())
    relationships = list(store.database["relationships"].find())
    documents = {document["_id"]: document for document in store.documents.find()}

    names: dict[str, str] = {}
    used: set[str] = set()
    for concept in sorted(concepts, key=lambda item: (-item.get("mention_count", 0), item["_id"])):
        base = note_name(concept["canonical_name"])
        name, suffix = base, 2
        while name.lower() in used:
            name, suffix = f"{base} ({suffix})", suffix + 1
        used.add(name.lower())
        names[concept["_id"]] = name

    neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in relationships:
        for this, other in ((edge["source_id"], edge["target_id"]), (edge["target_id"], edge["source_id"])):
            neighbors[this].append({"other": other, "edge": edge, "outgoing": this == edge["source_id"]})

    def title_of(document_id: str) -> str:
        meta = metadata.get(document_id)
        return meta.get("title") or documents.get(document_id, {}).get("filename") or UNKNOWN

    concept_dir = vault_dir / "Concepts"
    document_dir = vault_dir / "Documents"
    for folder in (concept_dir, document_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    for concept in concepts:
        lines = ["---", f"type: {_yaml(concept.get('concept_type') or 'concept')}"]
        for key, values in (("aliases", concept.get("aliases", [])[:50]),
                            ("source_documents", [title_of(doc_id) for doc_id in concept.get("document_ids", [])])):
            if values:
                lines.append(f"{key}:")
                lines += [f"  - {_yaml(value)}" for value in values]
            else:
                lines.append(f"{key}: []")
        lines += [f"mention_count: {concept.get('mention_count', 0)}", "---", "",
                  f"# {concept['canonical_name']}", ""]
        description = concept.get("description")
        if description:
            citation = format_citation(metadata.get(description["document_id"]), description["page_number"])
            lines += [f"> {collapse_whitespace(description['quote'])} {citation}", ""]
        else:
            lines += ["No description found in the documents.", ""]

        related = sorted(neighbors.get(concept["_id"], []), key=lambda item: -item["edge"].get("weight", 0))
        lines += ["## Related concepts", ""]
        if not related:
            lines.append("None found.")
        for item in related[:MAX_RELATED]:
            edge, other = item["edge"], item["other"]
            if other not in names:
                continue
            relation = edge["relation_type"]
            label = relation if relation == "related_to" or item["outgoing"] else f"{relation} (inverse)"
            evidence = edge["evidence"][0]
            citation = format_citation(metadata.get(evidence["document_id"]), evidence["page_number"])
            lines.append(f"- {label}: [[{names[other]}]] (confidence {edge['confidence']}) {citation}")
        lines += ["", "## Sources", ""]
        for mention in concept.get("mentions", [])[:MAX_SOURCES]:
            citation = format_citation(metadata.get(mention["document_id"]), mention["page_number"])
            quote = mention["quotes"][0]["quote"] if mention.get("quotes") else ""
            lines.append(f"- {citation}" + (f": \"{collapse_whitespace(quote)[:200]}\"" if quote else ""))
        (concept_dir / f"{names[concept['_id']]}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for document_id, document in documents.items():
        meta = metadata.get(document_id)
        members = sorted((concept for concept in concepts if document_id in concept.get("document_ids", [])),
                         key=lambda item: (-item.get("mention_count", 0), item["canonical_name"]))
        title = title_of(document_id)
        lines = ["---", "type: \"document\"", f"title: {_yaml(title)}",
                 f"author: {_yaml(meta.get('author') or UNKNOWN)}", f"year: {_yaml(meta.get('year') or UNKNOWN)}",
                 f"filename: {_yaml(document['filename'])}", f"pages: {document.get('page_count', 0)}", "---", "",
                 f"# {title}", "", f"Concepts found in this document: {len(members)}", ""]
        lines += [f"- [[{names[concept['_id']]}]] ({concept.get('mention_count', 0)} mentions)" for concept in members]
        (document_dir / f"{note_name(title)}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"vault": str(vault_dir), "concept_notes": len(concepts), "document_notes": len(documents)}
