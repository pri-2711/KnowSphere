"""Module boundaries, checked on the real import graph of ``src``.

Each component may import only the components listed in ``ALLOWED``. ``LEGACY`` names the
remaining exceptions file by file (older stages that still persist directly); a new exception
fails this test, and an entry that is no longer needed fails it too, so the list only shrinks.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
COMPONENTS = ("contracts", "document_intelligence", "model_runtime", "knowledge_extraction", "evidence_quality",
              "knowledge_repository", "retrieval", "orchestration", "api")
ALLOWED = {
    "contracts": set(),
    "model_runtime": {"contracts"},
    "evidence_quality": {"contracts"},
    "knowledge_repository": {"contracts"},
    "document_intelligence": {"contracts"},
    "knowledge_extraction": {"contracts", "evidence_quality"},
    "retrieval": {"contracts", "knowledge_repository", "evidence_quality"},   # conflicts and units in answers
    "orchestration": {"contracts", "document_intelligence", "model_runtime", "knowledge_extraction",
                      "evidence_quality", "knowledge_repository", "retrieval"},
    # Interfaces: services, plus the composition root (opening the store) and retrieval types.
    "api": {"contracts", "orchestration", "knowledge_repository", "retrieval"},
}
LEGACY = {
    ("document_intelligence", "images.py", "knowledge_repository"),        # image extraction writes its units
    ("knowledge_extraction", "gliner_concepts.py", "knowledge_repository"),  # GLiNER stage reads and writes pages
    ("knowledge_extraction", "gliner_concepts.py", "model_runtime"),       # its own CLI builds the GLiNER model
    ("knowledge_extraction", "merging.py", "knowledge_repository"),
    ("knowledge_extraction", "relationships.py", "knowledge_repository"),
    ("knowledge_extraction", "image_links.py", "knowledge_repository"),
    ("api", "cli.py", "knowledge_extraction"),                             # single-stage operator commands
    ("api", "kg_cli.py", "model_runtime"),                                 # builds the router for the CLI
    ("api", "kg_cli.py", "document_intelligence"),                         # passes the OCR function
}


def import_edges() -> set[tuple[str, str, str]]:
    edges = set()
    for path in SRC.rglob("*.py"):
        parts = path.relative_to(SRC).parts
        if len(parts) < 2 or parts[0] not in COMPONENTS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                pieces = module.split(".")
                if pieces[0] == "src" and len(pieces) > 1 and pieces[1] in COMPONENTS and pieces[1] != parts[0]:
                    edges.add((parts[0], path.name, pieces[1]))
    return edges


def test_components_only_import_what_their_boundary_allows() -> None:
    violations = sorted(edge for edge in import_edges()
                        if edge[2] not in ALLOWED[edge[0]] and edge not in LEGACY)
    assert violations == []


def test_the_legacy_exception_list_has_no_stale_entries() -> None:
    assert sorted(LEGACY - import_edges()) == []


def test_contracts_are_independent_of_models_parsers_and_storage() -> None:
    forbidden = ("pymongo", "anthropic", "fitz", "pymupdf", "gliner", "httpx", "docling", "fastembed", "torch")
    for path in (SRC / "contracts").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & set(forbidden), path.name


def test_each_component_is_its_own_package() -> None:
    by_folder = defaultdict(list)
    for path in SRC.glob("*/__init__.py"):
        by_folder[path.parent.name].append(path)
    assert set(COMPONENTS) <= set(by_folder)
    for retired in ("kg", "knowledge", "ingest", "search", "graph", "export", "storage"):
        assert not (SRC / retired).exists(), retired
