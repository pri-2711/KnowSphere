"""NetworkX graph from MongoDB and the Obsidian vault export."""

from __future__ import annotations

import re

from src.export.obsidian_exporter import export_vault
from src.graph.graph_builder import graph_stats, load_graph

CITATION = re.compile(r"\[[^|\]]+ \| [^|\]]+ \| [^|\]]+, p\. \d+\]")


def test_graph_keeps_evidence_and_computes_analysis(seeded) -> None:
    graph = load_graph(seeded)
    assert graph.number_of_nodes() == seeded.concepts.count_documents({})
    assert graph.number_of_edges() == seeded.database["relationships"].count_documents({})
    for _, _, data in graph.edges(data=True):
        assert data["evidence"]
    for _, data in graph.nodes(data=True):
        assert {"degree", "importance", "community", "concept_type"} <= set(data)
    stats = graph_stats(graph, top=3)
    assert stats["nodes"] == graph.number_of_nodes()
    assert len(stats["top_connected"]) == 3


def test_obsidian_vault_notes(seeded, metadata, tmp_path) -> None:
    vault = tmp_path / "vault"
    summary = export_vault(seeded, metadata, vault)
    notes = list((vault / "Concepts").glob("*.md"))
    assert len(notes) == summary["concept_notes"] == seeded.concepts.count_documents({})
    note = (vault / "Concepts" / "Requirements analysis.md").read_text(encoding="utf-8")
    assert note.startswith("---\ntype: \"process\"")
    assert "source_documents:" in note and "aliases:" in note
    assert "[[Systems engineering]]" in note
    assert "## Sources" in note and CITATION.search(note)
    assert "[Systems Handbook | Jane Doe | 2012, p. 1]" in note
    index = (vault / "Documents" / "Systems Handbook.md").read_text(encoding="utf-8")
    assert "[[Requirements analysis]]" in index and 'author: "Jane Doe"' in index
    # A second export replaces the notes instead of adding copies.
    export_vault(seeded, metadata, vault)
    assert len(list((vault / "Concepts").glob("*.md"))) == len(notes)
