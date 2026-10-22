"""Build an in-memory NetworkX graph from MongoDB concepts and relationships.

Nodes keep type, description (with its source), mention count and documents.
Edges keep weight, confidence, relation type, direction and the evidence list.
Analysis adds degree, weighted PageRank (used as importance, computed with NumPy) and Louvain
communities (used for colors).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, Mapping

import networkx as nx

from src.storage.mongodb_store import MongoKnowledgeStore, connect_from_environment


def build_graph(concepts: Iterable[Mapping[str, Any]], relationships: Iterable[Mapping[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for concept in concepts:
        graph.add_node(
            concept["_id"],
            name=concept["canonical_name"],
            concept_type=concept.get("concept_type"),
            mention_count=concept.get("mention_count", 0),
            page_count=concept.get("page_count", 0),
            document_ids=concept.get("document_ids", []),
            aliases=concept.get("aliases", []),
            description=concept.get("description"),
        )
    for edge in relationships:
        source, target = edge["source_id"], edge["target_id"]
        if source not in graph or target not in graph or not edge.get("evidence"):
            continue
        graph.add_edge(
            source, target,
            id=edge["_id"],
            source_id=source,
            target_id=target,
            relation_type=edge.get("relation_type", "related_to"),
            weight=float(edge.get("weight", 1.0)),
            confidence=float(edge.get("confidence", 0.0)),
            evidence=edge.get("evidence", []),
        )
    return graph


def weighted_pagerank(graph: nx.Graph, damping: float = 0.85, iterations: int = 100,
                      tolerance: float = 1e-10) -> dict[str, float]:
    """Weighted PageRank by power iteration with NumPy (NetworkX's version needs SciPy)."""
    import numpy as np

    nodes = list(graph)
    size = len(nodes)
    position = {node: index for index, node in enumerate(nodes)}
    out_weight = np.zeros(size)
    sources, targets, weights = [], [], []
    for a, b, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        for u, v in ((a, b), (b, a)):
            sources.append(position[u])
            targets.append(position[v])
            weights.append(weight)
            out_weight[position[u]] += weight
    sources_arr, targets_arr, weights_arr = np.array(sources, dtype=int), np.array(targets, dtype=int), np.array(weights)
    rank = np.full(size, 1.0 / size)
    for _ in range(iterations):
        share = np.divide(rank, out_weight, out=np.zeros(size), where=out_weight > 0)
        incoming = np.zeros(size)
        np.add.at(incoming, targets_arr, share[sources_arr] * weights_arr)
        dangling = rank[out_weight == 0].sum()
        updated = (1 - damping) / size + damping * (incoming + dangling / size)
        if np.abs(updated - rank).sum() < tolerance:
            rank = updated
            break
        rank = updated
    return {node: float(rank[position[node]]) for node in nodes}


def analyze(graph: nx.Graph, seed: int = 42) -> nx.Graph:
    """Store degree, importance (PageRank) and community on every node."""
    if graph.number_of_nodes() == 0:
        return graph
    pagerank = weighted_pagerank(graph)
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed) if graph.number_of_edges() \
        else [{node} for node in graph]
    community_of = {}
    for index, members in enumerate(sorted(communities, key=lambda group: (-len(group), sorted(group)[0]))):
        for node in members:
            community_of[node] = index
    for node in graph:
        graph.nodes[node]["degree"] = graph.degree(node)
        graph.nodes[node]["importance"] = pagerank[node]
        graph.nodes[node]["community"] = community_of[node]
    return graph


def load_graph(store: MongoKnowledgeStore) -> nx.Graph:
    concepts = store.concepts.find({}, {"mentions": 0})
    relationships = store.database["relationships"].find()
    return analyze(build_graph(concepts, relationships))


def graph_stats(graph: nx.Graph, top: int = 10) -> dict[str, Any]:
    ranked = sorted(graph.nodes(data=True), key=lambda item: (-item[1].get("degree", 0), item[1]["name"]))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len({data.get("community") for _, data in graph.nodes(data=True)}),
        "isolated_nodes": nx.number_of_isolates(graph),
        "top_connected": [{"name": data["name"], "degree": data.get("degree", 0)} for _, data in ranked[:top]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the knowledge graph from MongoDB and print statistics.")
    parser.add_argument("--top", type=int, default=10, help="How many top connected concepts to list")
    args = parser.parse_args(argv)
    client = None
    try:
        client, database = connect_from_environment()
        print(json.dumps(graph_stats(load_graph(MongoKnowledgeStore(database)), args.top), indent=2))
        return 0
    except Exception as exc:
        print(f"Graph build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
