"""Utilities for constructing cascade trees and computing structural metrics without networkx.

This module avoids external dependencies by representing a tree as a simple
adjacency list (a dictionary mapping nodes to sets of children).  It
provides functions to build such a structure from a `Cascade` and to compute
basic structural properties of the resulting tree.
"""

from __future__ import annotations

from typing import Dict, Set, List

from .cascade import Cascade


Adjacency = Dict[int, Set[int]]


def build_tree(cascade: Cascade) -> Adjacency:
    """Construct an adjacency list representing the cascade tree.

    Parameters
    ----------
    cascade: `Cascade`
        The cascade from which to build the tree.  The root user and all
        retweet events are used to build a directed tree where edges go
        from parent to child.

    Returns
    -------
    adjacency: dict
        A dictionary mapping each node to a set of its children.  Nodes
        with no children will have an empty set.  The root node is
        included even if it has no children.
    """
    adj: Adjacency = {}
    # ensure root exists
    adj.setdefault(cascade.root, set())
    for event in cascade.events:
        # ensure parent and child nodes are present
        adj.setdefault(event.parent, set())
        adj.setdefault(event.user, set())
        adj[event.parent].add(event.user)
    return adj


def compute_depths(adj: Adjacency, root: int) -> Dict[int, int]:
    """Compute depths of nodes reachable from the root in the tree.

    Nodes unreachable from the root have a depth of -1.  Depth of the root
    is 0.  Uses a simple breadth‑first search.
    """
    depths: Dict[int, int] = {node: -1 for node in adj}
    if root not in adj:
        return depths
    depths[root] = 0
    queue: List[int] = [root]
    while queue:
        current = queue.pop(0)
        for child in adj.get(current, set()):
            if depths[child] == -1 or depths[child] > depths[current] + 1:
                depths[child] = depths[current] + 1
                queue.append(child)
    return depths


def structural_metrics(adj: Adjacency, root: int) -> Dict[str, float]:
    """Compute structural metrics from an adjacency list.

    Metrics include:
    - `depth`: maximum distance from root to any reachable node.
    - `avg_depth`: average depth of reachable nodes.
    - `leaves`: number of leaf nodes (nodes with zero children).
    - `branching_factor`: average number of children of non‑leaf nodes.
    - `wiener_root_avg`: average distance from the root to all reachable nodes.

    Nodes unreachable from the root are ignored.
    """
    if root not in adj:
        # no tree
        return {
            "depth": 0.0,
            "avg_depth": 0.0,
            "leaves": 1.0,
            "branching_factor": 0.0,
            "wiener_root_avg": 0.0,
        }
    depths = compute_depths(adj, root)
    reachable = [node for node, d in depths.items() if d >= 0]
    if not reachable:
        return {
            "depth": 0.0,
            "avg_depth": 0.0,
            "leaves": 1.0,
            "branching_factor": 0.0,
            "wiener_root_avg": 0.0,
        }
    max_depth = max(depths[node] for node in reachable)
    avg_depth = sum(depths[node] for node in reachable) / len(reachable)
    leaves = [node for node in reachable if len(adj.get(node, set())) == 0]
    n_leaves = float(len(leaves))
    non_leaves = [node for node in reachable if len(adj.get(node, set())) > 0]
    if non_leaves:
        branching = sum(len(adj[node]) for node in non_leaves) / len(non_leaves)
    else:
        branching = 0.0
    wiener = sum(depths[node] for node in reachable) / len(reachable)
    return {
        "depth": float(max_depth),
        "avg_depth": float(avg_depth),
        "leaves": float(n_leaves),
        "branching_factor": float(branching),
        "wiener_root_avg": float(wiener),
    }
