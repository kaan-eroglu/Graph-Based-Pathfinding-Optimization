"""
Graph Architecture Module
=========================
Implements weighted, directed, and undirected graphs using an adjacency list
representation. The adjacency list provides O(V + E) space complexity, making
it efficient for both sparse and dense graphs.

Mathematical Foundation:
    A graph G = (V, E) where V is the set of vertices and E is the set of
    edges. Each edge e ∈ E has an associated weight w(e) ∈ ℝ⁺.
    For undirected graphs, (u, v) ∈ E ⟺ (v, u) ∈ E with w(u,v) = w(v,u).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Edge:
    """Represents a weighted edge in the graph."""

    source: str
    target: str
    weight: float


class Graph:
    """Weighted graph using adjacency list representation.

    Supports both directed and undirected graphs. Internally stores edges
    as a dict-of-dicts for O(1) edge lookup and O(deg(v)) neighbor iteration.

    Attributes:
        directed: Whether the graph is directed.
    """

    def __init__(self, directed: bool = False) -> None:
        self.directed: bool = directed
        # adj[u][v] = weight  →  O(1) lookup, O(deg(u)) iteration
        self._adj: dict[str, dict[str, float]] = {}
        # Optional 2D coordinates for heuristic computation
        self._positions: dict[str, tuple[float, float]] = {}

    # ── Vertex Operations ──────────────────────────────────────────────

    def add_vertex(self, vertex: str) -> None:
        """Add a vertex to the graph (no-op if it already exists)."""
        if vertex not in self._adj:
            self._adj[vertex] = {}

    def remove_vertex(self, vertex: str) -> None:
        """Remove a vertex and all its incident edges."""
        if vertex not in self._adj:
            raise KeyError(f"Vertex '{vertex}' not in graph")
        del self._adj[vertex]
        for neighbors in self._adj.values():
            neighbors.pop(vertex, None)
        self._positions.pop(vertex, None)

    @property
    def vertices(self) -> list[str]:
        """Return a list of all vertices."""
        return list(self._adj.keys())

    @property
    def vertex_count(self) -> int:
        return len(self._adj)

    # ── Edge Operations ────────────────────────────────────────────────

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a weighted edge. For undirected graphs, adds both directions.

        Args:
            source: Source vertex identifier.
            target: Target vertex identifier.
            weight: Non-negative edge weight (must be ≥ 0 for Dijkstra).

        Raises:
            ValueError: If weight is negative.
        """
        if weight < 0:
            raise ValueError(f"Edge weight must be ≥ 0, got {weight}")
        self.add_vertex(source)
        self.add_vertex(target)
        self._adj[source][target] = weight
        if not self.directed:
            self._adj[target][source] = weight

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        if source not in self._adj or target not in self._adj[source]:
            raise KeyError(f"Edge ({source} → {target}) not in graph")
        del self._adj[source][target]
        if not self.directed:
            self._adj[target].pop(source, None)

    def has_edge(self, source: str, target: str) -> bool:
        return source in self._adj and target in self._adj[source]

    def get_weight(self, source: str, target: str) -> float:
        """Return the weight of edge (source, target).

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(source, target):
            raise KeyError(f"Edge ({source} → {target}) not in graph")
        return self._adj[source][target]

    def neighbors(self, vertex: str) -> list[tuple[str, float]]:
        """Return (neighbor, weight) pairs for a vertex.

        Raises:
            KeyError: If the vertex does not exist.
        """
        if vertex not in self._adj:
            raise KeyError(f"Vertex '{vertex}' not in graph")
        return list(self._adj[vertex].items())

    @property
    def edges(self) -> list[Edge]:
        """Return all edges. For undirected graphs, each edge appears once."""
        seen: set[tuple[str, str]] = set()
        result: list[Edge] = []
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = (min(u, v), max(u, v)) if not self.directed else (u, v)
                if key not in seen:
                    seen.add(key)
                    result.append(Edge(source=u, target=v, weight=w))
        return result

    @property
    def edge_count(self) -> int:
        total = sum(len(nbrs) for nbrs in self._adj.values())
        return total if self.directed else total // 2

    # ── Position / Coordinates ─────────────────────────────────────────

    def set_position(self, vertex: str, x: float, y: float) -> None:
        """Assign 2D coordinates to a vertex (used by heuristic functions)."""
        self.add_vertex(vertex)
        self._positions[vertex] = (x, y)

    def get_position(self, vertex: str) -> Optional[tuple[float, float]]:
        """Return the (x, y) position of a vertex, or None."""
        return self._positions.get(vertex)

    @property
    def has_positions(self) -> bool:
        """True if all vertices have assigned positions."""
        return len(self._positions) == len(self._adj) and len(self._adj) > 0

    # ── Utilities ──────────────────────────────────────────────────────

    def density(self) -> float:
        """Graph density ∈ [0, 1].

        For directed:   |E| / (|V| × (|V| - 1))
        For undirected: 2|E| / (|V| × (|V| - 1))
        """
        v = self.vertex_count
        if v < 2:
            return 0.0
        max_edges = v * (v - 1) if self.directed else v * (v - 1) // 2
        return self.edge_count / max_edges

    def __repr__(self) -> str:
        kind = "Directed" if self.directed else "Undirected"
        return (
            f"Graph({kind}, V={self.vertex_count}, E={self.edge_count}, "
            f"density={self.density():.4f})"
        )

    def __contains__(self, vertex: str) -> bool:
        return vertex in self._adj
