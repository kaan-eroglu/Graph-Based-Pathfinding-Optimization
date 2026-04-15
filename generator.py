"""
Graph Generator
===============
Creates random weighted graphs for stress testing and benchmarking.
Supports generating both sparse and dense graphs with configurable
parameters and optional 2D coordinate assignment for heuristic testing.

Density Model:
    For a graph with V vertices, the maximum number of edges is:
        Directed:   V × (V − 1)
        Undirected: V × (V − 1) / 2

    The `density` parameter ∈ (0, 1] controls what fraction of the
    maximum edges are generated. Sparse graphs typically have
    density < 0.1, while dense graphs have density > 0.5.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from graph import Graph


class GraphGenerator:
    """Factory for generating random weighted graphs.

    Uses a seeded RNG for reproducibility. All generated graphs have
    vertex labels "v0", "v1", ..., "v{n-1}".
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the generator with an optional random seed.

        Args:
            seed: Seed for the random number generator. If None, a
                  random seed is used.
        """
        self._rng = random.Random(seed)

    def generate(
        self,
        num_vertices: int,
        density: float = 0.3,
        directed: bool = False,
        weight_range: tuple[float, float] = (1.0, 10.0),
        assign_positions: bool = True,
        grid_size: float = 100.0,
        ensure_connected: bool = True,
    ) -> Graph:
        """Generate a random weighted graph.

        Args:
            num_vertices: Number of vertices (V).
            density: Edge density ∈ (0, 1]. Fraction of max possible edges.
            directed: Whether to create a directed graph.
            weight_range: (min_weight, max_weight) for random edge weights.
            assign_positions: If True, assign random 2D coordinates to
                              each vertex for heuristic computation.
            grid_size: Coordinate range [0, grid_size) for positions.
            ensure_connected: If True, first create a spanning tree to
                              guarantee connectivity, then add random edges
                              up to the target density.

        Returns:
            A Graph instance populated with random vertices and edges.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if num_vertices < 2:
            raise ValueError("Need at least 2 vertices")
        if not (0.0 < density <= 1.0):
            raise ValueError(f"Density must be in (0, 1], got {density}")
        if weight_range[0] > weight_range[1] or weight_range[0] < 0:
            raise ValueError(f"Invalid weight range: {weight_range}")

        g = Graph(directed=directed)
        vertices = [f"v{i}" for i in range(num_vertices)]

        # Add all vertices
        for v in vertices:
            g.add_vertex(v)

        # Assign random 2D positions
        if assign_positions:
            for v in vertices:
                x = self._rng.uniform(0, grid_size)
                y = self._rng.uniform(0, grid_size)
                g.set_position(v, x, y)

        # Calculate target edge count
        max_edges = (
            num_vertices * (num_vertices - 1)
            if directed
            else num_vertices * (num_vertices - 1) // 2
        )
        target_edges = max(num_vertices - 1, int(max_edges * density))

        # Phase 1: Ensure connectivity with a random spanning tree
        edges_added: set[tuple[str, str]] = set()
        if ensure_connected:
            shuffled = vertices[:]
            self._rng.shuffle(shuffled)
            for i in range(1, len(shuffled)):
                u, v = shuffled[i - 1], shuffled[i]
                w = self._rng.uniform(*weight_range)
                g.add_edge(u, v, round(w, 2))
                key = (u, v) if directed else (min(u, v), max(u, v))
                edges_added.add(key)

        # Phase 2: Add random edges up to target density
        all_possible: list[tuple[str, str]] = []
        for i, u in enumerate(vertices):
            for j, v in enumerate(vertices):
                if i == j:
                    continue
                if directed:
                    key = (u, v)
                else:
                    if i > j:
                        continue
                    key = (u, v)
                if key not in edges_added:
                    all_possible.append(key)

        self._rng.shuffle(all_possible)
        remaining = target_edges - len(edges_added)
        for u, v in all_possible[:remaining]:
            w = self._rng.uniform(*weight_range)
            g.add_edge(u, v, round(w, 2))

        return g

    def generate_sparse(
        self,
        num_vertices: int,
        directed: bool = False,
        **kwargs,
    ) -> Graph:
        """Convenience method: generate a sparse graph (density ≈ 0.05–0.1).

        Sparse graphs have E ≈ O(V), making them ideal for testing
        algorithm performance on tree-like structures.
        """
        density = max(0.05, min(0.15, 3.0 / num_vertices))
        return self.generate(
            num_vertices, density=density, directed=directed, **kwargs
        )

    def generate_dense(
        self,
        num_vertices: int,
        directed: bool = False,
        **kwargs,
    ) -> Graph:
        """Convenience method: generate a dense graph (density ≈ 0.5–0.8).

        Dense graphs have E ≈ O(V²), stress-testing the priority queue
        operations and memory usage.
        """
        density = self._rng.uniform(0.5, 0.8)
        return self.generate(
            num_vertices, density=density, directed=directed, **kwargs
        )

    def generate_grid(
        self,
        rows: int,
        cols: int,
        weight_range: tuple[float, float] = (1.0, 5.0),
    ) -> Graph:
        """Generate a 2D grid graph with 4-connected neighbors.

        Grid graphs are the natural test case for Manhattan distance
        heuristic, where it achieves exact estimation.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            weight_range: Range for random edge weights.

        Returns:
            An undirected grid graph with positions.
        """
        g = Graph(directed=False)
        for r in range(rows):
            for c in range(cols):
                name = f"v{r}_{c}"
                g.add_vertex(name)
                g.set_position(name, float(c), float(r))

        for r in range(rows):
            for c in range(cols):
                current = f"v{r}_{c}"
                # Right neighbor
                if c + 1 < cols:
                    w = self._rng.uniform(*weight_range)
                    g.add_edge(current, f"v{r}_{c+1}", round(w, 2))
                # Down neighbor
                if r + 1 < rows:
                    w = self._rng.uniform(*weight_range)
                    g.add_edge(current, f"v{r+1}_{c}", round(w, 2))

        return g
