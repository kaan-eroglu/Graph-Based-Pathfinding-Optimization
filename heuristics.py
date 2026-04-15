"""
Heuristic Functions for A* Search
==================================
Provides admissible heuristic functions for A* pathfinding. A heuristic h(n)
is *admissible* if it never overestimates the true cost to reach the goal:

    ∀n : h(n) ≤ h*(n)

where h*(n) is the true shortest-path cost from n to the goal.

An admissible heuristic guarantees A* optimality (i.e., the first path found
is the shortest path).

A heuristic is *consistent* (monotone) if:

    h(n) ≤ c(n, n') + h(n')

for every successor n' of n. Consistency implies admissibility and ensures
that A* never needs to re-open closed nodes.

Supported Heuristics:
    - Euclidean distance  (L₂ norm) — admissible & consistent for geometric graphs
    - Manhattan distance  (L₁ norm) — admissible & consistent for grid graphs
    - Chebyshev distance  (L∞ norm) — admissible & consistent for 8-connected grids
    - Zero heuristic      (h ≡ 0)  — reduces A* to Dijkstra's algorithm
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Callable, Optional

from graph import Graph


# Type alias for heuristic functions: (current, goal) → estimated cost
HeuristicFn = Callable[[str, str], float]


class HeuristicType(Enum):
    """Enumeration of built-in heuristic strategies."""

    EUCLIDEAN = auto()
    MANHATTAN = auto()
    CHEBYSHEV = auto()
    ZERO = auto()


def euclidean_distance(
    graph: Graph, current: str, goal: str
) -> float:
    """Compute the Euclidean (L₂) distance between two vertices.

    Mathematical definition:
        d(p, q) = √((p₁ − q₁)² + (p₂ − q₂)²)

    This is the straight-line distance and is admissible because
    no path through edges of non-negative weight can be shorter than
    the straight-line distance (triangle inequality).

    Returns:
        The Euclidean distance, or 0.0 if positions are unavailable.
    """
    pos_a = graph.get_position(current)
    pos_b = graph.get_position(goal)
    if pos_a is None or pos_b is None:
        return 0.0
    return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


def manhattan_distance(
    graph: Graph, current: str, goal: str
) -> float:
    """Compute the Manhattan (L₁) distance between two vertices.

    Mathematical definition:
        d(p, q) = |p₁ − q₁| + |p₂ − q₂|

    Admissible for grid-based graphs where movement is restricted to
    the four cardinal directions, since the shortest grid path length
    is exactly the L₁ norm.

    Returns:
        The Manhattan distance, or 0.0 if positions are unavailable.
    """
    pos_a = graph.get_position(current)
    pos_b = graph.get_position(goal)
    if pos_a is None or pos_b is None:
        return 0.0
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def chebyshev_distance(
    graph: Graph, current: str, goal: str
) -> float:
    """Compute the Chebyshev (L∞) distance between two vertices.

    Mathematical definition:
        d(p, q) = max(|p₁ − q₁|, |p₂ − q₂|)

    Admissible for grids with 8-directional movement (including diagonals),
    where the cost of each move (including diagonal) is uniform.

    Returns:
        The Chebyshev distance, or 0.0 if positions are unavailable.
    """
    pos_a = graph.get_position(current)
    pos_b = graph.get_position(goal)
    if pos_a is None or pos_b is None:
        return 0.0
    return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))


def zero_heuristic(
    graph: Graph, current: str, goal: str
) -> float:
    """Trivially admissible heuristic that always returns 0.

    Using h(n) = 0 causes A* to degenerate into Dijkstra's algorithm,
    since the priority becomes f(n) = g(n) + 0 = g(n). This is useful
    as a baseline for benchmarking.
    """
    return 0.0


# ── Factory ────────────────────────────────────────────────────────────

_HEURISTIC_MAP: dict[HeuristicType, Callable[..., float]] = {
    HeuristicType.EUCLIDEAN: euclidean_distance,
    HeuristicType.MANHATTAN: manhattan_distance,
    HeuristicType.CHEBYSHEV: chebyshev_distance,
    HeuristicType.ZERO: zero_heuristic,
}


def _compute_scale_factor(
    graph: Graph,
    distance_fn: Callable[..., float],
) -> float:
    """Compute a scaling factor to guarantee heuristic admissibility.

    For arbitrary weighted graphs, the geometric distance between two
    vertices may exceed the edge weight connecting them, violating
    the admissibility condition h(n) ≤ h*(n).

    We compute:
        α = min over all edges (u,v):  w(u,v) / d(u,v)

    where d(u,v) is the geometric distance. Scaling the heuristic by α
    ensures that for any edge: α·d(u,v) ≤ w(u,v), which by induction
    over any path guarantees α·d(start,goal) ≤ shortest_path_weight.

    Returns:
        A scaling factor α ∈ (0, 1] that ensures admissibility.
        Returns 1.0 if positions are unavailable or if the graph has
        edges with zero geometric distance.
    """
    if not graph.has_positions:
        return 1.0

    min_ratio = float("inf")
    for edge in graph.edges:
        geo_dist = distance_fn(graph, edge.source, edge.target)
        if geo_dist > 1e-9:  # avoid division by zero
            ratio = edge.weight / geo_dist
            if ratio < min_ratio:
                min_ratio = ratio

    return min_ratio if min_ratio < float("inf") else 1.0


def get_heuristic(
    heuristic_type: HeuristicType, graph: Graph
) -> HeuristicFn:
    """Return a heuristic function bound to a specific graph.

    The returned callable has signature (current: str, goal: str) → float,
    which is the interface expected by the A* implementation.

    For non-grid random graphs, a scaling factor α is computed from
    the minimum weight-to-distance ratio across all edges. This ensures
    admissibility (h never overestimates) while still providing useful
    guidance toward the goal.

    Args:
        heuristic_type: The heuristic strategy to use.
        graph: The graph instance (needed for vertex positions).

    Returns:
        A callable heuristic function.
    """
    if heuristic_type == HeuristicType.ZERO:
        return lambda current, goal: 0.0

    raw = _HEURISTIC_MAP[heuristic_type]

    # Compute admissibility scaling factor
    scale = _compute_scale_factor(graph, raw)

    return lambda current, goal: scale * raw(graph, current, goal)
