"""
Pathfinding Algorithms
======================
Implements Dijkstra's and A* shortest-path algorithms with a unified
result interface and detailed performance metrics.

Complexity Analysis:
    Both algorithms use a binary min-heap (heapq) as the priority queue.

    Dijkstra's Algorithm:
        Time  : O((E + V) log V)  — each vertex extracted once, each edge relaxed once
        Space : O(V)              — distance table + predecessor map

    A* Algorithm:
        Time  : O((E + V) log V)  — same worst-case, but in practice explores fewer
                                     nodes when h(n) is a good heuristic
        Space : O(V)              — same as Dijkstra

    The key difference is in the priority function:
        Dijkstra:  priority = g(n)           (cost so far)
        A*:        priority = g(n) + h(n)    (cost so far + estimated remaining)

    where g(n) is the known shortest distance from start to n, and h(n) is
    the heuristic estimate from n to the goal.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional

from graph import Graph


# Type alias for the heuristic function signature
HeuristicFn = Callable[[str, str], float]


@dataclass
class PathResult:
    """Unified result container for pathfinding algorithms.

    Attributes:
        algorithm: Name of the algorithm used.
        found: Whether a path was found.
        path: Ordered list of vertices from start to goal.
        total_weight: Sum of edge weights along the path.
        nodes_visited: Number of unique nodes dequeued (expanded).
        nodes_explored: Number of times edges were relaxed.
    """

    algorithm: str
    found: bool
    path: list[str] = field(default_factory=list)
    total_weight: float = float("inf")
    nodes_visited: int = 0
    nodes_explored: int = 0

    def summary_line(self) -> str:
        """One-line summary suitable for CLI output."""
        if not self.found:
            return f"[{self.algorithm}] No path found."
        return (
            f"[{self.algorithm}] Weight: {self.total_weight:.4f} | "
            f"Path length: {len(self.path)} nodes | "
            f"Visited: {self.nodes_visited} | "
            f"Explored: {self.nodes_explored}"
        )


def _reconstruct_path(
    predecessors: dict[str, Optional[str]], start: str, goal: str
) -> list[str]:
    """Backtrack through the predecessor map to reconstruct the path."""
    path: list[str] = []
    current: Optional[str] = goal
    while current is not None:
        path.append(current)
        current = predecessors.get(current)
    path.reverse()
    return path if path and path[0] == start else []


# ── Dijkstra's Algorithm ──────────────────────────────────────────────


def dijkstra(
    graph: Graph, start: str, goal: str
) -> PathResult:
    """Find the shortest path using Dijkstra's algorithm.

    Dijkstra's algorithm greedily expands the vertex with the smallest
    known distance g(n). It is optimal for graphs with non-negative edge
    weights.

    Relaxation condition:
        if g(u) + w(u, v) < g(v):
            g(v) ← g(u) + w(u, v)
            predecessor(v) ← u

    Args:
        graph: The graph to search.
        start: Source vertex identifier.
        goal: Target vertex identifier.

    Returns:
        A PathResult with the shortest path and performance metrics.

    Raises:
        KeyError: If start or goal vertex is not in the graph.
    """
    if start not in graph:
        raise KeyError(f"Start vertex '{start}' not in graph")
    if goal not in graph:
        raise KeyError(f"Goal vertex '{goal}' not in graph")

    dist: dict[str, float] = {start: 0.0}
    predecessors: dict[str, Optional[str]] = {start: None}
    visited: set[str] = set()
    nodes_explored: int = 0

    # Priority queue: (distance, tiebreaker, vertex)
    counter: int = 0
    pq: list[tuple[float, int, str]] = [(0.0, counter, start)]

    while pq:
        current_dist, _, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        # Early termination when goal is reached
        if u == goal:
            path = _reconstruct_path(predecessors, start, goal)
            return PathResult(
                algorithm="Dijkstra",
                found=True,
                path=path,
                total_weight=current_dist,
                nodes_visited=len(visited),
                nodes_explored=nodes_explored,
            )

        for neighbor, weight in graph.neighbors(u):
            nodes_explored += 1
            new_dist = current_dist + weight
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                predecessors[neighbor] = u
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))

    return PathResult(
        algorithm="Dijkstra",
        found=False,
        nodes_visited=len(visited),
        nodes_explored=nodes_explored,
    )


# ── A* Algorithm ──────────────────────────────────────────────────────


def astar(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: HeuristicFn,
) -> PathResult:
    """Find the shortest path using the A* algorithm.

    A* extends Dijkstra by adding a heuristic estimate h(n) to guide the
    search toward the goal. The priority of a node n is:

        f(n) = g(n) + h(n)

    where:
        g(n) = known shortest distance from start to n
        h(n) = heuristic estimate of cost from n to goal

    Optimality guarantee:
        A* is optimal (finds the true shortest path) if h(n) is admissible,
        meaning h(n) ≤ h*(n) for all n, where h*(n) is the true cost.

    Efficiency:
        With a consistent heuristic (h(n) ≤ c(n,n') + h(n')), A* never
        re-expands a node once it has been visited, matching Dijkstra's
        worst-case complexity while typically visiting far fewer nodes.

    Args:
        graph: The graph to search.
        start: Source vertex identifier.
        goal: Target vertex identifier.
        heuristic: A callable h(current, goal) → float.

    Returns:
        A PathResult with the shortest path and performance metrics.

    Raises:
        KeyError: If start or goal vertex is not in the graph.
    """
    if start not in graph:
        raise KeyError(f"Start vertex '{start}' not in graph")
    if goal not in graph:
        raise KeyError(f"Goal vertex '{goal}' not in graph")

    g_score: dict[str, float] = {start: 0.0}
    predecessors: dict[str, Optional[str]] = {start: None}
    visited: set[str] = set()
    nodes_explored: int = 0

    # f(n) = g(n) + h(n)
    f_start = heuristic(start, goal)
    counter: int = 0
    pq: list[tuple[float, int, str]] = [(f_start, counter, start)]

    while pq:
        current_f, _, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == goal:
            path = _reconstruct_path(predecessors, start, goal)
            return PathResult(
                algorithm="A*",
                found=True,
                path=path,
                total_weight=g_score[goal],
                nodes_visited=len(visited),
                nodes_explored=nodes_explored,
            )

        for neighbor, weight in graph.neighbors(u):
            nodes_explored += 1
            tentative_g = g_score[u] + weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                predecessors[neighbor] = u
                f_val = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(pq, (f_val, counter, neighbor))

    return PathResult(
        algorithm="A*",
        found=False,
        nodes_visited=len(visited),
        nodes_explored=nodes_explored,
    )
