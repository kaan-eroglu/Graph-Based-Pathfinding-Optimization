# Graph-Based-Pathfinding-Optimization
A high-performance Python tool for implementing, benchmarking, and comparing **Dijkstra's** and **A\*** shortest-path algorithms on weighted graphs.

## Features

- **Graph Architecture** — Weighted, directed and undirected graphs using adjacency list representation with O(1) edge lookup
- **Dijkstra's Algorithm** — Priority queue (min-heap) implementation with O((E + V) log V) time complexity
- **A\* Algorithm** — Supports pluggable heuristic functions (Euclidean, Manhattan, Chebyshev, Zero)
- **Graph Generator** — Random sparse, dense, and grid graph generation with guaranteed connectivity
- **Benchmarking** — Decorator-based performance measurement with detailed comparison tables
- **CLI Interface** — ANSI-styled terminal output with configurable graph parameters

## Project Structure

```
├── graph.py        # Core graph data structure (adjacency list)
├── algorithms.py   # Dijkstra's and A* implementations
├── heuristics.py   # Heuristic functions with admissibility proofs
├── generator.py    # Random graph generation for stress testing
├── benchmark.py    # Performance measurement decorator & comparison
├── main.py         # CLI entry point
└── README.md
```

## Quick Start

```bash
# Default demo — 100 vertices, density 0.15
python main.py

# Sparse graph stress test
python main.py --vertices 500 --sparse

# Dense graph stress test
python main.py --vertices 300 --dense

# Grid graph (ideal for Manhattan heuristic)
python main.py --grid 20 20

# Custom configuration
python main.py --vertices 200 --density 0.3 --directed --seed 42

# Select specific heuristics
python main.py --vertices 150 --heuristic euclidean manhattan chebyshev zero
```

## Algorithm Comparison

| Property | Dijkstra | A\* |
|----------|----------|-----|
| Priority | g(n) | f(n) = g(n) + h(n) |
| Optimality | Always (non-negative weights) | When h(n) is admissible |
| Time Complexity | O((E + V) log V) | O((E + V) log V) worst-case |
| Nodes Visited | Typically more | Fewer with good heuristic |

## Heuristics

| Heuristic | Formula | Best For |
|-----------|---------|----------|
| Euclidean | √((x₁−x₂)² + (y₁−y₂)²) | Geometric / spatial graphs |
| Manhattan | \|x₁−x₂\| + \|y₁−y₂\| | Grid graphs (4-connected) |
| Chebyshev | max(\|x₁−x₂\|, \|y₁−y₂\|) | Grid graphs (8-connected) |
| Zero | h(n) = 0 | Baseline (degenerates to Dijkstra) |

## Requirements

- Python 3.10+ (uses `match` statements, `slots`, and modern type hints)
- No external dependencies — built entirely with the Python standard library

## Mathematical Foundation

The implementation is grounded in the following theoretical guarantees:

- **Dijkstra Optimality**: For graphs with non-negative edge weights, Dijkstra's greedy relaxation produces the true shortest path.
- **A\* Admissibility**: When h(n) ≤ h\*(n) for all nodes (heuristic never overestimates), A\* is guaranteed to find the optimal path.
- **A\* Consistency**: When h(n) ≤ c(n, n') + h(n') (triangle inequality), closed nodes are never re-expanded, ensuring worst-case O((E + V) log V).

## License

MIT
