#!/usr/bin/env python3
"""
Graph-Based Pathfinding Optimization — CLI Interface
=====================================================
A high-performance tool for comparing Dijkstra's and A* pathfinding
algorithms on various graph topologies.

Usage:
    python main.py                          # Run default demonstration
    python main.py --vertices 500 --dense   # Stress test with 500 vertices
    python main.py --grid 20 20             # 20×20 grid graph
    python main.py --help                   # Show all options
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from algorithms import PathResult, astar, dijkstra
from benchmark import BenchmarkResult, benchmark, compare_algorithms
from generator import GraphGenerator
from graph import Graph
from heuristics import HeuristicType, get_heuristic


# ── ANSI Color Codes ──────────────────────────────────────────────────

class Style:
    """ANSI escape codes for terminal styling."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    BG_DARK = "\033[48;5;235m"
    UNDERLINE = "\033[4m"


def _header(text: str) -> str:
    """Format a section header."""
    return f"\n{Style.BOLD}{Style.CYAN}{'━' * 60}{Style.RESET}\n{Style.BOLD}{Style.WHITE}  {text}{Style.RESET}\n{Style.BOLD}{Style.CYAN}{'━' * 60}{Style.RESET}"


def _sub_header(text: str) -> str:
    return f"\n  {Style.BOLD}{Style.YELLOW}▸ {text}{Style.RESET}"


def _info(label: str, value: str) -> str:
    return f"  {Style.DIM}{label}:{Style.RESET} {Style.WHITE}{value}{Style.RESET}"


def _success(text: str) -> str:
    return f"  {Style.GREEN}✓{Style.RESET} {text}"


def _error(text: str) -> str:
    return f"  {Style.RED}✗{Style.RESET} {text}"


# ── Benchmarked Algorithm Wrappers ────────────────────────────────────

@benchmark
def run_dijkstra(graph: Graph, start: str, goal: str) -> PathResult:
    """Benchmarked wrapper for Dijkstra's algorithm."""
    return dijkstra(graph, start, goal)


@benchmark
def run_astar(
    graph: Graph, start: str, goal: str, heuristic_type: HeuristicType
) -> PathResult:
    """Benchmarked wrapper for A* algorithm."""
    h = get_heuristic(heuristic_type, graph)
    return astar(graph, start, goal, h)


# ── Display Functions ─────────────────────────────────────────────────

def display_path(result: PathResult, label: str = "") -> None:
    """Display a single pathfinding result."""
    prefix = f" ({label})" if label else ""
    print(_sub_header(f"{result.algorithm}{prefix}"))

    if not result.found:
        print(_error("No path found between the specified vertices."))
        return

    print(_info("Total weight", f"{result.total_weight:.4f}"))
    print(_info("Path length ", f"{len(result.path)} nodes"))
    print(_info("Nodes visited", str(result.nodes_visited)))
    print(_info("Edges relaxed", str(result.nodes_explored)))

    # Show path (truncated for large paths)
    if len(result.path) <= 20:
        path_str = " → ".join(result.path)
    else:
        head = " → ".join(result.path[:8])
        tail = " → ".join(result.path[-4:])
        path_str = f"{head} → ... ({len(result.path) - 12} more) ... → {tail}"

    print(f"  {Style.DIM}Path:{Style.RESET} {Style.MAGENTA}{path_str}{Style.RESET}")


def display_graph_info(graph: Graph, label: str = "Graph") -> None:
    """Display graph statistics."""
    print(_header(f"📊 {label} Summary"))
    kind = "Directed" if graph.directed else "Undirected"
    print(_info("Type    ", kind))
    print(_info("Vertices", str(graph.vertex_count)))
    print(_info("Edges   ", str(graph.edge_count)))
    print(_info("Density ", f"{graph.density():.4f}"))
    print(_info("Positions", "Yes" if graph.has_positions else "No"))


def run_comparison(
    graph: Graph,
    start: str,
    goal: str,
    heuristic_types: list[HeuristicType],
    label: str = "Pathfinding",
) -> None:
    """Run Dijkstra and A* with multiple heuristics, then compare."""
    print(_header(f"🔍 {label}: {start} → {goal}"))

    benchmarks: list[BenchmarkResult] = []

    # Dijkstra
    b_dijkstra = run_dijkstra(graph, start, goal)
    display_path(b_dijkstra.path_result)
    benchmarks.append(b_dijkstra)

    # A* with each heuristic
    for h_type in heuristic_types:
        b_astar = run_astar(graph, start, goal, h_type)
        # Override algorithm name for clarity
        b_astar.algorithm = f"A*({h_type.name[:4]})"
        b_astar.path_result.algorithm = f"A*({h_type.name[:4]})"
        display_path(b_astar.path_result, h_type.name)
        benchmarks.append(b_astar)

    # Comparison summary
    print(_header("📈 Performance Comparison"))
    print(compare_algorithms(benchmarks))


# ── CLI Argument Parsing ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathfinder",
        description="Graph-Based Pathfinding Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                          # Default demo\n"
            "  python main.py --vertices 200 --sparse  # Sparse 200-vertex graph\n"
            "  python main.py --vertices 500 --dense   # Dense 500-vertex graph\n"
            "  python main.py --grid 15 15             # 15×15 grid graph\n"
            "  python main.py --seed 42                # Reproducible run\n"
        ),
    )

    graph_group = parser.add_argument_group("Graph Generation")
    graph_group.add_argument(
        "--vertices", "-v", type=int, default=100,
        help="Number of vertices (default: 100)",
    )
    graph_group.add_argument(
        "--density", "-d", type=float, default=None,
        help="Edge density ∈ (0, 1]. Overrides --sparse/--dense.",
    )
    graph_group.add_argument(
        "--sparse", action="store_true",
        help="Generate a sparse graph (auto-density).",
    )
    graph_group.add_argument(
        "--dense", action="store_true",
        help="Generate a dense graph (auto-density).",
    )
    graph_group.add_argument(
        "--grid", nargs=2, type=int, metavar=("ROWS", "COLS"),
        help="Generate a grid graph with ROWS × COLS vertices.",
    )
    graph_group.add_argument(
        "--directed", action="store_true",
        help="Create a directed graph (default: undirected).",
    )
    graph_group.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    path_group = parser.add_argument_group("Pathfinding")
    path_group.add_argument(
        "--start", type=str, default=None,
        help="Start vertex (default: first vertex).",
    )
    path_group.add_argument(
        "--goal", type=str, default=None,
        help="Goal vertex (default: last vertex).",
    )

    heuristic_group = parser.add_argument_group("Heuristics")
    heuristic_group.add_argument(
        "--heuristic", "-H",
        nargs="+",
        choices=["euclidean", "manhattan", "chebyshev", "zero"],
        default=["euclidean", "manhattan"],
        help="Heuristic(s) for A* (default: euclidean manhattan).",
    )

    return parser


# ── Main Entry Point ──────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Banner
    print(f"\n{Style.BOLD}{Style.CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     Graph-Based Pathfinding Optimization Tool       ║")
    print("  ║     Dijkstra's vs A* Algorithm Comparison           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(Style.RESET)

    # Generate graph
    gen = GraphGenerator(seed=args.seed)

    if args.grid:
        rows, cols = args.grid
        graph = gen.generate_grid(rows, cols)
        start = args.start or "v0_0"
        goal = args.goal or f"v{rows-1}_{cols-1}"
        graph_label = f"Grid Graph ({rows}×{cols})"
    elif args.sparse:
        graph = gen.generate_sparse(args.vertices, directed=args.directed)
        start = args.start or "v0"
        goal = args.goal or f"v{args.vertices - 1}"
        graph_label = f"Sparse Random Graph (V={args.vertices})"
    elif args.dense:
        graph = gen.generate_dense(args.vertices, directed=args.directed)
        start = args.start or "v0"
        goal = args.goal or f"v{args.vertices - 1}"
        graph_label = f"Dense Random Graph (V={args.vertices})"
    elif args.density is not None:
        graph = gen.generate(
            args.vertices, density=args.density, directed=args.directed
        )
        start = args.start or "v0"
        goal = args.goal or f"v{args.vertices - 1}"
        graph_label = f"Random Graph (V={args.vertices}, d={args.density:.2f})"
    else:
        # Default demo: moderate graph
        graph = gen.generate(args.vertices, density=0.15, directed=args.directed)
        start = args.start or "v0"
        goal = args.goal or f"v{args.vertices - 1}"
        graph_label = f"Random Graph (V={args.vertices}, d=0.15)"

    # Validate vertices
    if start not in graph:
        print(_error(f"Start vertex '{start}' not found in graph."))
        sys.exit(1)
    if goal not in graph:
        print(_error(f"Goal vertex '{goal}' not found in graph."))
        sys.exit(1)

    # Display graph info
    display_graph_info(graph, graph_label)

    # Map heuristic names to enum
    heuristic_map = {
        "euclidean": HeuristicType.EUCLIDEAN,
        "manhattan": HeuristicType.MANHATTAN,
        "chebyshev": HeuristicType.CHEBYSHEV,
        "zero": HeuristicType.ZERO,
    }
    heuristic_types = [heuristic_map[h] for h in args.heuristic]

    # Run comparison
    run_comparison(graph, start, goal, heuristic_types, graph_label)

    print(f"\n{Style.DIM}  Seed: {args.seed or 'random'}{Style.RESET}")
    print()


if __name__ == "__main__":
    main()
