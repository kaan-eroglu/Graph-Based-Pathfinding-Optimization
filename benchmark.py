"""
Benchmarking Module
===================
Provides a decorator and utilities to measure and compare the execution
time and node-visit metrics of pathfinding algorithms.

The benchmarking decorator captures:
    - Wall-clock execution time (via time.perf_counter for high resolution)
    - Nodes visited (from PathResult)
    - Nodes explored (edges relaxed, from PathResult)
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from algorithms import PathResult


@dataclass
class BenchmarkResult:
    """Container for benchmark measurements.

    Attributes:
        algorithm: Name of the benchmarked algorithm.
        execution_time_ms: Wall-clock time in milliseconds.
        path_result: The underlying PathResult from the algorithm.
    """

    algorithm: str
    execution_time_ms: float
    path_result: PathResult

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.algorithm}, "
            f"time={self.execution_time_ms:.3f}ms, "
            f"visited={self.path_result.nodes_visited})"
        )


def benchmark(func: Callable[..., PathResult]) -> Callable[..., BenchmarkResult]:
    """Decorator that measures execution time of a pathfinding function.

    Wraps any function that returns a PathResult and produces a
    BenchmarkResult with high-resolution timing information.

    Usage:
        @benchmark
        def my_dijkstra(graph, start, goal):
            return dijkstra(graph, start, goal)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> BenchmarkResult:
        start_time = time.perf_counter()
        result: PathResult = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000.0

        return BenchmarkResult(
            algorithm=result.algorithm,
            execution_time_ms=elapsed_ms,
            path_result=result,
        )

    return wrapper


def compare_algorithms(
    benchmarks: list[BenchmarkResult],
) -> str:
    """Generate a formatted comparison table from multiple benchmark results.

    Args:
        benchmarks: List of BenchmarkResult instances to compare.

    Returns:
        A formatted string containing the comparison table.
    """
    if not benchmarks:
        return "No benchmark results to compare."

    # Header
    header = (
        f"{'Algorithm':<12} │ {'Time (ms)':>12} │ {'Visited':>9} │ "
        f"{'Explored':>10} │ {'Path Len':>10} │ {'Weight':>12} │ {'Found':>6}"
    )
    separator = "─" * len(header)

    lines: list[str] = [
        "",
        "╔" + "═" * (len(header) + 2) + "╗",
        "║ " + " PERFORMANCE COMPARISON ".center(len(header)) + " ║",
        "╚" + "═" * (len(header) + 2) + "╝",
        "",
        header,
        separator,
    ]

    for b in benchmarks:
        pr = b.path_result
        weight_str = f"{pr.total_weight:.4f}" if pr.found else "∞"
        path_len = str(len(pr.path)) if pr.found else "—"
        found_str = "✓" if pr.found else "✗"
        lines.append(
            f"{b.algorithm:<12} │ {b.execution_time_ms:>12.3f} │ "
            f"{pr.nodes_visited:>9} │ {pr.nodes_explored:>10} │ "
            f"{path_len:>10} │ {weight_str:>12} │ {found_str:>6}"
        )

    lines.append(separator)

    # Speedup analysis
    if len(benchmarks) >= 2:
        times = [(b.algorithm, b.execution_time_ms) for b in benchmarks]
        times.sort(key=lambda x: x[1])
        fastest_name, fastest_time = times[0]
        slowest_name, slowest_time = times[-1]

        if fastest_time > 0:
            speedup = slowest_time / fastest_time
            lines.append(
                f"\n⚡ {fastest_name} was {speedup:.2f}× faster than {slowest_name}"
            )

        # Node efficiency
        visits = [(b.algorithm, b.path_result.nodes_visited) for b in benchmarks]
        visits.sort(key=lambda x: x[1])
        most_efficient = visits[0]
        least_efficient = visits[-1]
        if most_efficient[1] > 0 and most_efficient[1] != least_efficient[1]:
            ratio = least_efficient[1] / most_efficient[1]
            lines.append(
                f"🔍 {most_efficient[0]} visited {ratio:.2f}× fewer nodes "
                f"than {least_efficient[0]} "
                f"({most_efficient[1]} vs {least_efficient[1]})"
            )

    return "\n".join(lines)
