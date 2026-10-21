"""
Benchmarking, Accuracy Testing & Performance Analysis Module (Module 3).

This module provides tools for:
- Exact brute-force K-Nearest Neighbor search baseline.
- Empirical Recall@K accuracy evaluation.
- Microsecond-precision query latency profiling (mean, median, p95, p99, QPS).
- Heap memory usage profiling via tracemalloc.
- Index construction throughput and scaling analysis.
- Multi-dimensional parameter sweep across M, ef_construction, and ef_search.
- Publication-quality CSV reports, SVG vector charts, and Matplotlib PNG visualizations.
"""

from benchmark.baseline import ExactKNNBaseline
from benchmark.metrics import (
    LatencyStats,
    SystemProfiler,
    compute_latency_stats,
    compute_recall_at_k,
)
from benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
)
from benchmark.synthetic_data import generate_synthetic_vectors

__all__ = [
    "ExactKNNBaseline",
    "compute_recall_at_k",
    "compute_latency_stats",
    "LatencyStats",
    "SystemProfiler",
    "generate_synthetic_vectors",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
]
