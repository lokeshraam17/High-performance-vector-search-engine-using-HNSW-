"""
Benchmarking Metrics, Statistical Latency Profiling & System Diagnostics.

This module provides:
- Recall@K calculation comparing ANN retrieval against exact ground truth.
- Latency statistics (mean, median/p50, p90, p95, p99, min, max, std dev, QPS).
- System environment profiling (OS, CPU, Python build, total RAM).
- Memory tracking via Python standard library tracemalloc.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import datetime
import math
import os
import platform
import statistics
import tracemalloc
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple


@dataclass
class LatencyStats:
    """Statistical summary of query latencies in milliseconds and QPS."""
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    qps: float
    total_queries: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard dictionary."""
        return asdict(self)


def compute_recall_at_k(
    retrieved: Sequence[Tuple[Any, float]],
    ground_truth: Sequence[Tuple[Any, float]],
    k: int,
) -> float:
    """
    Compute Recall@K between retrieved ANN candidates and exact ground truth.
    
    Formula:
        Recall@K = |Retrieved_K ∩ GroundTruth_K| / K
    
    Args:
        retrieved: List of (node_id, distance) tuples returned by HNSW.
        ground_truth: List of (node_id, distance) tuples returned by exact baseline.
        k: Number of top neighbors evaluated (must be >= 1).
        
    Returns:
        Recall score in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer >= 1, got {k}.")
    if len(ground_truth) == 0:
        return 1.0 if len(retrieved) == 0 else 0.0

    actual_k = min(k, len(ground_truth))
    if actual_k == 0:
        return 1.0

    gt_ids: Set[Any] = {item[0] for item in ground_truth[:actual_k]}
    ret_ids: Set[Any] = {item[0] for item in retrieved[:actual_k]}

    exact_matches = len(ret_ids.intersection(gt_ids))
    recall = exact_matches / float(actual_k)
    return max(0.0, min(1.0, recall))


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """
    Calculate statistical distributions from an array of latencies in milliseconds.
    
    Args:
        latencies_ms: Sequence of latency measurements in milliseconds.
        
    Returns:
        LatencyStats dataclass with mean, percentiles, and QPS.
    """
    if not latencies_ms:
        raise ValueError("Cannot compute latency statistics on an empty sequence.")

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    mean_val = statistics.fmean(sorted_latencies) if hasattr(statistics, "fmean") else statistics.mean(sorted_latencies)
    median_val = statistics.median(sorted_latencies)
    std_val = statistics.stdev(sorted_latencies) if n > 1 else 0.0

    def percentile(p: float) -> float:
        """Nearest-rank percentile computation."""
        idx = int(math.ceil(p * n)) - 1
        return sorted_latencies[max(0, min(n - 1, idx))]

    p90 = percentile(0.90)
    p95 = percentile(0.95)
    p99 = percentile(0.99)
    min_val = sorted_latencies[0]
    max_val = sorted_latencies[-1]

    # QPS = total queries / total query time in seconds
    total_time_sec = sum(sorted_latencies) / 1000.0
    qps = (n / total_time_sec) if total_time_sec > 0 else 0.0

    return LatencyStats(
        mean_ms=round(mean_val, 4),
        median_ms=round(median_val, 4),
        p90_ms=round(p90, 4),
        p95_ms=round(p95, 4),
        p99_ms=round(p99, 4),
        min_ms=round(min_val, 4),
        max_ms=round(max_val, 4),
        std_ms=round(std_val, 4),
        qps=round(qps, 2),
        total_queries=n,
    )


class SystemProfiler:
    """System hardware, operating system, and runtime profiler."""

    @staticmethod
    def get_total_ram_gb() -> Optional[float]:
        """Estimate total system RAM in Gigabytes without third-party dependencies."""
        try:
            if platform.system() == "Windows":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return round(stat.ullTotalPhys / (1024 ** 3), 2)
            elif hasattr(os, "sysconf"):
                if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                    return round(
                        (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
                        / (1024 ** 3),
                        2,
                    )
        except Exception:
            pass
        return None

    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Collect platform and runtime diagnostic metadata."""
        return {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "Unknown",
            "cpu_count": os.cpu_count() or 1,
            "total_ram_gb": cls.get_total_ram_gb() or "Unknown",
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    @staticmethod
    @contextmanager
    def track_memory() -> Generator[Dict[str, float], None, None]:
        """
        Context manager measuring Python heap memory usage during an operation.
        
        Yields:
            Dictionary that will be populated with 'peak_memory_mb' and 'net_memory_mb'.
        """
        tracemalloc.start()
        tracemalloc.reset_peak()
        metrics: Dict[str, float] = {"peak_memory_mb": 0.0, "net_memory_mb": 0.0}
        start_current, _ = tracemalloc.get_traced_memory()
        try:
            yield metrics
        finally:
            end_current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            metrics["peak_memory_mb"] = round(peak / (1024 * 1024), 3)
            metrics["net_memory_mb"] = round(max(0, end_current - start_current) / (1024 * 1024), 3)
