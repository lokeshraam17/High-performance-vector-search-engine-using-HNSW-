"""
Benchmark Execution Engine for HNSW vs Exact KNN Baseline.

This module orchestrates rigorous, reproducible benchmarking experiments:
- Fair comparison under identical query vectors, datasets, and K.
- Multi-run repetitions with warmup queries for statistically valid latency metrics.
- Disentangled measurement of index construction vs query time.
- Memory tracking during index building.
- Parameter sensitivity sweeps for M, ef_construction, and ef_search.
- Dataset size scaling (N) sweeps.
- Real-world document corpus benchmarking using DocumentStore.
"""

from dataclasses import asdict, dataclass, field
import datetime
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.baseline import ExactKNNBaseline
from benchmark.metrics import (
    LatencyStats,
    SystemProfiler,
    compute_latency_stats,
    compute_recall_at_k,
)
from benchmark.synthetic_data import generate_synthetic_vectors, split_base_and_query
from docproc.store import DocumentStore
from docproc.loaders import load_from_csv
from hnsw.index import HNSWIndex


@dataclass
class BenchmarkConfig:
    """Configuration parameters for a benchmark experiment."""
    dim: int = 64
    metric: str = "cosine"
    M: int = 16
    M0: Optional[int] = None
    ef_construction: int = 64
    ef_search: int = 32
    k: int = 5
    num_queries: int = 25
    num_runs: int = 3
    warmup_runs: int = 1
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Structured metrics collected from an experiment."""
    experiment_name: str
    dataset_size: int
    dim: int
    metric: str
    M: int
    ef_construction: int
    ef_search: int
    k: int
    num_queries: int
    num_runs: int

    # HNSW Index Building Metrics
    hnsw_build_time_sec: float
    hnsw_build_throughput_docs_sec: float
    hnsw_peak_memory_mb: float
    hnsw_net_memory_mb: float

    # Exact Baseline Building Metrics
    bf_build_time_sec: float

    # Query Performance Metrics
    hnsw_latencies: LatencyStats
    bf_latencies: LatencyStats
    recall_at_k: float

    # Diagnostic & Complexity Metrics
    hnsw_avg_distance_comps_per_query: float
    bf_avg_distance_comps_per_query: float
    distance_comp_reduction_ratio: float
    speedup_ratio: float

    # System Info & Metadata
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for tabular export and serialization."""
        d = {
            "experiment_name": self.experiment_name,
            "dataset_size": self.dataset_size,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "k": self.k,
            "num_queries": self.num_queries,
            "num_runs": self.num_runs,
            "hnsw_build_time_sec": round(self.hnsw_build_time_sec, 4),
            "hnsw_build_throughput_docs_sec": round(self.hnsw_build_throughput_docs_sec, 2),
            "hnsw_peak_memory_mb": self.hnsw_peak_memory_mb,
            "hnsw_net_memory_mb": self.hnsw_net_memory_mb,
            "bf_build_time_sec": round(self.bf_build_time_sec, 4),
            "hnsw_mean_latency_ms": self.hnsw_latencies.mean_ms,
            "hnsw_median_latency_ms": self.hnsw_latencies.median_ms,
            "hnsw_p95_latency_ms": self.hnsw_latencies.p95_ms,
            "hnsw_qps": self.hnsw_latencies.qps,
            "bf_mean_latency_ms": self.bf_latencies.mean_ms,
            "bf_median_latency_ms": self.bf_latencies.median_ms,
            "bf_p95_latency_ms": self.bf_latencies.p95_ms,
            "bf_qps": self.bf_latencies.qps,
            "recall_at_k": round(self.recall_at_k, 4),
            "hnsw_dist_comps_per_query": round(self.hnsw_avg_distance_comps_per_query, 1),
            "bf_dist_comps_per_query": round(self.bf_avg_distance_comps_per_query, 1),
            "dist_comp_reduction": round(self.distance_comp_reduction_ratio, 2),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "timestamp": self.timestamp,
        }
        return d


class BenchmarkSuite:
    """Execution coordinator for empirical ANN evaluation."""

    def __init__(self, system_info: Optional[Dict[str, Any]] = None) -> None:
        self.system_info = system_info or SystemProfiler.get_system_info()

    def run_single_comparison(
        self,
        base_vectors: Sequence[Sequence[float]],
        query_vectors: Sequence[Sequence[float]],
        config: BenchmarkConfig,
        experiment_name: str = "single_run",
    ) -> BenchmarkResult:
        """
        Execute a controlled, head-to-head comparison between HNSW and Exact KNN.
        
        Guarantees:
        - Identical vectors and IDs for both indices.
        - Identical queries evaluated under the same conditions.
        - Independent timing for build vs search.
        - Multiple run repetitions for latency stability.
        
        Args:
            base_vectors: Vectors to index.
            query_vectors: Vectors to query against index.
            config: Benchmark experiment configuration.
            experiment_name: Descriptive name for the experiment.
            
        Returns:
            BenchmarkResult with comprehensive timing, recall, and complexity metrics.
        """
        n_docs = len(base_vectors)
        n_queries = len(query_vectors)

        if n_docs == 0:
            raise ValueError("base_vectors cannot be empty.")
        if n_queries == 0:
            raise ValueError("query_vectors cannot be empty.")

        # =====================================================================
        # 1. Build Exact Brute-Force Baseline
        # =====================================================================
        bf_index = ExactKNNBaseline(dim=config.dim, metric=config.metric)
        bf_start = time.perf_counter()
        for idx, vec in enumerate(base_vectors):
            bf_index.add_item(idx, vec)
        bf_build_time = time.perf_counter() - bf_start

        # =====================================================================
        # 2. Build HNSW Index with Memory Profiling
        # =====================================================================
        hnsw_index = HNSWIndex(
            dim=config.dim,
            metric=config.metric,
            M=config.M,
            M0=config.M0,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
            seed=config.seed,
        )

        with SystemProfiler.track_memory() as mem_metrics:
            hnsw_start = time.perf_counter()
            for idx, vec in enumerate(base_vectors):
                hnsw_index.add_item(idx, vec)
            hnsw_build_time = time.perf_counter() - hnsw_start

        build_throughput = n_docs / hnsw_build_time if hnsw_build_time > 0 else float("inf")

        # =====================================================================
        # 3. Query Exact Baseline (Ground Truth Generation & Timing)
        # =====================================================================
        # Warmup
        for _ in range(config.warmup_runs):
            for q_vec in query_vectors[: min(3, n_queries)]:
                bf_index.knn_search(q_vec, k=config.k)

        # Timed runs
        bf_latencies_ms: List[float] = []
        ground_truth_map: Dict[int, List[Tuple[Any, float]]] = {}
        bf_index.reset_distance_computations()

        for run in range(config.num_runs):
            for q_idx, q_vec in enumerate(query_vectors):
                t0 = time.perf_counter()
                results = bf_index.knn_search(q_vec, k=config.k)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                bf_latencies_ms.append(lat_ms)
                if run == 0:
                    ground_truth_map[q_idx] = results

        bf_total_comps = bf_index.total_distance_computations
        bf_avg_comps = bf_total_comps / float(config.num_runs * n_queries) if (config.num_runs * n_queries) > 0 else 0.0
        bf_stats = compute_latency_stats(bf_latencies_ms)

        # =====================================================================
        # 4. Query HNSW Index (ANN Timing & Accuracy Evaluation)
        # =====================================================================
        # Warmup
        for _ in range(config.warmup_runs):
            for q_vec in query_vectors[: min(3, n_queries)]:
                hnsw_index.knn_search(q_vec, k=config.k, ef_search=config.ef_search)

        hnsw_latencies_ms: List[float] = []
        recalls: List[float] = []
        hnsw_index.reset_distance_computations()

        for run in range(config.num_runs):
            for q_idx, q_vec in enumerate(query_vectors):
                t0 = time.perf_counter()
                hnsw_results = hnsw_index.knn_search(q_vec, k=config.k, ef_search=config.ef_search)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                hnsw_latencies_ms.append(lat_ms)

                if run == 0:
                    gt_results = ground_truth_map[q_idx]
                    recall = compute_recall_at_k(hnsw_results, gt_results, k=config.k)
                    recalls.append(recall)

        hnsw_total_comps = hnsw_index.total_distance_computations
        hnsw_avg_comps = hnsw_total_comps / float(config.num_runs * n_queries) if (config.num_runs * n_queries) > 0 else 0.0
        hnsw_stats = compute_latency_stats(hnsw_latencies_ms)
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

        # Efficiency ratios
        dist_reduction = (bf_avg_comps / hnsw_avg_comps) if hnsw_avg_comps > 0 else 1.0
        speedup = (bf_stats.mean_ms / hnsw_stats.mean_ms) if hnsw_stats.mean_ms > 0 else 1.0

        return BenchmarkResult(
            experiment_name=experiment_name,
            dataset_size=n_docs,
            dim=config.dim,
            metric=config.metric,
            M=config.M,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
            k=config.k,
            num_queries=n_queries,
            num_runs=config.num_runs,
            hnsw_build_time_sec=hnsw_build_time,
            hnsw_build_throughput_docs_sec=build_throughput,
            hnsw_peak_memory_mb=mem_metrics["peak_memory_mb"],
            hnsw_net_memory_mb=mem_metrics["net_memory_mb"],
            bf_build_time_sec=bf_build_time,
            hnsw_latencies=hnsw_stats,
            bf_latencies=bf_stats,
            recall_at_k=mean_recall,
            hnsw_avg_distance_comps_per_query=hnsw_avg_comps,
            bf_avg_distance_comps_per_query=bf_avg_comps,
            distance_comp_reduction_ratio=dist_reduction,
            speedup_ratio=speedup,
            system_info=self.system_info,
        )

    def run_dataset_scaling_benchmark(
        self,
        sizes: Sequence[int],
        config: Optional[BenchmarkConfig] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate index build and query performance across increasing dataset sizes N.
        
        Demonstrates:
        - Build time scaling O(N * M * log N)
        - HNSW query time scaling O(log N) vs Exact linear scan O(N)
        - Distance calculation count scaling
        """
        cfg = config or BenchmarkConfig()
        max_size = max(sizes)
        total_needed = max_size + cfg.num_queries

        all_vecs = generate_synthetic_vectors(
            num_vectors=total_needed,
            dim=cfg.dim,
            distribution="uniform_sphere",
            seed=cfg.seed,
        )
        _, query_vectors = split_base_and_query(all_vecs, cfg.num_queries)

        results: List[BenchmarkResult] = []
        for n in sizes:
            base_vecs = all_vecs[:n]
            res = self.run_single_comparison(
                base_vectors=base_vecs,
                query_vectors=query_vectors,
                config=cfg,
                experiment_name=f"scale_N_{n}",
            )
            results.append(res)
        return results

    def run_m_parameter_sweep(
        self,
        m_values: Sequence[int],
        base_vectors: Sequence[Sequence[float]],
        query_vectors: Sequence[Sequence[float]],
        base_config: Optional[BenchmarkConfig] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate effect of maximum connections per layer (M).
        
        Trade-offs:
        - Higher M increases graph density, build time, and memory.
        - Higher M improves graph navigability and Recall@K.
        """
        cfg = base_config or BenchmarkConfig()
        results: List[BenchmarkResult] = []
        for m in m_values:
            current_cfg = BenchmarkConfig(
                dim=cfg.dim,
                metric=cfg.metric,
                M=m,
                ef_construction=cfg.ef_construction,
                ef_search=cfg.ef_search,
                k=cfg.k,
                num_queries=cfg.num_queries,
                num_runs=cfg.num_runs,
                seed=cfg.seed,
            )
            res = self.run_single_comparison(
                base_vectors=base_vectors,
                query_vectors=query_vectors,
                config=current_cfg,
                experiment_name=f"sweep_M_{m}",
            )
            results.append(res)
        return results

    def run_ef_construction_sweep(
        self,
        ef_c_values: Sequence[int],
        base_vectors: Sequence[Sequence[float]],
        query_vectors: Sequence[Sequence[float]],
        base_config: Optional[BenchmarkConfig] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate effect of ef_construction beam size during index building.
        
        Trade-offs:
        - Higher ef_construction increases index build time.
        - Higher ef_construction creates higher-quality graph edges, boosting recall.
        """
        cfg = base_config or BenchmarkConfig()
        results: List[BenchmarkResult] = []
        for ef_c in ef_c_values:
            current_cfg = BenchmarkConfig(
                dim=cfg.dim,
                metric=cfg.metric,
                M=cfg.M,
                ef_construction=ef_c,
                ef_search=cfg.ef_search,
                k=cfg.k,
                num_queries=cfg.num_queries,
                num_runs=cfg.num_runs,
                seed=cfg.seed,
            )
            res = self.run_single_comparison(
                base_vectors=base_vectors,
                query_vectors=query_vectors,
                config=current_cfg,
                experiment_name=f"sweep_efConst_{ef_c}",
            )
            results.append(res)
        return results

    def run_ef_search_sweep(
        self,
        ef_s_values: Sequence[int],
        base_vectors: Sequence[Sequence[float]],
        query_vectors: Sequence[Sequence[float]],
        base_config: Optional[BenchmarkConfig] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate effect of ef_search beam size during query time.
        
        Trade-offs:
        - Higher ef_search explores more candidate nodes in Layer 0.
        - Increases query latency (lower QPS) while monotonically approaching 100% Recall.
        - Traces the classic Accuracy vs Speed Pareto curve.
        """
        cfg = base_config or BenchmarkConfig()
        results: List[BenchmarkResult] = []
        for ef_s in ef_s_values:
            current_cfg = BenchmarkConfig(
                dim=cfg.dim,
                metric=cfg.metric,
                M=cfg.M,
                ef_construction=cfg.ef_construction,
                ef_search=ef_s,
                k=cfg.k,
                num_queries=cfg.num_queries,
                num_runs=cfg.num_runs,
                seed=cfg.seed,
            )
            res = self.run_single_comparison(
                base_vectors=base_vectors,
                query_vectors=query_vectors,
                config=current_cfg,
                experiment_name=f"sweep_efSearch_{ef_s}",
            )
            results.append(res)
        return results

    def run_corpus_benchmark(
        self,
        csv_filepath: str,
        dim: int = 64,
        k: int = 5,
        num_query_docs: int = 5,
        metric: str = "cosine",
    ) -> BenchmarkResult:
        """
        Benchmark HNSW directly on real text documents ingested via DocumentStore.
        
        Demonstrates cross-module integration with Teammate 2's pipeline.
        """
        docs = load_from_csv(csv_filepath)
        if len(docs) <= num_query_docs:
            raise ValueError(f"Corpus size ({len(docs)}) must exceed num_query_docs ({num_query_docs}).")

        store = DocumentStore(dim=dim)
        store.add_documents(docs)

        all_doc_vectors = [doc.vector for doc in store.all_documents()]
        base_vectors = all_doc_vectors[:-num_query_docs]
        query_vectors = all_doc_vectors[-num_query_docs:]

        config = BenchmarkConfig(
            dim=dim,
            metric=metric,
            M=8,
            ef_construction=64,
            ef_search=32,
            k=k,
            num_queries=num_query_docs,
            num_runs=3,
        )

        return self.run_single_comparison(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            config=config,
            experiment_name="real_corpus_sample_csv",
        )
