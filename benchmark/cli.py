"""
Command-Line Interface (CLI) for HNSW Benchmarking & Performance Evaluation.

Usage Examples:
    # 1. Quick reproducible evaluation (< 3 seconds)
    python -m benchmark.cli --quick

    # 2. Comprehensive benchmark with N-scaling and parameter sweeps
    python -m benchmark.cli --full

    # 3. Benchmark real document corpus
    python -m benchmark.cli --corpus data/sample_documents.csv

    # 4. Custom parameter run
    python -m benchmark.cli --n 500 --dim 64 --k 10 --metric cosine --m 16 --ef-c 100 --ef-s 50
"""

import argparse
import os
import sys
from typing import List

from benchmark.metrics import SystemProfiler
from benchmark.reporting import (
    export_csv,
    generate_html_report,
    generate_png_charts,
    generate_svg_chart_scaling,
    print_terminal_table,
)
from benchmark.runner import BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from benchmark.synthetic_data import generate_synthetic_vectors, split_base_and_query


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with helpful defaults and descriptions."""
    parser = argparse.ArgumentParser(
        description="HNSW Vector Search Engine Performance & Accuracy Benchmarking Suite"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick reproducible benchmark (< 3 seconds).",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run a comprehensive benchmark (dataset size scaling + parameter sweeps).",
    )
    mode_group.add_argument(
        "--corpus",
        type=str,
        help="Path to CSV document file to benchmark with DocumentStore.",
    )

    # Custom run parameters
    parser.add_argument("--n", type=int, default=300, help="Number of base vectors (default: 300).")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension (default: 64).")
    parser.add_argument("--k", type=int, default=5, help="Top-K nearest neighbors (default: 5).")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot_product"],
        help="Distance metric (default: cosine).",
    )
    parser.add_argument("--m", type=int, default=16, help="HNSW M parameter (default: 16).")
    parser.add_argument("--ef-c", type=int, default=64, help="HNSW efConstruction (default: 64).")
    parser.add_argument("--ef-s", type=int, default=32, help="HNSW efSearch (default: 32).")
    parser.add_argument("--queries", type=int, default=20, help="Number of query vectors (default: 20).")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions for latency timing (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save CSV reports, HTML dashboard, and charts (default: results).",
    )
    return parser


def run_quick_benchmark(suite: BenchmarkSuite, output_dir: str) -> List[BenchmarkResult]:
    """Execute quick reproducible benchmark (< 3 seconds)."""
    print("\n>>> Running Quick Reproducible Benchmark (N=200, dim=32, k=5)...")
    config = BenchmarkConfig(
        dim=32,
        metric="cosine",
        M=12,
        ef_construction=48,
        ef_search=24,
        k=5,
        num_queries=15,
        num_runs=3,
        seed=42,
    )

    all_vecs = generate_synthetic_vectors(
        num_vectors=215,
        dim=config.dim,
        distribution="uniform_sphere",
        seed=config.seed,
    )
    base_vecs, q_vecs = split_base_and_query(all_vecs, config.num_queries)

    res = suite.run_single_comparison(
        base_vectors=base_vecs,
        query_vectors=q_vecs,
        config=config,
        experiment_name="quick_demo_N200",
    )

    results = [res]
    print_terminal_table(results, title="QUICK BENCHMARK RESULTS")

    # Export artifacts
    csv_path = os.path.join(output_dir, "benchmark_quick.csv")
    export_csv(results, csv_path)
    print(f"[*] CSV saved: {csv_path}")
    return results


def run_full_benchmark(suite: BenchmarkSuite, output_dir: str) -> List[BenchmarkResult]:
    """Execute comprehensive benchmarking including dataset scaling and parameter sweeps."""
    print("\n==================================================================")
    print(" HNSW SEARCH ENGINE COMPREHENSIVE BENCHMARKING SUITE")
    print("==================================================================")

    all_results: List[BenchmarkResult] = []

    # 1. Dataset Size Scaling Experiment
    print("\n[Phase 1/4] Running Dataset Size (N) Scaling Benchmark...")
    scale_sizes = [50, 100, 200, 400, 800]
    scale_cfg = BenchmarkConfig(dim=64, metric="cosine", M=16, ef_construction=64, ef_search=32, k=5, num_queries=20, num_runs=3, seed=42)
    scaling_results = suite.run_dataset_scaling_benchmark(sizes=scale_sizes, config=scale_cfg)
    all_results.extend(scaling_results)

    # 2. efSearch Parameter Sweep (Accuracy vs Speed Pareto Curve)
    print("\n[Phase 2/4] Running efSearch Sweep (Accuracy vs Speed)...")
    base_vecs = generate_synthetic_vectors(num_vectors=400, dim=64, distribution="uniform_sphere", seed=101)
    q_vecs = generate_synthetic_vectors(num_vectors=25, dim=64, distribution="uniform_sphere", seed=202)
    ef_s_values = [5, 10, 20, 40, 80, 160]
    ef_search_results = suite.run_ef_search_sweep(ef_s_values=ef_s_values, base_vectors=base_vecs, query_vectors=q_vecs, base_config=scale_cfg)
    all_results.extend(ef_search_results)

    # 3. efConstruction Parameter Sweep
    print("\n[Phase 3/4] Running efConstruction Sweep (Build Time vs Graph Quality)...")
    ef_c_values = [16, 32, 64, 128]
    ef_c_results = suite.run_ef_construction_sweep(ef_c_values=ef_c_values, base_vectors=base_vecs, query_vectors=q_vecs, base_config=scale_cfg)
    all_results.extend(ef_c_results)

    # 4. M Parameter Sweep
    print("\n[Phase 4/4] Running M Sweep (Graph Connectivity & Degree)...")
    m_values = [4, 8, 16, 24]
    m_results = suite.run_m_parameter_sweep(m_values=m_values, base_vectors=base_vecs, query_vectors=q_vecs, base_config=scale_cfg)
    all_results.extend(m_results)

    # Display console tables
    print_terminal_table(scaling_results, title="1. DATASET SIZE SCALING (N) RESULTS")
    print_terminal_table(ef_search_results, title="2. efSearch PARAMETER SWEEP (PARETO TRADEOFF)")
    print_terminal_table(ef_c_results, title="3. efConstruction PARAMETER SWEEP")
    print_terminal_table(m_results, title="4. M (CONNECTIVITY) PARAMETER SWEEP")

    # Export CSV
    summary_csv = os.path.join(output_dir, "benchmark_summary.csv")
    export_csv(all_results, summary_csv)
    print(f"[*] Summary CSV saved: {summary_csv}")

    # Generate Charts
    charts = generate_png_charts(
        scaling_results=scaling_results,
        ef_search_results=ef_search_results,
        ef_const_results=ef_c_results,
        output_dir=output_dir,
    )
    for c in charts:
        print(f"[*] Generated Chart: {c}")

    # Generate SVG Scaling Chart
    svg_path = os.path.join(output_dir, "chart_latency_vs_dataset_size.svg")
    generate_svg_chart_scaling(scaling_results, svg_path)
    print(f"[*] Generated Vector SVG Chart: {svg_path}")

    # Generate HTML Dashboard
    html_path = os.path.join(output_dir, "benchmark_report.html")
    generate_html_report(all_results, suite.system_info, html_path, chart_filenames=charts)
    print(f"[*] Generated Interactive HTML Report: {html_path}")

    return all_results


def run_custom_benchmark(args: argparse.Namespace, suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Execute a benchmark configured via custom CLI flags."""
    print(f"\n>>> Running Custom Benchmark: N={args.n}, dim={args.dim}, M={args.m}, efC={args.ef_c}, efS={args.ef_s}, metric={args.metric}...")
    config = BenchmarkConfig(
        dim=args.dim,
        metric=args.metric,
        M=args.m,
        ef_construction=args.ef_c,
        ef_search=args.ef_s,
        k=args.k,
        num_queries=args.queries,
        num_runs=args.runs,
        seed=args.seed,
    )

    all_vecs = generate_synthetic_vectors(
        num_vectors=args.n + args.queries,
        dim=args.dim,
        distribution="uniform_sphere",
        seed=args.seed,
    )
    base_vecs, q_vecs = split_base_and_query(all_vecs, args.queries)

    res = suite.run_single_comparison(
        base_vectors=base_vecs,
        query_vectors=q_vecs,
        config=config,
        experiment_name=f"custom_N{args.n}_M{args.m}",
    )

    results = [res]
    print_terminal_table(results, title="CUSTOM BENCHMARK RESULTS")

    csv_path = os.path.join(args.output_dir, "benchmark_custom.csv")
    export_csv(results, csv_path)
    print(f"[*] CSV saved: {csv_path}")
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suite = BenchmarkSuite()

    if args.quick:
        run_quick_benchmark(suite, args.output_dir)
    elif args.full:
        run_full_benchmark(suite, args.output_dir)
    elif args.corpus:
        print(f"\n>>> Running Document Corpus Benchmark on '{args.corpus}'...")
        res = suite.run_corpus_benchmark(csv_filepath=args.corpus, dim=args.dim, k=args.k)
        results = [res]
        print_terminal_table(results, title="CORPUS BENCHMARK RESULTS")
        csv_path = os.path.join(args.output_dir, "benchmark_corpus.csv")
        export_csv(results, csv_path)
        print(f"[*] CSV saved: {csv_path}")
    else:
        # Default to quick if no flag specified
        run_quick_benchmark(suite, args.output_dir)


if __name__ == "__main__":
    main()
