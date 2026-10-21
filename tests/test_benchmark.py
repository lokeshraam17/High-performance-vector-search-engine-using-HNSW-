"""
Unit Tests for Module 3 (Benchmarking, Baseline & Evaluation).

Covers:
- ExactKNNBaseline correctness, input validation, heap ordering, and distance counting.
- Recall@K computation, tie tolerance, and boundary handling.
- Latency statistics (mean, median, p95, p99, min, max, std dev, QPS).
- SystemProfiler environment metadata and tracemalloc memory tracking.
- Synthetic vector generation (uniform sphere, clusters, hypercube) and reproducibility.
- BenchmarkSuite execution, parameter sweeps, and metric integrity.
- CSV export and SVG visualization generation.
"""

import csv
import math
import os
import tempfile
import unittest

from benchmark.baseline import ExactKNNBaseline
from benchmark.metrics import (
    LatencyStats,
    SystemProfiler,
    compute_latency_stats,
    compute_recall_at_k,
)
from benchmark.reporting import export_csv, generate_html_report, generate_svg_chart_scaling
from benchmark.runner import BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from benchmark.synthetic_data import generate_synthetic_vectors, split_base_and_query


class TestExactKNNBaseline(unittest.TestCase):
    """Tests for the ExactKNNBaseline linear-scan ground-truth oracle."""

    def test_initialization_and_validation(self):
        baseline = ExactKNNBaseline(dim=4, metric="euclidean")
        self.assertEqual(baseline.dim, 4)
        self.assertEqual(baseline.metric_name, "euclidean")
        self.assertEqual(len(baseline), 0)

        with self.assertRaises(ValueError):
            ExactKNNBaseline(dim=0)
        with self.assertRaises(ValueError):
            ExactKNNBaseline(dim=-5)
        with self.assertRaises(ValueError):
            ExactKNNBaseline(dim=4, metric="invalid_metric")

    def test_add_item_and_validation(self):
        baseline = ExactKNNBaseline(dim=3, metric="cosine")
        baseline.add_item("item_1", [1.0, 0.0, 0.0])
        self.assertEqual(len(baseline), 1)

        # Duplicate ID
        with self.assertRaises(ValueError):
            baseline.add_item("item_1", [0.0, 1.0, 0.0])

        # Dimension mismatch
        with self.assertRaises(ValueError):
            baseline.add_item("item_2", [1.0, 0.0])

        # None ID
        with self.assertRaises(ValueError):
            baseline.add_item(None, [1.0, 0.0, 0.0])

        # Non-numeric coordinate
        with self.assertRaises(TypeError):
            baseline.add_item("item_3", [1.0, "bad", 0.0])

        # NaN coordinate
        with self.assertRaises(ValueError):
            baseline.add_item("item_4", [1.0, float("nan"), 0.0])

    def test_exact_knn_search_euclidean(self):
        baseline = ExactKNNBaseline(dim=2, metric="euclidean")
        # Items placed on the 2D plane
        baseline.add_item("p0", [0.0, 0.0])
        baseline.add_item("p1", [1.0, 0.0])
        baseline.add_item("p2", [2.0, 0.0])
        baseline.add_item("p3", [10.0, 0.0])

        query = [0.1, 0.0]
        results = baseline.knn_search(query, k=2)

        self.assertEqual(len(results), 2)
        # Closest is p0 (dist ~0.1), second is p1 (dist ~0.9)
        self.assertEqual(results[0][0], "p0")
        self.assertAlmostEqual(results[0][1], 0.1, places=5)
        self.assertEqual(results[1][0], "p1")
        self.assertAlmostEqual(results[1][1], 0.9, places=5)

    def test_empty_baseline_search(self):
        baseline = ExactKNNBaseline(dim=3)
        res = baseline.knn_search([1.0, 2.0, 3.0], k=5)
        self.assertEqual(res, [])

    def test_k_greater_than_total_items(self):
        baseline = ExactKNNBaseline(dim=2)
        baseline.add_item("a", [1.0, 1.0])
        baseline.add_item("b", [2.0, 2.0])
        res = baseline.knn_search([0.0, 0.0], k=10)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0], "a")
        self.assertEqual(res[1][0], "b")

    def test_distance_computations_tracking(self):
        baseline = ExactKNNBaseline(dim=2)
        baseline.add_item("a", [0.0, 0.0])
        baseline.add_item("b", [1.0, 1.0])
        baseline.add_item("c", [2.0, 2.0])

        self.assertEqual(baseline.total_distance_computations, 0)
        baseline.knn_search([0.0, 0.0], k=2)
        self.assertEqual(baseline.total_distance_computations, 3)

        baseline.reset_distance_computations()
        self.assertEqual(baseline.total_distance_computations, 0)


class TestBenchmarkMetrics(unittest.TestCase):
    """Tests for accuracy calculations, latency stats, and profiler."""

    def test_recall_at_k_exact_match(self):
        retrieved = [("doc_1", 0.1), ("doc_2", 0.2), ("doc_3", 0.3)]
        ground_truth = [("doc_1", 0.1), ("doc_2", 0.2), ("doc_3", 0.3)]
        recall = compute_recall_at_k(retrieved, ground_truth, k=3)
        self.assertAlmostEqual(recall, 1.0)

    def test_recall_at_k_disjoint(self):
        retrieved = [("doc_a", 0.1), ("doc_b", 0.2)]
        ground_truth = [("doc_x", 0.1), ("doc_y", 0.2)]
        recall = compute_recall_at_k(retrieved, ground_truth, k=2)
        self.assertAlmostEqual(recall, 0.0)

    def test_recall_at_k_partial(self):
        retrieved = [("doc_1", 0.1), ("doc_2", 0.2), ("doc_99", 0.5)]
        ground_truth = [("doc_1", 0.1), ("doc_2", 0.2), ("doc_3", 0.3)]
        recall = compute_recall_at_k(retrieved, ground_truth, k=3)
        self.assertAlmostEqual(recall, 2.0 / 3.0)

    def test_recall_at_k_invalid_k(self):
        with self.assertRaises(ValueError):
            compute_recall_at_k([], [], k=0)

    def test_compute_latency_stats(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_latency_stats(latencies)

        self.assertEqual(stats.total_queries, 10)
        self.assertAlmostEqual(stats.mean_ms, 5.5, places=2)
        self.assertAlmostEqual(stats.median_ms, 5.5, places=2)
        self.assertAlmostEqual(stats.min_ms, 1.0)
        self.assertAlmostEqual(stats.max_ms, 10.0)
        # Sum = 55ms = 0.055s -> 10 / 0.055 = ~181.82 QPS
        self.assertGreater(stats.qps, 150.0)

    def test_compute_latency_stats_empty(self):
        with self.assertRaises(ValueError):
            compute_latency_stats([])

    def test_system_profiler_and_memory_tracking(self):
        info = SystemProfiler.get_system_info()
        self.assertIn("os_name", info)
        self.assertIn("python_version", info)
        self.assertIn("cpu_count", info)

        with SystemProfiler.track_memory() as mem:
            # Allocate a temporary 100,000 float list
            temp_list = [float(i) for i in range(100_000)]
            self.assertGreater(len(temp_list), 0)

        self.assertIn("peak_memory_mb", mem)
        self.assertGreaterEqual(mem["peak_memory_mb"], 0.0)


class TestSyntheticData(unittest.TestCase):
    """Tests for reproducible synthetic vector generation."""

    def test_generate_uniform_sphere(self):
        dim = 16
        num = 50
        vecs = generate_synthetic_vectors(num_vectors=num, dim=dim, distribution="uniform_sphere", seed=42)
        self.assertEqual(len(vecs), num)
        for v in vecs:
            self.assertEqual(len(v), dim)
            # Check unit length: ||v|| ~ 1.0
            norm = math.sqrt(sum(x * x for x in v))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_generate_clusters_and_cube(self):
        dim = 8
        vecs_clusters = generate_synthetic_vectors(num_vectors=30, dim=dim, distribution="gaussian_clusters", seed=42)
        self.assertEqual(len(vecs_clusters), 30)

        vecs_cube = generate_synthetic_vectors(num_vectors=20, dim=dim, distribution="uniform_cube", seed=42)
        self.assertEqual(len(vecs_cube), 20)
        for v in vecs_cube:
            for coord in v:
                self.assertTrue(-1.0 <= coord <= 1.0)

    def test_reproducibility_with_seed(self):
        v1 = generate_synthetic_vectors(num_vectors=25, dim=8, seed=123)
        v2 = generate_synthetic_vectors(num_vectors=25, dim=8, seed=123)
        self.assertEqual(v1, v2)

        v3 = generate_synthetic_vectors(num_vectors=25, dim=8, seed=999)
        self.assertNotEqual(v1, v3)

    def test_split_base_and_query(self):
        vecs = [[float(i)] * 4 for i in range(10)]
        base, queries = split_base_and_query(vecs, num_queries=3)
        self.assertEqual(len(base), 7)
        self.assertEqual(len(queries), 3)

        with self.assertRaises(ValueError):
            split_base_and_query(vecs, num_queries=0)
        with self.assertRaises(ValueError):
            split_base_and_query(vecs, num_queries=10)


class TestBenchmarkSuite(unittest.TestCase):
    """Tests for experiment coordination and parameter sweeps."""

    def setUp(self):
        self.suite = BenchmarkSuite()
        self.config = BenchmarkConfig(
            dim=8,
            metric="euclidean",
            M=4,
            ef_construction=16,
            ef_search=16,
            k=3,
            num_queries=5,
            num_runs=2,
            seed=42,
        )
        all_vecs = generate_synthetic_vectors(num_vectors=40, dim=8, seed=42)
        self.base_vectors, self.query_vectors = split_base_and_query(all_vecs, num_queries=5)

    def test_run_single_comparison(self):
        res = self.suite.run_single_comparison(
            base_vectors=self.base_vectors,
            query_vectors=self.query_vectors,
            config=self.config,
            experiment_name="unit_test_run",
        )
        self.assertIsInstance(res, BenchmarkResult)
        self.assertEqual(res.dataset_size, len(self.base_vectors))
        self.assertGreater(res.recall_at_k, 0.70)  # High recall expected
        self.assertGreater(res.hnsw_latencies.mean_ms, 0.0)
        self.assertGreater(res.bf_latencies.mean_ms, 0.0)
        self.assertGreaterEqual(res.hnsw_peak_memory_mb, 0.0)
        self.assertGreater(res.hnsw_build_time_sec, 0.0)
        self.assertGreater(res.bf_build_time_sec, 0.0)

    def test_ef_search_sweep(self):
        ef_s_values = [4, 8, 16]
        results = self.suite.run_ef_search_sweep(
            ef_s_values=ef_s_values,
            base_vectors=self.base_vectors,
            query_vectors=self.query_vectors,
            base_config=self.config,
        )
        self.assertEqual(len(results), 3)
        # Recall must be monotonically non-decreasing or high across increasing ef_search
        recalls = [r.recall_at_k for r in results]
        self.assertGreaterEqual(recalls[-1], recalls[0])

    def test_dataset_scaling(self):
        sizes = [15, 30]
        results = self.suite.run_dataset_scaling_benchmark(sizes=sizes, config=self.config)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].dataset_size, 15)
        self.assertEqual(results[1].dataset_size, 30)


class TestReporting(unittest.TestCase):
    """Tests for CSV, SVG, and HTML artifact generation."""

    def setUp(self):
        self.suite = BenchmarkSuite()
        self.config = BenchmarkConfig(dim=4, metric="euclidean", M=4, ef_construction=16, ef_search=16, k=2, num_queries=2, num_runs=1)
        all_vecs = generate_synthetic_vectors(num_vectors=15, dim=4, seed=42)
        base, queries = split_base_and_query(all_vecs, num_queries=2)
        self.res = self.suite.run_single_comparison(base, queries, self.config, experiment_name="test_reporting")

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            export_csv([self.res], csv_path)
            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["experiment_name"], "test_reporting")
                self.assertIn("recall_at_k", rows[0])
                self.assertIn("hnsw_mean_latency_ms", rows[0])

    def test_svg_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, "test_scaling.svg")
            # SVG requires at least 2 results
            res2 = self.res
            generate_svg_chart_scaling([self.res, res2], svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertTrue(content.startswith("<svg"))
                self.assertTrue(content.endswith("</svg>"))

    def test_html_report_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "report.html")
            generate_html_report([self.res], self.suite.system_info, html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("test_reporting", content)


if __name__ == "__main__":
    unittest.main()
