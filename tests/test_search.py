"""
Unit and End-to-End Integration Tests for Module 4 (Search Interface & Integration).

Tests:
1. SearchEngine initialization and parameter validation.
2. Unindexed engine query handling and error raising.
3. Ingestion of CSV dataset and HNSW + baseline construction.
4. Approximate nearest-neighbor search with metadata retrieval.
5. Ground-truth baseline comparison and Recall@K accuracy verification.
6. Edge case validation: empty queries, whitespace, invalid Top-K, missing files.
7. Web API server request handling (status, search, comparison, and static HTML).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

from search.engine import ComparisonResponse, SearchEngine, SearchResponse, SearchResultItem
from search.web import create_web_server, SearchHTTPRequestHandler
from search.cli import format_comparison_results, format_search_results


class TestSearchEngine(unittest.TestCase):
    """Test suite for the core SearchEngine orchestrator."""

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.csv_path = os.path.join(cls.repo_dir, "data", "sample_documents.csv")

    def setUp(self):
        self.engine = SearchEngine(
            dim=64,
            metric="cosine",
            M=16,
            ef_construction=64,
            ef_search=32,
            seed=42,
        )

    def test_initialization(self):
        """Verify engine initializes with expected configuration and empty state."""
        self.assertEqual(self.engine.dim, 64)
        self.assertEqual(self.engine.metric, "cosine")
        self.assertEqual(self.engine.total_documents, 0)
        self.assertFalse(self.engine.is_indexed)

        # Invalid initialization parameters
        with self.assertRaises(ValueError):
            SearchEngine(dim=-1)
        with self.assertRaises(ValueError):
            SearchEngine(M=1)

    def test_unindexed_search_raises_error(self):
        """Querying an unindexed engine must raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.search("test query", k=5)
        self.assertIn("not indexed", str(ctx.exception).lower())

        with self.assertRaises(RuntimeError):
            self.engine.search_baseline("test query", k=5)

    def test_load_dataset_csv(self):
        """Loading sample_documents.csv must populate store, HNSW index, and baseline."""
        count = self.engine.load_dataset(self.csv_path)
        self.assertEqual(count, 25)
        self.assertEqual(self.engine.total_documents, 25)
        self.assertTrue(self.engine.is_indexed)
        self.assertIsNotNone(self.engine.index)
        self.assertIsNotNone(self.engine.baseline)
        self.assertEqual(len(self.engine.index.nodes), 25)
        self.assertEqual(len(self.engine.baseline.vectors), 25)

    def test_load_dataset_missing_file(self):
        """Non-existent file path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.engine.load_dataset("non_existent_file_path_xyz.csv")

    def test_search_valid_query(self):
        """Execute valid search and verify response structure and ranking."""
        self.engine.load_dataset(self.csv_path)
        resp = self.engine.search("binary search tree left right child", k=3)

        self.assertIsInstance(resp, SearchResponse)
        self.assertEqual(resp.query, "binary search tree left right child")
        self.assertEqual(resp.k, 3)
        self.assertEqual(len(resp.results), 3)
        self.assertGreater(resp.latency_ms, 0.0)
        self.assertGreater(resp.distance_computations, 0)

        # Verify results order and fields
        first = resp.results[0]
        self.assertEqual(first.rank, 1)
        self.assertIn("tree", first.title.lower())
        self.assertIsInstance(first.snippet, str)
        self.assertGreater(len(first.snippet), 10)
        self.assertGreaterEqual(first.similarity, 0.0)
        self.assertLessEqual(first.similarity, 1.0)
        self.assertGreaterEqual(first.distance, 0.0)

        # Verify monotonicity of ranks and distances
        for i in range(len(resp.results) - 1):
            self.assertEqual(resp.results[i].rank + 1, resp.results[i + 1].rank)
            self.assertLessEqual(resp.results[i].distance, resp.results[i + 1].distance + 1e-6)

    def test_empty_and_whitespace_query(self):
        """Empty or whitespace queries must raise ValueError."""
        self.engine.load_dataset(self.csv_path)
        with self.assertRaises(ValueError):
            self.engine.search("", k=5)
        with self.assertRaises(ValueError):
            self.engine.search("   \t\n  ", k=5)
        with self.assertRaises(ValueError):
            self.engine.search_baseline("", k=5)

    def test_invalid_top_k(self):
        """k <= 0 must raise ValueError."""
        self.engine.load_dataset(self.csv_path)
        with self.assertRaises(ValueError):
            self.engine.search("algorithm", k=0)
        with self.assertRaises(ValueError):
            self.engine.search("algorithm", k=-5)

    def test_k_greater_than_total_documents(self):
        """If k > total documents, engine returns all available documents without error."""
        self.engine.load_dataset(self.csv_path)
        resp = self.engine.search("database storage", k=100)
        self.assertEqual(len(resp.results), 25)

    def test_compare_search_ground_truth(self):
        """Compare HNSW against ExactKNNBaseline oracle."""
        self.engine.load_dataset(self.csv_path)
        comp = self.engine.compare_search("dijkstra shortest path graph", k=5)

        self.assertIsInstance(comp, ComparisonResponse)
        self.assertEqual(comp.k, 5)
        self.assertGreaterEqual(comp.recall_at_k, 0.8)  # High recall >= 80%
        self.assertEqual(comp.recall_percent, comp.recall_at_k * 100.0)
        self.assertEqual(len(comp.hnsw.results), 5)
        self.assertEqual(len(comp.baseline.results), 5)
        self.assertEqual(comp.baseline.mode, "Exact Brute-Force (Ground Truth)")

        # Verify serialization to dict
        d = comp.to_dict()
        self.assertIn("recall_at_k", d)
        self.assertIn("speedup", d)
        self.assertIn("hnsw", d)
        self.assertIn("baseline", d)

    def test_ef_search_override(self):
        """ef_search parameter override affects candidate evaluation."""
        self.engine.load_dataset(self.csv_path)
        resp1 = self.engine.search("neural network", k=3, ef_search=10)
        self.assertEqual(resp1.ef_search, 10)

        resp2 = self.engine.search("neural network", k=3, ef_search=50)
        self.assertEqual(resp2.ef_search, 50)

    def test_engine_status_dictionary(self):
        """Verify engine status provides complete diagnostics."""
        self.engine.load_dataset(self.csv_path)
        status = self.engine.get_status()

        self.assertTrue(status["is_indexed"])
        self.assertEqual(status["total_documents"], 25)
        self.assertEqual(status["embedding_dimension"], 64)
        self.assertEqual(status["distance_metric"], "cosine")
        self.assertIn("hnsw_config", status)
        self.assertIn("graph_diagnostics", status)
        self.assertEqual(status["graph_diagnostics"]["total_nodes"], 25)


class TestWebAPI(unittest.TestCase):
    """End-to-end integration tests for the HTTP server and REST endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.csv_path = os.path.join(cls.repo_dir, "data", "sample_documents.csv")
        
        # Start server on an ephemeral/free port
        cls.port = 8899
        cls.host = "127.0.0.1"
        cls.server = create_web_server(
            host=cls.host,
            port=cls.port,
            dataset_path=cls.csv_path,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.3)  # brief warm-up

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_status_endpoint(self):
        """GET /api/status returns 200 and system status JSON."""
        url = f"http://{self.host}:{self.port}/api/status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            self.assertTrue(data["is_indexed"])
            self.assertEqual(data["total_documents"], 25)

    def test_get_static_html(self):
        """GET / returns HTML document."""
        url = f"http://{self.host}:{self.port}/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers.get("Content-Type"))
            body = response.read().decode("utf-8")
            self.assertIn("HNSW Vector Search Engine", body)

    def test_post_search_endpoint(self):
        """POST /api/search returns ranked search results."""
        url = f"http://{self.host}:{self.port}/api/search"
        payload = json.dumps({"query": "binary search tree", "k": 3}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            self.assertEqual(len(data["results"]), 3)
            self.assertEqual(data["results"][0]["rank"], 1)

    def test_post_search_comparison_endpoint(self):
        """POST /api/search with compare_baseline=true returns ground-truth comparison."""
        url = f"http://{self.host}:{self.port}/api/search"
        payload = json.dumps({
            "query": "virtual memory paging",
            "k": 4,
            "compare_baseline": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            self.assertIn("recall_at_k", data)
            self.assertIn("hnsw", data)
            self.assertIn("baseline", data)
            self.assertEqual(len(data["hnsw"]["results"]), 4)

    def test_post_search_empty_query_error(self):
        """POST /api/search with empty query returns 400 Bad Request."""
        url = f"http://{self.host}:{self.port}/api/search"
        payload = json.dumps({"query": "   "}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


class TestCLIRendering(unittest.TestCase):
    """Test formatting and display functions in CLI without raising exceptions."""

    def test_format_results_smoke(self):
        """format_search_results and format_comparison_results execute cleanly."""
        item = SearchResultItem(
            rank=1,
            doc_id="doc_001",
            title="Binary Search Trees",
            category="Data Structures",
            snippet="A binary search tree is a rooted binary tree data structure...",
            distance=0.45,
            similarity=0.55,
        )
        resp = SearchResponse(
            query="test",
            k=1,
            ef_search=32,
            latency_ms=0.5,
            distance_computations=10,
            total_indexed=25,
            mode="HNSW",
            results=[item],
        )
        # Verify formatting executes without raising
        format_search_results(resp)

        comp = ComparisonResponse(
            query="test",
            k=1,
            recall_at_k=1.0,
            speedup=1.2,
            distance_reduction_ratio=1.5,
            hnsw=resp,
            baseline=resp,
        )
        format_comparison_results(comp)


if __name__ == "__main__":
    unittest.main()
