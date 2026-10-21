"""
Unit tests for the HNSW Core Algorithm module.

Tests cover:
- Distance metrics (Euclidean, Cosine, Dot product) & validation.
- Node data structure integrity.
- Graph insertion and hierarchical level assignments.
- SearchLayer and K-NN approximate search vs exact brute force.
- Edge cases: empty index, invalid inputs, duplicate IDs, dimension mismatches.
- Serialization and deserialization.
"""

import math
import os
import random
import tempfile
import unittest

from hnsw import (
    HNSWIndex,
    HNSWNode,
    cosine_distance,
    dot_product_distance,
    euclidean_distance,
    validate_vector,
)


class TestDistanceMetrics(unittest.TestCase):
    """Test vector distance calculations and validation routines."""

    def test_euclidean_distance_known_values(self):
        # 3D 3-4-5 right triangle equivalent
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        self.assertAlmostEqual(euclidean_distance(v1, v2), 5.0)

        # Identical vectors have 0 distance
        self.assertAlmostEqual(euclidean_distance(v2, v2), 0.0)

    def test_cosine_distance_known_values(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        # Identical direction -> similarity = 1 -> distance = 0
        self.assertAlmostEqual(cosine_distance(v1, v2), 0.0)

        # Orthogonal vectors -> similarity = 0 -> distance = 1
        v3 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_distance(v1, v3), 1.0)

        # Opposite vectors -> similarity = -1 -> distance = 2
        v4 = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_distance(v1, v4), 2.0)

    def test_dot_product_distance(self):
        v1 = [1.0, 2.0]
        v2 = [3.0, 4.0]
        # -(1*3 + 2*4) = -11
        self.assertAlmostEqual(dot_product_distance(v1, v2), -11.0)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            cosine_distance([1.0], [1.0, 2.0])

    def test_vector_validation(self):
        # Empty vector
        with self.assertRaises(ValueError):
            validate_vector([])

        # None vector
        with self.assertRaises(ValueError):
            validate_vector(None)

        # NaN value
        with self.assertRaises(ValueError):
            validate_vector([1.0, float("nan")])

        # Inf value
        with self.assertRaises(ValueError):
            validate_vector([float("inf"), 2.0])

        # Dimension mismatch
        with self.assertRaises(ValueError):
            validate_vector([1.0, 2.0], expected_dim=3)

        # Non-numeric item
        with self.assertRaises(TypeError):
            validate_vector([1.0, "two"])


class TestHNSWNode(unittest.TestCase):
    """Test HNSWNode creation and neighbor management."""

    def test_node_initialization(self):
        node = HNSWNode(node_id="doc_1", vector=[0.1, 0.2], level=2)
        self.assertEqual(node.node_id, "doc_1")
        self.assertEqual(node.vector, [0.1, 0.2])
        self.assertEqual(node.level, 2)
        # Layers 0, 1, 2 should be initialized
        for lvl in [0, 1, 2]:
            self.assertIn(lvl, node.neighbors)
            self.assertEqual(node.get_neighbors(lvl), [])

    def test_neighbor_manipulation(self):
        node = HNSWNode(node_id="doc_1", vector=[0.5], level=1)
        node.add_neighbor(0, "doc_2")
        node.add_neighbor(0, "doc_3")
        # Duplicate should be ignored
        node.add_neighbor(0, "doc_2")

        self.assertEqual(node.get_neighbors(0), ["doc_2", "doc_3"])
        self.assertEqual(node.degree(0), 2)

        node.remove_neighbor(0, "doc_2")
        self.assertEqual(node.get_neighbors(0), ["doc_3"])
        self.assertEqual(node.degree(0), 1)


class TestHNSWIndex(unittest.TestCase):
    """Test HNSWIndex insertion, search, and edge case handling."""

    def test_invalid_index_initialization(self):
        with self.assertRaises(ValueError):
            HNSWIndex(dim=0)
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, M=1)  # M must be >= 2
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, ef_construction=0)
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, ef_search=0)

    def test_empty_index_queries(self):
        index = HNSWIndex(dim=3)
        self.assertEqual(index.knn_search([1.0, 2.0, 3.0], k=5), [])
        self.assertEqual(index.brute_force_knn([1.0, 2.0, 3.0], k=5), [])

    def test_single_item_insertion_and_search(self):
        index = HNSWIndex(dim=2, seed=42)
        index.add_item("doc_1", [1.0, 2.0])

        self.assertEqual(len(index.nodes), 1)
        self.assertEqual(index.enter_point_id, "doc_1")

        # Query with exact vector -> distance must be 0
        res = index.knn_search([1.0, 2.0], k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "doc_1")
        self.assertAlmostEqual(res[0][1], 0.0)

    def test_invalid_add_item_inputs(self):
        index = HNSWIndex(dim=3)
        index.add_item(1, [1.0, 2.0, 3.0])

        # Duplicate ID
        with self.assertRaises(ValueError):
            index.add_item(1, [4.0, 5.0, 6.0])

        # None ID
        with self.assertRaises(ValueError):
            index.add_item(None, [1.0, 2.0, 3.0])

        # Dimension mismatch
        with self.assertRaises(ValueError):
            index.add_item(2, [1.0, 2.0])

    def test_invalid_knn_search_inputs(self):
        index = HNSWIndex(dim=2)
        index.add_item(1, [1.0, 1.0])

        # Invalid k <= 0
        with self.assertRaises(ValueError):
            index.knn_search([1.0, 1.0], k=0)

        # Dimension mismatch in query
        with self.assertRaises(ValueError):
            index.knn_search([1.0, 1.0, 1.0], k=1)

    def test_k_greater_than_total_nodes(self):
        index = HNSWIndex(dim=2, seed=123)
        index.add_item("a", [0.0, 0.0])
        index.add_item("b", [1.0, 1.0])

        # Request k=10 when only 2 items exist
        results = index.knn_search([0.0, 0.0], k=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "a")
        self.assertEqual(results[1][0], "b")

    def test_distance_ordering_and_monotonicity(self):
        index = HNSWIndex(dim=2, M=4, seed=42)
        # Add 15 grid points
        for i in range(15):
            index.add_item(i, [float(i), float(i * 2)])

        results = index.knn_search([5.0, 10.0], k=7)
        self.assertEqual(len(results), 7)

        # Distances must be sorted in non-decreasing order
        for idx in range(len(results) - 1):
            self.assertLessEqual(results[idx][1], results[idx + 1][1])

    def test_deterministic_behavior_with_seed(self):
        # Two indexes built with the exact same seed and data must yield identical results
        idx1 = HNSWIndex(dim=4, M=4, seed=999)
        idx2 = HNSWIndex(dim=4, M=4, seed=999)

        data = [
            (f"node_{i}", [random.uniform(-1, 1) for _ in range(4)])
            for i in range(25)
        ]

        for n_id, vec in data:
            idx1.add_item(n_id, vec)
            idx2.add_item(n_id, vec)

        query = [0.1, -0.2, 0.5, 0.0]
        res1 = idx1.knn_search(query, k=5)
        res2 = idx2.knn_search(query, k=5)

        self.assertEqual(res1, res2)

    def test_high_accuracy_recall_against_brute_force(self):
        """
        Verify that HNSW achieves high recall (> 90%) when compared against
        an exact linear brute-force scan on a 50-vector dataset.
        """
        rng = random.Random(777)
        dim = 8
        n_vectors = 60
        index = HNSWIndex(dim=dim, M=8, ef_construction=64, ef_search=32, seed=777)

        vectors = {}
        for i in range(n_vectors):
            vec = [rng.uniform(-5.0, 5.0) for _ in range(dim)]
            vectors[i] = vec
            index.add_item(i, vec)

        # Test 10 queries
        k = 4
        total_hits = 0
        total_queries = 10

        for _ in range(total_queries):
            query = [rng.uniform(-5.0, 5.0) for _ in range(dim)]
            hnsw_results = index.knn_search(query, k=k)
            exact_results = index.brute_force_knn(query, k=k)

            hnsw_ids = {item[0] for item in hnsw_results}
            exact_ids = {item[0] for item in exact_results}

            overlap = len(hnsw_ids.intersection(exact_ids))
            total_hits += overlap

        recall = total_hits / (total_queries * k)
        # HNSW should achieve >= 90% recall
        self.assertGreaterEqual(
            recall, 0.90, f"Expected recall >= 0.90, got {recall:.2f}"
        )

    def test_cosine_metric_index(self):
        index = HNSWIndex(dim=2, metric="cosine", seed=42)
        index.add_item("unit_x", [1.0, 0.0])
        index.add_item("unit_y", [0.0, 1.0])
        index.add_item("diag", [1.0, 1.0])

        # Query close to unit_x
        res = index.knn_search([0.9, 0.1], k=1)
        self.assertEqual(res[0][0], "unit_x")

    def test_serialization_save_and_load(self):
        index = HNSWIndex(dim=3, M=4, seed=42)
        index.add_item("item_1", [1.0, 0.0, 0.0])
        index.add_item("item_2", [0.0, 1.0, 0.0])
        index.add_item("item_3", [0.0, 0.0, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_hnsw.json")
            index.save(file_path)

            loaded_index = HNSWIndex.load(file_path)

            self.assertEqual(loaded_index.dim, index.dim)
            self.assertEqual(loaded_index.metric_name, index.metric_name)
            self.assertEqual(len(loaded_index.nodes), len(index.nodes))
            self.assertEqual(loaded_index.max_level, index.max_level)
            self.assertEqual(loaded_index.enter_point_id, index.enter_point_id)

            # Check query results match exactly
            q = [0.8, 0.1, 0.0]
            res_orig = index.knn_search(q, k=2)
            res_loaded = loaded_index.knn_search(q, k=2)
            self.assertEqual(res_orig, res_loaded)

    def test_index_diagnostics_and_stats(self):
        index = HNSWIndex(dim=2, M=4, seed=42)
        for i in range(10):
            index.add_item(i, [float(i), float(i)])

        stats = index.get_stats()
        self.assertEqual(stats["total_nodes"], 10)
        self.assertIn(0, stats["nodes_per_level"])
        self.assertEqual(stats["nodes_per_level"][0], 10)  # All nodes belong to layer 0
        self.assertGreater(stats["total_distance_computations"], 0)

        # Test counter reset
        index.reset_distance_computations()
        self.assertEqual(index.total_distance_computations, 0)


if __name__ == "__main__":
    unittest.main()
