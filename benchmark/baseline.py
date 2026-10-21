"""
Exact Brute-Force K-Nearest Neighbor (KNN) Baseline.

This module provides an independent, exact linear-scan search implementation
used as the ground-truth oracle for evaluating HNSW search accuracy (Recall@K)
and comparing search latency / computational efficiency.

Algorithmic Details:
- Scan: Computes exact distance to every stored vector (O(N * d)).
- Top-K Selection: Employs a bounded max-heap of size K (O(N * log K))
  to extract the exact nearest neighbors in ascending order of distance.
"""

import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hnsw.distance import get_distance_function, validate_vector


class ExactKNNBaseline:
    """
    Exact linear-scan K-Nearest Neighbor baseline index.
    
    Attributes:
        dim: Expected vector dimensionality.
        metric: Distance metric ('euclidean', 'cosine', 'dot_product').
        vectors: Dictionary mapping node_id to its vector coordinates.
        total_distance_computations: Counter tracking distance evaluations.
    """

    def __init__(self, dim: int, metric: str = "euclidean") -> None:
        """
        Initialize the exact baseline index.
        
        Args:
            dim: Dimension of vector embeddings (must be > 0).
            metric: Distance metric name ('euclidean', 'cosine', or 'dot_product').
        """
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")

        self.dim: int = dim
        self.metric_name: str = metric.lower()
        self.dist_fn = get_distance_function(self.metric_name)

        # Vector storage: node_id -> List[float]
        self.vectors: Dict[Any, List[float]] = {}

        # Diagnostic counter for complexity comparison
        self.total_distance_computations: int = 0

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def size(self) -> int:
        return len(self.vectors)

    def _distance(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        """Compute metric distance and increment operation counter."""
        self.total_distance_computations += 1
        return self.dist_fn(v1, v2)

    def add_item(self, node_id: Any, vector: Sequence[float]) -> None:
        """
        Store a vector associated with a unique node identifier.
        
        Args:
            node_id: Unique identifier for the item.
            vector: Numerical embedding matching index dimensionality.
            
        Raises:
            ValueError: If node_id is None, already exists, or vector dimension mismatches.
            TypeError: If vector coordinates are non-numeric.
        """
        if node_id is None:
            raise ValueError("node_id cannot be None.")
        if node_id in self.vectors:
            raise ValueError(f"Node with ID '{node_id}' already exists in baseline index.")

        validate_vector(vector, expected_dim=self.dim)
        self.vectors[node_id] = [float(x) for x in vector]

    def add_items(self, items: Iterable[Tuple[Any, Sequence[float]]]) -> None:
        """
        Batch-add multiple (node_id, vector) pairs to the baseline index.
        
        Args:
            items: Iterable of (node_id, vector) tuples.
        """
        for node_id, vector in items:
            self.add_item(node_id, vector)

    def knn_search(
        self,
        query: Sequence[float],
        k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """
        Find exact top-k nearest neighbors via exhaustive linear scan.
        
        DSA Complexity:
        - Time: O(N * d + N * log K) where N is index size, d is vector dimension,
          and K is the number of neighbors requested.
        - Space: O(K) auxiliary memory for the bounded max-heap.
        
        Args:
            query: Query vector matching index dimension.
            k: Number of nearest neighbors to retrieve (must be >= 1).
            
        Returns:
            List of (node_id, distance) tuples sorted in ascending order of distance.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer >= 1, got {k}.")
        if len(self.vectors) == 0:
            return []

        validate_vector(query, expected_dim=self.dim)
        query_list = [float(x) for x in query]

        # Use a max-heap of size at most k to maintain the k closest candidates.
        # Python's heapq is a min-heap, so we push (-distance, node_id).
        max_heap: List[Tuple[float, Any]] = []

        for node_id, vec in self.vectors.items():
            dist = self._distance(query_list, vec)

            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, node_id))
            elif dist < -max_heap[0][0]:
                # Closer than the current worst candidate in our top-k
                heapq.heapreplace(max_heap, (-dist, node_id))

        # Extract candidates from max-heap and sort ascending by true distance
        results = [(node_id, -neg_dist) for neg_dist, node_id in max_heap]
        results.sort(key=lambda item: item[1])
        return results

    def reset_distance_computations(self) -> None:
        """Reset distance calculation counter to zero."""
        self.total_distance_computations = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Return structural statistics for reporting and diagnostics.
        """
        return {
            "total_nodes": len(self.vectors),
            "dim": self.dim,
            "metric": self.metric_name,
            "total_distance_computations": self.total_distance_computations,
        }
