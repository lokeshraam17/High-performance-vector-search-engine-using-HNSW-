"""
Hierarchical Navigable Small World (HNSW) Index implementation.

This module implements the core HNSW algorithm for Approximate Nearest Neighbor (ANN)
search, following the fundamental algorithms by Malkov & Yashunin (2018).

Features:
- Pure Python implementation without external C++ or third-party ANN libraries.
- Support for multiple distance metrics (Euclidean, Cosine, Dot Product).
- Randomized geometric layer distribution.
- Greedy multi-layer routing with bounded beam search at layer 0.
- Bidirectional neighbor connection management with pruning.
- Deterministic seeding for reproducible testing.
- Built-in exact brute-force search for baseline recall calculation.
- Save/load serialization support.
"""

import heapq
import json
import math
import random
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from hnsw.distance import get_distance_function, validate_vector
from hnsw.node import HNSWNode


class HNSWIndex:
    """
    Hierarchical Navigable Small World Graph index.
    
    Attributes:
        dim: Dimensionality of vector embeddings.
        metric: Distance metric ('euclidean', 'cosine', 'dot_product').
        M: Maximum number of outgoing connections per node for layers > 0.
        M0: Maximum number of outgoing connections for layer 0 (typically 2 * M).
        ef_construction: Size of the dynamic candidate list during graph building.
        ef_search: Default size of the dynamic candidate list during query search.
        mL: Normalization factor for level generation (1.0 / ln(M)).
        seed: Optional random seed for reproducible level generation.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "euclidean",
        M: int = 16,
        M0: Optional[int] = None,
        ef_construction: int = 100,
        ef_search: int = 50,
        mL: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        if M < 2:
            raise ValueError(f"M must be at least 2, got {M}.")
        if ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {ef_construction}.")
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}.")

        self.dim: int = dim
        self.metric_name: str = metric.lower()
        self.dist_fn = get_distance_function(self.metric_name)

        self.M: int = M
        self.M0: int = M0 if M0 is not None else 2 * M
        self.ef_construction: int = ef_construction
        self.ef_search: int = ef_search
        self.mL: float = mL if mL is not None else (1.0 / math.log(M))

        self.rng = random.Random(seed)

        # Graph storage
        self.nodes: Dict[Any, HNSWNode] = {}
        self.enter_point_id: Optional[Any] = None
        self.max_level: int = -1

        # Diagnostic counters
        self.total_distance_computations: int = 0

    def _distance(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        """Calculate distance and track operation count."""
        self.total_distance_computations += 1
        return self.dist_fn(v1, v2)

    def _random_level(self) -> int:
        """
        Generate a random level for a new node according to an exponential distribution.
        Formula: floor(-ln(uniform(0, 1)) * mL)
        
        This creates a skip-list-like hierarchy where the probability of a node
        appearing at level l is (1/M)^l.
        """
        u = self.rng.random()
        # Avoid ln(0) error
        u = max(1e-10, u)
        level = int(math.floor(-math.log(u) * self.mL))
        return level

    def _search_layer(
        self,
        query: Sequence[float],
        entry_points: Sequence[Any],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, Any]]:
        """
        Greedy beam search for the nearest neighbors of a query vector at a specific layer.
        
        Data structures:
        - v: Set of visited node IDs to prevent cycles and redundant distance calculations.
        - C: Min-heap of candidate nodes to explore: (distance, node_id).
        - W: Max-heap of closest elements found so far: (-distance, node_id) bounded by ef.
        
        Args:
            query: Query vector.
            entry_points: Initial entry node IDs for the search.
            ef: Number of nearest neighbors to track in the candidate set.
            layer: The layer index to search within.
            
        Returns:
            List of (distance, node_id) pairs sorted in ascending order of distance.
        """
        visited = set(entry_points)
        candidates_min_heap: List[Tuple[float, Any]] = []
        best_w_max_heap: List[Tuple[float, Any]] = []

        # Initialize heaps with entry points
        for ep_id in entry_points:
            ep_vec = self.nodes[ep_id].vector
            d = self._distance(query, ep_vec)
            heapq.heappush(candidates_min_heap, (d, ep_id))
            heapq.heappush(best_w_max_heap, (-d, ep_id))
            if len(best_w_max_heap) > ef:
                heapq.heappop(best_w_max_heap)

        # Explore candidates greedily
        while candidates_min_heap:
            cand_dist, cand_id = heapq.heappop(candidates_min_heap)
            furthest_best_dist = -best_w_max_heap[0][0]

            # If the closest unexplored candidate is farther than the furthest
            # element currently in our best set W, we can safely terminate.
            if cand_dist > furthest_best_dist:
                break

            cand_node = self.nodes[cand_id]
            for neighbor_id in cand_node.get_neighbors(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    n_vec = self.nodes[neighbor_id].vector
                    n_dist = self._distance(query, n_vec)
                    furthest_best_dist = -best_w_max_heap[0][0]

                    # If this neighbor improves our top-ef set or W is not yet full
                    if n_dist < furthest_best_dist or len(best_w_max_heap) < ef:
                        heapq.heappush(candidates_min_heap, (n_dist, neighbor_id))
                        if len(best_w_max_heap) < ef:
                            heapq.heappush(best_w_max_heap, (-n_dist, neighbor_id))
                        else:
                            heapq.heapreplace(best_w_max_heap, (-n_dist, neighbor_id))

        # Convert max-heap into ascending list of (distance, node_id)
        results = [(-dist, node_id) for dist, node_id in best_w_max_heap]
        results.sort(key=lambda item: item[0])
        return results

    def _select_neighbors_simple(
        self,
        candidates: List[Tuple[float, Any]],
        max_m: int,
    ) -> List[Any]:
        """
        Select up to max_m nearest candidates.
        
        Args:
            candidates: List of (distance, node_id) pairs.
            max_m: Maximum number of neighbors to select.
            
        Returns:
            List of selected node IDs.
        """
        sorted_candidates = sorted(candidates, key=lambda item: item[0])
        return [node_id for _, node_id in sorted_candidates[:max_m]]

    def _prune_connections(self, node_id: Any, layer: int, max_m: int) -> None:
        """
        Shrink the neighbor connections of a node at a given layer to at most max_m.
        Keeps the neighbors closest to node_id.
        """
        node = self.nodes[node_id]
        neighbors = node.get_neighbors(layer)
        if len(neighbors) <= max_m:
            return

        # Measure distances from node_id to all of its current neighbors
        scored_neighbors: List[Tuple[float, Any]] = []
        for n_id in neighbors:
            d = self._distance(node.vector, self.nodes[n_id].vector)
            scored_neighbors.append((d, n_id))

        scored_neighbors.sort(key=lambda item: item[0])
        kept_neighbors = [n_id for _, n_id in scored_neighbors[:max_m]]
        node.set_neighbors(layer, kept_neighbors)

    def add_item(self, node_id: Any, vector: Sequence[float]) -> None:
        """
        Insert a new vector with the given node_id into the HNSW graph.
        
        Algorithm Flow:
        1. Input validation (dimensions, numerical validity, unique ID).
        2. Assign a random maximum level l for the node.
        3. If graph is empty, set this node as entry point and return.
        4. Greedily traverse down from top layer to level l + 1 finding the closest node.
        5. From layer min(max_level, l) down to 0:
           - Search layer for closest candidates using efConstruction.
           - Select up to M (or M0 at layer 0) neighbors and establish bidirectional edges.
           - Prune neighbor edges if they exceed degree capacity.
        6. If l > max_level, update graph max_level and entry point to this node.
        
        Args:
            node_id: Unique identifier for the item.
            vector: Numerical embedding vector matching index dimensionality.
        """
        if node_id is None:
            raise ValueError("node_id cannot be None.")
        if node_id in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' already exists in index.")

        validate_vector(vector, expected_dim=self.dim)
        vec_list = [float(x) for x in vector]

        node_level = self._random_level()
        new_node = HNSWNode(node_id=node_id, vector=vec_list, level=node_level)
        self.nodes[node_id] = new_node

        # First node in graph becomes the entry point
        if self.enter_point_id is None:
            self.enter_point_id = node_id
            self.max_level = node_level
            return

        curr_obj = self.enter_point_id

        # Phase 1: Fast greedy navigation through upper layers down to insertion level
        # At layers > node_level, we do not add connections; we just navigate greedily.
        for lc in range(self.max_level, node_level, -1):
            candidates = self._search_layer(vec_list, [curr_obj], ef=1, layer=lc)
            curr_obj = candidates[0][1]

        # Phase 2: Add connections at levels from min(max_level, node_level) down to 0
        top_connect_level = min(self.max_level, node_level)
        for lc in range(top_connect_level, -1, -1):
            candidates = self._search_layer(
                vec_list, [curr_obj], ef=self.ef_construction, layer=lc
            )
            max_conn = self.M0 if lc == 0 else self.M
            neighbors = self._select_neighbors_simple(candidates, max_conn)

            # Connect bidirectionally
            for n_id in neighbors:
                new_node.add_neighbor(lc, n_id)
                self.nodes[n_id].add_neighbor(lc, node_id)
                if self.nodes[n_id].degree(lc) > max_conn:
                    self._prune_connections(n_id, lc, max_conn)

            # Entry point for the next lower layer is the closest element found
            curr_obj = candidates[0][1]

        # Update entry point if new node has a higher level than current max_level
        if node_level > self.max_level:
            self.max_level = node_level
            self.enter_point_id = node_id

    def knn_search(
        self,
        query: Sequence[float],
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Perform approximate K-nearest neighbor search for a query vector.
        
        Args:
            query: Query vector matching index dimension.
            k: Number of nearest neighbors to retrieve (must be >= 1).
            ef_search: Dynamic candidate list size during search (defaults to max(k, self.ef_search)).
            
        Returns:
            List of (node_id, distance) tuples sorted in ascending order of distance.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer >= 1, got {k}.")
        if len(self.nodes) == 0:
            return []

        validate_vector(query, expected_dim=self.dim)
        query_list = [float(x) for x in query]

        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(k, ef)

        curr_obj = self.enter_point_id

        # Greedy zoom-in through layers from max_level down to 1
        for lc in range(self.max_level, 0, -1):
            candidates = self._search_layer(query_list, [curr_obj], ef=1, layer=lc)
            curr_obj = candidates[0][1]

        # Detailed beam search at layer 0 using ef candidates
        layer_0_results = self._search_layer(query_list, [curr_obj], ef=ef, layer=0)

        # Return top-k closest items formatted as (node_id, distance)
        return [(node_id, dist) for dist, node_id in layer_0_results[:k]]

    def brute_force_knn(
        self,
        query: Sequence[float],
        k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """
        Exact (brute force) linear scan baseline for comparison and accuracy calculation.
        Computes distances between query and every node in the index.
        
        Returns:
            List of (node_id, distance) sorted by ascending distance.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer >= 1, got {k}.")
        if len(self.nodes) == 0:
            return []

        validate_vector(query, expected_dim=self.dim)
        query_list = [float(x) for x in query]

        all_distances: List[Tuple[float, Any]] = []
        for n_id, node in self.nodes.items():
            d = self._distance(query_list, node.vector)
            all_distances.append((d, n_id))

        all_distances.sort(key=lambda item: item[0])
        return [(n_id, d) for d, n_id in all_distances[:k]]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return structural and diagnostic statistics about the graph.
        Useful for Teammate 3 (Benchmarking) and debugging.
        """
        nodes_per_level: Dict[int, int] = {}
        connections_per_level: Dict[int, int] = {}

        for node in self.nodes.values():
            for l in range(node.level + 1):
                nodes_per_level[l] = nodes_per_level.get(l, 0) + 1
                deg = node.degree(l)
                connections_per_level[l] = connections_per_level.get(l, 0) + deg

        avg_degrees = {
            l: (connections_per_level[l] / nodes_per_level[l] if nodes_per_level[l] > 0 else 0)
            for l in sorted(nodes_per_level.keys())
        }

        return {
            "total_nodes": len(self.nodes),
            "dim": self.dim,
            "metric": self.metric_name,
            "max_level": self.max_level,
            "enter_point_id": self.enter_point_id,
            "nodes_per_level": nodes_per_level,
            "avg_degree_per_level": avg_degrees,
            "total_distance_computations": self.total_distance_computations,
        }

    def reset_distance_computations(self) -> None:
        """Reset the distance computation counter."""
        self.total_distance_computations = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize index structure into a dictionary."""
        return {
            "dim": self.dim,
            "metric": self.metric_name,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "mL": self.mL,
            "max_level": self.max_level,
            "enter_point_id": self.enter_point_id,
            "nodes": {
                str(n_id): {
                    "id": n_id,
                    "vector": node.vector,
                    "level": node.level,
                    "neighbors": {str(lvl): nbrs for lvl, nbrs in node.neighbors.items()},
                }
                for n_id, node in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HNSWIndex":
        """Reconstruct index from a serialized dictionary."""
        index = cls(
            dim=data["dim"],
            metric=data["metric"],
            M=data["M"],
            M0=data["M0"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
            mL=data["mL"],
        )
        index.max_level = data["max_level"]
        index.enter_point_id = data["enter_point_id"]

        for n_id_str, n_data in data["nodes"].items():
            node_id = n_data["id"]
            node = HNSWNode(
                node_id=node_id,
                vector=n_data["vector"],
                level=n_data["level"],
            )
            # Restore neighbors with integer layer keys
            node.neighbors = {
                int(lvl): list(nbrs) for lvl, nbrs in n_data["neighbors"].items()
            }
            index.nodes[node_id] = node

        return index

    def save(self, filepath: str) -> None:
        """Save index to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "HNSWIndex":
        """Load index from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
