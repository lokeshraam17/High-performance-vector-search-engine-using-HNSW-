"""
Synthetic Vector Generation for ANN Benchmarking.

This module provides reproducible, parameterized vector generation:
- Uniform distribution over the unit hypersphere S^(d-1).
- Clustered Gaussian distributions (simulating semantic document embeddings).
- Uniform hypercube distribution [-1.0, 1.0]^d.

All generators use Python's random.Random with explicit seeds to guarantee
100% reproducibility across benchmark runs and platforms.
"""

import math
import random
from typing import List, Literal, Sequence, Tuple


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    """Normalize a vector to unit Euclidean length."""
    norm_sq = sum(x * x for x in vec)
    norm = math.sqrt(norm_sq)
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def generate_synthetic_vectors(
    num_vectors: int,
    dim: int,
    distribution: Literal["uniform_sphere", "gaussian_clusters", "uniform_cube"] = "uniform_sphere",
    seed: int = 42,
    num_clusters: int = 5,
) -> List[List[float]]:
    """
    Generate synthetic vector dataset for benchmarking.
    
    Args:
        num_vectors: Number of vectors to generate (must be >= 1).
        dim: Dimensionality of each vector (must be >= 1).
        distribution: Distribution strategy ('uniform_sphere', 'gaussian_clusters', 'uniform_cube').
        seed: Random seed for 100% reproducible benchmark datasets.
        num_clusters: Number of cluster centroids when using 'gaussian_clusters'.
        
    Returns:
        List of generated vectors, each represented as a List[float] of length dim.
        
    Raises:
        ValueError: If parameters are non-positive or distribution name is unknown.
    """
    if num_vectors <= 0:
        raise ValueError(f"num_vectors must be >= 1, got {num_vectors}.")
    if dim <= 0:
        raise ValueError(f"dim must be >= 1, got {dim}.")

    rng = random.Random(seed)
    vectors: List[List[float]] = []

    if distribution == "uniform_sphere":
        # By the Muller-Marsaglia method, sampling coordinates from standard normal
        # N(0, 1) and normalizing yields a mathematically uniform distribution on S^(d-1).
        for _ in range(num_vectors):
            coords = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            vectors.append(_l2_normalize(coords))

    elif distribution == "gaussian_clusters":
        # Simulates real semantic embeddings clustered around topical centroids
        cluster_centers = []
        for _ in range(num_clusters):
            center = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            cluster_centers.append(_l2_normalize(center))

        for _ in range(num_vectors):
            center = rng.choice(cluster_centers)
            coords = [c + rng.gauss(0.0, 0.15) for c in center]
            vectors.append(_l2_normalize(coords))

    elif distribution == "uniform_cube":
        # Uniform random coordinates in [-1.0, 1.0]
        for _ in range(num_vectors):
            coords = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors.append(coords)

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Supported: 'uniform_sphere', 'gaussian_clusters', 'uniform_cube'."
        )

    return vectors


def split_base_and_query(
    vectors: List[List[float]],
    num_queries: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Split a list of vectors into base dataset vectors and disjoint query vectors.
    
    Args:
        vectors: Total pool of generated vectors.
        num_queries: Number of vectors reserved for querying.
        
    Returns:
        (base_vectors, query_vectors) tuple.
    """
    if num_queries <= 0 or num_queries >= len(vectors):
        raise ValueError(
            f"num_queries ({num_queries}) must be between 1 and {len(vectors) - 1}."
        )

    base = vectors[:-num_queries]
    queries = vectors[-num_queries:]
    return base, queries
