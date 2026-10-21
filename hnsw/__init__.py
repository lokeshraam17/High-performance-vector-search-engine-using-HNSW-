"""
HNSW (Hierarchical Navigable Small World) Graph Core Package.

Provides core classes and utilities for approximate nearest neighbor search:
- HNSWIndex: Primary index for vector insertion and approximate nearest neighbor search.
- HNSWNode: Node representation within the hierarchical graph.
- Distance functions: euclidean_distance, cosine_distance, dot_product_distance.
"""

from hnsw.distance import (
    cosine_distance,
    dot_product_distance,
    euclidean_distance,
    get_distance_function,
    validate_vector,
)
from hnsw.index import HNSWIndex
from hnsw.node import HNSWNode

__all__ = [
    "HNSWIndex",
    "HNSWNode",
    "euclidean_distance",
    "cosine_distance",
    "dot_product_distance",
    "get_distance_function",
    "validate_vector",
]
