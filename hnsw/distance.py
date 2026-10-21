"""
Distance metrics for vector similarity and nearest-neighbor search.

Supported metrics:
- Euclidean distance (L2 distance)
- Cosine distance (1 - cosine similarity)
- Dot product distance (negative inner product)
"""

import math
from typing import Sequence


def validate_vector(vec: Sequence[float], expected_dim: int | None = None) -> None:
    """
    Validate that a vector is a non-empty sequence of finite numerical values.
    Optionally checks if the dimension matches expected_dim.
    
    Raises:
        TypeError: If vec is not a sequence or contains non-numeric elements.
        ValueError: If vec is empty, contains NaN/inf, or has mismatched dimension.
    """
    if vec is None:
        raise ValueError("Vector cannot be None.")
    if not isinstance(vec, (list, tuple)):
        try:
            vec = list(vec)
        except Exception as e:
            raise TypeError(f"Vector must be an iterable sequence: {e}")
            
    if len(vec) == 0:
        raise ValueError("Vector cannot be empty (dimension must be >= 1).")
        
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dim}, got {len(vec)}."
        )

    for i, val in enumerate(vec):
        if not isinstance(val, (int, float)):
            raise TypeError(f"Vector element at index {i} must be a number, got {type(val)}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Vector element at index {i} is invalid (NaN or Infinite).")


def euclidean_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.
    Formula: sqrt(sum((v1[i] - v2[i])^2))
    
    Time Complexity: O(d) where d is vector dimension.
    Space Complexity: O(1) auxiliary memory.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} != {len(v2)}.")
        
    sum_sq = 0.0
    for a, b in zip(v1, v2):
        diff = a - b
        sum_sq += diff * diff
    return math.sqrt(sum_sq)


def cosine_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute Cosine distance between two vectors.
    Formula: 1.0 - (v1 . v2) / (||v1|| * ||v2||)
    
    Returns a value in [0.0, 2.0].
    If either vector has zero norm, distance is 1.0 (orthogonal).
    
    Time Complexity: O(d) where d is vector dimension.
    Space Complexity: O(1) auxiliary memory.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} != {len(v2)}.")
        
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1_sq += a * a
        norm2_sq += b * b

    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)

    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0  # Zero vector treated as orthogonal to everything

    similarity = dot / (norm1 * norm2)
    # Clip to avoid floating point precision out of bounds [-1.0, 1.0]
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def dot_product_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute negative dot product distance between two vectors.
    Formula: - (v1 . v2)
    Used when vectors are already normalized and higher similarity means smaller distance.
    
    Time Complexity: O(d).
    Space Complexity: O(1).
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} != {len(v2)}.")
        
    dot = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
    return -dot


METRIC_REGISTRY = {
    "euclidean": euclidean_distance,
    "l2": euclidean_distance,
    "cosine": cosine_distance,
    "dot_product": dot_product_distance,
}


def get_distance_function(name: str):
    """Retrieve distance function by name."""
    name_lower = name.strip().lower()
    if name_lower not in METRIC_REGISTRY:
        raise ValueError(
            f"Unsupported distance metric '{name}'. Choose from: {list(METRIC_REGISTRY.keys())}."
        )
    return METRIC_REGISTRY[name_lower]
