"""
Abstract base class and utilities for document text embedders.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Sequence


class BaseEmbedder(ABC):
    """
    Abstract Base Class for text-to-vector embedding models.

    Subclasses must implement:
    - embed_text(text: str) -> list[float]
    - embedding_type: property returning the identifier of the method
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize embedder with fixed output dimension.

        Args:
            dim: Number of components in output vectors (must be >= 1).
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Vector dimension must be a positive integer, got {dim}.")
        self.dim: int = dim

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """Return the descriptive type/name of the embedding method."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text string into a fixed-length numerical vector.

        Args:
            text: Input string to embed.

        Returns:
            List of floats of length `self.dim`.
        """
        pass

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Convert a sequence of text strings into numerical vectors.

        Args:
            texts: Sequence of strings.

        Returns:
            List of vector embeddings.
        """
        return [self.embed_text(t) for t in texts]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize raw text into normalized lower-case word tokens.

        DSA Detail:
        Uses a regular expression scanning O(L) characters where L is string length,
        extracting contiguous alphanumeric character sequences.

        Args:
            text: Raw input string.

        Returns:
            List of lowercased string tokens.
        """
        if not text:
            return []
        # Match words (alphanumeric sequences, including hyphenated terms)
        return [token.lower() for token in re.findall(r"\b[a-zA-Z0-9_\-]+\b", text)]

    def _l2_normalize(self, vec: list[float]) -> list[float]:
        """
        Normalize a vector to unit L2 length (||v||_2 = 1.0).

        Mathematical Formula:
            v_norm[i] = v[i] / sqrt(sum(v[j]^2))

        Time Complexity: O(d) where d is vector dimension.
        Space Complexity: O(d) for normalized copy.

        Args:
            vec: Input numerical vector.

        Returns:
            L2-normalized numerical vector.
        """
        sum_sq = sum(x * x for x in vec)
        norm = math.sqrt(sum_sq)
        if norm < 1e-12:
            # Fallback for zero vector: return uniform unit vector
            val = 1.0 / math.sqrt(self.dim)
            return [val] * self.dim
        return [x / norm for x in vec]

    def _validate_vector_output(self, vec: list[float]) -> list[float]:
        """
        Verify output vector length and numeric validity.

        Args:
            vec: Vector to validate.

        Returns:
            Validated vector.
        """
        if len(vec) != self.dim:
            raise ValueError(
                f"Embedder '{self.embedding_type}' produced vector of length {len(vec)}, "
                f"expected {self.dim}."
            )
        for i, val in enumerate(vec):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Vector element {i} is not a number: {type(val)}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector element {i} is invalid (NaN or Inf).")
        return vec
