"""
Deterministic Feature-Hashing Text Embedder.

Provides a reproducible, zero-dependency embedding pipeline using the Hashing Trick
(Weinberger et al., 2009) to project arbitrary text documents into a fixed-dimensional
dense vector space.

Guarantees:
- 100% deterministic across Python processes, platforms, and restarts (uses cryptographic SHA-256).
- Configurable vector dimensionality.
- Fast O(T) embedding time where T is token count.
- Zero external package dependencies (standard library only).
"""

from __future__ import annotations

import collections
import hashlib
import math
from typing import Sequence

from docproc.embeddings.base import BaseEmbedder


class DeterministicHashEmbedder(BaseEmbedder):
    """
    Feature-Hashing Text Vectorizer.

    Uses a deterministic cryptographic hash function (SHA-256) to map token n-grams
    to coordinates in a fixed d-dimensional vector space:
    1. Tokenize input text into normalized lower-case tokens.
    2. Compute token frequency counts with sublinear term-frequency scaling (1 + ln(count)).
    3. Map each token to:
       - Dimension index: index = sha256("idx:" + token) mod dim
       - Sign multiplier: sign = +1 if sha256("sgn:" + token) is even else -1
    4. Accumulate signed term weights into coordinate buckets.
    5. Apply L2-normalization so that all output vectors reside on the unit hypersphere (||v||_2 = 1.0).

    Time Complexity:
        - embed_text: O(T * log(d)) ~ O(T) where T is token count.
    Space Complexity:
        - O(d) auxiliary storage for the vector accumulator.
    """

    def __init__(self, dim: int = 64) -> None:
        """
        Initialize the deterministic hash embedder.

        Args:
            dim: Dimension of output vectors (e.g. 64, 128, 384). Defaults to 64.
        """
        super().__init__(dim=dim)

    @property
    def embedding_type(self) -> str:
        """Identifier for this embedding model."""
        return "deterministic_hash"

    def embed_text(self, text: str) -> list[float]:
        """
        Convert text into a deterministic, normalized d-dimensional vector.

        Args:
            text: Input document or query string.

        Returns:
            List of floats of length `self.dim`.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text)}.")

        tokens = self._tokenize(text)
        if not tokens:
            # Handle empty token stream gracefully with uniform normalized unit vector
            val = 1.0 / math.sqrt(self.dim)
            return [val] * self.dim

        # Compute term frequencies
        tf_counts = collections.Counter(tokens)

        # Accumulator vector initialized to zeros
        vec = [0.0] * self.dim

        for token, count in tf_counts.items():
            token_bytes = token.encode("utf-8")

            # 1. Coordinate index derivation via SHA-256
            h_idx = hashlib.sha256(b"idx:" + token_bytes).digest()
            idx = int.from_bytes(h_idx[:4], byteorder="big") % self.dim

            # 2. Sign hashing to remove bias (expectation of dot product is preserved)
            h_sgn = hashlib.sha256(b"sgn:" + token_bytes).digest()
            sign = 1.0 if (h_sgn[0] % 2 == 0) else -1.0

            # 3. Sublinear TF weighting: 1.0 + ln(count)
            weight = 1.0 + math.log(count)

            vec[idx] += sign * weight

        # 4. L2 unit normalization
        normalized_vec = self._l2_normalize(vec)
        return self._validate_vector_output(normalized_vec)
