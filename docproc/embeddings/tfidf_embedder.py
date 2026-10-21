"""
TF-IDF (Term Frequency - Inverse Document Frequency) Vector Embedder.

Pure Python Information Retrieval vector space model without external dependencies.
Extracts vocabulary terms, computes smoothed inverse document frequencies, and projects
documents and queries into an L2-normalized vector space.
"""

from __future__ import annotations

import collections
import math
from typing import Sequence

from docproc.embeddings.base import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """
    Term Frequency - Inverse Document Frequency (TF-IDF) Vectorizer.

    Steps:
    1. Corpus Fitting:
       - Tokenizes all training documents.
       - Computes document frequencies: df(t) = number of documents containing term t.
       - Selects the top `dim` most informative terms by document frequency.
       - Computes smoothed inverse document frequency:
             idf(t) = ln(1 + (N + 1) / (1 + df(t))) + 1.0
    2. Vector Transformation:
       - For a document or search query, counts term frequencies tf(t).
       - Weight: w(t) = (1.0 + ln(tf(t))) * idf(t) for term t in vocabulary.
       - Normalizes output vector to unit L2 length: ||v||_2 = 1.0.

    Time Complexity:
        - fit: O(N * T) where N is corpus size and T is average tokens per document.
        - embed_text: O(T) where T is token count of input text.
    Space Complexity:
        - O(|V| + dim) where |V| is vocabulary size.
    """

    def __init__(self, dim: int = 64, corpus: Sequence[str] | None = None) -> None:
        """
        Initialize TFIDFEmbedder.

        Args:
            dim: Output vector dimension (vocabulary capacity).
            corpus: Optional initial corpus of document texts to fit immediately.
        """
        super().__init__(dim=dim)
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._is_fitted: bool = False

        if corpus is not None:
            self.fit(corpus)

    @property
    def embedding_type(self) -> str:
        """Identifier for this embedding model."""
        return "tfidf"

    @property
    def is_fitted(self) -> bool:
        """Check whether the TF-IDF vocabulary and IDF tables have been trained."""
        return self._is_fitted

    def fit(self, corpus: Sequence[str]) -> "TFIDFEmbedder":
        """
        Train vocabulary and IDF weights on a collection of texts.

        Args:
            corpus: Sequence of document texts.

        Returns:
            Self instance for method chaining.
        """
        if not corpus:
            raise ValueError("Cannot fit TFIDFEmbedder on an empty corpus.")

        num_docs = len(corpus)
        df_counts: collections.Counter[str] = collections.Counter()

        # Phase 1: Collect document frequencies for each unique term
        for text in corpus:
            if not isinstance(text, str):
                continue
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                df_counts[token] += 1

        if not df_counts:
            raise ValueError("Corpus contains no valid tokens to build vocabulary.")

        # Phase 2: Select top `dim` terms by document frequency
        # (Filtering out terms that only appear once if corpus is sufficiently large)
        most_common_terms = df_counts.most_common(self.dim)

        self.vocab = {term: idx for idx, (term, _) in enumerate(most_common_terms)}

        # Phase 3: Compute smoothed IDF for selected vocabulary terms
        self.idf = {}
        for term in self.vocab:
            df = df_counts[term]
            # Standard Scikit-Learn style smoothed IDF formula
            self.idf[term] = math.log(1.0 + (num_docs + 1.0) / (df + 1.0)) + 1.0

        self._is_fitted = True
        return self

    def fit_transform(self, corpus: Sequence[str]) -> list[list[float]]:
        """
        Fit vocabulary on corpus and return vector embeddings for all documents.

        Args:
            corpus: Sequence of document strings.

        Returns:
            List of L2-normalized TF-IDF vectors.
        """
        self.fit(corpus)
        return self.embed_batch(corpus)

    def embed_text(self, text: str) -> list[float]:
        """
        Transform a single text string into an L2-normalized TF-IDF vector.

        Args:
            text: Input string.

        Returns:
            List of floats of length `self.dim`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TFIDFEmbedder has not been fitted yet. Please call embedder.fit(corpus) "
                "or pass corpus to constructor before embedding text."
            )

        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text)}.")

        tokens = self._tokenize(text)
        vec = [0.0] * self.dim

        if not tokens:
            val = 1.0 / math.sqrt(self.dim)
            return [val] * self.dim

        tf_counts = collections.Counter(tokens)

        # Populate coordinates for tokens present in vocabulary
        for token, count in tf_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf_weight = 1.0 + math.log(count)
                idf_weight = self.idf[token]
                vec[idx] = tf_weight * idf_weight

        # Normalize vector to unit length
        normalized_vec = self._l2_normalize(vec)
        return self._validate_vector_output(normalized_vec)
