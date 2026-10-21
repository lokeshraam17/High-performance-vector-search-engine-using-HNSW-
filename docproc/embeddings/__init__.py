"""
Vector Embeddings Subpackage.

Provides text-to-vector embedding strategies:
- DeterministicHashEmbedder: Fast, reproducible feature-hashing with zero external dependencies.
- TFIDFEmbedder: Classic Information Retrieval term frequency - inverse document frequency model.
- SentenceTransformerEmbedder: Modern neural Transformer embeddings (requires sentence-transformers).
"""

from typing import Any

from docproc.embeddings.base import BaseEmbedder
from docproc.embeddings.hash_embedder import DeterministicHashEmbedder
from docproc.embeddings.sentence_transformer import SentenceTransformerEmbedder
from docproc.embeddings.tfidf_embedder import TFIDFEmbedder


def create_embedder(
    method: str = "deterministic_hash",
    dim: int = 64,
    **kwargs: Any,
) -> BaseEmbedder:
    """
    Factory function to instantiate a configured text embedder.

    Supported methods:
    - 'deterministic_hash' (or 'hash'): Zero-dependency feature hashing vectorizer.
    - 'tfidf': Pure Python TF-IDF vectorizer (can pass corpus=... in kwargs).
    - 'sentence_transformers' (or 'sentence_transformer', 'neural'): Pretrained Dense Transformer.

    Args:
        method: Embedder strategy name.
        dim: Vector dimension (ignored if sentence_transformers specifies fixed model dim).
        **kwargs: Additional parameters passed to the specific embedder constructor.

    Returns:
        Instance of BaseEmbedder.
    """
    normalized_method = method.strip().lower()

    if normalized_method in ("deterministic_hash", "hash", "test_vector"):
        return DeterministicHashEmbedder(dim=dim)

    if normalized_method in ("tfidf", "tf_idf"):
        corpus = kwargs.get("corpus")
        return TFIDFEmbedder(dim=dim, corpus=corpus)

    if normalized_method in ("sentence_transformers", "sentence_transformer", "neural"):
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        device = kwargs.get("device")
        return SentenceTransformerEmbedder(model_name=model_name, dim=dim if "dim" in kwargs else None, device=device)

    raise ValueError(
        f"Unknown embedding method '{method}'. "
        f"Supported options: 'deterministic_hash', 'tfidf', 'sentence_transformers'."
    )


__all__ = [
    "BaseEmbedder",
    "DeterministicHashEmbedder",
    "TFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
]
