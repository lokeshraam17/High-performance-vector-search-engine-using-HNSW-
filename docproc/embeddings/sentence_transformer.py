"""
Dense Neural Embeddings via Sentence-Transformers.

Integrates modern pre-trained Transformer embedding models (such as all-MiniLM-L6-v2)
for dense semantic vector representations.

Note:
Requires optional dependencies:
    pip install sentence-transformers torch

If dependencies are missing, provides clear instructions and recommends fallback to
DeterministicHashEmbedder or TFIDFEmbedder.
"""

from __future__ import annotations

from typing import Any, Sequence

from docproc.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Neural Dense Vector Embedder wrapping Hugging Face / Sentence-Transformers models.

    Provides state-of-the-art dense semantic text representations.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize SentenceTransformerEmbedder.

        Args:
            model_name: Name of the pre-trained model (default: 'all-MiniLM-L6-v2').
            dim: Expected vector dimension. If None, derived from model.
            device: Optional compute device ('cpu', 'cuda', etc.).

        Raises:
            ImportError: If sentence-transformers or torch is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "The 'sentence-transformers' package is not installed in the current environment.\n"
                "To enable dense neural embeddings, please run:\n"
                "    pip install sentence-transformers torch\n"
                "Alternatively, use 'DeterministicHashEmbedder' or 'TFIDFEmbedder' for zero-dependency "
                "deterministic vector generation."
            ) from err

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

        model_dim = self._model.get_sentence_embedding_dimension()
        if dim is not None and dim != model_dim:
            raise ValueError(
                f"Specified dimension {dim} does not match model '{model_name}' "
                f"native dimension {model_dim}."
            )

        super().__init__(dim=model_dim)

    @property
    def embedding_type(self) -> str:
        """Identifier for this embedding model."""
        return f"sentence_transformers ({self._model_name})"

    def embed_text(self, text: str) -> list[float]:
        """
        Encode single text string to a dense semantic vector.

        Args:
            text: Input string.

        Returns:
            Normalized float vector of length `self.dim`.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text)}.")

        # normalize_embeddings=True ensures ||v||_2 = 1.0 (ideal for cosine and euclidean metric)
        raw_vec = self._model.encode(text, normalize_embeddings=True)
        vec = [float(x) for x in raw_vec]
        return self._validate_vector_output(vec)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Batch encode multiple texts efficiently.

        Args:
            texts: Sequence of text strings.

        Returns:
            List of normalized float vectors.
        """
        if not texts:
            return []

        raw_vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        output = []
        for raw_vec in raw_vecs:
            vec = [float(x) for x in raw_vec]
            self._validate_vector_output(vec)
            output.append(vec)
        return output
