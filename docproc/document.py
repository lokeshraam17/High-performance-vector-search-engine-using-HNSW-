"""
Document data representation for the HNSW Vector Search Engine.

Represents a single document containing:
- Unique identifier (`doc_id`)
- Raw text content (`text`)
- Arbitrary key-value metadata (`metadata`)
- Numerical vector embedding (`vector`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Document:
    """
    Data structure representing a searchable document.

    Attributes:
        doc_id: Unique string identifier for the document.
        text: Raw text content of the document.
        metadata: Arbitrary key-value metadata (e.g. title, category, author, timestamp).
        vector: Optional numerical vector embedding generated from the text.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None

    def __post_init__(self) -> None:
        """Validate document fields upon instantiation."""
        # Validate doc_id
        if self.doc_id is None:
            raise ValueError("Document doc_id cannot be None.")
        self.doc_id = str(self.doc_id).strip()
        if not self.doc_id:
            raise ValueError("Document doc_id cannot be an empty string.")

        # Validate text
        if self.text is None:
            raise ValueError(f"Document '{self.doc_id}' text cannot be None.")
        if not isinstance(self.text, str):
            raise TypeError(f"Document '{self.doc_id}' text must be a string, got {type(self.text)}.")
        if not self.text.strip():
            raise ValueError(f"Document '{self.doc_id}' text cannot be empty or pure whitespace.")

        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError(f"Document '{self.doc_id}' metadata must be a dictionary, got {type(self.metadata)}.")

        # Validate vector if already populated
        if self.vector is not None:
            self.validate_vector()

    def validate_vector(self, expected_dim: int | None = None) -> None:
        """
        Ensure document vector contains valid, finite numeric values and matches expected dimension.

        Args:
            expected_dim: Optional required dimension length.

        Raises:
            ValueError: If vector is empty, contains NaN/inf, or has mismatched dimension.
            TypeError: If vector is not a sequence of numbers.
        """
        if self.vector is None:
            raise ValueError(f"Document '{self.doc_id}' has no vector embedding.")

        if not isinstance(self.vector, (list, tuple)):
            raise TypeError(f"Document '{self.doc_id}' vector must be a list or tuple, got {type(self.vector)}.")

        if len(self.vector) == 0:
            raise ValueError(f"Document '{self.doc_id}' vector cannot be empty.")

        if expected_dim is not None and len(self.vector) != expected_dim:
            raise ValueError(
                f"Document '{self.doc_id}' vector dimension mismatch: "
                f"expected {expected_dim}, got {len(self.vector)}."
            )

        for i, val in enumerate(self.vector):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"Document '{self.doc_id}' vector element at index {i} must be a number, got {type(val)}."
                )
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Document '{self.doc_id}' vector element at index {i} is invalid (NaN or Infinite)."
                )

    def snippet(self, max_length: int = 120) -> str:
        """
        Return a clean, truncated preview of the document text for search results and logs.

        Args:
            max_length: Maximum character length for preview.

        Returns:
            Truncated string snippet with ellipsis if truncated.
        """
        clean = " ".join(self.text.split())
        if len(clean) <= max_length:
            return clean
        return clean[:max_length].rstrip() + "..."

    def to_dict(self) -> dict[str, Any]:
        """Serialize document to a JSON-compatible dictionary."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "vector": list(self.vector) if self.vector is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """
        Reconstruct a Document instance from a dictionary.

        Args:
            data: Dictionary containing 'doc_id', 'text', and optional 'metadata', 'vector'.

        Returns:
            Document instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict to deserialize Document, got {type(data)}.")
        return cls(
            doc_id=data.get("doc_id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            vector=data.get("vector"),
        )
