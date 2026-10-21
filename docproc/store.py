"""
Document Store and Vector Management.

Maintains an in-memory document repository providing:
- O(1) hash map indexing of documents by Document ID.
- Integrated text-to-vector embedding generation.
- Strict vector dimensionality and numerical validity checking.
- Bidirectional Document ID <-> HNSW Node ID translation.
- Population bridge to HNSW graph index.
- Full JSON serialization and persistence.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from docproc.document import Document
from docproc.embeddings.base import BaseEmbedder
from docproc.embeddings.hash_embedder import DeterministicHashEmbedder
from docproc.mapper import IDMapper

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    In-memory Document and Vector Store.

    Acts as the single source of truth for raw document texts, arbitrary metadata,
    and numerical vector embeddings required by the HNSW index.

    Time Complexity:
        - add_document: O(T + d) where T is text token count and d is vector dimension.
        - get_document / get_text / get_vector: O(1) average hash map lookup.
        - remove_document: O(1) average.
        - populate_hnsw_index: O(N * (M * log(N) * d)) through HNSW insertion.
    Space Complexity:
        - O(N * (L + d)) where N is document count, L is average text length, d is dimension.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        id_mode: str = "identity",
        dim: int = 64,
    ) -> None:
        """
        Initialize the DocumentStore.

        Args:
            embedder: Embedding model instance. If None, defaults to DeterministicHashEmbedder(dim=dim).
            id_mode: ID mapping mode ('identity' or 'integer').
            dim: Default vector dimension if embedder is not explicitly provided.
        """
        if embedder is not None:
            if not isinstance(embedder, BaseEmbedder):
                raise TypeError(f"embedder must inherit from BaseEmbedder, got {type(embedder)}.")
            self.embedder: BaseEmbedder = embedder
        else:
            self.embedder = DeterministicHashEmbedder(dim=dim)

        self.dim: int = self.embedder.dim
        self._docs: dict[str, Document] = {}
        self.mapper: IDMapper = IDMapper(mode=id_mode)

    @property
    def total_documents(self) -> int:
        """Total number of documents stored."""
        return len(self._docs)

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return self.has_document(doc_id)

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store by its ID."""
        return str(doc_id).strip() in self._docs

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        vector: list[float] | None = None,
        node_id: Any | None = None,
        overwrite: bool = False,
        generate_vector: bool = True,
    ) -> Document:
        """
        Add a single document to the store, generating or validating its vector embedding.

        Args:
            doc_id: Unique string document identifier.
            text: Non-empty document text.
            metadata: Optional dictionary of metadata attributes.
            vector: Optional pre-computed vector. If None and generate_vector=True, computed via embedder.
            node_id: Optional explicit HNSW node ID.
            overwrite: If True, replaces existing document with same ID. Defaults to False.
            generate_vector: Whether to compute vector if not provided.

        Returns:
            The stored Document instance.

        Raises:
            ValueError: If doc_id already exists (and overwrite=False), or text is invalid.
        """
        doc_id_str = str(doc_id).strip()
        if not doc_id_str:
            raise ValueError("doc_id cannot be empty.")

        if doc_id_str in self._docs and not overwrite:
            raise ValueError(
                f"Document with ID '{doc_id_str}' already exists in DocumentStore. "
                f"Set overwrite=True to replace it."
            )

        if text is None or not str(text).strip():
            raise ValueError(f"Document '{doc_id_str}' text cannot be empty or pure whitespace.")

        clean_text = str(text).strip()
        meta = metadata.copy() if metadata else {}

        # Vector computation / validation
        if vector is not None:
            # Validate user-supplied vector
            self._validate_vector(vector)
            final_vector = list(vector)
        elif generate_vector:
            # Generate embedding via configured embedder
            try:
                final_vector = self.embedder.embed_text(clean_text)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate embedding for document '{doc_id_str}': {e}"
                ) from e
            self._validate_vector(final_vector)
        else:
            final_vector = None

        # Build Document object
        doc = Document(
            doc_id=doc_id_str,
            text=clean_text,
            metadata=meta,
            vector=final_vector,
        )

        # Update ID Mapper
        if doc_id_str in self._docs and overwrite:
            # Keep existing node_id if possible
            existing_node_id = self.mapper.to_node_id(doc_id_str)
            if node_id is not None and node_id != existing_node_id:
                self.mapper.unregister(doc_id_str)
                self.mapper.register(doc_id_str, node_id=node_id)
        else:
            self.mapper.register(doc_id_str, node_id=node_id)

        self._docs[doc_id_str] = doc
        return doc

    def add_documents(
        self,
        docs: Sequence[Document | dict[str, Any]],
        overwrite: bool = False,
        generate_vectors: bool = True,
    ) -> list[str]:
        """
        Batch add multiple documents to the store.

        Optimizes vector generation by batching texts through `embed_batch` when applicable.

        Args:
            docs: Sequence of Document objects or dictionaries with keys 'doc_id', 'text', 'metadata', etc.
            overwrite: Whether to overwrite existing documents.
            generate_vectors: Whether to generate vectors for documents lacking one.

        Returns:
            List of successfully added doc_id strings.
        """
        added_ids: list[str] = []
        pending_to_embed: list[tuple[str, str, dict[str, Any], Any | None]] = []

        for item in docs:
            if isinstance(item, Document):
                d_id = item.doc_id
                t = item.text
                m = item.metadata
                v = item.vector
                n_id = None
            elif isinstance(item, dict):
                d_id = str(item.get("doc_id") or item.get("id") or "").strip()
                t = str(item.get("text") or item.get("body") or item.get("content") or "").strip()
                m = item.get("metadata") or {k: v for k, v in item.items() if k not in ("doc_id", "id", "text", "body", "content", "vector", "node_id")}
                v = item.get("vector")
                n_id = item.get("node_id")
            else:
                raise TypeError(f"Unsupported document item type: {type(item)}")

            if not d_id:
                raise ValueError("Batch document item missing 'doc_id'.")

            if v is not None:
                # Vector already supplied, add directly
                self.add_document(
                    doc_id=d_id,
                    text=t,
                    metadata=m,
                    vector=v,
                    node_id=n_id,
                    overwrite=overwrite,
                    generate_vector=False,
                )
                added_ids.append(d_id)
            elif generate_vectors:
                pending_to_embed.append((d_id, t, m, n_id))
            else:
                self.add_document(
                    doc_id=d_id,
                    text=t,
                    metadata=m,
                    vector=None,
                    node_id=n_id,
                    overwrite=overwrite,
                    generate_vector=False,
                )
                added_ids.append(d_id)

        # Batch embed pending documents
        if pending_to_embed:
            texts = [item[1] for item in pending_to_embed]
            vectors = self.embedder.embed_batch(texts)

            for (d_id, t, m, n_id), vec in zip(pending_to_embed, vectors):
                self.add_document(
                    doc_id=d_id,
                    text=t,
                    metadata=m,
                    vector=vec,
                    node_id=n_id,
                    overwrite=overwrite,
                    generate_vector=False,
                )
                added_ids.append(d_id)

        return added_ids

    def get_document(self, doc_id: str) -> Document | None:
        """
        Retrieve document object by Document ID.

        Time Complexity: O(1) average.

        Args:
            doc_id: Unique string document identifier.

        Returns:
            Document instance if found, else None.
        """
        return self._docs.get(str(doc_id).strip())

    def get_document_by_node_id(self, node_id: Any) -> Document | None:
        """
        Retrieve document object using its HNSW graph Node ID.

        Time Complexity: O(1) average.

        Args:
            node_id: Node ID returned from HNSW search.

        Returns:
            Document instance if found, else None.
        """
        try:
            doc_id = self.mapper.to_doc_id(node_id)
            return self.get_document(doc_id)
        except KeyError:
            return None

    def get_text(self, doc_id: str) -> str | None:
        """Get raw text content for a document ID."""
        doc = self.get_document(doc_id)
        return doc.text if doc else None

    def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata dictionary for a document ID."""
        doc = self.get_document(doc_id)
        return doc.metadata if doc else None

    def get_vector(self, doc_id: str) -> list[float] | None:
        """Get numerical vector for a document ID."""
        doc = self.get_document(doc_id)
        return doc.vector if doc else None

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove document and its ID mapping from the store.

        Time Complexity: O(1) average.

        Args:
            doc_id: Document ID to remove.

        Returns:
            True if removed, False if document did not exist.
        """
        doc_id_str = str(doc_id).strip()
        if doc_id_str in self._docs:
            del self._docs[doc_id_str]
            self.mapper.unregister(doc_id_str)
            return True
        return False

    def all_documents(self) -> list[Document]:
        """Return a list of all Document objects in the store."""
        return list(self._docs.values())

    def all_vectors(self) -> dict[str, list[float]]:
        """
        Return a mapping from doc_id to vector for all documents with vectors.

        Returns:
            Dictionary of {doc_id: vector}.
        """
        return {
            doc_id: doc.vector
            for doc_id, doc in self._docs.items()
            if doc.vector is not None
        }

    def embed_query(self, query_text: str) -> list[float]:
        """
        Encode a user search query into a vector matching the store's dimension.

        Designed for Teammate 4 (Search Interface) and Teammate 3 (Benchmarking).

        Args:
            query_text: Raw query text.

        Returns:
            Numerical vector of length `self.dim`.
        """
        if not query_text or not str(query_text).strip():
            raise ValueError("Query text cannot be empty or whitespace.")
        return self.embedder.embed_text(str(query_text).strip())

    def populate_hnsw_index(self, index: Any) -> int:
        """
        Populate an HNSWIndex instance with all valid vectors in this store.

        Connects Module 2 (Document Store) with Module 1 (HNSW Core).

        Args:
            index: An HNSWIndex instance from hnsw.index.

        Returns:
            Number of vectors successfully inserted into the index.

        Raises:
            ValueError: If index dimension does not match store dimension.
        """
        if hasattr(index, "dim") and index.dim != self.dim:
            raise ValueError(
                f"Dimension mismatch between HNSW index (dim={index.dim}) "
                f"and DocumentStore (dim={self.dim})."
            )

        inserted_count = 0
        for doc_id, doc in self._docs.items():
            if doc.vector is None:
                continue
            node_id = self.mapper.to_node_id(doc_id)
            index.add_item(node_id, doc.vector)
            inserted_count += 1

        return inserted_count

    def _validate_vector(self, vector: list[float]) -> None:
        """Validate dimension and finite numbers."""
        if len(vector) != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}."
            )
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Vector coordinate {i} must be numeric, got {type(val)}.")
            import math
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector coordinate {i} is invalid (NaN or Infinite).")

    def save(self, filepath: str) -> None:
        """
        Persist the DocumentStore state to a JSON file.

        Args:
            filepath: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        payload = {
            "schema_version": "1.0",
            "dim": self.dim,
            "embedder_type": self.embedder.embedding_type,
            "id_mapper": self.mapper.to_dict(),
            "documents": [doc.to_dict() for doc in self._docs.values()],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(
        cls,
        filepath: str,
        embedder: BaseEmbedder | None = None,
    ) -> "DocumentStore":
        """
        Load a DocumentStore from a persisted JSON file.

        Args:
            filepath: Source file path.
            embedder: Optional embedder to attach. If None, instantiates a DeterministicHashEmbedder
                      matching the saved dimension.

        Returns:
            Reconstituted DocumentStore instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_dim = data.get("dim", 64)
        if embedder is None:
            embedder = DeterministicHashEmbedder(dim=saved_dim)
        elif embedder.dim != saved_dim:
            raise ValueError(
                f"Provided embedder dimension {embedder.dim} does not match "
                f"saved store dimension {saved_dim}."
            )

        store = cls(embedder=embedder)

        # Restore mapper
        if "id_mapper" in data:
            store.mapper = IDMapper.from_dict(data["id_mapper"])

        # Restore documents
        raw_docs = data.get("documents", [])
        for item in raw_docs:
            doc = Document.from_dict(item)
            store._docs[doc.doc_id] = doc

        return store
