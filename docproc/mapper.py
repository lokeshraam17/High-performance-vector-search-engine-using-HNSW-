"""
Bidirectional Identifier Mapper.

Maintains a 1-to-1 mapping between human-readable Document IDs (e.g. "doc_001", "wiki_article_42")
and HNSW Graph Node IDs (string or integer).
"""

from __future__ import annotations

from typing import Any


class IDMapper:
    """
    Bidirectional mapping between Document IDs and HNSW Node IDs.

    Features:
    - Mode 'identity': node_id is equal to doc_id (standard usage in Python HNSW).
    - Mode 'integer': assigns monotonic integer IDs (0, 1, 2, ...) for HNSW implementations
      requiring sequential numeric indexing.
    - O(1) average time complexity for both forward and reverse lookups.

    Time Complexity:
        - register: O(1) average.
        - to_node_id: O(1) average.
        - to_doc_id: O(1) average.
    Space Complexity:
        - O(N) where N is the number of mapped elements.
    """

    def __init__(self, mode: str = "identity") -> None:
        """
        Initialize IDMapper.

        Args:
            mode: Mapping strategy ('identity' or 'integer').
        """
        if mode not in ("identity", "integer"):
            raise ValueError(f"Unknown mapper mode '{mode}'. Choose 'identity' or 'integer'.")

        self.mode: str = mode
        self._doc_to_node: dict[str, Any] = {}
        self._node_to_doc: dict[Any, str] = {}
        self._next_integer_id: int = 0

    def register(self, doc_id: str, node_id: Any | None = None) -> Any:
        """
        Register a new document ID and determine/assign its corresponding HNSW node ID.

        Args:
            doc_id: String document identifier.
            node_id: Optional explicit node ID. If omitted, determined by mapper mode.

        Returns:
            The assigned HNSW node_id.

        Raises:
            ValueError: If doc_id already exists with a conflicting node_id, or node_id is taken.
        """
        doc_id = str(doc_id).strip()
        if not doc_id:
            raise ValueError("doc_id cannot be empty.")

        # If already registered, verify consistency
        if doc_id in self._doc_to_node:
            existing_node = self._doc_to_node[doc_id]
            if node_id is not None and node_id != existing_node:
                raise ValueError(
                    f"Document ID '{doc_id}' already registered with node ID '{existing_node}', "
                    f"cannot re-register with '{node_id}'."
                )
            return existing_node

        # Determine node_id
        if node_id is None:
            if self.mode == "identity":
                node_id = doc_id
            elif self.mode == "integer":
                node_id = self._next_integer_id
                self._next_integer_id += 1

        # Check collision on reverse map
        if node_id in self._node_to_doc:
            existing_doc = self._node_to_doc[node_id]
            raise ValueError(
                f"Node ID '{node_id}' is already assigned to document '{existing_doc}'."
            )

        self._doc_to_node[doc_id] = node_id
        self._node_to_doc[node_id] = doc_id
        return node_id

    def to_node_id(self, doc_id: str) -> Any:
        """
        Look up the HNSW node ID for a given document ID.

        Args:
            doc_id: String document identifier.

        Returns:
            The associated HNSW node ID.

        Raises:
            KeyError: If doc_id has not been registered.
        """
        doc_id = str(doc_id).strip()
        if doc_id not in self._doc_to_node:
            raise KeyError(f"Document ID '{doc_id}' is not registered in IDMapper.")
        return self._doc_to_node[doc_id]

    def to_doc_id(self, node_id: Any) -> str:
        """
        Look up the document ID for a given HNSW node ID.

        Args:
            node_id: The HNSW node ID.

        Returns:
            The associated document ID string.

        Raises:
            KeyError: If node_id has not been registered.
        """
        if node_id not in self._node_to_doc:
            raise KeyError(f"Node ID '{node_id}' is not registered in IDMapper.")
        return self._node_to_doc[node_id]

    def has_doc_id(self, doc_id: str) -> bool:
        """Check whether doc_id is registered."""
        return str(doc_id).strip() in self._doc_to_node

    def has_node_id(self, node_id: Any) -> bool:
        """Check whether node_id is registered."""
        return node_id in self._node_to_doc

    def unregister(self, doc_id: str) -> None:
        """Remove a document ID and its node ID mapping."""
        doc_id = str(doc_id).strip()
        if doc_id in self._doc_to_node:
            node_id = self._doc_to_node.pop(doc_id)
            self._node_to_doc.pop(node_id, None)

    def __len__(self) -> int:
        return len(self._doc_to_node)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapper state to JSON-serializable dictionary."""
        return {
            "mode": self.mode,
            "next_integer_id": self._next_integer_id,
            "doc_to_node": {k: v for k, v in self._doc_to_node.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IDMapper":
        """Reconstruct IDMapper from serialized dictionary."""
        mapper = cls(mode=data.get("mode", "identity"))
        mapper._next_integer_id = data.get("next_integer_id", 0)
        doc_to_node = data.get("doc_to_node", {})
        for doc_id, node_id in doc_to_node.items():
            mapper._doc_to_node[doc_id] = node_id
            mapper._node_to_doc[node_id] = doc_id
        return mapper
