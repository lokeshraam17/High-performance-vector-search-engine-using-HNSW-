"""
Data structures for nodes in the Hierarchical Navigable Small World (HNSW) graph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HNSWNode:
    """
    Represents an element stored inside the HNSW index.
    
    Attributes:
        node_id: Unique identifier for the element (e.g. document ID or integer index).
        vector: Dense numerical representation of the item.
        level: Maximum layer at which this node is present. Layer 0 is the bottom layer,
               containing all nodes. Higher layers contain exponentially fewer nodes.
        neighbors: Dictionary where key is layer index (int) and value is the list
                   of connected neighbor node IDs at that layer.
    """
    node_id: Any
    vector: List[float]
    level: int
    neighbors: Dict[int, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize neighbor lists for all layers from 0 up to self.level
        for l in range(self.level + 1):
            if l not in self.neighbors:
                self.neighbors[l] = []

    def add_neighbor(self, layer: int, neighbor_id: Any) -> None:
        """Add a neighbor to the node at a specific layer if not already present."""
        if layer not in self.neighbors:
            self.neighbors[layer] = []
        if neighbor_id not in self.neighbors[layer]:
            self.neighbors[layer].append(neighbor_id)

    def remove_neighbor(self, layer: int, neighbor_id: Any) -> None:
        """Remove a neighbor from the specified layer."""
        if layer in self.neighbors and neighbor_id in self.neighbors[layer]:
            self.neighbors[layer].remove(neighbor_id)

    def get_neighbors(self, layer: int) -> List[Any]:
        """Return list of neighbor IDs at given layer."""
        return self.neighbors.get(layer, [])

    def set_neighbors(self, layer: int, neighbors: List[Any]) -> None:
        """Set the neighbor list for a given layer."""
        self.neighbors[layer] = list(neighbors)

    def degree(self, layer: int) -> int:
        """Return number of connections at given layer."""
        return len(self.neighbors.get(layer, []))
