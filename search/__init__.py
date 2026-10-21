"""
Search Interface and Integration Module (Module 4).

Provides:
- SearchEngine: Unified orchestrator connecting DocumentStore (Module 2),
  HNSWIndex (Module 1), and ExactKNNBaseline (Module 3).
- Data models: SearchResultItem, SearchResponse, ComparisonResponse.
- Interactive CLI and Web user search interfaces.
"""

from search.engine import (
    ComparisonResponse,
    SearchEngine,
    SearchResponse,
    SearchResultItem,
)

__all__ = [
    "SearchEngine",
    "SearchResultItem",
    "SearchResponse",
    "ComparisonResponse",
]
