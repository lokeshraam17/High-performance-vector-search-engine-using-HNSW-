"""
Document Processing, Vector Generation & Storage Package (Module 2).

Provides:
- Document: Dataclass representing searchable documents with metadata and embeddings.
- DocumentStore: In-memory hash-map repository for document retrieval, persistence, and HNSW population.
- IDMapper: Bidirectional mapper between string Document IDs and HNSW Node IDs.
- Embedders:
  - DeterministicHashEmbedder: Zero-dependency feature hashing text vectorizer.
  - TFIDFEmbedder: Pure Python TF-IDF vectorizer with vocabulary building and IDF weighting.
  - SentenceTransformerEmbedder: Pre-trained Transformer dense semantic embeddings.
  - create_embedder: Unified factory function.
- Loaders:
  - load_from_csv: Flexible CSV ingestion with automatic column detection and error policies.
  - load_from_text_file: Single text file loader.
  - load_from_directory: Directory bulk loader.
  - load_from_json: JSON / JSONL loader.
"""

from docproc.document import Document
from docproc.embeddings import (
    BaseEmbedder,
    DeterministicHashEmbedder,
    SentenceTransformerEmbedder,
    TFIDFEmbedder,
    create_embedder,
)
from docproc.loaders import (
    load_from_csv,
    load_from_directory,
    load_from_json,
    load_from_text_file,
)
from docproc.mapper import IDMapper
from docproc.store import DocumentStore

__all__ = [
    "Document",
    "DocumentStore",
    "IDMapper",
    "BaseEmbedder",
    "DeterministicHashEmbedder",
    "TFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "load_from_csv",
    "load_from_text_file",
    "load_from_directory",
    "load_from_json",
]
