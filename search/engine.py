"""
Search Engine Integration Orchestrator.

Integrates:
- Module 1 (HNSW Core Algorithm): Multi-layer graph index for approximate search.
- Module 2 (Document Processing): Ingestion, tokenization, text-to-vector embeddings, and store.
- Module 3 (Benchmarking & Ground Truth): Exact brute-force baseline for recall comparison.

Provides a unified, thread-safe interface for natural language search queries,
real-time performance profiling, and side-by-side ground truth comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.baseline import ExactKNNBaseline
from benchmark.metrics import compute_recall_at_k
from docproc.document import Document
from docproc.embeddings import create_embedder
from docproc.loaders import load_from_csv, load_from_json, load_from_text_file
from docproc.store import DocumentStore
from hnsw.index import HNSWIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResultItem:
    """Represents a single retrieved document in search results."""
    rank: int
    doc_id: str
    title: str
    category: str
    snippet: str
    distance: float
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result item to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SearchResponse:
    """Represents the complete response from a search query."""
    query: str
    k: int
    ef_search: int
    latency_ms: float
    distance_computations: int
    total_indexed: int
    mode: str
    results: List[SearchResultItem]

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "k": self.k,
            "ef_search": self.ef_search,
            "latency_ms": round(self.latency_ms, 3),
            "distance_computations": self.distance_computations,
            "total_indexed": self.total_indexed,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ComparisonResponse:
    """Represents side-by-side comparison between HNSW and Exact Brute-Force."""
    query: str
    k: int
    recall_at_k: float
    speedup: float
    distance_reduction_ratio: float
    hnsw: SearchResponse
    baseline: SearchResponse

    @property
    def recall_percent(self) -> float:
        """Recall@K expressed as a percentage in [0.0, 100.0]."""
        return self.recall_at_k * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "k": self.k,
            "recall_at_k": round(self.recall_at_k, 4),
            "recall_percent": round(self.recall_at_k * 100.0, 1),
            "speedup": round(self.speedup, 2),
            "distance_reduction_ratio": round(self.distance_reduction_ratio, 2),
            "hnsw": self.hnsw.to_dict(),
            "baseline": self.baseline.to_dict(),
        }


class SearchEngine:
    """
    Unified High-Performance Search Engine using HNSW.
    
    Coordinates document loading, vector generation, HNSW graph construction,
    query vectorization, approximate nearest-neighbor search, and optional
    ground-truth accuracy comparison.
    """

    def __init__(
        self,
        dim: int = 64,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        embedder_type: str = "hash",
        id_mode: str = "identity",
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the SearchEngine configuration.

        Args:
            dim: Dimension of vector embeddings (default 64).
            metric: Distance metric ('cosine', 'euclidean', 'dot_product').
            M: Max connections per node per layer in HNSW (default 16).
            ef_construction: Candidate list size during graph build (default 64).
            ef_search: Default candidate list size during search (default 32).
            embedder_type: Embedding strategy ('hash', 'tfidf', 'sentence-transformers').
            id_mode: ID mapping mode ('identity' or 'integer').
            seed: Deterministic random seed for reproducibility.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if M < 2:
            raise ValueError(f"M must be at least 2, got {M}.")

        self.dim = dim
        self.metric = metric.lower()
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.embedder_type = embedder_type
        self.id_mode = id_mode
        self.seed = seed

        # Initialize embedder and document store (Module 2)
        embedder = create_embedder(model_type=embedder_type, dim=dim)
        self.store = DocumentStore(embedder=embedder, id_mode=id_mode, dim=dim)

        # HNSW graph index (Module 1)
        self.index: Optional[HNSWIndex] = None

        # Exact linear scan ground-truth baseline (Module 3)
        self.baseline: Optional[ExactKNNBaseline] = None

        # Ingestion metrics
        self.index_build_time_ms: float = 0.0
        self.last_loaded_path: Optional[str] = None
        self._is_indexed: bool = False

    @property
    def is_indexed(self) -> bool:
        """Return True if documents are loaded and HNSW graph is ready for search."""
        return self._is_indexed and self.index is not None and len(self.index.nodes) > 0

    @property
    def total_documents(self) -> int:
        """Return total number of documents indexed."""
        return self.store.total_documents

    def load_dataset(
        self,
        filepath: str,
        on_error: str = "skip",
    ) -> int:
        """
        Load documents from a file (CSV, JSON, or TXT) and build both
        the HNSW index and ExactKNNBaseline.

        Args:
            filepath: Path to the dataset file.
            on_error: Error policy ('skip' or 'raise').

        Returns:
            Number of documents indexed.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If file is empty or contains no valid documents.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            docs = load_from_csv(filepath, on_error=on_error)
        elif ext in (".json", ".jsonl"):
            docs = load_from_json(filepath)
        elif ext == ".txt":
            docs = [load_from_text_file(filepath)]
        else:
            # Fallback to CSV loader with auto-detection
            docs = load_from_csv(filepath, on_error=on_error)

        if not docs:
            raise ValueError(f"No valid documents found in {filepath}")

        return self.index_documents(docs, source_path=filepath)

    def index_documents(
        self,
        documents: Sequence[Document],
        source_path: Optional[str] = None,
    ) -> int:
        """
        Ingest Document objects into DocumentStore and construct the HNSW graph.

        Args:
            documents: Sequence of Document objects.
            source_path: Optional path for status reporting.

        Returns:
            Number of indexed documents.
        """
        if not documents:
            raise ValueError("Document sequence cannot be empty.")

        # Step 1: Add to DocumentStore (Module 2) - generates embeddings if missing
        t_start = time.perf_counter()
        self.store.add_documents(documents)

        # Step 2: Initialize HNSW Index (Module 1)
        self.index = HNSWIndex(
            dim=self.dim,
            metric=self.metric,
            M=self.M,
            M0=2 * self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            seed=self.seed,
        )

        # Step 3: Populate HNSW graph
        self.store.populate_hnsw_index(self.index)

        # Step 4: Populate Exact Baseline (Module 3) for ground truth comparison
        self.baseline = ExactKNNBaseline(dim=self.dim, metric=self.metric)
        all_vecs = self.store.all_vectors()
        for doc_id, vec in all_vecs.items():
            node_id = self.store.mapper.to_node_id(doc_id)
            self.baseline.add_item(node_id, vec)

        self.index_build_time_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_loaded_path = source_path
        self._is_indexed = True

        logger.info(
            f"Successfully indexed {self.total_documents} documents in {self.index_build_time_ms:.2f} ms."
        )
        return self.total_documents

    def search(
        self,
        query: str,
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> SearchResponse:
        """
        Perform approximate nearest-neighbor search using the HNSW graph index.

        Args:
            query: Natural language query string.
            k: Number of nearest documents to return (k >= 1).
            ef_search: Optional candidate beam width override at Layer 0.

        Returns:
            SearchResponse containing ranked SearchResultItems and latency stats.
            
        Raises:
            RuntimeError: If search engine is not indexed.
            ValueError: If query is empty or k is invalid.
        """
        self._validate_search_prerequisites(query, k)

        # 1. Transform query text to numerical embedding using same pipeline
        query_vector = self.store.embed_query(query)

        effective_ef = ef_search if ef_search is not None else self.ef_search
        effective_ef = max(k, effective_ef)

        # 2. Reset distance computation tracker for per-query profiling
        self.index.reset_distance_computations()

        # 3. Search HNSW graph (Module 1)
        t0 = time.perf_counter()
        raw_results: List[Tuple[Any, float]] = self.index.knn_search(
            query=query_vector,
            k=k,
            ef_search=effective_ef,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        dist_comps = self.index.total_distance_computations

        # 4. Map HNSW Node IDs back to Document metadata (Module 2)
        results = self._format_results(raw_results)

        return SearchResponse(
            query=query,
            k=k,
            ef_search=effective_ef,
            latency_ms=latency_ms,
            distance_computations=dist_comps,
            total_indexed=self.total_documents,
            mode="HNSW Approximate Search",
            results=results,
        )

    def search_baseline(
        self,
        query: str,
        k: int = 5,
    ) -> SearchResponse:
        """
        Perform exact nearest-neighbor search using the Brute-Force Baseline.
        Used as ground-truth oracle to measure HNSW Recall@K and speedup.

        Args:
            query: Natural language query string.
            k: Number of nearest documents to return (k >= 1).

        Returns:
            SearchResponse containing exact nearest documents.
        """
        self._validate_search_prerequisites(query, k)

        # 1. Transform query text to numerical embedding
        query_vector = self.store.embed_query(query)

        # 2. Reset baseline distance computations
        self.baseline.reset_distance_computations()

        # 3. Exact linear scan over all vectors (Module 3)
        t0 = time.perf_counter()
        raw_results: List[Tuple[Any, float]] = self.baseline.knn_search(
            query=query_vector,
            k=k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        dist_comps = self.baseline.total_distance_computations

        # 4. Format results
        results = self._format_results(raw_results)

        return SearchResponse(
            query=query,
            k=k,
            ef_search=len(self.baseline),
            latency_ms=latency_ms,
            distance_computations=dist_comps,
            total_indexed=self.total_documents,
            mode="Exact Brute-Force (Ground Truth)",
            results=results,
        )

    def compare_search(
        self,
        query: str,
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> ComparisonResponse:
        """
        Execute both HNSW approximate search and exact brute-force search,
        computing empirical Recall@K, latency speedup, and distance computation reduction.

        Args:
            query: Query text.
            k: Number of nearest items.
            ef_search: Optional ef_search parameter for HNSW.

        Returns:
            ComparisonResponse with side-by-side results and accuracy metrics.
        """
        hnsw_resp = self.search(query=query, k=k, ef_search=ef_search)
        baseline_resp = self.search_baseline(query=query, k=k)

        # Extract node_ids to compute Recall@K (Module 3 metrics)
        hnsw_pairs = [(r.doc_id, r.distance) for r in hnsw_resp.results]
        baseline_pairs = [(r.doc_id, r.distance) for r in baseline_resp.results]

        recall = compute_recall_at_k(hnsw_pairs, baseline_pairs, k=k)

        # Latency speedup: baseline_latency / hnsw_latency
        speedup = (
            (baseline_resp.latency_ms / max(1e-5, hnsw_resp.latency_ms))
            if hnsw_resp.latency_ms > 0
            else 1.0
        )

        # Distance computation reduction
        reduction = (
            (baseline_resp.distance_computations / max(1, hnsw_resp.distance_computations))
            if hnsw_resp.distance_computations > 0
            else 1.0
        )

        return ComparisonResponse(
            query=query,
            k=k,
            recall_at_k=recall,
            speedup=speedup,
            distance_reduction_ratio=reduction,
            hnsw=hnsw_resp,
            baseline=baseline_resp,
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Return comprehensive diagnostic information about current index state,
        graph architecture, and embedding settings.
        """
        graph_stats = self.index.get_stats() if self.index else {}
        return {
            "is_indexed": self.is_indexed,
            "total_documents": self.total_documents,
            "embedding_dimension": self.dim,
            "distance_metric": self.metric,
            "embedder_type": self.embedder_type,
            "id_mode": self.id_mode,
            "hnsw_config": {
                "M": self.M,
                "M0": 2 * self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            },
            "index_build_time_ms": round(self.index_build_time_ms, 2),
            "source_dataset": self.last_loaded_path,
            "graph_diagnostics": graph_stats,
        }

    def _validate_search_prerequisites(self, query: str, k: int) -> None:
        """Validate search inputs and index readiness."""
        if not self.is_indexed:
            raise RuntimeError(
                "Search engine is not indexed. Please call load_dataset() or index_documents() first."
            )
        if query is None or not str(query).strip():
            raise ValueError("Query text cannot be empty or whitespace.")
        if k <= 0:
            raise ValueError(f"Top-K must be a positive integer >= 1, got {k}.")

    def _format_results(
        self,
        raw_results: List[Tuple[Any, float]],
    ) -> List[SearchResultItem]:
        """
        Transform (node_id, distance) tuples into rich SearchResultItem objects.
        
        Calculates similarity score:
        - For cosine distance in [0, 2]: similarity = max(0.0, 1.0 - distance)
        - For euclidean distance: similarity = 1.0 / (1.0 + distance)
        """
        items: List[SearchResultItem] = []
        for rank, (node_id, distance) in enumerate(raw_results, start=1):
            doc = self.store.get_document_by_node_id(node_id)
            if doc is None:
                # Fallback if document lookup fails
                doc_id = str(node_id)
                title = f"Document {node_id}"
                category = "General"
                snippet = ""
                meta: Dict[str, Any] = {}
            else:
                doc_id = doc.doc_id
                title = doc.metadata.get("title", f"Document {doc_id}")
                category = doc.metadata.get("category", "Uncategorized")
                snippet = doc.snippet(max_length=160)
                meta = doc.metadata

            # Compute normalized similarity score in [0.0, 1.0]
            if self.metric == "cosine":
                similarity = max(0.0, min(1.0, 1.0 - distance))
            elif self.metric == "euclidean":
                similarity = 1.0 / (1.0 + distance)
            else:
                similarity = max(0.0, 1.0 / (1.0 + abs(distance)))

            items.append(
                SearchResultItem(
                    rank=rank,
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    snippet=snippet,
                    distance=round(distance, 5),
                    similarity=round(similarity, 4),
                    metadata=meta,
                )
            )
        return items
