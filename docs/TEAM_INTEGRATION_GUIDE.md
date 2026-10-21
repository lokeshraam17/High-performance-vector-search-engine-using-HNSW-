# HNSW Core Module Integration Guide

Welcome to the **HNSW Core Algorithm** module for our **"High-Performance Search Engine Using HNSW"** DSA project.

This document details the architecture, algorithmic theory, complexity analysis, and concrete instructions for **Teammates 2, 3, and 4** to integrate smoothly.

---

## 1. Algorithm Overview & Mathematical Foundation

### 1.1 How the Hierarchy Reduces Search Work (Skip-List Analogy)
In a standard graph (like NSW - Navigable Small World), greedy routing can easily stall in local minima or require scanning numerous intermediate neighbors, scaling poorly to large datasets ($O(N)$ or $O(N^{1/d})$).

HNSW (Hierarchical Navigable Small World) solves this by organizing nodes into a multi-layered geometric hierarchy, analogous to a multi-dimensional skip-list:
1. **Layer 0 (Bottom Layer):** Contains **all** $N$ nodes connected with local, fine-grained edges.
2. **Layer $l > 0$ (Upper Layers):** Contains exponentially fewer nodes (decaying with factor $1/M^l$). Edges between nodes at higher layers span long Euclidean/metric distances ("highways").

```
Layer 2 (Coarse):      [A] ------------------------------> [G]
                        |                                   |
Layer 1 (Medium):      [A] ------------> [D] ------------> [G]
                        |                 |                 |
Layer 0 (Fine):        [A] <-> [B] <-> [C] <-> [D] <-> [E] <-> [F] <-> [G]
```

**Query Workflow:**
- The query starts at the top layer (`enter_point_id`) and performs a **fast greedy search** ($ef=1$) along long-range highway edges to quickly zoom into the general basin of attraction of the target query vector.
- It steps down layer-by-layer, using the nearest node found in layer $l$ as the entry point for layer $l-1$.
- Only at Layer 0 does it perform a detailed beam search using a candidate pool of size `ef_search`.
- **Result:** Instead of scanning $N$ items linearly, HNSW navigates through $O(\log N)$ layers, performing only $O(\log N)$ distance calculations on average.

---

### 1.2 Why Approximate Search Can Differ from Exact Search
- **Exact Search (Brute Force):** Computes distances between the query and **every single vector** in the dataset. This guarantees 100% recall (optimal nearest neighbors) but takes $O(N \cdot d)$ time, which becomes prohibitively slow for millions of documents.
- **Approximate Search (HNSW):** Restricts the search space to the graph topology. Because edges are established based on greedy local heuristics during insertion, it is possible for a true nearest neighbor to be separated from the current path by a high-distance barrier (a "local minimum") if the beam width (`ef_search`) is too narrow or the degree (`M`) is insufficient.
- **Tuning Trade-off:** By increasing `ef_search`, the algorithm explores more branching paths in Layer 0, closing the gap with exact search. In practice, `ef_search` values between 32 and 128 routinely achieve **98% to 100% Recall@K** while executing in a small fraction of the brute-force time.

---

## 2. Time & Space Complexity Analysis

| Operation | Time Complexity | Auxiliary Space Complexity | Notes |
| :--- | :--- | :--- | :--- |
| **Distance Calculation (Euclidean)** | $O(d)$ | $O(1)$ | $d$ is vector dimensionality. |
| **Node Insertion (`add_item`)** | $O(M \cdot \log N \cdot d)$ | $O(ef_{construction})$ | Navigates upper layers in $O(\log N)$ steps, connects at lower layers. |
| **Query Search (`knn_search`)** | $O(\log N \cdot d)$ | $O(ef_{search} + k)$ | $O(\log N)$ greedy hops in upper layers + bounded beam search at layer 0. |
| **Brute-Force Search (`brute_force_knn`)** | $O(N \cdot d + N \log k)$ | $O(N)$ | Linear scan baseline used for benchmarking. |
| **Overall Graph Storage** | — | $O(N \cdot (M \cdot d))$ | Linear in the number of elements: each node stores its vector and at most $M$ neighbors per layer ($M_0$ at layer 0). |

---

## 3. Class Structure & Public API

### `HNSWIndex`
```python
from hnsw import HNSWIndex

index = HNSWIndex(
    dim=128,                 # Dimension of embeddings (e.g. 128, 384, 768)
    metric="euclidean",       # 'euclidean', 'cosine', or 'dot_product'
    M=16,                    # Max connections per node per layer (l > 0)
    M0=32,                   # Max connections at layer 0 (defaults to 2 * M)
    ef_construction=100,     # Candidate list size during insertion
    ef_search=50,            # Candidate list size during search
    seed=42                  # Optional seed for deterministic behavior
)
```

### Core Methods:
1. `index.add_item(node_id: str | int, vector: list[float]) -> None`
   - Inserts a document vector.
   - Raises `ValueError` if `vector` dimension does not match `dim` or if `node_id` already exists.
2. `index.knn_search(query: list[float], k: int = 5, ef_search: int | None = None) -> list[tuple[node_id, float]]`
   - Returns top $k$ nearest neighbors formatted as `[(node_id, distance), ...]`, ordered from closest to furthest.
3. `index.brute_force_knn(query: list[float], k: int = 5) -> list[tuple[node_id, float]]`
   - Exact linear scan ground truth for benchmarking.
4. `index.get_stats() -> dict`
   - Returns structural statistics: `total_nodes`, `max_level`, `nodes_per_level`, `avg_degree_per_level`, `total_distance_computations`.
5. `index.reset_distance_computations() -> None`
   - Resets the distance counter to 0 (crucial for per-query benchmark metrics).
6. `index.save(filepath: str) -> None` / `HNSWIndex.load(filepath: str) -> HNSWIndex`
   - Saves/loads the entire graph to/from a JSON file.

---

## 4. Teammate Handoff & Integration Instructions

### For Teammate 2: Document Processing, Vector Generation & Storage
- **Task:** Process text documents, generate vector embeddings (e.g. TF-IDF, Word2Vec, or sentence-transformers), and populate the index.
- **Integration snippet:**
  ```python
  from hnsw import HNSWIndex

  # Initialize index with your vector model's output dimension
  index = HNSWIndex(dim=384, metric="cosine", M=16, ef_construction=100)

  # Process and insert documents
  for doc in processed_documents:
      # doc["id"] can be int or string (e.g. "doc_001")
      # doc["vector"] must be list of floats matching dim
      index.add_item(doc["id"], doc["vector"])

  # Persist index for the web/search application
  index.save("storage/hnsw_index.json")
  ```
- **Tips:**
  - If using `metric="cosine"`, vectors do not strictly need to be normalized in advance (the distance metric handles norm computation), but pre-normalizing can be useful.
  - Document metadata (title, text, author) should be stored in your own storage dictionary/DB keyed by `doc_id`.

---

### For Teammate 3: Benchmarking, Accuracy Testing & Performance Analysis
- **Task:** Measure indexing speed, search latency (QPS), Recall@K, and distance calculation efficiency across varying parameters ($M$, $efConstruction$, $efSearch$).
- **Integration snippet:**
  ```python
  from hnsw import HNSWIndex

  # Ground truth vs HNSW Recall calculation
  index.reset_distance_computations()
  hnsw_results = index.knn_search(query_vec, k=10, ef_search=64)
  hnsw_comps = index.total_distance_computations

  index.reset_distance_computations()
  exact_results = index.brute_force_knn(query_vec, k=10)
  exact_comps = index.total_distance_computations

  hnsw_ids = {item[0] for item in hnsw_results}
  exact_ids = {item[0] for item in exact_results}
  recall_at_10 = len(hnsw_ids.intersection(exact_ids)) / 10.0

  print(f"Recall@10: {recall_at_10 * 100:.2f}%")
  print(f"HNSW checks: {hnsw_comps} vs Exact checks: {exact_comps}")
  ```

---

### For Teammate 4: Search Interface, Integration & End-to-End Testing
- **Task:** Build the frontend/CLI search interface and connect user search queries to document retrieval.
- **Integration snippet:**
  ```python
  from hnsw import HNSWIndex

  # Load persisted index at server startup
  index = HNSWIndex.load("storage/hnsw_index.json")

  def search_documents(query_text: str, k: int = 5):
      # 1. Convert text to vector via Teammate 2's encoder
      query_vector = text_encoder.transform(query_text)

      # 2. Query HNSW graph
      results = index.knn_search(query_vector, k=k)

      # 3. Retrieve document text using returned IDs
      documents = []
      for doc_id, distance in results:
          doc_data = document_store.get(doc_id)
          documents.append({
              "id": doc_id,
              "distance": distance,
              "title": doc_data["title"],
              "snippet": doc_data["body"][:200]
          })
      return documents
  ```

---

## 5. Verification & Testing

To run the complete test suite:
```bash
python -m unittest discover -s tests -v
```

To run the interactive visual demonstration:
```bash
python examples/demo_hnsw.py
```
