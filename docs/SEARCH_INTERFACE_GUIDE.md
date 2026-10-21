# Search Interface, Integration & End-to-End Testing Guide (Module 4)

Welcome to **Module 4 (Search Interface, System Integration, and End-to-End Testing)** for our **"High-Performance Search Engine Using HNSW"** DSA project.

This document details the architecture, integration interfaces, user interface design, error handling, empirical demonstration workflows, and presentation checklist for our team.

---

## 1. Module Overview & Responsibilities

Module 4 serves as the user-facing application layer and the unified integration orchestrator connecting all modules:
1. **Module 1 (HNSW Core Algorithm):** Consumes `HNSWIndex` for approximate nearest-neighbor graph search via multi-layer geometric routing.
2. **Module 2 (Document Processing & Vector Storage):** Consumes `DocumentStore`, `load_from_csv`, and `embed_query` to convert user text queries into dense normalized vectors and retrieve rich document metadata (title, category, snippet).
3. **Module 3 (Benchmarking & Ground Truth Oracle):** Integrates `ExactKNNBaseline` to evaluate live Recall@K accuracy, empirical speedup factors, and distance computation reduction ratios side-by-side with HNSW.
4. **Interactive Interfaces:**
   - **Modern Web Application (`search/web.py` & `search/static/index.html`):** Built with Python standard library `http.server` (zero external dependencies), featuring a responsive dark mode, glassmorphism aesthetics, live Top-K sliders, query chips, graph hierarchy visualizer, and side-by-side ground truth comparisons.
   - **Command-Line Interface (`search/cli.py`):** Interactive REPL and one-shot CLI with ANSI color formatting.
5. **End-to-End Testing (`tests/test_search.py`):** Comprehensive unit and integration test suite bringing repository coverage to **90 passing tests**.

---

## 2. End-to-End Integration Architecture

```text
User Text Query (e.g. "balanced binary search tree rotations")
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       SearchEngine Orchestrator              │
        │           (search/engine.py)                 │
        └──────────────────────┬───────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [Module 2: DocumentStore]             [Embedding Strategy]
 • embed_query(query_text)             • DeterministicHash / TF-IDF
 • Normalization ||q||_2 = 1.0         • Matches ingested doc vectors
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                        Query Vector q
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [Module 1: HNSW Graph Search]         [Module 3: Exact KNN Baseline]
 • Greedy Highway Navigation (L1..Lmax) • Exhaustive Linear Scan O(N*d)
 • Layer 0 Beam Search (efSearch)      • Exact Ground-Truth Oracle
 • Latency: O(log N * d)               • Latency: O(N * d)
            │                                     │
            ▼                                     ▼
  Top-K (Node ID, Distances)            Top-K (Node ID, Distances)
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
           [Metadata Resolution & Similarity Scoring]
           • mapper.to_doc_id(node_id)
           • store.get_document(doc_id) -> title, category, snippet
           • Cosine similarity = max(0.0, 1.0 - distance)
           • Recall@K = |HNSW Top-K ∩ Exact Top-K| / K
                               │
                               ▼
           [Presentation Layer (Module 4)]
           ┌──────────────────────┬──────────────────────┐
           │ Web UI (index.html)  │   CLI (search/cli)   │
           └──────────────────────┴──────────────────────┘
```

---

## 3. Time & Space Complexity Analysis

| Operation | Time Complexity | Auxiliary Space Complexity | Algorithmic Role |
| :--- | :--- | :--- | :--- |
| **Query Embedding (`store.embed_query`)** | $O(T)$ | $O(d)$ | Projects query text with $T$ tokens into $d$-dimensional space. |
| **HNSW Approximate Search (`engine.search`)** | $O(\log N \cdot d + ef_{search} \cdot M_0 \cdot d)$ | $O(ef_{search} + K)$ | Hierarchical skip-list routing down to Layer 0 beam search. |
| **Exact Baseline Search (`engine.search_baseline`)** | $O(N \cdot d + N \log K)$ | $O(K)$ | Linear scan oracle for ground truth. |
| **Recall@K Accuracy Evaluation** | $O(K)$ | $O(K)$ | Hash set intersection between HNSW and Oracle IDs. |
| **Result Formatting & Metadata Retrieval** | $O(K)$ | $O(K)$ | $O(1)$ dictionary lookups for titles and snippets. |
| **Overall Search Engine Memory** | — | $O(N \cdot (M \cdot d + L))$ | Nodes, vectors, text buffers, and bidirectional ID maps. |

---

## 4. Class Structure & Public APIs

### 4.1 `SearchEngine` (`search/engine.py`)

```python
from search.engine import SearchEngine

# 1. Initialize engine
engine = SearchEngine(
    dim=64,                  # Embedding dimension
    metric="cosine",         # 'cosine', 'euclidean', 'dot_product'
    M=16,                    # Max connections per node per layer
    ef_construction=64,      # Dynamic candidate list for graph building
    ef_search=32,            # Default Layer 0 beam width during search
    seed=42,                 # Deterministic seed
)

# 2. Ingest documents and construct HNSW graph + Exact Baseline
num_indexed = engine.load_dataset("data/sample_documents.csv")
print(f"Indexed {num_indexed} documents.")

# 3. Perform approximate search
response = engine.search("balanced binary tree", k=5, ef_search=32)
for item in response.results:
    print(f"#{item.rank} {item.title} (Score: {item.similarity*100:.1f}%, Dist: {item.distance:.4f})")
    print(f"    Snippet: {item.snippet}")

# 4. Compare side-by-side with ground-truth oracle
comparison = engine.compare_search("dijkstra shortest path", k=5)
print(f"Recall@5: {comparison.recall_percent:.1f}%")
print(f"HNSW Latency: {comparison.hnsw.latency_ms:.3f} ms vs Exact: {comparison.baseline.latency_ms:.3f} ms")
print(f"Speedup: {comparison.speedup:.2f}x faster")
```

### 4.2 Web API Endpoints (`search/web.py`)

The server runs on standard Python (`http.server`):
- `GET /`: Serves the single-page application dashboard.
- `GET /api/status`: Returns system status, document counts, and HNSW graph diagnostics.
- `POST /api/load`: Ingests dataset (defaults to `data/sample_documents.csv`) and rebuilds index.
- `POST /api/search`:
  - Request: `{"query": "attention mechanisms", "k": 5, "ef_search": 32, "compare_baseline": true}`
  - Response: Returns ranked documents, similarity percentages, snippets, latencies, distance counts, and Recall@K.

---

## 5. How to Run the System

### Single Run Command (Default Web Interface):
```bash
python app.py
```
*Automatically indexes `data/sample_documents.csv`, starts the HTTP server at `http://127.0.0.1:8000/`, and opens your default web browser.*

### Interactive Command-Line Interface (CLI):
```bash
python app.py --cli
```
*Or directly:*
```bash
python -m search.cli
```

### One-Shot CLI Search with Comparison:
```bash
python -m search.cli --query "graph algorithms dijkstra" -k 5 --compare
```

### End-to-End Demonstration:
```bash
python examples/demo_search.py
```

### Run All Unit and Integration Tests:
```bash
python -m unittest discover -s tests -v
```
*All 90 unit tests pass across all four modules.*

---

## 6. Live Demonstration Workflow for Project Defense

Follow this sequential workflow during your project demonstration:

1. **Step 1: Introduction & Architecture Walkthrough**
   - Open `http://127.0.0.1:8000/`.
   - Point out the **Status Badge**: `Active (25 Docs)` and the **Stats HUD**:
     - 25 Indexed Documents
     - 64-d Cosine Metric
     - HNSW Skip-List Graph Architecture ($M=16, ef_c=64$).
   - Click the **"Graph"** button in the header to show the multi-layer hierarchy modal (Layer 0 ground nodes vs Layer 1 highway nodes).

2. **Step 2: Sub-Millisecond Semantic Retrieval**
   - Click the sample query chip: `"Binary Search Trees"`.
   - Highlight the **sub-millisecond latency** ($\approx 0.20\text{ ms}$).
   - Show how the top result is `"Red-Black Self-Balancing Trees"` and `"Binary Search Trees"`, with similarity scores, category tags, and preview snippets.

3. **Step 3: Demonstrating the Ground-Truth Oracle (Module 3 Integration)**
   - Turn ON the toggle: **"Compare with Exact Brute-Force (Module 3)"**.
   - Type query: `"dijkstra shortest path algorithm priority queue"` and press Enter.
   - Point to the **Comparison Panel**:
     - **Recall@5: 100.0%**: Proves that HNSW retrieves the optimal nearest neighbors identical to an exhaustive brute-force search.
     - **Distance Computations**: HNSW routes hierarchically rather than scanning all vectors.

4. **Step 4: Demonstrating Parameter Sensitivity ($efSearch$)**
   - Adjust the **efSearch** slider between 5 and 50.
   - Explain how lower $efSearch$ reduces search time for large datasets, while higher $efSearch$ expands the candidate beam for maximum recall.

5. **Step 5: Edge Case & Fault Tolerance Handling**
   - Clear the search box and press Search $\to$ Notice clean validation alert.
   - Type an out-of-domain query $\to$ Notice graceful ranking and score degradation.

---

## 7. College Project Defense Checklist

Use this checklist to ensure all project rubric requirements are satisfied:

- [x] **Core Algorithm Scratch Implementation:** Pure Python HNSW implementation without external C++ or third-party ANN libraries (`hnsw/`).
- [x] **Distance Metrics:** Euclidean, Cosine, and Dot Product metrics implemented and validated (`hnsw/distance.py`).
- [x] **Document Ingestion & Embeddings:** CSV loading, deterministic feature hashing (SHA-256), and pure Python TF-IDF (`docproc/`).
- [x] **Ground-Truth Baseline:** Independent linear-scan oracle (`benchmark/baseline.py`).
- [x] **Performance Metrics:** Statistical latency distributions (mean, median, p95), QPS, heap tracking via `tracemalloc`, and Recall@K (`benchmark/metrics.py`).
- [x] **Visual Search Interface:** Polished, responsive Web UI and interactive CLI (`search/static/index.html`, `search/cli.py`).
- [x] **Side-by-Side Comparison:** Live ground-truth accuracy and speedup evaluation in both Web and CLI.
- [x] **Zero External Dependency Setup:** Built using Python standard libraries so any college evaluator can run it with `python app.py`.
- [x] **Comprehensive Testing:** 90 passing unit and integration tests (`tests/`).
- [x] **Complete Team Documentation:** Handoff guides for all 4 team modules in `docs/`.
