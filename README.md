# High-Performance Search Engine Using HNSW (Hierarchical Navigable Small World)

A Data Structures and Algorithms (DSA) project implementing a high-performance vector search engine from scratch using the Hierarchical Navigable Small World (HNSW) graph algorithm for Approximate Nearest Neighbor (ANN) search.

## Team Structure & Responsibilities
1. **Module 1 (HNSW Core Algorithm - Completed):** HNSW graph construction, hierarchical layer routing, distance metrics, neighbor selection, and top-$K$ approximate search.
2. **Module 2 (Document Processing & Vector Generation):** Text tokenization, chunking, vector embedding generation, and document metadata store.
3. **Module 3 (Benchmarking & Performance Analysis):** QPS, recall calculation vs brute-force ground truth, latency profiling, and parameter sensitivity curves ($M$, $efConstruction$, $efSearch$).
4. **Module 4 (Search Interface & Integration):** CLI/Web user search interface, query pipeline, end-to-end integration, and results display.

---

## Project Structure
```text
.
├── hnsw/                          # Core HNSW algorithm package
│   ├── __init__.py                # Package exports
│   ├── distance.py                # Euclidean, cosine, and dot-product metrics
│   ├── node.py                    # HNSWNode data structure
│   └── index.py                   # HNSWIndex graph construction & search
├── tests/
│   ├── __init__.py
│   └── test_hnsw.py               # 19 comprehensive unit tests
├── examples/
│   └── demo_hnsw.py               # Visual runnable demonstration
├── docs/
│   └── TEAM_INTEGRATION_GUIDE.md  # Detailed API specs and teammate handoff guide
└── README.md
```

---

## Quickstart

### 1. Run Unit Tests
```bash
python -m unittest discover -s tests -v
```
All 19 unit tests validate:
- Euclidean, Cosine, and Dot-Product distances and validation.
- Node creation, layer initialization, and neighbor manipulation.
- Insertion and multi-layer greedy routing.
- Recall comparison against exact brute-force search.
- Edge cases (empty graph, duplicate IDs, dimension mismatch, invalid $k$).
- JSON serialization and deserialization.

### 2. Run the Demonstration
```bash
python examples/demo_hnsw.py
```
This runs a 300-vector visual demonstration showing:
- Hierarchical exponential layer distribution.
- Top-$K$ query execution.
- Direct comparison with exact brute-force search (demonstrating ~54% distance computation savings with 100% recall).
- Index save and reload from disk.

---

## Detailed Integration Documentation
Refer to [`docs/TEAM_INTEGRATION_GUIDE.md`](docs/TEAM_INTEGRATION_GUIDE.md) for full algorithmic explanations, complexity tables, and code snippets for Teammates 2, 3, and 4.