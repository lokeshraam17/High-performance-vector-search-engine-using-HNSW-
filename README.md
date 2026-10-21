# High-Performance Search Engine Using HNSW (Hierarchical Navigable Small World)

A Data Structures and Algorithms (DSA) project implementing a high-performance vector search engine from scratch using the Hierarchical Navigable Small World (HNSW) graph algorithm for Approximate Nearest Neighbor (ANN) search.

## Team Structure & Responsibilities
1. **Module 1 (HNSW Core Algorithm - Completed):** HNSW graph construction, hierarchical layer routing, distance metrics, neighbor selection, and top-$K$ approximate search.
2. **Module 2 (Document Processing & Vector Generation - Completed):** Text tokenization, chunking, deterministic & TF-IDF vector embeddings, document store, and CSV/file ingestion.
3. **Module 3 (Benchmarking & Performance Analysis - Completed):** Empirical QPS, statistical latency profiling (mean, median, p95, p99), memory tracking via tracemalloc, Recall@K accuracy against exact brute-force ground truth, dataset scaling analysis, and parameter trade-off curves (M, efConstruction, efSearch).
4. **Module 4 (Search Interface & Integration):** CLI/Web user search interface, query pipeline, end-to-end integration, and results display.

---

## Project Structure
```text
.
├── hnsw/                          # Core HNSW algorithm package (Module 1)
│   ├── __init__.py                # Package exports
│   ├── distance.py                # Euclidean, cosine, and dot-product metrics
│   ├── node.py                    # HNSWNode data structure
│   └── index.py                   # HNSWIndex graph construction & search
├── docproc/                       # Document processing & vector storage (Module 2)
│   ├── __init__.py                # Package exports
│   ├── document.py                # Document data structure and validation
│   ├── mapper.py                  # IDMapper (doc_id <-> HNSW node_id)
│   ├── store.py                   # DocumentStore (CRUD, HNSW bridge, persistence)
│   ├── loaders.py                 # CSV, TXT, and JSON document loaders
│   └── embeddings/                # Vector embedding strategies
│       ├── __init__.py            # create_embedder factory
│       ├── base.py                # BaseEmbedder abstract class
│       ├── hash_embedder.py       # Deterministic feature hashing (zero dependencies)
│       ├── tfidf_embedder.py      # Pure Python TF-IDF vectorizer
│       └── sentence_transformer.py# Optional pre-trained Transformer embeddings
├── benchmark/                     # Benchmarking & Performance Evaluation (Module 3)
│   ├── __init__.py                # Package exports
│   ├── baseline.py                # ExactKNNBaseline linear scan ground-truth oracle
│   ├── metrics.py                 # Recall@K, LatencyStats, and SystemProfiler (tracemalloc)
│   ├── synthetic_data.py          # Reproducible vector generators (sphere, cluster, cube)
│   ├── runner.py                  # BenchmarkSuite execution engine (scaling & parameter sweeps)
│   ├── reporting.py               # CSV export, HTML report, PNG and SVG vector charts
│   └── cli.py                     # CLI entry point (--quick, --full, --corpus, custom)
├── data/                          # Sample datasets
│   ├── sample_documents.csv       # 25-document CS & Algorithms corpus
│   ├── sample_documents.txt       # Plain text sample
│   └── malformed_sample.csv       # Edge-case error testing dataset
├── tests/
│   ├── __init__.py
│   ├── test_hnsw.py               # 19 core HNSW unit tests
│   ├── test_docproc.py            # 31 document processing and integration tests
│   └── test_benchmark.py          # 23 benchmarking, oracle, metrics, and report tests
├── examples/
│   ├── demo_hnsw.py               # Visual HNSW algorithm demonstration
│   ├── demo_docproc.py            # Visual end-to-end document search demonstration
│   └── demo_benchmark.py          # Visual performance & accuracy evaluation demo
├── docs/
│   ├── TEAM_INTEGRATION_GUIDE.md  # Teammate 1 HNSW algorithm handoff guide
│   ├── DOCUMENT_PROCESSING_GUIDE.md # Teammate 2 Document Processing handoff guide
│   └── BENCHMARKING_GUIDE.md      # Teammate 3 Benchmarking & Performance handoff guide
├── results/                       # Generated benchmark artifacts
│   ├── benchmark_summary.csv      # Tabular empirical performance data
│   ├── benchmark_report.html      # Interactive HTML performance dashboard
│   ├── chart_latency_vs_dataset_size.png / .svg
│   ├── chart_tradeoff_recall_vs_efsearch.png
│   └── chart_build_time_vs_efconstruction.png
└── README.md
```

---

## Quickstart

### 1. Run Unit Tests
```bash
python -m unittest discover -s tests -v
```
All **73 unit tests** validate:
- Euclidean, Cosine, and Dot-Product distances and validation (19 tests).
- Document data structure, validation, snippet formatting, and serialization (7 tests).
- Deterministic feature hashing, TF-IDF, Transformer fallbacks, and embedder factory (7 tests).
- Bidirectional ID mapping in identity and integer modes (5 tests).
- Document store CRUD, duplicate handling, overwrite, batch ingestion, and JSON persistence (6 tests).
- File loaders: CSV auto-detection, error policies (`raise` vs `skip`), TXT, and JSON (5 tests).
- End-to-end integration: populating HNSW index directly from DocumentStore and querying (1 test).
- Exact KNN baseline oracle, Recall@K accuracy evaluation, latency statistics, and memory tracking (11 tests).
- Synthetic vector generation (sphere, clusters, cube) and reproducibility (4 tests).
- BenchmarkSuite execution, parameter sweeps, and scaling (3 tests).
- Reporting: CSV export, HTML report, and SVG vector chart generation (3 tests).
- End-to-end integration: populating HNSW index directly from DocumentStore and querying (2 tests).

### 2. Run Demonstrations
- **Core HNSW Algorithm Demo:**
  ```bash
  python examples/demo_hnsw.py
  ```
- **Document Processing & Search Demo:**
  ```bash
  python examples/demo_docproc.py
  ```
- **Benchmarking & Accuracy Evaluation Demo:**
  ```bash
  python examples/demo_benchmark.py
  ```

### 3. Run Benchmarking Suite
- **Quick reproducible run (< 3s):**
  ```bash
  python -m benchmark.cli --quick
  ```
- **Real document corpus benchmark:**
  ```bash
  python -m benchmark.cli --corpus data/sample_documents.csv
  ```
- **Full scaling and parameter sweep benchmark:**
  ```bash
  python -m benchmark.cli --full
  ```

---

## Detailed Integration Documentation
- For HNSW core algorithm specifications: [`docs/TEAM_INTEGRATION_GUIDE.md`](docs/TEAM_INTEGRATION_GUIDE.md)
- For Document processing, embedding models, and search interface integration: [`docs/DOCUMENT_PROCESSING_GUIDE.md`](docs/DOCUMENT_PROCESSING_GUIDE.md)
- For Benchmarking methodology, empirical trade-offs, and accuracy analysis: [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md)