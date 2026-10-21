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
├── search/                        # Search Interface & Integration (Module 4)
│   ├── __init__.py                # Package exports
│   ├── engine.py                  # SearchEngine unified orchestrator
│   ├── cli.py                     # Interactive REPL and one-shot terminal search
│   ├── web.py                     # Zero-dependency Python HTTP server (http.server)
│   └── static/
│       └── index.html             # Modern responsive Web UI dashboard
├── data/                          # Sample datasets
│   ├── sample_documents.csv       # 25-document CS & Algorithms corpus
│   ├── sample_documents.txt       # Plain text sample
│   └── malformed_sample.csv       # Edge-case error testing dataset
├── tests/
│   ├── __init__.py
│   ├── test_hnsw.py               # 19 core HNSW unit tests
│   ├── test_docproc.py            # 31 document processing and integration tests
│   ├── test_benchmark.py          # 23 benchmarking, oracle, metrics, and report tests
│   └── test_search.py             # 17 end-to-end search engine, web API & CLI tests
├── examples/
│   ├── demo_hnsw.py               # Visual HNSW algorithm demonstration
│   ├── demo_docproc.py            # Visual end-to-end document search demonstration
│   ├── demo_benchmark.py          # Visual performance & accuracy evaluation demo
│   └── demo_search.py             # Visual end-to-end search engine & oracle demo
├── docs/
│   ├── TEAM_INTEGRATION_GUIDE.md  # Teammate 1 HNSW algorithm handoff guide
│   ├── DOCUMENT_PROCESSING_GUIDE.md # Teammate 2 Document Processing handoff guide
│   ├── BENCHMARKING_GUIDE.md      # Teammate 3 Benchmarking & Performance handoff guide
│   └── SEARCH_INTERFACE_GUIDE.md  # Teammate 4 Search Interface handoff guide
├── results/                       # Generated benchmark artifacts
│   ├── benchmark_summary.csv      # Tabular empirical performance data
│   ├── benchmark_report.html      # Interactive HTML performance dashboard
│   ├── chart_latency_vs_dataset_size.png / .svg
│   ├── chart_tradeoff_recall_vs_efsearch.png
│   └── chart_build_time_vs_efconstruction.png
├── app.py                         # Single run command entry point (Web & CLI)
└── README.md
```

---

## Quickstart

### 1. Launch Search Engine (Single Run Command)
- **Modern Web Interface (Default):**
  ```bash
  python app.py
  ```
  *Opens the interactive Web UI at `http://127.0.0.1:8000/` with live search, Top-K sliders, graph diagnostics, and side-by-side ground truth comparisons.*

- **Interactive Terminal CLI:**
  ```bash
  python app.py --cli
  ```
  *Or one-shot query:*
  ```bash
  python -m search.cli --query "graph algorithms dijkstra" -k 5 --compare
  ```

### 2. Run Unit Tests
```bash
python -m unittest discover -s tests -v
```
All **90 unit and integration tests** validate:
- Euclidean, Cosine, and Dot-Product distances and validation (19 tests).
- Document data structure, validation, snippet formatting, and serialization (7 tests).
- Deterministic feature hashing, TF-IDF, Transformer fallbacks, and embedder factory (7 tests).
- Bidirectional ID mapping in identity and integer modes (5 tests).
- Document store CRUD, duplicate handling, overwrite, batch ingestion, and JSON persistence (6 tests).
- File loaders: CSV auto-detection, error policies (`raise` vs `skip`), TXT, and JSON (5 tests).
- Exact KNN baseline oracle, Recall@K accuracy evaluation, latency statistics, and memory tracking (11 tests).
- Synthetic vector generation (sphere, clusters, cube) and reproducibility (4 tests).
- BenchmarkSuite execution, parameter sweeps, and scaling (3 tests).
- Reporting: CSV export, HTML report, and SVG vector chart generation (3 tests).
- End-to-end HNSW and DocumentStore integration (2 tests).
- SearchEngine integration, unindexed error handling, query validation, and Top-K ranking (12 tests).
- Web API server: GET /api/status, POST /api/search, ground-truth comparison, static HTML (4 tests).
- Terminal formatting and visual rendering smoke tests (1 test).

### 3. Run Demonstrations
- **Web & End-to-End Search Demo (Module 4):**
  ```bash
  python examples/demo_search.py
  ```
- **Benchmarking & Accuracy Evaluation Demo (Module 3):**
  ```bash
  python examples/demo_benchmark.py
  ```
- **Document Processing & Search Demo (Module 2):**
  ```bash
  python examples/demo_docproc.py
  ```
- **Core HNSW Algorithm Demo (Module 1):**
  ```bash
  python examples/demo_hnsw.py
  ```

### 4. Run Benchmarking Suite
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
- For Document processing, embedding models, and vector storage: [`docs/DOCUMENT_PROCESSING_GUIDE.md`](docs/DOCUMENT_PROCESSING_GUIDE.md)
- For Benchmarking methodology, empirical trade-offs, and accuracy analysis: [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md)
- For Search interface, Web dashboard, CLI, and presentation workflow: [`docs/SEARCH_INTERFACE_GUIDE.md`](docs/SEARCH_INTERFACE_GUIDE.md)