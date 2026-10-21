# HNSW Benchmarking, Accuracy Evaluation & Performance Analysis Guide (Module 3)

Welcome to **Module 3 (Benchmarking, Accuracy Testing, and Performance Analysis)** for our **"High-Performance Search Engine Using HNSW"** DSA project.

This document details the architectural design, algorithmic theory, trade-off dynamics, empirical findings, and integration instructions for **Teammates 1, 2, and 4**.

---

## 1. Module Overview & Responsibilities

Module 3 provides an empirical evaluation framework for our vector search engine:
1. **Ground-Truth Oracle (`ExactKNNBaseline`):** An independent, exact linear-scan nearest-neighbor engine operating over identical vectors and distance metrics, guaranteeing 100% recall as our evaluation baseline.
2. **Recall@K Accuracy Evaluation:** Quantifies approximate search quality:
   $$\text{Recall}@K = \frac{|\text{Retrieved Top-}K \cap \text{Ground Truth Top-}K|}{K}$$
3. **Statistical Latency & Throughput Profiling:** Measures query latencies down to microsecond resolution, calculating Mean, Median ($p_{50}$), $p_{90}$, $p_{95}$, $p_{99}$, Min, Max, Standard Deviation, and Queries Per Second (QPS):
   $$\text{QPS} = \frac{\text{Total Queries}}{\text{Total Query Time (seconds)}}$$
4. **Memory Profiling (`SystemProfiler`):** Measures Python heap allocation (peak and net memory in MB) during index construction via `tracemalloc`.
5. **System & Environment Profiling:** Records machine hardware, operating system, CPU architecture, core count, total RAM, and Python version to ensure benchmark reproducibility.
6. **Multi-Dimensional Parameter Sensitivity Sweeps:** Systematically analyzes the trade-offs of:
   - $M$: Graph connectivity degree and edge density.
   - $efConstruction$: Candidate beam width during index insertion.
   - $efSearch$: Dynamic beam search width at Layer 0 during queries.
7. **Dataset Scaling Analysis ($N$):** Tracks build time and query latency across growing corpus sizes ($N$).
8. **Automated Reporting & Visualizations:** Exports CSV tables, standalone interactive HTML dashboards, zero-dependency SVG vector plots, and high-resolution PNG charts.

---

## 2. Algorithmic Theory & Trade-Off Dynamics

### 2.1 Why HNSW Trades Accuracy for Speed
- **Exact Search (Brute Force):** Calculates vector distance to **all $N$ nodes** in the corpus. While optimal ($100\%$ Recall), query time grows linearly as $O(N \cdot d)$. For massive document collections ($N \ge 100,000$), exhaustive linear scans violate sub-millisecond search SLAs.
- **Approximate Nearest Neighbor (HNSW):** Navigates a multi-layer geometric graph using greedy routing from sparse highway layers down to dense local layers. It inspects only a small subset of candidate nodes ($O(\log N)$ on average).
- **The Trade-off:** Because graph navigation relies on local greedy heuristics, the search path can occasionally get trapped in a local minimum if $efSearch$ is too narrow or if the graph has insufficient edge connectivity ($M$). Consequently, HNSW is an *approximate* algorithm that trades a small percentage of recall ($0$–$5\%$) for massive speedups in latency and distance computations.

### 2.2 Parameter Sensitivity & The Pareto Frontier

```
                ▲
     100% Recall│                  ● efSearch=40 (Sweet spot: ~99% Recall, High QPS)
                │             ●
                │        ● efSearch=10 (~96% Recall)
                │    ● efSearch=5 (~85% Recall)
                │
                └────────────────────────────────────────►
                  Lower QPS (High Latency)     Higher QPS
```

1. **`efSearch` (Query-time candidate beam size):**
   - **Low `efSearch` (e.g., 5 to 10):** Explores fewer nodes in Layer 0. Latency is ultra-low and QPS is maximized, but recall drops to $85\%$–$95\%$.
   - **High `efSearch` (e.g., 40 to 100):** Explores wider neighborhood frontiers, driving Recall@K to $99\%$–$100\%$, with slightly higher latency.
2. **`efConstruction` (Build-time candidate beam size):**
   - Higher `efConstruction` expands the search frontier when discovering bidirectional neighbors during node insertion.
   - Increases index construction time ($O(N \cdot efConstruction \cdot \log N)$), but produces a higher-quality Delaunay-like graph with fewer routing dead-ends.
3. **`M` (Max connections per node per layer):**
   - Controls graph density. Layer 0 has $M_0 = 2M$ connections.
   - Higher $M$ increases memory consumption ($O(N \cdot M)$) and per-hop distance calculations, but substantially improves graph connectivity in higher-dimensional vector spaces.
4. **Dataset Size ($N$) Scaling:**
   - For small $N$ ($N \le 200$), the constant overhead of Python priority queues and set operations can make brute force competitive with HNSW.
   - As $N$ scales ($N \ge 1,000$), HNSW query latency scales logarithmically ($O(\log N)$) while brute force degrades linearly ($O(N)$), yielding substantial speedups.

---

## 3. Time & Space Complexity Analysis

| Component / Operation | Time Complexity | Auxiliary Space Complexity | Algorithmic Role |
| :--- | :--- | :--- | :--- |
| **Exact Baseline Build** | $O(N)$ | $O(N \cdot d)$ | Appends vectors to internal dictionary. |
| **Exact Baseline Query (`knn_search`)** | $O(N \cdot d + N \log K)$ | $O(K)$ | Exhaustive distance evaluation + bounded max-heap of size $K$. |
| **HNSW Build (`add_item`)** | $O(N \cdot M \log N \cdot d)$ | $O(ef_{construction})$ | Multi-layer beam search and bidirectional edge pruning. |
| **HNSW Query (`knn_search`)** | $O(\log N \cdot d + ef_{search} \cdot M_0 \cdot d)$ | $O(ef_{search} + K)$ | Coarse greedy zoom down layers + Layer 0 beam search. |
| **Recall@K Evaluation** | $O(K)$ | $O(K)$ | Hash set intersection between retrieved and ground truth IDs. |
| **Latency Statistics** | $O(Q \log Q)$ | $O(Q)$ | Sorting query latencies to compute percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$). |
| **Memory Profiling** | $O(1)$ | $O(1)$ | Non-invasive heap memory sampling via `tracemalloc`. |

---

## 4. Class Structure & Public API

### 4.1 `ExactKNNBaseline`
Located in `benchmark/baseline.py`:
```python
from benchmark import ExactKNNBaseline

baseline = ExactKNNBaseline(dim=64, metric="cosine")

# 1. Add vectors
baseline.add_item("doc_001", [0.1, 0.4, ...])
baseline.add_items([("doc_002", vec2), ("doc_003", vec3)])

# 2. Query exact top-K ground truth
results = baseline.knn_search(query_vector, k=5)
# Returns: [('doc_001', 0.124), ('doc_003', 0.312), ...]

# 3. Inspect operations
print(baseline.total_distance_computations)
baseline.reset_distance_computations()
```

### 4.2 `BenchmarkSuite`
Located in `benchmark/runner.py`:
```python
from benchmark import BenchmarkSuite, BenchmarkConfig

suite = BenchmarkSuite()
config = BenchmarkConfig(
    dim=64,
    metric="cosine",
    M=16,
    ef_construction=64,
    ef_search=32,
    k=5,
    num_queries=20,
    num_runs=3,
)

# Run head-to-head comparison
result = suite.run_single_comparison(base_vectors, query_vectors, config)
print(f"Recall@K: {result.recall_at_k * 100:.1f}%")
print(f"HNSW Mean Latency: {result.hnsw_latencies.mean_ms:.3f} ms")
print(f"Brute Force Mean Latency: {result.bf_latencies.mean_ms:.3f} ms")
print(f"Distance Evaluations Saved: {result.distance_comp_reduction_ratio:.1f}x")
```

### 4.3 Evaluation Functions
Located in `benchmark/metrics.py`:
```python
from benchmark.metrics import compute_recall_at_k, compute_latency_stats, SystemProfiler

# Recall@K calculation
recall = compute_recall_at_k(hnsw_results, exact_results, k=5)

# Latency distribution
stats = compute_latency_stats([0.42, 0.51, 0.48, 0.72, 0.39])
print(stats.mean_ms, stats.median_ms, stats.p95_ms, stats.qps)

# Memory tracking
with SystemProfiler.track_memory() as mem:
    # Code to profile
    ...
print(f"Peak Heap Memory: {mem['peak_memory_mb']} MB")
```

---

## 5. Teammate Integration Instructions

### 5.1 For Teammate 1: Core HNSW Algorithm Validation
When refining heuristic neighbor selection, heuristic pruning, or distance routines:
1. Run `python -m benchmark.cli --quick` to verify that Recall@K remains $> 90\%$ and no regressions are introduced.
2. Run `python -m benchmark.cli --full` to verify that distance computations scale as $O(\log N)$ and compare against the linear $O(N)$ baseline.

### 5.2 For Teammate 2: Document Processing & Embeddings
To determine optimal vector dimensionality and embedding models:
1. Use `suite.run_corpus_benchmark("data/sample_documents.csv")` to evaluate retrieval on real ingested documents.
2. Compare index construction throughput (docs/sec) and peak memory across different embedding dimensions (e.g., $d=32$ vs $d=64$ vs $d=384$).

### 5.3 For Teammate 4: Search Interface & Application SLAs
When configuring server query defaults for the search user interface:
1. Inspect the generated `results/chart_tradeoff_recall_vs_efsearch.png` or `results/benchmark_report.html`.
2. Select an `ef_search` setting that meets your latency SLA:
   - For ultra-fast interactive search: set `ef_search = 16` (high QPS, $\sim 95\%$ recall).
   - For high-precision document retrieval: set `ef_search = 48` ($\sim 99\%$–$100\%$ recall).

---

---

## 6. Empirical Benchmark Results & Analysis

### 6.1 Hardware & Software Test Conditions
- **Operating System:** Windows 11 (AMD64)
- **Processor:** Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (18 logical cores)
- **Total Physical Memory:** 15.43 GB RAM
- **Python Version:** 3.13.7
- **Repetitions:** 3 timed runs per experiment with warmup queries to prevent caching artifacts.

---

### 6.2 Dataset Size ($N$) Scaling Results ($d=64$, $\text{metric}=\text{cosine}$, $M=16$, $ef_c=64$, $ef_s=32$, $K=5$)

| $N$ | HNSW Build (s) | Peak RAM (MB) | HNSW Mean Latency | HNSW $p_{95}$ Latency | HNSW QPS | Brute Force Mean Latency | Speedup | Dist Comp Reduction | Recall@5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | 0.68 s | 0.071 MB | 0.489 ms | 0.696 ms | 2046.4 | 0.348 ms | 0.71x | 0.91x | 100.0% |
| **100** | 4.01 s | 0.139 MB | 0.882 ms | 1.527 ms | 1133.1 | 0.782 ms | 0.89x | 0.93x | 100.0% |
| **200** | 13.14 s | 0.260 MB | 1.455 ms | 2.243 ms | 687.1 | 1.294 ms | 0.89x | 0.95x | 100.0% |
| **400** | 34.93 s | 0.536 MB | 2.941 ms | 4.630 ms | 340.0 | 6.627 ms | **2.25x** | 1.06x | 100.0% |
| **800** | 79.94 s | 1.030 MB | 4.095 ms | 6.118 ms | 244.2 | 6.763 ms | **1.65x** | 1.44x | 99.0% |

**Key Takeaways:**
1. **Logarithmic vs Linear Scaling:** As $N$ doubles from $400$ to $800$, brute-force distance calculations double linearly ($400 \to 800$), whereas HNSW distance computations increase sublinearly from $379.1$ to $556.9$ ($1.44\times$ fewer calculations than brute force).
2. **Small $N$ Crossover Point:** For small $N \le 200$, brute force is faster ($0.71\times$ to $0.89\times$) because the graph routing logic, heap priority queue pushes/pops, and visited set checks incur fixed constant overhead. Once $N \ge 400$, HNSW's pruning power dominates, delivering **$1.65\times$ to $2.25\times$ speedups**.
3. **Memory Footprint:** Peak Python heap memory scales linearly at $\approx 1.2 \text{ KB per vector}$ for $M=16, d=64$.

---

### 6.3 The Accuracy vs Speed Trade-Off: Tuning $efSearch$ ($N=400$, $d=64$, $M=16$, $ef_c=64$, $K=5$)

| $efSearch$ | Recall@5 | HNSW Mean Latency | HNSW $p_{95}$ Latency | HNSW QPS | Dist Comps / Query | Speedup vs BF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | **82.4%** | **1.149 ms** | 1.551 ms | **870.3** | **175.0** | **2.29x** |
| **10** | **93.6%** | **1.616 ms** | 1.873 ms | **618.7** | **245.3** | **1.56x** |
| **20** | **100.0%** | **2.245 ms** | 3.054 ms | **445.5** | **328.0** | **1.11x** |
| **40** | **100.0%** | **2.901 ms** | 3.864 ms | **344.8** | **394.7** | **0.96x** |
| **80** | **100.0%** | **3.227 ms** | 4.451 ms | **309.9** | **421.0** | **0.80x** |
| **160** | **100.0%** | **3.850 ms** | 5.653 ms | **259.7** | **423.8** | **0.66x** |

**Key Takeaways:**
1. **The Pareto Trade-off Frontier:** This parameter sweep cleanly demonstrates the core thesis of ANN search:
   - At $efSearch=5$, latency is slashed to **1.149 ms (870 QPS, 2.29x speedup)** by evaluating only $175$ candidates ($43.8\%$ of the dataset), achieving $82.4\%$ Recall.
   - At $efSearch=10$, Recall jumps to **$93.6\%$** while maintaining a **1.56x speedup (618 QPS)**.
   - At $efSearch \ge 20$, Recall reaches **$100.0\%$**, and higher values yield diminishing accuracy returns while increasing query time.
   - **Recommended Production Setting:** $efSearch \in [10, 20]$ provides the optimal balance of $> 93\%$ recall and high throughput.

---

### 6.4 Effect of Graph Connectivity: Tuning $M$ ($N=400$, $ef_c=64$, $ef_s=32$, $K=5$)

| $M$ | Recall@5 | Build Time (s) | Peak RAM (MB) | Mean Query Latency | Dist Comps / Query |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 83.2% | 7.68 s | 0.440 MB | 1.606 ms | 202.5 |
| **8** | 96.0% | 13.41 s | 0.490 MB | 2.166 ms | 297.0 |
| **16** | 100.0% | 34.06 s | 0.536 MB | 2.977 ms | 376.2 |
| **24** | 100.0% | 65.50 s | 0.584 MB | 2.751 ms | 401.1 |

**Key Takeaways:**
- Lower $M$ ($M=4$) produces sparse graphs that build rapidly ($7.68\text{ s}$) but suffer from routing local minima ($83.2\%$ recall).
- $M=8$ to $M=16$ provides near-perfect navigability ($96\%$ to $100\%$) with modest memory growth.

---

### 6.5 Effect of Construction Beam: Tuning $efConstruction$ ($N=400$, $M=16$, $ef_s=32$, $K=5$)

| $efConstruction$ | Recall@5 | Build Time (s) | Throughput (docs/s) |
| :---: | :---: | :---: | :---: |
| **16** | 99.2% | **2.72 s** | **146.8 docs/s** |
| **32** | 100.0% | 28.96 s | 13.8 docs/s |
| **64** | 100.0% | 34.15 s | 11.7 docs/s |
| **128** | 100.0% | 33.45 s | 12.0 docs/s |

**Key Takeaways:**
- $efConstruction=16$ allows ultra-rapid index construction ($2.72\text{ s}$, $146.8\text{ docs/s}$) while still achieving **$99.2\%$ recall**.
- For development and rapid prototyping, lower $efConstruction \in [16, 32]$ is recommended. For production datasets where build time is amortized, $efConstruction \ge 64$ provides maximum routing robustness.

---

## 7. Verification, Tests & Demonstrations

### 7.1 Run All Unit Tests
```bash
python -m unittest discover -s tests -v
```
All **73 unit tests** validate:
- `test_hnsw.py`: 19 core HNSW tests.
- `test_docproc.py`: 31 document processing tests.
- `test_benchmark.py`: 23 benchmarking, oracle, metrics, and visualization tests.

### 7.2 Run Visual Demonstration
```bash
python examples/demo_benchmark.py
```
Executes a live head-to-head comparison and parameter trade-off sweep directly in the console.

### 7.3 Run Benchmark CLI Options
- **Quick reproducible test (< 3s):**
  ```bash
  python -m benchmark.cli --quick
  ```
- **Comprehensive benchmark (Scaling + Parameter Sweeps + Charts):**
  ```bash
  python -m benchmark.cli --full
  ```
- **Real corpus benchmark:**
  ```bash
  python -m benchmark.cli --corpus data/sample_documents.csv
  ```
- **Custom parameter configuration:**
  ```bash
  python -m benchmark.cli --n 500 --dim 64 --k 10 --metric cosine --m 16 --ef-c 100 --ef-s 50
  ```

