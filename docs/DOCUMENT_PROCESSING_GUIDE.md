# Document Processing, Vector Generation & Storage Guide (Module 2)

Welcome to **Module 2 (Document Processing, Vector Generation, and Storage)** for our **"High-Performance Search Engine Using HNSW"** DSA project.

This document details the architecture, text-to-vector pipeline theory, complexity analysis, error handling, and concrete instructions for **Teammates 1, 3, and 4** to integrate smoothly.

---

## 1. Module Overview & Responsibilities

Module 2 acts as the data ingestion backbone and document metadata repository for the entire search engine:
1. **Document Ingestion:** Ingests raw documents from CSV files, plain text files (`.txt`), directories of text files, or JSON/JSONL.
2. **Text-to-Vector Pipeline:** Transforms unstructured text into dense, normalized numerical vectors of configurable dimension $d$.
3. **Document Store:** Maintains an $O(1)$ hash map mapping `doc_id` to text, metadata (title, category, timestamps), and vector embeddings.
4. **Identifier Mapping:** Maintains bidirectional translation between human-readable Document IDs (`"doc_001"`) and HNSW Node IDs (string or integer).
5. **Bridge to HNSW Graph:** Populates Teammate 1's `HNSWIndex` directly and provides vector encoding for user search queries (`embed_query`).

---

## 2. Text-to-Vector Pipeline Architecture

```text
Raw Document (CSV / TXT / JSON)
              │
              ▼
    [1. Text Tokenization]
    - Case folding (lower)
    - Regex word extraction: \b[a-zA-Z0-9_\-]+\b
              │
              ▼
   [2. Embedding Strategy]
    ┌──────────────────────────────┬─────────────────────────────┬──────────────────────────────┐
    │  DeterministicHashEmbedder   │        TFIDFEmbedder        │ SentenceTransformerEmbedder  │
    │  (Hashing Trick, SHA-256)    │ (Pure Python IR Vocabulary) │  (Dense Pre-trained Neural)  │
    │  • Zero dependencies         │ • Classic DSA IR model      │ • Semantic contextual dense  │
    │  • Configurable dim (e.g 64) │ • Vocab + IDF table         │ • all-MiniLM-L6-v2 (dim=384) │
    └──────────────────────────────┴─────────────────────────────┴──────────────────────────────┘
              │
              ▼
   [3. Coordinate Weighting]
    - Sublinear TF scaling: w = 1.0 + ln(count)
    - Signed hash assignment / IDF multiplier
              │
              ▼
   [4. L2 Unit Normalization]
    - v_norm = v / ||v||_2
    - Every vector resides on unit hypersphere (||v||_2 = 1.0)
              │
              ▼
   [5. Validation & Storage]
    - Verifies len(v) == dim
    - Asserts no NaN, no Infinite values
    - Persisted in DocumentStore
              │
              ▼
   [6. HNSW Graph Population]
    - index.add_item(node_id, v_norm)
```

### 2.1 Embedding Strategies

#### A. Deterministic Feature Hashing (`DeterministicHashEmbedder`)
- **Algorithm:** Uses cryptographic SHA-256 to map each token $t$ to coordinate $i = \text{hash}_1(t) \pmod d$ and sign $s \in \{-1, +1\}$ based on $\text{hash}_2(t)$.
- **Why SHA-256 instead of Python's built-in `hash()`?** Python randomizes the siphash seed per process (`PYTHONHASHSEED`), which would make vectors non-reproducible across restarts. SHA-256 is 100% deterministic and platform-independent.
- **Zero Dependencies:** Runs out-of-the-box on standard Python 3.8+ without external C libraries.

#### B. Pure Python TF-IDF (`TFIDFEmbedder`)
- **Algorithm:** Builds a vocabulary of top $d$ terms by document frequency across a training corpus. Computes smoothed inverse document frequency:
  $$\text{idf}(t) = \ln\left(1 + \frac{N + 1}{1 + \text{df}(t)}\right) + 1.0$$
- **Weight:** $w(t) = (1 + \ln(\text{tf}(t))) \cdot \text{idf}(t)$.

#### C. Pre-Trained Dense Transformers (`SentenceTransformerEmbedder`)
- **Algorithm:** Uses modern Transformer models (e.g. `all-MiniLM-L6-v2`) via `sentence-transformers`.
- **Graceful Fallback:** If `sentence-transformers` is not installed, catches `ImportError` and outputs clear installation instructions while advising fallback to `DeterministicHashEmbedder`.

---

## 3. Time & Space Complexity Analysis

| Operation | Time Complexity | Auxiliary Space Complexity | Notes |
| :--- | :--- | :--- | :--- |
| **Tokenization (`_tokenize`)** | $O(L)$ | $O(T)$ | $L$ is character length, $T$ is token count. |
| **Hash Embedding (`DeterministicHashEmbedder.embed_text`)** | $O(T)$ | $O(d)$ | Direct hash projection into $d$ coordinate buckets. |
| **TF-IDF Fit (`TFIDFEmbedder.fit`)** | $O(N \cdot T)$ | $O(\|V\| + d)$ | $N$ is corpus size, $\|V\|$ is vocabulary size. |
| **TF-IDF Transform (`TFIDFEmbedder.embed_text`)** | $O(T)$ | $O(d)$ | Vocabulary hash lookup for matched terms. |
| **L2 Normalization (`_l2_normalize`)** | $O(d)$ | $O(d)$ | Single pass vector sum-of-squares and scale. |
| **Add Document (`store.add_document`)** | $O(T + d)$ | $O(L + d)$ | Tokenizes, embeds, validates, and stores in hash map. |
| **Get Document / Text / Vector (`store.get_*`)** | $O(1)$ average | $O(1)$ | Hash table key lookup. |
| **ID Translation (`mapper.to_node_id` / `to_doc_id`)** | $O(1)$ average | $O(1)$ | Bidirectional dictionary lookup. |
| **Populate HNSW (`store.populate_hnsw_index`)** | $O(N \cdot (M \log N \cdot d))$ | $O(N \cdot d)$ | Feeds all vectors into HNSW graph index. |

---

## 4. Teammate Handoff & Integration Instructions

### 4.1 For Teammate 4: Search Interface & Integration
Teammate 4 builds the user-facing CLI or Web search interface.

```python
from docproc import DocumentStore, load_from_csv
from hnsw import HNSWIndex

# 1. Initialize DocumentStore and load data
store = DocumentStore(dim=64, id_mode="identity")
docs = load_from_csv("data/sample_documents.csv")
store.add_documents(docs)

# 2. Initialize HNSW Index and build graph
index = HNSWIndex(dim=64, metric="cosine", M=8, ef_construction=64, ef_search=32)
store.populate_hnsw_index(index)

# 3. Handle search query from user
def search(query_text: str, k: int = 5):
    # A. Encode query into normalized vector
    query_vector = store.embed_query(query_text)

    # B. Query HNSW graph for nearest neighbor node IDs
    results = index.knn_search(query_vector, k=k)

    # C. Retrieve full document objects and format response
    output = []
    for node_id, distance in results:
        doc = store.get_document_by_node_id(node_id)
        if doc is not None:
            output.append({
                "doc_id": doc.doc_id,
                "title": doc.metadata.get("title", "Untitled"),
                "category": doc.metadata.get("category", "General"),
                "distance": distance,
                "similarity": max(0.0, 1.0 - distance),
                "snippet": doc.snippet(max_length=120),
            })
    return output
```

### 4.2 For Teammate 3: Benchmarking & Performance Analysis
Teammate 3 evaluates ingestion throughput, search latency (QPS), and Recall@K.

```python
from docproc import DocumentStore, load_from_csv
import time

# 1. Ingestion Throughput Benchmark
docs = load_from_csv("data/sample_documents.csv")
store = DocumentStore(dim=64)

start = time.perf_counter()
store.add_documents(docs)
elapsed = time.perf_counter() - start
print(f"Ingestion Throughput: {len(docs) / elapsed:.1f} docs/sec")

# 2. Extract raw vector matrix for ground-truth calculation
all_vectors = store.all_vectors()  # {doc_id: [float, ...]}
```

### 4.3 For Teammate 1: HNSW Algorithm Compatibility
- `DocumentStore.populate_hnsw_index(index)` directly loops over documents and invokes `index.add_item(node_id, doc.vector)`.
- Vectors are guaranteed to match `index.dim` and contain finite numerical floats with unit L2 norm ($||v||_2 = 1.0$).

---

## 5. Error Handling & Edge Cases

| Scenario | Behavior | Configuration |
| :--- | :--- | :--- |
| **Missing Document Text** | Raises `ValueError` ("text cannot be empty") | Default behavior. In `load_from_csv(..., on_error='skip')`, skips the row and logs a warning. |
| **Duplicate Document ID** | Raises `ValueError` ("Document with ID already exists") | Default. Set `overwrite=True` in `add_document` to replace existing entry. |
| **Vector Dimension Mismatch** | Raises `ValueError` ("expected dim, got X") | Enforced during ingestion and HNSW population. |
| **Non-numeric / NaN / Inf Coordinates** | Raises `ValueError` / `TypeError` | Validated by `Document.validate_vector`. |
| **Missing Embedding Package** | Raises `ImportError` with setup command | `SentenceTransformerEmbedder` directs user to `pip install sentence-transformers torch` or use zero-dependency fallback. |

---

## 6. How to Run Tests and Demo

### Run All Unit Tests:
```bash
python -m unittest discover -s tests -v
```
Currently 50 unit tests pass across `test_hnsw.py` and `test_docproc.py`.

### Run Module 2 Visual Demonstration:
```bash
python examples/demo_docproc.py
```
This runs an end-to-end demonstration featuring CSV ingestion, vector generation, HNSW population, multi-query search, and JSON persistence.
