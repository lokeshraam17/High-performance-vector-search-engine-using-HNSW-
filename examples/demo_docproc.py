"""
Document Processing & Vector Generation Demonstration (Module 2).

This script demonstrates:
1. Loading documents from CSV ('data/sample_documents.csv').
2. Ingesting documents into DocumentStore with DeterministicHashEmbedder.
3. Inspecting tokenization, L2 normalization, and vector properties.
4. Seamlessly populating Teammate 1's HNSW graph index.
5. Executing natural language search queries and retrieving relevant documents.
6. Displaying titles, categories, distance metrics, and preview snippets.
7. Demonstrating JSON persistence (saving and reloading the document store).

Run directly:
    python examples/demo_docproc.py
"""

import os
import sys
import tempfile
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from docproc import (
    DocumentStore,
    DeterministicHashEmbedder,
    TFIDFEmbedder,
    load_from_csv,
)
from hnsw import HNSWIndex


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> None:
    print_banner("HNSW Search Engine - Document Processing & Vector Generation (Module 2)")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "sample_documents.csv")

    # Step 1: Ingest documents from CSV
    print(f"\n[Step 1] Ingesting documents from CSV: {os.path.relpath(csv_path, base_dir)}")
    raw_docs = load_from_csv(csv_path)
    print(f" -> Successfully loaded {len(raw_docs)} documents from CSV.")
    print(" -> Sample Document Ingested:")
    sample = raw_docs[0]
    print(f"    - ID:       {sample.doc_id}")
    print(f"    - Title:    {sample.metadata.get('title')}")
    print(f"    - Category: {sample.metadata.get('category')}")
    print(f"    - Snippet:  {sample.snippet(80)}")

    # Step 2: Initialize DocumentStore with DeterministicHashEmbedder
    dim = 64
    print(f"\n[Step 2] Initializing DocumentStore with DeterministicHashEmbedder (dim={dim})")
    embedder = DeterministicHashEmbedder(dim=dim)
    store = DocumentStore(embedder=embedder, id_mode="identity")

    t0 = time.perf_counter()
    added_ids = store.add_documents(raw_docs)
    ingest_time = (time.perf_counter() - t0) * 1000
    print(f" -> Generated embeddings and stored {len(added_ids)} documents in {ingest_time:.2f} ms.")

    # Inspect vector properties of doc_001
    doc_1_vec = store.get_vector("doc_001")
    print(f" -> Document 'doc_001' vector: length={len(doc_1_vec)}, norm={sum(x*x for x in doc_1_vec)**0.5:.4f}")
    print(f"    First 6 components: {[round(x, 4) for x in doc_1_vec[:6]]}...")

    # Step 3: Populate Teammate 1's HNSW Index
    print("\n[Step 3] Initializing Teammate 1's HNSWIndex and populating graph")
    index = HNSWIndex(
        dim=dim,
        metric="cosine",
        M=8,
        M0=16,
        ef_construction=64,
        ef_search=32,
        seed=42,
    )

    t0 = time.perf_counter()
    nodes_inserted = store.populate_hnsw_index(index)
    build_time = (time.perf_counter() - t0) * 1000
    print(f" -> Inserted {nodes_inserted} vectors into HNSW graph in {build_time:.2f} ms.")
    stats = index.get_stats()
    print(f" -> HNSW Graph Stats: max_level={stats['max_level']}, nodes_per_level={stats['nodes_per_level']}")

    # Step 4: Perform Semantic Queries
    print("\n[Step 4] Executing Natural Language Queries via HNSW Vector Search")

    queries = [
        "balanced binary search tree rotations",
        "deep neural networks transformer attention",
        "distributed consensus replication protocol",
        "operating system memory paging and tables",
    ]

    for q_idx, query_text in enumerate(queries, start=1):
        print(f"\n--- Query {q_idx}: \"{query_text}\" ---")
        # 1. Embed query text using the store's embedder
        q_vec = store.embed_query(query_text)

        # 2. Query HNSW graph for top-3 nearest neighbors
        t0 = time.perf_counter()
        results = index.knn_search(q_vec, k=3)
        q_time = (time.perf_counter() - t0) * 1000

        print(f"Search latency: {q_time:.3f} ms | Top matches:")
        for rank, (node_id, distance) in enumerate(results, start=1):
            doc = store.get_document_by_node_id(node_id)
            title = doc.metadata.get("title", "N/A") if doc else "Unknown"
            cat = doc.metadata.get("category", "N/A") if doc else "Unknown"
            snippet = doc.snippet(90) if doc else ""
            sim_score = max(0.0, 1.0 - distance)
            print(f"  [{rank}] {doc.doc_id} (Cosine Dist: {distance:.4f}, Sim: {sim_score:.4f})")
            print(f"      Title:    {title} [{cat}]")
            print(f"      Snippet:  {snippet}")

    # Step 5: DocumentStore Serialization (Save & Load)
    print("\n[Step 5] Demonstrating Persistence (Save / Load)")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        save_path = tmp.name

    try:
        store.save(save_path)
        file_size_kb = os.path.getsize(save_path) / 1024
        print(f" -> Saved document store ({store.total_documents} docs) to: {save_path} ({file_size_kb:.1f} KB)")

        reloaded_store = DocumentStore.load(save_path)
        print(f" -> Reloaded document store: {reloaded_store.total_documents} docs successfully recovered.")
        print(f" -> Verified doc_004 title: '{reloaded_store.get_metadata('doc_004').get('title')}'")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    # Step 6: Pure Python TF-IDF Embedder Demonstration
    print("\n[Step 6] Alternative Embedder: Pure Python TF-IDF Vectorizer")
    corpus = [d.text for d in raw_docs]
    tfidf_embedder = TFIDFEmbedder(dim=32, corpus=corpus)
    tfidf_store = DocumentStore(embedder=tfidf_embedder)
    tfidf_store.add_documents(raw_docs)

    tfidf_index = HNSWIndex(dim=32, metric="cosine", M=8, seed=42)
    tfidf_store.populate_hnsw_index(tfidf_index)

    tf_results = tfidf_index.knn_search(tfidf_store.embed_query("quicksort divide and conquer algorithm"), k=2)
    print(" -> TF-IDF Query: 'quicksort divide and conquer algorithm'")
    for rank, (nid, dist) in enumerate(tf_results, start=1):
        d = tfidf_store.get_document_by_node_id(nid)
        print(f"  [{rank}] {d.doc_id}: {d.metadata.get('title')} (dist: {dist:.4f})")

    print_banner("Module 2 Demonstration Complete! All modules ready for integration.")


if __name__ == "__main__":
    main()
