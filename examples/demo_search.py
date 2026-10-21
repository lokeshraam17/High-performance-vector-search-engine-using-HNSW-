"""
End-to-End Search Engine Demonstration (Module 4).

This demonstration script illustrates:
1. Initializing the unified SearchEngine orchestrator.
2. Ingesting documents from 'data/sample_documents.csv'.
3. Inspecting the multi-layer HNSW graph structure and indexing latency.
4. Performing natural language vector search across technical domains.
5. Configuring Top-K and dynamic Layer 0 beam width (efSearch).
6. Running side-by-side ground truth comparisons against Exact Brute-Force KNN.
7. Measuring empirical Recall@K and distance computation savings.
8. Verifying robust error handling for edge cases (empty query, unindexed search).

Run directly:
    python examples/demo_search.py
"""

import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search.engine import SearchEngine


def print_banner(title: str) -> None:
    """Print visually distinct banner."""
    print("\n" + "=" * 80)
    print(f"  {title.center(76)}")
    print("=" * 80)


def main() -> None:
    print_banner("HNSW Vector Search Engine - End-to-End Integration Demo (Module 4)")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "sample_documents.csv")

    # Step 1: Initialize Engine
    dim = 64
    metric = "cosine"
    print(f"\n[Step 1] Initializing SearchEngine (dim={dim}, metric='{metric}', M=16, efConstruction=64, efSearch=32)")
    engine = SearchEngine(
        dim=dim,
        metric=metric,
        M=16,
        ef_construction=64,
        ef_search=32,
        seed=42,
    )
    print(" -> Engine initialized successfully.")
    print(f"    - Ready status: {engine.is_indexed} (0 documents)")

    # Step 2: Edge Case Validation - Search before indexing
    print("\n[Step 2] Validating Edge Case: Search before indexing")
    try:
        engine.search("unindexed query test", k=3)
    except RuntimeError as e:
        print(f" -> Successfully caught expected error: '{e}'")

    # Step 3: Load Dataset and Construct HNSW Graph + Exact Baseline
    print(f"\n[Step 3] Ingesting dataset from: {os.path.relpath(csv_path, base_dir)}")
    count = engine.load_dataset(csv_path)
    status = engine.get_status()
    print(f" -> Successfully loaded and indexed {count} documents in {status['index_build_time_ms']:.2f} ms.")
    print(" -> HNSW Graph Hierarchy:")
    g = status["graph_diagnostics"]
    print(f"    - Max Level:            {g.get('max_level')}")
    print(f"    - Total Graph Nodes:    {g.get('total_nodes')}")
    print(f"    - Nodes per Level:      {g.get('nodes_per_level')}")
    print(f"    - Avg Degree per Level: {g.get('avg_degree_per_level')}")

    # Step 4: Multi-Domain Natural Language Queries
    print("\n[Step 4] Executing Natural Language Vector Searches (HNSW Approximate Search)")
    queries = [
        ("balanced binary search tree rotations", 3),
        ("dijkstra shortest path algorithm priority queue", 3),
        ("deep transformer attention mechanisms neural network", 3),
        ("distributed consensus fault tolerant raft protocol", 3),
    ]

    for q_text, top_k in queries:
        print(f"\n--- Query: \"{q_text}\" (Top-{top_k}) ---")
        resp = engine.search(query=q_text, k=top_k)
        print(f"Latency: {resp.latency_ms:.3f} ms | Distance Calculations: {resp.distance_computations}")
        for item in resp.results:
            sim_bar = "#" * int(item.similarity * 15)
            print(f"  #{item.rank} {item.title:<36} [{item.category}]")
            print(f"      Score: {item.similarity*100:.1f}% [{sim_bar:<15}] | Dist: {item.distance:.4f}")
            print(f"      Snippet: {item.snippet[:90]}...")

    # Step 5: Side-by-Side Ground Truth Comparison (Module 1 HNSW vs Module 3 Exact Baseline)
    print_banner("Step 5: Empirical Comparison Against Exact Brute-Force Oracle")
    comp_queries = [
        "operating system virtual memory paging tables",
        "public key cryptography rsa encryption",
    ]

    for q_text in comp_queries:
        print(f"\nEvaluating query: \"{q_text}\" (Top-5)")
        comp = engine.compare_search(query=q_text, k=5)
        
        print(f"  {'Metric':<28} | {'HNSW (Approximate)':<20} | {'Brute-Force (Exact)':<20}")
        print("  " + "-" * 74)
        print(f"  {'Recall@5 Accuracy':<28} | {comp.recall_percent:.1f}%{'':<15} | 100.0% (Ground Truth)")
        print(f"  {'Query Latency':<28} | {comp.hnsw.latency_ms:.3f} ms{'':<12} | {comp.baseline.latency_ms:.3f} ms")
        print(f"  {'Distance Computations':<28} | {comp.hnsw.distance_computations} checks{'':<11} | {comp.baseline.distance_computations} checks")
        print(f"  {'Speedup Factor':<28} | {comp.speedup:.2f}x{'':<16} | 1.00x")
        print(f"  {'Comp Reduction Ratio':<28} | {comp.distance_reduction_ratio:.2f}x fewer{'':<11} | baseline")
        print("  " + "-" * 74)

    # Step 6: Parameter Tuning Trade-Off Demo (efSearch = 5 vs efSearch = 40)
    print("\n[Step 6] Demonstrating Query-Time Parameter Sensitivity (efSearch)")
    test_q = "quicksort divide and conquer algorithm"
    for ef in [5, 15, 40]:
        comp = engine.compare_search(query=test_q, k=5, ef_search=ef)
        print(f"  efSearch={ef:<2} -> Recall@5: {comp.recall_percent:5.1f}% | "
              f"Dist Checks: {comp.hnsw.distance_computations:<3} | "
              f"Latency: {comp.hnsw.latency_ms:.3f} ms")

    # Step 7: Edge Case Error Handling
    print("\n[Step 7] Validating Edge Case Error Handling:")
    # Empty query
    try:
        engine.search("   ", k=5)
    except ValueError as e:
        print(f" -> Empty query correctly rejected: '{e}'")

    # Invalid k <= 0
    try:
        engine.search("valid query", k=-1)
    except ValueError as e:
        print(f" -> Invalid Top-K correctly rejected: '{e}'")

    print_banner("DEMO COMPLETED SUCCESSFULLY")
    print("To launch the interactive Web Interface:  python app.py")
    print("To launch the interactive Terminal CLI:    python app.py --cli\n")


if __name__ == "__main__":
    main()
