"""
Hierarchical Navigable Small World (HNSW) Core Algorithm Demonstration.

This script demonstrates:
1. Initializing an HNSW index with custom parameters.
2. Inserting 100 synthetic document embeddings (dimension = 4).
3. Inspecting the hierarchical multi-layer graph structure.
4. Performing a Top-K approximate nearest neighbor search.
5. Comparing HNSW results against exact brute-force search (computing recall and distance count).
6. Saving and reloading the index from disk.

Run this script directly:
    python examples/demo_hnsw.py
"""

import math
import os
import random
import sys
import time

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hnsw import HNSWIndex


def main():
    print("=" * 70)
    print("  HNSW (Hierarchical Navigable Small World) Algorithm Demonstration")
    print("=" * 70)

    # 1. Configuration
    dim = 4
    num_vectors = 300
    k = 5
    seed = 42

    print(f"\n[Step 1] Initializing HNSW Index (dim={dim}, metric='euclidean', M=8, ef_construction=64, ef_search=32)")
    index = HNSWIndex(
        dim=dim,
        metric="euclidean",
        M=8,
        M0=16,
        ef_construction=64,
        ef_search=32,
        seed=seed,
    )

    # 2. Vector generation and insertion
    print(f"\n[Step 2] Inserting {num_vectors} synthetic document vectors...")
    rng = random.Random(seed)
    sample_data = {}
    for i in range(num_vectors):
        doc_id = f"doc_{i:03d}"
        # Generate random vector in [-10, 10]
        vec = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(dim)]
        sample_data[doc_id] = vec
        index.add_item(doc_id, vec)

    print(f" -> Successfully inserted {len(index.nodes)} documents into the graph.")

    # 3. Inspect hierarchy statistics
    stats = index.get_stats()
    print(f"\n[Step 3] Graph Hierarchy Statistics:")
    print(f" - Maximum layer reached: Layer {stats['max_level']}")
    print(f" - Entry point node ID  : {stats['enter_point_id']}")
    print(f" - Node distribution across layers:")
    for lvl in range(stats['max_level'], -1, -1):
        count = stats['nodes_per_level'].get(lvl, 0)
        avg_deg = stats['avg_degree_per_level'].get(lvl, 0.0)
        bar = "#" * max(1, int(count * 30 / num_vectors))
        print(f"     Layer {lvl}: {count:3d} nodes (avg degree: {avg_deg:4.1f}) | {bar}")

    # 4. Top-K Query Execution
    query_vec = [1.5, -2.0, 4.0, -0.5]
    print(f"\n[Step 4] Performing Top-{k} Nearest-Neighbor Search...")
    print(f" - Query vector: {query_vec}")

    # Reset distance counter before search to measure query cost
    index.reset_distance_computations()
    start_time = time.perf_counter()
    hnsw_results = index.knn_search(query_vec, k=k)
    hnsw_time_ms = (time.perf_counter() - start_time) * 1000.0
    hnsw_dist_comps = index.total_distance_computations

    # 5. Exact ground-truth comparison (Brute-force)
    index.reset_distance_computations()
    start_time = time.perf_counter()
    exact_results = index.brute_force_knn(query_vec, k=k)
    bf_time_ms = (time.perf_counter() - start_time) * 1000.0
    bf_dist_comps = index.total_distance_computations

    # Calculate Recall@K
    hnsw_ids = [res[0] for res in hnsw_results]
    exact_ids = [res[0] for res in exact_results]
    matches = set(hnsw_ids).intersection(set(exact_ids))
    recall = len(matches) / k

    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'HNSW Result (ID, Dist)':<32}{'Exact Brute-Force (ID, Dist)':<32}")
    print("-" * 70)
    for rank in range(k):
        hnsw_str = f"{hnsw_results[rank][0]} (dist: {hnsw_results[rank][1]:.4f})"
        exact_str = f"{exact_results[rank][0]} (dist: {exact_results[rank][1]:.4f})"
        print(f"{rank + 1:<6}{hnsw_str:<32}{exact_str:<32}")
    print("-" * 70)

    print(f"\n[Step 5] Performance Comparison:")
    print(f" - Recall@{k}                     : {recall * 100.0:.1f}%")
    print(f" - HNSW Distance Computations    : {hnsw_dist_comps} calls")
    print(f" - Brute-force Computations      : {bf_dist_comps} calls")
    efficiency_gain = (1.0 - hnsw_dist_comps / bf_dist_comps) * 100.0
    print(f" - Work Reduction (Fewer checks) : {efficiency_gain:.1f}% savings")
    print(f" - Search Latency                : HNSW={hnsw_time_ms:.3f}ms | Brute-force={bf_time_ms:.3f}ms")

    # 6. Serialization Demonstration
    dump_path = "hnsw_demo_saved_index.json"
    print(f"\n[Step 6] Serialization Test (Save/Load):")
    index.save(dump_path)
    print(f" - Successfully saved index to '{dump_path}'.")

    reloaded_index = HNSWIndex.load(dump_path)
    reloaded_results = reloaded_index.knn_search(query_vec, k=k)
    assert reloaded_results == hnsw_results, "Reloaded search results must match original!"
    print(f" - Successfully reloaded index. Verified identical search results!")

    # Cleanup temporary demo file
    if os.path.exists(dump_path):
        os.remove(dump_path)
        print(f" - Cleaned up temporary file '{dump_path}'.")

    print("\n" + "=" * 70)
    print("  Demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
