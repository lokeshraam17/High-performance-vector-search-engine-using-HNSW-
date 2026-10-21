"""
Interactive Command-Line Search Interface (Module 4).

Provides an interactive terminal interface for the HNSW Vector Search Engine:
- Live text queries with customizable Top-K and efSearch.
- Document title, snippet, rank, cosine distance, and similarity score display.
- Side-by-side ground truth comparison against Exact Brute-Force KNN (Module 3).
- Instant indexing status, document count, and query latency profiling.
- Graph hierarchy diagnostic breakdown (layers, degree, entry point).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from search.engine import ComparisonResponse, SearchEngine, SearchResponse


class TerminalColors:
    """ANSI color codes for clean terminal presentation."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"

    # Background
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


def colorize(text: str, color: str) -> str:
    """Return colored text if stdout is a tty, else return plain text."""
    if sys.stdout.isatty():
        return f"{color}{text}{TerminalColors.RESET}"
    return text


def print_header(title: str) -> None:
    """Print a visually separated header banner."""
    line = "=" * 80
    print(colorize(f"\n{line}", TerminalColors.CYAN))
    print(colorize(f"  {title.center(76)}", TerminalColors.BOLD + TerminalColors.WHITE))
    print(colorize(f"{line}\n", TerminalColors.CYAN))


def format_search_results(response: SearchResponse, compare_with: Optional[SearchResponse] = None) -> None:
    """Format and print ranked search results to stdout."""
    print(colorize(f"\n--- Results for: \"{response.query}\" (Top-{response.k}) ---", TerminalColors.BOLD))
    print(
        f"Mode: {colorize(response.mode, TerminalColors.MAGENTA)} | "
        f"Latency: {colorize(f'{response.latency_ms:.3f} ms', TerminalColors.GREEN)} | "
        f"Dist Computations: {colorize(str(response.distance_computations), TerminalColors.YELLOW)} | "
        f"Total Indexed: {response.total_indexed}"
    )
    print("-" * 80)

    if not response.results:
        print(colorize("No documents found.", TerminalColors.YELLOW))
        return

    for item in response.results:
        rank_str = f"#{item.rank:<2}"
        score_bar = "#" * int(item.similarity * 20)
        sim_pct = f"{item.similarity * 100:.1f}%"

        print(
            f"{colorize(rank_str, TerminalColors.BOLD + TerminalColors.GREEN)} "
            f"{colorize(item.title, TerminalColors.BOLD + TerminalColors.WHITE)} "
            f"[{colorize(item.category, TerminalColors.CYAN)}] "
            f"(ID: {colorize(item.doc_id, TerminalColors.DIM)})"
        )
        print(
            f"    Score: {colorize(sim_pct, TerminalColors.GREEN)} [{colorize(score_bar, TerminalColors.GREEN):<20}] | "
            f"Distance: {item.distance:.4f}"
        )
        print(f"    Snippet: {colorize(item.snippet, TerminalColors.DIM)}")
        print()


def format_comparison_results(comparison: ComparisonResponse) -> None:
    """Format side-by-side comparison between HNSW and Exact Brute Force."""
    print_header("HNSW vs Exact Brute-Force Comparison")

    h = comparison.hnsw
    b = comparison.baseline

    print(f"Query: \"{colorize(comparison.query, TerminalColors.BOLD + TerminalColors.WHITE)}\" (Top-{comparison.k})")
    print(f"Corpus Size: {h.total_indexed} documents\n")

    # Metrics Summary Table
    print(colorize(f"{'Metric':<30} | {'HNSW (Approximate)':<22} | {'Brute-Force (Exact)':<22}", TerminalColors.BOLD))
    print("-" * 80)
    print(f"{'Recall@K Accuracy':<30} | {colorize(f'{comparison.recall_percent:.1f}%', TerminalColors.BOLD + TerminalColors.GREEN):<31} | {'100.0% (Oracle)':<22}")
    print(f"{'Search Latency':<30} | {colorize(f'{h.latency_ms:.3f} ms', TerminalColors.GREEN):<31} | {f'{b.latency_ms:.3f} ms':<22}")
    print(f"{'Distance Computations':<30} | {colorize(str(h.distance_computations), TerminalColors.YELLOW):<31} | {str(b.distance_computations):<22}")
    print(f"{'Speedup Multiplier':<30} | {colorize(f'{comparison.speedup:.2f}x', TerminalColors.CYAN):<31} | {'1.00x':<22}")
    print(f"{'Comp Reduction Ratio':<30} | {colorize(f'{comparison.distance_reduction_ratio:.2f}x fewer', TerminalColors.CYAN):<31} | {'baseline':<22}")
    print("-" * 80)

    # Detailed HNSW Results
    print(colorize("\n[HNSW Approximate Matches]:", TerminalColors.BOLD + TerminalColors.CYAN))
    for r in h.results:
        print(f"  #{r.rank} {r.title:<36} (ID: {r.doc_id}) | Sim: {r.similarity*100:.1f}% | Dist: {r.distance:.4f}")

    # Ground Truth Matches
    print(colorize("\n[Ground Truth Exact Matches]:", TerminalColors.BOLD + TerminalColors.MAGENTA))
    for r in b.results:
        matched = "MATCH" if any(hr.doc_id == r.doc_id for hr in h.results) else "MISSED"
        match_color = TerminalColors.GREEN if matched == "MATCH" else TerminalColors.RED
        print(f"  #{r.rank} {r.title:<36} (ID: {r.doc_id}) | Dist: {r.distance:.4f} {colorize(f'[{matched}]', match_color)}")
    print()


def interactive_loop(engine: SearchEngine, default_csv: str) -> None:
    """Run interactive REPL search console."""
    print_header("HNSW Vector Search Engine - Interactive CLI")
    print("Welcome! Type a search query, a command, or 'help'.")

    k = 5
    ef_search = 32
    compare_mode = False

    while True:
        try:
            status_tag = colorize(f"[{engine.total_documents} docs indexed]", TerminalColors.GREEN) if engine.is_indexed else colorize("[Not Indexed]", TerminalColors.RED)
            comp_tag = colorize("[Compare: ON]", TerminalColors.CYAN) if compare_mode else colorize("[Compare: OFF]", TerminalColors.DIM)

            prompt_str = f"\n{status_tag} {comp_tag} (k={k}, ef={ef_search})\nQuery > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ("exit", "quit", "q"):
                print(colorize("\nGoodbye!", TerminalColors.CYAN))
                break

            elif cmd_lower in ("help", "h", "?"):
                print("\nAvailable Commands:")
                print("  <text>                 : Run vector search query across documents")
                print("  :k <number>            : Set Top-K results count (e.g. :k 10)")
                print("  :ef <number>           : Set HNSW efSearch beam width (e.g. :ef 64)")
                print("  :compare [on|off]      : Toggle side-by-side ground truth comparison")
                print("  :load [path]           : Load dataset (default: data/sample_documents.csv)")
                print("  :status                : Display engine and HNSW graph stats")
                print("  :exit                  : Exit the program")
                continue

            elif cmd_lower.startswith(":k"):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    new_k = int(parts[1])
                    if new_k > 0:
                        k = new_k
                        print(colorize(f"Top-K updated to: {k}", TerminalColors.GREEN))
                    else:
                        print(colorize("Error: Top-K must be >= 1.", TerminalColors.RED))
                else:
                    print(colorize("Usage: :k <positive_integer>", TerminalColors.YELLOW))
                continue

            elif cmd_lower.startswith(":ef"):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    new_ef = int(parts[1])
                    if new_ef >= 1:
                        ef_search = new_ef
                        print(colorize(f"efSearch updated to: {ef_search}", TerminalColors.GREEN))
                    else:
                        print(colorize("Error: efSearch must be >= 1.", TerminalColors.RED))
                else:
                    print(colorize("Usage: :ef <positive_integer>", TerminalColors.YELLOW))
                continue

            elif cmd_lower.startswith(":compare"):
                parts = user_input.split()
                if len(parts) >= 2:
                    val = parts[1].lower()
                    compare_mode = val in ("on", "true", "1", "yes")
                else:
                    compare_mode = not compare_mode
                state = "ENABLED" if compare_mode else "DISABLED"
                print(colorize(f"Ground-truth comparison mode {state}.", TerminalColors.CYAN))
                continue

            elif cmd_lower.startswith(":load"):
                parts = user_input.split(maxsplit=1)
                path = parts[1].strip() if len(parts) > 1 else default_csv
                try:
                    count = engine.load_dataset(path)
                    print(colorize(f"Successfully loaded and indexed {count} documents from '{path}'.", TerminalColors.GREEN))
                except Exception as e:
                    print(colorize(f"Failed to load dataset: {e}", TerminalColors.RED))
                continue

            elif cmd_lower == ":status":
                status = engine.get_status()
                print("\n" + "=" * 50)
                print("  Search Engine & Index Status")
                print("=" * 50)
                print(f"Indexed:              {status['is_indexed']}")
                print(f"Total Documents:      {status['total_documents']}")
                print(f"Vector Dimension:     {status['embedding_dimension']}")
                print(f"Distance Metric:      {status['distance_metric']}")
                print(f"Embedder Type:        {status['embedder_type']}")
                print(f"Index Build Time:     {status['index_build_time_ms']} ms")
                print(f"Source Dataset:       {status['source_dataset']}")
                print("HNSW Parameters:     ", status['hnsw_config'])
                print("Graph Diagnostics:   ", status['graph_diagnostics'])
                continue

            # Execute Query
            if not engine.is_indexed:
                print(colorize("Notice: Engine is not indexed yet. Loading sample dataset automatically...", TerminalColors.YELLOW))
                engine.load_dataset(default_csv)
                print(colorize(f"Indexed {engine.total_documents} documents.", TerminalColors.GREEN))

            if compare_mode:
                comp = engine.compare_search(query=user_input, k=k, ef_search=ef_search)
                format_comparison_results(comp)
            else:
                resp = engine.search(query=user_input, k=k, ef_search=ef_search)
                format_search_results(resp)

        except KeyboardInterrupt:
            print(colorize("\nInterrupted. Type ':exit' to quit.", TerminalColors.YELLOW))
        except Exception as e:
            print(colorize(f"Error: {e}", TerminalColors.RED))


def main() -> None:
    """CLI entry point with argument parsing."""
    default_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents.csv")
    )

    parser = argparse.ArgumentParser(
        description="High-Performance HNSW Vector Search Engine CLI"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_csv,
        help="Path to CSV dataset (default: data/sample_documents.csv)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="One-shot search query (if omitted, launches interactive REPL)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5)",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=32,
        help="HNSW Layer 0 search candidate beam width (default: 32)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare HNSW against Exact Brute-Force ground truth",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dot_product"],
        help="Distance metric (default: cosine)",
    )

    args = parser.parse_args()

    engine = SearchEngine(
        dim=args.dim,
        metric=args.metric,
        M=16,
        ef_construction=64,
        ef_search=args.ef_search,
    )

    # If dataset file exists, preload it
    if os.path.exists(args.dataset):
        engine.load_dataset(args.dataset)

    # One-shot mode if --query is provided
    if args.query:
        if args.compare:
            comp = engine.compare_search(query=args.query, k=args.top_k, ef_search=args.ef_search)
            format_comparison_results(comp)
        else:
            resp = engine.search(query=args.query, k=args.top_k, ef_search=args.ef_search)
            format_search_results(resp)
    else:
        # Launch Interactive REPL
        interactive_loop(engine, default_csv=args.dataset)


if __name__ == "__main__":
    main()
