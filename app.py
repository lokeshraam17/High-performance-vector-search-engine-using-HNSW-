"""
Main Application Entry Point for the HNSW Vector Search Engine.

Usage:
    # 1. Launch Web Application Interface (Default):
    python app.py

    # 2. Launch Interactive Terminal CLI Interface:
    python app.py --cli

    # 3. Custom host and port:
    python app.py --port 8080 --host 0.0.0.0
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from search.cli import interactive_loop
from search.engine import SearchEngine
from search.web import run_web_app


def main():
    default_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample_documents.csv"))

    parser = argparse.ArgumentParser(
        description="High-Performance Vector Search Engine using HNSW"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch interactive terminal CLI instead of the Web UI",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the Web UI server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the Web UI server on (default: 8000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_csv,
        help="Path to initial CSV dataset (default: data/sample_documents.csv)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser",
    )

    args = parser.parse_args()

    if args.cli:
        print("Starting HNSW Vector Search Engine (CLI Mode)...")
        engine = SearchEngine(
            dim=64,
            metric="cosine",
            M=16,
            ef_construction=64,
            ef_search=32,
        )
        if os.path.exists(args.dataset):
            engine.load_dataset(args.dataset)
        interactive_loop(engine, default_csv=args.dataset)
    else:
        run_web_app(
            host=args.host,
            port=args.port,
            dataset_path=args.dataset,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
