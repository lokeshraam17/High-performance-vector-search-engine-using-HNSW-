"""
Standalone Web Application Server for HNSW Vector Search Engine (Module 4).

Built strictly using Python's standard library `http.server` for zero external dependencies.
Provides:
- REST API for querying (/api/search), status (/api/status), and indexing (/api/load).
- Serves the polished, modern Web UI frontend with dark mode, glassmorphism,
  live Top-K adjustments, and side-by-side Brute-Force baseline comparison.
"""

from __future__ import annotations

import cgi
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
import urllib.parse
import webbrowser

from search.engine import SearchEngine

logger = logging.getLogger(__name__)


class SearchHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler for HNSW Search Engine."""

    # Reference to the shared SearchEngine instance
    engine: Optional[SearchEngine] = None
    static_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

    def send_json_response(self, data: Any, status: int = HTTPStatus.OK) -> None:
        """Helper to send a JSON response with proper headers."""
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for static files and API status."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/status":
            if self.engine is None:
                self.send_json_response({"error": "Engine not initialized"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            status = self.engine.get_status()
            self.send_json_response(status)
            return

        # Serve static UI (index.html)
        if path in ("/", "/index.html"):
            index_path = os.path.join(self.static_dir, "index.html")
            if not os.path.exists(index_path):
                self.send_error(HTTPStatus.NOT_FOUND, "index.html not found.")
                return

            with open(index_path, "rb") as f:
                content = f.read()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # Fallback to standard static file serving if requested
        super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests for searching and indexing."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Read JSON body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                self.send_json_response({"error": "Malformed JSON in request body."}, status=HTTPStatus.BAD_REQUEST)
                return
        else:
            body = {}

        if path == "/api/search":
            self._handle_search(body)
        elif path == "/api/load":
            self._handle_load(body)
        else:
            self.send_json_response({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_search(self, body: Dict[str, Any]) -> None:
        """Process search query."""
        if self.engine is None or not self.engine.is_indexed:
            self.send_json_response(
                {
                    "error": "Engine is not indexed. Please load a dataset first.",
                    "code": "NOT_INDEXED",
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        query = body.get("query", "").strip()
        if not query:
            self.send_json_response(
                {"error": "Query string cannot be empty.", "code": "EMPTY_QUERY"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            k = int(body.get("k", 5))
            if k < 1:
                k = 5
        except (ValueError, TypeError):
            k = 5

        try:
            ef_search = int(body.get("ef_search", self.engine.ef_search))
            if ef_search < 1:
                ef_search = self.engine.ef_search
        except (ValueError, TypeError):
            ef_search = self.engine.ef_search

        compare_baseline = bool(body.get("compare_baseline", False))

        try:
            if compare_baseline:
                comparison = self.engine.compare_search(query=query, k=k, ef_search=ef_search)
                self.send_json_response(comparison.to_dict())
            else:
                resp = self.engine.search(query=query, k=k, ef_search=ef_search)
                self.send_json_response(resp.to_dict())
        except Exception as e:
            logger.exception("Error executing search")
            self.send_json_response({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_load(self, body: Dict[str, Any]) -> None:
        """Load dataset and rebuild index."""
        if self.engine is None:
            self.send_json_response({"error": "Engine not initialized"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        filepath = body.get("filepath")
        if not filepath:
            # Default to repo sample_documents.csv
            filepath = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents.csv")
            )

        try:
            count = self.engine.load_dataset(filepath)
            status = self.engine.get_status()
            self.send_json_response(
                {
                    "message": f"Successfully indexed {count} documents from {os.path.basename(filepath)}.",
                    "status": status,
                }
            )
        except Exception as e:
            logger.exception("Failed to load dataset")
            self.send_json_response({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logs to keep terminal clean."""
        # Only log errors or specific actions
        if "404" in format % args or "500" in format % args:
            super().log_message(format, *args)


def create_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    engine: Optional[SearchEngine] = None,
    dataset_path: Optional[str] = None,
) -> ThreadingHTTPServer:
    """
    Factory function to initialize and configure ThreadingHTTPServer.

    Args:
        host: Host IP or hostname (default 127.0.0.1).
        port: Port number (default 8000).
        engine: Pre-configured SearchEngine instance.
        dataset_path: Optional dataset file to preload.

    Returns:
        Configured ThreadingHTTPServer.
    """
    if engine is None:
        engine = SearchEngine(
            dim=64,
            metric="cosine",
            M=16,
            ef_construction=64,
            ef_search=32,
        )

    if dataset_path is None:
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents.csv")
        )

    if os.path.exists(dataset_path) and not engine.is_indexed:
        engine.load_dataset(dataset_path)

    SearchHTTPRequestHandler.engine = engine
    server = ThreadingHTTPServer((host, port), SearchHTTPRequestHandler)
    return server


def run_web_app(
    host: str = "127.0.0.1",
    port: int = 8000,
    dataset_path: Optional[str] = None,
    open_browser: bool = True,
) -> None:
    """Run Web Application server with auto-browser launch."""
    server = create_web_server(host=host, port=port, dataset_path=dataset_path)
    url = f"http://{host}:{port}/"

    print("=" * 80)
    print("  HNSW Vector Search Engine - Web Application Server")
    print("=" * 80)
    print(f"  Server URL:         {url}")
    print(f"  Dataset Loaded:     {dataset_path or 'data/sample_documents.csv'}")
    print(f"  Indexed Documents:  {SearchHTTPRequestHandler.engine.total_documents}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 80 + "\n")

    if open_browser:
        def _open():
            time.sleep(0.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    run_web_app()
