"""
Unit tests for the Document Processing, Vector Generation & Storage Module (Module 2).

Tests:
1. Document data structure, validation, snippet formatting, and serialization.
2. DeterministicHashEmbedder, TFIDFEmbedder, SentenceTransformerEmbedder, and factory.
3. IDMapper bidirectional mapping (identity and integer modes).
4. DocumentStore CRUD, batch addition, persistence, query embedding, and error handling.
5. CSV, TXT, Directory, and JSON loaders (including malformed data error policies).
6. End-to-end integration between DocumentStore and Teammate 1's HNSWIndex.
"""

import math
import os
import tempfile
import unittest

from docproc import (
    Document,
    DocumentStore,
    IDMapper,
    DeterministicHashEmbedder,
    TFIDFEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    load_from_csv,
    load_from_directory,
    load_from_json,
    load_from_text_file,
)
from hnsw import HNSWIndex


class TestDocument(unittest.TestCase):
    """Test the Document dataclass, validation rules, and helpers."""

    def test_document_creation_valid(self):
        doc = Document(
            doc_id="doc_1",
            text="Hello world test text.",
            metadata={"author": "Alice", "views": 10},
            vector=[0.1, 0.2, 0.3],
        )
        self.assertEqual(doc.doc_id, "doc_1")
        self.assertEqual(doc.text, "Hello world test text.")
        self.assertEqual(doc.metadata["author"], "Alice")
        self.assertEqual(len(doc.vector), 3)

    def test_document_invalid_id(self):
        with self.assertRaises(ValueError):
            Document(doc_id="", text="Valid text")
        with self.assertRaises(ValueError):
            Document(doc_id="   ", text="Valid text")
        with self.assertRaises(ValueError):
            Document(doc_id=None, text="Valid text")

    def test_document_invalid_text(self):
        with self.assertRaises(ValueError):
            Document(doc_id="doc_1", text="")
        with self.assertRaises(ValueError):
            Document(doc_id="doc_1", text="   \n\t  ")
        with self.assertRaises(ValueError):
            Document(doc_id="doc_1", text=None)
        with self.assertRaises(TypeError):
            Document(doc_id="doc_1", text=12345)

    def test_document_invalid_metadata(self):
        with self.assertRaises(TypeError):
            Document(doc_id="doc_1", text="Valid text", metadata="not-a-dict")

    def test_document_vector_validation(self):
        # Empty vector
        with self.assertRaises(ValueError):
            doc = Document(doc_id="doc_1", text="Test", vector=[])

        # Dimension mismatch check
        doc = Document(doc_id="doc_1", text="Test", vector=[1.0, 2.0])
        with self.assertRaises(ValueError):
            doc.validate_vector(expected_dim=3)

        # Non-numeric element
        with self.assertRaises(TypeError):
            doc_bad = Document(doc_id="doc_1", text="Test", vector=[1.0, "two"])

        # NaN element
        with self.assertRaises(ValueError):
            doc_nan = Document(doc_id="doc_1", text="Test", vector=[1.0, float("nan")])

        # Infinite element
        with self.assertRaises(ValueError):
            doc_inf = Document(doc_id="doc_1", text="Test", vector=[float("inf"), 2.0])

    def test_document_snippet(self):
        long_text = "This is a very long string that should definitely be truncated when rendered as a preview snippet."
        doc = Document(doc_id="d1", text=long_text)
        snip = doc.snippet(max_length=25)
        self.assertTrue(snip.endswith("..."))
        self.assertLessEqual(len(snip), 28)

        # Short text
        short_doc = Document(doc_id="d2", text="Short text")
        self.assertEqual(short_doc.snippet(max_length=50), "Short text")

    def test_document_serialization(self):
        doc = Document(
            doc_id="doc_42",
            text="Serialization test",
            metadata={"tag": "dsa"},
            vector=[0.5, -0.5],
        )
        data = doc.to_dict()
        reconstituted = Document.from_dict(data)
        self.assertEqual(reconstituted.doc_id, doc.doc_id)
        self.assertEqual(reconstituted.text, doc.text)
        self.assertEqual(reconstituted.metadata, doc.metadata)
        self.assertEqual(reconstituted.vector, doc.vector)


class TestEmbedders(unittest.TestCase):
    """Test embedding models, dimensions, determinism, and normalization."""

    def test_deterministic_hash_embedder_dimension_and_norm(self):
        dim = 32
        embedder = DeterministicHashEmbedder(dim=dim)
        self.assertEqual(embedder.dim, dim)
        self.assertEqual(embedder.embedding_type, "deterministic_hash")

        vec = embedder.embed_text("Binary search trees and dynamic programming algorithms.")
        self.assertEqual(len(vec), dim)

        # Verify unit L2 norm: sum(v[i]^2) == 1.0
        norm_sq = sum(x * x for x in vec)
        self.assertAlmostEqual(math.sqrt(norm_sq), 1.0, places=5)

    def test_deterministic_hash_embedder_reproducibility(self):
        """Verify that identical text yields identical vectors across instances."""
        dim = 48
        embedder1 = DeterministicHashEmbedder(dim=dim)
        embedder2 = DeterministicHashEmbedder(dim=dim)

        text = "Hierarchical Navigable Small World graphs enable logarithmic approximate search."
        v1 = embedder1.embed_text(text)
        v2 = embedder2.embed_text(text)

        self.assertEqual(v1, v2)

        # Different texts produce different vectors
        v3 = embedder1.embed_text("Completely unrelated culinary recipe for sourdough bread.")
        self.assertNotEqual(v1, v3)

    def test_deterministic_hash_embedder_empty_fallback(self):
        embedder = DeterministicHashEmbedder(dim=16)
        vec = embedder.embed_text("   !!! ??? ...   ")
        self.assertEqual(len(vec), 16)
        norm_sq = sum(x * x for x in vec)
        self.assertAlmostEqual(math.sqrt(norm_sq), 1.0, places=5)

    def test_tfidf_embedder(self):
        corpus = [
            "Data structures and algorithms in Python",
            "Binary search trees and balanced red black trees",
            "Graph search algorithms Dijkstra and A star",
            "Operating systems memory management and paging",
        ]
        dim = 16
        embedder = TFIDFEmbedder(dim=dim, corpus=corpus)
        self.assertTrue(embedder.is_fitted)
        self.assertEqual(embedder.embedding_type, "tfidf")

        vec = embedder.embed_text("balanced binary search trees")
        self.assertEqual(len(vec), dim)
        norm_sq = sum(x * x for x in vec)
        self.assertAlmostEqual(math.sqrt(norm_sq), 1.0, places=5)

    def test_tfidf_unfitted_error(self):
        embedder = TFIDFEmbedder(dim=16)
        self.assertFalse(embedder.is_fitted)
        with self.assertRaises(RuntimeError):
            embedder.embed_text("Query without fit")

    def test_sentence_transformer_error_or_model(self):
        """SentenceTransformerEmbedder raises clear ImportError if package is absent."""
        try:
            import sentence_transformers
            # If installed, test initialization
            embedder = SentenceTransformerEmbedder()
            vec = embedder.embed_text("Test sentence embedding.")
            self.assertEqual(len(vec), embedder.dim)
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                SentenceTransformerEmbedder()
            self.assertIn("pip install sentence-transformers", str(ctx.exception))

    def test_create_embedder_factory(self):
        emb_hash = create_embedder("deterministic_hash", dim=32)
        self.assertIsInstance(emb_hash, DeterministicHashEmbedder)
        self.assertEqual(emb_hash.dim, 32)

        emb_tfidf = create_embedder("tfidf", dim=20, corpus=["A quick brown fox", "Lazy dog"])
        self.assertIsInstance(emb_tfidf, TFIDFEmbedder)

        with self.assertRaises(ValueError):
            create_embedder("non_existent_method")


class TestIDMapper(unittest.TestCase):
    """Test bidirectional document ID and node ID translation."""

    def test_identity_mode(self):
        mapper = IDMapper(mode="identity")
        node_id = mapper.register("doc_101")
        self.assertEqual(node_id, "doc_101")
        self.assertEqual(mapper.to_node_id("doc_101"), "doc_101")
        self.assertEqual(mapper.to_doc_id("doc_101"), "doc_101")
        self.assertTrue(mapper.has_doc_id("doc_101"))
        self.assertTrue(mapper.has_node_id("doc_101"))

    def test_integer_mode(self):
        mapper = IDMapper(mode="integer")
        n0 = mapper.register("alpha")
        n1 = mapper.register("beta")
        n2 = mapper.register("gamma")

        self.assertEqual(n0, 0)
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 2)

        self.assertEqual(mapper.to_doc_id(1), "beta")
        self.assertEqual(mapper.to_node_id("gamma"), 2)

    def test_re_registration_and_collisions(self):
        mapper = IDMapper(mode="identity")
        mapper.register("doc_A", node_id="node_A")
        # Idempotent re-registration of same mapping
        self.assertEqual(mapper.register("doc_A", node_id="node_A"), "node_A")

        # Conflicting registration raises ValueError
        with self.assertRaises(ValueError):
            mapper.register("doc_A", node_id="node_DIFFERENT")

        # Collision on node ID
        with self.assertRaises(ValueError):
            mapper.register("doc_B", node_id="node_A")

    def test_unregister(self):
        mapper = IDMapper(mode="identity")
        mapper.register("doc_temp")
        self.assertTrue(mapper.has_doc_id("doc_temp"))
        mapper.unregister("doc_temp")
        self.assertFalse(mapper.has_doc_id("doc_temp"))
        with self.assertRaises(KeyError):
            mapper.to_node_id("doc_temp")

    def test_serialization(self):
        mapper = IDMapper(mode="integer")
        mapper.register("doc_1")
        mapper.register("doc_2")

        data = mapper.to_dict()
        restored = IDMapper.from_dict(data)
        self.assertEqual(restored.to_node_id("doc_1"), 0)
        self.assertEqual(restored.to_node_id("doc_2"), 1)
        self.assertEqual(restored.to_doc_id(0), "doc_1")


class TestDocumentStore(unittest.TestCase):
    """Test document store operations, duplicate handling, and persistence."""

    def setUp(self):
        self.embedder = DeterministicHashEmbedder(dim=16)
        self.store = DocumentStore(embedder=self.embedder)

    def test_add_and_retrieve_document(self):
        doc = self.store.add_document(
            doc_id="doc_1",
            text="HNSW is a graph-based nearest neighbor algorithm.",
            metadata={"category": "Graph Algorithms"},
        )
        self.assertEqual(self.store.total_documents, 1)
        self.assertTrue(self.store.has_document("doc_1"))
        self.assertEqual(len(doc.vector), 16)

        retrieved = self.store.get_document("doc_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.doc_id, "doc_1")
        self.assertEqual(self.store.get_text("doc_1"), doc.text)
        self.assertEqual(self.store.get_metadata("doc_1")["category"], "Graph Algorithms")
        self.assertEqual(self.store.get_vector("doc_1"), doc.vector)

    def test_duplicate_rejection_and_overwrite(self):
        self.store.add_document("doc_dup", "First version")

        # Duplicate without overwrite raises ValueError
        with self.assertRaises(ValueError):
            self.store.add_document("doc_dup", "Second version")

        # With overwrite=True, succeeds
        updated = self.store.add_document("doc_dup", "Updated version", overwrite=True)
        self.assertEqual(self.store.get_text("doc_dup"), "Updated version")
        self.assertEqual(len(self.store), 1)

    def test_batch_addition(self):
        items = [
            {"id": "b1", "text": "Batch document one text."},
            {"id": "b2", "text": "Batch document two text."},
            {"id": "b3", "text": "Batch document three text."},
        ]
        added = self.store.add_documents(items)
        self.assertEqual(len(added), 3)
        self.assertEqual(self.store.total_documents, 3)

    def test_remove_document(self):
        self.store.add_document("to_remove", "Temporary text.")
        self.assertTrue(self.store.has_document("to_remove"))
        removed = self.store.remove_document("to_remove")
        self.assertTrue(removed)
        self.assertFalse(self.store.has_document("to_remove"))
        self.assertIsNone(self.store.get_document("to_remove"))

        # Removing non-existent returns False
        self.assertFalse(self.store.remove_document("does_not_exist"))

    def test_embed_query(self):
        query_vec = self.store.embed_query("nearest neighbor search")
        self.assertEqual(len(query_vec), 16)
        with self.assertRaises(ValueError):
            self.store.embed_query("")

    def test_persistence_save_and_load(self):
        self.store.add_document("p1", "Document for persistence test 1", {"tag": "A"})
        self.store.add_document("p2", "Document for persistence test 2", {"tag": "B"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.store.save(tmp_path)
            loaded_store = DocumentStore.load(tmp_path)

            self.assertEqual(loaded_store.total_documents, 2)
            self.assertTrue(loaded_store.has_document("p1"))
            self.assertTrue(loaded_store.has_document("p2"))
            self.assertEqual(loaded_store.get_metadata("p1")["tag"], "A")
            self.assertEqual(loaded_store.get_vector("p1"), self.store.get_vector("p1"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestLoaders(unittest.TestCase):
    """Test loading files (CSV, TXT, JSON, and error edge cases)."""

    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sample_csv = os.path.join(self.base_dir, "data", "sample_documents.csv")
        self.sample_txt = os.path.join(self.base_dir, "data", "sample_documents.txt")
        self.malformed_csv = os.path.join(self.base_dir, "data", "malformed_sample.csv")

    def test_load_sample_csv(self):
        docs = load_from_csv(self.sample_csv)
        self.assertEqual(len(docs), 25)
        # Verify first document
        first = docs[0]
        self.assertEqual(first.doc_id, "doc_001")
        self.assertIn("Binary Search Trees", first.metadata["title"])
        self.assertEqual(first.metadata["category"], "Data Structures")
        self.assertTrue(len(first.text) > 0)

    def test_load_sample_txt(self):
        doc = load_from_text_file(self.sample_txt)
        self.assertEqual(doc.doc_id, "sample_documents")
        self.assertIn("Hierarchical Navigable Small World", doc.text)
        self.assertIn("source_file", doc.metadata)

    def test_load_malformed_csv_raise(self):
        """on_error='raise' raises ValueError upon encountering duplicate or missing text."""
        with self.assertRaises(ValueError):
            load_from_csv(self.malformed_csv, on_error="raise")

    def test_load_malformed_csv_skip(self):
        """on_error='skip' gracefully skips bad rows and returns valid documents."""
        docs = load_from_csv(self.malformed_csv, on_error="skip")
        # In malformed_sample.csv:
        # valid_01 (valid)
        # valid_01 (duplicate -> skipped)
        # valid_02 (empty text -> skipped)
        # (empty ID -> skipped)
        # valid_03 (valid)
        # Exactly 2 valid documents should be loaded
        self.assertEqual(len(docs), 2)
        loaded_ids = [d.doc_id for d in docs]
        self.assertIn("valid_01", loaded_ids)
        self.assertIn("valid_03", loaded_ids)

    def test_load_from_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write('[{"doc_id": "j1", "text": "Json doc 1"}, {"doc_id": "j2", "text": "Json doc 2"}]')
            tmp_path = tmp.name

        try:
            docs = load_from_json(tmp_path)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].doc_id, "j1")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestHNSWIntegration(unittest.TestCase):
    """Test seamless interoperability between Module 2 and Module 1."""

    def test_populate_hnsw_and_search(self):
        dim = 16
        embedder = DeterministicHashEmbedder(dim=dim)
        store = DocumentStore(embedder=embedder)

        # Ingest 10 sample documents
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(base_dir, "data", "sample_documents.csv")
        raw_docs = load_from_csv(csv_path)[:10]

        for doc in raw_docs:
            store.add_document(doc_id=doc.doc_id, text=doc.text, metadata=doc.metadata)

        self.assertEqual(store.total_documents, 10)

        # Initialize Teammate 1's HNSW index
        index = HNSWIndex(dim=dim, metric="cosine", M=8, ef_construction=32, ef_search=16, seed=42)

        # Bridge call: populate index directly from DocumentStore
        inserted = store.populate_hnsw_index(index)
        self.assertEqual(inserted, 10)
        self.assertEqual(len(index.nodes), 10)

        # Teammate 4 style query workflow:
        query_text = "balanced binary search tree rotations"
        query_vec = store.embed_query(query_text)

        # HNSW k-NN search
        results = index.knn_search(query_vec, k=3)
        self.assertGreaterEqual(len(results), 1)

        # Resolve node_ids back to documents
        top_node_id, top_distance = results[0]
        top_doc = store.get_document_by_node_id(top_node_id)
        self.assertIsNotNone(top_doc)
        self.assertIn("doc_", top_doc.doc_id)
        self.assertTrue(len(top_doc.text) > 0)
        self.assertIn("title", top_doc.metadata)


if __name__ == "__main__":
    unittest.main()
