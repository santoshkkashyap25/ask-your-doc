"""Comprehensive automated tests for AskMyPDF RAG components."""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from src.core.domain.models import Document, DocumentChunk
from src.infrastructure.caching.sqlite_cache import SQLiteDiskCache
from src.infrastructure.llm.mock_llm import MockExtractiveLLM
from src.infrastructure.llm.groq_llm import GroqLLM
from src.infrastructure.llm.grok_llm import GrokLLM
from src.infrastructure.splitters.recursive_splitter import RecursiveCharacterSplitter
from src.infrastructure.vector_stores.memory_store import MemoryVectorStore
from src.services.workflow.langgraph_workflow import LangGraphRAGWorkflow


class TestRAGComponents(unittest.TestCase):
    """Unit tests for OOP modular RAG interfaces and implementations."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_cache.db"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_recursive_splitter(self):
        """Test that RecursiveCharacterSplitter creates chunks respecting bounds."""
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        docs = [
            Document(
                doc_id="doc1",
                filename="test.pdf",
                page_number=1,
                text="Artificial intelligence is transforming software engineering. Modern RAG architectures provide fast, accurate retrieval without hallucinations. Clean OOP code ensures modularity."
            )
        ]
        chunks = splitter.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.doc_id, "doc1")
            self.assertEqual(chunk.filename, "test.pdf")
            self.assertEqual(chunk.page_number, 1)
            self.assertLessEqual(len(chunk.text), 150)
            self.assertTrue(len(chunk.chunk_id) > 0)

    def test_sqlite_cache(self):
        """Test SQLiteDiskCache storage, retrieval, TTL, and metrics."""
        cache = SQLiteDiskCache(self.db_path)

        # Basic Set & Get
        cache.set("answers", "query_1", {"text": "Answer 1"})
        val = cache.get("answers", "query_1")
        self.assertIsNotNone(val)
        self.assertEqual(val["text"], "Answer 1")

        # Cache Miss
        miss = cache.get("answers", "non_existent")
        self.assertIsNone(miss)

        # TTL Expiration
        cache.set("answers", "temp_query", "temporary", ttl=1)
        time.sleep(1.2)
        expired = cache.get("answers", "temp_query")
        self.assertIsNone(expired)

        # Stats
        stats = cache.get_stats()
        self.assertGreater(stats["hits"], 0)
        self.assertGreater(stats["misses"], 0)

        # Clear
        cache.clear("answers")
        self.assertIsNone(cache.get("answers", "query_1"))

    def test_memory_vector_store(self):
        """Test in-memory vector store indexing and cosine search."""
        store = MemoryVectorStore()

        chunk1 = DocumentChunk(
            chunk_id="c1",
            doc_id="doc1",
            filename="doc1.pdf",
            page_number=1,
            chunk_index=0,
            text="Vector databases store high-dimensional embeddings for nearest neighbor search.",
            embedding=[1.0, 0.0, 0.0]
        )
        chunk2 = DocumentChunk(
            chunk_id="c2",
            doc_id="doc2",
            filename="doc2.pdf",
            page_number=1,
            chunk_index=0,
            text="Transformers use self-attention to process natural language tokens.",
            embedding=[0.0, 1.0, 0.0]
        )
        store.add_chunks([chunk1, chunk2])

        # Query aligned with chunk1
        results = store.similarity_search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=3)

        # Scoped filter by doc_id
        filtered = store.similarity_search(query_embedding=[1.0, 0.0, 0.0], top_k=2, filter_doc_id="doc2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk.doc_id, "doc2")

        # Delete document
        store.delete_document("doc1")
        remaining = store.similarity_search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].chunk.chunk_id, "c2")

    def test_mock_llm_extractive(self):
        """Test that MockExtractiveLLM parses prompt and answers accurately."""
        llm = MockExtractiveLLM()
        prompt = (
            "Context:\n"
            "AskYourDoc uses a hybrid LangGraph architecture with FastAPI and ChromaDB.\n\n"
            "Question: What architecture does AskYourDoc use?\n\nAnswer:"
        )
        answer = llm.generate(prompt)
        self.assertIn("LangGraph", answer)
        self.assertIn("FastAPI", answer)

        # Test streaming
        stream_tokens = list(llm.stream(prompt))
        full_streamed = "".join(stream_tokens)
        self.assertIn("LangGraph", full_streamed)

    def test_groq_and_grok_adapters(self):
        """Test GroqLLM and GrokLLM parameter validation and provider naming."""
        # Must raise ValueError without API key
        with self.assertRaises(ValueError):
            GroqLLM(api_key="")

        with self.assertRaises(ValueError):
            GrokLLM(api_key="")

        groq_llm = GroqLLM(api_key="mock_key", model_name="llama-3.3-70b-versatile")
        self.assertEqual(groq_llm.provider_name, "Groq (llama-3.3-70b-versatile)")

        grok_llm = GrokLLM(api_key="mock_key", model_name="grok-beta")
        self.assertEqual(grok_llm.provider_name, "xAI Grok (grok-beta)")

    def test_end_to_end_rag_workflow(self):
        """Test LangGraphRAGWorkflow execution with vector store and cache integration."""
        cache = SQLiteDiskCache(self.db_path)
        store = MemoryVectorStore()
        llm = MockExtractiveLLM()

        # Dummy embeddings returning fixed vector
        class DummyEmbeddings:
            def embed_query(self, text):
                return [1.0, 0.0, 0.0]
            def embed_documents(self, texts):
                return [[1.0, 0.0, 0.0] for _ in texts]

        # Index chunk
        chunk = DocumentChunk(
            chunk_id="chunk_test_1",
            doc_id="doc_alpha",
            filename="contract.pdf",
            page_number=3,
            chunk_index=0,
            text="The termination notice period is exactly thirty days written notice.",
            embedding=[1.0, 0.0, 0.0]
        )
        store.add_chunks([chunk])

        workflow = LangGraphRAGWorkflow(
            vector_store=store,
            embeddings=DummyEmbeddings(),
            llm=llm,
            cache=cache,
            top_k=2,
            similarity_threshold=0.1
        )

        # 1. First run (Cache Miss)
        result1 = workflow.run("What is the termination notice period?")
        self.assertFalse(result1.is_cached)
        self.assertIn("thirty days", result1.answer)
        self.assertEqual(len(result1.citations), 1)
        self.assertEqual(result1.citations[0].page_number, 3)

        # 2. Second run (Cache Hit)
        result2 = workflow.run("What is the termination notice period?")
        self.assertTrue(result2.is_cached)
        self.assertIn("thirty days", result2.answer)

    def test_document_service_constraints(self):
        """Test max documents (10) and total storage (25MB) limits in DocumentService."""
        from src.services.document_service import DocumentService

        class DummyLoader:
            def load(self, path, doc_id=""):
                return [Document(doc_id=doc_id, filename=path.name, page_number=1, text="Sample text")]

        class DummySplitter:
            def split_documents(self, docs):
                return [DocumentChunk(chunk_id=f"c_{docs[0].doc_id}", doc_id=docs[0].doc_id, filename=docs[0].filename, page_number=1, chunk_index=0, text="Sample text")]

        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.1, 0.2] for _ in texts]

        doc_service = DocumentService(
            loader=DummyLoader(),
            splitter=DummySplitter(),
            embeddings=DummyEmbeddings(),
            vector_store=MemoryVectorStore(),
            upload_dir=self.test_dir / "uploads",
            max_documents=3,
            max_total_storage_mb=1
        )

        # Ingest 3 documents successfully
        for i in range(3):
            data = f"PDF content {i}".encode()
            doc_service.ingest_pdf(f"doc_{i}.pdf", data)

        # 4th document should exceed max_documents (3)
        with self.assertRaises(ValueError) as ctx:
            doc_service.ingest_pdf("doc_4.pdf", b"PDF content 4")
        self.assertIn("Maximum document limit reached", str(ctx.exception))

        # Test per-file limit and storage limit
        doc_service2 = DocumentService(
            loader=DummyLoader(),
            splitter=DummySplitter(),
            embeddings=DummyEmbeddings(),
            vector_store=MemoryVectorStore(),
            upload_dir=self.test_dir / "uploads2",
            max_documents=5,
            max_total_storage_mb=10,
            max_file_size_mb=1
        )
        large_bytes = b"0" * (2 * 1024 * 1024)
        with self.assertRaises(ValueError) as ctx2:
            doc_service2.ingest_pdf("large.pdf", large_bytes)
        self.assertIn("File exceeds the maximum", str(ctx2.exception))

    def test_doc_id_validation_on_delete(self):
        """Test that invalid doc_id inputs are strictly rejected to prevent security vulnerabilities."""
        from src.services.document_service import DocumentService

        class DummyLoader:
            def load(self, path, doc_id=""):
                return [Document(doc_id=doc_id, filename=path.name, page_number=1, text="Text")]

        class DummySplitter:
            def split_documents(self, docs):
                return [DocumentChunk(chunk_id="c1", doc_id="d1", filename="f.pdf", page_number=1, chunk_index=0, text="Text")]

        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.1, 0.2] for _ in texts]

        upload_dir = self.test_dir / "security_uploads"
        doc_service = DocumentService(
            loader=DummyLoader(),
            splitter=DummySplitter(),
            embeddings=DummyEmbeddings(),
            vector_store=MemoryVectorStore(),
            upload_dir=upload_dir
        )

        invalid_ids = ["", "*", "*_*", "../etc", "../../passwd", "a" * 65, "bad doc id!"]
        for bad_id in invalid_ids:
            with self.assertRaises(ValueError):
                doc_service.delete_document(bad_id)

    def test_document_service_restart_hydration(self):
        """Test that DocumentService hydrates metadata and storage stats across instance restarts."""
        from src.services.document_service import DocumentService

        class DummyLoader:
            def load(self, path, doc_id=""):
                return [Document(doc_id=doc_id, filename="sample.pdf", page_number=1, text="Hello world")]

        class DummySplitter:
            def split_documents(self, docs):
                return [DocumentChunk(chunk_id=f"c_{docs[0].doc_id}", doc_id=docs[0].doc_id, filename="sample.pdf", page_number=1, chunk_index=0, text="Hello world")]

        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.1, 0.2] for _ in texts]

        upload_dir = self.test_dir / "hydration_uploads"
        cache = SQLiteDiskCache(self.test_dir / "hydration_cache.db")

        # Session 1: ingest document
        s1 = DocumentService(
            loader=DummyLoader(),
            splitter=DummySplitter(),
            embeddings=DummyEmbeddings(),
            vector_store=MemoryVectorStore(),
            upload_dir=upload_dir,
            cache=cache
        )
        meta = s1.ingest_pdf("sample.pdf", b"%PDF-1.4 sample data")
        self.assertEqual(meta.filename, "sample.pdf")

        # Session 2: simulate new server startup pointing to same upload_dir and cache
        s2 = DocumentService(
            loader=DummyLoader(),
            splitter=DummySplitter(),
            embeddings=DummyEmbeddings(),
            vector_store=MemoryVectorStore(),
            upload_dir=upload_dir,
            cache=cache
        )
        docs = s2.list_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["doc_id"], meta.doc_id)
        self.assertEqual(docs[0]["filename"], "sample.pdf")
        self.assertGreater(docs[0]["file_size_bytes"], 0)

        # Deletion cleans disk and cache
        s2.delete_document(meta.doc_id)
        self.assertEqual(len(s2.list_documents()), 0)

    def test_pypdf_loader_clean_filename(self):
        """Test that PyPDFLoader strips the {doc_id}_ prefix from filename in documents."""
        from src.infrastructure.loaders.pypdf_loader import PyPDFLoader
        from unittest.mock import patch, MagicMock

        loader = PyPDFLoader()
        fake_path = Path("data/uploads/abc12345_quarterly_report.pdf")

        # Mock pypdf.PdfReader to avoid needing an actual PDF on disk
        with patch.object(Path, "exists", return_value=True):
            with patch("pypdf.PdfReader") as mock_reader_cls:
                mock_page = MagicMock()
                mock_page.extract_text.return_value = "Quarterly results"
                mock_reader_cls.return_value.pages = [mock_page]

                docs = loader.load(fake_path, doc_id="abc12345")
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0].filename, "quarterly_report.pdf")
                self.assertEqual(docs[0].doc_id, "abc12345")

    def test_groq_llm_stream_reasoning_stripping(self):
        """Test that GroqLLM.stream() strips <think>...</think> without dropping trailing tokens."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        llm = GroqLLM(api_key="gsk_fake_key")

        # Case 1: Reasoning in chunks followed by answer
        def make_chunks(texts):
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]) for t in texts]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_chunks([
            "<th", "ink>I am reasoning deeply...</th", "ink>The answer is 42."
        ])
        llm._client = mock_client

        output = "".join(list(llm.stream("test prompt")))
        self.assertEqual(output, "The answer is 42.")

        # Case 2: Regular streaming without think tags
        mock_client.chat.completions.create.return_value = make_chunks([
            "Direct", " answer", " without reasoning."
        ])
        output2 = "".join(list(llm.stream("test prompt 2")))
        self.assertEqual(output2, "Direct answer without reasoning.")


if __name__ == "__main__":
    unittest.main()
