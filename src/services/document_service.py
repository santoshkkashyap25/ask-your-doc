"""Document ingestion and lifecycle management service."""

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.domain.models import Document, DocumentChunk, DocumentMetadata
from src.core.interfaces.embeddings import BaseEmbeddings
from src.core.interfaces.loader import BaseDocumentLoader
from src.core.interfaces.splitter import BaseTextSplitter
from src.core.interfaces.vector_store import BaseVectorStore
from src.core.interfaces.cache import BaseCache

logger = logging.getLogger("AskMyPDF.DocService")
DOC_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class DocumentService:
    """Orchestrates document uploading, validation, parsing, chunking, and indexing."""

    def __init__(
        self,
        loader: BaseDocumentLoader,
        splitter: BaseTextSplitter,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        upload_dir: Path,
        max_documents: int = 5,
        max_total_storage_mb: int = 10,
        max_file_size_mb: int = 10,
        cache: Optional[BaseCache] = None,
        zero_retention: bool = True
    ):
        self.loader = loader
        self.splitter = splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_documents = max_documents
        self.max_total_storage_mb = max_total_storage_mb
        self.max_file_size_mb = max_file_size_mb
        self.cache = cache
        self.zero_retention = zero_retention
        self._documents_meta: Dict[str, DocumentMetadata] = {}

    def _compute_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Return current document count and storage usage metrics."""
        current_docs = self.list_documents()
        if self.zero_retention:
            total_bytes = sum(m.file_size_bytes for m in self._documents_meta.values())
        else:
            total_bytes = sum(f.stat().st_size for f in self.upload_dir.glob("*_*") if f.is_file())
        max_total_bytes = self.max_total_storage_mb * 1024 * 1024
        return {
            "total_documents": len(current_docs),
            "max_documents": self.max_documents,
            "total_bytes": total_bytes,
            "max_total_bytes": max_total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "max_total_mb": self.max_total_storage_mb
        }

    def _hydrate_metadata(self) -> None:
        """Hydrate in-memory metadata from cache and disk files if not already in memory."""
        if self.zero_retention:
            return  # In zero-retention mode, metadata lives strictly in memory
        try:
            vector_docs = {d["doc_id"]: d for d in self.vector_store.list_documents()}
        except Exception:
            vector_docs = {}

        for f in self.upload_dir.glob("*_*"):
            if not f.is_file():
                continue
            parts = f.name.split("_", 1)
            if len(parts) == 2 and DOC_ID_REGEX.match(parts[0]):
                d_id = parts[0]
                if d_id in self._documents_meta:
                    continue

                if self.cache:
                    try:
                        cached = self.cache.get("doc_meta", d_id)
                        if cached and isinstance(cached, dict):
                            self._documents_meta[d_id] = DocumentMetadata(
                                doc_id=cached.get("doc_id", d_id),
                                filename=cached.get("filename", parts[1]),
                                file_size_bytes=cached.get("file_size_bytes", f.stat().st_size),
                                content_hash=cached.get("content_hash", ""),
                                total_pages=cached.get("total_pages", 1),
                                total_chunks=cached.get("total_chunks", 0),
                                created_at=cached.get("created_at", "")
                            )
                            continue
                    except Exception as e:
                        logger.warning("Error reading doc_meta cache for '%s': %s", d_id, e)

                v_info = vector_docs.get(d_id, {})
                meta = DocumentMetadata(
                    doc_id=d_id,
                    filename=v_info.get("filename") or parts[1],
                    file_size_bytes=f.stat().st_size,
                    content_hash="",
                    total_pages=v_info.get("total_pages", 1),
                    total_chunks=v_info.get("chunk_count", 0),
                    created_at=""
                )
                self._documents_meta[d_id] = meta
                if self.cache:
                    try:
                        self.cache.set("doc_meta", d_id, {
                            "doc_id": meta.doc_id,
                            "filename": meta.filename,
                            "file_size_bytes": meta.file_size_bytes,
                            "content_hash": meta.content_hash,
                            "total_pages": meta.total_pages,
                            "total_chunks": meta.total_chunks,
                            "created_at": meta.created_at
                        })
                    except Exception:
                        pass

    def ingest_pdf(self, filename: str, file_bytes: bytes) -> DocumentMetadata:
        """Ingest a PDF file from bytes, chunk it, embed it, and store in vector database."""
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files are supported. Received: {filename}")

        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        max_file_bytes = self.max_file_size_mb * 1024 * 1024
        if len(file_bytes) > max_file_bytes:
            file_mb = round(len(file_bytes) / (1024 * 1024), 2)
            raise ValueError(
                f"File exceeds the maximum {self.max_file_size_mb} MB limit (uploaded: {file_mb} MB). "
                f"Please upload a smaller document."
            )

        content_hash = self._compute_hash(file_bytes)

        # Check if identical file was already indexed
        self._hydrate_metadata()
        for existing in self._documents_meta.values():
            if existing.content_hash and existing.content_hash == content_hash:
                logger.info("PDF '%s' with hash '%s' already indexed under doc_id='%s'",
                            filename, content_hash[:8], existing.doc_id)
                return existing

        # Constraint 1: Maximum document count limit
        current_docs = self.list_documents()
        if len(current_docs) >= self.max_documents:
            raise ValueError(
                f"Maximum document limit reached ({self.max_documents} documents). "
                f"Please delete existing documents to upload new ones."
            )

        # Constraint 2: Cumulative storage limit (max 10 MB across all documents)
        if self.zero_retention:
            current_total_bytes = sum(m.file_size_bytes for m in self._documents_meta.values())
        else:
            current_total_bytes = sum(f.stat().st_size for f in self.upload_dir.glob("*_*") if f.is_file())
        max_total_bytes = self.max_total_storage_mb * 1024 * 1024
        new_file_bytes = len(file_bytes)

        if current_total_bytes + new_file_bytes > max_total_bytes:
            current_mb = round(current_total_bytes / (1024 * 1024), 2)
            file_mb = round(new_file_bytes / (1024 * 1024), 2)
            remaining_mb = max(0.0, round((max_total_bytes - current_total_bytes) / (1024 * 1024), 2))
            raise ValueError(
                f"Total storage limit exceeded (maximum {self.max_total_storage_mb} MB). "
                f"Current usage: {current_mb} MB, new file: {file_mb} MB ({remaining_mb} MB remaining). "
                f"Please delete existing documents to free up space."
            )

        doc_id = str(uuid.uuid4())[:8]

        # 1. Load document pages (In-memory zero retention vs disk)
        if self.zero_retention and hasattr(self.loader, "load_bytes"):
            logger.info("Zero-retention mode: parsing '%s' directly in memory (size: %d bytes)", filename, len(file_bytes))
            pages: List[Document] = self.loader.load_bytes(file_bytes, filename=filename, doc_id=doc_id)
        else:
            saved_path = self.upload_dir / f"{doc_id}_{filename}"
            saved_path.write_bytes(file_bytes)
            logger.info("Saved upload to '%s' (size: %d bytes)", saved_path, len(file_bytes))
            pages: List[Document] = self.loader.load(saved_path, doc_id=doc_id)

        if not pages:
            raise ValueError(f"Could not extract any readable text from '{filename}'. It might be a scanned image.")

        # 2. Split into chunks
        chunks: List[DocumentChunk] = self.splitter.split_documents(pages)
        if not chunks:
            raise ValueError(f"No valid text chunks generated for '{filename}'.")

        # 3. Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings_list = self.embeddings.embed_documents(chunk_texts)
        for chunk, emb in zip(chunks, embeddings_list):
            chunk.embedding = emb

        # 4. Store in vector database
        self.vector_store.add_chunks(chunks)

        if self.cache:
            self.cache.clear("answers")
            logger.info("Invalidated 'answers' cache following ingestion of '%s'", filename)

        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            total_pages=len(pages),
            total_chunks=len(chunks)
        )
        self._documents_meta[doc_id] = meta

        if self.cache and not self.zero_retention:
            try:
                self.cache.set("doc_meta", doc_id, {
                    "doc_id": meta.doc_id,
                    "filename": meta.filename,
                    "file_size_bytes": meta.file_size_bytes,
                    "content_hash": meta.content_hash,
                    "total_pages": meta.total_pages,
                    "total_chunks": meta.total_chunks,
                    "created_at": meta.created_at
                })
            except Exception as e:
                logger.warning("Failed to persist doc_meta to cache for '%s': %s", doc_id, e)

        logger.info("Successfully ingested '%s' (doc_id=%s, pages=%d, chunks=%d)",
                    filename, doc_id, len(pages), len(chunks))
        return meta

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents combining metadata, disk files, and vector store counts."""
        self._hydrate_metadata()
        vector_docs = self.vector_store.list_documents()
        docs_by_id = {d["doc_id"]: d for d in vector_docs}

        # Merge with in-memory metadata if available
        results: List[Dict[str, Any]] = []
        all_ids = set(docs_by_id.keys()) | set(self._documents_meta.keys())

        for d_id in all_ids:
            if d_id in self._documents_meta:
                m = self._documents_meta[d_id]
                results.append({
                    "doc_id": m.doc_id,
                    "filename": m.filename,
                    "file_size_bytes": m.file_size_bytes,
                    "total_pages": m.total_pages,
                    "total_chunks": m.total_chunks,
                    "created_at": m.created_at
                })
            elif d_id in docs_by_id:
                v = docs_by_id[d_id]
                results.append({
                    "doc_id": v["doc_id"],
                    "filename": v.get("filename", "Unknown"),
                    "file_size_bytes": 0,
                    "total_pages": v.get("total_pages", 1),
                    "total_chunks": v.get("chunk_count", 0),
                    "created_at": ""
                })

        return results

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector store, disk, and metadata cache."""
        if not doc_id or not DOC_ID_REGEX.match(doc_id):
            raise ValueError(f"Invalid doc_id format: '{doc_id}'")

        self.vector_store.delete_document(doc_id)
        if doc_id in self._documents_meta:
            del self._documents_meta[doc_id]

        if self.cache:
            if not self.zero_retention:
                try:
                    self.cache.delete("doc_meta", doc_id)
                except Exception as e:
                    logger.warning("Failed to delete doc_meta for '%s': %s", doc_id, e)
            self.cache.clear("answers")
            logger.info("Invalidated 'answers' cache following deletion of doc_id='%s'", doc_id)

        # Clean up disk files only if not in zero-retention mode
        if not self.zero_retention:
            for f in self.upload_dir.glob(f"{doc_id}_*"):
                try:
                    f.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Could not delete file %s: %s", f, e)

        logger.info("Deleted document doc_id='%s'", doc_id)
        return True
