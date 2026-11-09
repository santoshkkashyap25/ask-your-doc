"""Document loader implementation using pypdf and pdfplumber."""

import logging
import re
from pathlib import Path
from typing import List
from src.core.domain.models import Document
from src.core.interfaces.loader import BaseDocumentLoader

logger = logging.getLogger("AskMyPDF.Loader")


def clean_extracted_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts, font icon glitches, and unicode noise."""
    if not text:
        return ""
    # Normalize unicode non-breaking spaces and narrow spaces
    text = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\ufeff", "")
    # Remove non-printable control characters except tab and newline
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Fix common icon font extraction artifacts in resumes (e.g. FontAwesome ligatures)
    text = re.sub(r"[\u2640\u2642\u00b6\u25cb\u25cf\u2022]*\s*obile[- ]?alt", " Mobile: ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\u2640\u2642\u00b6\u25cb\u25cf\u2022]*\s*envelope", " Email: ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\u2640\u2642\u00b6\u25cb\u25cf\u2022]*\s*phone", " Phone: ", text, flags=re.IGNORECASE)
    # Normalize multiple horizontal whitespaces on single lines
    lines = [re.sub(r"[ \t]+", " ", line) for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PyPDFLoader(BaseDocumentLoader):
    """Loads PDF documents using PyPDF with fallback to pdfplumber for resilient extraction."""

    def load(self, file_path: Path, doc_id: str) -> List[Document]:
        """Load and extract text page-by-page from a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents: List[Document] = []
        filename = file_path.name
        if doc_id and filename.startswith(f"{doc_id}_"):
            filename = filename[len(doc_id) + 1:]

        # Attempt extraction via pypdf first
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_num = page_idx + 1
                cleaned_text = clean_extracted_pdf_text(page_text)
                if cleaned_text:
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_num,
                            text=cleaned_text,
                            metadata={"total_pages": total_pages, "extractor": "pypdf"}
                        )
                    )
            
            if documents:
                logger.info("Loaded %d pages from '%s' using pypdf", len(documents), filename)
                return documents
        except Exception as e:
            logger.warning("pypdf extraction failed or was incomplete for '%s': %s. Trying pdfplumber fallback.", filename, e)

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                total_pages = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_num = page_idx + 1
                    cleaned_text = clean_extracted_pdf_text(page_text)
                    if cleaned_text:
                        documents.append(
                            Document(
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=cleaned_text,
                                metadata={"total_pages": total_pages, "extractor": "pdfplumber"}
                            )
                        )
            logger.info("Loaded %d pages from '%s' using pdfplumber", len(documents), filename)
            return documents
        except Exception as e:
            logger.error("All PDF extraction methods failed for '%s': %s", filename, e)
            raise RuntimeError(f"Failed to extract text from PDF '{filename}': {e}") from e

    def load_bytes(self, file_bytes: bytes, filename: str, doc_id: str) -> List[Document]:
        """Extract text from in-memory PDF bytes using pypdf with pdfplumber fallback."""
        import io
        documents: List[Document] = []
        clean_filename = filename
        if doc_id and clean_filename.startswith(f"{doc_id}_"):
            clean_filename = clean_filename[len(doc_id) + 1:]

        # Attempt extraction via pypdf first
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_num = page_idx + 1
                cleaned_text = clean_extracted_pdf_text(page_text)
                if cleaned_text:
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            filename=clean_filename,
                            page_number=page_num,
                            text=cleaned_text,
                            metadata={"total_pages": total_pages, "extractor": "pypdf"}
                        )
                    )

            if documents:
                logger.info("Loaded %d pages from in-memory '%s' using pypdf", len(documents), clean_filename)
                return documents
        except Exception as e:
            logger.warning("pypdf in-memory extraction failed for '%s': %s. Trying pdfplumber fallback.", clean_filename, e)

        # Fallback to pdfplumber in-memory
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                total_pages = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_num = page_idx + 1
                    cleaned_text = clean_extracted_pdf_text(page_text)
                    if cleaned_text:
                        documents.append(
                            Document(
                                doc_id=doc_id,
                                filename=clean_filename,
                                page_number=page_num,
                                text=cleaned_text,
                                metadata={"total_pages": total_pages, "extractor": "pdfplumber"}
                            )
                        )
            logger.info("Loaded %d pages from in-memory '%s' using pdfplumber", len(documents), clean_filename)
            return documents
        except Exception as e:
            logger.error("All in-memory PDF extraction methods failed for '%s': %s", clean_filename, e)
            raise RuntimeError(f"Failed to extract text from PDF '{clean_filename}': {e}") from e
