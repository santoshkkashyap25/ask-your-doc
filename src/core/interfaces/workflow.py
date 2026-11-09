"""Abstract base class for RAG orchestration workflows."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Iterator, List, Optional
from src.core.domain.models import QueryResult


class BaseRAGWorkflow(ABC):
    """Interface contract for end-to-end RAG query execution workflow (e.g. LangGraph)."""

    @abstractmethod
    def run(
        self,
        query: str,
        filter_doc_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> QueryResult:
        """Execute the RAG workflow and return the complete grounded result.

        Args:
            query: User's question.
            filter_doc_id: Optional document ID to restrict context.
            history: Optional conversation history turns.

        Returns:
            QueryResult containing answer, citations, caching status, and metrics.
        """
        pass

    @abstractmethod
    def stream(
        self,
        query: str,
        filter_doc_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Iterator[Dict]:
        """Execute the RAG workflow yielding events (token events, source events, completion events).

        Args:
            query: User's question.
            filter_doc_id: Optional document ID to restrict context.
            history: Optional conversation history turns.

        Yields:
            Dict events such as {"type": "token", "content": "hello"} or {"type": "sources", "data": [...]}
        """
        pass
