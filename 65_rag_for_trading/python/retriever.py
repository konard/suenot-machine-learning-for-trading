"""
Document Retrieval Module for RAG Trading System

This module provides semantic search capabilities for financial documents
using embedding-based retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Document:
    """
    Financial document with metadata.

    Attributes:
        id: Unique document identifier
        text: Document content
        ticker: Associated stock ticker (optional)
        source: Document source (e.g., "Reuters", "SEC")
        date: Publication/filing date
        doc_type: Document type ("news", "filing", "earnings", "research")
        metadata: Additional metadata dictionary
    """
    id: str
    text: str
    ticker: Optional[str]
    source: str
    date: datetime
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        text: str,
        ticker: Optional[str] = None,
        source: str = "unknown",
        date: Optional[datetime] = None,
        doc_type: str = "news",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Document":
        """
        Factory method to create a document with auto-generated ID.

        Args:
            text: Document content
            ticker: Stock ticker symbol
            source: Document source
            date: Publication date (defaults to now)
            doc_type: Type of document
            metadata: Additional metadata

        Returns:
            Document instance with generated ID
        """
        doc_id = hashlib.md5(
            f"{text[:100]}{source}{date}".encode()
        ).hexdigest()[:12]

        return cls(
            id=doc_id,
            text=text,
            ticker=ticker,
            source=source,
            date=date or datetime.now(),
            doc_type=doc_type,
            metadata=metadata or {}
        )


@dataclass
class SearchResult:
    """
    Search result with relevance score.

    Attributes:
        document: The matched document
        score: Relevance score (0-1, higher is better)
        highlights: Relevant text snippets from the document
    """
    document: Document
    score: float
    highlights: List[str] = field(default_factory=list)


class FinancialDocumentRetriever:
    """
    Retrieval system for financial documents.

    Uses semantic search with embeddings to find relevant documents
    based on natural language queries. Supports filtering by ticker,
    document type, and date.

    Examples:
        >>> retriever = FinancialDocumentRetriever()
        >>> retriever.add_documents([
        ...     Document.create(
        ...         text="Tesla reported record Q4 deliveries...",
        ...         ticker="TSLA",
        ...         source="Reuters",
        ...         doc_type="news"
        ...     )
        ... ])
        >>> results = retriever.search("Tesla delivery numbers", top_k=5)
        >>> print(results[0].score)
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the retriever.

        Args:
            embedding_model: Name of the sentence-transformers model to use
        """
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None

    def _get_encoder(self):
        """Lazy load the embedding model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for embedding generation. "
                    "Install with: pip install sentence-transformers"
                )
        return self._encoder

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        encoder = self._get_encoder()
        return encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the retrieval index.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Generate embeddings for new documents
        texts = [doc.text for doc in documents]
        new_embeddings = self._compute_embeddings(texts)

        # Add to index
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(documents)

    def add_document(self, document: Document) -> None:
        """
        Add a single document to the index.

        Args:
            document: Document to add
        """
        self.add_documents([document])

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            ticker: Filter by ticker symbol
            doc_type: Filter by document type
            min_date: Filter by minimum date
            max_date: Filter by maximum date
            min_score: Minimum relevance score threshold

        Returns:
            List of search results sorted by relevance
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Encode query
        query_embedding = self._compute_embeddings([query])[0]

        # Compute cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(self.embeddings, query_embedding) / norms

        # Apply filters
        mask = np.ones(len(self.documents), dtype=bool)

        for i, doc in enumerate(self.documents):
            if ticker and doc.ticker != ticker:
                mask[i] = False
            if doc_type and doc.doc_type != doc_type:
                mask[i] = False
            if min_date and doc.date < min_date:
                mask[i] = False
            if max_date and doc.date > max_date:
                mask[i] = False

        # Set filtered documents to low score
        similarities = np.where(mask, similarities, -1)

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=score,
                    highlights=self._extract_highlights(
                        self.documents[idx].text, query
                    )
                ))

        return results

    def _extract_highlights(
        self,
        text: str,
        query: str,
        max_highlights: int = 3
    ) -> List[str]:
        """
        Extract relevant text snippets from a document.

        Args:
            text: Document text
            query: Search query
            max_highlights: Maximum number of highlights to return

        Returns:
            List of relevant text snippets
        """
        # Split into sentences
        sentences = []
        for delimiter in ['. ', '! ', '? ', '\n']:
            if delimiter in text:
                parts = text.split(delimiter)
                sentences = [
                    s.strip() + ('.' if not s.endswith(('.', '!', '?')) else '')
                    for s in parts if s.strip()
                ]
                break

        if not sentences:
            sentences = [text[:500] + "..." if len(text) > 500 else text]

        # Score sentences by keyword overlap
        query_words = set(query.lower().split())
        scored_sentences = []

        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > 0:
                scored_sentences.append((overlap, sent))

        # Sort by score and return top highlights
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        return [sent for _, sent in scored_sentences[:max_highlights]]

    def get_document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self.documents)

    def get_documents_by_ticker(self, ticker: str) -> List[Document]:
        """
        Get all documents for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of documents for the ticker
        """
        return [doc for doc in self.documents if doc.ticker == ticker]

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents = []
        self.embeddings = None


class SimpleRetriever:
    """
    Simple keyword-based retriever (no ML dependencies).

    Use this when sentence-transformers is not available.
    Falls back to TF-IDF style matching.
    """

    def __init__(self):
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the index."""
        self.documents.extend(documents)
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using keyword matching.

        Args:
            query: Search query
            top_k: Number of results
            ticker: Filter by ticker

        Returns:
            List of search results
        """
        query_words = set(query.lower().split())
        results = []

        for doc in self.documents:
            if ticker and doc.ticker != ticker:
                continue

            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)

            if overlap > 0:
                score = overlap / len(query_words)
                results.append(SearchResult(
                    document=doc,
                    score=min(score, 1.0),
                    highlights=[]
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def create_retriever(use_embeddings: bool = True) -> FinancialDocumentRetriever:
    """
    Factory function to create a retriever.

    Args:
        use_embeddings: Whether to use embedding-based retrieval

    Returns:
        Appropriate retriever instance
    """
    if use_embeddings:
        try:
            retriever = FinancialDocumentRetriever()
            # Test if sentence-transformers is available
            retriever._get_encoder()
            return retriever
        except ImportError:
            print("Warning: sentence-transformers not available, using SimpleRetriever")
            return SimpleRetriever()
    else:
        return SimpleRetriever()


if __name__ == "__main__":
    print("RAG Document Retriever Demo")
    print("=" * 50)

    # Create sample documents
    sample_docs = [
        Document.create(
            text="Tesla reported record Q4 deliveries of 484,507 vehicles, "
                 "beating analyst expectations. The company's Shanghai factory "
                 "contributed significantly to the growth.",
            ticker="TSLA",
            source="Reuters",
            doc_type="news"
        ),
        Document.create(
            text="Apple announced strong iPhone 15 sales in the holiday quarter, "
                 "with particular strength in emerging markets. Services revenue "
                 "also reached an all-time high.",
            ticker="AAPL",
            source="Bloomberg",
            doc_type="news"
        ),
        Document.create(
            text="NVIDIA's data center revenue surged 200% year-over-year as "
                 "AI chip demand continues to accelerate. The company raised "
                 "guidance for next quarter.",
            ticker="NVDA",
            source="WSJ",
            doc_type="news"
        ),
        Document.create(
            text="Tesla's Cybertruck production is ramping up at Gigafactory Texas. "
                 "Initial deliveries have begun, though supply remains limited.",
            ticker="TSLA",
            source="Electrek",
            doc_type="news"
        ),
        Document.create(
            text="Federal Reserve minutes showed officials remain cautious about "
                 "cutting interest rates too quickly. Markets reacted negatively "
                 "to the hawkish tone.",
            ticker=None,
            source="Fed",
            doc_type="filing"
        ),
    ]

    # Use simple retriever to avoid dependency issues in demo
    retriever = SimpleRetriever()
    retriever.add_documents(sample_docs)

    print(f"\nIndexed {len(sample_docs)} documents")

    # Test searches
    test_queries = [
        ("Tesla delivery", "TSLA"),
        ("Apple iPhone sales", "AAPL"),
        ("AI chip demand", None),
        ("interest rates", None),
    ]

    for query, ticker in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'" + (f" (ticker: {ticker})" if ticker else ""))
        print("-" * 50)

        results = retriever.search(query, top_k=3, ticker=ticker)

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.score:.3f}")
                print(f"   Source: {result.document.source}")
                print(f"   Ticker: {result.document.ticker or 'N/A'}")
                print(f"   Text: {result.document.text[:100]}...")
        else:
            print("No results found")
