# Chapter 65: Retrieval-Augmented Generation (RAG) for Trading

This chapter explores **Retrieval-Augmented Generation (RAG)**, a powerful technique that combines large language models with external knowledge retrieval to enhance trading decisions. RAG enables traders to leverage real-time financial data, news, research reports, and historical market information to generate contextually relevant trading signals and analysis.

<p align="center">
<img src="https://i.imgur.com/RAGTrade.png" width="70%">
</p>

## Contents

1. [Introduction to RAG for Trading](#introduction-to-rag-for-trading)
    * [What is RAG?](#what-is-rag)
    * [Why RAG for Trading?](#why-rag-for-trading)
    * [RAG vs Traditional NLP](#rag-vs-traditional-nlp)
2. [RAG Architecture](#rag-architecture)
    * [Core Components](#core-components)
    * [Document Processing Pipeline](#document-processing-pipeline)
    * [Vector Stores and Embeddings](#vector-stores-and-embeddings)
3. [Trading Applications](#trading-applications)
    * [Real-Time News Analysis](#real-time-news-analysis)
    * [SEC Filing Analysis](#sec-filing-analysis)
    * [Earnings Call Intelligence](#earnings-call-intelligence)
    * [Market Research Synthesis](#market-research-synthesis)
4. [Practical Examples](#practical-examples)
    * [01: Document Retrieval System](#01-document-retrieval-system)
    * [02: Trading Signal Generation](#02-trading-signal-generation)
    * [03: Portfolio Analysis with RAG](#03-portfolio-analysis-with-rag)
    * [04: Backtesting RAG Signals](#04-backtesting-rag-signals)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to RAG for Trading

### What is RAG?

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines the generative capabilities of Large Language Models (LLMs) with information retrieval systems. Instead of relying solely on the model's parametric knowledge, RAG retrieves relevant documents from an external knowledge base and uses them as context for generation.

```
RAG WORKFLOW FOR TRADING:
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  User Query: "What's the latest news about Tesla and how might it            │
│               affect tomorrow's stock price?"                                 │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Step 1: RETRIEVAL                                    │  │
│  │  ┌─────────────────────┐     ┌─────────────────────────────────────┐   │  │
│  │  │   Query Embedding    │ ──▶ │     Vector Database Search           │   │  │
│  │  │   "Tesla news..."   │     │  • Recent news articles              │   │  │
│  │  └─────────────────────┘     │  • SEC filings                       │   │  │
│  │                              │  • Analyst reports                   │   │  │
│  │                              │  • Social media sentiment            │   │  │
│  │                              └─────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                      │
│                                        ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Step 2: AUGMENTATION                                 │  │
│  │  Retrieved Documents:                                                   │  │
│  │  • "Tesla Q3 deliveries beat estimates by 15%..."                      │  │
│  │  • "Elon Musk announces new factory in Texas..."                       │  │
│  │  • "Analyst upgrades TSLA to Buy with $300 target..."                  │  │
│  │  + Original Query                                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                      │
│                                        ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Step 3: GENERATION                                   │  │
│  │  LLM generates response with retrieved context:                        │  │
│  │  "Based on recent developments, Tesla has several positive catalysts:  │  │
│  │   1. Q3 deliveries exceeded expectations (+15%)                        │  │
│  │   2. New Texas factory announcement signals expansion                  │  │
│  │   3. Multiple analyst upgrades suggest bullish sentiment               │  │
│  │   Signal: MODERATE BUY (confidence: 72%)"                              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why RAG for Trading?

Traditional LLMs have several limitations for trading applications:

| Challenge | Traditional LLM | RAG Solution |
|-----------|----------------|--------------|
| **Knowledge Cutoff** | Trained on historical data, unaware of recent events | Retrieves real-time information |
| **Hallucinations** | May generate plausible but incorrect facts | Grounds responses in retrieved documents |
| **Source Attribution** | Cannot cite sources for claims | Provides explicit document references |
| **Domain Specificity** | General knowledge may miss financial nuances | Retrieves domain-specific documents |
| **Updatability** | Requires expensive retraining | Simply update document database |

### RAG vs Traditional NLP

```
COMPARISON: RAG vs FINE-TUNED LLM vs TRADITIONAL NLP
═══════════════════════════════════════════════════════════════════════════════

                    Traditional NLP      Fine-tuned LLM       RAG
                    ────────────────    ────────────────    ────────────────
Knowledge Update    Retrain model       Retrain model       Update DB only
                    (days/weeks)        (hours/days)        (seconds)

Real-time Info      Not possible        Not possible        ✅ Supported

Cost per Update     $$$$                $$$                 $

Explainability      Low                 Medium              High (citations)

Accuracy on         Moderate            Good                Very Good
Recent Events

Hallucination       N/A                 High risk           Low risk
Risk

Scalability         Fixed capacity      Fixed capacity      Scales with DB

Best Use Case       Static tasks        Domain adaptation   Dynamic knowledge
```

## RAG Architecture

### Core Components

A RAG system for trading consists of four main components:

```
RAG ARCHITECTURE FOR TRADING
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     DOCUMENT INGESTION LAYER                         │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│   │  │ News APIs   │  │ SEC Edgar   │  │ Research    │  │ Social     │  │    │
│   │  │ (Bloomberg, │  │ (10-K, 10-Q,│  │ Reports     │  │ Media      │  │    │
│   │  │  Reuters)   │  │  8-K, etc)  │  │ (Analysts)  │  │ (Twitter)  │  │    │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │    │
│   └─────────┼────────────────┼────────────────┼───────────────┼─────────┘    │
│             └────────────────┼────────────────┼───────────────┘              │
│                              ▼                ▼                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     DOCUMENT PROCESSING                              │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│   │  │   Chunking  │──│  Cleaning   │──│  Metadata   │──│ Embedding  │  │    │
│   │  │  (by topic, │  │ (normalize, │  │ Extraction  │  │ Generation │  │    │
│   │  │   section)  │  │  dedupe)    │  │ (date,ticker│  │ (OpenAI,   │  │    │
│   │  │             │  │             │  │  source)    │  │  local)    │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                      │
│                                        ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     VECTOR STORE                                     │    │
│   │  ┌─────────────────────────────────────────────────────────────┐    │    │
│   │  │  Document Embeddings (1536-dim vectors)                      │    │    │
│   │  │  ┌──────────────────────────────────────────────────────┐   │    │    │
│   │  │  │ ID: doc_001 | Ticker: TSLA | Date: 2024-01-15        │   │    │    │
│   │  │  │ Vector: [0.021, -0.045, 0.089, ..., 0.012]           │   │    │    │
│   │  │  │ Text: "Tesla reported Q4 deliveries of 484,507..."   │   │    │    │
│   │  │  └──────────────────────────────────────────────────────┘   │    │    │
│   │  │  ┌──────────────────────────────────────────────────────┐   │    │    │
│   │  │  │ ID: doc_002 | Ticker: AAPL | Date: 2024-01-16        │   │    │    │
│   │  │  │ Vector: [0.015, 0.032, -0.067, ..., 0.045]           │   │    │    │
│   │  │  │ Text: "Apple Vision Pro pre-orders exceed..."        │   │    │    │
│   │  │  └──────────────────────────────────────────────────────┘   │    │    │
│   │  └─────────────────────────────────────────────────────────────┘    │    │
│   │  Storage Options: ChromaDB | Pinecone | Weaviate | FAISS            │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                      │
│                                        ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     RETRIEVAL & GENERATION                           │    │
│   │  ┌──────────────────────────┐    ┌──────────────────────────┐       │    │
│   │  │   Semantic Search        │    │   LLM Generation         │       │    │
│   │  │  • Query embedding       │───▶│  • Context integration   │       │    │
│   │  │  • k-NN retrieval        │    │  • Trading analysis      │       │    │
│   │  │  • Re-ranking            │    │  • Signal generation     │       │    │
│   │  │  • Filtering (date,      │    │  • Risk assessment       │       │    │
│   │  │    ticker, relevance)    │    │  • Source citation       │       │    │
│   │  └──────────────────────────┘    └──────────────────────────┘       │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Document Processing Pipeline

Effective RAG requires careful document processing:

```python
# Document processing pipeline for financial documents
class FinancialDocumentProcessor:
    """
    Process financial documents for RAG indexing.

    Handles:
    - News articles
    - SEC filings (10-K, 10-Q, 8-K)
    - Earnings transcripts
    - Research reports
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, document: str, metadata: dict) -> List[DocumentChunk]:
        """
        Process document into chunks with metadata.

        Args:
            document: Raw document text
            metadata: Document metadata (source, date, tickers)

        Returns:
            List of processed document chunks
        """
        # Step 1: Clean and normalize
        cleaned = self._clean_text(document)

        # Step 2: Extract additional metadata
        entities = self._extract_entities(cleaned)
        metadata["entities"] = entities

        # Step 3: Chunk document
        chunks = self._chunk_document(cleaned)

        # Step 4: Generate embeddings
        embeddings = self._generate_embeddings(chunks)

        return [
            DocumentChunk(
                text=chunk,
                embedding=emb,
                metadata={**metadata, "chunk_idx": i}
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
```

### Vector Stores and Embeddings

Choosing the right embedding model and vector store is crucial:

| Embedding Model | Dimensions | Best For | Cost |
|----------------|------------|----------|------|
| OpenAI text-embedding-3-large | 3072 | High accuracy | API cost |
| OpenAI text-embedding-3-small | 1536 | Balance | API cost |
| Sentence-BERT | 768 | Privacy, offline | Free |
| FinBERT Embeddings | 768 | Financial domain | Free |

| Vector Store | Scalability | Features | Deployment |
|-------------|-------------|----------|------------|
| **ChromaDB** | Small-Medium | Easy setup, metadata | Local/Cloud |
| **FAISS** | Large | High performance | Local |
| **Pinecone** | Very Large | Managed, fast | Cloud |
| **Weaviate** | Large | GraphQL, hybrid search | Self-hosted/Cloud |
| **Qdrant** | Large | Fast, Rust-based | Self-hosted/Cloud |

## Trading Applications

### Real-Time News Analysis

RAG enables sophisticated news analysis for trading:

```python
# Real-time news analysis with RAG
class NewsRAGAnalyzer:
    """
    Analyze real-time news for trading signals using RAG.
    """

    def analyze_news(self, query: str, tickers: List[str]) -> TradingSignal:
        """
        Analyze news and generate trading signals.

        Example:
            >>> analyzer = NewsRAGAnalyzer()
            >>> signal = analyzer.analyze_news(
            ...     "What's the market sentiment on TSLA today?",
            ...     tickers=["TSLA"]
            ... )
            >>> print(signal)
            TradingSignal(
                ticker="TSLA",
                direction="LONG",
                confidence=0.72,
                reasoning="Based on 3 recent news articles...",
                sources=["Reuters", "Bloomberg", "WSJ"]
            )
        """
        # Retrieve relevant documents
        docs = self.retriever.search(
            query=query,
            filters={"ticker": {"$in": tickers}},
            top_k=10
        )

        # Generate analysis with LLM
        context = self._format_context(docs)
        prompt = self._build_prompt(query, context)

        response = self.llm.generate(prompt)

        return self._parse_signal(response, docs)
```

### SEC Filing Analysis

Automated analysis of regulatory filings:

```python
# Example: 10-K Filing Analysis
filing_analysis = rag_analyzer.analyze(
    query="What are the main risk factors mentioned in Tesla's latest 10-K?",
    document_types=["10-K"],
    tickers=["TSLA"]
)

# Output:
"""
Based on Tesla's 2023 10-K filing, the main risk factors include:

1. **Production Capacity Risks** (Section 1A, Page 15)
   - Dependency on Gigafactory output
   - Supply chain constraints for batteries

2. **Regulatory Risks** (Section 1A, Page 18)
   - EV tax credit eligibility changes
   - Autonomous driving regulations

3. **Competition Risks** (Section 1A, Page 21)
   - Increasing EV competition from legacy automakers
   - Chinese EV manufacturers entering US market

4. **Key Person Risk** (Section 1A, Page 24)
   - Heavy reliance on Elon Musk

Sources: [SEC 10-K Filing dated 2024-01-29, pages 15-24]
"""
```

### Earnings Call Intelligence

Extract insights from earnings calls:

```
EARNINGS CALL RAG ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Query: "What guidance did Apple provide for next quarter?"

Retrieved Context:
┌──────────────────────────────────────────────────────────────────────────────┐
│ [1] Apple Q4 2024 Earnings Call Transcript (Oct 31, 2024)                    │
│     "Looking ahead to Q1, we expect revenue between $118-122 billion,        │
│      representing 5-8% year-over-year growth. Services should continue       │
│      its strong momentum with double-digit growth expected."                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ [2] Apple CFO Prepared Remarks                                               │
│     "Gross margin guidance for Q1 is between 45% and 46%, consistent        │
│      with our historical Q1 seasonality patterns."                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ [3] Analyst Q&A Session                                                      │
│     Q: "Can you comment on iPhone demand in China?"                          │
│     A: "We're seeing healthy demand across all geographies. China           │
│         continues to be our fastest-growing market for Services."            │
└──────────────────────────────────────────────────────────────────────────────┘

Generated Analysis:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Apple Q1 FY2025 Guidance Summary:                                            │
│                                                                               │
│ • Revenue: $118-122B (5-8% YoY growth)                                       │
│ • Gross Margin: 45-46%                                                       │
│ • Services: Double-digit growth expected                                     │
│ • Geographic: Strong China demand, especially in Services                    │
│                                                                               │
│ Trading Implication: NEUTRAL to MILDLY BULLISH                               │
│ - Guidance in-line with consensus ($120B)                                    │
│ - Services growth provides margin tailwind                                   │
│ - China commentary addresses key investor concern                            │
│                                                                               │
│ Confidence: 78%                                                              │
│ Sources: Q4 2024 Earnings Call Transcript, CFO Remarks, Q&A Session          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Market Research Synthesis

Combine multiple research sources:

```python
# Multi-source research synthesis
synthesis = rag_system.synthesize(
    query="What's the consensus view on semiconductor stocks for 2024?",
    sources=[
        "analyst_reports",
        "earnings_calls",
        "news_articles",
        "industry_reports"
    ],
    tickers=["NVDA", "AMD", "INTC", "TSM"]
)

# Returns structured analysis combining all sources
```

## Practical Examples

### 01: Document Retrieval System

```python
"""
Example 01: Building a Financial Document Retrieval System

This example demonstrates how to build a document retrieval system
for financial documents using embeddings and vector search.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    """Financial document with metadata."""
    id: str
    text: str
    ticker: Optional[str]
    source: str
    date: datetime
    doc_type: str  # "news", "filing", "earnings", "research"

@dataclass
class SearchResult:
    """Search result with relevance score."""
    document: Document
    score: float
    highlights: List[str]

class FinancialDocumentRetriever:
    """
    Retrieval system for financial documents.

    Uses semantic search with embeddings to find relevant documents
    based on natural language queries.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._encoder

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retrieval index.

        Args:
            documents: List of documents to index
        """
        encoder = self._get_encoder()

        # Generate embeddings for new documents
        texts = [doc.text for doc in documents]
        new_embeddings = encoder.encode(texts, convert_to_numpy=True)

        # Add to index
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
        min_date: Optional[datetime] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            ticker: Filter by ticker symbol
            doc_type: Filter by document type
            min_date: Filter by minimum date

        Returns:
            List of search results with scores
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        encoder = self._get_encoder()

        # Encode query
        query_embedding = encoder.encode([query], convert_to_numpy=True)[0]

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Apply filters
        mask = np.ones(len(self.documents), dtype=bool)

        for i, doc in enumerate(self.documents):
            if ticker and doc.ticker != ticker:
                mask[i] = False
            if doc_type and doc.doc_type != doc_type:
                mask[i] = False
            if min_date and doc.date < min_date:
                mask[i] = False

        # Set filtered documents to low score
        similarities[~mask] = -1

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(similarities[idx]),
                    highlights=self._extract_highlights(
                        self.documents[idx].text, query
                    )
                ))

        return results

    def _extract_highlights(self, text: str, query: str) -> List[str]:
        """Extract relevant text snippets."""
        # Simple keyword-based highlighting
        sentences = text.split('. ')
        query_words = set(query.lower().split())

        scored_sentences = []
        for sent in sentences:
            score = sum(1 for word in query_words if word in sent.lower())
            if score > 0:
                scored_sentences.append((score, sent))

        scored_sentences.sort(reverse=True)
        return [sent for _, sent in scored_sentences[:3]]
```

### 02: Trading Signal Generation

```python
"""
Example 02: RAG-based Trading Signal Generation

This example shows how to generate trading signals by combining
document retrieval with LLM-based analysis.
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

@dataclass
class TradingSignal:
    """Generated trading signal with reasoning."""
    ticker: str
    direction: SignalDirection
    confidence: float  # 0-1
    reasoning: str
    sources: List[str]
    timestamp: datetime

class RAGTradingSignalGenerator:
    """
    Generate trading signals using RAG.

    Combines document retrieval with LLM analysis to produce
    actionable trading signals with explanations.
    """

    def __init__(
        self,
        retriever: FinancialDocumentRetriever,
        llm_client: Optional[object] = None
    ):
        self.retriever = retriever
        self.llm_client = llm_client

    def generate_signal(
        self,
        ticker: str,
        query: Optional[str] = None
    ) -> TradingSignal:
        """
        Generate a trading signal for a ticker.

        Args:
            ticker: Stock ticker symbol
            query: Optional custom query (default: general sentiment)

        Returns:
            Trading signal with reasoning and sources
        """
        # Default query if not provided
        if query is None:
            query = f"What is the current market sentiment and outlook for {ticker}?"

        # Retrieve relevant documents
        results = self.retriever.search(
            query=query,
            ticker=ticker,
            top_k=5
        )

        if not results:
            return TradingSignal(
                ticker=ticker,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning="No relevant documents found for analysis.",
                sources=[],
                timestamp=datetime.now()
            )

        # Build context from retrieved documents
        context = self._build_context(results)

        # Generate analysis (using LLM or rule-based)
        if self.llm_client:
            analysis = self._llm_analysis(ticker, query, context)
        else:
            analysis = self._rule_based_analysis(ticker, results)

        return analysis

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] Source: {result.document.source} "
                f"({result.document.date.strftime('%Y-%m-%d')})\n"
                f"{result.document.text[:500]}..."
            )

        return "\n\n".join(context_parts)

    def _rule_based_analysis(
        self,
        ticker: str,
        results: List[SearchResult]
    ) -> TradingSignal:
        """
        Simple rule-based sentiment analysis.

        Used when LLM is not available.
        """
        positive_keywords = [
            "beat", "exceeded", "growth", "upgrade", "bullish",
            "strong", "positive", "increase", "surge", "rally"
        ]
        negative_keywords = [
            "miss", "below", "decline", "downgrade", "bearish",
            "weak", "negative", "decrease", "drop", "fall"
        ]

        positive_count = 0
        negative_count = 0

        for result in results:
            text_lower = result.document.text.lower()
            positive_count += sum(
                1 for word in positive_keywords if word in text_lower
            )
            negative_count += sum(
                1 for word in negative_keywords if word in text_lower
            )

        total = positive_count + negative_count

        if total == 0:
            direction = SignalDirection.NEUTRAL
            confidence = 0.3
        elif positive_count > negative_count:
            direction = SignalDirection.LONG
            confidence = min(0.9, 0.5 + (positive_count - negative_count) / total * 0.4)
        else:
            direction = SignalDirection.SHORT
            confidence = min(0.9, 0.5 + (negative_count - positive_count) / total * 0.4)

        sources = list(set(r.document.source for r in results))

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            reasoning=(
                f"Based on {len(results)} documents: "
                f"{positive_count} positive signals, {negative_count} negative signals."
            ),
            sources=sources,
            timestamp=datetime.now()
        )

    def _llm_analysis(
        self,
        ticker: str,
        query: str,
        context: str
    ) -> TradingSignal:
        """Generate analysis using LLM."""
        prompt = f"""Analyze the following financial documents and generate a trading signal.

Ticker: {ticker}
Query: {query}

Retrieved Documents:
{context}

Based on the above documents, provide:
1. Trading direction (LONG, SHORT, or NEUTRAL)
2. Confidence level (0-100%)
3. Brief reasoning (2-3 sentences)

Format your response as:
DIRECTION: [direction]
CONFIDENCE: [percentage]
REASONING: [your analysis]
"""

        response = self.llm_client.generate(prompt)
        return self._parse_llm_response(ticker, response, context)
```

### 03: Portfolio Analysis with RAG

```python
"""
Example 03: Portfolio-Level RAG Analysis

Analyze entire portfolios using RAG for holistic insights.
"""

@dataclass
class PortfolioPosition:
    """Single portfolio position."""
    ticker: str
    shares: float
    entry_price: float
    current_price: float

@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis."""
    total_value: float
    risk_assessment: str
    sector_exposure: Dict[str, float]
    key_risks: List[str]
    opportunities: List[str]
    recommended_actions: List[str]
    sources_used: int

class PortfolioRAGAnalyzer:
    """
    Analyze portfolio using RAG for comprehensive insights.
    """

    def __init__(
        self,
        retriever: FinancialDocumentRetriever,
        signal_generator: RAGTradingSignalGenerator
    ):
        self.retriever = retriever
        self.signal_generator = signal_generator

    def analyze_portfolio(
        self,
        positions: List[PortfolioPosition]
    ) -> PortfolioAnalysis:
        """
        Perform comprehensive portfolio analysis.

        Args:
            positions: List of portfolio positions

        Returns:
            Complete portfolio analysis with recommendations
        """
        # Calculate basic metrics
        total_value = sum(
            pos.shares * pos.current_price for pos in positions
        )

        # Analyze each position
        position_signals = {}
        all_risks = []
        all_opportunities = []
        sources_count = 0

        for position in positions:
            # Get signal for each position
            signal = self.signal_generator.generate_signal(position.ticker)
            position_signals[position.ticker] = signal
            sources_count += len(signal.sources)

            # Retrieve risk-specific documents
            risk_results = self.retriever.search(
                query=f"risks and challenges for {position.ticker}",
                ticker=position.ticker,
                top_k=3
            )

            for result in risk_results:
                all_risks.append({
                    "ticker": position.ticker,
                    "risk": result.highlights[0] if result.highlights else result.document.text[:100]
                })

            # Retrieve opportunity documents
            opp_results = self.retriever.search(
                query=f"growth opportunities and catalysts for {position.ticker}",
                ticker=position.ticker,
                top_k=3
            )

            for result in opp_results:
                all_opportunities.append({
                    "ticker": position.ticker,
                    "opportunity": result.highlights[0] if result.highlights else result.document.text[:100]
                })

        # Generate recommendations
        recommendations = self._generate_recommendations(
            positions, position_signals
        )

        # Assess overall risk
        risk_assessment = self._assess_risk(position_signals)

        return PortfolioAnalysis(
            total_value=total_value,
            risk_assessment=risk_assessment,
            sector_exposure=self._calculate_sector_exposure(positions),
            key_risks=[r["risk"] for r in all_risks[:5]],
            opportunities=[o["opportunity"] for o in all_opportunities[:5]],
            recommended_actions=recommendations,
            sources_used=sources_count
        )

    def _generate_recommendations(
        self,
        positions: List[PortfolioPosition],
        signals: Dict[str, TradingSignal]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        for position in positions:
            signal = signals.get(position.ticker)
            if not signal:
                continue

            pnl_pct = (position.current_price - position.entry_price) / position.entry_price

            if signal.direction == SignalDirection.SHORT and signal.confidence > 0.7:
                recommendations.append(
                    f"Consider reducing {position.ticker} position "
                    f"(Signal: {signal.direction.value}, Confidence: {signal.confidence:.0%})"
                )
            elif signal.direction == SignalDirection.LONG and signal.confidence > 0.7:
                if pnl_pct < 0:
                    recommendations.append(
                        f"Consider averaging down on {position.ticker} "
                        f"(Signal: {signal.direction.value}, Confidence: {signal.confidence:.0%})"
                    )
                else:
                    recommendations.append(
                        f"Hold {position.ticker}, bullish outlook "
                        f"(Signal: {signal.direction.value}, Confidence: {signal.confidence:.0%})"
                    )

        return recommendations

    def _assess_risk(self, signals: Dict[str, TradingSignal]) -> str:
        """Assess overall portfolio risk."""
        bearish_count = sum(
            1 for s in signals.values()
            if s.direction == SignalDirection.SHORT
        )
        bullish_count = sum(
            1 for s in signals.values()
            if s.direction == SignalDirection.LONG
        )

        if bearish_count > bullish_count:
            return "HIGH - Multiple positions showing bearish signals"
        elif bearish_count == bullish_count:
            return "MODERATE - Mixed signals across positions"
        else:
            return "LOW - Majority of positions showing bullish signals"

    def _calculate_sector_exposure(
        self,
        positions: List[PortfolioPosition]
    ) -> Dict[str, float]:
        """Calculate sector exposure (simplified)."""
        # In production, would use actual sector mappings
        return {"Technology": 0.4, "Healthcare": 0.3, "Finance": 0.3}
```

### 04: Backtesting RAG Signals

```python
"""
Example 04: Backtesting RAG-Generated Trading Signals

Backtest trading strategies based on RAG-generated signals.
"""

@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float

class RAGBacktester:
    """
    Backtest RAG-based trading strategies.
    """

    def __init__(
        self,
        signal_generator: RAGTradingSignalGenerator,
        initial_capital: float = 100000.0
    ):
        self.signal_generator = signal_generator
        self.initial_capital = initial_capital

    def backtest(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        signal_dates: List[datetime]
    ) -> BacktestResult:
        """
        Backtest a RAG-based strategy.

        Args:
            ticker: Stock ticker
            price_data: OHLCV DataFrame with DatetimeIndex
            signal_dates: Dates to generate signals

        Returns:
            Backtest results with metrics
        """
        capital = self.initial_capital
        position = 0  # Number of shares
        trades = []
        equity_curve = [capital]

        for date in signal_dates:
            if date not in price_data.index:
                continue

            price = price_data.loc[date, 'close']

            # Generate signal for this date
            signal = self.signal_generator.generate_signal(ticker)

            # Execute trades based on signal
            if signal.direction == SignalDirection.LONG and position == 0:
                # Buy
                shares_to_buy = int(capital * 0.95 / price)  # 95% of capital
                if shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= shares_to_buy * price
                    trades.append({
                        "date": date,
                        "action": "BUY",
                        "shares": shares_to_buy,
                        "price": price,
                        "confidence": signal.confidence
                    })

            elif signal.direction == SignalDirection.SHORT and position > 0:
                # Sell
                capital += position * price
                trades.append({
                    "date": date,
                    "action": "SELL",
                    "shares": position,
                    "price": price,
                    "confidence": signal.confidence
                })
                position = 0

            # Update equity
            equity = capital + position * price
            equity_curve.append(equity)

        # Close any remaining position
        if position > 0 and len(price_data) > 0:
            final_price = price_data.iloc[-1]['close']
            capital += position * final_price

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital

        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0
        )

        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calculate win rate
        buy_prices = [t["price"] for t in trades if t["action"] == "BUY"]
        sell_prices = [t["price"] for t in trades if t["action"] == "SELL"]

        wins = sum(
            1 for b, s in zip(buy_prices, sell_prices) if s > b
        )
        total_completed_trades = min(len(buy_prices), len(sell_prices))
        win_rate = wins / total_completed_trades if total_completed_trades > 0 else 0

        trade_returns = [
            (s - b) / b for b, s in zip(buy_prices, sell_prices)
        ]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return
        )
```

## Rust Implementation

See the `rust_rag_trading/` directory for the complete Rust implementation featuring:

- **Async/await support** with Tokio for high-performance I/O
- **Vector similarity search** using efficient SIMD operations
- **Document processing** with chunking and metadata extraction
- **Bybit API integration** for cryptocurrency data
- **Yahoo Finance** data loading for stock market data

```rust
// Example usage of Rust RAG implementation
use rag_trading::{
    DocumentRetriever, RAGSignalGenerator, Document,
    BybitDataLoader, YahooFinanceLoader
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize retriever
    let mut retriever = DocumentRetriever::new()?;

    // Add documents
    retriever.add_documents(vec![
        Document::new(
            "Tesla reported record Q4 deliveries...",
            Some("TSLA"),
            "news",
            chrono::Utc::now()
        ),
    ]).await?;

    // Generate signal
    let generator = RAGSignalGenerator::new(retriever);
    let signal = generator.generate_signal("TSLA").await?;

    println!("Signal: {:?}", signal);

    // Load market data for backtesting
    let yahoo = YahooFinanceLoader::new();
    let tsla_data = yahoo.load("TSLA", "1y").await?;

    let bybit = BybitDataLoader::new(false);
    let btc_data = bybit.load("BTCUSDT", 30).await?;

    Ok(())
}
```

## Python Implementation

See the `python/` directory for the complete Python implementation including:

- **`retriever.py`**: Document retrieval with semantic search
- **`signals.py`**: Trading signal generation
- **`backtest.py`**: Backtesting framework
- **`data_loader.py`**: Yahoo Finance and Bybit data loading
- **`examples/`**: Demo scripts

```python
# Example usage
from rag_trading import (
    FinancialDocumentRetriever,
    RAGTradingSignalGenerator,
    RAGBacktester,
    DataLoader
)

# Initialize components
retriever = FinancialDocumentRetriever()
generator = RAGTradingSignalGenerator(retriever)

# Add documents
retriever.add_documents([
    Document(
        id="doc_001",
        text="Tesla reported record Q4 deliveries of 484,507 vehicles...",
        ticker="TSLA",
        source="Reuters",
        date=datetime.now(),
        doc_type="news"
    )
])

# Generate signal
signal = generator.generate_signal("TSLA")
print(f"Signal: {signal.direction.value}, Confidence: {signal.confidence:.0%}")

# Load market data
loader = DataLoader()
tsla_data = loader.load("TSLA", source="yahoo", period="1y")
btc_data = loader.load("BTCUSDT", source="bybit", days=30)

# Backtest
backtester = RAGBacktester(generator)
results = backtester.backtest("TSLA", tsla_data.ohlcv, signal_dates)
print(f"Total Return: {results.total_return:.2%}")
```

## Best Practices

### 1. Document Quality

```
DOCUMENT QUALITY CHECKLIST:
✓ Remove boilerplate (disclaimers, legal text)
✓ Normalize dates to consistent format
✓ Extract and validate ticker symbols
✓ Remove duplicate or near-duplicate documents
✓ Verify source reliability
✓ Tag with document type and date
```

### 2. Chunking Strategy

```python
# Recommended chunking strategies for financial documents
CHUNKING_STRATEGIES = {
    "news": {
        "chunk_size": 512,
        "overlap": 50,
        "strategy": "paragraph"  # Split by paragraphs
    },
    "10-K": {
        "chunk_size": 1024,
        "overlap": 100,
        "strategy": "section"  # Split by SEC sections
    },
    "earnings_call": {
        "chunk_size": 768,
        "overlap": 75,
        "strategy": "speaker"  # Split by speaker turns
    }
}
```

### 3. Retrieval Optimization

- Use **hybrid search** (semantic + keyword) for better results
- Apply **metadata filtering** (date, ticker) before semantic search
- Implement **re-ranking** for top results
- Cache frequent queries

### 4. Signal Generation

- Always provide **source attribution**
- Include **confidence scores**
- Log all signal generations for analysis
- Implement **position sizing** based on confidence

### 5. Backtesting

- Use **out-of-sample** data for validation
- Account for **look-ahead bias** in document timestamps
- Include **transaction costs**
- Test across different **market regimes**

## Resources

### Papers

1. **Retrieval-Augmented Generation for Large Language Models: A Survey**
   - arXiv: [2312.10997](https://arxiv.org/abs/2312.10997)
   - Comprehensive overview of RAG techniques

2. **REALM: Retrieval-Augmented Language Model Pre-Training**
   - arXiv: [2002.08909](https://arxiv.org/abs/2002.08909)
   - Foundation paper for retrieval-augmented LMs

3. **FinGPT: Open-Source Financial Large Language Models**
   - arXiv: [2306.06031](https://arxiv.org/abs/2306.06031)
   - Open-source financial LLM with RAG capabilities

### Tools & Libraries

| Tool | Purpose | Link |
|------|---------|------|
| LangChain | RAG framework | [langchain.com](https://www.langchain.com/) |
| LlamaIndex | Document indexing | [llamaindex.ai](https://www.llamaindex.ai/) |
| ChromaDB | Vector store | [trychroma.com](https://www.trychroma.com/) |
| Sentence-Transformers | Embeddings | [sbert.net](https://www.sbert.net/) |
| yfinance | Stock data | [pypi.org/project/yfinance](https://pypi.org/project/yfinance/) |

### Data Sources

- **SEC EDGAR**: Free SEC filings ([sec.gov/edgar](https://www.sec.gov/edgar))
- **Yahoo Finance**: Stock data via yfinance
- **Bybit API**: Cryptocurrency data
- **Alpha Vantage**: News and market data
- **Polygon.io**: Real-time market data

---

## Summary

RAG for trading combines the power of LLMs with real-time information retrieval to create intelligent trading systems that:

1. **Stay Current**: Access real-time news and filings
2. **Ground Responses**: Base analysis on actual documents
3. **Provide Transparency**: Cite sources for all claims
4. **Scale Efficiently**: Update knowledge without retraining

Key takeaways:
- Choose appropriate **embedding models** for financial domain
- Implement proper **document processing** pipelines
- Use **hybrid retrieval** for best results
- Always **backtest** strategies before deployment
- Monitor and **log** all signals for continuous improvement

The code examples in this chapter provide a foundation for building production-grade RAG systems for trading applications.
