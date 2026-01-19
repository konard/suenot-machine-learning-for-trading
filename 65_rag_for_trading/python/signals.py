"""
Trading Signal Generation Module

This module provides RAG-based trading signal generation by combining
document retrieval with sentiment analysis and LLM-based reasoning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .retriever import FinancialDocumentRetriever, SearchResult, SimpleRetriever


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = "LONG"       # Buy signal
    SHORT = "SHORT"     # Sell signal
    NEUTRAL = "NEUTRAL" # No action


@dataclass
class TradingSignal:
    """
    Generated trading signal with reasoning and sources.

    Attributes:
        ticker: Stock ticker symbol
        direction: Signal direction (LONG, SHORT, NEUTRAL)
        confidence: Confidence score (0-1)
        reasoning: Explanation for the signal
        sources: List of source names used for analysis
        timestamp: When the signal was generated
        metadata: Additional signal metadata
    """
    ticker: str
    direction: SignalDirection
    confidence: float
    reasoning: str
    sources: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "ticker": self.ticker,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return (
            f"TradingSignal({self.ticker}: {self.direction.value}, "
            f"confidence={self.confidence:.0%})"
        )


class SentimentAnalyzer:
    """
    Simple sentiment analyzer for financial text.

    Uses keyword-based analysis for sentiment scoring.
    For production use, consider integrating FinBERT or similar models.
    """

    POSITIVE_KEYWORDS = [
        "beat", "exceeded", "growth", "upgrade", "bullish", "strong",
        "positive", "increase", "surge", "rally", "outperform", "buy",
        "profit", "gains", "record", "breakthrough", "optimistic",
        "accelerate", "expand", "success", "momentum", "upside"
    ]

    NEGATIVE_KEYWORDS = [
        "miss", "below", "decline", "downgrade", "bearish", "weak",
        "negative", "decrease", "drop", "fall", "underperform", "sell",
        "loss", "losses", "concern", "risk", "warning", "slowdown",
        "cut", "reduce", "disappoint", "downside", "pressure"
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment scores and details
        """
        text_lower = text.lower()

        positive_count = sum(
            1 for word in self.POSITIVE_KEYWORDS if word in text_lower
        )
        negative_count = sum(
            1 for word in self.NEGATIVE_KEYWORDS if word in text_lower
        )

        total = positive_count + negative_count

        if total == 0:
            sentiment = "NEUTRAL"
            score = 0.5
        elif positive_count > negative_count:
            sentiment = "POSITIVE"
            score = 0.5 + (positive_count - negative_count) / (total * 2)
        else:
            sentiment = "NEGATIVE"
            score = 0.5 - (negative_count - positive_count) / (total * 2)

        return {
            "sentiment": sentiment,
            "score": min(max(score, 0.0), 1.0),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "keywords_found": total
        }


class RAGTradingSignalGenerator:
    """
    Generate trading signals using RAG.

    Combines document retrieval with sentiment analysis and optional
    LLM-based reasoning to produce actionable trading signals.

    Examples:
        >>> retriever = FinancialDocumentRetriever()
        >>> # Add documents to retriever...
        >>> generator = RAGTradingSignalGenerator(retriever)
        >>> signal = generator.generate_signal("TSLA")
        >>> print(signal)
        TradingSignal(TSLA: LONG, confidence=72%)
    """

    def __init__(
        self,
        retriever: FinancialDocumentRetriever,
        llm_client: Optional[Any] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        """
        Initialize the signal generator.

        Args:
            retriever: Document retriever instance
            llm_client: Optional LLM client for advanced analysis
            sentiment_analyzer: Optional sentiment analyzer (default: built-in)
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()

    def generate_signal(
        self,
        ticker: str,
        query: Optional[str] = None,
        top_k: int = 5
    ) -> TradingSignal:
        """
        Generate a trading signal for a ticker.

        Args:
            ticker: Stock ticker symbol
            query: Optional custom query (default: general sentiment query)
            top_k: Number of documents to retrieve

        Returns:
            Trading signal with reasoning and sources
        """
        # Build query if not provided
        if query is None:
            query = f"What is the current market sentiment and outlook for {ticker}?"

        # Retrieve relevant documents
        results = self.retriever.search(
            query=query,
            ticker=ticker,
            top_k=top_k
        )

        if not results:
            return TradingSignal(
                ticker=ticker,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning="No relevant documents found for analysis.",
                sources=[],
                metadata={"documents_analyzed": 0}
            )

        # Analyze with LLM if available, otherwise use rule-based
        if self.llm_client:
            return self._llm_analysis(ticker, query, results)
        else:
            return self._rule_based_analysis(ticker, results)

    def _rule_based_analysis(
        self,
        ticker: str,
        results: List[SearchResult]
    ) -> TradingSignal:
        """
        Generate signal using rule-based sentiment analysis.

        Args:
            ticker: Stock ticker
            results: Retrieved documents

        Returns:
            Trading signal
        """
        # Analyze sentiment of each document
        total_score = 0.0
        total_weight = 0.0
        positive_signals = 0
        negative_signals = 0
        sources = set()

        for result in results:
            sentiment = self.sentiment_analyzer.analyze(result.document.text)

            # Weight by retrieval relevance score
            weight = result.score
            total_score += sentiment["score"] * weight
            total_weight += weight

            if sentiment["sentiment"] == "POSITIVE":
                positive_signals += 1
            elif sentiment["sentiment"] == "NEGATIVE":
                negative_signals += 1

            sources.add(result.document.source)

        # Calculate weighted average sentiment
        if total_weight > 0:
            avg_sentiment = total_score / total_weight
        else:
            avg_sentiment = 0.5

        # Determine signal direction
        if avg_sentiment > 0.6:
            direction = SignalDirection.LONG
            confidence = min(0.9, avg_sentiment)
        elif avg_sentiment < 0.4:
            direction = SignalDirection.SHORT
            confidence = min(0.9, 1.0 - avg_sentiment)
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.3 + abs(avg_sentiment - 0.5)

        # Build reasoning
        reasoning = (
            f"Analysis of {len(results)} documents: "
            f"{positive_signals} positive, {negative_signals} negative signals. "
            f"Average sentiment score: {avg_sentiment:.2f}."
        )

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            sources=list(sources),
            metadata={
                "documents_analyzed": len(results),
                "positive_signals": positive_signals,
                "negative_signals": negative_signals,
                "avg_sentiment": avg_sentiment
            }
        )

    def _llm_analysis(
        self,
        ticker: str,
        query: str,
        results: List[SearchResult]
    ) -> TradingSignal:
        """
        Generate signal using LLM analysis.

        Args:
            ticker: Stock ticker
            query: User query
            results: Retrieved documents

        Returns:
            Trading signal with LLM-generated reasoning
        """
        # Build context from retrieved documents
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] Source: {result.document.source} "
                f"({result.document.date.strftime('%Y-%m-%d')})\n"
                f"{result.document.text[:500]}"
            )
            sources.append(result.document.source)

        context = "\n\n".join(context_parts)

        # Build prompt
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

        # Call LLM
        try:
            response = self.llm_client.generate(prompt)
            return self._parse_llm_response(ticker, response, sources)
        except Exception as e:
            # Fall back to rule-based on LLM error
            print(f"LLM analysis failed: {e}. Falling back to rule-based analysis.")
            return self._rule_based_analysis(ticker, results)

    def _parse_llm_response(
        self,
        ticker: str,
        response: str,
        sources: List[str]
    ) -> TradingSignal:
        """
        Parse LLM response into a TradingSignal.

        Args:
            ticker: Stock ticker
            response: LLM response text
            sources: Source documents used

        Returns:
            Parsed trading signal
        """
        lines = response.strip().split('\n')

        direction = SignalDirection.NEUTRAL
        confidence = 0.5
        reasoning = "Unable to parse LLM response."

        for line in lines:
            line = line.strip()
            if line.startswith("DIRECTION:"):
                direction_str = line.replace("DIRECTION:", "").strip().upper()
                if "LONG" in direction_str or "BUY" in direction_str:
                    direction = SignalDirection.LONG
                elif "SHORT" in direction_str or "SELL" in direction_str:
                    direction = SignalDirection.SHORT
                else:
                    direction = SignalDirection.NEUTRAL

            elif line.startswith("CONFIDENCE:"):
                conf_str = line.replace("CONFIDENCE:", "").strip()
                conf_str = conf_str.replace("%", "")
                try:
                    confidence = float(conf_str) / 100.0
                except ValueError:
                    confidence = 0.5

            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            sources=list(set(sources)),
            metadata={"analysis_type": "llm"}
        )

    def batch_generate(
        self,
        tickers: List[str],
        **kwargs
    ) -> Dict[str, TradingSignal]:
        """
        Generate signals for multiple tickers.

        Args:
            tickers: List of ticker symbols
            **kwargs: Additional arguments passed to generate_signal

        Returns:
            Dict mapping ticker to signal
        """
        return {
            ticker: self.generate_signal(ticker, **kwargs)
            for ticker in tickers
        }


if __name__ == "__main__":
    from .retriever import Document

    print("RAG Trading Signal Generator Demo")
    print("=" * 50)

    # Create sample documents
    sample_docs = [
        Document.create(
            text="Tesla reported record Q4 deliveries of 484,507 vehicles, "
                 "beating analyst expectations by 15%. Strong demand in China "
                 "and Europe drove the growth. Analysts upgraded their price targets.",
            ticker="TSLA",
            source="Reuters",
            doc_type="news"
        ),
        Document.create(
            text="Tesla faces increased competition from Chinese EV makers. "
                 "BYD surpassed Tesla in quarterly sales for the first time. "
                 "Margin pressure concerns are rising among analysts.",
            ticker="TSLA",
            source="Bloomberg",
            doc_type="news"
        ),
        Document.create(
            text="Apple announced strong holiday quarter results with iPhone "
                 "sales exceeding expectations. Services revenue hit all-time "
                 "high. Company raised guidance for next quarter.",
            ticker="AAPL",
            source="WSJ",
            doc_type="news"
        ),
        Document.create(
            text="Apple faces regulatory challenges in EU over app store fees. "
                 "New rules could reduce services revenue by 5-10% in Europe.",
            ticker="AAPL",
            source="FT",
            doc_type="news"
        ),
    ]

    # Create retriever and add documents
    retriever = SimpleRetriever()
    retriever.add_documents(sample_docs)

    # Create signal generator
    generator = RAGTradingSignalGenerator(retriever)

    # Generate signals
    for ticker in ["TSLA", "AAPL"]:
        print(f"\n{'='*50}")
        print(f"Generating signal for {ticker}")
        print("-" * 50)

        signal = generator.generate_signal(ticker)

        print(f"Direction: {signal.direction.value}")
        print(f"Confidence: {signal.confidence:.0%}")
        print(f"Reasoning: {signal.reasoning}")
        print(f"Sources: {', '.join(signal.sources)}")

        if signal.metadata:
            print(f"Metadata: {signal.metadata}")
