"""
Chapter 67: LLM Sentiment Analysis for Trading
04_gpt_sentiment.py - GPT-based sentiment analysis with reasoning

This module demonstrates how to use GPT models (OpenAI or compatible)
for advanced sentiment analysis with detailed reasoning and explanation.

Requirements:
    pip install openai
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPTSentimentResult:
    """Result of GPT-based sentiment analysis."""
    text: str
    sentiment: str
    confidence: float
    reasoning: str
    key_factors: List[str]
    market_impact: str
    raw_response: Dict[str, Any]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        try:
            import openai
            self.openai = openai
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                self.client = None
                logger.warning("No API key provided. Using mock mode.")
        except ImportError:
            self.openai = None
            self.client = None
            logger.warning("openai package not installed. Using mock mode.")

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using OpenAI API."""
        if self.client is None:
            return self._mock_generate(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent analysis
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API error: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Mock response for demonstration."""
        # Simple rule-based mock
        text_lower = prompt.lower()

        if any(w in text_lower for w in ['surge', 'rise', 'beat', 'growth', 'rally']):
            sentiment = "POSITIVE"
            confidence = 0.85
            reasoning = "The text contains positive indicators suggesting upward momentum."
            key_factors = ["Strong performance language", "Positive price action"]
            market_impact = "Likely bullish for the asset"
        elif any(w in text_lower for w in ['fall', 'drop', 'miss', 'crash', 'decline']):
            sentiment = "NEGATIVE"
            confidence = 0.82
            reasoning = "The text contains negative indicators suggesting downward pressure."
            key_factors = ["Negative performance language", "Bearish indicators"]
            market_impact = "Likely bearish for the asset"
        else:
            sentiment = "NEUTRAL"
            confidence = 0.65
            reasoning = "The text appears balanced without strong directional bias."
            key_factors = ["Mixed signals", "Informational content"]
            market_impact = "Limited immediate impact expected"

        return json.dumps({
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_factors": key_factors,
            "market_impact": market_impact
        })


class GPTSentimentAnalyzer:
    """
    Advanced sentiment analyzer using GPT models.

    Features:
    - Detailed reasoning for sentiment decisions
    - Key factor extraction
    - Market impact assessment
    - Configurable prompts for different domains
    """

    SYSTEM_PROMPT = """You are a financial sentiment analyst specializing in market news and social media analysis.
Your task is to analyze the sentiment of financial text and provide actionable insights for trading.

Always respond in valid JSON format with the following structure:
{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence": <float between 0 and 1>,
    "reasoning": "<detailed explanation>",
    "key_factors": ["<factor1>", "<factor2>", ...],
    "market_impact": "<expected market reaction>"
}

Be objective and consider:
1. The overall tone and language used
2. Specific financial metrics or events mentioned
3. Context and potential market implications
4. Any uncertainty or mixed signals"""

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        model: str = "gpt-4"
    ):
        """
        Initialize the GPT sentiment analyzer.

        Args:
            client: LLM client to use (defaults to OpenAI)
            model: Model name for default OpenAI client
        """
        self.client = client or OpenAIClient(model=model)

    def analyze(self, text: str) -> GPTSentimentResult:
        """
        Analyze sentiment with detailed reasoning.

        Args:
            text: Financial text to analyze

        Returns:
            GPTSentimentResult with sentiment, reasoning, and factors
        """
        prompt = f"""Analyze the sentiment of the following financial text:

"{text}"

Provide your analysis in JSON format."""

        response = self.client.generate(prompt, self.SYSTEM_PROMPT)

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response: {response}")
            parsed = {
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": "Failed to parse LLM response",
                "key_factors": [],
                "market_impact": "Unknown"
            }

        return GPTSentimentResult(
            text=text,
            sentiment=parsed.get("sentiment", "NEUTRAL"),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning", ""),
            key_factors=parsed.get("key_factors", []),
            market_impact=parsed.get("market_impact", ""),
            raw_response=parsed
        )

    def analyze_with_context(
        self,
        text: str,
        symbol: str,
        recent_price_change: Optional[float] = None,
        market_condition: Optional[str] = None
    ) -> GPTSentimentResult:
        """
        Analyze sentiment with additional market context.

        Args:
            text: Financial text to analyze
            symbol: Asset symbol (e.g., 'BTC', 'AAPL')
            recent_price_change: Recent price change percentage
            market_condition: Current market condition ('bull', 'bear', 'sideways')

        Returns:
            GPTSentimentResult with context-aware analysis
        """
        context_parts = [f"Asset: {symbol}"]

        if recent_price_change is not None:
            direction = "up" if recent_price_change > 0 else "down"
            context_parts.append(f"Recent price change: {abs(recent_price_change):.1f}% {direction}")

        if market_condition:
            context_parts.append(f"Current market: {market_condition}")

        context = "\n".join(context_parts)

        prompt = f"""Analyze the sentiment of the following financial text with market context:

Context:
{context}

Text:
"{text}"

Consider how the news might affect the asset given the current market conditions.
Provide your analysis in JSON format."""

        response = self.client.generate(prompt, self.SYSTEM_PROMPT)

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = {
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": "Failed to parse",
                "key_factors": [],
                "market_impact": "Unknown"
            }

        return GPTSentimentResult(
            text=text,
            sentiment=parsed.get("sentiment", "NEUTRAL"),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning", ""),
            key_factors=parsed.get("key_factors", []),
            market_impact=parsed.get("market_impact", ""),
            raw_response=parsed
        )

    def compare_sentiments(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze and compare sentiments of multiple texts.

        Args:
            texts: List of texts to compare

        Returns:
            Comparison analysis with consensus and disagreements
        """
        results = [self.analyze(text) for text in texts]

        # Count sentiments
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for result in results:
            sentiment_counts[result.sentiment] += 1

        # Find consensus
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        consensus_strength = sentiment_counts[dominant_sentiment] / len(results)

        # Collect all factors
        all_factors = []
        for result in results:
            all_factors.extend(result.key_factors)

        return {
            "individual_results": results,
            "sentiment_counts": sentiment_counts,
            "consensus_sentiment": dominant_sentiment,
            "consensus_strength": consensus_strength,
            "all_key_factors": list(set(all_factors)),
            "total_texts": len(texts)
        }


def main():
    """Demonstrate GPT-based sentiment analysis."""

    print("=" * 60)
    print("LLM Sentiment Analysis - GPT Demo")
    print("=" * 60)

    # Initialize analyzer (will use mock mode without API key)
    analyzer = GPTSentimentAnalyzer()

    # Example financial texts
    texts = [
        "Bitcoin surges 15% after BlackRock's ETF application receives SEC approval. "
        "Institutional investors are finally getting the regulated exposure they've been waiting for.",

        "Tesla misses Q3 delivery targets by 12%, citing supply chain disruptions. "
        "Margins compress as competition intensifies in the EV market.",

        "Federal Reserve signals potential rate pause in upcoming meeting. "
        "Markets await further economic data before adjusting positions.",

        "Ethereum's Shanghai upgrade enables staking withdrawals, potentially "
        "creating short-term selling pressure but long-term confidence in the network.",
    ]

    print("\nDetailed Analysis Results:")
    print("-" * 40)

    for i, text in enumerate(texts, 1):
        print(f"\n=== Text {i} ===")
        print(f"Input: {text[:80]}...")

        result = analyzer.analyze(text)

        print(f"\nSentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Factors: {', '.join(result.key_factors)}")
        print(f"Market Impact: {result.market_impact}")

    # Context-aware analysis example
    print("\n" + "=" * 60)
    print("Context-Aware Analysis")
    print("=" * 60)

    context_result = analyzer.analyze_with_context(
        text="Bitcoin drops 5% overnight amid broader market selloff",
        symbol="BTC",
        recent_price_change=-5.0,
        market_condition="bear"
    )

    print(f"\nWith context analysis:")
    print(f"Sentiment: {context_result.sentiment}")
    print(f"Confidence: {context_result.confidence:.0%}")
    print(f"Reasoning: {context_result.reasoning}")

    # Comparison example
    print("\n" + "=" * 60)
    print("Multi-Text Comparison")
    print("=" * 60)

    comparison_texts = [
        "Apple stock rallies on strong iPhone sales",
        "Tech sector faces headwinds from rising rates",
        "Apple beats earnings estimates, raises dividend",
    ]

    comparison = analyzer.compare_sentiments(comparison_texts)

    print(f"\nAnalyzed {comparison['total_texts']} texts")
    print(f"Sentiment Distribution: {comparison['sentiment_counts']}")
    print(f"Consensus: {comparison['consensus_sentiment']} "
          f"(strength: {comparison['consensus_strength']:.0%})")
    print(f"Key Factors Identified: {', '.join(comparison['all_key_factors'][:5])}")


if __name__ == "__main__":
    main()
