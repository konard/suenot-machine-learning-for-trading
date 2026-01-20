"""
Financial Sentiment Analysis using Prompt Engineering

This module provides sentiment analysis capabilities using
various prompt engineering techniques.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio

from .prompts.sentiment import SENTIMENT_PROMPTS


class Sentiment(Enum):
    """Sentiment classification."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: Sentiment
    confidence: float
    reasoning: str
    entities: Optional[Dict[str, Sentiment]] = None
    raw_response: Optional[str] = None


class FinancialSentimentAnalyzer:
    """
    Prompt engineering-based financial sentiment analyzer.

    Uses carefully crafted prompts to extract trading-relevant
    sentiment from financial text.

    Example:
        >>> analyzer = FinancialSentimentAnalyzer(llm_client)
        >>> result = await analyzer.analyze("Apple beats earnings expectations")
        >>> print(result.sentiment)
        Sentiment.POSITIVE
    """

    def __init__(
        self,
        llm_client,
        prompt_type: str = "few_shot",
        temperature: float = 0.3
    ):
        """
        Initialize sentiment analyzer.

        Args:
            llm_client: LLM client instance
            prompt_type: Type of prompt to use ("zero_shot", "few_shot", "chain_of_thought")
            temperature: LLM temperature (lower = more consistent)
        """
        self.llm = llm_client
        self.prompt_type = prompt_type
        self.temperature = temperature

        if prompt_type not in SENTIMENT_PROMPTS:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        self.prompt_template = SENTIMENT_PROMPTS[prompt_type]

    async def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news, report, or social media post

        Returns:
            SentimentResult with sentiment classification and reasoning
        """
        prompt = self.prompt_template.format(text=text)

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=500
        )

        return self._parse_response(response)

    async def analyze_with_entities(
        self,
        text: str,
        entities: List[str]
    ) -> SentimentResult:
        """
        Analyze sentiment toward specific entities.

        Args:
            text: Financial text
            entities: List of entities to analyze sentiment for

        Returns:
            SentimentResult with entity-specific sentiments
        """
        prompt = SENTIMENT_PROMPTS["aspect_based"].format(
            text=text,
            entities=", ".join(entities)
        )

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=800
        )

        return self._parse_aspect_response(response)

    async def analyze_batch(
        self,
        texts: List[str],
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze multiple texts and optionally aggregate sentiment.

        Args:
            texts: List of financial texts
            aggregate: Whether to aggregate into overall sentiment

        Returns:
            Dictionary with individual results and optional aggregate
        """
        # Process texts concurrently
        tasks = [self.analyze(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if isinstance(r, SentimentResult)]

        if not aggregate:
            return {"individual": valid_results}

        # Aggregate sentiment with confidence weighting
        sentiment_scores = {
            Sentiment.POSITIVE: 1,
            Sentiment.NEUTRAL: 0,
            Sentiment.NEGATIVE: -1
        }

        if not valid_results:
            return {
                "individual": valid_results,
                "aggregate": {
                    "sentiment": Sentiment.NEUTRAL,
                    "score": 0,
                    "confidence": 0,
                    "sample_count": 0
                }
            }

        weighted_sum = sum(
            sentiment_scores[r.sentiment] * r.confidence
            for r in valid_results
        )
        total_confidence = sum(r.confidence for r in valid_results)

        if total_confidence > 0:
            avg_score = weighted_sum / total_confidence
            if avg_score > 0.3:
                aggregate_sentiment = Sentiment.POSITIVE
            elif avg_score < -0.3:
                aggregate_sentiment = Sentiment.NEGATIVE
            else:
                aggregate_sentiment = Sentiment.NEUTRAL
        else:
            aggregate_sentiment = Sentiment.NEUTRAL
            avg_score = 0

        return {
            "individual": valid_results,
            "aggregate": {
                "sentiment": aggregate_sentiment,
                "score": avg_score,
                "confidence": total_confidence / len(valid_results),
                "sample_count": len(valid_results)
            }
        }

    async def self_consistent_analyze(
        self,
        text: str,
        num_samples: int = 5,
        temperature: float = 0.7
    ) -> SentimentResult:
        """
        Use self-consistency technique for more reliable analysis.

        Generates multiple samples and uses majority voting.

        Args:
            text: Financial text to analyze
            num_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            SentimentResult with aggregated confidence
        """
        prompt = self.prompt_template.format(text=text)

        # Generate multiple samples
        tasks = [
            self.llm.complete(prompt, temperature=temperature, max_tokens=500)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse all responses
        results = []
        for response in responses:
            if isinstance(response, str):
                try:
                    parsed = self._parse_response(response)
                    results.append(parsed)
                except Exception:
                    continue

        if not results:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0,
                reasoning="Failed to generate consistent responses"
            )

        # Majority vote
        sentiment_counts = {}
        for r in results:
            sentiment_counts[r.sentiment] = sentiment_counts.get(r.sentiment, 0) + 1

        majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        agreement_ratio = sentiment_counts[majority_sentiment] / len(results)

        # Average confidence adjusted by agreement
        avg_confidence = sum(r.confidence for r in results) / len(results)
        final_confidence = avg_confidence * agreement_ratio

        # Combine reasoning from matching results
        matching_reasoning = [
            r.reasoning for r in results
            if r.sentiment == majority_sentiment
        ]

        return SentimentResult(
            sentiment=majority_sentiment,
            confidence=final_confidence,
            reasoning=f"[Agreement: {agreement_ratio:.0%}] " + matching_reasoning[0] if matching_reasoning else ""
        )

    def _parse_response(self, response: str) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            sentiment_str = data.get("sentiment", "NEUTRAL").upper()
            if sentiment_str == "BULLISH":
                sentiment_str = "POSITIVE"
            elif sentiment_str == "BEARISH":
                sentiment_str = "NEGATIVE"

            return SentimentResult(
                sentiment=Sentiment(sentiment_str),
                confidence=float(data.get("confidence", 50)),
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback parsing
            response_lower = response.lower()
            if "positive" in response_lower or "bullish" in response_lower:
                sentiment = Sentiment.POSITIVE
            elif "negative" in response_lower or "bearish" in response_lower:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL

            return SentimentResult(
                sentiment=sentiment,
                confidence=30,  # Low confidence for fallback
                reasoning=f"Fallback parsing: {str(e)}",
                raw_response=response
            )

    def _parse_aspect_response(self, response: str) -> SentimentResult:
        """Parse aspect-based sentiment response."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            sentiment_str = data.get("overall_sentiment", "NEUTRAL").upper()
            entities = {}

            if "entity_sentiments" in data:
                for entity, info in data["entity_sentiments"].items():
                    entity_sentiment_str = info.get("sentiment", "NEUTRAL").upper()
                    if entity_sentiment_str == "BULLISH":
                        entity_sentiment_str = "POSITIVE"
                    elif entity_sentiment_str == "BEARISH":
                        entity_sentiment_str = "NEGATIVE"
                    entities[entity] = Sentiment(entity_sentiment_str)

            return SentimentResult(
                sentiment=Sentiment(sentiment_str),
                confidence=float(data.get("overall_confidence", 50)),
                reasoning=str(data.get("entity_sentiments", {})),
                entities=entities,
                raw_response=response
            )

        except Exception as e:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0,
                reasoning=f"Parse error: {e}",
                raw_response=response
            )


# Example usage
async def main():
    """Demo of sentiment analysis."""
    from .llm_client import MockLLMClient

    # Create mock client with predefined responses
    mock_responses = {
        "apple": json.dumps({
            "sentiment": "POSITIVE",
            "confidence": 85,
            "reasoning": "Earnings beat indicates strong performance."
        }),
        "fed": json.dumps({
            "sentiment": "NEGATIVE",
            "confidence": 72,
            "reasoning": "Higher rates typically pressure equity valuations."
        })
    }

    client = MockLLMClient(responses=mock_responses)
    analyzer = FinancialSentimentAnalyzer(client)

    # Analyze news
    news_items = [
        "Apple reported record quarterly earnings, beating analyst expectations.",
        "The Federal Reserve announced plans to maintain higher interest rates."
    ]

    print("Financial Sentiment Analysis Demo")
    print("=" * 50)

    for news in news_items:
        result = await analyzer.analyze(news)
        print(f"\nNews: {news[:60]}...")
        print(f"Sentiment: {result.sentiment.value}")
        print(f"Confidence: {result.confidence}%")
        print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
