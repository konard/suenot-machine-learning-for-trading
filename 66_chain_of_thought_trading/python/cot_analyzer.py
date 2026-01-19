"""
Chain-of-Thought Trading Analyzer

This module provides the core CoT analysis functionality for trading,
using LLMs to generate explainable trading decisions with full
reasoning chains.
"""

import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReasoningStep:
    """A single step in the chain of thought reasoning."""
    step_number: int
    title: str
    reasoning: str
    conclusion: str


@dataclass
class CoTAnalysis:
    """Complete chain-of-thought analysis result."""
    symbol: str
    timestamp: datetime
    steps: List[ReasoningStep]
    final_signal: str
    confidence: float
    reasoning_summary: str
    raw_response: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class ChainOfThoughtAnalyzer:
    """
    Chain-of-Thought trading analyzer using LLM.

    This class demonstrates how to use CoT prompting for
    explainable trading decisions.

    Attributes:
        model: The LLM model to use (e.g., "gpt-4")
        temperature: Sampling temperature for generation
        api_key: OpenAI API key

    Example:
        >>> analyzer = ChainOfThoughtAnalyzer()
        >>> market_data = {
        ...     "price": 43250,
        ...     "change_24h": 2.5,
        ...     "rsi": 55,
        ...     "macd": 150
        ... }
        >>> result = analyzer.analyze("BTCUSDT", market_data)
        >>> print(result.final_signal)
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the analyzer.

        Args:
            model: LLM model name
            temperature: Sampling temperature (lower = more deterministic)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        """Initialize OpenAI client lazily."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai required. Install with: pip install openai")

    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        news: Optional[List[str]] = None
    ) -> CoTAnalysis:
        """
        Perform chain-of-thought analysis on market data.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
            market_data: Dict with price, indicators, etc.
            news: Optional list of recent news headlines

        Returns:
            CoTAnalysis with full reasoning chain
        """
        self._ensure_client()

        prompt = self._build_prompt(symbol, market_data, news)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=2000
        )

        raw_response = response.choices[0].message.content
        return self._parse_response(symbol, raw_response)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the analyzer."""
        return """You are an expert quantitative trading analyst.
Your role is to analyze market data and provide trading recommendations
with clear, step-by-step reasoning.

Always structure your analysis as follows:
1. State each reasoning step clearly with STEP X prefix
2. Support conclusions with specific data points
3. Consider both bullish and bearish scenarios
4. Quantify confidence levels
5. Provide actionable recommendations

Your analysis should be thorough but concise. Each step should
logically lead to the next, building toward a final recommendation.

End your analysis with:
SIGNAL: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
CONFIDENCE: [0-100]%
ENTRY: $[price]
STOP_LOSS: $[price]
TAKE_PROFIT: $[price]"""

    def _build_prompt(
        self,
        symbol: str,
        market_data: Dict,
        news: Optional[List[str]]
    ) -> str:
        """Build the analysis prompt."""
        prompt = f"""
Analyze {symbol} and provide a trading recommendation.

=== MARKET DATA ===
Current Price: ${market_data.get('price', 'N/A')}
24h Change: {market_data.get('change_24h', 'N/A')}%
Volume: {market_data.get('volume', 'N/A')}
RSI(14): {market_data.get('rsi', 'N/A')}
MACD: {market_data.get('macd', 'N/A')}
MACD Signal: {market_data.get('macd_signal', 'N/A')}
50-day SMA: ${market_data.get('sma_50', 'N/A')}
200-day SMA: ${market_data.get('sma_200', 'N/A')}
ATR(14): {market_data.get('atr', 'N/A')}
"""

        if news:
            prompt += "\n=== RECENT NEWS ===\n"
            for i, headline in enumerate(news[:5], 1):
                prompt += f"{i}. {headline}\n"

        prompt += """
=== ANALYSIS REQUIRED ===
Think through each step carefully:

STEP 1 - PRICE ACTION ANALYSIS:
Analyze current price relative to key moving averages and recent action.

STEP 2 - MOMENTUM EVALUATION:
Assess RSI and MACD for momentum signals and potential divergences.

STEP 3 - SENTIMENT REVIEW:
Consider news sentiment and its potential market impact.

STEP 4 - RISK ASSESSMENT:
Identify key risks, support/resistance levels, and calculate risk/reward.

STEP 5 - FINAL RECOMMENDATION:
Provide your signal with specific levels.

Begin your analysis:
"""
        return prompt

    def _parse_response(self, symbol: str, raw_response: str) -> CoTAnalysis:
        """Parse LLM response into structured format."""
        steps = []

        # Extract steps using regex
        step_pattern = r'STEP\s*(\d+)\s*[-:]\s*([^\n]+)\n([\s\S]*?)(?=STEP\s*\d+|SIGNAL:|$)'
        matches = re.findall(step_pattern, raw_response, re.IGNORECASE)

        for match in matches:
            step_num = int(match[0])
            title = match[1].strip()
            content = match[2].strip()
            conclusion = content.split('\n')[-1].strip() if content else ""

            steps.append(ReasoningStep(
                step_number=step_num,
                title=title,
                reasoning=content,
                conclusion=conclusion
            ))

        # Extract final signal
        signal_match = re.search(r'SIGNAL:\s*(\w+)', raw_response, re.IGNORECASE)
        signal = signal_match.group(1).upper() if signal_match else "HOLD"

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', raw_response, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

        # Extract price levels
        entry_match = re.search(r'ENTRY:\s*\$?([\d,]+(?:\.\d+)?)', raw_response, re.IGNORECASE)
        entry_price = float(entry_match.group(1).replace(',', '')) if entry_match else None

        stop_match = re.search(r'STOP_LOSS:\s*\$?([\d,]+(?:\.\d+)?)', raw_response, re.IGNORECASE)
        stop_loss = float(stop_match.group(1).replace(',', '')) if stop_match else None

        target_match = re.search(r'TAKE_PROFIT:\s*\$?([\d,]+(?:\.\d+)?)', raw_response, re.IGNORECASE)
        take_profit = float(target_match.group(1).replace(',', '')) if target_match else None

        summary = f"Analysis of {symbol}: {signal} with {confidence:.0%} confidence. "
        if steps:
            summary += f"Based on {len(steps)} reasoning steps."

        return CoTAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            steps=steps,
            final_signal=signal,
            confidence=confidence,
            reasoning_summary=summary,
            raw_response=raw_response,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )


class MockChainOfThoughtAnalyzer(ChainOfThoughtAnalyzer):
    """
    Mock analyzer for testing without API calls.

    This class generates realistic-looking analysis without
    making actual LLM API calls. Useful for testing and
    demonstrations.
    """

    def __init__(self):
        """Initialize mock analyzer (no API key needed)."""
        super().__init__()

    def _ensure_client(self):
        """No client needed for mock."""
        pass

    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        news: Optional[List[str]] = None
    ) -> CoTAnalysis:
        """
        Generate mock analysis for demonstration.

        Args:
            symbol: Trading symbol
            market_data: Market data dict
            news: Optional news headlines

        Returns:
            CoTAnalysis with simulated reasoning
        """
        price = market_data.get('price', 100)
        rsi = market_data.get('rsi', 50)
        change = market_data.get('change_24h', 0)
        sma_50 = market_data.get('sma_50', price)
        sma_200 = market_data.get('sma_200', price)
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)

        # Determine signal based on indicators
        score = 0
        if rsi < 30:
            score += 2
        elif rsi < 45:
            score += 1
        elif rsi > 70:
            score -= 2
        elif rsi > 55:
            score -= 1

        if price > sma_50:
            score += 1
        else:
            score -= 1

        if price > sma_200:
            score += 1
        else:
            score -= 1

        if macd > macd_signal:
            score += 1
        else:
            score -= 1

        if change > 0:
            score += 0.5
        else:
            score -= 0.5

        # Determine signal and confidence
        if score >= 3:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif score >= 1:
            signal = "BUY"
            confidence = 0.70
        elif score <= -3:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif score <= -1:
            signal = "SELL"
            confidence = 0.70
        else:
            signal = "HOLD"
            confidence = 0.55

        # Build reasoning steps
        trend_assessment = "bullish" if price > sma_50 else "bearish"
        momentum_assessment = "positive" if macd > macd_signal else "negative"
        rsi_assessment = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"

        steps = [
            ReasoningStep(
                step_number=1,
                title="PRICE ACTION ANALYSIS",
                reasoning=f"Current price ${price:,.2f} shows {change:+.1f}% change over 24h. "
                         f"Price is {'above' if price > sma_50 else 'below'} the 50-day SMA (${sma_50:,.2f}) "
                         f"and {'above' if price > sma_200 else 'below'} the 200-day SMA (${sma_200:,.2f}). "
                         f"This indicates a {trend_assessment} trend structure.",
                conclusion=f"Price action is {trend_assessment}."
            ),
            ReasoningStep(
                step_number=2,
                title="MOMENTUM EVALUATION",
                reasoning=f"RSI at {rsi:.1f} indicates {rsi_assessment} conditions. "
                         f"MACD ({macd:.2f}) is {'above' if macd > macd_signal else 'below'} "
                         f"signal line ({macd_signal:.2f}), suggesting {momentum_assessment} momentum. "
                         f"No significant divergences detected.",
                conclusion=f"Momentum is {momentum_assessment}."
            ),
            ReasoningStep(
                step_number=3,
                title="SENTIMENT REVIEW",
                reasoning="News sentiment analysis indicates mixed signals. Recent headlines "
                         "show both positive and negative factors. Market sentiment appears "
                         f"{'slightly positive' if change > 0 else 'slightly negative'} based on recent price action.",
                conclusion="Sentiment is neutral with slight bias."
            ),
            ReasoningStep(
                step_number=4,
                title="RISK ASSESSMENT",
                reasoning=f"Key support level at ${price * 0.95:,.2f}, resistance at ${price * 1.05:,.2f}. "
                         f"ATR suggests normal volatility. Stop-loss at ${price * 0.97:,.2f} "
                         f"and take-profit at ${price * 1.06:,.2f} yields risk/reward of 1:2.",
                conclusion="Risk/reward is favorable for the suggested direction."
            )
        ]

        entry_price = price
        stop_loss = price * 0.97 if signal in ["BUY", "STRONG_BUY"] else price * 1.03
        take_profit = price * 1.06 if signal in ["BUY", "STRONG_BUY"] else price * 0.94

        raw_response = self._build_raw_response(steps, signal, confidence,
                                                 entry_price, stop_loss, take_profit)

        return CoTAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            steps=steps,
            final_signal=signal,
            confidence=confidence,
            reasoning_summary=f"{symbol}: {signal} ({confidence:.0%} confidence)",
            raw_response=raw_response,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _build_raw_response(
        self,
        steps: List[ReasoningStep],
        signal: str,
        confidence: float,
        entry: float,
        stop: float,
        target: float
    ) -> str:
        """Build a raw response string from components."""
        response = ""
        for step in steps:
            response += f"\nSTEP {step.step_number} - {step.title}:\n"
            response += f"{step.reasoning}\n"
            response += f"Conclusion: {step.conclusion}\n"

        response += f"\nSIGNAL: {signal}\n"
        response += f"CONFIDENCE: {int(confidence * 100)}%\n"
        response += f"ENTRY: ${entry:,.2f}\n"
        response += f"STOP_LOSS: ${stop:,.2f}\n"
        response += f"TAKE_PROFIT: ${target:,.2f}\n"

        return response


def analyze_with_self_consistency(
    analyzer: ChainOfThoughtAnalyzer,
    symbol: str,
    market_data: Dict,
    news: Optional[List[str]] = None,
    n_samples: int = 5
) -> Dict:
    """
    Use self-consistency for more reliable signals.

    Runs multiple analyses and takes majority vote.

    Args:
        analyzer: The CoT analyzer to use
        symbol: Trading symbol
        market_data: Market data dict
        news: Optional news headlines
        n_samples: Number of samples to generate

    Returns:
        Dict with consensus signal and agreement level
    """
    from collections import Counter

    signals = []
    all_analyses = []

    for _ in range(n_samples):
        result = analyzer.analyze(symbol, market_data, news)
        signals.append(result.final_signal)
        all_analyses.append(result)

    # Majority vote
    counter = Counter(signals)
    most_common = counter.most_common(1)[0]

    return {
        'signal': most_common[0],
        'agreement': most_common[1] / n_samples,
        'all_signals': signals,
        'all_analyses': all_analyses,
        'signal_distribution': dict(counter)
    }


if __name__ == "__main__":
    print("Chain-of-Thought Trading Analysis Demo")
    print("=" * 50)

    # Use mock analyzer for demonstration
    analyzer = MockChainOfThoughtAnalyzer()

    # Sample market data
    market_data = {
        "price": 43250,
        "change_24h": -2.5,
        "volume": 25000000000,
        "rsi": 35,
        "macd": -150,
        "macd_signal": -100,
        "sma_50": 44000,
        "sma_200": 42000,
        "atr": 800
    }

    news = [
        "Bitcoin ETF sees record inflows for third consecutive day",
        "Federal Reserve signals potential rate pause",
        "Major exchange reports technical issues",
    ]

    # Perform analysis
    result = analyzer.analyze("BTCUSDT", market_data, news)

    print(f"\nAnalysis for: {result.symbol}")
    print(f"Timestamp: {result.timestamp}")
    print(f"\n{'=' * 50}")
    print("REASONING CHAIN:")
    print("=" * 50)

    for step in result.steps:
        print(f"\nSTEP {step.step_number}: {step.title}")
        print("-" * 40)
        print(step.reasoning)
        print(f"\nConclusion: {step.conclusion}")

    print(f"\n{'=' * 50}")
    print("FINAL RECOMMENDATION:")
    print("=" * 50)
    print(f"Signal: {result.final_signal}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Entry: ${result.entry_price:,.2f}")
    print(f"Stop Loss: ${result.stop_loss:,.2f}")
    print(f"Take Profit: ${result.take_profit:,.2f}")
    print(f"\nSummary: {result.reasoning_summary}")
