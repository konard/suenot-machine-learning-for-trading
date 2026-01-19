# Chapter 66: Chain-of-Thought Trading — Explainable LLM Trading Decisions

This chapter explores **Chain-of-Thought (CoT) prompting** for trading applications. We examine how step-by-step reasoning from Large Language Models can generate **explainable, transparent trading signals** with full audit trails of the decision-making process.

<p align="center">
<img src="https://i.imgur.com/JRK5m6x.png" width="70%">
</p>

## Contents

1. [Introduction to Chain-of-Thought Trading](#introduction-to-chain-of-thought-trading)
    * [Why Explainability Matters in Trading](#why-explainability-matters-in-trading)
    * [CoT vs Traditional ML Signals](#cot-vs-traditional-ml-signals)
    * [Key Benefits](#key-benefits)
2. [Chain-of-Thought Fundamentals](#chain-of-thought-fundamentals)
    * [The CoT Prompting Technique](#the-cot-prompting-technique)
    * [Zero-Shot vs Few-Shot CoT](#zero-shot-vs-few-shot-cot)
    * [Self-Consistency Decoding](#self-consistency-decoding)
3. [Trading Applications](#trading-applications)
    * [Market Analysis with CoT](#market-analysis-with-cot)
    * [Risk Assessment Chains](#risk-assessment-chains)
    * [Multi-Factor Signal Generation](#multi-factor-signal-generation)
    * [Trade Execution Reasoning](#trade-execution-reasoning)
4. [Practical Examples](#practical-examples)
    * [01: Basic CoT Trading Analysis](#01-basic-cot-trading-analysis)
    * [02: Multi-Step Signal Generation](#02-multi-step-signal-generation)
    * [03: Risk-Adjusted Position Sizing](#03-risk-adjusted-position-sizing)
    * [04: Full Backtesting Pipeline](#04-full-backtesting-pipeline)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Chain-of-Thought Trading

Chain-of-Thought prompting revolutionizes how we can use LLMs for trading by making the reasoning process **explicit and auditable**. Instead of getting a simple "buy" or "sell" signal, we get a complete breakdown of the factors considered and how they led to the final decision.

### Why Explainability Matters in Trading

```
TRADITIONAL ML TRADING SIGNALS:
┌──────────────────────────────────────────────────────────────────┐
│  Input: Market Data + News + Indicators                          │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │   BLACK BOX  │                              │
│                    │    MODEL     │                              │
│                    └──────────────┘                              │
│                           │                                      │
│                           ▼                                      │
│  Output: "BUY" (confidence: 0.73)                                │
│                                                                  │
│  Problem: WHY should we buy? We don't know!                      │
└──────────────────────────────────────────────────────────────────┘

CHAIN-OF-THOUGHT TRADING:
┌──────────────────────────────────────────────────────────────────┐
│  Input: Market Data + News + Indicators                          │
│                           │                                      │
│                           ▼                                      │
│  Step 1: "The RSI is at 32, indicating oversold conditions..."   │
│                           │                                      │
│                           ▼                                      │
│  Step 2: "Recent earnings beat expectations by 15%..."           │
│                           │                                      │
│                           ▼                                      │
│  Step 3: "Sector momentum is positive with peers up 3%..."       │
│                           │                                      │
│                           ▼                                      │
│  Step 4: "Risk/reward ratio favors long position..."             │
│                           │                                      │
│                           ▼                                      │
│  Output: "BUY" with full reasoning chain                         │
│                                                                  │
│  Benefit: Complete audit trail for every decision!               │
└──────────────────────────────────────────────────────────────────┘
```

### CoT vs Traditional ML Signals

| Aspect | Traditional ML | Chain-of-Thought |
|--------|---------------|------------------|
| Explainability | Low (black box) | High (full reasoning) |
| Regulatory Compliance | Challenging | Straightforward |
| Debugging | Difficult | Easy (inspect each step) |
| Human Oversight | Limited | Natural integration |
| Adaptability | Requires retraining | Prompt modification |
| Computational Cost | Lower (single inference) | Higher (longer generation) |
| Consistency | High | Requires techniques (self-consistency) |

### Key Benefits

1. **Regulatory Compliance**
   - Full audit trail for every trade decision
   - Explainable rationale for regulators
   - Documentation of risk considerations

2. **Risk Management**
   - Explicit reasoning about downside scenarios
   - Clear articulation of position sizing logic
   - Transparent stop-loss and take-profit reasoning

3. **Human-AI Collaboration**
   - Traders can verify LLM reasoning
   - Easy to identify flawed logic
   - Natural integration with human decision-making

4. **Continuous Improvement**
   - Analyze reasoning chains for patterns
   - Identify systematic biases
   - Refine prompts based on outcomes

## Chain-of-Thought Fundamentals

### The CoT Prompting Technique

Chain-of-Thought prompting, introduced by Wei et al. (2022), enables LLMs to decompose complex problems into intermediate reasoning steps:

```python
# Standard prompting
prompt_standard = """
Analyze AAPL stock and provide a trading signal.
"""
# Output: "Buy" (no explanation)

# Chain-of-Thought prompting
prompt_cot = """
Analyze AAPL stock and provide a trading signal.
Think step by step:

1. First, analyze the current price action and technical indicators
2. Then, consider recent fundamental developments
3. Next, evaluate market sentiment and sector trends
4. Assess the risk/reward ratio
5. Finally, provide your trading recommendation with reasoning
"""
# Output: Detailed step-by-step analysis with final recommendation
```

### Zero-Shot vs Few-Shot CoT

**Zero-Shot CoT**: Simply add "Let's think step by step" to the prompt

```python
zero_shot_cot = """
Given the following market data for BTCUSDT:
- Current price: $43,250
- 24h change: +2.3%
- RSI(14): 58
- Volume: 1.2x average

Should we enter a long position?

Let's think step by step.
"""
```

**Few-Shot CoT**: Provide examples of reasoning chains

```python
few_shot_cot = """
Example 1:
Market: ETHUSDT at $2,150, RSI: 28, Volume: 2x average
Analysis:
Step 1: RSI at 28 indicates oversold territory (below 30)
Step 2: High volume suggests strong interest at these levels
Step 3: This combination often precedes a bounce
Decision: LONG with tight stop at $2,050

Example 2:
Market: BTCUSDT at $45,000, RSI: 72, Volume: 0.8x average
Analysis:
Step 1: RSI at 72 indicates overbought conditions
Step 2: Below-average volume suggests weak momentum
Step 3: Risk of pullback is elevated
Decision: REDUCE position or stay FLAT

Now analyze:
Market: SOLUSDT at $98, RSI: 45, Volume: 1.5x average
"""
```

### Self-Consistency Decoding

For more reliable trading signals, we can use self-consistency:

```
SELF-CONSISTENCY FOR TRADING DECISIONS
═══════════════════════════════════════════════════════════════════

Input: Market analysis request

                    ┌─────────────────────┐
                    │   Same prompt       │
                    │   multiple times    │
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Path 1   │   │ Path 2   │   │ Path 3   │
        │ Reasoning│   │ Reasoning│   │ Reasoning│
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              ▼               ▼               ▼
           BUY             BUY            HOLD
           0.8             0.6            0.4

                    ┌─────────────────────┐
                    │  Majority Vote:     │
                    │  BUY (2/3 paths)    │
                    │  Avg confidence: 0.6│
                    └─────────────────────┘

Benefit: More robust decisions by sampling multiple reasoning paths
```

## Trading Applications

### Market Analysis with CoT

```python
MARKET_ANALYSIS_PROMPT = """
You are an expert quantitative analyst. Analyze the following market data
and provide a comprehensive assessment.

=== MARKET DATA ===
Asset: {symbol}
Current Price: ${price}
24h Change: {change_24h}%
7d Change: {change_7d}%

Technical Indicators:
- RSI(14): {rsi}
- MACD: {macd} (Signal: {macd_signal})
- Bollinger Bands: Lower={bb_lower}, Middle={bb_middle}, Upper={bb_upper}
- ATR(14): {atr}

Volume Analysis:
- Current Volume: {volume}
- 20-day Avg Volume: {avg_volume}
- Volume Ratio: {volume_ratio}x

=== ANALYSIS FRAMEWORK ===
Think through each step carefully:

STEP 1 - TREND ANALYSIS:
Analyze the current trend direction and strength based on price action
and moving averages.

STEP 2 - MOMENTUM ASSESSMENT:
Evaluate momentum using RSI and MACD. Identify any divergences.

STEP 3 - VOLATILITY CONTEXT:
Consider current volatility (ATR, Bollinger Bands) and its implications
for position sizing and stop placement.

STEP 4 - VOLUME CONFIRMATION:
Assess whether volume supports the current price action.

STEP 5 - RISK/REWARD CALCULATION:
Calculate potential entry, stop-loss, and take-profit levels.
Determine the risk/reward ratio.

STEP 6 - FINAL RECOMMENDATION:
Provide a clear trading signal (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
with confidence level and specific action items.

Begin your analysis:
"""
```

### Risk Assessment Chains

```python
RISK_ASSESSMENT_PROMPT = """
You are a risk manager at a quantitative trading firm. Evaluate the
following proposed trade using a structured risk assessment framework.

=== PROPOSED TRADE ===
Asset: {symbol}
Direction: {direction}
Entry Price: ${entry}
Position Size: ${size}
Account Value: ${account_value}
Current Positions: {current_positions}

=== RISK ASSESSMENT CHAIN ===

STEP 1 - POSITION SIZE RISK:
What percentage of the portfolio does this trade represent?
Is this within acceptable concentration limits?

STEP 2 - CORRELATION RISK:
How does this position correlate with existing holdings?
Does it increase or decrease portfolio diversification?

STEP 3 - LIQUIDITY RISK:
Can this position be exited quickly if needed?
What is the expected slippage for emergency exit?

STEP 4 - TAIL RISK:
What is the worst-case scenario for this trade?
How would a 3-sigma move affect the portfolio?

STEP 5 - OPPORTUNITY COST:
What capital is being allocated to this trade?
Are there better risk-adjusted opportunities?

STEP 6 - RISK DECISION:
APPROVE, APPROVE WITH MODIFICATIONS, or REJECT
If modifications needed, specify:
- Adjusted position size
- Required hedges
- Stop-loss requirements

Provide your risk assessment:
"""
```

### Multi-Factor Signal Generation

```python
MULTI_FACTOR_PROMPT = """
You are a multi-factor portfolio analyst. Generate a trading signal
by analyzing multiple factors and weighting their importance.

=== FACTOR DATA ===
Asset: {symbol}

FACTOR 1 - VALUE:
- P/E Ratio: {pe_ratio} (Sector avg: {sector_pe})
- P/B Ratio: {pb_ratio}
- Free Cash Flow Yield: {fcf_yield}%

FACTOR 2 - MOMENTUM:
- 1-month return: {return_1m}%
- 3-month return: {return_3m}%
- 12-month return: {return_12m}%
- Relative strength vs benchmark: {relative_strength}

FACTOR 3 - QUALITY:
- ROE: {roe}%
- Debt/Equity: {debt_equity}
- Earnings stability (5yr): {earnings_stability}

FACTOR 4 - SENTIMENT:
- Analyst consensus: {analyst_rating}
- News sentiment (7d): {news_sentiment}
- Social sentiment: {social_sentiment}

FACTOR 5 - TECHNICAL:
- Trend: {trend}
- Support levels: {support}
- Resistance levels: {resistance}

=== MULTI-FACTOR ANALYSIS ===

STEP 1 - VALUE ASSESSMENT:
Score the value factor from -1 (expensive) to +1 (cheap)
Reasoning: [explain]

STEP 2 - MOMENTUM ASSESSMENT:
Score momentum from -1 (weak) to +1 (strong)
Reasoning: [explain]

STEP 3 - QUALITY ASSESSMENT:
Score quality from -1 (poor) to +1 (excellent)
Reasoning: [explain]

STEP 4 - SENTIMENT ASSESSMENT:
Score sentiment from -1 (bearish) to +1 (bullish)
Reasoning: [explain]

STEP 5 - TECHNICAL ASSESSMENT:
Score technical setup from -1 (bearish) to +1 (bullish)
Reasoning: [explain]

STEP 6 - COMPOSITE SIGNAL:
Weight each factor (should sum to 1.0):
- Value weight: [0.0-1.0]
- Momentum weight: [0.0-1.0]
- Quality weight: [0.0-1.0]
- Sentiment weight: [0.0-1.0]
- Technical weight: [0.0-1.0]

Calculate weighted score and provide final signal:
SIGNAL: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
CONFIDENCE: [0-100]%

Begin your multi-factor analysis:
"""
```

### Trade Execution Reasoning

```python
EXECUTION_PROMPT = """
You are a trade execution specialist. Plan the optimal execution
strategy for the following trade.

=== TRADE DETAILS ===
Asset: {symbol}
Direction: {direction}
Target Size: {target_size} units
Urgency: {urgency} (LOW/MEDIUM/HIGH)
Market Conditions:
- Bid: ${bid}
- Ask: ${ask}
- Spread: {spread_bps} bps
- Daily Volume: {daily_volume}
- Current Volatility: {volatility}%

=== EXECUTION PLANNING ===

STEP 1 - MARKET IMPACT ANALYSIS:
Estimate the market impact of executing full size immediately.
Calculate participation rate relative to average volume.

STEP 2 - EXECUTION STRATEGY SELECTION:
Choose between:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Implementation Shortfall
- Aggressive/Passive
Justify your choice.

STEP 3 - ORDER SLICING:
If slicing the order:
- Number of slices: [N]
- Slice size: [X] units each
- Time between slices: [T] minutes
- Price limits for each slice

STEP 4 - LIMIT PRICE DETERMINATION:
Set appropriate limit prices considering:
- Spread capture opportunity
- Urgency requirements
- Market direction

STEP 5 - CONTINGENCY PLANNING:
What if:
- Price moves against us during execution?
- Liquidity dries up?
- News event occurs mid-execution?

STEP 6 - EXECUTION PLAN SUMMARY:
Provide complete execution instructions ready for implementation.

Begin your execution analysis:
"""
```

## Practical Examples

### 01: Basic CoT Trading Analysis

```python
# python/01_cot_analysis.py

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import re


@dataclass
class ReasoningStep:
    """A single step in the chain of thought."""
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


class ChainOfThoughtAnalyzer:
    """
    Chain-of-Thought trading analyzer using LLM.

    This class demonstrates how to use CoT prompting for
    explainable trading decisions.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
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
            symbol: Trading symbol
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

        # Parse the response into structured format
        return self._parse_response(symbol, raw_response)

    def _get_system_prompt(self) -> str:
        return """You are an expert quantitative trading analyst.
Your role is to analyze market data and provide trading recommendations
with clear, step-by-step reasoning.

Always structure your analysis as follows:
1. State each reasoning step clearly
2. Support conclusions with specific data points
3. Consider both bullish and bearish scenarios
4. Quantify confidence levels
5. Provide actionable recommendations

Your analysis should be thorough but concise. Each step should
logically lead to the next, building toward a final recommendation."""

    def _build_prompt(
        self,
        symbol: str,
        market_data: Dict,
        news: Optional[List[str]]
    ) -> str:
        prompt = f"""
Analyze {symbol} and provide a trading recommendation.

=== MARKET DATA ===
Current Price: ${market_data.get('price', 'N/A')}
24h Change: {market_data.get('change_24h', 'N/A')}%
Volume: {market_data.get('volume', 'N/A')}
RSI(14): {market_data.get('rsi', 'N/A')}
MACD: {market_data.get('macd', 'N/A')}
50-day SMA: ${market_data.get('sma_50', 'N/A')}
200-day SMA: ${market_data.get('sma_200', 'N/A')}
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
Assess RSI and MACD for momentum signals.

STEP 3 - SENTIMENT REVIEW:
Consider news sentiment and its potential impact.

STEP 4 - RISK ASSESSMENT:
Identify key risks and potential downside.

STEP 5 - FINAL RECOMMENDATION:
Provide your signal (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL) with:
- Confidence level (0-100%)
- Entry price suggestion
- Stop-loss level
- Take-profit target

Format your final recommendation as:
SIGNAL: [signal]
CONFIDENCE: [X]%
ENTRY: $[price]
STOP_LOSS: $[price]
TAKE_PROFIT: $[price]

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

            # Extract conclusion (last sentence or paragraph)
            conclusion = content.split('\n')[-1].strip()

            steps.append(ReasoningStep(
                step_number=step_num,
                title=title,
                reasoning=content,
                conclusion=conclusion
            ))

        # Extract final signal
        signal_match = re.search(r'SIGNAL:\s*(\w+)', raw_response, re.IGNORECASE)
        signal = signal_match.group(1) if signal_match else "HOLD"

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', raw_response, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

        # Create summary
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
            raw_response=raw_response
        )


# Mock analyzer for demonstration without API
class MockChainOfThoughtAnalyzer(ChainOfThoughtAnalyzer):
    """Mock analyzer for testing without API calls."""

    def __init__(self):
        super().__init__()

    def _ensure_client(self):
        pass  # No client needed for mock

    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        news: Optional[List[str]] = None
    ) -> CoTAnalysis:
        """Generate mock analysis for demonstration."""

        price = market_data.get('price', 100)
        rsi = market_data.get('rsi', 50)
        change = market_data.get('change_24h', 0)

        # Determine signal based on simple rules
        if rsi < 30 and change < -5:
            signal = "BUY"
            confidence = 0.75
        elif rsi > 70 and change > 5:
            signal = "SELL"
            confidence = 0.70
        else:
            signal = "HOLD"
            confidence = 0.55

        steps = [
            ReasoningStep(
                step_number=1,
                title="PRICE ACTION ANALYSIS",
                reasoning=f"Current price ${price} shows {change:+.1f}% change over 24h. "
                         f"Price is {'above' if price > market_data.get('sma_50', price) else 'below'} "
                         f"the 50-day SMA, indicating {'bullish' if price > market_data.get('sma_50', price) else 'bearish'} trend.",
                conclusion=f"Price action is {'favorable' if change > 0 else 'concerning'}."
            ),
            ReasoningStep(
                step_number=2,
                title="MOMENTUM EVALUATION",
                reasoning=f"RSI at {rsi} indicates {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} "
                         f"conditions. MACD shows {market_data.get('macd', 'neutral')} momentum.",
                conclusion=f"Momentum is {'bullish' if rsi < 40 else 'bearish' if rsi > 60 else 'neutral'}."
            ),
            ReasoningStep(
                step_number=3,
                title="SENTIMENT REVIEW",
                reasoning="News sentiment analysis indicates mixed signals. Recent headlines "
                         "show both positive and negative factors affecting the asset.",
                conclusion="Sentiment is neutral with slight positive bias."
            ),
            ReasoningStep(
                step_number=4,
                title="RISK ASSESSMENT",
                reasoning=f"Key support at ${price * 0.95:.2f}, resistance at ${price * 1.05:.2f}. "
                         f"Risk/reward ratio is approximately 1:1.5.",
                conclusion="Risk is manageable with proper position sizing."
            )
        ]

        raw_response = f"""
STEP 1 - PRICE ACTION ANALYSIS:
{steps[0].reasoning}
{steps[0].conclusion}

STEP 2 - MOMENTUM EVALUATION:
{steps[1].reasoning}
{steps[1].conclusion}

STEP 3 - SENTIMENT REVIEW:
{steps[2].reasoning}
{steps[2].conclusion}

STEP 4 - RISK ASSESSMENT:
{steps[3].reasoning}
{steps[3].conclusion}

SIGNAL: {signal}
CONFIDENCE: {int(confidence * 100)}%
ENTRY: ${price:.2f}
STOP_LOSS: ${price * 0.95:.2f}
TAKE_PROFIT: ${price * 1.08:.2f}
"""

        return CoTAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            steps=steps,
            final_signal=signal,
            confidence=confidence,
            reasoning_summary=f"{symbol}: {signal} ({confidence:.0%} confidence)",
            raw_response=raw_response
        )


def main():
    """Demonstrate Chain-of-Thought analysis."""
    print("Chain-of-Thought Trading Analysis Demo")
    print("=" * 50)

    # Use mock analyzer (no API key required)
    analyzer = MockChainOfThoughtAnalyzer()

    # Sample market data
    market_data = {
        "price": 43250,
        "change_24h": -2.5,
        "volume": 1500000000,
        "rsi": 35,
        "macd": -150,
        "sma_50": 44000,
        "sma_200": 42000
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
    print(f"\n{'='*50}")
    print("REASONING CHAIN:")
    print("="*50)

    for step in result.steps:
        print(f"\nSTEP {step.step_number}: {step.title}")
        print("-" * 40)
        print(step.reasoning)
        print(f"\nConclusion: {step.conclusion}")

    print(f"\n{'='*50}")
    print("FINAL RECOMMENDATION:")
    print("="*50)
    print(f"Signal: {result.final_signal}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nSummary: {result.reasoning_summary}")


if __name__ == "__main__":
    main()
```

### 02: Multi-Step Signal Generation

```python
# python/02_signal_generation.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class CoTSignal:
    """Trading signal with chain-of-thought reasoning."""
    symbol: str
    signal: Signal
    confidence: float
    reasoning_chain: List[str]
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MultiStepSignalGenerator:
    """
    Generate trading signals using multi-step reasoning.

    Each signal is produced through a chain of analytical steps,
    providing full transparency into the decision process.
    """

    def __init__(
        self,
        risk_tolerance: float = 0.02,  # 2% risk per trade
        min_confidence: float = 0.6
    ):
        self.risk_tolerance = risk_tolerance
        self.min_confidence = min_confidence
        self.reasoning_chain: List[str] = []

    def _add_reasoning(self, step: str):
        """Add a step to the reasoning chain."""
        self.reasoning_chain.append(step)

    def generate_signal(
        self,
        symbol: str,
        price_data: Dict,
        indicators: Dict,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None
    ) -> CoTSignal:
        """
        Generate a trading signal with full reasoning chain.

        Args:
            symbol: Trading symbol
            price_data: Dict with open, high, low, close, volume
            indicators: Dict with technical indicators
            fundamentals: Optional fundamental data
            sentiment: Optional sentiment scores

        Returns:
            CoTSignal with reasoning chain
        """
        self.reasoning_chain = []

        # Step 1: Trend Analysis
        trend_score, trend_reasoning = self._analyze_trend(price_data, indicators)
        self._add_reasoning(f"STEP 1 - TREND: {trend_reasoning}")

        # Step 2: Momentum Analysis
        momentum_score, momentum_reasoning = self._analyze_momentum(indicators)
        self._add_reasoning(f"STEP 2 - MOMENTUM: {momentum_reasoning}")

        # Step 3: Volume Analysis
        volume_score, volume_reasoning = self._analyze_volume(price_data)
        self._add_reasoning(f"STEP 3 - VOLUME: {volume_reasoning}")

        # Step 4: Sentiment Analysis (if available)
        if sentiment:
            sentiment_score, sentiment_reasoning = self._analyze_sentiment(sentiment)
            self._add_reasoning(f"STEP 4 - SENTIMENT: {sentiment_reasoning}")
        else:
            sentiment_score = 0
            self._add_reasoning("STEP 4 - SENTIMENT: No sentiment data available, neutral assumption.")

        # Step 5: Risk/Reward Calculation
        rr_score, rr_reasoning, levels = self._calculate_risk_reward(
            price_data, indicators
        )
        self._add_reasoning(f"STEP 5 - RISK/REWARD: {rr_reasoning}")

        # Step 6: Aggregate and Decide
        final_signal, confidence, decision_reasoning = self._aggregate_signals(
            trend_score, momentum_score, volume_score, sentiment_score, rr_score
        )
        self._add_reasoning(f"STEP 6 - DECISION: {decision_reasoning}")

        # Calculate position size
        position_size = self._calculate_position_size(
            confidence, levels.get('entry', price_data['close']),
            levels.get('stop_loss', price_data['close'] * 0.95)
        )

        return CoTSignal(
            symbol=symbol,
            signal=final_signal,
            confidence=confidence,
            reasoning_chain=self.reasoning_chain.copy(),
            entry_price=levels.get('entry'),
            stop_loss=levels.get('stop_loss'),
            take_profit=levels.get('take_profit'),
            position_size_pct=position_size
        )

    def _analyze_trend(
        self,
        price_data: Dict,
        indicators: Dict
    ) -> Tuple[float, str]:
        """Analyze price trend. Returns score (-1 to 1) and reasoning."""
        close = price_data['close']
        sma_20 = indicators.get('sma_20', close)
        sma_50 = indicators.get('sma_50', close)
        sma_200 = indicators.get('sma_200', close)

        score = 0.0
        reasons = []

        # Price vs SMAs
        if close > sma_20:
            score += 0.2
            reasons.append("Price above SMA(20)")
        else:
            score -= 0.2
            reasons.append("Price below SMA(20)")

        if close > sma_50:
            score += 0.3
            reasons.append("Price above SMA(50)")
        else:
            score -= 0.3
            reasons.append("Price below SMA(50)")

        if close > sma_200:
            score += 0.3
            reasons.append("Price above SMA(200) - bullish long-term trend")
        else:
            score -= 0.3
            reasons.append("Price below SMA(200) - bearish long-term trend")

        # SMA alignment
        if sma_20 > sma_50 > sma_200:
            score += 0.2
            reasons.append("SMAs properly aligned (20>50>200)")
        elif sma_20 < sma_50 < sma_200:
            score -= 0.2
            reasons.append("SMAs negatively aligned (20<50<200)")

        score = np.clip(score, -1, 1)
        reasoning = f"Trend score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_momentum(self, indicators: Dict) -> Tuple[float, str]:
        """Analyze momentum indicators. Returns score (-1 to 1) and reasoning."""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        score = 0.0
        reasons = []

        # RSI analysis
        if rsi < 30:
            score += 0.4
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 45:
            score += 0.2
            reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.4
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 55:
            score -= 0.2
            reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        else:
            reasons.append(f"RSI neutral ({rsi:.1f})")

        # MACD analysis
        if macd > macd_signal:
            score += 0.3
            reasons.append("MACD above signal line (bullish)")
        else:
            score -= 0.3
            reasons.append("MACD below signal line (bearish)")

        if macd > 0:
            score += 0.15
            reasons.append("MACD positive")
        else:
            score -= 0.15
            reasons.append("MACD negative")

        score = np.clip(score, -1, 1)
        reasoning = f"Momentum score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_volume(self, price_data: Dict) -> Tuple[float, str]:
        """Analyze volume patterns. Returns score (-1 to 1) and reasoning."""
        volume = price_data.get('volume', 0)
        avg_volume = price_data.get('avg_volume', volume)
        close = price_data['close']
        prev_close = price_data.get('prev_close', close)

        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        price_change = (close - prev_close) / prev_close if prev_close > 0 else 0

        score = 0.0
        reasons = []

        # Volume relative to average
        if volume_ratio > 1.5:
            if price_change > 0:
                score += 0.4
                reasons.append(f"High volume ({volume_ratio:.1f}x avg) on up day - strong buying")
            else:
                score -= 0.4
                reasons.append(f"High volume ({volume_ratio:.1f}x avg) on down day - strong selling")
        elif volume_ratio > 1.0:
            if price_change > 0:
                score += 0.2
                reasons.append(f"Above-average volume ({volume_ratio:.1f}x) confirms upward move")
            else:
                score -= 0.2
                reasons.append(f"Above-average volume ({volume_ratio:.1f}x) confirms downward move")
        else:
            reasons.append(f"Below-average volume ({volume_ratio:.1f}x) - weak conviction")

        score = np.clip(score, -1, 1)
        reasoning = f"Volume score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_sentiment(self, sentiment: Dict) -> Tuple[float, str]:
        """Analyze sentiment data. Returns score (-1 to 1) and reasoning."""
        news_sentiment = sentiment.get('news', 0)
        social_sentiment = sentiment.get('social', 0)
        analyst_rating = sentiment.get('analyst', 0)

        # Weight sentiment sources
        score = (
            news_sentiment * 0.4 +
            social_sentiment * 0.3 +
            analyst_rating * 0.3
        )

        reasons = []
        if news_sentiment > 0.3:
            reasons.append("Positive news sentiment")
        elif news_sentiment < -0.3:
            reasons.append("Negative news sentiment")

        if social_sentiment > 0.3:
            reasons.append("Bullish social media sentiment")
        elif social_sentiment < -0.3:
            reasons.append("Bearish social media sentiment")

        if analyst_rating > 0.3:
            reasons.append("Positive analyst outlook")
        elif analyst_rating < -0.3:
            reasons.append("Negative analyst outlook")

        if not reasons:
            reasons.append("Mixed/neutral sentiment across sources")

        score = np.clip(score, -1, 1)
        reasoning = f"Sentiment score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _calculate_risk_reward(
        self,
        price_data: Dict,
        indicators: Dict
    ) -> Tuple[float, str, Dict]:
        """Calculate risk/reward and key levels."""
        close = price_data['close']
        high = price_data.get('high', close)
        low = price_data.get('low', close)
        atr = indicators.get('atr', close * 0.02)

        # Support and resistance estimation
        support = low - (0.5 * atr)
        resistance = high + (0.5 * atr)

        # Entry, stop, target levels
        entry = close
        stop_loss = support - (0.5 * atr)
        take_profit = resistance + atr

        # Risk/reward ratio
        risk = entry - stop_loss
        reward = take_profit - entry
        rr_ratio = reward / risk if risk > 0 else 0

        # Score based on R/R ratio
        if rr_ratio >= 3:
            score = 0.8
            assessment = "Excellent"
        elif rr_ratio >= 2:
            score = 0.5
            assessment = "Good"
        elif rr_ratio >= 1.5:
            score = 0.2
            assessment = "Acceptable"
        elif rr_ratio >= 1:
            score = 0
            assessment = "Marginal"
        else:
            score = -0.5
            assessment = "Poor"

        reasoning = (
            f"R/R ratio: {rr_ratio:.2f} ({assessment}). "
            f"Entry: ${entry:.2f}, Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}. "
            f"Risk: ${risk:.2f}, Potential reward: ${reward:.2f}"
        )

        levels = {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'support': support,
            'resistance': resistance
        }

        return score, reasoning, levels

    def _aggregate_signals(
        self,
        trend: float,
        momentum: float,
        volume: float,
        sentiment: float,
        risk_reward: float
    ) -> Tuple[Signal, float, str]:
        """Aggregate all scores into final signal."""
        # Weight the factors
        weights = {
            'trend': 0.25,
            'momentum': 0.25,
            'volume': 0.15,
            'sentiment': 0.15,
            'risk_reward': 0.20
        }

        weighted_score = (
            trend * weights['trend'] +
            momentum * weights['momentum'] +
            volume * weights['volume'] +
            sentiment * weights['sentiment'] +
            risk_reward * weights['risk_reward']
        )

        # Calculate confidence
        score_variance = np.var([trend, momentum, volume, sentiment, risk_reward])
        alignment_factor = 1 - min(score_variance, 0.5)  # Lower variance = higher confidence
        confidence = 0.5 + (abs(weighted_score) * 0.4 * alignment_factor)
        confidence = min(confidence, 0.95)

        # Determine signal
        if weighted_score > 0.5:
            signal = Signal.STRONG_BUY
        elif weighted_score > 0.2:
            signal = Signal.BUY
        elif weighted_score < -0.5:
            signal = Signal.STRONG_SELL
        elif weighted_score < -0.2:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        reasoning = (
            f"Weighted score: {weighted_score:.3f}. "
            f"Factors: Trend({trend:.2f}), Momentum({momentum:.2f}), "
            f"Volume({volume:.2f}), Sentiment({sentiment:.2f}), R/R({risk_reward:.2f}). "
            f"Signal: {signal.name} with {confidence:.1%} confidence."
        )

        return signal, confidence, reasoning

    def _calculate_position_size(
        self,
        confidence: float,
        entry: float,
        stop_loss: float
    ) -> float:
        """Calculate position size as percentage of portfolio."""
        if confidence < self.min_confidence:
            return 0.0

        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0.0

        # Scale position by confidence
        confidence_factor = (confidence - self.min_confidence) / (1 - self.min_confidence)
        base_risk = self.risk_tolerance * confidence_factor

        # This would be: (portfolio_value * base_risk) / risk_per_share
        # For now, return the risk percentage
        return base_risk


def main():
    """Demonstrate multi-step signal generation."""
    print("Multi-Step Signal Generation Demo")
    print("=" * 50)

    generator = MultiStepSignalGenerator()

    # Sample market data
    price_data = {
        'open': 42500,
        'high': 43800,
        'low': 42200,
        'close': 43250,
        'prev_close': 42800,
        'volume': 25000000000,
        'avg_volume': 20000000000
    }

    indicators = {
        'rsi': 55,
        'macd': 250,
        'macd_signal': 180,
        'sma_20': 42800,
        'sma_50': 41500,
        'sma_200': 38000,
        'atr': 800
    }

    sentiment = {
        'news': 0.35,
        'social': 0.25,
        'analyst': 0.40
    }

    # Generate signal
    signal = generator.generate_signal(
        "BTCUSDT",
        price_data,
        indicators,
        sentiment=sentiment
    )

    print(f"\nSignal for: {signal.symbol}")
    print(f"Generated at: {signal.timestamp}")
    print("\n" + "=" * 50)
    print("REASONING CHAIN:")
    print("=" * 50)

    for step in signal.reasoning_chain:
        print(f"\n{step}")

    print("\n" + "=" * 50)
    print("FINAL SIGNAL:")
    print("=" * 50)
    print(f"Signal: {signal.signal.name}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Entry: ${signal.entry_price:,.2f}")
    print(f"Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"Take Profit: ${signal.take_profit:,.2f}")
    print(f"Position Size: {signal.position_size_pct:.2%} of portfolio")


if __name__ == "__main__":
    main()
```

### 03: Risk-Adjusted Position Sizing

```python
# python/03_position_sizing.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation with reasoning."""
    symbol: str
    recommended_size: float
    size_in_units: int
    risk_amount: float
    reasoning_chain: List[str]
    risk_metrics: Dict[str, float]


class CoTPositionSizer:
    """
    Chain-of-Thought position sizing for trading.

    Uses explicit reasoning steps to determine optimal position sizes
    based on risk tolerance, account size, and market conditions.
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade: float = 0.02,  # 2%
        max_position_size: float = 0.10,   # 10%
        max_correlated_exposure: float = 0.25  # 25%
    ):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_correlated_exposure = max_correlated_exposure
        self.reasoning_chain: List[str] = []

    def _add_reasoning(self, step: str):
        """Add a step to the reasoning chain."""
        self.reasoning_chain.append(step)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        current_positions: Optional[Dict[str, float]] = None,
        correlations: Optional[Dict[str, float]] = None,
        volatility: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size with explicit reasoning chain.

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            signal_confidence: Confidence in the signal (0-1)
            current_positions: Dict of symbol -> position value
            correlations: Dict of symbol -> correlation with new trade
            volatility: Current volatility (e.g., ATR as % of price)

        Returns:
            PositionSizeResult with recommended size and reasoning
        """
        self.reasoning_chain = []
        current_positions = current_positions or {}
        correlations = correlations or {}

        # Step 1: Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        risk_percent = risk_per_share / entry_price
        self._add_reasoning(
            f"STEP 1 - RISK PER SHARE: "
            f"Entry ${entry_price:.2f}, Stop ${stop_loss:.2f}. "
            f"Risk per share: ${risk_per_share:.2f} ({risk_percent:.2%} of entry price)."
        )

        # Step 2: Calculate base position size from risk tolerance
        risk_amount = self.account_size * self.max_risk_per_trade
        base_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        base_position_value = base_shares * entry_price
        base_position_pct = base_position_value / self.account_size
        self._add_reasoning(
            f"STEP 2 - BASE POSITION SIZE: "
            f"Max risk per trade: {self.max_risk_per_trade:.1%} = ${risk_amount:,.2f}. "
            f"Base position: {base_shares} shares (${base_position_value:,.2f}, {base_position_pct:.1%} of portfolio)."
        )

        # Step 3: Apply confidence adjustment
        confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
        confidence_adjusted_shares = int(base_shares * confidence_multiplier)
        self._add_reasoning(
            f"STEP 3 - CONFIDENCE ADJUSTMENT: "
            f"Signal confidence: {signal_confidence:.1%}. "
            f"Confidence multiplier: {confidence_multiplier:.2f}. "
            f"Adjusted shares: {confidence_adjusted_shares}."
        )

        # Step 4: Apply volatility adjustment
        if volatility:
            vol_multiplier = self._calculate_volatility_multiplier(volatility)
            volatility_adjusted_shares = int(confidence_adjusted_shares * vol_multiplier)
            self._add_reasoning(
                f"STEP 4 - VOLATILITY ADJUSTMENT: "
                f"Current volatility: {volatility:.2%}. "
                f"Volatility multiplier: {vol_multiplier:.2f}. "
                f"Vol-adjusted shares: {volatility_adjusted_shares}."
            )
        else:
            volatility_adjusted_shares = confidence_adjusted_shares
            self._add_reasoning(
                f"STEP 4 - VOLATILITY ADJUSTMENT: "
                f"No volatility data provided. Skipping adjustment."
            )

        # Step 5: Check correlation constraints
        corr_adjusted_shares, corr_reasoning = self._apply_correlation_constraint(
            volatility_adjusted_shares,
            entry_price,
            current_positions,
            correlations
        )
        self._add_reasoning(f"STEP 5 - CORRELATION CHECK: {corr_reasoning}")

        # Step 6: Apply maximum position size constraint
        max_shares = int((self.account_size * self.max_position_size) / entry_price)
        final_shares = min(corr_adjusted_shares, max_shares)
        if corr_adjusted_shares > max_shares:
            self._add_reasoning(
                f"STEP 6 - MAX SIZE CONSTRAINT: "
                f"Position capped at {self.max_position_size:.0%} of portfolio. "
                f"Reduced from {corr_adjusted_shares} to {final_shares} shares."
            )
        else:
            self._add_reasoning(
                f"STEP 6 - MAX SIZE CONSTRAINT: "
                f"Position ({final_shares} shares) within max limit ({max_shares} shares). No adjustment needed."
            )

        # Step 7: Final recommendation
        final_position_value = final_shares * entry_price
        final_position_pct = final_position_value / self.account_size
        final_risk = final_shares * risk_per_share
        final_risk_pct = final_risk / self.account_size

        self._add_reasoning(
            f"STEP 7 - FINAL RECOMMENDATION: "
            f"Buy {final_shares} shares at ${entry_price:.2f} = ${final_position_value:,.2f} "
            f"({final_position_pct:.2%} of portfolio). "
            f"Risk: ${final_risk:,.2f} ({final_risk_pct:.2%} of portfolio)."
        )

        risk_metrics = {
            'risk_per_share': risk_per_share,
            'risk_percent_of_entry': risk_percent,
            'total_risk_amount': final_risk,
            'total_risk_pct': final_risk_pct,
            'position_pct': final_position_pct,
            'confidence_multiplier': confidence_multiplier,
        }

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=final_position_pct,
            size_in_units=final_shares,
            risk_amount=final_risk,
            reasoning_chain=self.reasoning_chain.copy(),
            risk_metrics=risk_metrics
        )

    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position multiplier based on signal confidence."""
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.8:
            return 0.8
        elif confidence >= 0.7:
            return 0.6
        elif confidence >= 0.6:
            return 0.4
        else:
            return 0.2

    def _calculate_volatility_multiplier(self, volatility: float) -> float:
        """Adjust position size based on volatility."""
        # Higher volatility = smaller position
        # Assume "normal" volatility is 2%
        normal_vol = 0.02

        if volatility <= normal_vol * 0.5:
            return 1.2  # Low vol, can size up slightly
        elif volatility <= normal_vol:
            return 1.0
        elif volatility <= normal_vol * 1.5:
            return 0.75
        elif volatility <= normal_vol * 2:
            return 0.5
        else:
            return 0.25  # Very high vol, reduce significantly

    def _apply_correlation_constraint(
        self,
        shares: int,
        entry_price: float,
        current_positions: Dict[str, float],
        correlations: Dict[str, float]
    ) -> Tuple[int, str]:
        """Apply correlation-based position constraints."""
        if not current_positions or not correlations:
            return shares, "No existing positions or correlation data. No adjustment needed."

        # Calculate correlated exposure
        correlated_exposure = 0
        position_value = shares * entry_price

        for symbol, pos_value in current_positions.items():
            corr = correlations.get(symbol, 0)
            if corr > 0.5:  # Only count significantly correlated positions
                correlated_exposure += pos_value * corr

        # Add new position's contribution
        new_total_correlated = correlated_exposure + position_value
        max_correlated = self.account_size * self.max_correlated_exposure

        if new_total_correlated > max_correlated:
            # Need to reduce position
            available_for_new = max_correlated - correlated_exposure
            adjusted_shares = int(max(0, available_for_new / entry_price))
            reasoning = (
                f"Correlated exposure would be ${new_total_correlated:,.2f} "
                f"(limit: ${max_correlated:,.2f}). "
                f"Reduced from {shares} to {adjusted_shares} shares."
            )
            return adjusted_shares, reasoning
        else:
            reasoning = (
                f"Correlated exposure: ${new_total_correlated:,.2f} "
                f"(within {self.max_correlated_exposure:.0%} limit). Position size OK."
            )
            return shares, reasoning


def main():
    """Demonstrate chain-of-thought position sizing."""
    print("Chain-of-Thought Position Sizing Demo")
    print("=" * 50)

    # Initialize position sizer
    sizer = CoTPositionSizer(
        account_size=100000,
        max_risk_per_trade=0.02,
        max_position_size=0.10,
        max_correlated_exposure=0.25
    )

    # Calculate position for a trade
    result = sizer.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=43250,
        stop_loss=42000,
        signal_confidence=0.75,
        current_positions={
            "ETHUSDT": 15000,
            "SOLUSDT": 5000
        },
        correlations={
            "ETHUSDT": 0.85,  # High correlation
            "SOLUSDT": 0.70   # Moderate correlation
        },
        volatility=0.025  # 2.5% ATR
    )

    print(f"\nPosition Sizing for: {result.symbol}")
    print("\n" + "=" * 50)
    print("REASONING CHAIN:")
    print("=" * 50)

    for step in result.reasoning_chain:
        print(f"\n{step}")

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(f"Recommended Position: {result.size_in_units} units ({result.recommended_size:.2%} of portfolio)")
    print(f"Risk Amount: ${result.risk_amount:,.2f}")
    print(f"\nRisk Metrics:")
    for key, value in result.risk_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
```

### 04: Full Backtesting Pipeline

```python
# python/04_backtest.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """A single trade with reasoning."""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class BacktestResult:
    """Results from backtesting with full reasoning audit."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    reasoning_audit: List[Dict]


class CoTBacktester:
    """
    Backtest Chain-of-Thought trading strategies.

    This backtester maintains full audit trails of all reasoning
    chains that led to trading decisions.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.10,
        risk_per_trade_pct: float = 0.02,
        commission_pct: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.commission_pct = commission_pct

        self.capital = initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.reasoning_audit: List[Dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        signal_generator,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            prices: DataFrame with OHLCV data, indexed by datetime
            signal_generator: Signal generator with generate_signal method
            start_date: Start of backtest period
            end_date: End of backtest period

        Returns:
            BacktestResult with trades, equity curve, and reasoning audit
        """
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.reasoning_audit = []

        # Filter date range
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        # Iterate through each bar
        for timestamp in prices.index:
            bar = prices.loc[timestamp]

            # Update equity
            equity = self._calculate_equity(bar)
            self.equity_history.append((timestamp, equity))

            # Check for exits on existing positions
            self._check_exits(timestamp, bar)

            # Generate signals and potentially enter new positions
            self._process_signals(timestamp, bar, signal_generator)

        # Close any remaining positions
        final_bar = prices.iloc[-1]
        self._close_all_positions(prices.index[-1], final_bar)

        # Calculate metrics
        equity_series = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        metrics = self._calculate_metrics(equity_series)

        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=equity_series,
            metrics=metrics,
            reasoning_audit=self.reasoning_audit
        )

    def _calculate_equity(self, bar) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital

        for symbol, trade in self.positions.items():
            current_price = bar['close'] if isinstance(bar, pd.Series) else bar
            if trade.direction == TradeDirection.LONG:
                unrealized_pnl = (current_price - trade.entry_price) * trade.size
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.size
            equity += unrealized_pnl

        return equity

    def _check_exits(self, timestamp: datetime, bar):
        """Check if any positions should be exited."""
        symbols_to_close = []

        for symbol, trade in self.positions.items():
            current_price = bar['close']

            # Simple exit logic: close after 5 bars or if significant move
            # In real implementation, would check stop loss and take profit
            entry_return = (current_price - trade.entry_price) / trade.entry_price

            if trade.direction == TradeDirection.LONG:
                if entry_return <= -0.02:  # 2% stop loss hit
                    symbols_to_close.append((symbol, "Stop loss hit", current_price))
                elif entry_return >= 0.04:  # 4% take profit hit
                    symbols_to_close.append((symbol, "Take profit hit", current_price))
            else:
                if entry_return >= 0.02:  # Short stop loss
                    symbols_to_close.append((symbol, "Stop loss hit (short)", current_price))
                elif entry_return <= -0.04:  # Short take profit
                    symbols_to_close.append((symbol, "Take profit hit (short)", current_price))

        for symbol, reason, price in symbols_to_close:
            self._close_position(symbol, timestamp, price, reason)

    def _process_signals(
        self,
        timestamp: datetime,
        bar,
        signal_generator
    ):
        """Process signals and enter positions."""
        # Prepare data for signal generation
        price_data = {
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar.get('volume', 0),
            'prev_close': bar['close'] * 0.99  # Mock previous close
        }

        indicators = {
            'rsi': 50 + np.random.randn() * 15,
            'macd': np.random.randn() * 100,
            'macd_signal': np.random.randn() * 100,
            'sma_20': bar['close'] * (1 + np.random.randn() * 0.02),
            'sma_50': bar['close'] * (1 + np.random.randn() * 0.03),
            'sma_200': bar['close'] * (1 + np.random.randn() * 0.05),
            'atr': bar['close'] * 0.02
        }

        # Generate signal
        signal = signal_generator.generate_signal(
            "BTCUSDT",
            price_data,
            indicators
        )

        # Record reasoning
        self.reasoning_audit.append({
            'timestamp': timestamp,
            'signal': signal.signal.name,
            'confidence': signal.confidence,
            'reasoning_chain': signal.reasoning_chain
        })

        # Enter position if signal is strong enough
        if signal.confidence >= 0.6 and "BTCUSDT" not in self.positions:
            if signal.signal.value > 0:  # BUY or STRONG_BUY
                self._enter_position(
                    "BTCUSDT",
                    TradeDirection.LONG,
                    timestamp,
                    bar['close'],
                    signal.stop_loss,
                    signal.reasoning_chain
                )
            elif signal.signal.value < 0:  # SELL or STRONG_SELL
                self._enter_position(
                    "BTCUSDT",
                    TradeDirection.SHORT,
                    timestamp,
                    bar['close'],
                    signal.stop_loss,
                    signal.reasoning_chain
                )

    def _enter_position(
        self,
        symbol: str,
        direction: TradeDirection,
        timestamp: datetime,
        price: float,
        stop_loss: float,
        reasoning_chain: List[str]
    ):
        """Enter a new position."""
        # Calculate position size based on risk
        risk_per_share = abs(price - stop_loss) if stop_loss else price * 0.02
        risk_amount = self.capital * self.risk_per_trade_pct
        size = risk_amount / risk_per_share

        # Cap at max position size
        max_size = (self.capital * self.max_position_pct) / price
        size = min(size, max_size)

        # Apply commission
        commission = size * price * self.commission_pct
        self.capital -= commission

        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            size=size,
            reasoning_chain=reasoning_chain
        )

        self.positions[symbol] = trade

        self.reasoning_audit.append({
            'timestamp': timestamp,
            'action': 'ENTRY',
            'symbol': symbol,
            'direction': direction.name,
            'price': price,
            'size': size,
            'reasoning': reasoning_chain[-1] if reasoning_chain else "No reasoning"
        })

    def _close_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        reason: str
    ):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.exit_time = timestamp
        trade.exit_price = price

        # Calculate PnL
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - price) * trade.size

        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size)

        # Apply commission
        commission = trade.size * price * self.commission_pct
        trade.pnl -= commission

        # Update capital
        self.capital += trade.pnl

        # Add exit reasoning
        trade.reasoning_chain.append(f"EXIT: {reason} at ${price:.2f}")

        self.closed_trades.append(trade)
        del self.positions[symbol]

        self.reasoning_audit.append({
            'timestamp': timestamp,
            'action': 'EXIT',
            'symbol': symbol,
            'price': price,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'reason': reason
        })

    def _close_all_positions(self, timestamp: datetime, bar):
        """Close all remaining positions."""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, timestamp, bar['close'], "End of backtest")

    def _calculate_metrics(self, equity: pd.Series) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        returns = equity.pct_change().dropna()

        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Risk metrics
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity.iloc[-1]
        }


def generate_mock_prices(
    start_date: str = "2024-01-01",
    periods: int = 252,
    initial_price: float = 40000,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate mock price data for backtesting."""
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq='D')

    # Generate price path
    returns = np.random.randn(periods) * volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(periods) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(periods) * 0.015)),
        'low': prices * (1 - np.abs(np.random.randn(periods) * 0.015)),
        'close': prices,
        'volume': np.random.randint(1e9, 5e10, periods)
    }, index=dates)

    return df


def main():
    """Demonstrate Chain-of-Thought backtesting."""
    from signal_generation import MultiStepSignalGenerator

    print("Chain-of-Thought Backtesting Demo")
    print("=" * 50)

    # Generate mock data
    prices = generate_mock_prices()

    # Initialize components
    signal_generator = MultiStepSignalGenerator()
    backtester = CoTBacktester(
        initial_capital=100000,
        max_position_pct=0.10,
        risk_per_trade_pct=0.02
    )

    # Run backtest
    result = backtester.run(prices, signal_generator)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {result.metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {result.metrics['num_trades']}")
    print(f"  Win Rate: {result.metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"  Final Equity: ${result.metrics['final_equity']:,.2f}")

    print(f"\nSample Trade Reasoning (first 3 trades):")
    for i, trade in enumerate(result.trades[:3], 1):
        print(f"\n  Trade {i}: {trade.symbol}")
        print(f"    Direction: {trade.direction.name}")
        print(f"    Entry: ${trade.entry_price:.2f} at {trade.entry_time}")
        print(f"    Exit: ${trade.exit_price:.2f} at {trade.exit_time}")
        print(f"    PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        print(f"    Reasoning: {trade.reasoning_chain[-1][:80]}...")

    print(f"\nReasoning Audit: {len(result.reasoning_audit)} entries recorded")
    print("Full audit trail available for compliance and review.")


if __name__ == "__main__":
    main()
```

## Rust Implementation

The Rust implementation provides high-performance Chain-of-Thought trading analysis.

### Project Structure

```
rust_cot_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── analysis.rs      # CoT analysis logic
│   ├── signals.rs       # Signal generation
│   ├── position.rs      # Position sizing
│   ├── backtest.rs      # Backtesting engine
│   ├── data.rs          # Data loading
│   ├── api.rs           # LLM API client
│   └── bin/
│       ├── analysis_demo.rs
│       ├── signal_demo.rs
│       └── backtest_demo.rs
```

### Quick Start (Rust)

```bash
cd rust_cot_trading
cargo build --release
cargo run --bin analysis_demo
```

## Python Implementation

### Directory Structure

```
python/
├── __init__.py
├── cot_analyzer.py      # Chain-of-Thought analyzer
├── signal_generator.py  # Multi-step signal generation
├── position_sizer.py    # Position sizing with reasoning
├── backtest.py          # Backtesting engine
├── data_loader.py       # Data loading utilities
├── requirements.txt
└── examples/
    ├── 01_basic_analysis.py
    ├── 02_signal_generation.py
    ├── 03_position_sizing.py
    └── 04_backtest_demo.py
```

### Quick Start (Python)

```bash
cd python
pip install -r requirements.txt
python examples/01_basic_analysis.py
```

## Best Practices

### 1. Prompt Engineering for Trading

```python
# DO: Be specific about the analysis framework
prompt = """
Analyze using these specific steps:
1. Check RSI for oversold/overbought
2. Verify MACD crossover direction
3. Confirm volume supports price action
4. Calculate risk/reward ratio
5. Provide specific entry, stop, and target
"""

# DON'T: Use vague prompts
prompt = "Tell me if I should buy this stock"
```

### 2. Self-Consistency for Reliability

```python
def get_reliable_signal(analyzer, data, n_samples=5):
    """Use self-consistency for more reliable signals."""
    signals = []
    for _ in range(n_samples):
        result = analyzer.analyze(data)
        signals.append(result.final_signal)

    # Majority vote
    from collections import Counter
    most_common = Counter(signals).most_common(1)[0]

    return {
        'signal': most_common[0],
        'agreement': most_common[1] / n_samples,
        'all_signals': signals
    }
```

### 3. Reasoning Audit Trail

```python
class AuditableAnalyzer:
    """Maintain full audit trail for compliance."""

    def __init__(self):
        self.audit_log = []

    def analyze(self, symbol, data):
        result = self._perform_analysis(symbol, data)

        # Log for audit
        self.audit_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'input_data': data,
            'reasoning_chain': result.reasoning_chain,
            'signal': result.signal,
            'confidence': result.confidence
        })

        return result

    def export_audit(self, filepath):
        """Export audit trail for compliance review."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2, default=str)
```

### 4. Error Handling in Reasoning Chains

```python
def safe_reasoning_step(step_func, fallback_result):
    """Wrap reasoning steps with error handling."""
    try:
        return step_func()
    except Exception as e:
        return {
            'result': fallback_result,
            'error': str(e),
            'reasoning': f"Step failed: {e}. Using fallback."
        }
```

### 5. Combining CoT with Traditional Signals

```python
class HybridSignalGenerator:
    """Combine CoT reasoning with quantitative signals."""

    def __init__(self, cot_weight=0.4, quant_weight=0.6):
        self.cot_weight = cot_weight
        self.quant_weight = quant_weight

    def generate(self, data):
        # Get CoT-based signal
        cot_signal = self.cot_analyzer.analyze(data)

        # Get quantitative signal
        quant_signal = self.quant_model.predict(data)

        # Combine with reasoning
        combined_score = (
            cot_signal.score * self.cot_weight +
            quant_signal.score * self.quant_weight
        )

        return {
            'signal': combined_score,
            'cot_reasoning': cot_signal.reasoning_chain,
            'quant_factors': quant_signal.factors,
            'combination_logic': f"CoT({self.cot_weight}) + Quant({self.quant_weight})"
        }
```

## Common Pitfalls

1. **Over-reliance on LLM reasoning**: Always validate with quantitative backtests
2. **Inconsistent reasoning**: Use self-consistency and temperature tuning
3. **Token limits**: Break long analyses into focused sub-analyses
4. **Latency in live trading**: Pre-compute reasoning templates, use caching
5. **Hallucinated data**: Always verify LLM doesn't invent market data

## Resources

### Papers

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   - Wei et al., 2022
   - https://arxiv.org/abs/2201.11903

2. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**
   - Wang et al., 2023
   - https://arxiv.org/abs/2203.11171

3. **Large Language Models are Zero-Shot Reasoners**
   - Kojima et al., 2022
   - https://arxiv.org/abs/2205.11916

### Open-Source Alternatives

- **LangChain**: Framework for chaining LLM calls
- **Guidance**: Constrained generation for structured outputs
- **DSPy**: Programming framework for LLM pipelines

### Related Chapters

- [Chapter 61: FinGPT Financial LLM](../61_fingpt_financial_llm/)
- [Chapter 62: BloombergGPT Trading](../62_bloomberggpt_trading/)
- [Chapter 65: RAG for Trading](../65_rag_for_trading/)
- [Chapter 71: Prompt Engineering Trading](../71_prompt_engineering_trading/)

## Difficulty Level

**Intermediate to Advanced**

### Prerequisites
- Understanding of LLM prompting techniques
- Basic knowledge of trading indicators
- Python programming experience
- Familiarity with backtesting concepts

### Skills You'll Learn
- Chain-of-Thought prompting for finance
- Explainable AI in trading
- Multi-step reasoning systems
- Audit trail implementation for compliance

## References

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.

2. Wang, X., et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.

3. Kojima, T., et al. (2022). "Large Language Models are Zero-Shot Reasoners." NeurIPS 2022.

4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.

5. OpenAI. (2023). "GPT-4 Technical Report." arXiv:2303.08774.
