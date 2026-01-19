# Chapter 63: LLM Alpha Mining — Generating Trading Factors with Large Language Models

This chapter explores **LLM Alpha Mining**, a cutting-edge approach that leverages Large Language Models to automatically discover, generate, and refine alpha factors for quantitative trading. We examine how LLMs can process diverse financial data sources to produce actionable trading signals that outperform traditional factor mining methods.

<p align="center">
<img src="https://i.imgur.com/LLMAlpha.png" width="70%">
</p>

## Contents

1. [Introduction to LLM Alpha Mining](#introduction-to-llm-alpha-mining)
    * [What is Alpha Mining?](#what-is-alpha-mining)
    * [Why LLMs for Alpha Discovery?](#why-llms-for-alpha-discovery)
    * [Key Approaches](#key-approaches)
2. [LLM Alpha Mining Architectures](#llm-alpha-mining-architectures)
    * [LLM as Alpha Generator](#llm-as-alpha-generator)
    * [LLM as Factor Optimizer](#llm-as-factor-optimizer)
    * [Self-Improving Agents (QuantAgent)](#self-improving-agents-quantagent)
3. [Trading Applications](#trading-applications)
    * [Factor Expression Generation](#factor-expression-generation)
    * [Multi-Modal Alpha Mining](#multi-modal-alpha-mining)
    * [Crypto Market Alpha](#crypto-market-alpha)
4. [Practical Examples](#practical-examples)
    * [01: Basic Alpha Factor Generation](#01-basic-alpha-factor-generation)
    * [02: Multi-Source Alpha Mining](#02-multi-source-alpha-mining)
    * [03: Self-Improving Alpha Agent](#03-self-improving-alpha-agent)
    * [04: Backtesting LLM Alphas](#04-backtesting-llm-alphas)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to LLM Alpha Mining

### What is Alpha Mining?

Alpha mining is the process of discovering trading signals (alpha factors) that can predict future asset returns. Traditional quantitative analysts spend months manually crafting mathematical expressions and testing them against historical data.

```
TRADITIONAL ALPHA MINING PROCESS:
┌─────────────────────────────────────────────────────────────────────┐
│  1. HYPOTHESIS FORMATION                                             │
│     Quant: "Maybe momentum over 20 days predicts returns..."        │
│     Time: Days to weeks                                              │
├─────────────────────────────────────────────────────────────────────┤
│  2. FACTOR CONSTRUCTION                                              │
│     Quant: Writes mathematical formula                               │
│     Example: alpha = (close - close_20d_ago) / close_20d_ago        │
│     Time: Hours to days                                              │
├─────────────────────────────────────────────────────────────────────┤
│  3. BACKTESTING                                                      │
│     Quant: Tests against historical data                             │
│     Result: Sharpe = 0.8, IC = 0.03                                  │
│     Time: Hours                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  4. ITERATION                                                        │
│     Quant: "What if I add volume? Different lookback?"              │
│     Repeat steps 1-3 hundreds of times                               │
│     Time: Months                                                     │
└─────────────────────────────────────────────────────────────────────┘

TOTAL TIME: 3-6 months per viable alpha factor
SUCCESS RATE: ~1-5% of tested factors are useful
```

### Why LLMs for Alpha Discovery?

LLMs can dramatically accelerate alpha mining by:

1. **Generating Novel Factor Expressions**
   - LLMs can propose thousands of factor formulas based on financial intuition
   - They understand market microstructure, technical analysis, and fundamental principles

2. **Processing Diverse Data Sources**
   - News, social media, earnings calls, SEC filings
   - Order book data, alternative data
   - Cross-market correlations

3. **Self-Improvement Through Feedback**
   - LLMs can learn from backtesting results
   - Iteratively refine factors based on performance metrics

```
LLM ALPHA MINING ADVANTAGES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  SPEED            Traditional: Months  →  LLM: Hours                 │
│  ─────────────────────────────────────────────────────────────       │
│                                                                      │
│  COVERAGE         Traditional: ~100 factors/month                    │
│                   LLM: ~10,000 factors/day                           │
│  ─────────────────────────────────────────────────────────────       │
│                                                                      │
│  NOVELTY          Traditional: Based on quant's experience           │
│                   LLM: Combines patterns from vast training data     │
│  ─────────────────────────────────────────────────────────────       │
│                                                                      │
│  INTERPRETABILITY Traditional: Human-designed, clear logic           │
│                   LLM: Can explain factor rationale                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Approaches

| Approach | Description | Example |
|----------|-------------|---------|
| **Direct Generation** | LLM outputs factor expressions directly | "Generate a momentum-based alpha factor" |
| **Code Synthesis** | LLM writes Python/Rust code for factors | "Write a factor using RSI and volume" |
| **Agent-Based** | LLM iteratively improves factors | QuantAgent self-improving loop |
| **Multi-Modal** | LLM combines text + numerical data | News sentiment + price data |

## LLM Alpha Mining Architectures

### LLM as Alpha Generator

The simplest approach uses LLMs to directly generate factor expressions:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LLM AS ALPHA GENERATOR                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐   │
│  │                 │     │                   │     │                     │   │
│  │  Market Data    │────▶│      LLM          │────▶│  Factor Expression  │   │
│  │  + Prompt       │     │   (GPT/Claude)    │     │  + Explanation      │   │
│  │                 │     │                   │     │                     │   │
│  └─────────────────┘     └───────────────────┘     └─────────────────────┘   │
│                                                                               │
│  INPUT PROMPT:                                                                │
│  "Generate a mean-reversion alpha factor for BTC using                       │
│   price and volume data with a 5-day lookback period."                       │
│                                                                               │
│  OUTPUT:                                                                      │
│  Factor: (close - rolling_mean(close, 5)) / rolling_std(close, 5)           │
│  Logic: Buy when price is below moving average (oversold)                    │
│  Risk: High volatility assets may trigger false signals                      │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### LLM as Factor Optimizer

More advanced systems use LLMs to refine existing factors:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LLM AS FACTOR OPTIMIZER                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  OPTIMIZATION LOOP:                                                           │
│                                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Initial     │───▶│ Backtest     │───▶│ Performance │───▶│ LLM         │   │
│  │ Factor      │    │ Engine       │    │ Metrics     │    │ Optimizer   │   │
│  └─────────────┘    └──────────────┘    └─────────────┘    └──────┬──────┘   │
│        ▲                                                          │          │
│        │                                                          │          │
│        └──────────────────────────────────────────────────────────┘          │
│                              Refined Factor                                   │
│                                                                               │
│  EXAMPLE ITERATION:                                                           │
│                                                                               │
│  Iteration 1:                                                                 │
│    Factor: momentum_5d                                                        │
│    Sharpe: 0.65, IC: 0.02                                                     │
│                                                                               │
│  Iteration 2:                                                                 │
│    LLM: "Add volume confirmation to reduce false signals"                    │
│    Factor: momentum_5d * sign(volume - avg_volume_20d)                       │
│    Sharpe: 0.89, IC: 0.04                                                     │
│                                                                               │
│  Iteration 3:                                                                 │
│    LLM: "Incorporate volatility regime for risk adjustment"                  │
│    Factor: momentum_5d * volume_signal / realized_vol_10d                    │
│    Sharpe: 1.12, IC: 0.05                                                     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Self-Improving Agents (QuantAgent)

The QuantAgent framework introduces a two-loop self-improvement mechanism:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     QUANTAGENT ARCHITECTURE                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗   │
│  ║                         OUTER LOOP (Learning)                          ║   │
│  ║                                                                        ║   │
│  ║  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐   ║   │
│  ║  │ Generate    │───▶│ Real-World  │───▶│ Update Knowledge Base   │   ║   │
│  ║  │ Factor      │    │ Backtest    │    │ with Feedback           │   ║   │
│  ║  └─────────────┘    └─────────────┘    └─────────────────────────┘   ║   │
│  ║        ▲                                         │                    ║   │
│  ║        │                                         │                    ║   │
│  ║        └─────────────────────────────────────────┘                    ║   │
│  ║                                                                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════╝   │
│                                    │                                          │
│                                    ▼                                          │
│  ╔═══════════════════════════════════════════════════════════════════════╗   │
│  ║                         INNER LOOP (Reasoning)                         ║   │
│  ║                                                                        ║   │
│  ║  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐  ║   │
│  ║  │  Writer    │───▶│  Answer    │───▶│   Judge    │───▶│  Refine  │  ║   │
│  ║  │  Agent     │    │ Generation │    │   Agent    │    │  Answer  │  ║   │
│  ║  └────────────┘    └────────────┘    └────────────┘    └──────────┘  ║   │
│  ║        │                                    │                         ║   │
│  ║        ▼                                    ▼                         ║   │
│  ║  ┌────────────────────────────────────────────────────────────────┐  ║   │
│  ║  │                    KNOWLEDGE BASE                               │  ║   │
│  ║  │  • Successful factor patterns                                   │  ║   │
│  ║  │  • Failed approaches to avoid                                   │  ║   │
│  ║  │  • Market regime insights                                       │  ║   │
│  ║  │  • Performance benchmarks                                       │  ║   │
│  ║  └────────────────────────────────────────────────────────────────┘  ║   │
│  ╚═══════════════════════════════════════════════════════════════════════╝   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Writer Agent**: Generates initial factor expressions using knowledge base
2. **Judge Agent**: Reviews and scores generated factors, provides improvement suggestions
3. **Knowledge Base**: Stores successful patterns and failed attempts
4. **Real-World Feedback**: Backtest results update the knowledge base

## Trading Applications

### Factor Expression Generation

LLMs can generate factors in multiple formats:

```python
# Example factor expressions generated by LLM

# 1. Mathematical Expression
factor_math = "(close - SMA(close, 20)) / STD(close, 20)"

# 2. DSL (Domain-Specific Language)
factor_dsl = "zscore(close, 20) * rank(volume)"

# 3. Python Code
def factor_python(df):
    """
    Mean reversion factor with volume confirmation.
    Buy signal when price is below moving average and volume is increasing.
    """
    zscore = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    volume_signal = df['volume'] / df['volume'].rolling(20).mean()
    return -zscore * np.where(volume_signal > 1, 1.5, 1.0)
```

### Multi-Modal Alpha Mining

Combining text and numerical data for richer signals:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODAL ALPHA MINING                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  TEXT SOURCES                     NUMERICAL SOURCES                           │
│  ─────────────                    ─────────────────                           │
│  • News articles                  • OHLCV price data                          │
│  • Social media                   • Order book snapshots                      │
│  • Earnings calls                 • Funding rates (crypto)                    │
│  • SEC filings                    • Technical indicators                      │
│  • Analyst reports                • On-chain metrics                          │
│                                                                               │
│         │                                  │                                  │
│         ▼                                  ▼                                  │
│  ┌──────────────┐                 ┌──────────────────┐                        │
│  │ Text         │                 │ Numerical        │                        │
│  │ Embedding    │                 │ Feature          │                        │
│  │ (LLM)        │                 │ Engineering      │                        │
│  └──────────────┘                 └──────────────────┘                        │
│         │                                  │                                  │
│         └──────────────┬───────────────────┘                                  │
│                        ▼                                                      │
│              ┌──────────────────────┐                                         │
│              │   FUSION MODULE      │                                         │
│              │   (Cross-Attention)  │                                         │
│              └──────────────────────┘                                         │
│                        │                                                      │
│                        ▼                                                      │
│              ┌──────────────────────┐                                         │
│              │   ALPHA SIGNAL       │                                         │
│              │   Combined signal    │                                         │
│              └──────────────────────┘                                         │
│                                                                               │
│  EXAMPLE:                                                                     │
│  News: "Ethereum upgrade successfully deployed, gas fees drop 40%"           │
│  Price: ETH at $2,450, up 2% in last hour                                    │
│  Volume: 150% of 20-day average                                              │
│                                                                               │
│  Combined Alpha: Strong BUY signal (0.85 confidence)                         │
│  - Positive fundamental development                                           │
│  - Price momentum confirmation                                                │
│  - Volume surge validation                                                    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Crypto Market Alpha

LLMs are particularly effective for crypto markets due to:

1. **24/7 News Cycle**: Continuous information flow requires automated processing
2. **On-Chain Data**: LLMs can interpret blockchain metrics
3. **Social Sentiment**: Twitter, Reddit, Telegram signals
4. **DeFi Metrics**: TVL, yield farming APYs, liquidation levels

```python
# Crypto-specific alpha factors that LLMs can generate

crypto_factors = {
    "funding_momentum": """
        Signal based on perpetual futures funding rate changes.
        Negative funding + price recovery = potential long signal.
    """,

    "whale_accumulation": """
        Monitor large wallet accumulation patterns.
        Increased whale holdings during price consolidation = bullish.
    """,

    "exchange_flow": """
        Track net exchange inflows/outflows.
        Large outflows to cold wallets = reduced sell pressure.
    """,

    "social_momentum": """
        Aggregate sentiment from Twitter, Reddit, Telegram.
        Sentiment divergence from price = potential reversal signal.
    """,

    "defi_tvl_momentum": """
        Total Value Locked momentum in DeFi protocols.
        Increasing TVL + stablecoin inflows = ecosystem growth signal.
    """
}
```

## Practical Examples

### 01: Basic Alpha Factor Generation

```python
# python/01_alpha_generation.py

import openai
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class LLMAlphaGenerator:
    """
    Generate alpha factors using Large Language Models.

    This class demonstrates how to use LLMs to create
    trading factors from natural language descriptions.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        if api_key:
            openai.api_key = api_key

    def generate_factor(
        self,
        asset_type: str,
        factor_type: str,
        lookback_period: int = 20,
        additional_context: str = ""
    ) -> Dict:
        """
        Generate an alpha factor based on specifications.

        Args:
            asset_type: "stock", "crypto", "forex"
            factor_type: "momentum", "mean_reversion", "volatility", etc.
            lookback_period: Number of periods for calculations
            additional_context: Extra instructions or constraints

        Returns:
            Dict containing factor expression, explanation, and code
        """
        prompt = self._build_prompt(
            asset_type, factor_type, lookback_period, additional_context
        )

        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _build_prompt(
        self,
        asset_type: str,
        factor_type: str,
        lookback: int,
        context: str
    ) -> str:
        """Build the prompt for factor generation."""
        return f"""You are a quantitative analyst specializing in alpha factor development.

Generate an alpha factor with the following specifications:
- Asset Type: {asset_type}
- Factor Type: {factor_type}
- Lookback Period: {lookback} periods
- Additional Requirements: {context if context else "None"}

Provide your response in the following format:

FACTOR_NAME: [Name of the factor]

MATHEMATICAL_EXPRESSION: [Mathematical formula using standard notation]

PYTHON_CODE:
```python
def calculate_factor(df):
    # df contains: open, high, low, close, volume
    # Return: Series of factor values
    pass
```

RATIONALE: [2-3 sentences explaining why this factor might work]

RISK_FACTORS: [Potential weaknesses or market conditions where this may fail]

EXPECTED_PERFORMANCE:
- Typical IC: [Expected Information Coefficient range]
- Turnover: [Expected daily turnover percentage]
- Suitable Holding Period: [Intraday/Daily/Weekly]
"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        # For demonstration, return a mock response
        # In production, use actual API call
        return self._mock_response()

    def _mock_response(self) -> str:
        """Mock response for demonstration."""
        return """
FACTOR_NAME: Volume-Adjusted Momentum

MATHEMATICAL_EXPRESSION:
momentum = (close - close_lag_5) / close_lag_5
vol_ratio = volume / SMA(volume, 20)
factor = momentum * log(vol_ratio + 1) * sign(momentum)

PYTHON_CODE:
```python
def calculate_factor(df):
    # Calculate price momentum
    momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

    # Calculate volume ratio
    vol_ma = df['volume'].rolling(20).mean()
    vol_ratio = df['volume'] / vol_ma

    # Combine with volume confirmation
    factor = momentum * np.log(vol_ratio + 1) * np.sign(momentum)

    return factor
```

RATIONALE: This factor combines price momentum with volume confirmation.
Strong price moves accompanied by above-average volume are more likely to
continue, while low-volume moves are often mean-reverting. The log
transformation prevents extreme volume spikes from dominating the signal.

RISK_FACTORS:
- May underperform in low-volatility, range-bound markets
- Volume patterns differ between crypto (24/7) and traditional markets
- Earnings/news events can cause volume spikes that distort signals

EXPECTED_PERFORMANCE:
- Typical IC: 0.02 - 0.05
- Turnover: 15-25% daily
- Suitable Holding Period: Daily to Weekly
"""

    def _parse_response(self, response: str) -> Dict:
        """Parse the LLM response into structured format."""
        # Simple parsing - in production, use more robust parsing
        sections = {}
        current_section = None
        current_content = []

        for line in response.strip().split('\n'):
            if line.startswith('FACTOR_NAME:'):
                sections['name'] = line.replace('FACTOR_NAME:', '').strip()
            elif line.startswith('MATHEMATICAL_EXPRESSION:'):
                current_section = 'expression'
                current_content = []
            elif line.startswith('PYTHON_CODE:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'code'
                current_content = []
            elif line.startswith('RATIONALE:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'rationale'
                current_content = []
            elif line.startswith('RISK_FACTORS:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'risks'
                current_content = []
            elif line.startswith('EXPECTED_PERFORMANCE:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'performance'
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections


def calculate_volume_adjusted_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Example factor implementation.

    This is a standalone function demonstrating the factor logic
    that can be used directly without LLM generation.
    """
    # Calculate price momentum
    momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

    # Calculate volume ratio
    vol_ma = df['volume'].rolling(20).mean()
    vol_ratio = df['volume'] / vol_ma

    # Combine with volume confirmation
    factor = momentum * np.log(vol_ratio + 1) * np.sign(momentum)

    return factor


# Example usage
def main():
    # Initialize generator
    generator = LLMAlphaGenerator()

    # Generate a momentum factor for crypto
    factor_spec = generator.generate_factor(
        asset_type="crypto",
        factor_type="momentum",
        lookback_period=5,
        additional_context="Focus on volume confirmation, suitable for BTC and ETH"
    )

    print("Generated Factor:")
    print(f"Name: {factor_spec.get('name', 'Unknown')}")
    print(f"\nExpression:\n{factor_spec.get('expression', 'N/A')}")
    print(f"\nRationale:\n{factor_spec.get('rationale', 'N/A')}")

    # Test with sample data
    print("\n--- Testing with Sample Data ---")

    # Create sample crypto data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': 40000 + np.random.randn(100).cumsum() * 500,
        'high': 40500 + np.random.randn(100).cumsum() * 500,
        'low': 39500 + np.random.randn(100).cumsum() * 500,
        'close': 40000 + np.random.randn(100).cumsum() * 500,
        'volume': np.random.exponential(1e9, 100)
    }, index=dates)

    # Calculate factor
    df['factor'] = calculate_volume_adjusted_momentum(df)

    print(f"\nFactor Statistics:")
    print(f"  Mean: {df['factor'].mean():.4f}")
    print(f"  Std: {df['factor'].std():.4f}")
    print(f"  Min: {df['factor'].min():.4f}")
    print(f"  Max: {df['factor'].max():.4f}")

    # Show sample values
    print(f"\nSample Factor Values:")
    print(df[['close', 'volume', 'factor']].tail(10))


if __name__ == "__main__":
    main()
```

### 02: Multi-Source Alpha Mining

```python
# python/02_multi_source_alpha.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class AlphaSignal:
    """Represents an alpha signal from any source."""
    timestamp: datetime
    symbol: str
    signal_type: str
    value: float
    confidence: float
    source: str
    metadata: Dict

class MultiSourceAlphaMiner:
    """
    Combine multiple data sources for comprehensive alpha mining.

    This class demonstrates how LLMs can orchestrate alpha generation
    from diverse data sources including price data, news, social media,
    and on-chain metrics for crypto assets.
    """

    def __init__(self):
        self.sources = {}
        self.signal_weights = {
            'price_momentum': 0.25,
            'volume_breakout': 0.20,
            'sentiment_news': 0.20,
            'sentiment_social': 0.15,
            'onchain_flow': 0.20
        }

    def mine_price_alpha(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[AlphaSignal]:
        """
        Extract alpha signals from price data.

        Args:
            df: OHLCV DataFrame
            symbol: Asset symbol

        Returns:
            List of AlphaSignal objects
        """
        signals = []

        # Momentum signal
        momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        momentum_20d = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

        # Latest momentum signal
        latest_momentum = momentum_5d.iloc[-1]
        momentum_confidence = min(abs(latest_momentum) / 0.05, 1.0)  # Normalize

        signals.append(AlphaSignal(
            timestamp=df.index[-1],
            symbol=symbol,
            signal_type='price_momentum',
            value=latest_momentum,
            confidence=momentum_confidence,
            source='price_data',
            metadata={
                'momentum_5d': latest_momentum,
                'momentum_20d': momentum_20d.iloc[-1],
                'close': df['close'].iloc[-1]
            }
        ))

        # Volume breakout signal
        vol_ma = df['volume'].rolling(20).mean()
        vol_ratio = df['volume'] / vol_ma
        latest_vol_ratio = vol_ratio.iloc[-1]

        if latest_vol_ratio > 1.5:  # Significant volume increase
            signals.append(AlphaSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                signal_type='volume_breakout',
                value=np.sign(latest_momentum) * (latest_vol_ratio - 1),
                confidence=min(latest_vol_ratio / 3, 1.0),
                source='price_data',
                metadata={
                    'volume_ratio': latest_vol_ratio,
                    'avg_volume': vol_ma.iloc[-1]
                }
            ))

        return signals

    def mine_sentiment_alpha(
        self,
        news_items: List[Dict],
        symbol: str
    ) -> List[AlphaSignal]:
        """
        Extract alpha signals from news sentiment.

        In production, this would use an LLM for sentiment analysis.
        Here we demonstrate the structure with mock analysis.
        """
        signals = []

        # Mock sentiment analysis
        # In production: sentiment = llm.analyze_sentiment(news_items)
        if news_items:
            # Simple keyword-based sentiment for demo
            positive_keywords = ['surge', 'rally', 'bullish', 'upgrade', 'breakthrough']
            negative_keywords = ['crash', 'drop', 'bearish', 'downgrade', 'concern']

            sentiment_scores = []
            for item in news_items:
                text = item.get('title', '') + ' ' + item.get('content', '')
                text_lower = text.lower()

                pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
                neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

                if pos_count + neg_count > 0:
                    sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    sentiment = 0

                sentiment_scores.append(sentiment)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

            signals.append(AlphaSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type='sentiment_news',
                value=avg_sentiment,
                confidence=len(news_items) / 10,  # More news = higher confidence
                source='news',
                metadata={
                    'num_articles': len(news_items),
                    'individual_scores': sentiment_scores
                }
            ))

        return signals

    def mine_onchain_alpha(
        self,
        onchain_data: Dict,
        symbol: str
    ) -> List[AlphaSignal]:
        """
        Extract alpha signals from on-chain data (for crypto).

        This analyzes exchange flows, whale movements, and
        other blockchain-specific metrics.
        """
        signals = []

        # Exchange net flow signal
        if 'exchange_netflow' in onchain_data:
            netflow = onchain_data['exchange_netflow']
            # Negative netflow (outflow) is bullish
            signal_value = -np.sign(netflow) * np.log(abs(netflow) + 1)

            signals.append(AlphaSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type='onchain_flow',
                value=signal_value,
                confidence=0.7,
                source='onchain',
                metadata={
                    'netflow': netflow,
                    'interpretation': 'outflow' if netflow < 0 else 'inflow'
                }
            ))

        # Whale accumulation signal
        if 'whale_balance_change' in onchain_data:
            whale_change = onchain_data['whale_balance_change']
            signal_value = np.sign(whale_change) * min(abs(whale_change) / 1000, 1)

            signals.append(AlphaSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type='whale_accumulation',
                value=signal_value,
                confidence=0.6,
                source='onchain',
                metadata={
                    'balance_change': whale_change
                }
            ))

        return signals

    def combine_signals(
        self,
        signals: List[AlphaSignal]
    ) -> Dict[str, float]:
        """
        Combine multiple alpha signals into a unified trading signal.

        Uses weighted average with confidence adjustment.
        """
        if not signals:
            return {'combined_signal': 0, 'confidence': 0}

        # Group signals by type
        signal_by_type = {}
        for sig in signals:
            if sig.signal_type not in signal_by_type:
                signal_by_type[sig.signal_type] = []
            signal_by_type[sig.signal_type].append(sig)

        # Calculate weighted combined signal
        weighted_sum = 0
        weight_total = 0

        for sig_type, sigs in signal_by_type.items():
            base_weight = self.signal_weights.get(sig_type, 0.1)

            for sig in sigs:
                adjusted_weight = base_weight * sig.confidence
                weighted_sum += sig.value * adjusted_weight
                weight_total += adjusted_weight

        combined_signal = weighted_sum / weight_total if weight_total > 0 else 0
        combined_confidence = weight_total / len(signals)

        return {
            'combined_signal': np.clip(combined_signal, -1, 1),
            'confidence': min(combined_confidence, 1),
            'num_signals': len(signals),
            'signal_breakdown': {
                sig_type: np.mean([s.value for s in sigs])
                for sig_type, sigs in signal_by_type.items()
            }
        }


def demo_multi_source_mining():
    """Demonstrate multi-source alpha mining."""
    miner = MultiSourceAlphaMiner()

    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    price_data = pd.DataFrame({
        'open': 45000 + np.random.randn(100).cumsum() * 100,
        'high': 45200 + np.random.randn(100).cumsum() * 100,
        'low': 44800 + np.random.randn(100).cumsum() * 100,
        'close': 45000 + np.random.randn(100).cumsum() * 100,
        'volume': np.random.exponential(1e9, 100) * (1 + 0.5 * np.random.randn(100))
    }, index=dates)

    # Sample news items
    news_items = [
        {'title': 'Bitcoin Surges Past $45,000 as ETF Speculation Grows', 'content': 'Bullish momentum continues...'},
        {'title': 'Analysts Upgrade Crypto Outlook for 2024', 'content': 'Several firms raised price targets...'},
        {'title': 'Exchange Sees Record Volume', 'content': 'Trading activity surges as rally continues...'}
    ]

    # Sample on-chain data
    onchain_data = {
        'exchange_netflow': -5000,  # BTC leaving exchanges (bullish)
        'whale_balance_change': 2500  # Whales accumulating
    }

    # Mine signals from all sources
    all_signals = []

    price_signals = miner.mine_price_alpha(price_data, 'BTCUSDT')
    all_signals.extend(price_signals)
    print(f"Price signals: {len(price_signals)}")

    sentiment_signals = miner.mine_sentiment_alpha(news_items, 'BTCUSDT')
    all_signals.extend(sentiment_signals)
    print(f"Sentiment signals: {len(sentiment_signals)}")

    onchain_signals = miner.mine_onchain_alpha(onchain_data, 'BTCUSDT')
    all_signals.extend(onchain_signals)
    print(f"On-chain signals: {len(onchain_signals)}")

    # Combine all signals
    combined = miner.combine_signals(all_signals)

    print("\n=== Combined Alpha Signal ===")
    print(f"Signal Value: {combined['combined_signal']:.4f}")
    print(f"Confidence: {combined['confidence']:.4f}")
    print(f"Signal Breakdown: {combined['signal_breakdown']}")

    # Trading decision
    signal = combined['combined_signal']
    confidence = combined['confidence']

    if signal > 0.3 and confidence > 0.5:
        decision = "LONG"
    elif signal < -0.3 and confidence > 0.5:
        decision = "SHORT"
    else:
        decision = "HOLD"

    print(f"\nTrading Decision: {decision}")


if __name__ == "__main__":
    demo_multi_source_mining()
```

### 03: Self-Improving Alpha Agent

```python
# python/03_quantagent.py

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json

@dataclass
class KnowledgeEntry:
    """Entry in the agent's knowledge base."""
    factor_expression: str
    performance_metrics: Dict
    market_context: str
    success: bool
    timestamp: datetime
    iterations: int = 1

@dataclass
class FactorCandidate:
    """A candidate alpha factor."""
    name: str
    expression: str
    code: Callable
    rationale: str

class QuantAgent:
    """
    Self-improving alpha mining agent inspired by QuantAgent paper.

    Implements a two-loop architecture:
    - Inner Loop: Writer generates factors, Judge evaluates
    - Outer Loop: Real-world backtesting updates knowledge base

    Reference: Wang et al. (2024) "QuantAgent: Seeking Holy Grail in
    Trading by Self-Improving Large Language Model"
    """

    def __init__(
        self,
        max_inner_iterations: int = 5,
        performance_threshold: float = 0.5
    ):
        self.knowledge_base: List[KnowledgeEntry] = []
        self.max_inner_iterations = max_inner_iterations
        self.performance_threshold = performance_threshold
        self.generation_count = 0

    def generate_factor(
        self,
        market_context: str,
        asset_type: str = "crypto"
    ) -> FactorCandidate:
        """
        Generate a factor using the Writer agent.

        In production, this would use an LLM. Here we demonstrate
        the mechanism with rule-based generation.
        """
        self.generation_count += 1

        # Retrieve relevant knowledge
        relevant_knowledge = self._retrieve_knowledge(market_context)

        # Generate factor based on knowledge and context
        factor = self._writer_generate(market_context, relevant_knowledge)

        return factor

    def _retrieve_knowledge(self, context: str) -> List[KnowledgeEntry]:
        """Retrieve relevant entries from knowledge base."""
        # In production: use embedding similarity
        # Here: simple keyword matching
        relevant = []
        for entry in self.knowledge_base:
            if entry.success and entry.performance_metrics.get('sharpe', 0) > 0.5:
                relevant.append(entry)
        return sorted(relevant, key=lambda x: x.performance_metrics.get('sharpe', 0), reverse=True)[:5]

    def _writer_generate(
        self,
        context: str,
        knowledge: List[KnowledgeEntry]
    ) -> FactorCandidate:
        """
        Writer agent generates factor based on context and knowledge.

        In production, this would prompt an LLM. Here we use
        predefined factor templates enhanced by knowledge.
        """
        # Base factor templates
        templates = [
            {
                'name': 'momentum_volume',
                'expression': 'momentum_5d * log(volume_ratio + 1)',
                'rationale': 'Momentum confirmed by volume'
            },
            {
                'name': 'mean_reversion_volatility',
                'expression': '-zscore_20d / realized_vol_10d',
                'rationale': 'Mean reversion scaled by volatility regime'
            },
            {
                'name': 'breakout_filter',
                'expression': 'sign(close - high_20d) * volume_zscore',
                'rationale': 'Breakout signals filtered by volume'
            },
            {
                'name': 'rsi_momentum',
                'expression': '(rsi_14 - 50) / 50 * momentum_5d',
                'rationale': 'RSI-weighted momentum'
            },
            {
                'name': 'volatility_adjusted_return',
                'expression': 'return_5d / realized_vol_5d',
                'rationale': 'Risk-adjusted returns'
            }
        ]

        # Select template based on generation count and knowledge
        if knowledge and self.generation_count > 1:
            # Build on successful patterns
            best_entry = knowledge[0]
            template_idx = hash(best_entry.factor_expression) % len(templates)
        else:
            template_idx = self.generation_count % len(templates)

        template = templates[template_idx]

        # Create factor code
        def factor_code(df):
            # Implement the factor logic
            if 'momentum' in template['expression']:
                momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            else:
                momentum_5d = 0

            if 'volume_ratio' in template['expression']:
                vol_ma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'] / vol_ma
            else:
                volume_ratio = 1

            if 'zscore' in template['expression']:
                zscore_20d = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            else:
                zscore_20d = 0

            if 'realized_vol' in template['expression']:
                realized_vol = df['close'].pct_change().rolling(10).std() * np.sqrt(252)
            else:
                realized_vol = 1

            if 'rsi' in template['expression']:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_14 = 100 - (100 / (1 + rs))
            else:
                rsi_14 = 50

            # Calculate factor based on template
            if template['name'] == 'momentum_volume':
                return momentum_5d * np.log(volume_ratio + 1)
            elif template['name'] == 'mean_reversion_volatility':
                return -zscore_20d / (realized_vol + 0.01)
            elif template['name'] == 'breakout_filter':
                high_20d = df['high'].rolling(20).max()
                volume_zscore = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
                return np.sign(df['close'] - high_20d) * volume_zscore
            elif template['name'] == 'rsi_momentum':
                return (rsi_14 - 50) / 50 * momentum_5d
            else:
                return_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
                vol_5d = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
                return return_5d / (vol_5d + 0.01)

        return FactorCandidate(
            name=template['name'],
            expression=template['expression'],
            code=factor_code,
            rationale=template['rationale']
        )

    def judge_factor(
        self,
        factor: FactorCandidate,
        metrics: Dict
    ) -> Dict:
        """
        Judge agent evaluates factor and provides feedback.

        Returns score and improvement suggestions.
        """
        score = 0
        feedback = []

        # Evaluate based on metrics
        sharpe = metrics.get('sharpe', 0)
        ic = metrics.get('ic', 0)
        turnover = metrics.get('turnover', 0)
        max_dd = metrics.get('max_drawdown', -1)

        # Sharpe ratio scoring
        if sharpe > 1.5:
            score += 3
            feedback.append("Excellent risk-adjusted returns")
        elif sharpe > 1.0:
            score += 2
            feedback.append("Good risk-adjusted returns")
        elif sharpe > 0.5:
            score += 1
            feedback.append("Acceptable Sharpe, consider volatility scaling")
        else:
            feedback.append("Low Sharpe ratio, factor may be too noisy")

        # IC scoring
        if abs(ic) > 0.05:
            score += 2
            feedback.append("Strong predictive power")
        elif abs(ic) > 0.02:
            score += 1
            feedback.append("Moderate IC, may benefit from signal conditioning")
        else:
            feedback.append("Low IC, consider alternative formulations")

        # Turnover scoring
        if turnover < 0.3:
            score += 1
            feedback.append("Reasonable turnover for most strategies")
        elif turnover > 0.7:
            feedback.append("High turnover may erode returns due to costs")

        # Drawdown scoring
        if max_dd > -0.15:
            score += 1
            feedback.append("Acceptable drawdown profile")
        else:
            feedback.append("Large drawdown, consider adding risk limits")

        return {
            'score': score,
            'max_score': 7,
            'feedback': feedback,
            'pass': score >= 4
        }

    def backtest_factor(
        self,
        factor: FactorCandidate,
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Backtest factor against historical data.

        Returns performance metrics.
        """
        # Calculate factor values
        factor_values = factor.code(price_data)

        # Calculate forward returns
        forward_returns = price_data['close'].pct_change().shift(-1)

        # Remove NaN values
        valid_idx = ~(factor_values.isna() | forward_returns.isna())
        factor_clean = factor_values[valid_idx]
        returns_clean = forward_returns[valid_idx]

        if len(factor_clean) < 20:
            return {'error': 'Insufficient data'}

        # Calculate IC (Information Coefficient)
        ic = factor_clean.corr(returns_clean)

        # Calculate strategy returns (go long top quintile, short bottom quintile)
        factor_rank = factor_clean.rank(pct=True)
        strategy_position = np.where(factor_rank > 0.8, 1, np.where(factor_rank < 0.2, -1, 0))
        strategy_returns = pd.Series(strategy_position, index=factor_clean.index) * returns_clean

        # Calculate metrics
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # Calculate max drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calculate turnover
        position_changes = np.abs(np.diff(strategy_position))
        turnover = position_changes.mean() if len(position_changes) > 0 else 0

        return {
            'ic': ic,
            'sharpe': sharpe,
            'total_return': (1 + strategy_returns).prod() - 1,
            'volatility': strategy_returns.std() * np.sqrt(252),
            'max_drawdown': max_drawdown,
            'turnover': turnover,
            'num_trades': int(position_changes.sum())
        }

    def run_improvement_loop(
        self,
        market_context: str,
        price_data: pd.DataFrame,
        max_outer_iterations: int = 10
    ) -> FactorCandidate:
        """
        Run the full self-improvement loop.

        Outer loop: Generate, backtest, update knowledge
        Inner loop: Generate, judge, refine
        """
        best_factor = None
        best_metrics = {'sharpe': -np.inf}

        for outer_iter in range(max_outer_iterations):
            print(f"\n=== Outer Iteration {outer_iter + 1} ===")

            # Inner loop: generate and refine
            for inner_iter in range(self.max_inner_iterations):
                # Generate factor
                factor = self.generate_factor(market_context)

                # Quick evaluation
                metrics = self.backtest_factor(factor, price_data)

                if 'error' in metrics:
                    continue

                # Judge evaluation
                judgment = self.judge_factor(factor, metrics)

                print(f"  Inner {inner_iter + 1}: {factor.name}, Sharpe={metrics['sharpe']:.3f}, Score={judgment['score']}")

                if judgment['pass']:
                    break

            # Full backtest on best factor from inner loop
            metrics = self.backtest_factor(factor, price_data)

            if 'error' not in metrics:
                # Update knowledge base
                entry = KnowledgeEntry(
                    factor_expression=factor.expression,
                    performance_metrics=metrics,
                    market_context=market_context,
                    success=metrics['sharpe'] > self.performance_threshold,
                    timestamp=datetime.now()
                )
                self.knowledge_base.append(entry)

                # Track best
                if metrics['sharpe'] > best_metrics['sharpe']:
                    best_factor = factor
                    best_metrics = metrics
                    print(f"  New best! Sharpe={metrics['sharpe']:.3f}")

        return best_factor, best_metrics


def demo_quantagent():
    """Demonstrate QuantAgent self-improvement."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='H')

    # Generate synthetic price data with some structure
    trend = np.linspace(0, 10, 500)
    noise = np.random.randn(500).cumsum() * 0.5
    seasonal = 3 * np.sin(np.linspace(0, 20 * np.pi, 500))

    close_prices = 45000 + trend * 100 + noise * 100 + seasonal * 100

    price_data = pd.DataFrame({
        'open': close_prices + np.random.randn(500) * 50,
        'high': close_prices + np.abs(np.random.randn(500)) * 100,
        'low': close_prices - np.abs(np.random.randn(500)) * 100,
        'close': close_prices,
        'volume': np.random.exponential(1e9, 500) * (1 + 0.3 * np.random.randn(500))
    }, index=dates)

    # Initialize agent
    agent = QuantAgent(
        max_inner_iterations=3,
        performance_threshold=0.3
    )

    # Run improvement loop
    best_factor, best_metrics = agent.run_improvement_loop(
        market_context="crypto_trending_market",
        price_data=price_data,
        max_outer_iterations=5
    )

    print("\n=== Best Factor Found ===")
    print(f"Name: {best_factor.name}")
    print(f"Expression: {best_factor.expression}")
    print(f"Rationale: {best_factor.rationale}")
    print(f"\nPerformance Metrics:")
    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nKnowledge Base Size: {len(agent.knowledge_base)} entries")
    successful = sum(1 for e in agent.knowledge_base if e.success)
    print(f"Successful Factors: {successful}")


if __name__ == "__main__":
    demo_quantagent()
```

### 04: Backtesting LLM Alphas

```python
# python/04_backtest_alphas.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from datetime import datetime

@dataclass
class BacktestConfig:
    """Configuration for alpha backtesting."""
    initial_capital: float = 100000
    max_position_pct: float = 0.2
    transaction_cost_bps: float = 10
    slippage_bps: float = 5
    rebalance_frequency: str = "daily"
    long_short: bool = True  # Long-short or long-only

@dataclass
class AlphaBacktestResult:
    """Results from backtesting an alpha factor."""
    returns: pd.Series
    positions: pd.DataFrame
    factor_values: pd.Series
    metrics: Dict[str, float]
    trades: List[Dict]

class AlphaBacktester:
    """
    Backtest alpha factors generated by LLMs.

    Supports both long-short and long-only strategies,
    with realistic transaction costs and slippage.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def backtest(
        self,
        factor_func: Callable,
        price_data: pd.DataFrame,
        universe: Optional[List[str]] = None
    ) -> AlphaBacktestResult:
        """
        Run backtest on a factor.

        Args:
            factor_func: Function that takes price_data and returns factor values
            price_data: OHLCV DataFrame (single asset) or Dict of DataFrames
            universe: List of symbols (if multi-asset)

        Returns:
            AlphaBacktestResult with performance metrics
        """
        # Handle single asset case
        if isinstance(price_data, pd.DataFrame):
            return self._backtest_single_asset(factor_func, price_data)
        else:
            return self._backtest_multi_asset(factor_func, price_data, universe)

    def _backtest_single_asset(
        self,
        factor_func: Callable,
        df: pd.DataFrame
    ) -> AlphaBacktestResult:
        """Backtest factor on single asset."""
        # Calculate factor values
        factor_values = factor_func(df)

        # Normalize factor to [-1, 1]
        factor_normalized = factor_values / (factor_values.abs().rolling(20).max() + 1e-10)
        factor_normalized = factor_normalized.clip(-1, 1)

        # Generate positions based on factor
        if self.config.long_short:
            # Long-short: position = factor value
            raw_position = factor_normalized
        else:
            # Long-only: position = max(0, factor)
            raw_position = factor_normalized.clip(lower=0)

        # Scale to max position size
        position = raw_position * self.config.max_position_pct

        # Calculate returns
        price_returns = df['close'].pct_change()

        # Account for transaction costs
        position_change = position.diff().abs()
        transaction_costs = position_change * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000

        # Strategy returns
        strategy_returns = position.shift(1) * price_returns - transaction_costs
        strategy_returns = strategy_returns.dropna()

        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns, factor_values, price_returns)

        # Record trades
        trades = self._extract_trades(position, df['close'])

        return AlphaBacktestResult(
            returns=strategy_returns,
            positions=pd.DataFrame({'position': position}),
            factor_values=factor_values,
            metrics=metrics,
            trades=trades
        )

    def _backtest_multi_asset(
        self,
        factor_func: Callable,
        price_data: Dict[str, pd.DataFrame],
        universe: List[str]
    ) -> AlphaBacktestResult:
        """Backtest factor across multiple assets."""
        # Calculate factor for each asset
        factor_df = pd.DataFrame()
        returns_df = pd.DataFrame()

        for symbol in universe:
            if symbol in price_data:
                df = price_data[symbol]
                factor_df[symbol] = factor_func(df)
                returns_df[symbol] = df['close'].pct_change()

        # Cross-sectional ranking
        factor_rank = factor_df.rank(axis=1, pct=True)

        if self.config.long_short:
            # Long top quintile, short bottom quintile
            positions = pd.DataFrame(0.0, index=factor_rank.index, columns=factor_rank.columns)
            positions[factor_rank > 0.8] = 1
            positions[factor_rank < 0.2] = -1
            # Normalize to sum to 0 (market neutral)
            positions = positions.sub(positions.mean(axis=1), axis=0)
        else:
            # Long only: weight by rank
            positions = factor_rank

        # Scale positions
        positions = positions * self.config.max_position_pct / positions.abs().sum(axis=1).clip(lower=1)

        # Calculate returns
        position_change = positions.diff().abs()
        transaction_costs = position_change * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000

        strategy_returns = (positions.shift(1) * returns_df - transaction_costs).sum(axis=1)
        strategy_returns = strategy_returns.dropna()

        # Calculate metrics
        combined_factor = factor_df.mean(axis=1)
        combined_returns = returns_df.mean(axis=1)
        metrics = self._calculate_metrics(strategy_returns, combined_factor, combined_returns)

        return AlphaBacktestResult(
            returns=strategy_returns,
            positions=positions,
            factor_values=combined_factor,
            metrics=metrics,
            trades=[]
        )

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        factor_values: pd.Series,
        price_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(strategy_returns) < 20:
            return {'error': 'Insufficient data'}

        # Basic return metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Information coefficient
        aligned = pd.concat([factor_values, price_returns.shift(-1)], axis=1).dropna()
        if len(aligned) > 0:
            ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        else:
            ic = 0

        # Win rate
        win_rate = (strategy_returns > 0).mean()

        # Profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = -strategy_returns[strategy_returns < 0].sum()
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'ic': ic,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar,
            'num_periods': len(strategy_returns)
        }

    def _extract_trades(
        self,
        positions: pd.Series,
        prices: pd.Series
    ) -> List[Dict]:
        """Extract individual trades from position changes."""
        trades = []
        pos_diff = positions.diff()

        for ts, change in pos_diff.items():
            if abs(change) > 0.01:  # Significant position change
                trades.append({
                    'timestamp': ts,
                    'change': change,
                    'price': prices.loc[ts] if ts in prices.index else np.nan,
                    'type': 'BUY' if change > 0 else 'SELL'
                })

        return trades

    def compare_factors(
        self,
        factors: Dict[str, Callable],
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare multiple factors on the same data.

        Args:
            factors: Dict mapping factor name to factor function
            price_data: OHLCV DataFrame

        Returns:
            DataFrame comparing factor performance
        """
        results = []

        for name, factor_func in factors.items():
            result = self.backtest(factor_func, price_data)
            metrics = result.metrics
            metrics['factor_name'] = name
            results.append(metrics)

        return pd.DataFrame(results).set_index('factor_name')


def demo_backtest():
    """Demonstrate alpha backtesting."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='H')

    # Create price data with momentum characteristics
    returns = np.random.randn(500) * 0.01
    returns = returns + 0.3 * np.roll(returns, 1)  # Add momentum
    close_prices = 45000 * (1 + returns).cumprod()

    price_data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(500) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(500)) * 0.005),
        'low': close_prices * (1 - np.abs(np.random.randn(500)) * 0.005),
        'close': close_prices,
        'volume': np.random.exponential(1e9, 500)
    }, index=dates)

    # Define factors to test
    def momentum_factor(df):
        return (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

    def mean_reversion_factor(df):
        ma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        return -(df['close'] - ma) / std

    def volume_momentum_factor(df):
        price_mom = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
        return price_mom * np.log(vol_ratio + 1)

    def rsi_factor(df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # Normalize to [-1, 1]

    # Initialize backtester
    config = BacktestConfig(
        initial_capital=100000,
        max_position_pct=0.5,
        transaction_cost_bps=10,
        slippage_bps=5,
        long_short=True
    )
    backtester = AlphaBacktester(config)

    # Compare factors
    factors = {
        'momentum': momentum_factor,
        'mean_reversion': mean_reversion_factor,
        'volume_momentum': volume_momentum_factor,
        'rsi': rsi_factor
    }

    comparison = backtester.compare_factors(factors, price_data)

    print("=== Factor Comparison ===")
    print(comparison[['sharpe_ratio', 'total_return', 'max_drawdown', 'ic', 'win_rate']].round(4))

    # Detailed analysis of best factor
    best_factor_name = comparison['sharpe_ratio'].idxmax()
    print(f"\n=== Best Factor: {best_factor_name} ===")

    best_result = backtester.backtest(factors[best_factor_name], price_data)

    print("\nDetailed Metrics:")
    for key, value in best_result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nNumber of trades: {len(best_result.trades)}")

    # Show recent positions
    print("\nRecent Factor Values and Positions:")
    recent = pd.concat([
        best_result.factor_values.tail(10),
        best_result.positions['position'].tail(10)
    ], axis=1)
    recent.columns = ['factor', 'position']
    print(recent)


if __name__ == "__main__":
    demo_backtest()
```

## Rust Implementation

For production-grade alpha mining, we provide a Rust implementation with high performance and low latency.

```
rust_llm_alpha/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── alpha/              # Alpha generation
│   │   ├── mod.rs
│   │   ├── generator.rs    # Factor generation
│   │   ├── optimizer.rs    # Factor optimization
│   │   └── expressions.rs  # Factor expression DSL
│   ├── data/               # Data handling
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API client
│   │   ├── yahoo.rs        # Yahoo Finance client
│   │   └── types.rs        # Data structures
│   ├── backtest/           # Backtesting engine
│   │   ├── mod.rs
│   │   ├── engine.rs       # Core backtest logic
│   │   └── metrics.rs      # Performance metrics
│   └── agent/              # QuantAgent implementation
│       ├── mod.rs
│       ├── writer.rs       # Writer agent
│       ├── judge.rs        # Judge agent
│       └── knowledge.rs    # Knowledge base
└── examples/
    ├── generate_alpha.rs
    ├── backtest_factor.rs
    └── quantagent_demo.rs
```

See [rust_llm_alpha](rust_llm_alpha/) for complete Rust implementation.

### Quick Start (Rust)

```bash
cd rust_llm_alpha

# Run alpha generation example
cargo run --example generate_alpha

# Backtest a factor on Bybit data
cargo run --example backtest_factor -- --symbol BTCUSDT --days 30

# Run QuantAgent self-improvement loop
cargo run --example quantagent_demo
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── alpha_generator.py      # Factor generation
├── multi_source_alpha.py   # Multi-modal mining
├── quantagent.py           # Self-improving agent
├── backtest.py             # Backtesting engine
├── data_loader.py          # Data utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_basic_generation.py
    ├── 02_multi_source.py
    ├── 03_quantagent.py
    └── 04_full_backtest.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run basic alpha generation
python examples/01_basic_generation.py

# Multi-source alpha mining
python examples/02_multi_source.py --symbol BTCUSDT

# Run QuantAgent
python examples/03_quantagent.py

# Full backtest
python examples/04_full_backtest.py --capital 100000
```

## Best Practices

### Factor Generation Guidelines

1. **Specify Clear Constraints**
   ```python
   # Good prompt
   "Generate a momentum factor for BTC with 5-day lookback,
    incorporating volume confirmation, suitable for hourly rebalancing"

   # Poor prompt
   "Generate a trading factor"
   ```

2. **Iterative Refinement**
   ```python
   # Start simple, add complexity based on performance
   iteration_1: "momentum_5d"  # Simple momentum
   iteration_2: "momentum_5d * volume_ratio"  # Add volume
   iteration_3: "momentum_5d * volume_ratio / volatility"  # Risk adjust
   ```

3. **Validation Requirements**
   ```python
   # Always validate generated factors
   validation_checks = {
       "no_lookahead": check_lookahead_bias(factor),
       "reasonable_values": factor.abs().quantile(0.99) < 10,
       "sufficient_variation": factor.std() > 0.01,
       "low_correlation": abs(factor.corr(existing_factors)) < 0.7
   }
   ```

### Backtesting Best Practices

1. **Realistic Costs**
   - Include transaction costs (5-20 bps depending on asset)
   - Account for slippage (especially for crypto)
   - Consider funding rates for perpetual futures

2. **Out-of-Sample Testing**
   - Train on 70% of data
   - Validate on 15%
   - Test on final 15%

3. **Multiple Regime Testing**
   - Bull markets
   - Bear markets
   - High volatility periods
   - Low liquidity periods

### Common Pitfalls

1. **Overfitting to LLM Training Data** - LLMs may suggest factors similar to those in their training data
2. **Look-ahead Bias** - Ensure generated code doesn't use future information
3. **Survivorship Bias** - Test on delisted assets as well
4. **Cost Neglect** - High-turnover factors may look good before costs

## Resources

### Papers

- [QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model](https://arxiv.org/abs/2402.03755) — QuantAgent paper (2024)
- [Can Large Language Models Mine Interpretable Financial Factors?](https://aclanthology.org/2024.findings-acl.92/) — ACL Findings (2024)
- [Automate Strategy Finding with LLM in Quant Investment](https://arxiv.org/abs/2409.06289) — LLM strategy automation (2024)
- [Large Language Model Agent in Financial Trading: A Survey](https://arxiv.org/abs/2408.06361) — Comprehensive survey (2024)

### Open-Source Tools

| Tool | Description | Link |
|------|-------------|------|
| FinGPT | Open-source financial LLM | [GitHub](https://github.com/AI4Finance-Foundation/FinGPT) |
| Qlib | Quantitative research platform | [GitHub](https://github.com/microsoft/qlib) |
| Backtrader | Python backtesting library | [GitHub](https://github.com/mementum/backtrader) |
| Freqtrade | Crypto trading bot | [GitHub](https://github.com/freqtrade/freqtrade) |

### Related Chapters

- [Chapter 62: BloombergGPT Trading](../62_bloomberggpt_trading) — Financial LLM foundations
- [Chapter 64: Multi-Agent LLM Trading](../64_multi_agent_llm_trading) — Multi-agent systems
- [Chapter 65: RAG for Trading](../65_rag_for_trading) — Retrieval-augmented generation
- [Chapter 75: LLM Factor Discovery](../75_llm_factor_discovery) — Advanced factor mining

---

## Difficulty Level

**Advanced**

Prerequisites:
- Understanding of alpha factors and quantitative trading
- Experience with LLMs and prompt engineering
- Python/Rust programming experience
- Familiarity with backtesting concepts

## References

1. Wang, S., et al. (2024). "QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model." arXiv:2402.03755
2. Li, H., et al. (2024). "Can Large Language Models Mine Interpretable Financial Factors More Effectively?" ACL Findings 2024
3. Nie, Y., et al. (2024). "Large Language Model Agent in Financial Trading: A Survey." arXiv:2408.06361
4. Yang, H., et al. (2023). "FinGPT: Open-Source Financial Large Language Models." arXiv:2306.06031
