# Chapter 73: LLM Risk Assessment for Trading

## Overview

Large Language Models (LLMs) can analyze unstructured text data to assess various types of risks in financial markets. This chapter explores how to use LLMs for risk assessment in trading strategies, combining news analysis, sentiment extraction, and risk scoring to make informed trading decisions.

## Trading Strategy

**Core Concept:** LLMs process financial news, earnings reports, SEC filings, and social media to generate risk scores that inform position sizing and trading decisions.

**Entry Signals:**
- Long: Low risk score + positive sentiment momentum
- Short: High risk score + negative sentiment indicators
- Exit: Risk score crosses threshold or sentiment reversal

**Edge:** LLMs can process vast amounts of unstructured text faster than human analysts, identifying subtle risk signals in language patterns that correlate with future price movements.

## Technical Specification

### Key Components

1. **Text Data Pipeline** - Collect news, filings, social media
2. **LLM Risk Scoring** - Generate risk assessments from text
3. **Signal Generation** - Convert scores to trading signals
4. **Position Management** - Risk-adjusted position sizing
5. **Backtesting Framework** - Validate strategy performance

### Architecture

```
                    ┌─────────────────────┐
                    │   Data Sources      │
                    │  (News, Filings)    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Processor    │
                    │  (Cleaning, Chunk)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Risk Engine   │
                    │  (Prompt + Model)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Aggregator   │
                    │  (Score + History)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Trading Engine    │
                    │  (Signals + Orders) │
                    └─────────────────────┘
```

### Data Requirements

```
News Sources:
├── Financial news APIs (Alpha Vantage, NewsAPI)
├── SEC EDGAR filings (10-K, 10-Q, 8-K)
├── Earnings call transcripts
├── Social media (Twitter/X, Reddit, StockTwits)
└── Cryptocurrency news (CoinDesk, CryptoNews)

Market Data:
├── OHLCV price data (Bybit for crypto, Yahoo for stocks)
├── Order book snapshots
├── Trading volume analytics
└── Volatility metrics
```

### Risk Categories

The LLM evaluates multiple risk dimensions:

| Risk Type | Description | Indicators |
|-----------|-------------|------------|
| **Market Risk** | Exposure to market movements | Volatility mentions, macro concerns |
| **Credit Risk** | Counterparty default probability | Debt levels, rating changes |
| **Liquidity Risk** | Ability to trade without impact | Volume concerns, spread widening |
| **Operational Risk** | System/process failures | Tech issues, management changes |
| **Regulatory Risk** | Legal/compliance exposure | Lawsuits, regulatory actions |
| **Sentiment Risk** | Market perception shifts | Social media tone, analyst coverage |

### Prompt Engineering

```python
RISK_ASSESSMENT_PROMPT = """
Analyze the following financial text and provide a risk assessment.

Text: {text}

Evaluate the following risk dimensions on a scale of 1-10 (1=lowest risk, 10=highest risk):

1. Market Risk: Exposure to market volatility and systematic risk
2. Credit Risk: Counterparty or default risk indicators
3. Liquidity Risk: Trading and market depth concerns
4. Operational Risk: Business execution and management risks
5. Regulatory Risk: Legal and compliance exposure
6. Sentiment Risk: Market perception and reputation

For each dimension, provide:
- Score (1-10)
- Key factors identified
- Confidence level (low/medium/high)

Also provide:
- Overall Risk Score (weighted average)
- Risk Direction (increasing/stable/decreasing)
- Time Horizon (short/medium/long term)

Output as JSON format.
"""
```

### Key Metrics

**Risk Assessment Quality:**
- Prediction accuracy (risk score vs realized volatility)
- Information Coefficient (IC) with future returns
- False positive/negative rates for risk events

**Strategy Performance:**
- Sharpe Ratio
- Maximum Drawdown
- Risk-adjusted returns
- Hit rate on risk predictions

### Dependencies

```python
# Python dependencies
openai>=1.0.0           # OpenAI API client
anthropic>=0.5.0        # Claude API client
transformers>=4.30.0    # HuggingFace models
torch>=2.0.0            # PyTorch
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
yfinance>=0.2.0         # Stock data
requests>=2.28.0        # HTTP client
beautifulsoup4>=4.12.0  # HTML parsing
```

```rust
// Rust dependencies
reqwest = "0.12"        // HTTP client
serde = "1.0"           // Serialization
tokio = "1.0"           // Async runtime
ndarray = "0.16"        // Arrays
polars = "0.46"         // DataFrames
```

## Python Implementation

### Basic Risk Assessment

```python
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import openai

class RiskLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3
    SEVERE = 4

@dataclass
class RiskScore:
    market_risk: float
    credit_risk: float
    liquidity_risk: float
    operational_risk: float
    regulatory_risk: float
    sentiment_risk: float
    overall_score: float
    confidence: str
    direction: str

    @property
    def risk_level(self) -> RiskLevel:
        if self.overall_score <= 3:
            return RiskLevel.LOW
        elif self.overall_score <= 5:
            return RiskLevel.MODERATE
        elif self.overall_score <= 7:
            return RiskLevel.HIGH
        return RiskLevel.SEVERE

class LLMRiskAssessor:
    """LLM-based risk assessment engine."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def assess_risk(self, text: str) -> RiskScore:
        """Analyze text and return risk scores."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial risk analyst."},
                {"role": "user", "content": self._build_prompt(text)}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_response(result)

    def _build_prompt(self, text: str) -> str:
        return f"""Analyze this financial text for risk:

{text}

Return JSON with scores (1-10) for: market_risk, credit_risk,
liquidity_risk, operational_risk, regulatory_risk, sentiment_risk,
overall_score, confidence (low/medium/high), direction (increasing/stable/decreasing)"""

    def _parse_response(self, data: dict) -> RiskScore:
        return RiskScore(
            market_risk=float(data.get("market_risk", 5)),
            credit_risk=float(data.get("credit_risk", 5)),
            liquidity_risk=float(data.get("liquidity_risk", 5)),
            operational_risk=float(data.get("operational_risk", 5)),
            regulatory_risk=float(data.get("regulatory_risk", 5)),
            sentiment_risk=float(data.get("sentiment_risk", 5)),
            overall_score=float(data.get("overall_score", 5)),
            confidence=data.get("confidence", "medium"),
            direction=data.get("direction", "stable")
        )
```

### Trading Strategy Integration

```python
import pandas as pd
import numpy as np
from typing import Tuple

class RiskBasedTrader:
    """Trading strategy based on LLM risk assessment."""

    def __init__(
        self,
        risk_assessor: LLMRiskAssessor,
        risk_threshold_long: float = 4.0,
        risk_threshold_short: float = 7.0,
        max_position_size: float = 1.0
    ):
        self.assessor = risk_assessor
        self.risk_threshold_long = risk_threshold_long
        self.risk_threshold_short = risk_threshold_short
        self.max_position_size = max_position_size
        self.risk_history: List[RiskScore] = []

    def generate_signal(self, text: str, current_price: float) -> Tuple[str, float]:
        """Generate trading signal from text analysis.

        Returns:
            Tuple of (signal, position_size)
            signal: 'long', 'short', or 'neutral'
        """
        risk_score = self.assessor.assess_risk(text)
        self.risk_history.append(risk_score)

        # Calculate position size inversely proportional to risk
        position_size = self._calculate_position_size(risk_score)

        # Generate signal based on risk level and direction
        if risk_score.overall_score <= self.risk_threshold_long:
            if risk_score.direction in ['stable', 'decreasing']:
                return ('long', position_size)

        elif risk_score.overall_score >= self.risk_threshold_short:
            if risk_score.direction in ['stable', 'increasing']:
                return ('short', position_size * 0.5)  # Smaller short positions

        return ('neutral', 0.0)

    def _calculate_position_size(self, risk: RiskScore) -> float:
        """Calculate position size based on risk score."""
        # Lower risk = larger position (inverse relationship)
        risk_factor = 1 - (risk.overall_score / 10)

        # Adjust by confidence
        confidence_multiplier = {
            'high': 1.0,
            'medium': 0.75,
            'low': 0.5
        }.get(risk.confidence, 0.5)

        return self.max_position_size * risk_factor * confidence_multiplier

    def get_risk_trend(self, window: int = 5) -> str:
        """Analyze recent risk trend."""
        if len(self.risk_history) < window:
            return 'insufficient_data'

        recent_scores = [r.overall_score for r in self.risk_history[-window:]]
        slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        return 'stable'
```

## Rust Implementation

See the `rust_llm_risk/` directory for the complete Rust implementation, which includes:

- **Data fetching** from Bybit and news APIs
- **Text processing** and chunking
- **LLM API integration** (OpenAI compatible)
- **Risk scoring** engine
- **Backtesting** framework
- **Trading strategy** implementation

### Quick Start (Rust)

```bash
cd rust_llm_risk

# Build the project
cargo build --release

# Fetch market data
cargo run --example fetch_data

# Run risk assessment
cargo run --example assess_risk -- --symbol BTCUSDT --days 30

# Backtest the strategy
cargo run --example backtest -- --start 2024-01-01 --end 2024-06-01
```

## Expected Outcomes

1. **Risk Assessment Pipeline** - End-to-end system for LLM-based risk scoring
2. **Trading Signals** - Risk-adjusted entry/exit signals
3. **Position Sizing** - Dynamic sizing based on risk levels
4. **Performance Metrics** - Sharpe ratio improvement vs baseline
5. **Real-time Monitoring** - Dashboard for risk tracking

## Use Cases

### Cryptocurrency Trading
- Monitor social media for sentiment shifts
- Analyze exchange announcements for risks
- Track regulatory news across jurisdictions

### Stock Trading
- Earnings call analysis for company-specific risks
- SEC filing parsing for hidden risk factors
- News sentiment aggregation for sector risks

### Options Trading
- Volatility event prediction from news
- Risk scoring for earnings plays
- Merger/acquisition risk assessment

## Best Practices

1. **Prompt Engineering** - Test and refine prompts for consistent scoring
2. **Model Selection** - Use appropriate model for task complexity
3. **Rate Limiting** - Implement caching to manage API costs
4. **Validation** - Backtest extensively before live trading
5. **Human Oversight** - Review high-risk assessments manually

## References

- [Risk Assessment with Large Language Models](https://arxiv.org/abs/2310.01926)
- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031)
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
- [Sentiment Analysis in Financial Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4135500)

## Difficulty Level

Expert

Required knowledge: LLM prompting, NLP, financial risk management, trading systems, API integration
