# Chapter 61: FinGPT — Open-Source Financial Large Language Model

This chapter explores **FinGPT**, an open-source Large Language Model specifically designed for the financial domain. Unlike proprietary models like BloombergGPT, FinGPT provides accessible, customizable solutions for financial NLP tasks including sentiment analysis, news understanding, and trading signal generation.

<p align="center">
<img src="https://i.imgur.com/QVZ8kXp.png" width="70%">
</p>

## Contents

1. [Introduction to FinGPT](#introduction-to-fingpt)
    * [Why Open-Source Financial LLMs?](#why-open-source-financial-llms)
    * [FinGPT Architecture Overview](#fingpt-architecture-overview)
    * [Key Features](#key-features)
2. [FinGPT Training Approach](#fingpt-training-approach)
    * [Data-Centric Architecture](#data-centric-architecture)
    * [Reinforcement Learning from Stock Prices (RLSP)](#reinforcement-learning-from-stock-prices-rlsp)
    * [Low-Rank Adaptation (LoRA) Fine-tuning](#low-rank-adaptation-lora-fine-tuning)
3. [Trading Applications](#trading-applications)
    * [Sentiment Analysis](#sentiment-analysis)
    * [News-Based Trading Signals](#news-based-trading-signals)
    * [Robo-Advisor Integration](#robo-advisor-integration)
    * [Cryptocurrency Analysis](#cryptocurrency-analysis)
4. [Practical Examples](#practical-examples)
    * [01: Financial Sentiment Analysis](#01-financial-sentiment-analysis)
    * [02: Trading Signal Generation](#02-trading-signal-generation)
    * [03: Crypto Market Analysis with Bybit](#03-crypto-market-analysis-with-bybit)
    * [04: Backtesting FinGPT Signals](#04-backtesting-fingpt-signals)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Comparison with Other Models](#comparison-with-other-models)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to FinGPT

FinGPT is an open-source initiative by the AI4Finance Foundation that aims to democratize financial large language models. While proprietary models like BloombergGPT require significant resources and are not publicly available, FinGPT provides a framework for building and fine-tuning financial LLMs using publicly available data and open-source base models.

### Why Open-Source Financial LLMs?

```
CHALLENGES WITH PROPRIETARY FINANCIAL LLMs:
┌──────────────────────────────────────────────────────────────────┐
│  1. ACCESSIBILITY                                                 │
│     BloombergGPT: Not publicly available                          │
│     FinGPT: Open-source, anyone can use and modify                │
├──────────────────────────────────────────────────────────────────┤
│  2. CUSTOMIZABILITY                                               │
│     BloombergGPT: Fixed model, no fine-tuning                     │
│     FinGPT: Fine-tune for your specific use case                  │
├──────────────────────────────────────────────────────────────────┤
│  3. DATA FRESHNESS                                                │
│     Proprietary: Training data may be outdated                    │
│     FinGPT: Continuous updates with real-time data pipelines      │
├──────────────────────────────────────────────────────────────────┤
│  4. COST                                                          │
│     Proprietary: Expensive API costs, licensing fees              │
│     FinGPT: Free to use, self-host, and modify                    │
├──────────────────────────────────────────────────────────────────┤
│  5. TRANSPARENCY                                                  │
│     Proprietary: Black box, unknown biases                        │
│     FinGPT: Full visibility into training data and methods        │
└──────────────────────────────────────────────────────────────────┘
```

### FinGPT Architecture Overview

FinGPT uses a modular, data-centric approach:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FinGPT ARCHITECTURE                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        DATA LAYER                                        │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │ │
│  │  │ News APIs    │ │ SEC Filings  │ │ Social Media │ │ Market Data  │    │ │
│  │  │ (Reuters,    │ │ (EDGAR,      │ │ (Twitter,    │ │ (Yahoo,      │    │ │
│  │  │  Finnhub)    │ │  10-K/Q)     │ │  Reddit)     │ │  Binance)    │    │ │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │ │
│  │         │                │                │                │             │ │
│  │         └────────────────┴────────────────┴────────────────┘             │ │
│  │                                    │                                      │ │
│  │                                    ▼                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐    │ │
│  │  │              DATA ENGINEERING & PREPROCESSING                     │    │ │
│  │  │   • Clean & normalize financial text                              │    │ │
│  │  │   • Extract structured data (entities, numbers, dates)            │    │ │
│  │  │   • Create instruction-following datasets                         │    │ │
│  │  └──────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        MODEL LAYER                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   BASE MODEL OPTIONS                              │   │ │
│  │  │   LLaMA-2 (7B/13B/70B) | Falcon | MPT | Mistral | ChatGLM        │   │ │
│  │  └────────────────────────────────┬─────────────────────────────────┘   │ │
│  │                                   │                                      │ │
│  │                                   ▼                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │ │
│  │  │              FINE-TUNING METHODS                                  │   │ │
│  │  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │   │ │
│  │  │   │   LoRA      │   │    QLoRA    │   │ Full Fine-tuning    │   │   │ │
│  │  │   │ (Low-Rank   │   │ (Quantized  │   │ (Resource-          │   │   │ │
│  │  │   │  Adapters)  │   │   LoRA)     │   │  intensive)         │   │   │ │
│  │  │   └─────────────┘   └─────────────┘   └─────────────────────┘   │   │ │
│  │  └──────────────────────────────────────────────────────────────────┘   │ │
│  │                                   │                                      │ │
│  │                                   ▼                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │ │
│  │  │              REINFORCEMENT LEARNING                               │   │ │
│  │  │   RLSP (RL from Stock Prices) - Align predictions with returns    │   │ │
│  │  └──────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     APPLICATION LAYER                                    │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │ │
│  │  │ Sentiment  │  │  Trading   │  │   Robo-    │  │ Risk Assessment   │ │ │
│  │  │ Analysis   │  │  Signals   │  │  Advisor   │  │ & Compliance      │ │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Data-Centric Design**
   - Automated data pipelines for real-time financial data
   - Support for multiple data sources (news, filings, social media)
   - Built-in data cleaning and preprocessing

2. **Efficient Fine-tuning**
   - LoRA and QLoRA for parameter-efficient training
   - Train on consumer GPUs (RTX 3090, 4090)
   - Quick adaptation to new domains

3. **Novel Training Methods**
   - RLSP (Reinforcement Learning from Stock Prices)
   - Aligns model outputs with actual market returns
   - Better trading signal generation

4. **Multiple Base Models**
   - LLaMA-2, Falcon, MPT, Mistral support
   - Choose based on your hardware and requirements
   - Easy to swap and experiment

## FinGPT Training Approach

### Data-Centric Architecture

FinGPT emphasizes data quality and freshness over model size:

```python
# FinGPT data pipeline concept
class FinGPTDataPipeline:
    """
    Real-time data ingestion and processing for FinGPT.
    """

    def __init__(self):
        self.sources = {
            "news": ["reuters", "finnhub", "alpaca_news"],
            "filings": ["sec_edgar"],
            "social": ["twitter", "reddit_wallstreetbets"],
            "market": ["yahoo_finance", "binance", "bybit"]
        }

    def collect_data(self, source_type: str, symbols: list):
        """Collect data from specified sources."""
        pass

    def preprocess(self, raw_data):
        """
        Preprocess steps:
        1. Remove duplicates
        2. Clean HTML/special characters
        3. Normalize dates and numbers
        4. Extract entities
        5. Create instruction-following format
        """
        pass

    def create_training_samples(self, data, task_type: str):
        """
        Create instruction-following samples.

        Example format:
        {
            "instruction": "Analyze the sentiment of this news headline.",
            "input": "Apple reports record Q4 earnings beating estimates",
            "output": "POSITIVE. The headline indicates strong financial..."
        }
        """
        pass
```

### Reinforcement Learning from Stock Prices (RLSP)

A key innovation in FinGPT is RLSP, which aligns model predictions with actual market outcomes:

```
RLSP: REINFORCEMENT LEARNING FROM STOCK PRICES
═══════════════════════════════════════════════════════════════════════════════

TRADITIONAL APPROACH (Sentiment Labels):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  News Text ──────► LLM ──────► Sentiment ──────► Trading Signal             │
│                     │           (Human labels)                               │
│                     │                                                        │
│  Problem: Human labels may not correlate with actual price movements!       │
│                                                                              │
│  "Apple maintains guidance" → Human: NEUTRAL → Stock: UP 5%                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

RLSP APPROACH (Stock Price Feedback):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  News Text ──────► LLM ──────► Prediction ──────► Compare with              │
│                     │                              Actual Returns            │
│                     │                                   │                    │
│                     │◄─────────────── Reward/Loss ◄─────┘                    │
│                                                                              │
│  Model learns what news ACTUALLY moves prices, not human interpretation!     │
│                                                                              │
│  "Apple maintains guidance" → RLSP: +0.7 signal → Stock: UP 5% ✓            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

RLSP Implementation concept:

```python
def compute_rlsp_reward(
    prediction: float,      # Model's predicted direction (-1 to 1)
    actual_return: float,   # Actual stock return over horizon
    confidence: float       # Model's confidence
) -> float:
    """
    Compute reward for RLSP training.

    Args:
        prediction: Model's directional prediction
        actual_return: Actual market return
        confidence: Model's confidence score

    Returns:
        Reward signal for RL training
    """
    # Reward correct direction predictions
    direction_match = (prediction * actual_return) > 0

    if direction_match:
        # Positive reward scaled by confidence and magnitude
        reward = abs(actual_return) * confidence
    else:
        # Penalty for wrong direction
        reward = -abs(actual_return) * confidence

    return reward
```

### Low-Rank Adaptation (LoRA) Fine-tuning

FinGPT uses LoRA for efficient fine-tuning:

```
LoRA: PARAMETER-EFFICIENT FINE-TUNING
═══════════════════════════════════════════════════════════════════════════════

FULL FINE-TUNING (Traditional):
┌─────────────────────────────────────────────────────────────────────────────┐
│  Base Model: 7B parameters ──► Update ALL 7B parameters                     │
│                                                                              │
│  Requirements:                                                               │
│  • GPU Memory: 80GB+ (A100)                                                  │
│  • Training Time: Days to weeks                                              │
│  • Cost: $$$$$                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

LoRA FINE-TUNING (FinGPT):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Base Model (Frozen): 7B params     +     LoRA Adapters: 8M params          │
│         ┌────────────────────┐              ┌────────────────┐               │
│         │                    │              │                │               │
│         │   W (original)     │      +       │   BA (LoRA)    │               │
│         │   [4096 x 4096]    │              │   [4096 x r]   │               │
│         │    (FROZEN)        │              │   x [r x 4096] │               │
│         │                    │              │   r = 8 (rank) │               │
│         └────────────────────┘              │   (TRAINABLE)  │               │
│                                             └────────────────┘               │
│                                                                              │
│  Requirements:                                                               │
│  • GPU Memory: 16GB (RTX 4090)                                              │
│  • Training Time: Hours                                                      │
│  • Cost: $                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

LoRA configuration for FinGPT:

```python
from peft import LoraConfig, get_peft_model

# FinGPT LoRA configuration
lora_config = LoraConfig(
    r=8,                       # Rank of adaptation matrices
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} = {100*trainable_params/total_params:.2f}%")
# Output: Trainable: 8,388,608 / 6,738,415,616 = 0.12%
```

## Trading Applications

### Sentiment Analysis

FinGPT excels at financial sentiment analysis:

```python
# Example: FinGPT sentiment analysis prompt
prompt = """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.

Input: NVIDIA reports record revenue driven by AI chip demand, stock surges 10%.

Answer:"""

# FinGPT response: "positive"
```

**Benchmark Results (Accuracy):**

| Dataset | FinGPT-LLaMA | FinBERT | ChatGPT | BloombergGPT |
|---------|-------------|---------|---------|--------------|
| FPB (Financial PhraseBank) | **87.2%** | 86.5% | 78.3% | - |
| FiQA-SA | **85.4%** | 83.7% | 73.2% | - |
| TFNS (Twitter Financial) | **82.1%** | 80.3% | 71.5% | - |
| Headlines | **79.8%** | 77.2% | 68.4% | - |

### News-Based Trading Signals

Convert sentiment into actionable signals:

```python
class FinGPTSignalGenerator:
    """Generate trading signals from FinGPT sentiment analysis."""

    def __init__(self, model, confidence_threshold: float = 0.7):
        self.model = model
        self.confidence_threshold = confidence_threshold

        # Signal mapping
        self.sentiment_to_signal = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }

    def generate_signal(
        self,
        news: str,
        symbol: str
    ) -> dict:
        """
        Generate trading signal from news text.

        Returns:
            dict with signal, confidence, and reasoning
        """
        # Get FinGPT prediction
        prompt = self._create_prompt(news)
        response = self.model.generate(prompt)

        sentiment, confidence = self._parse_response(response)

        if confidence < self.confidence_threshold:
            return {
                "symbol": symbol,
                "signal": 0.0,
                "action": "HOLD",
                "confidence": confidence,
                "reason": "Low confidence prediction"
            }

        signal = self.sentiment_to_signal[sentiment] * confidence

        return {
            "symbol": symbol,
            "signal": signal,
            "action": "BUY" if signal > 0.3 else "SELL" if signal < -0.3 else "HOLD",
            "confidence": confidence,
            "reason": f"Sentiment: {sentiment}"
        }
```

### Robo-Advisor Integration

FinGPT can power AI financial advisors:

```python
# Example: FinGPT robo-advisor prompt
prompt = """You are a financial advisor assistant. Based on the following market conditions and user portfolio, provide investment advice.

Market Conditions:
- S&P 500: Up 2% this week
- Fed signaling potential rate cuts
- Tech sector showing strong momentum
- Inflation data came in lower than expected

User Portfolio:
- 60% stocks (heavy tech), 30% bonds, 10% cash
- Risk tolerance: Moderate
- Investment horizon: 10 years

Question: Should I rebalance my portfolio given current conditions?

Advice:"""

# FinGPT provides contextual, data-driven advice
```

### Cryptocurrency Analysis

FinGPT supports crypto market analysis with Bybit data:

```python
class CryptoAnalyzer:
    """Analyze cryptocurrency markets using FinGPT."""

    def __init__(self, fingpt_model, bybit_client):
        self.model = fingpt_model
        self.bybit = bybit_client

    async def analyze_crypto_news(
        self,
        symbol: str = "BTCUSDT",
        news_items: list = None
    ) -> dict:
        """
        Analyze crypto news and generate trading signals.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            news_items: List of news headlines

        Returns:
            Analysis with sentiment and signal
        """
        # Get recent price data from Bybit
        price_data = await self.bybit.get_klines(
            symbol=symbol,
            interval="1h",
            limit=24
        )

        # Analyze each news item
        signals = []
        for news in news_items:
            signal = self._analyze_single_news(news, symbol)
            signals.append(signal)

        # Aggregate signals
        avg_signal = sum(s["signal"] for s in signals) / len(signals)

        return {
            "symbol": symbol,
            "current_price": price_data[-1]["close"],
            "24h_change": self._calc_change(price_data),
            "news_signal": avg_signal,
            "recommendation": self._get_recommendation(avg_signal),
            "individual_signals": signals
        }
```

## Practical Examples

### 01: Financial Sentiment Analysis

```python
# python/examples/01_sentiment_demo.py

"""
FinGPT Sentiment Analysis Demo

This example demonstrates how to use FinGPT for financial sentiment analysis.
Since the full FinGPT model requires significant resources, we provide
both a mock implementation and integration with the actual model.
"""

import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: str  # positive, negative, neutral
    confidence: float
    score: float  # -1 to 1


class FinGPTSentimentAnalyzer:
    """
    Financial sentiment analyzer using FinGPT approach.

    This class provides both mock and real implementations
    for demonstration and production use.
    """

    def __init__(self, use_mock: bool = True, model_path: str = None):
        """
        Initialize the analyzer.

        Args:
            use_mock: If True, use rule-based mock for demos
            model_path: Path to FinGPT model weights
        """
        self.use_mock = use_mock
        self.model = None

        if not use_mock and model_path:
            self._load_model(model_path)

        # Financial sentiment keywords
        self.positive_keywords = {
            "beat", "exceed", "surge", "soar", "record", "growth",
            "profit", "gain", "bullish", "upgrade", "outperform",
            "strong", "positive", "optimistic", "rally", "boom"
        }
        self.negative_keywords = {
            "miss", "decline", "drop", "plunge", "loss", "fall",
            "bearish", "downgrade", "underperform", "weak", "negative",
            "pessimistic", "crash", "slump", "warning", "concern"
        }

    def _load_model(self, model_path: str):
        """Load the actual FinGPT model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"Warning: Could not load model, using mock: {e}")
            self.use_mock = True

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news or document

        Returns:
            SentimentResult with sentiment label and confidence
        """
        if self.use_mock:
            return self._mock_analyze(text)
        else:
            return self._model_analyze(text)

    def _mock_analyze(self, text: str) -> SentimentResult:
        """Rule-based mock analysis for demonstration."""
        text_lower = text.lower()

        pos_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in self.negative_keywords if kw in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(
                sentiment="neutral",
                confidence=0.5,
                score=0.0
            )

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if pos_ratio > 0.6:
            sentiment = "positive"
            score = pos_ratio
            confidence = min(0.5 + pos_count * 0.1, 0.95)
        elif neg_ratio > 0.6:
            sentiment = "negative"
            score = -neg_ratio
            confidence = min(0.5 + neg_count * 0.1, 0.95)
        else:
            sentiment = "neutral"
            score = pos_ratio - neg_ratio
            confidence = 0.6

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score
        )

    def _model_analyze(self, text: str) -> SentimentResult:
        """Analyze using the actual FinGPT model."""
        prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.

Input: {text}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = self._extract_sentiment(response)

        return SentimentResult(
            sentiment=sentiment,
            confidence=0.85,  # Default confidence for model predictions
            score={"positive": 1.0, "neutral": 0.0, "negative": -1.0}[sentiment]
        )

    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment label from model response."""
        response_lower = response.lower()
        if "positive" in response_lower:
            return "positive"
        elif "negative" in response_lower:
            return "negative"
        return "neutral"

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]


def main():
    """Demo of FinGPT sentiment analysis."""
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)

    # Test cases
    test_texts = [
        "Apple reported record quarterly revenue of $124 billion, beating analyst expectations by 5%.",
        "Tesla shares plunged 12% after disappointing delivery numbers and concerns about demand.",
        "The Federal Reserve kept interest rates unchanged at the latest meeting.",
        "NVIDIA stock surged to new all-time highs amid strong AI chip demand.",
        "Bitcoin dropped below $40,000 as regulatory concerns mount.",
        "Microsoft cloud revenue growth exceeded expectations, driving stock higher.",
    ]

    print("=" * 70)
    print("FinGPT Financial Sentiment Analysis Demo")
    print("=" * 70)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:70]}...")
        print(f"Sentiment: {result.sentiment.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Score: {result.score:+.2f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
```

### 02: Trading Signal Generation

```python
# python/examples/02_signal_generation.py

"""
FinGPT Trading Signal Generation

Generate trading signals from financial news using FinGPT sentiment analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TradingSignal:
    """Trading signal generated from news analysis."""
    timestamp: datetime
    symbol: str
    signal: float  # -1 (strong sell) to 1 (strong buy)
    confidence: float
    source: str
    news_text: str


class FinGPTTradingEngine:
    """
    Generate and manage trading signals using FinGPT.

    This engine combines sentiment analysis with signal generation
    and aggregation for actionable trading decisions.
    """

    def __init__(
        self,
        sentiment_analyzer,
        signal_threshold: float = 0.3,
        confidence_threshold: float = 0.6
    ):
        self.analyzer = sentiment_analyzer
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

        # Source importance weights
        self.source_weights = {
            "earnings": 1.0,     # Highest weight for earnings news
            "sec_filing": 0.9,  # SEC filings are reliable
            "news_wire": 0.8,   # Major news sources
            "analyst": 0.7,     # Analyst reports
            "social": 0.4,      # Social media (noisy)
        }

    def generate_signal(
        self,
        news_text: str,
        symbol: str,
        source: str = "news_wire",
        timestamp: datetime = None
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from news text.

        Args:
            news_text: News headline or article
            symbol: Stock/crypto symbol
            source: Source type for weighting
            timestamp: News timestamp

        Returns:
            TradingSignal if confidence meets threshold
        """
        timestamp = timestamp or datetime.now()

        # Analyze sentiment
        result = self.analyzer.analyze(news_text)

        # Check confidence threshold
        if result.confidence < self.confidence_threshold:
            return None

        # Calculate weighted signal
        source_weight = self.source_weights.get(source, 0.5)
        signal = result.score * result.confidence * source_weight

        # Check signal threshold
        if abs(signal) < self.signal_threshold:
            return None

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal,
            confidence=result.confidence,
            source=source,
            news_text=news_text
        )

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        decay_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Aggregate multiple signals with time decay.

        Args:
            signals: List of trading signals
            decay_hours: Half-life for signal decay

        Returns:
            Dict mapping symbols to aggregated position scores
        """
        now = datetime.now()
        positions = {}

        # Group by symbol
        symbol_signals = {}
        for sig in signals:
            if sig.symbol not in symbol_signals:
                symbol_signals[sig.symbol] = []
            symbol_signals[sig.symbol].append(sig)

        # Aggregate with exponential time decay
        for symbol, sigs in symbol_signals.items():
            weighted_sum = 0.0
            weight_sum = 0.0

            for sig in sigs:
                hours_old = (now - sig.timestamp).total_seconds() / 3600
                decay = np.exp(-hours_old / decay_hours)

                weight = sig.confidence * decay
                weighted_sum += sig.signal * weight
                weight_sum += weight

            if weight_sum > 0:
                positions[symbol] = np.clip(weighted_sum / weight_sum, -1, 1)

        return positions

    def get_recommendations(
        self,
        positions: Dict[str, float]
    ) -> List[Dict]:
        """
        Convert position scores to recommendations.

        Args:
            positions: Dict of symbol -> position score

        Returns:
            List of recommendations with actions
        """
        recommendations = []

        for symbol, score in sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True):
            if score > 0.5:
                action = "STRONG BUY"
            elif score > 0.2:
                action = "BUY"
            elif score < -0.5:
                action = "STRONG SELL"
            elif score < -0.2:
                action = "SELL"
            else:
                action = "HOLD"

            recommendations.append({
                "symbol": symbol,
                "score": score,
                "action": action,
                "confidence": abs(score)
            })

        return recommendations


def demo_signal_generation():
    """Demonstrate signal generation with sample news."""
    from fingpt_sentiment import FinGPTSentimentAnalyzer

    analyzer = FinGPTSentimentAnalyzer(use_mock=True)
    engine = FinGPTTradingEngine(analyzer)

    # Sample news items
    news_items = [
        {
            "text": "Apple reports record iPhone sales in China, shares rally 5%",
            "symbol": "AAPL",
            "source": "earnings"
        },
        {
            "text": "Apple faces antitrust investigation in EU over App Store",
            "symbol": "AAPL",
            "source": "news_wire"
        },
        {
            "text": "NVIDIA AI chips dominate data center market, revenue beats estimates",
            "symbol": "NVDA",
            "source": "earnings"
        },
        {
            "text": "Tesla delivery numbers miss expectations, stock drops 8%",
            "symbol": "TSLA",
            "source": "earnings"
        },
        {
            "text": "Bitcoin surges past $50,000 on ETF approval optimism",
            "symbol": "BTC",
            "source": "news_wire"
        },
    ]

    print("=" * 70)
    print("FinGPT Trading Signal Generation Demo")
    print("=" * 70)

    signals = []
    for item in news_items:
        signal = engine.generate_signal(
            news_text=item["text"],
            symbol=item["symbol"],
            source=item["source"]
        )

        if signal:
            signals.append(signal)
            print(f"\n{signal.symbol}:")
            print(f"  News: {signal.news_text[:50]}...")
            print(f"  Signal: {signal.signal:+.2f}")
            print(f"  Confidence: {signal.confidence:.1%}")

    # Aggregate signals
    print("\n" + "=" * 70)
    print("Aggregated Recommendations:")
    print("=" * 70)

    positions = engine.aggregate_signals(signals)
    recommendations = engine.get_recommendations(positions)

    for rec in recommendations:
        print(f"\n{rec['symbol']}:")
        print(f"  Action: {rec['action']}")
        print(f"  Score: {rec['score']:+.2f}")


if __name__ == "__main__":
    demo_signal_generation()
```

### 03: Crypto Market Analysis with Bybit

```python
# python/examples/03_crypto_bybit.py

"""
Cryptocurrency Analysis with FinGPT and Bybit Data

Demonstrates integration of FinGPT sentiment analysis with
Bybit cryptocurrency exchange data for crypto trading signals.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CryptoAnalysis:
    """Result of cryptocurrency analysis."""
    symbol: str
    current_price: float
    price_change_24h: float
    sentiment_score: float
    signal: str
    news_summary: List[Dict]


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Note: This is a mock implementation for demonstration.
    In production, use the pybit library or Bybit API directly.
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        # In production: self.client = HTTP(api_key=api_key, api_secret=api_secret)

    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        # Mock data for demonstration
        mock_prices = {
            "BTCUSDT": {"price": 52340.50, "change_24h": 2.35},
            "ETHUSDT": {"price": 2890.25, "change_24h": 3.12},
            "SOLUSDT": {"price": 108.45, "change_24h": -1.24},
            "BNBUSDT": {"price": 315.80, "change_24h": 0.85},
        }

        data = mock_prices.get(symbol, {"price": 100.0, "change_24h": 0.0})
        return {
            "symbol": symbol,
            "lastPrice": data["price"],
            "price24hPcnt": data["change_24h"] / 100
        }

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 24
    ) -> List[Dict]:
        """Get candlestick/kline data."""
        # Mock OHLCV data
        import numpy as np

        base_price = 52000 if "BTC" in symbol else 2800
        prices = base_price * (1 + np.cumsum(np.random.randn(limit) * 0.005))

        return [
            {
                "timestamp": datetime.now().timestamp() - (limit - i) * 3600,
                "open": prices[i],
                "high": prices[i] * 1.002,
                "low": prices[i] * 0.998,
                "close": prices[i] * (1 + np.random.randn() * 0.001),
                "volume": np.random.uniform(100, 1000)
            }
            for i in range(limit)
        ]

    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get order book data."""
        base_price = 52000 if "BTC" in symbol else 2800

        return {
            "bids": [
                {"price": base_price - i * 10, "qty": 0.5 + i * 0.1}
                for i in range(limit)
            ],
            "asks": [
                {"price": base_price + i * 10, "qty": 0.5 + i * 0.1}
                for i in range(limit)
            ]
        }


class CryptoNewsAnalyzer:
    """
    Analyze cryptocurrency news using FinGPT.
    """

    def __init__(self, sentiment_analyzer, bybit_loader: BybitDataLoader):
        self.analyzer = sentiment_analyzer
        self.bybit = bybit_loader

        # Crypto-specific keywords for enhanced analysis
        self.crypto_positive = {
            "adoption", "institutional", "etf", "approval", "bullish",
            "halving", "accumulation", "whale", "partnership"
        }
        self.crypto_negative = {
            "hack", "regulatory", "ban", "crash", "bearish",
            "liquidation", "scam", "fraud", "investigation"
        }

    async def analyze(
        self,
        symbol: str,
        news_items: List[str]
    ) -> CryptoAnalysis:
        """
        Perform comprehensive crypto analysis.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            news_items: List of news headlines

        Returns:
            CryptoAnalysis with price data and sentiment
        """
        # Get market data
        ticker = await self.bybit.get_ticker(symbol)

        # Analyze each news item
        news_analysis = []
        sentiment_scores = []

        for news in news_items:
            result = self.analyzer.analyze(news)
            sentiment_scores.append(result.score)
            news_analysis.append({
                "text": news,
                "sentiment": result.sentiment,
                "confidence": result.confidence
            })

        # Calculate aggregate sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Generate signal
        if avg_sentiment > 0.3:
            signal = "BUY"
        elif avg_sentiment < -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"

        return CryptoAnalysis(
            symbol=symbol,
            current_price=ticker["lastPrice"],
            price_change_24h=ticker["price24hPcnt"] * 100,
            sentiment_score=avg_sentiment,
            signal=signal,
            news_summary=news_analysis
        )


async def demo_crypto_analysis():
    """Demonstrate crypto analysis with Bybit data."""
    from fingpt_sentiment import FinGPTSentimentAnalyzer

    analyzer = FinGPTSentimentAnalyzer(use_mock=True)
    bybit = BybitDataLoader()
    crypto_analyzer = CryptoNewsAnalyzer(analyzer, bybit)

    # Sample crypto news
    btc_news = [
        "BlackRock Bitcoin ETF sees record inflows of $500 million",
        "Bitcoin hashrate reaches new all-time high, network security strengthens",
        "Institutional investors accumulating BTC according to on-chain data",
    ]

    eth_news = [
        "Ethereum gas fees drop to lowest level in months",
        "Major DeFi protocol hacked, $50 million stolen",
        "Ethereum Foundation announces major protocol upgrade roadmap",
    ]

    print("=" * 70)
    print("Cryptocurrency Analysis with FinGPT + Bybit Data")
    print("=" * 70)

    for symbol, news in [("BTCUSDT", btc_news), ("ETHUSDT", eth_news)]:
        analysis = await crypto_analyzer.analyze(symbol, news)

        print(f"\n{analysis.symbol}")
        print("-" * 50)
        print(f"Current Price: ${analysis.current_price:,.2f}")
        print(f"24h Change: {analysis.price_change_24h:+.2f}%")
        print(f"Sentiment Score: {analysis.sentiment_score:+.2f}")
        print(f"Signal: {analysis.signal}")
        print("\nNews Analysis:")
        for item in analysis.news_summary:
            print(f"  • {item['sentiment'].upper()}: {item['text'][:50]}...")


if __name__ == "__main__":
    asyncio.run(demo_crypto_analysis())
```

### 04: Backtesting FinGPT Signals

```python
# python/examples/04_backtest_demo.py

"""
Backtesting FinGPT Trading Signals

Demonstrates how to backtest trading strategies based on
FinGPT sentiment analysis signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    max_position_pct: float = 0.1  # Max 10% per position
    transaction_cost_bps: float = 10  # 10 basis points
    signal_threshold: float = 0.3
    rebalance_frequency: str = "daily"


@dataclass
class BacktestResult:
    """Results of backtesting."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    portfolio_values: pd.Series


class FinGPTBacktester:
    """
    Backtest FinGPT-based trading strategies.

    This backtester is designed specifically for news-driven
    trading signals with irregular timing.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame
    ) -> BacktestResult:
        """
        Run backtest on FinGPT signals.

        Args:
            signals: DataFrame with [timestamp, symbol, signal, confidence]
            prices: DataFrame with OHLC data indexed by timestamp

        Returns:
            BacktestResult with performance metrics
        """
        capital = self.config.initial_capital
        positions = {}  # symbol -> shares
        portfolio_values = []
        trades = []

        # Get all unique dates
        dates = prices.index.unique().sort_values()

        for date in dates:
            # Get active signals for this date
            day_signals = self._get_active_signals(signals, date)

            # Calculate target positions
            target = self._calculate_targets(day_signals, prices.loc[date], capital)

            # Execute trades
            new_trades, capital = self._execute_trades(
                positions, target, prices.loc[date], capital, date
            )
            trades.extend(new_trades)
            positions = target.copy()

            # Calculate portfolio value
            position_value = sum(
                shares * prices.loc[date, symbol].iloc[0]
                if isinstance(prices.loc[date, symbol], pd.Series)
                else shares * prices.loc[date, symbol]
                for symbol, shares in positions.items()
                if symbol in prices.columns
            )
            portfolio_values.append({
                "date": date,
                "value": capital + position_value
            })

        # Calculate metrics
        pv = pd.DataFrame(portfolio_values).set_index("date")["value"]
        returns = pv.pct_change().dropna()

        return BacktestResult(
            total_return=(pv.iloc[-1] / pv.iloc[0]) - 1,
            annualized_return=self._annualize_return(returns),
            sharpe_ratio=self._calc_sharpe(returns),
            max_drawdown=self._calc_max_drawdown(pv),
            num_trades=len(trades),
            win_rate=self._calc_win_rate(trades),
            portfolio_values=pv
        )

    def _get_active_signals(
        self,
        signals: pd.DataFrame,
        date: datetime
    ) -> pd.DataFrame:
        """Get signals active on a given date."""
        lookback = timedelta(hours=24)
        mask = (
            (signals["timestamp"] >= date - lookback) &
            (signals["timestamp"] <= date)
        )
        return signals[mask]

    def _calculate_targets(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        capital: float
    ) -> Dict[str, float]:
        """Calculate target positions from signals."""
        if signals.empty:
            return {}

        targets = {}
        max_position = capital * self.config.max_position_pct

        for _, row in signals.iterrows():
            symbol = row["symbol"]
            if symbol not in prices.index:
                continue

            signal = row["signal"]
            if abs(signal) < self.config.signal_threshold:
                continue

            price = prices[symbol]
            position_value = signal * max_position
            shares = position_value / price
            targets[symbol] = shares

        return targets

    def _execute_trades(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        prices: pd.Series,
        capital: float,
        date: datetime
    ) -> tuple:
        """Execute rebalancing trades."""
        trades = []
        all_symbols = set(current.keys()) | set(target.keys())

        for symbol in all_symbols:
            curr_shares = current.get(symbol, 0)
            tgt_shares = target.get(symbol, 0)
            delta = tgt_shares - curr_shares

            if abs(delta) < 0.01 or symbol not in prices.index:
                continue

            price = prices[symbol]
            cost_mult = 1 + self.config.transaction_cost_bps / 10000
            trade_value = abs(delta) * price * cost_mult

            if delta > 0:
                capital -= trade_value
            else:
                capital += trade_value / cost_mult

            trades.append({
                "date": date,
                "symbol": symbol,
                "shares": delta,
                "price": price,
                "value": trade_value
            })

        return trades, capital

    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize returns assuming daily frequency."""
        total = (1 + returns).prod()
        n_years = len(returns) / 252
        return total ** (1 / n_years) - 1 if n_years > 0 else 0

    def _calc_sharpe(self, returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess = returns - rf / 252
        return np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    def _calc_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = values.expanding().max()
        dd = (values - peak) / peak
        return dd.min()

    def _calc_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate of trades."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return wins / len(trades)


def generate_mock_data():
    """Generate mock signals and prices for demonstration."""
    np.random.seed(42)

    # Generate price data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    symbols = ["AAPL", "NVDA", "TSLA", "BTC"]

    prices = pd.DataFrame(index=dates)
    base_prices = {"AAPL": 180, "NVDA": 500, "TSLA": 250, "BTC": 45000}

    for symbol, base in base_prices.items():
        returns = np.random.randn(len(dates)) * 0.02
        prices[symbol] = base * (1 + returns).cumprod()

    # Generate signals
    signal_dates = np.random.choice(dates, size=100, replace=False)
    signals = pd.DataFrame({
        "timestamp": signal_dates,
        "symbol": np.random.choice(symbols, size=100),
        "signal": np.random.uniform(-1, 1, size=100),
        "confidence": np.random.uniform(0.6, 0.95, size=100)
    })

    return signals, prices


def main():
    """Run backtest demonstration."""
    print("=" * 70)
    print("FinGPT Trading Strategy Backtest")
    print("=" * 70)

    # Generate test data
    signals, prices = generate_mock_data()

    # Configure and run backtest
    config = BacktestConfig(
        initial_capital=100000,
        max_position_pct=0.1,
        transaction_cost_bps=10,
        signal_threshold=0.3
    )

    backtester = FinGPTBacktester(config)
    result = backtester.run_backtest(signals, prices)

    print(f"\nBacktest Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Final Portfolio Value: ${result.portfolio_values.iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
```

## Rust Implementation

```
rust_fingpt/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── sentiment.rs        # Sentiment analysis
│   ├── signals.rs          # Trading signal generation
│   ├── backtest.rs         # Backtesting engine
│   ├── data.rs             # Market data loading
│   ├── api.rs              # External API clients
│   └── error.rs            # Error types
└── src/bin/
    ├── sentiment_demo.rs   # Sentiment analysis demo
    ├── signal_demo.rs      # Signal generation demo
    ├── crypto_demo.rs      # Crypto analysis with Bybit
    └── backtest_demo.rs    # Backtesting demo
```

See [rust_fingpt](rust_fingpt/) for complete Rust implementation.

### Quick Start (Rust)

```bash
cd rust_fingpt

# Run sentiment analysis demo
cargo run --bin sentiment_demo

# Generate trading signals
cargo run --bin signal_demo -- --symbol AAPL

# Crypto analysis with Bybit
cargo run --bin crypto_demo -- --symbol BTCUSDT

# Run backtest
cargo run --bin backtest_demo -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── fingpt_sentiment.py    # Sentiment analysis
├── signals.py             # Trading signal generation
├── backtest.py            # Backtesting engine
├── data_loader.py         # Data loading utilities
├── bybit_client.py        # Bybit API client
├── requirements.txt       # Dependencies
└── examples/
    ├── 01_sentiment_demo.py
    ├── 02_signal_generation.py
    ├── 03_crypto_bybit.py
    └── 04_backtest_demo.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run sentiment analysis
python examples/01_sentiment_demo.py

# Generate trading signals
python examples/02_signal_generation.py

# Crypto analysis
python examples/03_crypto_bybit.py

# Run backtest
python examples/04_backtest_demo.py
```

## Comparison with Other Models

| Feature | FinGPT | BloombergGPT | FinBERT | ChatGPT |
|---------|--------|--------------|---------|---------|
| **Open Source** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Financial Training** | ✅ Extensive | ✅ Extensive | ✅ Financial | ❌ General |
| **Customizable** | ✅ LoRA fine-tuning | ❌ Fixed | ✅ Fine-tunable | ❌ API only |
| **Real-time Data** | ✅ Data pipelines | ❌ Static training | ❌ Static | ❌ Knowledge cutoff |
| **RLSP Training** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Model Sizes** | 7B-70B | 50B | 110M | ~1T |
| **Hardware Req.** | Consumer GPU | Datacenter | Low | API |
| **Cost** | Free | $$$$$ | Free | $$ |

### Performance Benchmarks

| Task | FinGPT-LLaMA | FinBERT | ChatGPT | BloombergGPT |
|------|-------------|---------|---------|--------------|
| Sentiment (FPB) | **87.2%** | 86.5% | 78.3% | N/A |
| Sentiment (FiQA) | **85.4%** | 83.7% | 73.2% | N/A |
| NER (Finance) | 82.1% | **84.2%** | 76.5% | N/A |
| QA (ConvFinQA) | **71.3%** | 58.4% | 68.9% | N/A |

## Best Practices

### When to Use FinGPT

**Ideal use cases:**
- Sentiment analysis on financial news
- Building custom trading signal pipelines
- Research and experimentation
- Cost-sensitive applications
- Self-hosted deployments

**Consider alternatives when:**
- Ultra-low latency required (use FinBERT)
- Maximum accuracy critical (consider commercial options)
- No ML expertise available (use ChatGPT API)

### Fine-tuning Tips

1. **Data Quality**
   ```python
   # Always clean and validate training data
   def clean_financial_text(text: str) -> str:
       # Remove HTML
       text = re.sub(r'<[^>]+>', '', text)
       # Normalize whitespace
       text = ' '.join(text.split())
       # Keep financial entities
       return text
   ```

2. **LoRA Configuration**
   ```python
   # Recommended settings for FinGPT
   lora_config = LoraConfig(
       r=8,                    # Start with r=8, increase if underfitting
       lora_alpha=32,          # alpha = 4 * r is common
       target_modules=[        # Target attention layers
           "q_proj", "v_proj", "k_proj", "o_proj"
       ],
       lora_dropout=0.05,      # Light regularization
   )
   ```

3. **Training Hyperparameters**
   ```python
   training_args = TrainingArguments(
       learning_rate=2e-4,     # Higher than full fine-tuning
       num_train_epochs=3,     # Usually sufficient
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,  # Effective batch=16
       warmup_steps=100,
       fp16=True,              # Mixed precision
   )
   ```

### Signal Generation Guidelines

1. **Confidence Thresholds**
   ```python
   # Filter low-confidence predictions
   if result.confidence < 0.6:
       return None  # Skip uncertain signals
   ```

2. **Source Weighting**
   ```python
   weights = {
       "earnings": 1.0,    # Direct financial data
       "sec_filing": 0.9,  # Regulatory filings
       "news": 0.7,        # News sources
       "social": 0.3,      # Social media (noisy)
   }
   ```

3. **Signal Decay**
   ```python
   # News impact decays over time
   hours_since_news = (now - news_time).total_seconds() / 3600
   decay_factor = np.exp(-hours_since_news / 24)  # 24-hour half-life
   adjusted_signal = raw_signal * decay_factor
   ```

## Resources

### Papers

- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) — Original FinGPT paper (2023)
- [FinGPT-HPC: Efficient Pretraining and Fine-tuning](https://arxiv.org/abs/2402.12659) — High-performance computing for FinGPT
- [Instruct-FinGPT: Financial Sentiment Analysis](https://arxiv.org/abs/2306.12659) — Instruction-tuning for finance

### Repositories

- [FinGPT Official](https://github.com/AI4Finance-Foundation/FinGPT) — Main FinGPT repository
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) — Companion RL library
- [FinNLP](https://github.com/AI4Finance-Foundation/FinNLP) — Financial NLP tools

### Related Chapters

- [Chapter 62: BloombergGPT Trading](../62_bloomberggpt_trading) — Proprietary alternative
- [Chapter 241: FinBERT Sentiment](../241_finbert_sentiment) — Smaller, faster model
- [Chapter 67: LLM Sentiment Analysis](../67_llm_sentiment_analysis) — Deep dive on sentiment
- [Chapter 70: Fine-tuning LLM for Finance](../70_fine_tuning_llm_finance) — Advanced fine-tuning

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Understanding of transformer architecture and LLMs
- Python programming experience
- Basic knowledge of financial markets
- Familiarity with PyTorch (for fine-tuning)

## References

1. Yang, H., et al. (2023). "FinGPT: Open-Source Financial Large Language Models." arXiv:2306.06031
2. Liu, X., et al. (2023). "FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications."
3. Wu, S., et al. (2023). "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564
4. Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models."
