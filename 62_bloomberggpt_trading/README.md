# Chapter 62: BloombergGPT for Trading — Financial LLM Applications

This chapter explores **BloombergGPT**, Bloomberg's 50-billion parameter Large Language Model specifically designed for the financial domain. We examine how domain-specific LLMs can be leveraged for trading applications including sentiment analysis, entity recognition, and financial question answering.

<p align="center">
<img src="https://i.imgur.com/YZB9kWp.png" width="70%">
</p>

## Contents

1. [Introduction to BloombergGPT](#introduction-to-bloomberggpt)
    * [Why Domain-Specific LLMs?](#why-domain-specific-llms)
    * [Key Innovations](#key-innovations)
    * [Comparison with Other Models](#comparison-with-other-models)
2. [BloombergGPT Architecture](#bloomberggpt-architecture)
    * [Model Specifications](#model-specifications)
    * [Training Data Composition](#training-data-composition)
    * [Training Methodology](#training-methodology)
3. [Trading Applications](#trading-applications)
    * [Sentiment Analysis for Trading](#sentiment-analysis-for-trading)
    * [Named Entity Recognition](#named-entity-recognition)
    * [Financial Question Answering](#financial-question-answering)
    * [News Classification](#news-classification)
4. [Practical Examples](#practical-examples)
    * [01: Financial Sentiment Analysis](#01-financial-sentiment-analysis)
    * [02: Trading Signal Generation](#02-trading-signal-generation)
    * [03: News Impact Prediction](#03-news-impact-prediction)
    * [04: Backtesting LLM Signals](#04-backtesting-llm-signals)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to BloombergGPT

BloombergGPT represents a paradigm shift in financial NLP. While general-purpose LLMs like GPT-4 or BLOOM can handle financial tasks, BloombergGPT was trained specifically on financial data, achieving superior performance on domain-specific tasks without sacrificing general language understanding.

### Why Domain-Specific LLMs?

General-purpose LLMs face challenges with financial language:

```
GENERAL LLM CHALLENGES:
┌──────────────────────────────────────────────────────────────────┐
│  1. DOMAIN JARGON                                                │
│     "The stock is trading at 15x forward P/E with a 2% div yield"│
│     General LLM: May misinterpret technical terms                │
│     BloombergGPT: Understands valuation metrics natively         │
├──────────────────────────────────────────────────────────────────┤
│  2. ENTITY DISAMBIGUATION                                        │
│     "Apple announced quarterly earnings"                          │
│     General LLM: Is this the fruit or the company?               │
│     BloombergGPT: Clearly identifies AAPL context                │
├──────────────────────────────────────────────────────────────────┤
│  3. TEMPORAL REASONING                                           │
│     "Q3 results beat consensus by 200bps"                        │
│     General LLM: May not link Q3 to specific timeframe           │
│     BloombergGPT: Trained on temporal financial patterns         │
├──────────────────────────────────────────────────────────────────┤
│  4. SENTIMENT NUANCE                                             │
│     "The company maintained guidance despite headwinds"          │
│     General LLM: Neutral or slightly negative?                   │
│     BloombergGPT: Recognizes as mildly positive (guidance held)  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Massive Financial Dataset (FinPile)**
   - 363 billion tokens of proprietary Bloomberg financial data
   - 40 years of financial documents, news, filings, and transcripts
   - Largest domain-specific dataset for financial LLM training

2. **Mixed Training Strategy**
   - Combined financial data (51.27%) with general data (48.73%)
   - Maintains general language capabilities
   - Achieves best-of-both-worlds performance

3. **Aspect-Specific Sentiment**
   - Goes beyond binary positive/negative
   - Identifies sentiment toward specific entities in text
   - Crucial for trading signals from multi-topic news

4. **Financial Entity Disambiguation**
   - Links mentions to Bloomberg entity IDs
   - Distinguishes between similarly-named companies
   - Critical for accurate trading signal generation

### Comparison with Other Models

| Feature | GPT-4 | BLOOM-176B | FinBERT | BloombergGPT |
|---------|-------|------------|---------|--------------|
| Parameters | ~1.7T | 176B | 110M | 50.6B |
| Financial pretraining | Limited | None | Yes | Extensive |
| General NLP | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| Financial sentiment | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Entity disambiguation | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ |
| Publicly available | API only | Yes | Yes | No |

## BloombergGPT Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           BloombergGPT ARCHITECTURE                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Input Text ─────────────────────────────────────────────────────────────┐   │
│  "Apple reported Q3 earnings above expectations, stock up 5%"            │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                         │                                     │
│                                         ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                    Token Embedding + ALiBi Positions                   │   │
│  │    Vocabulary: 131,072 tokens | Hidden Dim: 7,680                      │   │
│  └─────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                     Layer Normalization                                │   │
│  └─────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │              TRANSFORMER DECODER BLOCK (×70 layers)                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Multi-Head Self-Attention (40 heads)                           │  │   │
│  │  │  • Causal masking for autoregressive generation                 │  │   │
│  │  │  • ALiBi positional encoding (no position embeddings)           │  │   │
│  │  │  • Head dimension: 7,680 / 40 = 192                             │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Feed-Forward Network                                           │  │   │
│  │  │  • Hidden: 7,680 → 30,720 → 7,680                               │  │   │
│  │  │  • GELU activation                                              │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                         │   │
│  │  + Residual Connections + Layer Normalization                         │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                Final Layer Norm → Linear → Softmax                     │   │
│  │                      (tied with input embeddings)                      │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  Output: Next token probabilities / Task-specific predictions                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Specification | Value |
|--------------|-------|
| Total Parameters | 50.6 billion |
| Layers | 70 transformer decoder blocks |
| Attention Heads | 40 |
| Hidden Dimension | 7,680 |
| FFN Hidden Dimension | 30,720 (4× hidden) |
| Vocabulary Size | 131,072 tokens |
| Context Length | 2,048 tokens |
| Positional Encoding | ALiBi (Attention with Linear Biases) |
| Activation Function | GELU |

### Training Data Composition

```
TRAINING DATA BREAKDOWN (708B tokens, ~569B used)
═══════════════════════════════════════════════════════════════════════════════

FINPILE (363B tokens, 51.27%)                    PUBLIC DATA (345B tokens, 48.73%)
┌─────────────────────────────────────────┐     ┌──────────────────────────────┐
│  Bloomberg Web     (298.0B)  81.98%     │     │  The Pile    (184.6B) 53.50% │
│  Bloomberg News     (38.1B)  10.48%     │     │  C4          (138.5B) 40.14% │
│  Bloomberg Filings  (14.1B)   3.88%     │     │  Wikipedia    (21.9B)  6.36% │
│  Bloomberg Press    (13.3B)   3.66%     │     └──────────────────────────────┘
└─────────────────────────────────────────┘

FINPILE SOURCES:
• Web: Curated financial websites and portals
• News: Bloomberg news articles and wire services
• Filings: SEC filings, regulatory documents, financial reports
• Press: Press releases and company announcements
```

### Training Methodology

```python
# Training configuration used for BloombergGPT
training_config = {
    # Optimization
    "optimizer": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "weight_decay": 0.1,

    # Learning rate schedule
    "max_learning_rate": 6e-5,
    "min_learning_rate": 6e-6,
    "lr_schedule": "cosine_decay",
    "warmup_steps": 1600,

    # Batch size
    "batch_size": 1024,  # Initial
    "batch_size_final": 2048,  # After ramp-up
    "sequence_length": 2048,

    # Training duration
    "total_steps": 139200,
    "training_days": 53,

    # Hardware
    "gpus": 512,  # A100 40GB
    "gpu_instances": 64,
    "throughput_tflops": 102,

    # Efficiency techniques
    "zero_stage": 3,
    "activation_checkpointing": True,
    "mixed_precision": "bf16",  # Forward/backward
    "param_precision": "fp32",  # Optimizer states
}
```

## Trading Applications

### Sentiment Analysis for Trading

BloombergGPT excels at aspect-specific sentiment analysis, which is crucial for generating actionable trading signals:

```python
# Example: Aspect-specific sentiment for trading
text = """
Microsoft reported strong cloud growth, with Azure revenue up 29% YoY.
However, the company lowered guidance for the PC segment due to weak
consumer demand. CEO Nadella emphasized AI investments as key priority.
"""

# BloombergGPT can identify:
aspects = {
    "Microsoft_Cloud": "POSITIVE",     # Strong growth, beats
    "Microsoft_PC": "NEGATIVE",        # Lowered guidance
    "Microsoft_AI": "NEUTRAL/POSITIVE", # Strategic priority
    "Microsoft_Overall": "POSITIVE"     # Net positive narrative
}

# Trading signal generation
def generate_signal(aspects, weights):
    """
    Generate trading signal from aspect sentiments.

    Args:
        aspects: Dict of aspect -> sentiment
        weights: Dict of aspect -> importance weight

    Returns:
        Signal strength (-1 to 1)
    """
    sentiment_scores = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}

    total_weight = sum(weights.values())
    signal = sum(
        weights[aspect] * sentiment_scores[sentiment]
        for aspect, sentiment in aspects.items()
    ) / total_weight

    return signal  # e.g., 0.4 -> Mild Buy signal
```

**Benchmark Results (F1 Score):**

| Task | BloombergGPT | BLOOM-176B | GPT-NeoX |
|------|-------------|------------|----------|
| Equity News Sentiment | **79.63%** | 19.96% | 14.72% |
| Social Media Sentiment | **63.96%** | 21.63% | 17.22% |
| Transcript Sentiment | **52.70%** | 14.51% | 13.46% |
| ES News Sentiment | **58.36%** | 42.86% | 16.39% |
| Average | **62.47%** | 24.24% | 15.45% |

### Named Entity Recognition

Identifying financial entities accurately is critical for trading:

```python
# Example: NER for trading
text = "Apple stock surged after Tim Cook announced new iPhone sales records."

# BloombergGPT NER output:
entities = [
    {"text": "Apple", "type": "ORG", "bloomberg_id": "AAPL US Equity"},
    {"text": "Tim Cook", "type": "PER", "role": "CEO"},
    {"text": "iPhone", "type": "PRODUCT", "company": "AAPL"}
]

# Entity disambiguation example
ambiguous_text = "Apple reported earnings while apple harvest season begins."

# BloombergGPT correctly identifies:
# "Apple" (first) -> AAPL US Equity (company)
# "apple" (second) -> Not a financial entity (fruit)
```

**NER + Named Entity Disambiguation Results:**

| Dataset | BloombergGPT | BLOOM-176B | GPT-NeoX |
|---------|-------------|------------|----------|
| News Wire | **68.15%** | 45.87% | 39.45% |
| Filings | **62.34%** | 48.21% | 42.31% |
| Headlines | **65.92%** | 44.76% | 38.94% |
| Transcripts | **61.48%** | 43.89% | 37.82% |
| Average | **64.83%** | 45.43% | 39.26% |

### Financial Question Answering

BloombergGPT can answer complex financial questions:

```python
# Example: ConvFinQA task (conversational financial QA)

context = """
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per share data)

                          2023        2022        2021
Revenue                 $98,456     $87,321     $76,845
Cost of Revenue         $42,187     $38,542     $34,521
Gross Profit            $56,269     $48,779     $42,324
Operating Expenses      $28,456     $25,321     $22,187
Operating Income        $27,813     $23,458     $20,137
"""

question = "What was the year-over-year growth in operating income from 2022 to 2023?"

# BloombergGPT can:
# 1. Extract relevant numbers ($27,813 and $23,458)
# 2. Calculate: (27,813 - 23,458) / 23,458 = 18.55%
# 3. Provide answer: "Operating income grew 18.55% YoY"
```

### News Classification

Classifying news for trading relevance:

```python
# Example: News classification for trading
news_items = [
    {
        "headline": "Fed signals potential rate cuts in Q2 2024",
        "classification": {
            "topic": "MONETARY_POLICY",
            "market_impact": "HIGH",
            "direction": "RISK_ON",
            "assets_affected": ["SPY", "QQQ", "TLT", "GLD"]
        }
    },
    {
        "headline": "Tesla recalls 2M vehicles over autopilot concerns",
        "classification": {
            "topic": "REGULATORY",
            "market_impact": "MEDIUM",
            "direction": "NEGATIVE",
            "assets_affected": ["TSLA"]
        }
    }
]
```

## Practical Examples

### 01: Financial Sentiment Analysis

```python
# python/01_sentiment_analysis.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple

class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer using LLM-based approach.

    Since BloombergGPT is not publicly available, we demonstrate
    the approach using open-source alternatives (FinBERT, FinGPT)
    with the same interface BloombergGPT would provide.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.label_map = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news or document

        Returns:
            Dict with sentiment probabilities
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            "positive": probs[0].item(),
            "negative": probs[1].item(),
            "neutral": probs[2].item(),
            "label": self.label_map[probs.argmax().item()],
            "confidence": probs.max().item()
        }

    def analyze_aspects(
        self,
        text: str,
        entities: List[str]
    ) -> Dict[str, Dict]:
        """
        Analyze sentiment toward specific entities (aspect-based).

        This mimics BloombergGPT's aspect-specific sentiment capability.
        """
        results = {}

        for entity in entities:
            # Extract sentences mentioning the entity
            sentences = [s for s in text.split('.') if entity.lower() in s.lower()]

            if sentences:
                entity_text = '. '.join(sentences)
                results[entity] = self.analyze(entity_text)
            else:
                results[entity] = {"label": "NOT_MENTIONED", "confidence": 0.0}

        return results


# Example usage
def main():
    analyzer = FinancialSentimentAnalyzer()

    # Sample financial news
    news = """
    Apple Inc reported record quarterly revenue of $123.9 billion,
    beating analyst expectations. iPhone sales surged 8% in China
    despite concerns about economic slowdown. However, Mac sales
    declined 10% year-over-year amid weak PC demand. Tim Cook
    expressed optimism about AI features driving future growth.
    """

    # Overall sentiment
    overall = analyzer.analyze(news)
    print(f"Overall Sentiment: {overall['label']} ({overall['confidence']:.2%})")

    # Aspect-based sentiment
    aspects = analyzer.analyze_aspects(
        news,
        entities=["iPhone", "Mac", "AI", "China"]
    )

    for entity, sentiment in aspects.items():
        print(f"  {entity}: {sentiment['label']} ({sentiment.get('confidence', 0):.2%})")


if __name__ == "__main__":
    main()
```

### 02: Trading Signal Generation

```python
# python/02_trading_signals.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class TradingSignal:
    """Trading signal generated from LLM analysis."""
    timestamp: datetime
    symbol: str
    signal: float  # -1 (strong sell) to 1 (strong buy)
    confidence: float
    reasoning: str
    source_type: str  # "news", "earnings", "filing"

class LLMSignalGenerator:
    """
    Generate trading signals from LLM sentiment analysis.

    This class demonstrates how BloombergGPT-style analysis
    can be converted into actionable trading signals.
    """

    def __init__(
        self,
        sentiment_analyzer,
        signal_threshold: float = 0.6,
        confidence_threshold: float = 0.7
    ):
        self.analyzer = sentiment_analyzer
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

        # Sentiment to signal mapping
        self.sentiment_weights = {
            "POSITIVE": 1.0,
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0
        }

        # Source importance weights
        self.source_weights = {
            "earnings": 1.0,
            "filing": 0.8,
            "news": 0.6,
            "social": 0.3
        }

    def generate_signal(
        self,
        text: str,
        symbol: str,
        source_type: str = "news",
        timestamp: Optional[datetime] = None
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from financial text.

        Args:
            text: Financial text to analyze
            symbol: Stock symbol
            source_type: Type of source (news, earnings, filing)
            timestamp: Time of the text

        Returns:
            TradingSignal if confidence threshold met, else None
        """
        timestamp = timestamp or datetime.now()

        # Analyze sentiment
        sentiment = self.analyzer.analyze(text)

        # Check confidence threshold
        if sentiment['confidence'] < self.confidence_threshold:
            return None

        # Calculate signal strength
        base_signal = self.sentiment_weights[sentiment['label']]
        source_weight = self.source_weights.get(source_type, 0.5)
        signal_strength = base_signal * sentiment['confidence'] * source_weight

        # Apply threshold
        if abs(signal_strength) < self.signal_threshold * source_weight:
            return None

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal_strength,
            confidence=sentiment['confidence'],
            reasoning=f"Sentiment: {sentiment['label']}, Source: {source_type}",
            source_type=source_type
        )

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Aggregate multiple signals into a single position recommendation.

        Args:
            signals: List of trading signals
            window_hours: Time window for aggregation

        Returns:
            Dict of symbol -> recommended position
        """
        now = datetime.now()

        # Filter to recent signals
        recent_signals = [
            s for s in signals
            if (now - s.timestamp).total_seconds() < window_hours * 3600
        ]

        # Group by symbol
        symbol_signals: Dict[str, List[TradingSignal]] = {}
        for signal in recent_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)

        # Aggregate with time decay
        positions = {}
        for symbol, sigs in symbol_signals.items():
            weighted_sum = 0
            weight_total = 0

            for sig in sigs:
                # Time decay: more recent = higher weight
                hours_ago = (now - sig.timestamp).total_seconds() / 3600
                time_weight = np.exp(-hours_ago / window_hours)

                weighted_sum += sig.signal * sig.confidence * time_weight
                weight_total += sig.confidence * time_weight

            if weight_total > 0:
                positions[symbol] = np.clip(weighted_sum / weight_total, -1, 1)

        return positions


# Example usage with mock data
def demo_signal_generation():
    from sentiment_analysis import FinancialSentimentAnalyzer

    analyzer = FinancialSentimentAnalyzer()
    generator = LLMSignalGenerator(analyzer)

    # Sample news items
    news_items = [
        {
            "symbol": "AAPL",
            "text": "Apple reports record iPhone sales in China, shares surge",
            "source": "news"
        },
        {
            "symbol": "AAPL",
            "text": "Apple faces antitrust scrutiny in EU over App Store practices",
            "source": "news"
        },
        {
            "symbol": "MSFT",
            "text": "Microsoft Azure growth exceeds expectations, cloud dominance continues",
            "source": "earnings"
        }
    ]

    signals = []
    for item in news_items:
        signal = generator.generate_signal(
            text=item["text"],
            symbol=item["symbol"],
            source_type=item["source"]
        )
        if signal:
            signals.append(signal)
            print(f"Signal: {signal.symbol} = {signal.signal:.2f} ({signal.reasoning})")

    # Aggregate positions
    positions = generator.aggregate_signals(signals)
    print("\nAggregated Positions:")
    for symbol, position in positions.items():
        direction = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
        print(f"  {symbol}: {direction} ({position:+.2f})")


if __name__ == "__main__":
    demo_signal_generation()
```

### 03: News Impact Prediction

```python
# python/03_news_impact.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class NewsImpactPredictor(nn.Module):
    """
    Predict market impact of financial news.

    This model combines LLM embeddings with market data
    to predict the magnitude and direction of price moves
    following news events.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        market_features: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Text encoding branch
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Market features branch (volume, volatility, etc.)
        self.market_encoder = nn.Sequential(
            nn.Linear(market_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Combined prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # [direction, magnitude, confidence]
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        market_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict news impact.

        Args:
            text_embedding: LLM embedding of news text [batch, embedding_dim]
            market_features: Market state features [batch, market_features]

        Returns:
            Dict with direction (-1 to 1), magnitude (0 to inf), confidence (0 to 1)
        """
        text_encoded = self.text_encoder(text_embedding)
        market_encoded = self.market_encoder(market_features)

        combined = torch.cat([text_encoded, market_encoded], dim=-1)
        output = self.predictor(combined)

        return {
            "direction": torch.tanh(output[:, 0]),  # -1 to 1
            "magnitude": torch.exp(output[:, 1]),   # Positive, log-scale
            "confidence": torch.sigmoid(output[:, 2])  # 0 to 1
        }


def prepare_market_features(
    symbol: str,
    timestamp,
    lookback_minutes: int = 60
) -> np.ndarray:
    """
    Prepare market context features for impact prediction.

    Features include:
    - Recent volatility
    - Volume relative to average
    - Bid-ask spread
    - Time of day
    - Day of week
    - VIX level
    - Recent price momentum
    """
    # In production, fetch from market data API
    # Here we return mock features
    features = np.array([
        0.02,   # Recent 1h volatility
        1.5,    # Volume ratio vs 20-day avg
        0.001,  # Bid-ask spread (%)
        10.5,   # Hours since market open
        2,      # Day of week (0=Mon)
        18.5,   # VIX level
        0.005,  # 1h momentum
        0.012,  # 1d momentum
        100.0,  # Current price
        50000,  # Average daily volume
    ])

    return features


# Example: Combining with LLM for impact prediction
def predict_news_impact(
    news_text: str,
    symbol: str,
    llm_model,
    impact_predictor: NewsImpactPredictor
) -> Dict:
    """
    End-to-end news impact prediction.
    """
    # Get LLM embedding
    with torch.no_grad():
        # In production: text_embedding = llm_model.encode(news_text)
        text_embedding = torch.randn(1, 768)  # Mock embedding

    # Get market features
    market_features = torch.tensor(
        prepare_market_features(symbol, None),
        dtype=torch.float32
    ).unsqueeze(0)

    # Predict impact
    with torch.no_grad():
        prediction = impact_predictor(text_embedding, market_features)

    return {
        "symbol": symbol,
        "expected_direction": prediction["direction"].item(),
        "expected_magnitude_pct": prediction["magnitude"].item() * 100,
        "confidence": prediction["confidence"].item(),
        "interpretation": interpret_prediction(prediction)
    }


def interpret_prediction(pred: Dict) -> str:
    """Convert prediction to human-readable interpretation."""
    direction = pred["direction"].item()
    magnitude = pred["magnitude"].item() * 100
    confidence = pred["confidence"].item()

    if confidence < 0.5:
        return "Low confidence - uncertain impact"

    dir_str = "positive" if direction > 0.1 else "negative" if direction < -0.1 else "neutral"
    mag_str = "significant" if magnitude > 2 else "moderate" if magnitude > 0.5 else "minor"

    return f"Expected {mag_str} {dir_str} impact ({magnitude:.1f}% move, {confidence:.0%} confidence)"
```

### 04: Backtesting LLM Signals

```python
# python/04_backtest.py

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    """Configuration for LLM signal backtesting."""
    initial_capital: float = 100000
    max_position_size: float = 0.1  # Max 10% per position
    transaction_cost_bps: float = 10  # 10 basis points
    slippage_bps: float = 5
    signal_decay_hours: float = 24
    rebalance_frequency: str = "daily"  # "hourly", "daily"

@dataclass
class BacktestResult:
    """Results from backtesting LLM signals."""
    returns: pd.Series
    positions: pd.DataFrame
    trades: List[Dict]
    metrics: Dict[str, float]

class LLMSignalBacktester:
    """
    Backtest trading signals generated from LLM analysis.

    This backtester specifically handles the unique characteristics
    of LLM-derived signals:
    - Irregular signal timing (news-driven)
    - Signal decay over time
    - Varying confidence levels
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on LLM signals.

        Args:
            signals: DataFrame with columns [timestamp, symbol, signal, confidence]
            prices: DataFrame with OHLCV data, indexed by timestamp
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Filter date range
        if start_date:
            signals = signals[signals['timestamp'] >= start_date]
            prices = prices[prices.index >= start_date]
        if end_date:
            signals = signals[signals['timestamp'] <= end_date]
            prices = prices[prices.index <= end_date]

        # Initialize tracking
        capital = self.config.initial_capital
        positions: Dict[str, float] = {}  # symbol -> shares
        position_history = []
        trades = []
        portfolio_values = []

        # Get rebalance points
        if self.config.rebalance_frequency == "daily":
            rebalance_points = prices.index.normalize().unique()
        else:
            rebalance_points = prices.index

        for ts in rebalance_points:
            # Get active signals (with decay)
            active_signals = self._get_active_signals(signals, ts)

            # Calculate target positions
            target_positions = self._calculate_positions(
                active_signals,
                prices.loc[ts] if ts in prices.index else prices.iloc[-1],
                capital
            )

            # Execute rebalance
            trades_executed, capital = self._execute_rebalance(
                positions,
                target_positions,
                prices.loc[ts] if ts in prices.index else prices.iloc[-1],
                capital,
                ts
            )
            trades.extend(trades_executed)

            # Update positions
            positions = target_positions.copy()

            # Calculate portfolio value
            portfolio_value = capital + sum(
                shares * prices.loc[ts, symbol] if symbol in prices.columns else 0
                for symbol, shares in positions.items()
            )
            portfolio_values.append({"timestamp": ts, "value": portfolio_value})
            position_history.append({"timestamp": ts, **positions})

        # Calculate returns
        pv_df = pd.DataFrame(portfolio_values).set_index('timestamp')
        returns = pv_df['value'].pct_change().dropna()

        # Calculate metrics
        metrics = self._calculate_metrics(returns, trades)

        return BacktestResult(
            returns=returns,
            positions=pd.DataFrame(position_history),
            trades=trades,
            metrics=metrics
        )

    def _get_active_signals(
        self,
        signals: pd.DataFrame,
        current_time: datetime
    ) -> pd.DataFrame:
        """Get signals that are still active (with decay applied)."""
        decay_hours = self.config.signal_decay_hours

        # Filter to signals within decay window
        cutoff = current_time - timedelta(hours=decay_hours)
        active = signals[
            (signals['timestamp'] >= cutoff) &
            (signals['timestamp'] <= current_time)
        ].copy()

        if active.empty:
            return active

        # Apply time decay to signal strength
        active['hours_ago'] = (current_time - active['timestamp']).dt.total_seconds() / 3600
        active['decay_factor'] = np.exp(-active['hours_ago'] / decay_hours)
        active['adjusted_signal'] = active['signal'] * active['confidence'] * active['decay_factor']

        # Aggregate by symbol (latest signal wins with decay)
        aggregated = active.groupby('symbol').agg({
            'adjusted_signal': 'sum',
            'confidence': 'mean'
        }).reset_index()

        return aggregated

    def _calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        capital: float
    ) -> Dict[str, float]:
        """Calculate target positions from signals."""
        if signals.empty:
            return {}

        positions = {}
        max_position_value = capital * self.config.max_position_size

        for _, row in signals.iterrows():
            symbol = row['symbol']
            if symbol not in prices.index:
                continue

            signal_strength = row['adjusted_signal']
            price = prices[symbol]

            # Position size based on signal strength
            position_value = signal_strength * max_position_value
            shares = position_value / price

            positions[symbol] = shares

        return positions

    def _execute_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        prices: pd.Series,
        capital: float,
        timestamp: datetime
    ) -> Tuple[List[Dict], float]:
        """Execute rebalance trades."""
        trades = []

        all_symbols = set(current.keys()) | set(target.keys())

        for symbol in all_symbols:
            current_shares = current.get(symbol, 0)
            target_shares = target.get(symbol, 0)

            if symbol not in prices.index:
                continue

            delta = target_shares - current_shares
            if abs(delta) < 0.01:  # Skip tiny trades
                continue

            price = prices[symbol]

            # Apply transaction costs and slippage
            cost_factor = 1 + (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
            if delta > 0:  # Buy
                trade_value = delta * price * cost_factor
            else:  # Sell
                trade_value = delta * price / cost_factor

            capital -= trade_value

            trades.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "shares": delta,
                "price": price,
                "value": trade_value,
                "type": "BUY" if delta > 0 else "SELL"
            })

        return trades, capital

    def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        if returns.empty:
            return {}

        # Annualization factor (assuming daily returns)
        ann_factor = 252

        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)

        sharpe = ann_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Trade statistics
        n_trades = len(trades)
        if trades:
            winning_trades = sum(1 for t in trades if t['value'] > 0)
            win_rate = winning_trades / n_trades if n_trades > 0 else 0
        else:
            win_rate = 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": n_trades,
            "win_rate": win_rate
        }


# Example usage
def run_example_backtest():
    """Run example backtest with synthetic data."""
    config = BacktestConfig(
        initial_capital=100000,
        max_position_size=0.1,
        signal_decay_hours=48
    )

    backtester = LLMSignalBacktester(config)

    # Generate synthetic signals and prices
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    # Mock price data
    np.random.seed(42)
    prices = pd.DataFrame({
        "AAPL": 150 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
        "MSFT": 350 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
        "GOOGL": 140 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
    }, index=dates)

    # Mock LLM signals (random for demo)
    signal_dates = np.random.choice(dates, size=50, replace=False)
    signals = pd.DataFrame({
        "timestamp": signal_dates,
        "symbol": np.random.choice(["AAPL", "MSFT", "GOOGL"], size=50),
        "signal": np.random.uniform(-1, 1, size=50),
        "confidence": np.random.uniform(0.5, 1.0, size=50)
    })

    # Run backtest
    result = backtester.run_backtest(signals, prices)

    print("Backtest Results:")
    print(f"  Total Return: {result.metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {result.metrics['num_trades']}")

    return result


if __name__ == "__main__":
    run_example_backtest()
```

## Rust Implementation

Since BloombergGPT is not publicly available, we implement a BloombergGPT-style financial LLM wrapper in Rust that can work with open-source alternatives.

```
rust_bloomberggpt/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # External API clients
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit crypto data
│   │   └── yahoo.rs        # Yahoo Finance data
│   ├── llm/                # LLM interface
│   │   ├── mod.rs
│   │   ├── client.rs       # LLM API client
│   │   ├── prompts.rs      # Financial prompts
│   │   └── embeddings.rs   # Text embeddings
│   ├── analysis/           # Financial analysis
│   │   ├── mod.rs
│   │   ├── sentiment.rs    # Sentiment analysis
│   │   ├── ner.rs          # Named entity recognition
│   │   └── qa.rs           # Question answering
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── sentiment_analysis.rs
    ├── generate_signals.rs
    └── backtest.rs
```

See [rust_bloomberggpt](rust_bloomberggpt/) for complete Rust implementation.

### Quick Start (Rust)

```bash
cd rust_bloomberggpt

# Run sentiment analysis example
cargo run --example sentiment_analysis

# Generate trading signals from news
cargo run --example generate_signals -- --symbol BTCUSDT

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── sentiment_analysis.py   # Financial sentiment
├── trading_signals.py      # Signal generation
├── news_impact.py          # Impact prediction
├── backtest.py             # Backtesting
├── data_loader.py          # Data loading utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_sentiment_demo.py
    ├── 02_signal_generation.py
    ├── 03_impact_prediction.py
    └── 04_full_backtest.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run sentiment analysis
python examples/01_sentiment_demo.py

# Generate signals
python examples/02_signal_generation.py --symbol AAPL

# Run backtest
python examples/04_full_backtest.py --capital 100000
```

## Best Practices

### When to Use Financial LLMs for Trading

**Good use cases:**
- Sentiment analysis on earnings calls and news
- Event-driven trading signals
- Entity extraction from financial documents
- Summarizing SEC filings
- Classifying news by impact

**Not ideal for:**
- High-frequency trading (latency too high)
- Pure price prediction (use quantitative models)
- Replacing fundamental analysis entirely

### Signal Generation Guidelines

1. **Confidence Filtering**
   ```python
   # Only act on high-confidence signals
   if signal.confidence < 0.7:
       continue  # Skip low-confidence signals
   ```

2. **Signal Decay**
   ```python
   # News impact decays over time
   signal_strength *= np.exp(-hours_since_news / 24)
   ```

3. **Source Weighting**
   ```python
   source_weights = {
       "earnings": 1.0,  # Highest impact
       "sec_filing": 0.8,
       "news": 0.6,
       "social": 0.3  # Noisy, lower weight
   }
   ```

4. **Position Sizing**
   ```python
   # Scale position by confidence
   position_size = base_size * confidence * signal_strength
   ```

### Common Pitfalls

1. **Overfitting to sentiment** - Don't trade purely on sentiment; combine with price/volume
2. **Latency issues** - LLM inference is slow; not suitable for HFT
3. **Hallucination risk** - Always verify entity extraction with database lookup
4. **Cost management** - LLM API calls are expensive; batch when possible

## Resources

### Papers

- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) — Original BloombergGPT paper (2023)
- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063) — FinBERT paper
- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) — Open-source financial LLM
- [Large Language Models in Finance: A Survey](https://arxiv.org/abs/2311.10723) — Comprehensive survey

### Open-Source Alternatives

Since BloombergGPT is proprietary, consider these alternatives:

| Model | Size | Availability | Best For |
|-------|------|--------------|----------|
| [FinBERT](https://huggingface.co/ProsusAI/finbert) | 110M | Open | Sentiment analysis |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Various | Open | General financial NLP |
| [FinMA](https://huggingface.co/ChanceFocus/finma-7b-nlp) | 7B | Open | Financial tasks |
| GPT-4 | ~1.7T | API | General + financial |

### Related Chapters

- [Chapter 61: FinGPT Financial LLM](../61_fingpt_financial_llm) — Open-source alternative
- [Chapter 67: LLM Sentiment Analysis](../67_llm_sentiment_analysis) — Deep dive on sentiment
- [Chapter 241: FinBERT Sentiment](../241_finbert_sentiment) — Smaller, faster model
- [Chapter 37: Sentiment Momentum Fusion](../37_sentiment_momentum_fusion) — Combining signals

---

## Difficulty Level

**Advanced**

Prerequisites:
- Understanding of transformer architecture and LLMs
- Financial markets knowledge (sentiment, trading signals)
- Python/Rust programming experience
- Experience with NLP tasks (sentiment analysis, NER)

## References

1. Wu, S., et al. (2023). "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564
2. Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models."
3. Yang, H., et al. (2023). "FinGPT: Open-Source Financial Large Language Models."
4. Liu, X., et al. (2023). "Large Language Models in Finance: A Survey."
