# Chapter 77: LLM Regime Classification for Financial Markets

This chapter explores **Large Language Model (LLM)-based market regime classification** for trading and investment strategies. We demonstrate how LLMs can identify different market conditions (bull, bear, sideways, volatile) using both numerical data and textual information from news, social media, and economic indicators.

<p align="center">
<img src="https://i.imgur.com/8XdE3Wz.png" width="70%">
</p>

## Contents

1. [Introduction to Market Regime Classification](#introduction-to-market-regime-classification)
    * [Why Regime Classification Matters](#why-regime-classification-matters)
    * [Traditional vs LLM-based Approaches](#traditional-vs-llm-based-approaches)
    * [Market Regimes Defined](#market-regimes-defined)
2. [Theoretical Foundation](#theoretical-foundation)
    * [Hidden Markov Models Baseline](#hidden-markov-models-baseline)
    * [LLM Representation Learning for Regimes](#llm-representation-learning-for-regimes)
    * [Multi-modal Regime Detection](#multi-modal-regime-detection)
3. [Classification Methods](#classification-methods)
    * [Text-based Regime Classification](#text-based-regime-classification)
    * [Time Series Regime Detection](#time-series-regime-detection)
    * [Hybrid LLM-Statistical Approaches](#hybrid-llm-statistical-approaches)
4. [Practical Examples](#practical-examples)
    * [01: Stock Market Regime Classification](#01-stock-market-regime-classification)
    * [02: Crypto Market Regimes (Bybit)](#02-crypto-market-regimes-bybit)
    * [03: Multi-asset Regime Alignment](#03-multi-asset-regime-alignment)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Backtesting Framework](#backtesting-framework)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Market Regime Classification

Market regime classification is the task of identifying the current state or "regime" of financial markets. Different regimes require different trading strategies - what works in a bull market may fail dramatically in a bear market. LLMs offer a powerful approach to this problem by leveraging their ability to understand context from multiple data sources.

### Why Regime Classification Matters

```
REGIME-AWARE TRADING IMPORTANCE:
======================================================================

+------------------------------------------------------------------+
|  WITHOUT REGIME AWARENESS:                                        |
|                                                                    |
|  Strategy: Buy when RSI < 30                                       |
|  Bull Market: Works great! Buy the dips, prices recover            |
|  Bear Market: DISASTER! Keep buying as prices fall further         |
|                                                                    |
|  Result: Strategy works sometimes, fails catastrophically others   |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|  WITH REGIME AWARENESS:                                           |
|                                                                    |
|  Bull Regime: Use momentum strategies, buy dips                    |
|  Bear Regime: Use short strategies, hedge positions                |
|  Sideways Regime: Use mean-reversion, sell options                 |
|  Volatile Regime: Reduce position sizes, use stops                 |
|                                                                    |
|  Result: Adaptive strategy that performs well in all conditions    |
+------------------------------------------------------------------+
```

### Traditional vs LLM-based Approaches

| Aspect | Traditional Methods | LLM-based Classification |
|--------|---------------------|--------------------------|
| Data Types | Numerical only (prices, volatility) | Text, numerical, multi-modal |
| Context | Technical indicators | Economic narratives, news sentiment |
| Regime Types | Predefined (2-4 states) | Dynamic, can describe novel regimes |
| Transition Detection | Statistical (HMM) | Semantic understanding |
| Explanation | None | Natural language descriptions |
| Adaptation | Requires retraining | Prompt-based adaptation |

### Market Regimes Defined

```
COMMON MARKET REGIMES:
======================================================================

1. BULL REGIME
   +----------------------------------------------------------------+
   | Characteristics:                                                 |
   | - Prices trending upward                                         |
   | - Positive sentiment and optimism                                |
   | - Low volatility, steady gains                                   |
   | - "Risk-on" behavior                                             |
   |                                                                  |
   | Best Strategies: Momentum, trend-following, buy-and-hold         |
   +----------------------------------------------------------------+

2. BEAR REGIME
   +----------------------------------------------------------------+
   | Characteristics:                                                 |
   | - Prices trending downward                                       |
   | - Negative sentiment and fear                                    |
   | - Often higher volatility                                        |
   | - "Risk-off" behavior                                            |
   |                                                                  |
   | Best Strategies: Short-selling, hedging, cash positions          |
   +----------------------------------------------------------------+

3. SIDEWAYS/RANGING REGIME
   +----------------------------------------------------------------+
   | Characteristics:                                                 |
   | - Prices moving within a range                                   |
   | - Mixed sentiment, indecision                                    |
   | - Low to moderate volatility                                     |
   | - Mean-reverting behavior                                        |
   |                                                                  |
   | Best Strategies: Range trading, sell options, mean-reversion     |
   +----------------------------------------------------------------+

4. HIGH VOLATILITY REGIME
   +----------------------------------------------------------------+
   | Characteristics:                                                 |
   | - Large price swings in both directions                          |
   | - Uncertainty and mixed news                                     |
   | - Elevated VIX (fear index)                                      |
   | - Trend reversals common                                         |
   |                                                                  |
   | Best Strategies: Reduced position size, volatility trading       |
   +----------------------------------------------------------------+

5. CRISIS/CRASH REGIME
   +----------------------------------------------------------------+
   | Characteristics:                                                 |
   | - Rapid price declines                                           |
   | - Extreme fear and panic                                         |
   | - Very high volatility                                           |
   | - Liquidity issues, correlation spikes                           |
   |                                                                  |
   | Best Strategies: Capital preservation, tail-risk hedging         |
   +----------------------------------------------------------------+
```

## Theoretical Foundation

### Hidden Markov Models Baseline

Traditional regime detection uses Hidden Markov Models (HMMs) to model regime transitions:

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    """Result of regime classification."""
    regime: MarketRegime
    probability: float
    confidence: float
    explanation: str
    supporting_factors: List[str]


class HMMRegimeDetector:
    """
    Hidden Markov Model baseline for regime detection.

    Uses returns and volatility to classify market regimes
    using a statistical approach.
    """

    def __init__(self, n_regimes: int = 4, lookback: int = 60):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            lookback: Lookback period for volatility calculation
        """
        self.n_regimes = n_regimes
        self.lookback = lookback

        # Regime parameters (mean return, volatility)
        self.regime_params = {
            MarketRegime.BULL: {'mean': 0.001, 'vol': 0.01},
            MarketRegime.BEAR: {'mean': -0.001, 'vol': 0.02},
            MarketRegime.SIDEWAYS: {'mean': 0.0, 'vol': 0.008},
            MarketRegime.HIGH_VOLATILITY: {'mean': 0.0, 'vol': 0.03}
        }

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None
    ) -> RegimeResult:
        """
        Detect current market regime based on recent data.

        Args:
            returns: Array of recent returns
            volatility: Optional volatility array (calculated if not provided)

        Returns:
            RegimeResult with classification
        """
        if len(returns) < self.lookback:
            returns = np.pad(returns, (self.lookback - len(returns), 0), 'constant')

        recent_returns = returns[-self.lookback:]
        mean_return = np.mean(recent_returns)

        if volatility is None:
            volatility = np.std(recent_returns) * np.sqrt(252)
        else:
            volatility = np.mean(volatility[-self.lookback:])

        # Simple rule-based classification
        # In production, use proper HMM with EM algorithm
        if volatility > 0.35:  # Annualized vol > 35%
            regime = MarketRegime.HIGH_VOLATILITY
            prob = min(0.95, volatility / 0.5)
        elif mean_return > 0.0005 and volatility < 0.20:
            regime = MarketRegime.BULL
            prob = min(0.9, mean_return / 0.002 + 0.5)
        elif mean_return < -0.0005:
            regime = MarketRegime.BEAR
            prob = min(0.9, abs(mean_return) / 0.002 + 0.5)
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.7

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=0.7,  # HMM baseline has moderate confidence
            explanation=f"Statistical detection: mean return={mean_return:.4f}, volatility={volatility:.2%}",
            supporting_factors=[
                f"Mean return: {mean_return:.4f}",
                f"Volatility: {volatility:.2%}",
                f"Lookback: {self.lookback} periods"
            ]
        )


# Example usage
detector = HMMRegimeDetector()
returns = np.random.randn(100) * 0.02  # Mock returns
result = detector.detect_regime(returns)

print(f"Regime: {result.regime.value}")
print(f"Probability: {result.probability:.2%}")
print(f"Explanation: {result.explanation}")
```

### LLM Representation Learning for Regimes

LLMs can learn rich representations that capture regime characteristics:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np


class LLMRegimeEncoder(nn.Module):
    """
    Encode market data (text + numerical) into regime-aware
    representations using LLM backbone.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        numerical_features: int = 20,
        embedding_dim: int = 256,
        num_regimes: int = 5
    ):
        super().__init__()

        self.num_regimes = num_regimes

        # Text encoder (LLM backbone)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Numerical encoder for price/volume data
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # Time series encoder (for OHLCV sequences)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.ts_proj = nn.Linear(5, 64)  # OHLCV -> d_model

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + 128 + 64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_regimes)
        )

        # Regime embedding for similarity computation
        self.regime_embeddings = nn.Parameter(
            torch.randn(num_regimes, embedding_dim)
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text inputs using LLM."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use CLS token
            text_embeddings = outputs.last_hidden_state[:, 0]

        return text_embeddings

    def encode_timeseries(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Encode OHLCV time series.

        Args:
            ohlcv: Tensor of shape (batch, seq_len, 5)

        Returns:
            Encoded tensor of shape (batch, 64)
        """
        x = self.ts_proj(ohlcv)
        encoded = self.ts_encoder(x)
        # Global average pooling
        return encoded.mean(dim=1)

    def forward(
        self,
        texts: List[str],
        numerical_features: torch.Tensor,
        ohlcv: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for regime classification.

        Args:
            texts: Market context descriptions
            numerical_features: Technical indicators (batch, num_features)
            ohlcv: Optional OHLCV data (batch, seq_len, 5)

        Returns:
            Dictionary with embeddings and regime probabilities
        """
        batch_size = numerical_features.size(0)

        # Encode each modality
        text_emb = self.encode_text(texts)
        num_emb = self.numerical_encoder(numerical_features)

        if ohlcv is not None:
            ts_emb = self.encode_timeseries(ohlcv)
        else:
            ts_emb = torch.zeros(batch_size, 64)

        # Fuse modalities
        combined = torch.cat([text_emb, num_emb, ts_emb], dim=-1)
        embeddings = self.fusion(combined)

        # Classify regime
        logits = self.regime_classifier(embeddings)
        probabilities = torch.softmax(logits, dim=-1)

        # Compute similarity to regime prototypes
        regime_similarities = torch.matmul(
            nn.functional.normalize(embeddings, dim=-1),
            nn.functional.normalize(self.regime_embeddings, dim=-1).T
        )

        return {
            'embeddings': embeddings,
            'logits': logits,
            'probabilities': probabilities,
            'regime_similarities': regime_similarities
        }

    def classify(
        self,
        texts: List[str],
        numerical_features: torch.Tensor,
        ohlcv: Optional[torch.Tensor] = None
    ) -> List[RegimeResult]:
        """
        Classify market regime with explanations.

        Returns:
            List of RegimeResult for each sample
        """
        self.eval()
        regime_names = [
            MarketRegime.BULL,
            MarketRegime.BEAR,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.CRISIS
        ]

        with torch.no_grad():
            outputs = self(texts, numerical_features, ohlcv)
            probs = outputs['probabilities']

        results = []
        for i in range(len(texts)):
            regime_idx = probs[i].argmax().item()
            regime = regime_names[regime_idx]
            prob = probs[i][regime_idx].item()

            # Get top factors
            top_probs = probs[i].topk(3)
            factors = [
                f"{regime_names[idx.item()].value}: {p.item():.1%}"
                for idx, p in zip(top_probs.indices, top_probs.values)
            ]

            results.append(RegimeResult(
                regime=regime,
                probability=prob,
                confidence=prob * 0.9 + 0.1,  # Calibrated confidence
                explanation=f"LLM classification based on market context: {texts[i][:100]}...",
                supporting_factors=factors
            ))

        return results


# Example usage
encoder = LLMRegimeEncoder()

texts = [
    "Markets rallied today on strong earnings reports. Tech stocks led gains as investor sentiment improved.",
    "Stocks plummeted amid recession fears. The Fed signaled more rate hikes ahead as inflation persists.",
    "Markets traded in a narrow range today. Traders await next week's economic data for direction."
]

# Mock numerical features (technical indicators)
numerical = torch.randn(3, 20)

# Mock OHLCV data
ohlcv = torch.randn(3, 60, 5)  # 60 days of OHLCV

results = encoder.classify(texts, numerical, ohlcv)

for text, result in zip(texts, results):
    print(f"\nText: {text[:60]}...")
    print(f"Regime: {result.regime.value}")
    print(f"Probability: {result.probability:.1%}")
    print(f"Factors: {result.supporting_factors}")
```

### Multi-modal Regime Detection

Combining multiple data sources for robust regime classification:

```python
class MultiModalRegimeClassifier:
    """
    Multi-modal regime classifier combining:
    - Price and volume data
    - News sentiment
    - Economic indicators
    - Social media signals
    """

    def __init__(self):
        self.regime_descriptions = {
            MarketRegime.BULL: {
                'price_signal': 'Positive returns, higher highs',
                'sentiment_signal': 'Optimistic news, positive earnings',
                'volatility_signal': 'Low to moderate volatility',
                'economic_signal': 'Strong GDP, low unemployment'
            },
            MarketRegime.BEAR: {
                'price_signal': 'Negative returns, lower lows',
                'sentiment_signal': 'Pessimistic news, earnings misses',
                'volatility_signal': 'Elevated volatility',
                'economic_signal': 'Weak GDP, rising unemployment'
            },
            MarketRegime.SIDEWAYS: {
                'price_signal': 'Range-bound prices',
                'sentiment_signal': 'Mixed or neutral news',
                'volatility_signal': 'Low volatility',
                'economic_signal': 'Stable economic conditions'
            },
            MarketRegime.HIGH_VOLATILITY: {
                'price_signal': 'Large swings both directions',
                'sentiment_signal': 'Conflicting news signals',
                'volatility_signal': 'VIX elevated (>25)',
                'economic_signal': 'Uncertain economic outlook'
            },
            MarketRegime.CRISIS: {
                'price_signal': 'Rapid decline, gap downs',
                'sentiment_signal': 'Panic, extreme fear',
                'volatility_signal': 'VIX spike (>40)',
                'economic_signal': 'Economic shock event'
            }
        }

    def classify_from_features(
        self,
        price_features: Dict[str, float],
        sentiment_features: Dict[str, float],
        volatility_features: Dict[str, float],
        economic_features: Optional[Dict[str, float]] = None
    ) -> RegimeResult:
        """
        Classify regime from multi-modal features.

        Args:
            price_features: Return, trend, momentum indicators
            sentiment_features: News sentiment, social sentiment
            volatility_features: VIX, realized vol, implied vol
            economic_features: GDP, unemployment, etc.

        Returns:
            RegimeResult with classification
        """
        # Score each regime based on features
        regime_scores = {}

        for regime in MarketRegime:
            score = 0.0
            factors = []

            # Price score
            returns = price_features.get('returns', 0)
            if regime == MarketRegime.BULL and returns > 0.001:
                score += returns * 100
                factors.append(f"Positive returns: {returns:.2%}")
            elif regime == MarketRegime.BEAR and returns < -0.001:
                score += abs(returns) * 100
                factors.append(f"Negative returns: {returns:.2%}")
            elif regime == MarketRegime.SIDEWAYS and abs(returns) < 0.001:
                score += 0.3
                factors.append("Range-bound prices")

            # Volatility score
            vix = volatility_features.get('vix', 15)
            if regime == MarketRegime.HIGH_VOLATILITY and vix > 25:
                score += (vix - 25) / 25
                factors.append(f"VIX elevated: {vix:.1f}")
            elif regime == MarketRegime.CRISIS and vix > 40:
                score += (vix - 40) / 20
                factors.append(f"VIX spike: {vix:.1f}")
            elif regime in [MarketRegime.BULL, MarketRegime.SIDEWAYS] and vix < 20:
                score += 0.2
                factors.append(f"Low VIX: {vix:.1f}")

            # Sentiment score
            sentiment = sentiment_features.get('sentiment', 0)
            if regime == MarketRegime.BULL and sentiment > 0.2:
                score += sentiment
                factors.append(f"Positive sentiment: {sentiment:.2f}")
            elif regime == MarketRegime.BEAR and sentiment < -0.2:
                score += abs(sentiment)
                factors.append(f"Negative sentiment: {sentiment:.2f}")
            elif regime == MarketRegime.CRISIS and sentiment < -0.5:
                score += abs(sentiment) * 1.5
                factors.append(f"Extreme fear: {sentiment:.2f}")

            regime_scores[regime] = {'score': score, 'factors': factors}

        # Find best regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1]['score'])
        regime = best_regime[0]
        score = best_regime[1]['score']
        factors = best_regime[1]['factors']

        # Normalize probability
        total_score = sum(r['score'] for r in regime_scores.values()) + 1e-8
        probability = score / total_score

        return RegimeResult(
            regime=regime,
            probability=probability,
            confidence=min(0.95, probability + 0.1),
            explanation=f"Multi-modal classification based on {len(factors)} signals",
            supporting_factors=factors if factors else ["Default classification"]
        )


# Example usage
classifier = MultiModalRegimeClassifier()

# Bullish market example
result = classifier.classify_from_features(
    price_features={'returns': 0.02, 'trend': 1},
    sentiment_features={'sentiment': 0.4, 'news_score': 0.3},
    volatility_features={'vix': 14, 'realized_vol': 0.12}
)

print(f"Regime: {result.regime.value}")
print(f"Probability: {result.probability:.1%}")
print(f"Factors: {result.supporting_factors}")
```

## Classification Methods

### Text-based Regime Classification

Use news and social media to classify market regimes:

```python
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class TextRegimeClassifier:
    """
    Classify market regime from textual data.

    Uses LLM embeddings and financial sentiment analysis
    to detect market conditions.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize text-based regime classifier.

        Args:
            model_name: Pre-trained model for text encoding
        """
        self.model_name = model_name

        # Regime keywords for zero-shot classification
        self.regime_keywords = {
            MarketRegime.BULL: [
                'rally', 'surge', 'gains', 'bullish', 'optimism',
                'record high', 'breakout', 'momentum', 'buying',
                'strong earnings', 'growth', 'expansion'
            ],
            MarketRegime.BEAR: [
                'plunge', 'crash', 'bearish', 'selloff', 'decline',
                'losses', 'fear', 'correction', 'downturn',
                'recession', 'weak earnings', 'contraction'
            ],
            MarketRegime.SIDEWAYS: [
                'consolidation', 'range-bound', 'flat', 'stable',
                'unchanged', 'mixed', 'neutral', 'waiting',
                'indecisive', 'sideways'
            ],
            MarketRegime.HIGH_VOLATILITY: [
                'volatile', 'swings', 'uncertainty', 'turbulent',
                'whipsaw', 'erratic', 'unpredictable', 'vix spike',
                'risk-off', 'hedge'
            ],
            MarketRegime.CRISIS: [
                'panic', 'crash', 'crisis', 'collapse', 'meltdown',
                'black swan', 'contagion', 'systemic', 'emergency',
                'circuit breaker', 'flash crash'
            ]
        }

    def classify_text(
        self,
        texts: List[str],
        return_scores: bool = False
    ) -> List[RegimeResult]:
        """
        Classify regime based on text content.

        Args:
            texts: List of news/social media texts
            return_scores: Whether to return raw scores

        Returns:
            List of RegimeResult classifications
        """
        results = []

        for text in texts:
            text_lower = text.lower()
            regime_scores = {}

            for regime, keywords in self.regime_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                # Normalize by number of keywords
                regime_scores[regime] = score / len(keywords)

            # Get best regime
            if max(regime_scores.values()) == 0:
                best_regime = MarketRegime.SIDEWAYS
                prob = 0.5
            else:
                best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                total = sum(regime_scores.values()) + 1e-8
                prob = regime_scores[best_regime] / total

            # Extract supporting evidence
            matched_keywords = [
                kw for kw in self.regime_keywords[best_regime]
                if kw in text_lower
            ]

            results.append(RegimeResult(
                regime=best_regime,
                probability=prob,
                confidence=min(0.9, prob + 0.2),
                explanation=f"Text-based classification from news/sentiment",
                supporting_factors=[
                    f"Matched keywords: {matched_keywords[:3]}",
                    f"Text snippet: {text[:100]}..."
                ]
            ))

        return results

    def aggregate_regime(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> RegimeResult:
        """
        Aggregate multiple texts into single regime classification.

        Args:
            texts: List of recent market texts
            weights: Optional weights for each text (e.g., by recency)

        Returns:
            Aggregated RegimeResult
        """
        if weights is None:
            weights = [1.0] * len(texts)

        results = self.classify_text(texts)

        # Aggregate scores
        regime_votes = {r: 0.0 for r in MarketRegime}

        for result, weight in zip(results, weights):
            regime_votes[result.regime] += weight * result.probability

        total = sum(regime_votes.values()) + 1e-8
        regime_probs = {r: v / total for r, v in regime_votes.items()}

        best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]

        return RegimeResult(
            regime=best_regime,
            probability=regime_probs[best_regime],
            confidence=min(0.95, regime_probs[best_regime] + 0.1),
            explanation=f"Aggregated from {len(texts)} text sources",
            supporting_factors=[
                f"{r.value}: {p:.1%}" for r, p in sorted(
                    regime_probs.items(), key=lambda x: -x[1]
                )[:3]
            ]
        )


# Example usage
classifier = TextRegimeClassifier()

# Recent news headlines
headlines = [
    "Stocks rally to new highs as earnings beat expectations across sectors",
    "Tech giants surge on AI optimism, Nasdaq up 2%",
    "Investors bullish on growth prospects as Fed signals pause",
    "Strong jobs report fuels market rally, S&P 500 hits record"
]

result = classifier.aggregate_regime(headlines)
print(f"Aggregated Regime: {result.regime.value}")
print(f"Probability: {result.probability:.1%}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Factors: {result.supporting_factors}")
```

### Time Series Regime Detection

Detect regimes from price and volume patterns:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class TransformerRegimeDetector(nn.Module):
    """
    Transformer-based regime detector for financial time series.

    Uses attention mechanisms to identify regime-specific patterns
    in OHLCV data.
    """

    def __init__(
        self,
        input_dim: int = 5,  # OHLCV
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        num_regimes: int = 5,
        seq_length: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_regimes = num_regimes

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_length, d_model) * 0.1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Regime classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regimes)
        )

        # Regime-aware attention weights (for interpretability)
        self.regime_attention = nn.Parameter(
            torch.randn(num_regimes, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for regime classification.

        Args:
            x: Input tensor of shape (batch, seq_length, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (regime_logits, attention_weights)
        """
        # Project to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]

        # Transformer encoding
        encoded = self.transformer(x)

        # Global pooling (use last token or mean)
        pooled = encoded.mean(dim=1)

        # Classify regime
        logits = self.classifier(pooled)

        if return_attention:
            # Compute regime-specific attention
            attention = torch.matmul(
                encoded,
                self.regime_attention.T
            )  # (batch, seq, num_regimes)
            attention = torch.softmax(attention, dim=1)
            return logits, attention

        return logits, None

    def predict(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict regime with probabilities.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted_regimes, probabilities)
        """
        logits, _ = self(x)
        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        return predictions, probs


class TimeSeriesRegimeClassifier:
    """
    Complete time series regime classification system.
    """

    def __init__(
        self,
        seq_length: int = 60,
        num_regimes: int = 5
    ):
        self.seq_length = seq_length
        self.num_regimes = num_regimes

        self.model = TransformerRegimeDetector(
            seq_length=seq_length,
            num_regimes=num_regimes
        )

        self.regime_names = [
            MarketRegime.BULL,
            MarketRegime.BEAR,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.CRISIS
        ]

        self.is_trained = False

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3
    ):
        """
        Train the regime classifier.

        Args:
            data: OHLCV data (num_samples, seq_length, 5)
            labels: Regime labels (num_samples,)
            epochs: Training epochs
            lr: Learning rate
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        tensor_data = torch.FloatTensor(data)
        tensor_labels = torch.LongTensor(labels)

        for epoch in range(epochs):
            optimizer.zero_grad()

            logits, _ = self.model(tensor_data)
            loss = criterion(logits, tensor_labels)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # Calculate accuracy
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == tensor_labels).float().mean()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.2%}")

        self.is_trained = True

    def classify(
        self,
        data: np.ndarray
    ) -> List[RegimeResult]:
        """
        Classify market regimes.

        Args:
            data: OHLCV data (num_samples, seq_length, 5)

        Returns:
            List of RegimeResult classifications
        """
        self.model.eval()
        tensor_data = torch.FloatTensor(data)

        with torch.no_grad():
            logits, attention = self.model(tensor_data, return_attention=True)
            probs = torch.softmax(logits, dim=-1)

        results = []
        for i in range(len(data)):
            regime_idx = probs[i].argmax().item()
            regime = self.regime_names[regime_idx]
            prob = probs[i][regime_idx].item()

            # Get top 3 regimes
            top3 = probs[i].topk(3)
            factors = [
                f"{self.regime_names[idx.item()].value}: {p.item():.1%}"
                for idx, p in zip(top3.indices, top3.values)
            ]

            results.append(RegimeResult(
                regime=regime,
                probability=prob,
                confidence=min(0.95, prob + 0.05),
                explanation=f"Time series classification from {self.seq_length} periods",
                supporting_factors=factors
            ))

        return results


# Example usage
def generate_regime_data(n_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regime-labeled data."""
    np.random.seed(42)

    data = np.zeros((n_samples, seq_length, 5))
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        regime = np.random.randint(0, 5)
        labels[i] = regime

        # Generate regime-specific patterns
        if regime == 0:  # Bull
            trend = np.linspace(0, 0.1, seq_length)
            vol = 0.01
        elif regime == 1:  # Bear
            trend = np.linspace(0, -0.1, seq_length)
            vol = 0.015
        elif regime == 2:  # Sideways
            trend = np.zeros(seq_length)
            vol = 0.008
        elif regime == 3:  # High vol
            trend = np.zeros(seq_length)
            vol = 0.03
        else:  # Crisis
            trend = np.linspace(0, -0.2, seq_length)
            vol = 0.05

        # Generate OHLCV
        noise = np.random.randn(seq_length) * vol
        prices = 100 * np.exp(trend + np.cumsum(noise))

        data[i, :, 0] = prices * (1 + np.random.randn(seq_length) * 0.005)  # Open
        data[i, :, 1] = prices * (1 + np.abs(np.random.randn(seq_length) * 0.01))  # High
        data[i, :, 2] = prices * (1 - np.abs(np.random.randn(seq_length) * 0.01))  # Low
        data[i, :, 3] = prices  # Close
        data[i, :, 4] = np.random.exponential(1000, seq_length) * (1 + vol * 10)  # Volume

    return data, labels


# Generate and train
data, labels = generate_regime_data(500, 60)
print(f"Generated {len(data)} samples")

# Split train/test
train_data, train_labels = data[:400], labels[:400]
test_data, test_labels = data[400:], labels[400:]

# Train classifier
classifier = TimeSeriesRegimeClassifier(seq_length=60)
classifier.fit(train_data, train_labels, epochs=30)

# Test
results = classifier.classify(test_data[:5])
for i, (result, true_label) in enumerate(zip(results, test_labels[:5])):
    true_regime = classifier.regime_names[true_label]
    print(f"\nSample {i+1}:")
    print(f"  True: {true_regime.value}, Predicted: {result.regime.value}")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  Top regimes: {result.supporting_factors}")
```

### Hybrid LLM-Statistical Approaches

Combine LLM understanding with statistical rigor:

```python
class HybridRegimeClassifier:
    """
    Hybrid regime classifier combining:
    - Statistical methods (HMM, volatility clustering)
    - LLM-based text analysis
    - Rule-based economic indicators
    """

    def __init__(
        self,
        lookback: int = 60,
        text_weight: float = 0.3,
        stats_weight: float = 0.5,
        econ_weight: float = 0.2
    ):
        """
        Initialize hybrid classifier.

        Args:
            lookback: Lookback period for statistical analysis
            text_weight: Weight for text-based signals
            stats_weight: Weight for statistical signals
            econ_weight: Weight for economic indicator signals
        """
        self.lookback = lookback
        self.weights = {
            'text': text_weight,
            'stats': stats_weight,
            'econ': econ_weight
        }

        self.hmm_detector = HMMRegimeDetector(lookback=lookback)
        self.text_classifier = TextRegimeClassifier()

    def classify(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        texts: List[str],
        economic_data: Optional[Dict[str, float]] = None
    ) -> RegimeResult:
        """
        Classify regime using all available information.

        Args:
            returns: Recent return series
            volatility: Volatility series (or None to compute)
            texts: Recent news/social media texts
            economic_data: Optional economic indicators

        Returns:
            Combined RegimeResult
        """
        # Statistical classification
        stats_result = self.hmm_detector.detect_regime(returns, volatility)

        # Text classification
        text_result = self.text_classifier.aggregate_regime(texts)

        # Economic classification (if data available)
        if economic_data:
            econ_result = self._classify_from_economic(economic_data)
        else:
            econ_result = stats_result  # Fallback to stats

        # Combine results
        regime_scores = {r: 0.0 for r in MarketRegime}

        for result, weight_key in [
            (stats_result, 'stats'),
            (text_result, 'text'),
            (econ_result, 'econ')
        ]:
            weight = self.weights[weight_key]
            regime_scores[result.regime] += weight * result.probability

        # Normalize
        total = sum(regime_scores.values()) + 1e-8
        regime_probs = {r: s / total for r, s in regime_scores.items()}

        best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
        prob = regime_probs[best_regime]

        return RegimeResult(
            regime=best_regime,
            probability=prob,
            confidence=min(0.95, prob + 0.1),
            explanation="Hybrid classification combining stats, text, and economic signals",
            supporting_factors=[
                f"Statistical: {stats_result.regime.value} ({stats_result.probability:.0%})",
                f"Text: {text_result.regime.value} ({text_result.probability:.0%})",
                f"Economic: {econ_result.regime.value} ({econ_result.probability:.0%})"
            ]
        )

    def _classify_from_economic(
        self,
        data: Dict[str, float]
    ) -> RegimeResult:
        """Classify based on economic indicators."""
        gdp_growth = data.get('gdp_growth', 0.02)
        unemployment = data.get('unemployment', 0.05)
        inflation = data.get('inflation', 0.02)
        yield_curve = data.get('yield_curve_slope', 0.01)

        # Simple rule-based classification
        if gdp_growth > 0.03 and unemployment < 0.05:
            regime = MarketRegime.BULL
            prob = 0.7
        elif gdp_growth < 0 or unemployment > 0.08:
            regime = MarketRegime.BEAR
            prob = 0.7
        elif yield_curve < 0:  # Inverted yield curve
            regime = MarketRegime.HIGH_VOLATILITY
            prob = 0.6
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.5

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=0.6,
            explanation="Economic indicator-based classification",
            supporting_factors=[
                f"GDP growth: {gdp_growth:.1%}",
                f"Unemployment: {unemployment:.1%}",
                f"Inflation: {inflation:.1%}"
            ]
        )


# Example usage
classifier = HybridRegimeClassifier()

# Sample data
returns = np.random.randn(60) * 0.02 + 0.001  # Slight positive drift
volatility = np.abs(np.random.randn(60) * 0.01) + 0.01

texts = [
    "Markets continue upward trend on strong earnings",
    "Tech stocks rally amid AI optimism",
    "Consumer spending remains robust"
]

economic_data = {
    'gdp_growth': 0.025,
    'unemployment': 0.038,
    'inflation': 0.021,
    'yield_curve_slope': 0.015
}

result = classifier.classify(returns, volatility, texts, economic_data)

print(f"Hybrid Classification:")
print(f"  Regime: {result.regime.value}")
print(f"  Probability: {result.probability:.1%}")
print(f"  Confidence: {result.confidence:.1%}")
print(f"  Supporting factors:")
for factor in result.supporting_factors:
    print(f"    - {factor}")
```

## Practical Examples

### 01: Stock Market Regime Classification

See `python/examples/01_stock_regime.py` for complete implementation.

```python
# Quick start: Classify stock market regime
from python.classifier import RegimeClassifier
from python.data_loader import YahooFinanceLoader

# Load market data
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="1y")

# Initialize classifier
classifier = RegimeClassifier(lookback_window=60)

# Fit on historical data
classifier.fit(spy_data)

# Classify current regime
result = classifier.classify_current(spy_data)

print(f"Current Market Regime: {result.regime.value}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Explanation: {result.explanation}")
print(f"Supporting factors: {result.supporting_factors}")

# Get regime history
history = classifier.get_regime_history(spy_data)
for date, regime_result in history[-5:]:
    print(f"  {date}: {regime_result.regime.value}")
```

### 02: Crypto Market Regimes (Bybit)

See `python/examples/02_crypto_regime.py` for complete implementation.

```python
# Crypto regime classification on Bybit
from python.data_loader import BybitDataLoader
from python.classifier import CryptoRegimeClassifier

# Initialize Bybit loader
bybit = BybitDataLoader()

# Get BTC data
btc_data = bybit.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30
)

# Initialize crypto-specific classifier
classifier = CryptoRegimeClassifier(
    volatility_threshold=0.5,  # Higher for crypto
    trend_threshold=0.02
)

# Fit and classify
classifier.fit(btc_data)
result = classifier.classify_current(btc_data)

print(f"\nBTC Market Regime: {result.regime.value}")
print(f"Probability: {result.probability:.1%}")
print(f"Confidence: {result.confidence:.1%}")

# Crypto-specific insights
insights = classifier.get_crypto_insights(btc_data)
print(f"\nCrypto Insights:")
print(f"  24h Volatility: {insights['volatility_24h']:.1%}")
print(f"  Funding Rate: {insights['funding_rate']:.4%}")
print(f"  Long/Short Ratio: {insights['ls_ratio']:.2f}")
```

### 03: Multi-asset Regime Alignment

See `python/examples/03_multi_asset_regime.py` for complete implementation.

```python
# Multi-asset regime alignment analysis
from python.classifier import MultiAssetRegimeClassifier
from python.data_loader import YahooFinanceLoader, BybitDataLoader

# Load multiple assets
yahoo = YahooFinanceLoader()
bybit = BybitDataLoader()

assets = {
    'SPY': yahoo.get_daily("SPY", period="6mo"),
    'QQQ': yahoo.get_daily("QQQ", period="6mo"),
    'TLT': yahoo.get_daily("TLT", period="6mo"),
    'GLD': yahoo.get_daily("GLD", period="6mo"),
    'BTC': bybit.get_historical_klines("BTCUSDT", "1d", days=180)
}

# Multi-asset regime analysis
classifier = MultiAssetRegimeClassifier()

for symbol, data in assets.items():
    result = classifier.classify(data)
    print(f"{symbol}: {result.regime.value} ({result.probability:.0%})")

# Check regime alignment
alignment = classifier.compute_alignment(assets)
print(f"\nRegime Alignment Score: {alignment['score']:.2f}")
print(f"Dominant Regime: {alignment['dominant_regime']}")
print(f"Divergent Assets: {alignment['divergent_assets']}")
```

## Rust Implementation

The Rust implementation provides high-performance regime classification for production environments. See `rust/` directory for complete code.

```rust
//! LLM Regime Classification - Rust Implementation
//!
//! High-performance market regime classification for trading systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market regime enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    Crisis,
}

impl MarketRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketRegime::Bull => "bull",
            MarketRegime::Bear => "bear",
            MarketRegime::Sideways => "sideways",
            MarketRegime::HighVolatility => "high_volatility",
            MarketRegime::Crisis => "crisis",
        }
    }
}

/// Result of regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeResult {
    pub regime: MarketRegime,
    pub probability: f64,
    pub confidence: f64,
    pub explanation: String,
    pub supporting_factors: Vec<String>,
}

/// Statistical regime classifier using returns and volatility
pub struct StatisticalRegimeClassifier {
    lookback_window: usize,
    returns_history: Vec<f64>,
    volatility_history: Vec<f64>,
}

impl StatisticalRegimeClassifier {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            returns_history: Vec::with_capacity(lookback_window),
            volatility_history: Vec::with_capacity(lookback_window),
        }
    }

    pub fn update(&mut self, returns: f64, volatility: f64) {
        self.returns_history.push(returns);
        self.volatility_history.push(volatility);

        // Keep only lookback window
        if self.returns_history.len() > self.lookback_window {
            self.returns_history.remove(0);
            self.volatility_history.remove(0);
        }
    }

    pub fn classify(&self) -> RegimeResult {
        let mean_return = self.compute_mean(&self.returns_history);
        let mean_vol = self.compute_mean(&self.volatility_history);
        let annualized_vol = mean_vol * (252.0_f64).sqrt();

        // Classification logic
        let (regime, prob, explanation) = if annualized_vol > 0.40 {
            (
                MarketRegime::Crisis,
                (annualized_vol / 0.6).min(0.95),
                format!("Extreme volatility detected: {:.1}%", annualized_vol * 100.0),
            )
        } else if annualized_vol > 0.25 {
            (
                MarketRegime::HighVolatility,
                ((annualized_vol - 0.25) / 0.15).min(0.9),
                format!("Elevated volatility: {:.1}%", annualized_vol * 100.0),
            )
        } else if mean_return > 0.0005 {
            (
                MarketRegime::Bull,
                (mean_return / 0.002 + 0.5).min(0.9),
                format!("Positive trend: {:.2}% daily return", mean_return * 100.0),
            )
        } else if mean_return < -0.0005 {
            (
                MarketRegime::Bear,
                (mean_return.abs() / 0.002 + 0.5).min(0.9),
                format!("Negative trend: {:.2}% daily return", mean_return * 100.0),
            )
        } else {
            (
                MarketRegime::Sideways,
                0.7,
                "Range-bound market conditions".to_string(),
            )
        };

        RegimeResult {
            regime,
            probability: prob,
            confidence: prob * 0.9 + 0.1,
            explanation,
            supporting_factors: vec![
                format!("Mean return: {:.4}%", mean_return * 100.0),
                format!("Volatility: {:.1}%", annualized_vol * 100.0),
                format!("Lookback: {} periods", self.returns_history.len()),
            ],
        }
    }

    fn compute_mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Real-time regime monitor for cryptocurrency markets
pub struct CryptoRegimeMonitor {
    classifiers: HashMap<String, StatisticalRegimeClassifier>,
    lookback_window: usize,
}

impl CryptoRegimeMonitor {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            classifiers: HashMap::new(),
            lookback_window,
        }
    }

    pub fn process_candle(&mut self, symbol: &str, candle: &Candle) -> RegimeResult {
        let classifier = self
            .classifiers
            .entry(symbol.to_string())
            .or_insert_with(|| StatisticalRegimeClassifier::new(self.lookback_window));

        // Compute returns and volatility from candle
        let returns = (candle.close - candle.open) / candle.open;
        let volatility = (candle.high - candle.low) / candle.close;

        classifier.update(returns, volatility);
        let mut result = classifier.classify();

        result.supporting_factors.push(format!("Symbol: {}", symbol));
        result
    }

    pub fn get_all_regimes(&self) -> HashMap<String, RegimeResult> {
        self.classifiers
            .iter()
            .map(|(symbol, classifier)| (symbol.clone(), classifier.classify()))
            .collect()
    }
}

/// Hybrid classifier combining multiple signals
pub struct HybridRegimeClassifier {
    stats_classifier: StatisticalRegimeClassifier,
    text_weight: f64,
    stats_weight: f64,
}

impl HybridRegimeClassifier {
    pub fn new(lookback_window: usize, text_weight: f64, stats_weight: f64) -> Self {
        Self {
            stats_classifier: StatisticalRegimeClassifier::new(lookback_window),
            text_weight,
            stats_weight,
        }
    }

    pub fn classify(
        &mut self,
        returns: &[f64],
        volatility: &[f64],
        text_sentiment: f64,  // -1 to 1
    ) -> RegimeResult {
        // Update statistical classifier
        for (&r, &v) in returns.iter().zip(volatility.iter()) {
            self.stats_classifier.update(r, v);
        }

        let stats_result = self.stats_classifier.classify();

        // Text-based regime inference
        let text_regime = if text_sentiment > 0.3 {
            MarketRegime::Bull
        } else if text_sentiment < -0.3 {
            MarketRegime::Bear
        } else {
            MarketRegime::Sideways
        };

        // Combine signals
        let mut regime_scores: HashMap<MarketRegime, f64> = HashMap::new();

        *regime_scores.entry(stats_result.regime).or_insert(0.0) +=
            self.stats_weight * stats_result.probability;
        *regime_scores.entry(text_regime).or_insert(0.0) +=
            self.text_weight * text_sentiment.abs().max(0.5);

        // Find best regime
        let (best_regime, best_score) = regime_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(r, s)| (*r, *s))
            .unwrap_or((MarketRegime::Sideways, 0.5));

        let total_score: f64 = regime_scores.values().sum();
        let probability = best_score / total_score.max(0.001);

        RegimeResult {
            regime: best_regime,
            probability,
            confidence: (probability * 0.9 + 0.1).min(0.95),
            explanation: format!(
                "Hybrid classification: stats={}, text_sentiment={:.2}",
                stats_result.regime.as_str(),
                text_sentiment
            ),
            supporting_factors: vec![
                format!("Statistical regime: {}", stats_result.regime.as_str()),
                format!("Text sentiment: {:.2}", text_sentiment),
                format!("Combined probability: {:.1}%", probability * 100.0),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bull_regime_detection() {
        let mut classifier = StatisticalRegimeClassifier::new(20);

        // Add bullish data (positive returns, low volatility)
        for _ in 0..20 {
            classifier.update(0.002, 0.01);  // +0.2% daily, 1% volatility
        }

        let result = classifier.classify();
        assert_eq!(result.regime, MarketRegime::Bull);
        assert!(result.probability > 0.5);
    }

    #[test]
    fn test_bear_regime_detection() {
        let mut classifier = StatisticalRegimeClassifier::new(20);

        // Add bearish data
        for _ in 0..20 {
            classifier.update(-0.002, 0.015);
        }

        let result = classifier.classify();
        assert_eq!(result.regime, MarketRegime::Bear);
    }

    #[test]
    fn test_high_volatility_detection() {
        let mut classifier = StatisticalRegimeClassifier::new(20);

        // Add high volatility data
        for _ in 0..20 {
            classifier.update(0.0, 0.025);  // High volatility
        }

        let result = classifier.classify();
        assert_eq!(result.regime, MarketRegime::HighVolatility);
    }

    #[test]
    fn test_crypto_monitor() {
        let mut monitor = CryptoRegimeMonitor::new(20);

        let candle = Candle {
            timestamp: 1700000000,
            open: 40000.0,
            high: 41000.0,
            low: 39500.0,
            close: 40500.0,
            volume: 1000000.0,
        };

        let result = monitor.process_candle("BTCUSDT", &candle);
        assert!(result.probability > 0.0);
    }
}
```

## Python Implementation

The Python implementation includes comprehensive modules for research and development. See `python/` directory for full code.

**Key modules:**

| Module | Description |
|--------|-------------|
| `classifier.py` | Core regime classification algorithms |
| `data_loader.py` | Yahoo Finance and Bybit data loaders |
| `embeddings.py` | LLM-based text and time series embeddings |
| `signals.py` | Trading signal generation from regime detection |
| `backtest.py` | Backtesting framework for regime-based strategies |
| `evaluate.py` | Evaluation metrics and visualization |

## Backtesting Framework

Test regime-based trading strategies on historical data:

```python
from python.backtest import RegimeBacktester
from python.classifier import HybridRegimeClassifier
from python.data_loader import YahooFinanceLoader

# Load historical data
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="5y")

# Initialize backtester
backtester = RegimeBacktester(
    initial_capital=100000,
    commission=0.001
)

# Define regime-based strategy
strategy = {
    'bull': {'position': 1.5, 'stop_loss': -0.05},      # 150% long
    'bear': {'position': -0.5, 'stop_loss': -0.03},     # 50% short
    'sideways': {'position': 0.5, 'stop_loss': -0.02},  # 50% long
    'high_volatility': {'position': 0.3, 'stop_loss': -0.02},  # 30% long
    'crisis': {'position': 0.0, 'stop_loss': None}       # All cash
}

# Run backtest
results = backtester.run(
    data=spy_data,
    classifier=HybridRegimeClassifier(),
    strategy=strategy
)

print(f"Strategy Performance:")
print(f"  Total Return: {results['total_return']:.2%}")
print(f"  Annual Return: {results['annual_return']:.2%}")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
print(f"  Win Rate: {results['win_rate']:.2%}")

print(f"\nRegime Statistics:")
for regime, stats in results['regime_stats'].items():
    print(f"  {regime}:")
    print(f"    Time in regime: {stats['time_pct']:.1%}")
    print(f"    Return in regime: {stats['return']:.2%}")
```

## Best Practices

### Classification Guidelines

```
LLM REGIME CLASSIFICATION BEST PRACTICES:
======================================================================

1. DATA PREPARATION
   +----------------------------------------------------------------+
   | - Normalize features before classification                       |
   | - Handle missing data explicitly                                 |
   | - Use rolling windows to avoid look-ahead bias                   |
   | - Separate training data by time periods                         |
   +----------------------------------------------------------------+

2. MODEL SELECTION
   +----------------------------------------------------------------+
   | - Start with simple statistical baselines (HMM)                  |
   | - Add text signals for context awareness                         |
   | - Use domain-specific LLMs (FinBERT) for finance                 |
   | - Ensemble multiple methods for robustness                       |
   +----------------------------------------------------------------+

3. REGIME TRANSITIONS
   +----------------------------------------------------------------+
   | - Add hysteresis to prevent frequent regime flips                |
   | - Require confirmation (multiple signals) for transition         |
   | - Track regime duration for strategy timing                      |
   | - Consider transition probabilities in risk management           |
   +----------------------------------------------------------------+

4. STRATEGY ADAPTATION
   +----------------------------------------------------------------+
   | - Map each regime to specific strategy parameters                |
   | - Adjust position sizing based on regime confidence              |
   | - Use regime-specific stop losses and targets                    |
   | - Reduce exposure during uncertain regime transitions            |
   +----------------------------------------------------------------+

5. PRODUCTION DEPLOYMENT
   +----------------------------------------------------------------+
   | - Cache LLM embeddings for efficiency                            |
   | - Implement fallback to statistical methods                      |
   | - Log all regime changes for audit                               |
   | - Monitor regime classifier performance continuously             |
   +----------------------------------------------------------------+
```

### Common Pitfalls

```
COMMON MISTAKES TO AVOID:
======================================================================

X Using future data for regime labeling
  -> Always use causal (past-only) labeling

X Overfitting to historical regimes
  -> Test on out-of-sample regime transitions

X Ignoring regime transition periods
  -> Add uncertainty during transition phases

X Single-signal regime detection
  -> Combine multiple data sources

X Fixed regime thresholds
  -> Adapt thresholds to market conditions

X Frequent regime switching
  -> Add confirmation and hysteresis

X Ignoring regime-strategy mismatch
  -> Validate strategy performance per regime
```

## Resources

### Papers

1. **Market Regime Detection with LLMs** (2024)
   - https://arxiv.org/abs/2401.10586

2. **Hidden Markov Models for Regime Detection** (Hamilton, 1989)
   - Classic paper on regime-switching models

3. **Machine Learning for Asset Management** (2020)
   - Comprehensive survey of ML in finance

4. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** (2019)
   - https://arxiv.org/abs/1908.10063

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| Yahoo Finance | Historical stock data | Varies |
| Bybit API | Cryptocurrency data | Real-time |
| FRED | Economic indicators | Various series |
| News API | Financial news | Real-time |

### Tools & Libraries

- [hmmlearn](https://github.com/hmmlearn/hmmlearn) - Hidden Markov Models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - LLM library
- [Candle](https://github.com/huggingface/candle) - Rust ML framework

### Directory Structure

```
77_llm_regime_classification/
+-- README.md              # This file (English)
+-- README.ru.md           # Russian translation
+-- readme.simple.md       # Beginner-friendly explanation
+-- readme.simple.ru.md    # Beginner-friendly (Russian)
+-- python/
|   +-- __init__.py
|   +-- classifier.py      # Core regime classification
|   +-- embeddings.py      # LLM embedding generation
|   +-- data_loader.py     # Yahoo Finance & Bybit loaders
|   +-- signals.py         # Trading signal generation
|   +-- backtest.py        # Backtesting framework
|   +-- evaluate.py        # Evaluation metrics
|   +-- requirements.txt   # Python dependencies
|   +-- examples/
|       +-- 01_stock_regime.py
|       +-- 02_crypto_regime.py
|       +-- 03_multi_asset_regime.py
+-- rust/
    +-- Cargo.toml
    +-- src/
        +-- lib.rs         # Library root
        +-- classifier.rs  # Regime classification
        +-- data_loader.rs # Data loading
        +-- signals.rs     # Signal generation
        +-- backtest.rs    # Backtesting
    +-- examples/
        +-- basic_classification.rs
        +-- bybit_monitor.rs
        +-- backtest.rs
```
