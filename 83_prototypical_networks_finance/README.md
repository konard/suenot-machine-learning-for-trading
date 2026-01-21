# Chapter 83: Prototypical Networks for Finance

## Overview

Prototypical Networks are a meta-learning approach designed for few-shot classification tasks. In financial markets, they excel at classifying market regimes, detecting trading patterns, and adapting to new market conditions with minimal labeled data. This is particularly valuable in crypto markets where regimes can shift rapidly and historical patterns may have limited examples.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture Components](#architecture-components)
4. [Application to Financial Markets](#application-to-financial-markets)
5. [Few-Shot Market Regime Classification](#few-shot-market-regime-classification)
6. [Implementation Strategy](#implementation-strategy)
7. [Bybit Integration](#bybit-integration)
8. [Risk Management](#risk-management)
9. [Performance Metrics](#performance-metrics)
10. [References](#references)

---

## Introduction

Traditional machine learning approaches for trading require large amounts of labeled data for each market regime or pattern. However, financial markets present unique challenges:

- **Regime scarcity**: Some market conditions (crashes, squeezes) occur rarely
- **Concept drift**: Markets evolve, making historical data less relevant
- **Rapid adaptation**: Need to recognize new patterns with few examples

### Why Prototypical Networks for Trading?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Few-Shot Trading Problem                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional ML:                  Prototypical Networks:                │
│   ─────────────────                ──────────────────────                │
│   Need 1000s of examples           Need only 5-20 examples              │
│   of each pattern                  per pattern (support set)            │
│                                                                          │
│   Problem: Rare events like        Solution: Learn to compute           │
│   flash crashes have few           "prototypes" from few examples       │
│   historical examples              and classify new data                │
│                                                                          │
│   ┌────────────┐                  ┌────────────┐                        │
│   │ Bull Market│ 1000 samples     │ Bull Market│ 10 samples             │
│   │ Bear Market│ 1000 samples     │ Bear Market│ 10 samples             │
│   │ Crash      │   12 samples ❌  │ Crash      │  5 samples ✓          │
│   │ Squeeze    │    8 samples ❌  │ Squeeze    │  5 samples ✓          │
│   └────────────┘                  └────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Advantages

| Aspect | Traditional ML | Prototypical Networks |
|--------|---------------|----------------------|
| Data requirements | High (1000s per class) | Low (5-20 per class) |
| New pattern adaptation | Requires retraining | Few examples sufficient |
| Rare event handling | Poor | Excellent |
| Computational cost | High for retraining | Low for adaptation |
| Interpretability | Low (black box) | High (prototype distances) |

## Theoretical Foundation

### The Prototypical Network Framework

A Prototypical Network learns a metric space where classification is performed by computing distances to class prototypes (centroids).

### Mathematical Formulation

**Embedding Function**: $f_\phi: \mathbb{R}^D \rightarrow \mathbb{R}^M$

Maps input data to an M-dimensional embedding space via a neural network with parameters $\phi$.

**Prototype Computation**: Given a support set $S_k$ of examples for class $k$:

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

where $c_k$ is the prototype (centroid) for class $k$.

**Classification**: For a query point $x$, compute probability distribution over classes:

$$p(y = k | x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

where $d$ is a distance function (typically squared Euclidean distance).

### Training via Episodic Learning

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Episodic Training Process                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Episode = One training iteration simulating few-shot scenario         │
│                                                                         │
│   Step 1: Sample N classes (e.g., 5 market regimes)                    │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Bull  │  Bear  │  Sideways  │  Crash  │  Recovery  │              │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 2: For each class, sample K support examples + Q query examples │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Bull: [s1, s2, s3, s4, s5] | [q1, q2, q3]                 │      │
│   │  Bear: [s1, s2, s3, s4, s5] | [q1, q2, q3]                 │      │
│   │  ...                                                        │      │
│   └─────────────────────────────────────────────────────────────┘      │
│        Support Set (5-shot)       Query Set                             │
│                                                                         │
│   Step 3: Compute prototypes from support set                          │
│   Step 4: Classify query examples using prototype distances            │
│   Step 5: Compute loss and backpropagate                               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### N-way K-shot Classification

- **N-way**: Number of classes to distinguish
- **K-shot**: Number of examples per class in the support set

For trading: typically 3-5 way (regimes) with 5-10 shot (examples per regime)

### Distance Functions

**Squared Euclidean Distance** (default):
$$d(x, y) = \|x - y\|^2 = \sum_i (x_i - y_i)^2$$

**Cosine Distance** (alternative):
$$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$$

**Mahalanobis Distance** (accounts for covariance):
$$d(x, y) = \sqrt{(x-y)^T \Sigma^{-1} (x-y)}$$

## Architecture Components

### Embedding Network Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Embedding Network for Trading                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: Market features [price, volume, volatility, ...]              │
│   Shape: (batch_size, sequence_length, feature_dim)                    │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Temporal Convolutional Block                               │      │
│   │  ─────────────────────────────                              │      │
│   │  Conv1D(in=features, out=64, kernel=3) → BatchNorm → ReLU   │      │
│   │  Conv1D(in=64, out=128, kernel=3) → BatchNorm → ReLU        │      │
│   │  Conv1D(in=128, out=128, kernel=3) → BatchNorm → ReLU       │      │
│   │  MaxPool1D(kernel=2)                                        │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  LSTM/Transformer Block                                      │      │
│   │  ───────────────────────                                    │      │
│   │  LSTM(hidden=256, layers=2, bidirectional=True)             │      │
│   │  OR                                                          │      │
│   │  TransformerEncoder(d_model=256, nhead=8, layers=2)         │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Projection Head                                             │      │
│   │  ───────────────                                            │      │
│   │  Linear(in=256, out=128) → ReLU                             │      │
│   │  Linear(in=128, out=embedding_dim)                          │      │
│   │  L2 Normalize (optional)                                     │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   Output: Embedding vector of shape (batch_size, embedding_dim)        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering for Trading

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Input Features for Market Regime                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Price Features:                                                       │
│   • Returns (1m, 5m, 15m, 1h, 4h, 24h)                                 │
│   • Log returns                                                         │
│   • High-low range                                                      │
│   • Close position in range                                             │
│                                                                         │
│   Volume Features:                                                      │
│   • Volume (normalized)                                                 │
│   • Volume change                                                       │
│   • Buy/Sell volume ratio                                              │
│   • Volume profile                                                      │
│                                                                         │
│   Volatility Features:                                                  │
│   • Rolling volatility (multiple windows)                              │
│   • ATR (Average True Range)                                           │
│   • Bollinger Band width                                               │
│   • Parkinson volatility                                               │
│                                                                         │
│   Market Structure Features:                                            │
│   • RSI, MACD, Stochastic                                              │
│   • Moving average relationships                                        │
│   • Support/Resistance levels                                          │
│   • Order book imbalance                                               │
│                                                                         │
│   Crypto-Specific Features:                                            │
│   • Funding rate                                                        │
│   • Open interest                                                       │
│   • Long/Short ratio                                                   │
│   • Liquidation levels                                                 │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Prototype Computation Module

```rust
/// Compute prototypes from support set embeddings
pub struct PrototypeComputer {
    /// Distance function type
    distance_type: DistanceType,
    /// Optional prototype refinement
    refine_prototypes: bool,
}

impl PrototypeComputer {
    /// Compute class prototypes from support embeddings
    pub fn compute_prototypes(
        &self,
        support_embeddings: &Array2<f32>,  // (n_support, embed_dim)
        support_labels: &Array1<usize>,     // (n_support,)
        n_classes: usize,
    ) -> Array2<f32> {                      // (n_classes, embed_dim)
        let embed_dim = support_embeddings.ncols();
        let mut prototypes = Array2::zeros((n_classes, embed_dim));
        let mut counts = vec![0usize; n_classes];

        // Sum embeddings per class
        for (i, &label) in support_labels.iter().enumerate() {
            prototypes.row_mut(label).add_assign(&support_embeddings.row(i));
            counts[label] += 1;
        }

        // Average to get prototypes
        for (class_idx, count) in counts.iter().enumerate() {
            if *count > 0 {
                prototypes.row_mut(class_idx).mapv_inplace(|x| x / *count as f32);
            }
        }

        prototypes
    }

    /// Classify query points using prototype distances
    pub fn classify(
        &self,
        query_embeddings: &Array2<f32>,  // (n_query, embed_dim)
        prototypes: &Array2<f32>,         // (n_classes, embed_dim)
    ) -> (Array1<usize>, Array2<f32>) {   // (predictions, probabilities)
        let n_query = query_embeddings.nrows();
        let n_classes = prototypes.nrows();

        // Compute distances to all prototypes
        let mut distances = Array2::zeros((n_query, n_classes));
        for i in 0..n_query {
            for j in 0..n_classes {
                distances[[i, j]] = self.compute_distance(
                    &query_embeddings.row(i),
                    &prototypes.row(j),
                );
            }
        }

        // Convert to probabilities via softmax over negative distances
        let neg_distances = -&distances;
        let probabilities = softmax(&neg_distances);

        // Get predictions (argmax)
        let predictions = probabilities
            .outer_iter()
            .map(|row| row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0)
            .collect();

        (Array1::from_vec(predictions), probabilities)
    }
}
```

## Application to Financial Markets

### Market Regime Classification

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Market Regime Classes                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Class 0: STRONG_UPTREND                                              │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Sustained positive returns (>2% daily avg)             │        │
│   │  • Higher highs, higher lows                               │        │
│   │  • Volume confirming moves                                 │        │
│   │  • Positive funding rate                                   │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 1: WEAK_UPTREND                                                │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Mild positive returns (0.5-2% daily avg)               │        │
│   │  • Choppy price action with upward bias                    │        │
│   │  • Mixed volume signals                                    │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 2: SIDEWAYS / CONSOLIDATION                                    │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Returns near zero                                       │        │
│   │  • Price bouncing between support/resistance              │        │
│   │  • Declining volatility                                    │        │
│   │  • Low volume                                              │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 3: WEAK_DOWNTREND                                              │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Mild negative returns (-0.5 to -2% daily avg)          │        │
│   │  • Lower highs pattern emerging                            │        │
│   │  • Negative funding rate                                   │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 4: STRONG_DOWNTREND / CRASH                                    │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Sharp negative returns (<-2% daily avg)                │        │
│   │  • Spike in volatility                                     │        │
│   │  • High volume on down moves                               │        │
│   │  • Liquidation cascades                                    │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Trading Strategy Based on Regime

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Regime-Based Trading Signals                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Regime Detection Pipeline:                                            │
│   ─────────────────────────                                            │
│   1. Collect recent market data (e.g., last 24 hours)                  │
│   2. Extract features for current window                               │
│   3. Embed features using trained network                              │
│   4. Compute distances to regime prototypes                            │
│   5. Classify current regime with confidence                           │
│   6. Generate trading signal based on regime                           │
│                                                                         │
│   Regime → Signal Mapping:                                              │
│   ┌─────────────────┬──────────────────────────────────────────┐       │
│   │ Regime          │ Action                                    │       │
│   ├─────────────────┼──────────────────────────────────────────┤       │
│   │ STRONG_UPTREND  │ Long with trailing stop, add on dips    │       │
│   │ WEAK_UPTREND    │ Light long, tight stops                  │       │
│   │ SIDEWAYS        │ Mean reversion, range trading            │       │
│   │ WEAK_DOWNTREND  │ Light short or stay flat                 │       │
│   │ STRONG_DOWNTREND│ Short with protection, hedge longs       │       │
│   └─────────────────┴──────────────────────────────────────────┘       │
│                                                                         │
│   Confidence-Based Position Sizing:                                     │
│   ───────────────────────────────                                      │
│   position_size = base_size × classification_confidence                │
│                                                                         │
│   If confidence < 0.6: Reduce position or stay flat                    │
│   If confidence > 0.8: Full position size                              │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Few-Shot Market Regime Classification

### Episode Generation for Training

```python
# Pseudocode for episode generation
def generate_episode(dataset, n_way=5, k_shot=5, n_query=10):
    """
    Generate one training episode for prototypical network.

    Args:
        dataset: Historical market data with regime labels
        n_way: Number of regime classes per episode
        k_shot: Number of support examples per class
        n_query: Number of query examples per class

    Returns:
        support_set: (n_way * k_shot, features) tensor
        support_labels: (n_way * k_shot,) tensor
        query_set: (n_way * n_query, features) tensor
        query_labels: (n_way * n_query,) tensor
    """
    # Sample n_way classes from available regime classes
    available_classes = dataset.get_regime_classes()
    sampled_classes = random.sample(available_classes, n_way)

    support_set = []
    support_labels = []
    query_set = []
    query_labels = []

    for class_idx, regime_class in enumerate(sampled_classes):
        # Get all samples for this regime
        class_samples = dataset.get_samples_for_regime(regime_class)

        # Sample k_shot + n_query examples
        sampled_indices = random.sample(
            range(len(class_samples)),
            k_shot + n_query
        )

        # Split into support and query
        support_indices = sampled_indices[:k_shot]
        query_indices = sampled_indices[k_shot:]

        support_set.extend([class_samples[i] for i in support_indices])
        support_labels.extend([class_idx] * k_shot)

        query_set.extend([class_samples[i] for i in query_indices])
        query_labels.extend([class_idx] * n_query)

    return (
        torch.stack(support_set),
        torch.tensor(support_labels),
        torch.stack(query_set),
        torch.tensor(query_labels)
    )
```

### Regime Labeling Strategies

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Automatic Regime Labeling                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Method 1: Return-Based Quantiles                                     │
│   ───────────────────────────────                                      │
│   Label based on cumulative return over window:                        │
│   • Top 20%: STRONG_UPTREND                                            │
│   • 20-40%: WEAK_UPTREND                                               │
│   • 40-60%: SIDEWAYS                                                   │
│   • 60-80%: WEAK_DOWNTREND                                             │
│   • Bottom 20%: STRONG_DOWNTREND                                       │
│                                                                         │
│   Method 2: Volatility-Adjusted Returns                                │
│   ─────────────────────────────────────                                │
│   risk_adjusted_return = return / volatility                           │
│   Label based on risk-adjusted thresholds                              │
│                                                                         │
│   Method 3: Hidden Markov Model                                        │
│   ─────────────────────────────                                        │
│   Train HMM to discover latent regimes                                 │
│   Use HMM state assignments as labels                                  │
│                                                                         │
│   Method 4: Clustering-Based                                           │
│   ──────────────────────────                                           │
│   Extract features → K-means clustering → Use clusters as regimes      │
│                                                                         │
│   Method 5: Manual Expert Labeling (Gold Standard)                     │
│   ───────────────────────────────────────────────                      │
│   Domain experts label key market periods                              │
│   Used for validation and rare event examples                          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Handling Class Imbalance

Financial markets naturally have imbalanced regime distributions:
- Bull/Bear markets: Common
- Crashes: Rare but critical

```
Strategies for Imbalanced Few-Shot Learning:

1. Weighted Prototype Computation
   - Give more weight to rare class examples
   - prototype_k = weighted_mean(support_k, weights)

2. Synthetic Example Generation
   - Generate synthetic crash/squeeze examples
   - Use data augmentation for rare regimes

3. Episode Sampling Strategy
   - Oversample episodes containing rare classes
   - Ensure all classes appear equally often

4. Distance Scaling
   - Scale distances inversely with class frequency
   - Rare classes have smaller distances (easier to classify)
```

## Implementation Strategy

### Module Architecture

```
83_prototypical_networks_finance/
├── Cargo.toml
├── README.md
├── README.ru.md
├── readme.simple.md
├── readme.simple.ru.md
├── src/
│   ├── lib.rs                    # Library root
│   ├── network/
│   │   ├── mod.rs               # Network module
│   │   ├── embedding.rs         # Embedding network
│   │   ├── prototype.rs         # Prototype computation
│   │   └── distance.rs          # Distance functions
│   ├── training/
│   │   ├── mod.rs               # Training module
│   │   ├── episode.rs           # Episode generation
│   │   ├── trainer.rs           # Training loop
│   │   └── scheduler.rs         # Learning rate scheduling
│   ├── data/
│   │   ├── mod.rs               # Data module
│   │   ├── bybit.rs             # Bybit API client
│   │   ├── features.rs          # Feature engineering
│   │   ├── regime.rs            # Regime labeling
│   │   └── types.rs             # Data types
│   ├── strategy/
│   │   ├── mod.rs               # Strategy module
│   │   ├── classifier.rs        # Regime classifier
│   │   ├── signals.rs           # Signal generation
│   │   └── execution.rs         # Order execution
│   └── utils/
│       ├── mod.rs               # Utilities
│       └── metrics.rs           # Performance metrics
├── examples/
│   ├── basic_prototypical.rs    # Basic example
│   ├── regime_trading.rs        # Regime-based trading
│   └── backtest.rs              # Backtesting
├── python/
│   ├── prototypical_network.py  # PyTorch implementation
│   ├── train.py                 # Training script
│   └── notebooks/
│       └── example.ipynb        # Jupyter notebook example
└── tests/
    └── integration.rs           # Integration tests
```

### Key Design Principles

1. **Modularity**: Each component (embedding, prototype, distance) is independent
2. **Type Safety**: Leverage Rust's type system for financial data integrity
3. **Performance**: Efficient batch operations for real-time inference
4. **Flexibility**: Support different distance functions and embedding architectures
5. **Testability**: Comprehensive unit and integration tests

### Core Types in Rust

```rust
/// Market regime enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    StrongUptrend,
    WeakUptrend,
    Sideways,
    WeakDowntrend,
    StrongDowntrend,
}

impl MarketRegime {
    /// Get trading bias for this regime
    pub fn trading_bias(&self) -> TradingBias {
        match self {
            Self::StrongUptrend => TradingBias::StrongLong,
            Self::WeakUptrend => TradingBias::WeakLong,
            Self::Sideways => TradingBias::Neutral,
            Self::WeakDowntrend => TradingBias::WeakShort,
            Self::StrongDowntrend => TradingBias::StrongShort,
        }
    }
}

/// Distance function types
#[derive(Debug, Clone, Copy)]
pub enum DistanceType {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    Mahalanobis,
}

/// Prototypical network configuration
#[derive(Debug, Clone)]
pub struct PrototypicalConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of classes (regimes)
    pub n_classes: usize,
    /// Number of shots for support set
    pub k_shot: usize,
    /// Number of query examples
    pub n_query: usize,
    /// Distance function type
    pub distance_type: DistanceType,
    /// Temperature for softmax
    pub temperature: f32,
    /// Input feature dimension
    pub input_dim: usize,
    /// Sequence length for temporal features
    pub sequence_length: usize,
}

impl Default for PrototypicalConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            n_classes: 5,
            k_shot: 5,
            n_query: 15,
            distance_type: DistanceType::SquaredEuclidean,
            temperature: 1.0,
            input_dim: 32,
            sequence_length: 48,  // e.g., 48 hours of hourly data
        }
    }
}
```

## Bybit Integration

### Data Collection Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Bybit Data Pipeline                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. Historical Data Collection (Training)                             │
│   ────────────────────────────────────────                             │
│   GET /v5/market/kline → Klines for multiple timeframes               │
│   GET /v5/market/tickers → Current market state                       │
│   GET /v5/market/funding/history → Historical funding rates           │
│   GET /v5/market/open-interest → Open interest history                │
│                                                                         │
│   2. Real-Time Data (Inference)                                        │
│   ──────────────────────────────                                       │
│   WebSocket subscription:                                               │
│   • kline.{interval}.{symbol} → Real-time candles                     │
│   • ticker.{symbol} → Price updates                                    │
│   • liquidation.{symbol} → Liquidation events                         │
│                                                                         │
│   3. Feature Computation                                                │
│   ──────────────────────                                               │
│   Raw data → Feature extractor → Normalized features                   │
│                                                                         │
│   4. Regime Classification                                              │
│   ───────────────────────                                              │
│   Features → Embedding network → Distance to prototypes → Regime      │
│                                                                         │
│   5. Trading Signal                                                     │
│   ────────────────                                                     │
│   Regime + Confidence → Position sizing → Order submission             │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Bybit Client Integration

```rust
use crate::data::{BybitClient, Kline, FundingRate};

/// Fetch and process data for prototypical network
pub async fn collect_training_data(
    client: &BybitClient,
    symbols: &[&str],
    start_time: u64,
    end_time: u64,
) -> Result<TrainingDataset, BybitError> {
    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    for symbol in symbols {
        // Fetch klines
        let klines = client.get_klines(
            symbol,
            "1h",
            1000  // Maximum limit
        ).await?;

        // Fetch funding rates
        let funding = client.get_funding_rate(symbol).await?;

        // Extract features for each window
        for window in klines.windows(48) {  // 48-hour windows
            let features = extract_features(window, &funding)?;
            let label = compute_regime_label(window)?;

            all_features.push(features);
            all_labels.push(label);
        }
    }

    Ok(TrainingDataset::new(all_features, all_labels))
}
```

## Risk Management

### Regime-Specific Risk Controls

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Risk Management by Regime                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STRONG_UPTREND:                                                       │
│   • Max position: 100% of allocation                                   │
│   • Stop loss: 5% trailing                                             │
│   • Take profit: Let winners run                                       │
│   • Leverage: Up to 3x                                                 │
│                                                                         │
│   WEAK_UPTREND:                                                         │
│   • Max position: 60% of allocation                                    │
│   • Stop loss: 3% fixed                                                │
│   • Take profit: 5-8% target                                           │
│   • Leverage: Up to 2x                                                 │
│                                                                         │
│   SIDEWAYS:                                                             │
│   • Max position: 40% of allocation                                    │
│   • Stop loss: 2% fixed                                                │
│   • Take profit: Range boundaries                                      │
│   • Leverage: 1x only                                                  │
│                                                                         │
│   WEAK_DOWNTREND:                                                       │
│   • Max position: 30% short or 20% hedge                              │
│   • Stop loss: 3% fixed                                                │
│   • Take profit: 5-8% target                                           │
│   • Leverage: Up to 2x                                                 │
│                                                                         │
│   STRONG_DOWNTREND:                                                     │
│   • Max position: 50% short                                            │
│   • Stop loss: 5% trailing                                             │
│   • Take profit: Panic levels                                          │
│   • Leverage: Up to 3x (with caution)                                 │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Classification Confidence Thresholds

```rust
/// Risk parameters based on classification confidence
pub struct RiskParameters {
    pub position_size_multiplier: f32,
    pub max_leverage: f32,
    pub stop_loss_pct: f32,
    pub take_profit_pct: Option<f32>,
}

impl RiskParameters {
    pub fn from_confidence(confidence: f32, regime: MarketRegime) -> Self {
        let base = Self::base_for_regime(regime);

        // Scale position size with confidence
        let size_mult = if confidence < 0.6 {
            0.25  // Very uncertain - minimal position
        } else if confidence < 0.75 {
            0.5   // Moderately confident
        } else if confidence < 0.9 {
            0.75  // Confident
        } else {
            1.0   // Very confident - full position
        };

        Self {
            position_size_multiplier: base.position_size_multiplier * size_mult,
            max_leverage: base.max_leverage,
            stop_loss_pct: base.stop_loss_pct,
            take_profit_pct: base.take_profit_pct,
        }
    }
}
```

### Circuit Breakers

1. **Confidence Drop**: If classification confidence drops below 0.5, flatten positions
2. **Regime Flip-Flop**: If regime changes > 3 times in 4 hours, reduce exposure
3. **Distance Spike**: If min distance to all prototypes exceeds threshold, unknown regime
4. **Drawdown Limit**: If strategy drawdown > 10%, pause trading

## Performance Metrics

### Model Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall classification accuracy | > 70% |
| F1-Score (macro) | Balanced accuracy across regimes | > 0.65 |
| F1-Score (crash) | Crash detection specifically | > 0.80 |
| AUC-ROC | Discrimination ability | > 0.85 |
| Calibration Error | Confidence reliability | < 0.10 |

### Trading Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted returns | > 2.0 |
| Sortino Ratio | Downside risk-adjusted | > 2.5 |
| Max Drawdown | Largest peak-to-trough | < 15% |
| Win Rate | Profitable trades | > 55% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Regime Detection Lag | Time to detect regime change | < 4 hours |

### Latency Budget

```
┌─────────────────────────────────────────────────┐
│              Latency Requirements               │
├─────────────────────────────────────────────────┤
│ Feature Computation:       < 10ms               │
│ Embedding Forward Pass:    < 20ms               │
│ Prototype Distance:        < 5ms                │
│ Classification:            < 5ms                │
├─────────────────────────────────────────────────┤
│ Total Inference:           < 40ms               │
└─────────────────────────────────────────────────┘
```

## References

1. **Prototypical Networks for Few-shot Learning**
   - Snell, J., Swersky, K., & Zemel, R. (2017)
   - URL: https://arxiv.org/abs/1703.05175

2. **Matching Networks for One Shot Learning**
   - Vinyals, O., et al. (2016)
   - URL: https://arxiv.org/abs/1606.04080

3. **Meta-Learning for Semi-Supervised Few-Shot Classification**
   - Ren, M., et al. (2018)
   - URL: https://arxiv.org/abs/1803.00676

4. **Meta-Learning: A Survey**
   - Hospedales, T., et al. (2020)
   - URL: https://arxiv.org/abs/2004.05439

5. **Few-Shot Learning for Financial Time Series**
   - Recent applications of meta-learning to trading

6. **Market Regime Detection**
   - Nystrup, P., et al. (2020). "Learning Hidden Markov Models with Persistent States"

---

## Next Steps

- [View Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Русская версия
- [Run Examples](examples/) - Working Rust code
- [Python Implementation](python/) - PyTorch reference implementation

---

*Chapter 83 of Machine Learning for Trading*
