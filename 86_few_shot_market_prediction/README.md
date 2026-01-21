# Chapter 86: Few-Shot Market Prediction

## Overview

Few-Shot Market Prediction addresses the challenge of making accurate predictions on new or unseen markets with minimal historical data. This is particularly crucial in cryptocurrency markets where new trading pairs frequently emerge, and traditional machine learning models require extensive retraining. Using meta-learning techniques, we can adapt quickly to new market conditions with just 5-20 examples.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Few-Shot Learning Approaches](#few-shot-learning-approaches)
4. [Architecture Design](#architecture-design)
5. [Application to Market Prediction](#application-to-market-prediction)
6. [Implementation Strategy](#implementation-strategy)
7. [Bybit Integration](#bybit-integration)
8. [Risk Management](#risk-management)
9. [Performance Metrics](#performance-metrics)
10. [References](#references)

---

## Introduction

Traditional machine learning models for market prediction face a critical limitation: they require thousands of labeled examples for each market or asset to achieve reasonable accuracy. This becomes problematic when:

- **New Assets Emerge**: New cryptocurrencies, tokens, or trading pairs are listed daily
- **Market Dynamics Shift**: The statistical properties of markets change over time
- **Cross-Market Transfer**: Knowledge from one market may not directly apply to another
- **Limited Historical Data**: Some market conditions (crashes, squeezes) have few historical examples

### The Few-Shot Learning Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Traditional ML vs Few-Shot Learning                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TRADITIONAL ML:                                                            │
│   ───────────────                                                            │
│   Training: 10,000+ samples per asset                                        │
│   Problem: New asset listed → Need weeks of data                             │
│   Result: Cannot trade profitably during early adoption phase                │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  BTC  │████████████████████████████████│ 50,000 samples            │   │
│   │  ETH  │████████████████████████████████│ 45,000 samples            │   │
│   │  NEW! │██                              │ 500 samples ❌ Not enough │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   FEW-SHOT LEARNING:                                                         │
│   ──────────────────                                                         │
│   Training: Learn "how to learn" from many assets                            │
│   Inference: Adapt to new asset with 5-20 examples                           │
│   Result: Trade profitably within hours of listing                           │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Meta-train on: BTC, ETH, SOL, AVAX, DOT, LINK, UNI, ...           │   │
│   │  Meta-test on:  NEW_TOKEN with just 10 examples ✓                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits for Trading

| Aspect | Traditional ML | Few-Shot Learning |
|--------|---------------|-------------------|
| New asset adaptation | Weeks to months | Hours |
| Data requirements | 10,000+ samples | 5-20 samples |
| Cross-market transfer | Limited | Excellent |
| Regime adaptation | Slow retraining | Fast adaptation |
| Model maintenance | High | Low |
| Market coverage | Limited to trained assets | Expandable on-the-fly |

## Theoretical Foundation

### Meta-Learning Framework

Few-shot market prediction is built on meta-learning, also known as "learning to learn." Instead of training a model for a specific prediction task, we train a model that can quickly adapt to new tasks.

### Mathematical Formulation

**Task Distribution**: Let $p(\mathcal{T})$ be a distribution over trading tasks (e.g., different assets, timeframes, or prediction targets).

**Support Set**: For each task $\mathcal{T}_i$, we have a support set $\mathcal{S}_i = \{(x_j, y_j)\}_{j=1}^K$ with $K$ examples (K-shot).

**Query Set**: We evaluate on a query set $\mathcal{Q}_i = \{(x_j, y_j)\}_{j=1}^Q$.

**Meta-Learning Objective**:

$$\min_\theta \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})} \left[ \mathcal{L}(\mathcal{Q}_i; f_{\theta'_i}) \right]$$

where $\theta'_i = \text{adapt}(\theta, \mathcal{S}_i)$ represents the adapted parameters for task $i$.

### Episodic Training

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Episodic Training for Market Prediction                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Episode Generation:                                                       │
│   ──────────────────                                                        │
│                                                                             │
│   Step 1: Sample N tasks (e.g., N different assets)                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Task 1: BTC price direction prediction                              │  │
│   │  Task 2: ETH volatility forecasting                                  │  │
│   │  Task 3: SOL trend classification                                    │  │
│   │  Task 4: AVAX regime detection                                       │  │
│   │  Task 5: DOT support/resistance prediction                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Step 2: For each task, create support and query sets                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  BTC Task:                                                           │  │
│   │  Support: [Day 1-5 features → labels]   (5-shot)                    │  │
│   │  Query:   [Day 6-10 features → labels]  (5 query examples)          │  │
│   │                                                                      │  │
│   │  ETH Task:                                                           │  │
│   │  Support: [Day 1-5 features → labels]   (5-shot)                    │  │
│   │  Query:   [Day 6-10 features → labels]  (5 query examples)          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Step 3: Meta-learning update                                              │
│   • Adapt model to each task using support set                             │
│   • Evaluate on query set                                                   │
│   • Update meta-parameters to improve adaptation                           │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### N-way K-shot Problem Formulation

For market prediction, we typically frame problems as:

- **N-way Classification**: Classify into N categories (e.g., Up/Down/Sideways)
- **K-shot**: Use K examples per category in the support set
- **Regression**: Predict continuous values (price, volatility)

Common configurations for trading:
- 3-way 5-shot: Up/Down/Sideways with 5 examples each
- 5-way 10-shot: Five market regimes with 10 examples each
- 1-shot regression: Predict returns with 1 reference example

## Few-Shot Learning Approaches

### 1. Metric-Based Methods

Learn a metric space where similar market conditions cluster together.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Metric-Based Few-Shot Learning                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Core Idea: Learn embeddings where distance = similarity                   │
│                                                                             │
│   Embedding Space Visualization:                                            │
│                                                                             │
│              ↑ Dimension 2                                                  │
│              │                                                              │
│              │    ★ Query (new asset)                                      │
│              │   /                                                          │
│              │  / distance                                                  │
│              │ /                                                            │
│    ●─●─●────●────────●                                                     │
│    Support examples   │                                                     │
│    (same class)       │                                                     │
│              │        ○──○──○                                              │
│              │        Support examples                                      │
│              │        (different class)                                     │
│              └────────────────────────→ Dimension 1                        │
│                                                                             │
│   Methods:                                                                  │
│   ─────────                                                                 │
│   • Siamese Networks: Learn pairwise similarity                            │
│   • Prototypical Networks: Compare to class prototypes (centroids)         │
│   • Matching Networks: Attention-weighted similarity                        │
│   • Relation Networks: Learn similarity function with neural network       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2. Optimization-Based Methods (MAML Family)

Learn initial parameters that can be quickly fine-tuned.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Model-Agnostic Meta-Learning (MAML)                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Core Idea: Find initialization θ that adapts quickly                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │           θ (initial parameters)                                    │  │
│   │           │                                                          │  │
│   │     ┌─────┼─────┬─────────┐                                         │  │
│   │     │     │     │         │                                         │  │
│   │     ↓     ↓     ↓         ↓                                         │  │
│   │    θ'₁   θ'₂   θ'₃   ... θ'ₙ  (task-specific params)               │  │
│   │     │     │     │         │                                         │  │
│   │   Task1 Task2 Task3   TaskN  (inner loop: few gradient steps)       │  │
│   │     │     │     │         │                                         │  │
│   │     └─────┴─────┴─────────┘                                         │  │
│   │           │                                                          │  │
│   │           ↓                                                          │  │
│   │    Meta-update θ (outer loop: optimize for fast adaptation)         │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Algorithm:                                                                │
│   ──────────                                                                │
│   1. Sample batch of tasks                                                 │
│   2. For each task:                                                        │
│      - Adapt θ → θ' using support set (1-5 gradient steps)               │
│      - Compute loss on query set with θ'                                  │
│   3. Update θ using sum of query losses                                   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3. Hybrid Approach for Market Prediction

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Hybrid Few-Shot Market Predictor                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Market Features                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Price features: returns, volatility, momentum                       │  │
│   │  Volume features: volume, VWAP, order flow                          │  │
│   │  Technical indicators: RSI, MACD, Bollinger Bands                   │  │
│   │  Crypto-specific: funding rate, OI, liquidations                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ↓                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Temporal Encoder (shared across tasks)                              │  │
│   │  ───────────────────────────────────────                            │  │
│   │  • 1D-CNN for local patterns                                        │  │
│   │  • LSTM/Transformer for sequence modeling                           │  │
│   │  • Attention mechanism for important features                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ↓                                                │
│   ┌────────────────────────┴────────────────────────┐                      │
│   │                                                  │                      │
│   ↓                                                  ↓                      │
│   Metric-Based Path                          Optimization-Based Path       │
│   ┌────────────────────┐                    ┌────────────────────┐        │
│   │ Prototype Matching │                    │ MAML Adaptation    │        │
│   │ (fast, no grad)    │                    │ (accurate, grad)   │        │
│   └────────────────────┘                    └────────────────────┘        │
│            │                                          │                    │
│            └───────────────┬──────────────────────────┘                    │
│                            │                                                │
│                            ↓                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Ensemble / Confidence-Weighted Output                               │  │
│   │  ─────────────────────────────────────                              │  │
│   │  High confidence: Use metric-based (faster)                         │  │
│   │  Low confidence: Use optimization-based (more accurate)             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ↓                                                │
│   Output: Prediction + Confidence Score                                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Design

### Feature Extraction Module

```rust
/// Market feature types for few-shot learning
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Price-based features
    pub price_features: PriceFeatures,
    /// Volume-based features
    pub volume_features: VolumeFeatures,
    /// Technical indicators
    pub technical_features: TechnicalFeatures,
    /// Crypto-specific features
    pub crypto_features: CryptoFeatures,
}

#[derive(Debug, Clone)]
pub struct PriceFeatures {
    /// Returns over multiple horizons
    pub returns: Vec<f32>,           // [1m, 5m, 15m, 1h, 4h, 24h]
    /// Logarithmic returns
    pub log_returns: Vec<f32>,
    /// Realized volatility
    pub volatility: Vec<f32>,        // Multiple windows
    /// High-low range normalized
    pub range: f32,
    /// Close position in range [0, 1]
    pub close_position: f32,
    /// Price momentum
    pub momentum: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct VolumeFeatures {
    /// Normalized volume
    pub volume_normalized: f32,
    /// Volume change rate
    pub volume_change: f32,
    /// Buy/Sell volume ratio
    pub buy_sell_ratio: f32,
    /// Volume-weighted average price deviation
    pub vwap_deviation: f32,
    /// Volume profile features
    pub volume_profile: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TechnicalFeatures {
    /// RSI values at different periods
    pub rsi: Vec<f32>,
    /// MACD line, signal, histogram
    pub macd: [f32; 3],
    /// Bollinger Band position
    pub bb_position: f32,
    /// Moving average crossover signals
    pub ma_signals: Vec<f32>,
    /// ATR normalized
    pub atr: f32,
}

#[derive(Debug, Clone)]
pub struct CryptoFeatures {
    /// Funding rate (for perpetuals)
    pub funding_rate: f32,
    /// Open interest change
    pub oi_change: f32,
    /// Long/Short ratio
    pub long_short_ratio: f32,
    /// Recent liquidation volume
    pub liquidation_volume: f32,
    /// Spot-Futures basis
    pub basis: f32,
}
```

### Embedding Network Architecture

```rust
/// Configuration for the embedding network
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Sequence length (number of time steps)
    pub seq_length: usize,
    /// Hidden dimension for temporal encoder
    pub hidden_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_dim: 48,          // Total feature count
            seq_length: 96,         // 96 time steps (e.g., 96 hours)
            hidden_dim: 256,
            embedding_dim: 128,
            num_heads: 8,
            dropout: 0.1,
        }
    }
}

/// Embedding network structure
pub struct EmbeddingNetwork {
    config: EmbeddingConfig,
    // Temporal convolution layers
    conv_layers: Vec<Conv1dLayer>,
    // Transformer/LSTM encoder
    temporal_encoder: TemporalEncoder,
    // Projection head
    projection: ProjectionHead,
}

impl EmbeddingNetwork {
    /// Forward pass through the embedding network
    pub fn forward(&self, x: &Array3<f32>) -> Array2<f32> {
        // x shape: (batch, seq_length, features)

        // 1. Temporal convolutions for local patterns
        let conv_out = self.apply_conv_layers(x);

        // 2. Temporal encoding (LSTM or Transformer)
        let temporal_out = self.temporal_encoder.forward(&conv_out);

        // 3. Project to embedding space
        let embeddings = self.projection.forward(&temporal_out);

        // 4. L2 normalize embeddings
        self.l2_normalize(&embeddings)
    }

    fn l2_normalize(&self, x: &Array2<f32>) -> Array2<f32> {
        let norms = x.mapv(|v| v * v)
            .sum_axis(Axis(1))
            .mapv(f32::sqrt);

        x / &norms.insert_axis(Axis(1))
    }
}
```

### Few-Shot Predictor Module

```rust
/// Few-shot market predictor configuration
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Number of classes/categories
    pub n_way: usize,
    /// Number of support examples per class
    pub k_shot: usize,
    /// Number of query examples for evaluation
    pub n_query: usize,
    /// Embedding configuration
    pub embedding_config: EmbeddingConfig,
    /// Distance metric type
    pub distance_type: DistanceType,
    /// Temperature for softmax
    pub temperature: f32,
    /// Whether to use MAML adaptation
    pub use_maml: bool,
    /// MAML inner loop learning rate
    pub maml_inner_lr: f32,
    /// MAML inner loop steps
    pub maml_inner_steps: usize,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            n_way: 3,               // Up, Down, Sideways
            k_shot: 5,              // 5 examples per class
            n_query: 15,            // 15 query examples
            embedding_config: EmbeddingConfig::default(),
            distance_type: DistanceType::SquaredEuclidean,
            temperature: 1.0,
            use_maml: false,        // Start with metric-based
            maml_inner_lr: 0.01,
            maml_inner_steps: 5,
        }
    }
}

/// Distance metric types
#[derive(Debug, Clone, Copy)]
pub enum DistanceType {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    Learned,  // Relation network style
}

/// Prediction result with confidence
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted class/category
    pub prediction: usize,
    /// Class probabilities
    pub probabilities: Vec<f32>,
    /// Confidence score (max probability)
    pub confidence: f32,
    /// Distance to each prototype
    pub distances: Vec<f32>,
}
```

## Application to Market Prediction

### Prediction Tasks

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Few-Shot Market Prediction Tasks                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Task Type 1: DIRECTION PREDICTION (Classification)                       │
│   ─────────────────────────────────────────────────                        │
│   Predict: Price direction in next N periods                               │
│   Classes: [Up, Down, Sideways]                                            │
│   N-way: 3                                                                  │
│   K-shot: 5-10                                                              │
│                                                                             │
│   Support Set Example:                                                      │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Class "Up":                                                        │   │
│   │  • Period 1: features → return +2.3%                               │   │
│   │  • Period 2: features → return +1.8%                               │   │
│   │  • Period 3: features → return +3.1%                               │   │
│   │  ...                                                                │   │
│   │  Class "Down":                                                      │   │
│   │  • Period 1: features → return -2.1%                               │   │
│   │  • Period 2: features → return -1.5%                               │   │
│   │  ...                                                                │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Task Type 2: REGIME DETECTION (Classification)                           │
│   ─────────────────────────────────────────────                            │
│   Predict: Current market regime                                           │
│   Classes: [Bull, Bear, Sideways, High_Vol, Low_Vol]                       │
│   N-way: 5                                                                  │
│   K-shot: 5-10                                                              │
│                                                                             │
│   Task Type 3: VOLATILITY FORECAST (Regression)                            │
│   ─────────────────────────────────────────────                            │
│   Predict: Future realized volatility                                      │
│   Output: Continuous value                                                  │
│   K-shot: 5-20 reference periods                                           │
│                                                                             │
│   Task Type 4: SUPPORT/RESISTANCE (Classification)                         │
│   ─────────────────────────────────────────────────                        │
│   Predict: Will price bounce or break?                                     │
│   Classes: [Bounce, Break]                                                  │
│   N-way: 2                                                                  │
│   K-shot: 5-10                                                              │
│                                                                             │
│   Task Type 5: CROSS-ASSET TRANSFER                                        │
│   ─────────────────────────────────                                        │
│   Train on: BTC, ETH, SOL, AVAX                                            │
│   Adapt to: New token with 5 examples                                      │
│   Use Case: Trade new listings profitably                                   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Trading Strategy Integration

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Few-Shot Trading Strategy Pipeline                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 1: SUPPORT SET CONSTRUCTION (Online)                               │
│   ───────────────────────────────────────────                              │
│   • Monitor recent market data for asset                                   │
│   • Label recent periods based on realized outcomes                        │
│   • Create support set with balanced class representation                  │
│   • Update support set as new data arrives (sliding window)               │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Rolling Support Set for BTC:                                       │   │
│   │  ┌──────────────────────────────────────────────────────────────┐  │   │
│   │  │ t-48h │ t-47h │ ... │ t-25h │ ... │ t-1h │ Current           │  │   │
│   │  │  Up   │  Up   │     │ Down  │     │ Side │     ?             │  │   │
│   │  └──────────────────────────────────────────────────────────────┘  │   │
│   │  Support: Last 48 labeled periods                                   │   │
│   │  Query: Current period (to predict)                                 │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Phase 2: PREDICTION (Real-time)                                          │
│   ────────────────────────────────                                         │
│   • Extract features for current period                                    │
│   • Compute embedding                                                      │
│   • Compare to prototypes/adapted model                                    │
│   • Generate prediction with confidence                                    │
│                                                                             │
│   Phase 3: SIGNAL GENERATION                                               │
│   ──────────────────────────                                               │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Prediction → Trading Signal Mapping                                │   │
│   │                                                                      │   │
│   │  UP (conf > 0.7)    →  LONG signal, size = conf × base_size        │   │
│   │  DOWN (conf > 0.7)  →  SHORT signal, size = conf × base_size       │   │
│   │  SIDEWAYS           →  No trade or range-bound strategy             │   │
│   │  Low confidence     →  Reduce position size or skip                 │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Phase 4: POSITION MANAGEMENT                                             │
│   ────────────────────────────                                             │
│   • Enter position based on signal                                         │
│   • Set stop-loss and take-profit based on regime                         │
│   • Monitor prediction confidence continuously                             │
│   • Exit when: target hit, stop hit, or confidence drops                  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### New Asset Adaptation Protocol

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    New Asset Adaptation Protocol                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   When a new asset is listed (e.g., new token on Bybit):                   │
│                                                                             │
│   Hour 0-1: DATA COLLECTION                                                │
│   ─────────────────────────                                                │
│   • Start collecting OHLCV data                                            │
│   • Monitor order book dynamics                                            │
│   • Record funding rates (if perpetual)                                    │
│                                                                             │
│   Hour 1-4: INITIAL SUPPORT SET                                            │
│   ──────────────────────────────                                           │
│   • Label first periods based on realized returns                          │
│   • Create minimal support set (3-5 examples per class)                   │
│   • Use conservative predictions (high confidence threshold)               │
│                                                                             │
│   Hour 4-24: ACTIVE LEARNING                                               │
│   ──────────────────────────                                               │
│   • Expand support set with new labeled examples                          │
│   • Lower confidence threshold as data accumulates                        │
│   • Start small position sizing                                            │
│                                                                             │
│   After 24h: FULL ADAPTATION                                               │
│   ──────────────────────────                                               │
│   • Support set reaches target size (20-50 examples)                      │
│   • Full position sizing enabled                                           │
│   • Continuous support set updates (sliding window)                       │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Confidence Threshold Schedule                                      │   │
│   │                                                                      │   │
│   │  100% ─────────────────────────────────────────                    │   │
│   │   95% ▓▓▓▓▓                                                        │   │
│   │   90% ▓▓▓▓▓▓▓▓▓                                                    │   │
│   │   85%           ▓▓▓▓▓▓                                             │   │
│   │   80%                 ▓▓▓▓▓▓▓▓                                     │   │
│   │   75%                         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓             │   │
│   │   70%                                               ▓▓▓▓▓▓▓▓▓▓▓   │   │
│   │       ────┬────┬────┬────┬────┬────┬────┬────┬────┬────→ Hours    │   │
│   │           1    4    8   12   16   20   24   36   48   72           │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Module Structure

```
86_few_shot_market_prediction/
├── Cargo.toml
├── README.md                     # Main documentation (English)
├── README.ru.md                  # Russian translation
├── readme.simple.md              # Simplified explanation
├── readme.simple.ru.md           # Simplified Russian
├── src/
│   ├── lib.rs                    # Library root
│   ├── config.rs                 # Configuration types
│   ├── features/
│   │   ├── mod.rs               # Feature module
│   │   ├── extractor.rs         # Feature extraction
│   │   ├── normalizer.rs        # Feature normalization
│   │   └── types.rs             # Feature types
│   ├── embedding/
│   │   ├── mod.rs               # Embedding module
│   │   ├── network.rs           # Embedding network
│   │   ├── temporal.rs          # Temporal encoding
│   │   └── attention.rs         # Attention mechanisms
│   ├── fewshot/
│   │   ├── mod.rs               # Few-shot module
│   │   ├── prototypical.rs      # Prototypical networks
│   │   ├── maml.rs              # MAML implementation
│   │   ├── episode.rs           # Episode generation
│   │   └── predictor.rs         # Prediction interface
│   ├── data/
│   │   ├── mod.rs               # Data module
│   │   ├── bybit.rs             # Bybit API client
│   │   ├── dataset.rs           # Dataset handling
│   │   └── support_set.rs       # Support set management
│   ├── trading/
│   │   ├── mod.rs               # Trading module
│   │   ├── signals.rs           # Signal generation
│   │   ├── strategy.rs          # Trading strategy
│   │   └── risk.rs              # Risk management
│   └── utils/
│       ├── mod.rs               # Utilities
│       ├── metrics.rs           # Evaluation metrics
│       └── math.rs              # Math utilities
├── examples/
│   ├── basic_fewshot.rs         # Basic few-shot example
│   ├── direction_prediction.rs  # Direction prediction demo
│   ├── new_asset_trading.rs     # New asset adaptation
│   └── backtest.rs              # Backtesting example
├── python/
│   ├── few_shot_predictor.py    # PyTorch implementation
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── requirements.txt         # Python dependencies
└── tests/
    ├── integration.rs           # Integration tests
    └── unit_tests.rs            # Unit tests
```

### Core Implementation (Rust)

```rust
//! Few-Shot Market Prediction Core Module

use ndarray::{Array1, Array2, Array3, Axis};
use std::collections::HashMap;

/// Market prediction categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictionCategory {
    /// Price expected to increase
    Up,
    /// Price expected to decrease
    Down,
    /// Price expected to remain stable
    Sideways,
}

impl PredictionCategory {
    /// Convert to trading bias
    pub fn to_trading_bias(&self) -> TradingBias {
        match self {
            Self::Up => TradingBias::Long,
            Self::Down => TradingBias::Short,
            Self::Sideways => TradingBias::Neutral,
        }
    }

    /// Get all categories
    pub fn all() -> Vec<Self> {
        vec![Self::Up, Self::Down, Self::Sideways]
    }
}

/// Trading bias based on prediction
#[derive(Debug, Clone, Copy)]
pub enum TradingBias {
    Long,
    Short,
    Neutral,
}

/// Support set for few-shot learning
#[derive(Debug, Clone)]
pub struct SupportSet {
    /// Features for each example: (n_examples, seq_len, feature_dim)
    pub features: Array3<f32>,
    /// Labels for each example
    pub labels: Vec<PredictionCategory>,
    /// Timestamps for each example
    pub timestamps: Vec<u64>,
    /// Maximum examples per class
    max_per_class: usize,
}

impl SupportSet {
    /// Create a new empty support set
    pub fn new(max_per_class: usize) -> Self {
        Self {
            features: Array3::zeros((0, 0, 0)),
            labels: Vec::new(),
            timestamps: Vec::new(),
            max_per_class,
        }
    }

    /// Add a new example to the support set
    pub fn add_example(
        &mut self,
        features: Array2<f32>,
        label: PredictionCategory,
        timestamp: u64,
    ) {
        // Count current examples for this class
        let class_count = self.labels.iter()
            .filter(|&l| *l == label)
            .count();

        // If at max capacity, remove oldest example of this class
        if class_count >= self.max_per_class {
            self.remove_oldest_of_class(label);
        }

        // Add new example
        self.labels.push(label);
        self.timestamps.push(timestamp);

        // Concatenate features
        // (Implementation depends on actual array handling)
    }

    /// Remove the oldest example of a given class
    fn remove_oldest_of_class(&mut self, class: PredictionCategory) {
        if let Some(idx) = self.labels.iter()
            .enumerate()
            .filter(|(_, l)| **l == class)
            .min_by_key(|(i, _)| self.timestamps[*i])
            .map(|(i, _)| i)
        {
            self.labels.remove(idx);
            self.timestamps.remove(idx);
            // Remove from features array
        }
    }

    /// Get examples for a specific class
    pub fn get_class_examples(&self, class: PredictionCategory) -> Vec<usize> {
        self.labels.iter()
            .enumerate()
            .filter(|(_, l)| **l == class)
            .map(|(i, _)| i)
            .collect()
    }

    /// Check if support set is ready for prediction
    pub fn is_ready(&self, min_per_class: usize) -> bool {
        PredictionCategory::all().iter().all(|class| {
            self.get_class_examples(*class).len() >= min_per_class
        })
    }
}

/// Few-shot market predictor
pub struct FewShotPredictor {
    /// Configuration
    config: FewShotConfig,
    /// Embedding network
    embedding_network: EmbeddingNetwork,
    /// Current support set
    support_set: SupportSet,
    /// Cached prototypes for each class
    prototypes: HashMap<PredictionCategory, Array1<f32>>,
}

impl FewShotPredictor {
    /// Create a new predictor
    pub fn new(config: FewShotConfig) -> Self {
        Self {
            embedding_network: EmbeddingNetwork::new(config.embedding_config.clone()),
            support_set: SupportSet::new(config.k_shot * 2), // 2x buffer
            prototypes: HashMap::new(),
            config,
        }
    }

    /// Update support set with new labeled example
    pub fn update_support_set(
        &mut self,
        features: Array2<f32>,
        label: PredictionCategory,
        timestamp: u64,
    ) {
        self.support_set.add_example(features, label, timestamp);
        self.update_prototypes();
    }

    /// Update prototypes from current support set
    fn update_prototypes(&mut self) {
        self.prototypes.clear();

        for class in PredictionCategory::all() {
            let indices = self.support_set.get_class_examples(class);
            if indices.is_empty() {
                continue;
            }

            // Get embeddings for all examples of this class
            // Compute prototype as mean embedding
            // (Simplified - actual implementation would batch process)
        }
    }

    /// Make prediction for new query
    pub fn predict(&self, query_features: &Array2<f32>) -> PredictionResult {
        // 1. Embed query
        let query_embedding = self.embedding_network.forward_single(query_features);

        // 2. Compute distances to all prototypes
        let mut distances = HashMap::new();
        for (class, prototype) in &self.prototypes {
            let dist = self.compute_distance(&query_embedding, prototype);
            distances.insert(*class, dist);
        }

        // 3. Convert distances to probabilities
        let (probabilities, prediction, confidence) =
            self.distances_to_probabilities(&distances);

        PredictionResult {
            prediction: prediction as usize,
            probabilities,
            confidence,
            distances: distances.values().cloned().collect(),
        }
    }

    /// Compute distance between two embeddings
    fn compute_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        match self.config.distance_type {
            DistanceType::Euclidean => {
                (a - b).mapv(|x| x * x).sum().sqrt()
            }
            DistanceType::SquaredEuclidean => {
                (a - b).mapv(|x| x * x).sum()
            }
            DistanceType::Cosine => {
                let dot = a.dot(b);
                let norm_a = a.mapv(|x| x * x).sum().sqrt();
                let norm_b = b.mapv(|x| x * x).sum().sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
            DistanceType::Learned => {
                // Would use a learned relation network
                unimplemented!("Learned distance requires relation network")
            }
        }
    }

    /// Convert distances to probabilities via softmax
    fn distances_to_probabilities(
        &self,
        distances: &HashMap<PredictionCategory, f32>,
    ) -> (Vec<f32>, PredictionCategory, f32) {
        let classes: Vec<_> = distances.keys().cloned().collect();
        let neg_distances: Vec<f32> = classes.iter()
            .map(|c| -distances[c] / self.config.temperature)
            .collect();

        // Softmax
        let max_val = neg_distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = neg_distances.iter()
            .map(|x| (x - max_val).exp())
            .collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probabilities: Vec<f32> = exp_vals.iter()
            .map(|x| x / sum_exp)
            .collect();

        // Find prediction (class with highest probability)
        let (max_idx, &max_prob) = probabilities.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        (probabilities, classes[max_idx], max_prob)
    }
}
```

## Bybit Integration

### Data Collection Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Bybit Data Pipeline for Few-Shot Learning                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. HISTORICAL DATA (Meta-Training)                                       │
│   ──────────────────────────────────                                       │
│   Purpose: Train the base model on many assets                             │
│                                                                             │
│   API Endpoints:                                                            │
│   • GET /v5/market/kline → OHLCV candles for multiple symbols             │
│   • GET /v5/market/tickers → Current market state                          │
│   • GET /v5/market/funding/history → Funding rate history                  │
│   • GET /v5/market/open-interest → Open interest history                   │
│                                                                             │
│   Data Collection Strategy:                                                 │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Symbols: BTC, ETH, SOL, AVAX, DOT, LINK, UNI, ATOM, ...          │   │
│   │  Timeframe: 1h candles                                              │   │
│   │  History: 1 year per symbol                                         │   │
│   │  Task sampling: Different windows as different tasks                │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   2. REAL-TIME DATA (Inference/Adaptation)                                 │
│   ────────────────────────────────────────                                 │
│   Purpose: Build support sets and make predictions                         │
│                                                                             │
│   WebSocket Subscriptions:                                                  │
│   • kline.1.{symbol} → 1-minute candles for feature updates               │
│   • ticker.{symbol} → Real-time price for signal execution                │
│   • liquidation.{symbol} → Liquidation events for features                │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Real-time Loop:                                                    │   │
│   │  1. Receive new candle via WebSocket                               │   │
│   │  2. Update feature buffer                                          │   │
│   │  3. Every hour: Label previous period, update support set          │   │
│   │  4. Make prediction for next period                                │   │
│   │  5. Generate trading signal if confidence > threshold              │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   3. ORDER EXECUTION                                                        │
│   ──────────────────                                                       │
│   • POST /v5/order/create → Place market/limit orders                      │
│   • GET /v5/position/list → Check current positions                        │
│   • POST /v5/order/cancel → Cancel pending orders                          │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Bybit Client Implementation

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_tungstenite::WebSocketStream;

/// Bybit API client for few-shot learning
pub struct BybitClient {
    /// HTTP client for REST API
    http_client: Client,
    /// API key (optional for public endpoints)
    api_key: Option<String>,
    /// API secret
    api_secret: Option<String>,
    /// Base URL for REST API
    base_url: String,
    /// WebSocket URL
    ws_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(api_key: Option<String>, api_secret: Option<String>) -> Self {
        Self {
            http_client: Client::new(),
            api_key,
            api_secret,
            base_url: "https://api.bybit.com".to_string(),
            ws_url: "wss://stream.bybit.com/v5/public/linear".to_string(),
        }
    }

    /// Fetch historical klines for meta-training
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: u64,
        end_time: u64,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&end={}&limit={}",
            self.base_url, symbol, interval, start_time, end_time, limit
        );

        let response: KlineResponse = self.http_client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result.list.into_iter().map(Kline::from).collect())
    }

    /// Fetch funding rate history
    pub async fn get_funding_history(
        &self,
        symbol: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<FundingRate>, BybitError> {
        let url = format!(
            "{}/v5/market/funding/history?category=linear&symbol={}&startTime={}&endTime={}",
            self.base_url, symbol, start_time, end_time
        );

        let response: FundingResponse = self.http_client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result.list)
    }

    /// Subscribe to real-time kline updates
    pub async fn subscribe_klines(
        &self,
        symbols: &[&str],
        interval: &str,
    ) -> Result<WebSocketStream<impl tokio::io::AsyncRead + tokio::io::AsyncWrite>, BybitError> {
        use tokio_tungstenite::connect_async;

        let (ws_stream, _) = connect_async(&self.ws_url).await?;

        // Send subscription message
        let topics: Vec<String> = symbols.iter()
            .map(|s| format!("kline.{}.{}", interval, s))
            .collect();

        let subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": topics
        });

        // ... send subscription and return stream

        Ok(ws_stream)
    }
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Funding rate data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_rate_timestamp: u64,
}
```

## Risk Management

### Confidence-Based Position Sizing

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Risk Management for Few-Shot Trading                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. CONFIDENCE-BASED POSITION SIZING                                      │
│   ─────────────────────────────────────                                    │
│                                                                             │
│   Position Size = Base Size × Confidence × Regime Multiplier               │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Confidence Thresholds:                                             │   │
│   │                                                                      │   │
│   │  Confidence < 0.50:  NO TRADE (too uncertain)                       │   │
│   │  Confidence 0.50-0.65: 25% of base size                            │   │
│   │  Confidence 0.65-0.75: 50% of base size                            │   │
│   │  Confidence 0.75-0.85: 75% of base size                            │   │
│   │  Confidence 0.85-0.95: 100% of base size                           │   │
│   │  Confidence > 0.95:    100% of base size (cap to avoid overfit)    │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   2. SUPPORT SET QUALITY ASSESSMENT                                        │
│   ─────────────────────────────────                                        │
│                                                                             │
│   Quality Score = f(class_balance, recency, diversity)                     │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Quality Factors:                                                   │   │
│   │                                                                      │   │
│   │  • Class Balance: All classes should have similar # examples       │   │
│   │  • Recency: Recent examples should be weighted higher              │   │
│   │  • Diversity: Examples should cover different market conditions    │   │
│   │  • Consistency: Similar examples should have similar labels        │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   3. STOP-LOSS AND TAKE-PROFIT                                             │
│   ────────────────────────────                                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  Prediction    │ Stop-Loss │ Take-Profit │ Max Hold Time        │      │
│   │────────────────┼───────────┼─────────────┼──────────────────────│      │
│   │  UP (high conf)│ -2.0%     │ +3.0%       │ 4 hours              │      │
│   │  UP (med conf) │ -1.5%     │ +2.0%       │ 2 hours              │      │
│   │  DOWN (high)   │ -2.0%     │ +3.0%       │ 4 hours              │      │
│   │  DOWN (med)    │ -1.5%     │ +2.0%       │ 2 hours              │      │
│   │  SIDEWAYS      │ -1.0%     │ +1.0%       │ 1 hour               │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   4. CIRCUIT BREAKERS                                                       │
│   ───────────────────                                                       │
│                                                                             │
│   Automatic position closure when:                                          │
│   • Confidence drops below 0.40 during position hold                       │
│   • Prediction changes direction (UP → DOWN or vice versa)                 │
│   • Support set becomes stale (>4h since last update)                      │
│   • Daily loss exceeds 3% of capital                                       │
│   • 5 consecutive losing trades                                             │
│                                                                             │
│   5. NEW ASSET RISK SCALING                                                 │
│   ──────────────────────────                                               │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Hours Since Listing │ Max Position Size                           │   │
│   │──────────────────────┼──────────────────────────────────────────── │   │
│   │  0-4 hours           │ 10% of normal                               │   │
│   │  4-12 hours          │ 25% of normal                               │   │
│   │  12-24 hours         │ 50% of normal                               │   │
│   │  24-48 hours         │ 75% of normal                               │   │
│   │  48+ hours           │ 100% of normal                              │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Risk Parameters Implementation

```rust
/// Risk parameters for few-shot trading
#[derive(Debug, Clone)]
pub struct RiskParameters {
    /// Base position size as fraction of capital
    pub base_position_size: f32,
    /// Maximum position size multiplier
    pub max_position_multiplier: f32,
    /// Minimum confidence to trade
    pub min_confidence: f32,
    /// Stop-loss percentage
    pub stop_loss_pct: f32,
    /// Take-profit percentage
    pub take_profit_pct: f32,
    /// Maximum holding time in seconds
    pub max_hold_time: u64,
    /// Daily loss limit as fraction of capital
    pub daily_loss_limit: f32,
    /// Maximum consecutive losses before pause
    pub max_consecutive_losses: usize,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            base_position_size: 0.02,     // 2% of capital per trade
            max_position_multiplier: 3.0,  // Maximum 6% per trade
            min_confidence: 0.50,
            stop_loss_pct: 0.02,           // 2% stop loss
            take_profit_pct: 0.03,         // 3% take profit
            max_hold_time: 4 * 3600,       // 4 hours
            daily_loss_limit: 0.03,        // 3% daily loss limit
            max_consecutive_losses: 5,
        }
    }
}

impl RiskParameters {
    /// Calculate position size based on confidence
    pub fn calculate_position_size(&self, confidence: f32, capital: f32) -> f32 {
        if confidence < self.min_confidence {
            return 0.0;
        }

        // Scale position size with confidence
        let confidence_multiplier = match confidence {
            c if c < 0.65 => 0.25,
            c if c < 0.75 => 0.50,
            c if c < 0.85 => 0.75,
            _ => 1.00,
        };

        let position_size = self.base_position_size
            * confidence_multiplier
            * self.max_position_multiplier;

        capital * position_size.min(self.base_position_size * self.max_position_multiplier)
    }

    /// Scale parameters for new assets
    pub fn scale_for_new_asset(&self, hours_since_listing: f32) -> Self {
        let scale_factor = match hours_since_listing {
            h if h < 4.0 => 0.10,
            h if h < 12.0 => 0.25,
            h if h < 24.0 => 0.50,
            h if h < 48.0 => 0.75,
            _ => 1.00,
        };

        Self {
            base_position_size: self.base_position_size * scale_factor,
            min_confidence: self.min_confidence + (1.0 - scale_factor) * 0.2,
            stop_loss_pct: self.stop_loss_pct * (1.0 + (1.0 - scale_factor) * 0.5),
            ..*self
        }
    }
}
```

## Performance Metrics

### Model Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Few-Shot Accuracy** | Classification accuracy with K examples | > 65% |
| **1-Shot Accuracy** | Accuracy with single example per class | > 55% |
| **5-Shot Accuracy** | Accuracy with 5 examples per class | > 70% |
| **Cross-Asset Transfer** | Accuracy on unseen assets | > 60% |
| **Adaptation Speed** | Episodes needed to reach target accuracy | < 10 |
| **Confidence Calibration** | Reliability of confidence scores | ECE < 0.10 |

### Trading Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted returns | > 2.0 |
| **Sortino Ratio** | Downside risk-adjusted returns | > 2.5 |
| **Max Drawdown** | Largest peak-to-trough decline | < 15% |
| **Win Rate** | Percentage of profitable trades | > 55% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **New Asset ROI** | Returns on newly listed assets | > 10% monthly |

### Latency Budget

```
┌─────────────────────────────────────────────────┐
│              Latency Requirements               │
├─────────────────────────────────────────────────┤
│ Feature Extraction:        < 5ms                │
│ Embedding Computation:     < 15ms               │
│ Prototype Distance:        < 3ms                │
│ Prediction:                < 2ms                │
├─────────────────────────────────────────────────┤
│ Total Inference:           < 25ms               │
├─────────────────────────────────────────────────┤
│ Support Set Update:        < 50ms               │
│ MAML Adaptation (if used): < 500ms              │
└─────────────────────────────────────────────────┘
```

## References

1. **Short-Term Stock Price-Trend Prediction Using Meta-Learning**
   - URL: https://arxiv.org/abs/2105.13599
   - Year: 2021

2. **Prototypical Networks for Few-shot Learning**
   - Snell, J., Swersky, K., & Zemel, R. (2017)
   - URL: https://arxiv.org/abs/1703.05175

3. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**
   - Finn, C., Abbeel, P., & Levine, S. (2017)
   - URL: https://arxiv.org/abs/1703.03400

4. **Matching Networks for One Shot Learning**
   - Vinyals, O., et al. (2016)
   - URL: https://arxiv.org/abs/1606.04080

5. **Meta-Learning: A Survey**
   - Hospedales, T., et al. (2020)
   - URL: https://arxiv.org/abs/2004.05439

6. **Few-Shot Learning for Time Series Forecasting**
   - Recent advances in applying meta-learning to temporal data

7. **Cross-Asset Learning in Financial Markets**
   - Transfer learning approaches for multi-asset trading

---

## Next Steps

- [Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Russian translation
- [Run Examples](examples/) - Working Rust code
- [Python Implementation](python/) - PyTorch reference implementation

---

*Chapter 86 of Machine Learning for Trading*
