# Chapter 87: Task-Agnostic Trading

## Overview

Task-Agnostic Trading develops universal models that can perform well across multiple trading tasks without requiring task-specific fine-tuning. Unlike traditional approaches that train separate models for price prediction, volatility forecasting, regime detection, and portfolio optimization, task-agnostic models learn general representations that transfer effectively to any downstream task.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Task-Agnostic Architectures](#task-agnostic-architectures)
4. [Universal Feature Learning](#universal-feature-learning)
5. [Multi-Task Training Strategy](#multi-task-training-strategy)
6. [Application to Trading](#application-to-trading)
7. [Bybit Integration](#bybit-integration)
8. [Implementation](#implementation)
9. [Risk Management](#risk-management)
10. [Performance Metrics](#performance-metrics)
11. [References](#references)

---

## Introduction

Traditional machine learning for trading faces a fundamental problem: models are typically task-specific. A model trained for price direction prediction cannot be directly used for volatility forecasting or regime detection. This requires maintaining multiple models, each with its own training pipeline, hyperparameters, and maintenance overhead.

### The Task-Specific Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Traditional Task-Specific Approach                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SEPARATE MODELS FOR EACH TASK:                                            │
│   ───────────────────────────────                                           │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │  Price Direction │   │   Volatility    │   │     Regime      │          │
│   │    Predictor     │   │   Forecaster    │   │    Detector     │          │
│   ├─────────────────┤   ├─────────────────┤   ├─────────────────┤          │
│   │ • Own features  │   │ • Own features  │   │ • Own features  │          │
│   │ • Own training  │   │ • Own training  │   │ • Own training  │          │
│   │ • Own tuning    │   │ • Own tuning    │   │ • Own tuning    │          │
│   │ • Own deployment│   │ • Own deployment│   │ • Own deployment│          │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘          │
│                                                                              │
│   Problems:                                                                  │
│   • 3× development effort                                                   │
│   • 3× maintenance overhead                                                 │
│   • No knowledge sharing between tasks                                      │
│   • Inconsistent predictions across tasks                                   │
│   • Each model overfits to its specific task                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Task-Agnostic Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Task-Agnostic Unified Approach                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SINGLE UNIVERSAL MODEL:                                                    │
│   ───────────────────────                                                   │
│                                                                              │
│                      ┌─────────────────────────┐                            │
│                      │   Universal Encoder     │                            │
│                      │   (Shared Backbone)     │                            │
│                      ├─────────────────────────┤                            │
│                      │ • Market representation │                            │
│                      │ • Cross-task features   │                            │
│                      │ • General patterns      │                            │
│                      └───────────┬─────────────┘                            │
│                                  │                                           │
│               ┌──────────────────┼──────────────────┐                       │
│               │                  │                  │                       │
│               ▼                  ▼                  ▼                       │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│   │ Direction Head  │ │ Volatility Head │ │  Regime Head    │              │
│   │   (tiny MLP)    │ │   (tiny MLP)    │ │   (tiny MLP)    │              │
│   └─────────────────┘ └─────────────────┘ └─────────────────┘              │
│                                                                              │
│   Benefits:                                                                  │
│   • Single training pipeline                                                │
│   • Shared knowledge across tasks                                           │
│   • Consistent market understanding                                         │
│   • Better generalization                                                   │
│   • Reduced overfitting                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits for Trading

| Aspect | Task-Specific | Task-Agnostic |
|--------|--------------|---------------|
| Development effort | N × models | 1 unified model |
| Training data | Separate per task | Shared across tasks |
| Knowledge transfer | None | Automatic |
| Consistency | Independent predictions | Coherent predictions |
| New task adaptation | Full retraining | Task head only |
| Maintenance | High | Low |
| Generalization | Task-specific | Universal |

## Theoretical Foundation

### Universal Representation Learning

The core principle of task-agnostic learning is to discover representations that capture the fundamental structure of market data, independent of any specific downstream task.

**Definition**: A task-agnostic representation $\phi: X \to Z$ maps raw market data $X$ to a latent space $Z$ such that:

$$\forall \text{task } T_i: \exists \text{ simple predictor } h_i: Z \to Y_i$$

where $h_i$ can solve task $T_i$ using only the shared representation $Z$.

### Mathematical Framework

**Multi-Task Objective**:

$$\mathcal{L}_{total} = \sum_{i=1}^{N} \lambda_i \mathcal{L}_i(\phi, h_i)$$

where:
- $\phi$ is the shared encoder (task-agnostic)
- $h_i$ is the task-specific head for task $i$
- $\lambda_i$ is the task weight
- $\mathcal{L}_i$ is the loss for task $i$

**Gradient Balancing**:

To prevent any single task from dominating training:

$$\lambda_i = \frac{\overline{L} / L_i}{\sum_j \overline{L} / L_j}$$

where $\overline{L}$ is the average loss across all tasks.

### Information-Theoretic View

A good task-agnostic representation maximizes mutual information with all tasks:

$$\max_\phi \sum_{i=1}^{N} I(Z; Y_i)$$

subject to:

$$I(Z; T) = 0$$

where $T$ is the task identity. This ensures the representation doesn't encode task-specific biases.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                Information Flow in Task-Agnostic Learning                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Market Data X                                                         │
│         │                                                                   │
│         │ Contains: price, volume, orderbook, news, etc.                   │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              Task-Agnostic Encoder φ                                 │  │
│   │  ─────────────────────────────────────────────────────────────────  │  │
│   │                                                                      │  │
│   │  Goal: Extract information relevant to ALL trading tasks            │  │
│   │                                                                      │  │
│   │  Keeps:                              Discards:                       │  │
│   │  • Trend structure                   • Task-specific biases          │  │
│   │  • Volatility patterns               • Noise irrelevant to trading  │  │
│   │  • Market microstructure             • Spurious correlations         │  │
│   │  • Cross-asset relationships         • Overfitting artifacts         │  │
│   │  • Regime characteristics                                            │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│   Universal Representation Z                                                │
│         │                                                                   │
│         ├───── High I(Z; Price Direction)                                  │
│         ├───── High I(Z; Volatility)                                       │
│         ├───── High I(Z; Regime)                                           │
│         └───── Zero I(Z; Task Identity)                                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Task-Agnostic Architectures

### Architecture 1: Shared Encoder with Task Heads

The simplest and most common approach:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Shared Encoder Architecture                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Market Features (OHLCV + Indicators + Orderbook)                  │
│   ────────────────────────────────────────────────────                     │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Shared Encoder (Frozen after pretraining)        │  │
│   │  ─────────────────────────────────────────────────────────────────  │  │
│   │                                                                      │  │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │  │
│   │   │ 1D Conv     │──▶│ 1D Conv     │──▶│ 1D Conv     │              │  │
│   │   │ (local)     │   │ (mid-range) │   │ (long-range)│              │  │
│   │   └─────────────┘   └─────────────┘   └─────────────┘              │  │
│   │         │                 │                 │                       │  │
│   │         └────────────────┬┴─────────────────┘                       │  │
│   │                          ▼                                          │  │
│   │              ┌─────────────────────────┐                            │  │
│   │              │  Multi-Head Attention   │                            │  │
│   │              │  (temporal relationships)│                            │  │
│   │              └─────────────────────────┘                            │  │
│   │                          │                                          │  │
│   │                          ▼                                          │  │
│   │              ┌─────────────────────────┐                            │  │
│   │              │  Global Average Pool    │                            │  │
│   │              │  → 256-dim embedding    │                            │  │
│   │              └─────────────────────────┘                            │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│                  [256-dim Universal Representation]                         │
│                           │                                                 │
│         ┌─────────────────┼─────────────────┬─────────────────┐            │
│         │                 │                 │                 │            │
│         ▼                 ▼                 ▼                 ▼            │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐       │
│   │ Direction │    │ Volatility│    │  Regime   │    │  Return   │       │
│   │   Head    │    │   Head    │    │   Head    │    │   Head    │       │
│   ├───────────┤    ├───────────┤    ├───────────┤    ├───────────┤       │
│   │ 256→64→3  │    │ 256→64→1  │    │ 256→64→5  │    │ 256→64→1  │       │
│   │ softmax   │    │ ReLU      │    │ softmax   │    │ linear    │       │
│   └───────────┘    └───────────┘    └───────────┘    └───────────┘       │
│         │                 │                 │                 │            │
│         ▼                 ▼                 ▼                 ▼            │
│    [Up/Down/      [Predicted      [Bull/Bear/    [Expected              │
│     Sideways]      Volatility]     Sideways/      Return]               │
│                                    HighVol/                              │
│                                    LowVol]                               │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Architecture 2: Transformer-Based Universal Encoder

Using attention mechanisms for richer representations:

```rust
/// Transformer-based task-agnostic encoder configuration
#[derive(Debug, Clone)]
pub struct TransformerEncoderConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Model dimension (d_model)
    pub model_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Feedforward dimension
    pub ff_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Output embedding dimension
    pub embedding_dim: usize,
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            model_dim: 256,
            num_heads: 8,
            num_layers: 6,
            ff_dim: 1024,
            max_seq_len: 256,
            dropout: 0.1,
            embedding_dim: 256,
        }
    }
}

/// Task-agnostic transformer encoder
pub struct TransformerEncoder {
    config: TransformerEncoderConfig,
    // Input projection
    input_projection: Linear,
    // Positional encoding
    positional_encoding: PositionalEncoding,
    // Transformer layers
    layers: Vec<TransformerLayer>,
    // Output projection
    output_projection: Linear,
    // Layer normalization
    layer_norm: LayerNorm,
}

impl TransformerEncoder {
    /// Forward pass producing task-agnostic embeddings
    pub fn forward(&self, x: &Array3<f32>, mask: Option<&Array2<bool>>) -> Array2<f32> {
        // x shape: (batch, seq_len, input_dim)

        // 1. Project input to model dimension
        let projected = self.input_projection.forward(x);

        // 2. Add positional encoding
        let encoded = self.positional_encoding.forward(&projected);

        // 3. Pass through transformer layers
        let mut hidden = encoded;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask);
        }

        // 4. Apply layer normalization
        let normalized = self.layer_norm.forward(&hidden);

        // 5. Global pooling (mean over sequence)
        let pooled = normalized.mean_axis(Axis(1)).unwrap();

        // 6. Project to embedding dimension
        self.output_projection.forward(&pooled)
    }
}
```

### Architecture 3: Mixture of Experts (MoE)

Dynamic routing for different market conditions:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Mixture of Experts Architecture                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Market Features                                                    │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Routing Network                                   │  │
│   │  ─────────────────────────────────────                              │  │
│   │  Analyzes input and determines expert weights                       │  │
│   │                                                                      │  │
│   │  Output: [w₁=0.1, w₂=0.6, w₃=0.2, w₄=0.1]                          │  │
│   │          (weights sum to 1)                                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         │ Sparse routing (top-k experts)                                   │
│         │                                                                   │
│   ┌─────┴─────┬─────────────┬─────────────┬─────────────┐                 │
│   │           │             │             │             │                 │
│   ▼           ▼             ▼             ▼             ▼                 │
│ ┌─────┐   ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐             │
│ │ E₁  │   │ E₂  │       │ E₃  │       │ E₄  │       │ E₅  │             │
│ │Trend│   │Vol  │       │Mean │       │Tail │       │Event│             │
│ │Expert│  │Expert│      │Revert│      │Risk │       │Expert│            │
│ └──┬──┘   └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘             │
│    │         │             │             │             │                 │
│    │ ×w₁     │ ×w₂         │ ×w₃         │ ×w₄         │ ×w₅            │
│    │         │             │             │             │                 │
│    └─────────┴──────┬──────┴─────────────┴─────────────┘                 │
│                     │                                                     │
│                     ▼                                                     │
│            Weighted Sum of Expert Outputs                                 │
│                     │                                                     │
│                     ▼                                                     │
│          [Universal Representation Z]                                     │
│                     │                                                     │
│          ┌─────────┴─────────┐                                           │
│          ▼                   ▼                                           │
│     Task Heads          Expert Usage                                     │
│                         Statistics                                        │
│                         (for analysis)                                    │
│                                                                             │
│   Expert Specializations:                                                  │
│   • E₁ (Trend): Strong directional moves                                  │
│   • E₂ (Volatility): High uncertainty periods                            │
│   • E₃ (Mean Revert): Consolidation patterns                             │
│   • E₄ (Tail Risk): Extreme events                                       │
│   • E₅ (Event): News/earnings driven moves                               │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Universal Feature Learning

### Self-Supervised Pre-training

Train the encoder without labels using self-supervised objectives:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Self-Supervised Pre-training Tasks                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. MASKED AUTOENCODING (MAE for Time Series)                             │
│   ─────────────────────────────────────────────                            │
│                                                                             │
│   Original:  [p₁] [p₂] [p₃] [p₄] [p₅] [p₆] [p₇] [p₈]                     │
│   Masked:    [p₁] [##] [p₃] [##] [##] [p₆] [p₇] [##]  (30% masked)       │
│   Task:      Reconstruct masked values                                     │
│                                                                             │
│   This teaches the model to understand temporal patterns!                  │
│                                                                             │
│   2. CONTRASTIVE LEARNING                                                   │
│   ─────────────────────────                                                │
│                                                                             │
│   Create pairs:                                                            │
│   • Positive: Same asset, close time windows (similar patterns)           │
│   • Negative: Different assets or distant time windows                    │
│                                                                             │
│   ┌─────────┐   similar    ┌─────────┐                                    │
│   │ BTC t=1 │ ◀──────────▶ │ BTC t=2 │  Pull together                    │
│   └─────────┘              └─────────┘                                    │
│        │                                                                   │
│        │ different                                                         │
│        │                                                                   │
│   ┌─────────┐                                                             │
│   │ ETH t=1 │  Push apart                                                 │
│   └─────────┘                                                             │
│                                                                             │
│   3. TEMPORAL ORDER PREDICTION                                              │
│   ────────────────────────────                                             │
│                                                                             │
│   Given: [Window A] [Window B]                                             │
│   Task:  Did A come before B or after B?                                   │
│                                                                             │
│   This teaches temporal causality!                                         │
│                                                                             │
│   4. FUTURE PREDICTION (Next-Step)                                         │
│   ─────────────────────────────────                                        │
│                                                                             │
│   Given: [p₁] [p₂] [p₃] [p₄] [p₅]                                         │
│   Task:  Predict some statistics of [p₆]                                   │
│          (not exact value, but distribution properties)                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Feature Hierarchy

The encoder learns features at multiple levels of abstraction:

```rust
/// Feature hierarchy extracted by task-agnostic encoder
#[derive(Debug, Clone)]
pub struct FeatureHierarchy {
    /// Low-level features (local patterns)
    pub local_features: LocalFeatures,
    /// Mid-level features (structural patterns)
    pub structural_features: StructuralFeatures,
    /// High-level features (semantic patterns)
    pub semantic_features: SemanticFeatures,
}

/// Low-level features from early layers
#[derive(Debug, Clone)]
pub struct LocalFeatures {
    /// Price momentum at various scales
    pub momentum: Vec<f32>,
    /// Local volatility estimates
    pub local_volatility: Vec<f32>,
    /// Volume patterns
    pub volume_patterns: Vec<f32>,
    /// Bid-ask spread dynamics
    pub spread_dynamics: Vec<f32>,
}

/// Mid-level structural features
#[derive(Debug, Clone)]
pub struct StructuralFeatures {
    /// Trend strength and direction
    pub trend_structure: Vec<f32>,
    /// Support/resistance levels (learned)
    pub price_levels: Vec<f32>,
    /// Cycle detection
    pub cyclical_patterns: Vec<f32>,
    /// Correlation structure with other assets
    pub correlation_patterns: Vec<f32>,
}

/// High-level semantic features
#[derive(Debug, Clone)]
pub struct SemanticFeatures {
    /// Market regime encoding
    pub regime_embedding: Vec<f32>,
    /// Risk state encoding
    pub risk_embedding: Vec<f32>,
    /// Sentiment proxy
    pub sentiment_proxy: Vec<f32>,
    /// Anomaly score
    pub anomaly_score: f32,
}
```

## Multi-Task Training Strategy

### Gradient Harmonization

Prevent tasks from conflicting during training:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Task Gradient Harmonization                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Problem: Task gradients can conflict!                                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Example conflicting gradients:                                      │  │
│   │                                                                      │  │
│   │  Direction task gradient: ▲ (increase this weight)                  │  │
│   │  Volatility task gradient: ▼ (decrease this weight)                 │  │
│   │                                                                      │  │
│   │  Naive averaging: ▲ + ▼ = → (weights barely move!)                  │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Solution: Gradient Harmonization Methods                                 │
│                                                                             │
│   Method 1: PCGrad (Projecting Conflicting Gradients)                     │
│   ────────────────────────────────────────────────                        │
│   If g₁ · g₂ < 0 (conflicting):                                           │
│       g₁' = g₁ - (g₁·g₂/|g₂|²) × g₂                                      │
│   Project away the conflicting component                                   │
│                                                                             │
│   Method 2: GradNorm (Gradient Normalization)                             │
│   ──────────────────────────────────────────                              │
│   Dynamically adjust task weights to balance gradient magnitudes          │
│                                                                             │
│   Method 3: Uncertainty Weighting                                          │
│   ──────────────────────────────                                           │
│   w_i = 1/(2σ_i²) where σ_i is task-specific uncertainty                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Training Dynamics with Harmonization:                               │  │
│   │                                                                      │  │
│   │  Epoch 1-10:   All tasks improve steadily                           │  │
│   │  Epoch 10-50:  Direction task plateaus                               │  │
│   │                → Increase direction weight automatically            │  │
│   │  Epoch 50-100: All tasks converge together                          │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Task Curriculum

Train tasks in a structured order:

```rust
/// Task curriculum for training
#[derive(Debug, Clone)]
pub struct TaskCurriculum {
    /// Training phases
    pub phases: Vec<TrainingPhase>,
    /// Current phase index
    pub current_phase: usize,
    /// Metrics for phase transition
    pub transition_metrics: PhaseTransitionMetrics,
}

#[derive(Debug, Clone)]
pub struct TrainingPhase {
    /// Phase name
    pub name: String,
    /// Tasks active in this phase
    pub active_tasks: Vec<TaskType>,
    /// Task weights for this phase
    pub task_weights: HashMap<TaskType, f32>,
    /// Minimum epochs in this phase
    pub min_epochs: usize,
    /// Transition condition
    pub transition_condition: TransitionCondition,
}

/// Trading tasks supported by the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Predict price direction (up/down/sideways)
    DirectionPrediction,
    /// Forecast future volatility
    VolatilityForecast,
    /// Detect market regime
    RegimeDetection,
    /// Predict expected return
    ReturnPrediction,
    /// Detect anomalies
    AnomalyDetection,
    /// Predict optimal trade timing
    TradeTiming,
}

impl TaskCurriculum {
    /// Create standard trading curriculum
    pub fn standard() -> Self {
        Self {
            phases: vec![
                // Phase 1: Self-supervised pre-training
                TrainingPhase {
                    name: "Self-Supervised".to_string(),
                    active_tasks: vec![],  // No task heads, just encoder
                    task_weights: HashMap::new(),
                    min_epochs: 50,
                    transition_condition: TransitionCondition::ReconstructionLoss(0.01),
                },
                // Phase 2: Easy tasks (direction, regime)
                TrainingPhase {
                    name: "Easy Tasks".to_string(),
                    active_tasks: vec![
                        TaskType::DirectionPrediction,
                        TaskType::RegimeDetection,
                    ],
                    task_weights: [
                        (TaskType::DirectionPrediction, 0.5),
                        (TaskType::RegimeDetection, 0.5),
                    ].into_iter().collect(),
                    min_epochs: 30,
                    transition_condition: TransitionCondition::Accuracy(0.65),
                },
                // Phase 3: Add regression tasks
                TrainingPhase {
                    name: "Full Multi-Task".to_string(),
                    active_tasks: vec![
                        TaskType::DirectionPrediction,
                        TaskType::RegimeDetection,
                        TaskType::VolatilityForecast,
                        TaskType::ReturnPrediction,
                    ],
                    task_weights: [
                        (TaskType::DirectionPrediction, 0.25),
                        (TaskType::RegimeDetection, 0.25),
                        (TaskType::VolatilityForecast, 0.25),
                        (TaskType::ReturnPrediction, 0.25),
                    ].into_iter().collect(),
                    min_epochs: 50,
                    transition_condition: TransitionCondition::Convergence,
                },
            ],
            current_phase: 0,
            transition_metrics: PhaseTransitionMetrics::default(),
        }
    }
}
```

## Application to Trading

### Task-Agnostic Trading System

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Task-Agnostic Trading System                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   REAL-TIME DATA INGESTION                                                 │
│   ────────────────────────                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Bybit WebSocket → Feature Pipeline → Feature Buffer                │  │
│   │  • OHLCV (1m, 5m, 15m, 1h, 4h)                                       │  │
│   │  • Orderbook (L2 top 20 levels)                                     │  │
│   │  • Trades (tick-by-tick)                                            │  │
│   │  • Funding rates                                                     │  │
│   │  • Open interest                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   TASK-AGNOSTIC ENCODER (Pre-trained, Frozen)                             │
│   ─────────────────────────────────────────────                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Market Features → Universal Embedding (256-dim)                    │  │
│   │  Latency: < 5ms per inference                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   MULTI-TASK INFERENCE (Parallel Task Heads)                              │
│   ──────────────────────────────────────────                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│   │  │  Direction  │  │  Volatility │  │   Regime    │  │  Return   │  │  │
│   │  │   68% Up    │  │   2.3% exp  │  │  Bull (73%) │  │  +0.5%    │  │  │
│   │  │   22% Down  │  │             │  │             │  │  expected │  │  │
│   │  │   10% Side  │  │             │  │             │  │           │  │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   DECISION FUSION (Combine All Task Outputs)                              │
│   ──────────────────────────────────────────                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Consistency Check:                                                  │  │
│   │  • Direction=Up + Regime=Bull + Return=Positive → CONSISTENT ✓      │  │
│   │  • Direction=Up + Regime=Bear → CONFLICT! → Reduce confidence       │  │
│   │                                                                      │  │
│   │  Risk Assessment:                                                    │  │
│   │  • Volatility forecast determines position sizing                   │  │
│   │  • Regime determines stop-loss/take-profit levels                   │  │
│   │                                                                      │  │
│   │  Final Signal:                                                       │  │
│   │  • Action: LONG                                                     │  │
│   │  • Confidence: 72% (consistent predictions)                         │  │
│   │  • Size: 1.5% of capital (moderate volatility)                      │  │
│   │  • Stop: -1.5% (bull regime, tight stop)                            │  │
│   │  • Target: +2.0% (expected return + buffer)                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   ORDER EXECUTION                                                          │
│   ───────────────                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Bybit API → Market/Limit Order → Position Monitoring               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Decision Fusion Logic

```rust
/// Decision fusion from multiple task outputs
#[derive(Debug, Clone)]
pub struct TaskOutputs {
    /// Direction prediction
    pub direction: DirectionPrediction,
    /// Volatility forecast
    pub volatility: VolatilityForecast,
    /// Regime detection
    pub regime: RegimeDetection,
    /// Return prediction
    pub expected_return: ReturnPrediction,
}

#[derive(Debug, Clone)]
pub struct DirectionPrediction {
    pub up_prob: f32,
    pub down_prob: f32,
    pub sideways_prob: f32,
}

#[derive(Debug, Clone)]
pub struct VolatilityForecast {
    pub expected_volatility: f32,
    pub volatility_percentile: f32,  // Relative to historical
}

#[derive(Debug, Clone)]
pub struct RegimeDetection {
    pub regime: MarketRegime,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    LowVolatility,
}

#[derive(Debug, Clone)]
pub struct ReturnPrediction {
    pub expected_return: f32,
    pub return_std: f32,
}

/// Fuse multiple task outputs into trading decision
pub struct DecisionFusion {
    /// Consistency weight
    consistency_weight: f32,
    /// Minimum confidence threshold
    min_confidence: f32,
}

impl DecisionFusion {
    pub fn fuse(&self, outputs: &TaskOutputs) -> TradingDecision {
        // 1. Determine base direction
        let (direction, dir_confidence) = self.get_direction(&outputs.direction);

        // 2. Check consistency with regime
        let regime_agreement = self.check_regime_consistency(
            direction,
            outputs.regime.regime,
        );

        // 3. Check consistency with expected return
        let return_agreement = self.check_return_consistency(
            direction,
            outputs.expected_return.expected_return,
        );

        // 4. Calculate overall confidence
        let consistency_score = (regime_agreement + return_agreement) / 2.0;
        let final_confidence = dir_confidence *
            (1.0 - self.consistency_weight + self.consistency_weight * consistency_score);

        // 5. Determine position size based on volatility
        let position_size = self.calculate_position_size(
            final_confidence,
            outputs.volatility.expected_volatility,
            outputs.volatility.volatility_percentile,
        );

        // 6. Set stop-loss and take-profit based on regime and volatility
        let (stop_loss, take_profit) = self.calculate_exits(
            outputs.regime.regime,
            outputs.volatility.expected_volatility,
            outputs.expected_return.expected_return,
        );

        TradingDecision {
            action: if final_confidence >= self.min_confidence {
                match direction {
                    TradeDirection::Long => TradeAction::Buy,
                    TradeDirection::Short => TradeAction::Sell,
                    TradeDirection::Neutral => TradeAction::Hold,
                }
            } else {
                TradeAction::Hold
            },
            confidence: final_confidence,
            position_size,
            stop_loss_pct: stop_loss,
            take_profit_pct: take_profit,
            reasoning: self.generate_reasoning(outputs, direction, consistency_score),
        }
    }

    fn check_regime_consistency(&self, direction: TradeDirection, regime: MarketRegime) -> f32 {
        match (direction, regime) {
            // High consistency
            (TradeDirection::Long, MarketRegime::Bull) => 1.0,
            (TradeDirection::Short, MarketRegime::Bear) => 1.0,
            (TradeDirection::Neutral, MarketRegime::Sideways) => 1.0,
            // Medium consistency
            (TradeDirection::Long, MarketRegime::LowVolatility) => 0.7,
            (TradeDirection::Short, MarketRegime::HighVolatility) => 0.6,
            // Low consistency (conflicting signals)
            (TradeDirection::Long, MarketRegime::Bear) => 0.2,
            (TradeDirection::Short, MarketRegime::Bull) => 0.2,
            // Neutral
            _ => 0.5,
        }
    }

    fn calculate_position_size(
        &self,
        confidence: f32,
        volatility: f32,
        volatility_percentile: f32,
    ) -> f32 {
        let base_size = 0.02;  // 2% base position

        // Scale down for high volatility
        let vol_multiplier = if volatility_percentile > 0.8 {
            0.5  // High vol: half size
        } else if volatility_percentile > 0.6 {
            0.75  // Elevated vol: 3/4 size
        } else {
            1.0  // Normal vol: full size
        };

        // Scale with confidence
        let conf_multiplier = (confidence - self.min_confidence) /
            (1.0 - self.min_confidence);

        base_size * vol_multiplier * conf_multiplier
    }
}

#[derive(Debug, Clone)]
pub struct TradingDecision {
    pub action: TradeAction,
    pub confidence: f32,
    pub position_size: f32,
    pub stop_loss_pct: f32,
    pub take_profit_pct: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Copy)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
}
```

## Bybit Integration

### Data Collection for Multiple Tasks

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Bybit Data Collection Pipeline                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATA SOURCES FOR TASK-AGNOSTIC TRAINING                                  │
│   ──────────────────────────────────────────                               │
│                                                                             │
│   1. HISTORICAL DATA (Training)                                            │
│   ─────────────────────────────                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Endpoint: GET /v5/market/kline                                     │  │
│   │  Symbols: BTCUSDT, ETHUSDT, SOLUSDT, ... (top 50 by volume)        │  │
│   │  Intervals: 1m, 5m, 15m, 1h, 4h, 1d                                 │  │
│   │  History: 2 years                                                    │  │
│   │                                                                      │  │
│   │  Label Generation for Tasks:                                        │  │
│   │  ┌─────────────────────────────────────────────────────────────┐    │  │
│   │  │ Direction: Based on forward return sign                     │    │  │
│   │  │ Volatility: Realized volatility over next N periods         │    │  │
│   │  │ Regime: HMM clustering on return/volatility                 │    │  │
│   │  │ Return: Actual forward return (continuous)                  │    │  │
│   │  └─────────────────────────────────────────────────────────────┘    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. REAL-TIME DATA (Inference)                                            │
│   ─────────────────────────────                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  WebSocket: wss://stream.bybit.com/v5/public/linear                 │  │
│   │                                                                      │  │
│   │  Subscriptions:                                                      │  │
│   │  • kline.1.BTCUSDT (1-minute candles)                               │  │
│   │  • orderbook.50.BTCUSDT (L2 orderbook)                              │  │
│   │  • publicTrade.BTCUSDT (tick trades)                                │  │
│   │  • tickers.BTCUSDT (market stats)                                   │  │
│   │                                                                      │  │
│   │  Additional REST calls:                                              │  │
│   │  • GET /v5/market/funding/history (funding rates)                   │  │
│   │  • GET /v5/market/open-interest (open interest)                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. FEATURE ENGINEERING                                                    │
│   ──────────────────────                                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Raw Data → Feature Pipeline → Normalized Features                  │  │
│   │                                                                      │  │
│   │  Feature Groups:                                                     │  │
│   │  • Price features (returns, log-returns, ranges)                    │  │
│   │  • Volume features (volume ratios, VWAP deviation)                  │  │
│   │  • Volatility features (realized vol, ATR, Parkinson)               │  │
│   │  • Momentum features (ROC, RSI, MACD)                               │  │
│   │  • Microstructure (spread, depth imbalance, trade flow)            │  │
│   │  • Crypto-specific (funding, OI change, liquidations)              │  │
│   │                                                                      │  │
│   │  Total features: ~64 per timepoint                                  │  │
│   │  Sequence length: 96 timepoints (96 hours at 1h resolution)        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Bybit Client Implementation

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

/// Bybit client for task-agnostic trading
pub struct BybitClient {
    http_client: Client,
    api_key: Option<String>,
    api_secret: Option<String>,
    base_url: String,
    ws_url: String,
}

impl BybitClient {
    pub fn new(api_key: Option<String>, api_secret: Option<String>) -> Self {
        Self {
            http_client: Client::new(),
            api_key,
            api_secret,
            base_url: "https://api.bybit.com".to_string(),
            ws_url: "wss://stream.bybit.com/v5/public/linear".to_string(),
        }
    }

    /// Fetch historical klines for multiple symbols
    pub async fn fetch_training_data(
        &self,
        symbols: &[&str],
        interval: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<HashMap<String, Vec<Kline>>, BybitError> {
        let mut all_data = HashMap::new();

        for symbol in symbols {
            let klines = self.get_klines(symbol, interval, start_time, end_time).await?;
            all_data.insert(symbol.to_string(), klines);
        }

        Ok(all_data)
    }

    /// Get klines for a single symbol
    async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut current_start = start_time;

        while current_start < end_time {
            let url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&limit=1000",
                self.base_url, symbol, interval, current_start
            );

            let response: KlineResponse = self.http_client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.result.list.is_empty() {
                break;
            }

            let klines: Vec<Kline> = response.result.list
                .into_iter()
                .map(Kline::from)
                .collect();

            current_start = klines.last().map(|k| k.timestamp + 1).unwrap_or(end_time);
            all_klines.extend(klines);

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(all_klines)
    }

    /// Start real-time data streaming
    pub async fn start_streaming(
        &self,
        symbols: &[&str],
        tx: mpsc::Sender<MarketUpdate>,
    ) -> Result<(), BybitError> {
        let (ws_stream, _) = connect_async(&self.ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe to channels
        let topics: Vec<String> = symbols.iter()
            .flat_map(|s| vec![
                format!("kline.1.{}", s),
                format!("orderbook.50.{}", s),
                format!("publicTrade.{}", s),
            ])
            .collect();

        let subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": topics
        });

        write.send(Message::Text(subscribe_msg.to_string())).await?;

        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(update) = serde_json::from_str::<MarketUpdate>(&text) {
                        tx.send(update).await.ok();
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Err(e) => {
                    eprintln!("WebSocket error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

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

#[derive(Debug, Clone, Deserialize)]
pub struct MarketUpdate {
    pub topic: String,
    pub data: serde_json::Value,
    pub ts: u64,
}
```

## Implementation

### Module Structure

```
87_task_agnostic_trading/
├── Cargo.toml
├── README.md                     # Main documentation (English)
├── README.ru.md                  # Russian translation
├── readme.simple.md              # Simplified explanation
├── readme.simple.ru.md           # Simplified Russian
├── src/
│   ├── lib.rs                    # Library root
│   ├── config.rs                 # Configuration types
│   ├── encoder/
│   │   ├── mod.rs               # Encoder module
│   │   ├── transformer.rs       # Transformer encoder
│   │   ├── cnn.rs               # CNN encoder
│   │   └── moe.rs               # Mixture of Experts
│   ├── tasks/
│   │   ├── mod.rs               # Task module
│   │   ├── direction.rs         # Direction prediction
│   │   ├── volatility.rs        # Volatility forecast
│   │   ├── regime.rs            # Regime detection
│   │   └── returns.rs           # Return prediction
│   ├── training/
│   │   ├── mod.rs               # Training module
│   │   ├── multitask.rs         # Multi-task training
│   │   ├── curriculum.rs        # Task curriculum
│   │   └── pretraining.rs       # Self-supervised pretraining
│   ├── fusion/
│   │   ├── mod.rs               # Fusion module
│   │   └── decision.rs          # Decision fusion
│   ├── data/
│   │   ├── mod.rs               # Data module
│   │   ├── bybit.rs             # Bybit API client
│   │   ├── features.rs          # Feature engineering
│   │   └── dataset.rs           # Dataset handling
│   └── trading/
│       ├── mod.rs               # Trading module
│       ├── strategy.rs          # Trading strategy
│       └── risk.rs              # Risk management
├── examples/
│   ├── train_encoder.rs         # Train task-agnostic encoder
│   ├── multi_task_inference.rs  # Multi-task inference example
│   ├── bybit_trading.rs         # Live trading with Bybit
│   └── backtest.rs              # Backtesting example
├── python/
│   ├── task_agnostic_model.py   # PyTorch implementation
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── requirements.txt         # Python dependencies
└── tests/
    ├── integration.rs           # Integration tests
    └── unit_tests.rs            # Unit tests
```

### Core Implementation

```rust
//! Task-Agnostic Trading Core Module

use ndarray::{Array1, Array2, Array3, Axis};
use std::collections::HashMap;

/// Task-agnostic model configuration
#[derive(Debug, Clone)]
pub struct TaskAgnosticConfig {
    /// Encoder configuration
    pub encoder_config: EncoderConfig,
    /// Active tasks
    pub tasks: Vec<TaskConfig>,
    /// Multi-task training configuration
    pub training_config: MultiTaskTrainingConfig,
    /// Decision fusion configuration
    pub fusion_config: FusionConfig,
}

impl Default for TaskAgnosticConfig {
    fn default() -> Self {
        Self {
            encoder_config: EncoderConfig::default(),
            tasks: vec![
                TaskConfig::direction_prediction(),
                TaskConfig::volatility_forecast(),
                TaskConfig::regime_detection(),
                TaskConfig::return_prediction(),
            ],
            training_config: MultiTaskTrainingConfig::default(),
            fusion_config: FusionConfig::default(),
        }
    }
}

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Sequence length
    pub seq_length: usize,
    /// Encoder type
    pub encoder_type: EncoderType,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            seq_length: 96,
            encoder_type: EncoderType::Transformer {
                num_layers: 6,
                num_heads: 8,
                ff_dim: 256,
            },
            hidden_dim: 256,
            embedding_dim: 256,
            dropout: 0.1,
        }
    }
}

/// Encoder architecture types
#[derive(Debug, Clone)]
pub enum EncoderType {
    /// Transformer-based encoder
    Transformer {
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
    },
    /// CNN-based encoder
    CNN {
        num_layers: usize,
        kernel_sizes: Vec<usize>,
    },
    /// Mixture of Experts
    MoE {
        num_experts: usize,
        top_k: usize,
        expert_dim: usize,
    },
}

/// Task configuration
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Task type
    pub task_type: TaskType,
    /// Task head hidden dimension
    pub head_hidden_dim: usize,
    /// Task weight in training
    pub weight: f32,
    /// Loss function type
    pub loss_type: LossType,
}

impl TaskConfig {
    pub fn direction_prediction() -> Self {
        Self {
            task_type: TaskType::DirectionPrediction,
            head_hidden_dim: 64,
            weight: 1.0,
            loss_type: LossType::CrossEntropy,
        }
    }

    pub fn volatility_forecast() -> Self {
        Self {
            task_type: TaskType::VolatilityForecast,
            head_hidden_dim: 64,
            weight: 1.0,
            loss_type: LossType::MSE,
        }
    }

    pub fn regime_detection() -> Self {
        Self {
            task_type: TaskType::RegimeDetection,
            head_hidden_dim: 64,
            weight: 1.0,
            loss_type: LossType::CrossEntropy,
        }
    }

    pub fn return_prediction() -> Self {
        Self {
            task_type: TaskType::ReturnPrediction,
            head_hidden_dim: 64,
            weight: 1.0,
            loss_type: LossType::MSE,
        }
    }
}

/// Task types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    DirectionPrediction,
    VolatilityForecast,
    RegimeDetection,
    ReturnPrediction,
    AnomalyDetection,
    TradeTiming,
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossType {
    CrossEntropy,
    MSE,
    MAE,
    Huber,
}

/// Task-agnostic trading model
pub struct TaskAgnosticModel {
    /// Configuration
    config: TaskAgnosticConfig,
    /// Shared encoder
    encoder: Box<dyn Encoder>,
    /// Task heads
    task_heads: HashMap<TaskType, Box<dyn TaskHead>>,
    /// Decision fusion
    fusion: DecisionFusion,
}

/// Encoder trait for different architectures
pub trait Encoder: Send + Sync {
    /// Forward pass producing embeddings
    fn forward(&self, x: &Array3<f32>) -> Array2<f32>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;
}

/// Task head trait for different tasks
pub trait TaskHead: Send + Sync {
    /// Forward pass from embeddings to task output
    fn forward(&self, embeddings: &Array2<f32>) -> TaskOutput;

    /// Get task type
    fn task_type(&self) -> TaskType;

    /// Compute loss
    fn compute_loss(&self, output: &TaskOutput, target: &TaskTarget) -> f32;
}

/// Task output enum
#[derive(Debug, Clone)]
pub enum TaskOutput {
    /// Classification output (logits)
    Classification(Array2<f32>),
    /// Regression output (values)
    Regression(Array2<f32>),
}

/// Task target enum
#[derive(Debug, Clone)]
pub enum TaskTarget {
    /// Classification target (class indices)
    Classification(Vec<usize>),
    /// Regression target (values)
    Regression(Array2<f32>),
}

impl TaskAgnosticModel {
    /// Create a new task-agnostic model
    pub fn new(config: TaskAgnosticConfig) -> Self {
        let encoder = Self::create_encoder(&config.encoder_config);
        let task_heads = Self::create_task_heads(&config);
        let fusion = DecisionFusion::new(config.fusion_config.clone());

        Self {
            config,
            encoder,
            task_heads,
            fusion,
        }
    }

    /// Run inference on all tasks
    pub fn forward(&self, features: &Array3<f32>) -> AllTaskOutputs {
        // 1. Get universal embeddings
        let embeddings = self.encoder.forward(features);

        // 2. Run each task head
        let mut outputs = HashMap::new();
        for (task_type, head) in &self.task_heads {
            let output = head.forward(&embeddings);
            outputs.insert(*task_type, output);
        }

        AllTaskOutputs { outputs, embeddings }
    }

    /// Make trading decision from multi-task outputs
    pub fn make_decision(&self, outputs: &AllTaskOutputs) -> TradingDecision {
        // Convert task outputs to structured format
        let structured = self.structure_outputs(outputs);

        // Fuse decisions
        self.fusion.fuse(&structured)
    }

    fn create_encoder(config: &EncoderConfig) -> Box<dyn Encoder> {
        match &config.encoder_type {
            EncoderType::Transformer { num_layers, num_heads, ff_dim } => {
                Box::new(TransformerEncoder::new(
                    config.input_dim,
                    config.seq_length,
                    config.hidden_dim,
                    config.embedding_dim,
                    *num_layers,
                    *num_heads,
                    *ff_dim,
                    config.dropout,
                ))
            }
            EncoderType::CNN { num_layers, kernel_sizes } => {
                Box::new(CNNEncoder::new(
                    config.input_dim,
                    config.seq_length,
                    config.hidden_dim,
                    config.embedding_dim,
                    *num_layers,
                    kernel_sizes.clone(),
                    config.dropout,
                ))
            }
            EncoderType::MoE { num_experts, top_k, expert_dim } => {
                Box::new(MoEEncoder::new(
                    config.input_dim,
                    config.seq_length,
                    config.hidden_dim,
                    config.embedding_dim,
                    *num_experts,
                    *top_k,
                    *expert_dim,
                    config.dropout,
                ))
            }
        }
    }

    fn create_task_heads(config: &TaskAgnosticConfig) -> HashMap<TaskType, Box<dyn TaskHead>> {
        let mut heads = HashMap::new();

        for task_config in &config.tasks {
            let head: Box<dyn TaskHead> = match task_config.task_type {
                TaskType::DirectionPrediction => {
                    Box::new(DirectionHead::new(
                        config.encoder_config.embedding_dim,
                        task_config.head_hidden_dim,
                    ))
                }
                TaskType::VolatilityForecast => {
                    Box::new(VolatilityHead::new(
                        config.encoder_config.embedding_dim,
                        task_config.head_hidden_dim,
                    ))
                }
                TaskType::RegimeDetection => {
                    Box::new(RegimeHead::new(
                        config.encoder_config.embedding_dim,
                        task_config.head_hidden_dim,
                    ))
                }
                TaskType::ReturnPrediction => {
                    Box::new(ReturnHead::new(
                        config.encoder_config.embedding_dim,
                        task_config.head_hidden_dim,
                    ))
                }
                _ => continue,
            };

            heads.insert(task_config.task_type, head);
        }

        heads
    }

    fn structure_outputs(&self, outputs: &AllTaskOutputs) -> TaskOutputs {
        // Extract and structure individual task outputs
        // (Implementation details)
        TaskOutputs {
            direction: self.extract_direction(&outputs.outputs[&TaskType::DirectionPrediction]),
            volatility: self.extract_volatility(&outputs.outputs[&TaskType::VolatilityForecast]),
            regime: self.extract_regime(&outputs.outputs[&TaskType::RegimeDetection]),
            expected_return: self.extract_return(&outputs.outputs[&TaskType::ReturnPrediction]),
        }
    }
}

/// All task outputs from forward pass
#[derive(Debug, Clone)]
pub struct AllTaskOutputs {
    /// Individual task outputs
    pub outputs: HashMap<TaskType, TaskOutput>,
    /// Universal embeddings (for analysis)
    pub embeddings: Array2<f32>,
}
```

## Risk Management

### Multi-Task Risk Assessment

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Task Risk Assessment                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. TASK AGREEMENT SCORING                                                │
│   ────────────────────────                                                 │
│                                                                             │
│   When tasks agree → Higher confidence, larger positions                   │
│   When tasks conflict → Lower confidence, smaller positions or no trade   │
│                                                                             │
│   Agreement Matrix:                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │              │ Direction │ Volatility │ Regime │ Return │ Score   │   │
│   │──────────────┼───────────┼────────────┼────────┼────────┼─────────│   │
│   │ Scenario 1   │   UP      │   LOW      │  BULL  │  +0.5% │  0.95  │   │
│   │ (All agree)  │           │            │        │        │ TRADE  │   │
│   │──────────────┼───────────┼────────────┼────────┼────────┼─────────│   │
│   │ Scenario 2   │   UP      │   HIGH     │  BULL  │  +0.3% │  0.70  │   │
│   │ (Vol warns)  │           │            │        │        │ CAUTION│   │
│   │──────────────┼───────────┼────────────┼────────┼────────┼─────────│   │
│   │ Scenario 3   │   UP      │   HIGH     │  BEAR  │  -0.2% │  0.30  │   │
│   │ (Conflict)   │           │            │        │        │ NO TRADE│  │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   2. DYNAMIC POSITION SIZING                                               │
│   ──────────────────────────                                               │
│                                                                             │
│   Position Size = Base × Confidence × (1 / Volatility_Rank)                │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Confidence   │  Vol Rank │  Multiplier │  Example (2% base)      │   │
│   │───────────────┼───────────┼─────────────┼─────────────────────────│   │
│   │  0.90+        │  Low      │  1.5x       │  3.0% position          │   │
│   │  0.90+        │  High     │  0.75x      │  1.5% position          │   │
│   │  0.70-0.90    │  Low      │  1.0x       │  2.0% position          │   │
│   │  0.70-0.90    │  High     │  0.5x       │  1.0% position          │   │
│   │  0.50-0.70    │  Low      │  0.5x       │  1.0% position          │   │
│   │  0.50-0.70    │  High     │  0.25x      │  0.5% position          │   │
│   │  <0.50        │  Any      │  0x         │  No trade               │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   3. REGIME-AWARE STOP-LOSS                                                │
│   ─────────────────────────                                                │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Regime         │  Stop-Loss │  Take-Profit │  Rationale          │   │
│   │─────────────────┼────────────┼──────────────┼─────────────────────│   │
│   │  Bull           │  -1.5%     │  +3.0%       │  Ride the trend     │   │
│   │  Bear           │  -1.5%     │  +3.0%       │  Catch the move     │   │
│   │  Sideways       │  -1.0%     │  +1.5%       │  Tight range play   │   │
│   │  High Volatility│  -2.5%     │  +4.0%       │  Wide stops         │   │
│   │  Low Volatility │  -1.0%     │  +1.5%       │  Tight control      │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   4. PORTFOLIO-LEVEL CONSTRAINTS                                           │
│   ──────────────────────────────                                           │
│                                                                             │
│   • Maximum total exposure: 10% of capital                                 │
│   • Maximum correlation: Don't trade highly correlated assets together    │
│   • Daily loss limit: -3% → stop trading for the day                      │
│   • Weekly drawdown: -5% → reduce position sizes by 50%                   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Risk Parameters

```rust
/// Risk management parameters for task-agnostic trading
#[derive(Debug, Clone)]
pub struct RiskParameters {
    /// Base position size (fraction of capital)
    pub base_position_size: f32,
    /// Maximum position size multiplier
    pub max_position_multiplier: f32,
    /// Minimum confidence to trade
    pub min_confidence: f32,
    /// Minimum task agreement score
    pub min_agreement_score: f32,
    /// Base stop-loss percentage
    pub base_stop_loss: f32,
    /// Base take-profit percentage
    pub base_take_profit: f32,
    /// Maximum total exposure
    pub max_total_exposure: f32,
    /// Daily loss limit
    pub daily_loss_limit: f32,
    /// Weekly drawdown limit
    pub weekly_drawdown_limit: f32,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            base_position_size: 0.02,        // 2% per trade
            max_position_multiplier: 1.5,     // Up to 3% max
            min_confidence: 0.50,
            min_agreement_score: 0.60,
            base_stop_loss: 0.015,           // 1.5% stop
            base_take_profit: 0.025,         // 2.5% take profit
            max_total_exposure: 0.10,        // 10% total
            daily_loss_limit: 0.03,          // 3% daily limit
            weekly_drawdown_limit: 0.05,     // 5% weekly limit
        }
    }
}

impl RiskParameters {
    /// Calculate position size based on multi-task outputs
    pub fn calculate_position_size(
        &self,
        confidence: f32,
        agreement_score: f32,
        volatility_percentile: f32,
        current_exposure: f32,
    ) -> f32 {
        // Check minimum thresholds
        if confidence < self.min_confidence || agreement_score < self.min_agreement_score {
            return 0.0;
        }

        // Check exposure limit
        let remaining_capacity = self.max_total_exposure - current_exposure;
        if remaining_capacity <= 0.0 {
            return 0.0;
        }

        // Base calculation
        let confidence_factor = (confidence - self.min_confidence) /
            (1.0 - self.min_confidence);
        let agreement_factor = (agreement_score - self.min_agreement_score) /
            (1.0 - self.min_agreement_score);

        // Volatility adjustment (inverse)
        let vol_factor = if volatility_percentile > 0.8 {
            0.5
        } else if volatility_percentile > 0.6 {
            0.75
        } else {
            1.0
        };

        let position_size = self.base_position_size
            * confidence_factor
            * agreement_factor
            * vol_factor
            * self.max_position_multiplier;

        // Cap at remaining capacity
        position_size.min(remaining_capacity)
    }

    /// Get regime-adjusted stop-loss
    pub fn get_stop_loss(&self, regime: MarketRegime, volatility_percentile: f32) -> f32 {
        let base = match regime {
            MarketRegime::Bull | MarketRegime::Bear => self.base_stop_loss,
            MarketRegime::Sideways | MarketRegime::LowVolatility => self.base_stop_loss * 0.67,
            MarketRegime::HighVolatility => self.base_stop_loss * 1.67,
        };

        // Widen for high volatility
        if volatility_percentile > 0.8 {
            base * 1.5
        } else {
            base
        }
    }

    /// Get regime-adjusted take-profit
    pub fn get_take_profit(&self, regime: MarketRegime, volatility_percentile: f32) -> f32 {
        let base = match regime {
            MarketRegime::Bull | MarketRegime::Bear => self.base_take_profit * 1.2,
            MarketRegime::Sideways | MarketRegime::LowVolatility => self.base_take_profit * 0.6,
            MarketRegime::HighVolatility => self.base_take_profit * 1.6,
        };

        // Widen for high volatility
        if volatility_percentile > 0.8 {
            base * 1.3
        } else {
            base
        }
    }
}
```

## Performance Metrics

### Model Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Multi-Task Accuracy** | Average accuracy across all classification tasks | > 65% |
| **Direction Accuracy** | Price direction prediction accuracy | > 60% |
| **Regime Accuracy** | Market regime detection accuracy | > 70% |
| **Volatility RMSE** | Root mean squared error for volatility forecast | < 0.02 |
| **Return Correlation** | Correlation between predicted and actual returns | > 0.3 |
| **Task Transfer** | Performance when adding new task | > 90% of from-scratch |
| **Consistency Score** | How often task predictions agree | > 70% |

### Trading Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted returns | > 2.0 |
| **Sortino Ratio** | Downside risk-adjusted returns | > 2.5 |
| **Max Drawdown** | Largest peak-to-trough decline | < 15% |
| **Win Rate** | Percentage of profitable trades | > 55% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Calmar Ratio** | Annual return / Max drawdown | > 1.0 |

### Latency Budget

```
┌─────────────────────────────────────────────────┐
│              Latency Requirements               │
├─────────────────────────────────────────────────┤
│ Feature Engineering:       < 10ms               │
│ Encoder Forward Pass:      < 15ms               │
│ All Task Heads:            < 5ms                │
│ Decision Fusion:           < 2ms                │
├─────────────────────────────────────────────────┤
│ Total Inference:           < 32ms               │
├─────────────────────────────────────────────────┤
│ Order Placement:           < 50ms               │
│ End-to-End:                < 100ms              │
└─────────────────────────────────────────────────┘
```

## References

1. **Model-Agnostic Meta-Learning for Natural Language Understanding Tasks in Finance**
   - URL: https://arxiv.org/abs/2303.02841
   - Year: 2023

2. **Trading in Fast-Changing Markets with Meta-Reinforcement Learning**
   - MAML-based trading method for multiple tasks
   - Year: 2024

3. **Multi-Task Learning in Deep Neural Networks: A Survey**
   - Comprehensive overview of multi-task architectures
   - URL: https://arxiv.org/abs/2009.09796

4. **Universal Language Model Fine-tuning for Text Classification**
   - Transfer learning techniques applicable to time series
   - URL: https://arxiv.org/abs/1801.06146

5. **Gradient Surgery for Multi-Task Learning**
   - PCGrad method for handling gradient conflicts
   - URL: https://arxiv.org/abs/2001.06782

6. **Adaptive Event-Driven Labeling with Meta-Learning for Financial Time Series**
   - Multi-scale temporal analysis with MAML
   - Year: 2025

7. **FinRL: Deep Reinforcement Learning for Quantitative Finance**
   - Comprehensive framework for financial RL
   - URL: https://github.com/AI4Finance-Foundation/FinRL

---

## Next Steps

- [Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Russian translation
- [Run Examples](examples/) - Working Rust code
- [Python Implementation](python/) - PyTorch reference implementation

---

*Chapter 87 of Machine Learning for Trading*
