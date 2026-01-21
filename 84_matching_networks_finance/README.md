# Chapter 84: Matching Networks for Finance

## Overview

Matching Networks are a meta-learning approach for one-shot and few-shot learning that uses an attention-based classifier over learned embeddings. Unlike Prototypical Networks that compute class centroids, Matching Networks use a soft nearest-neighbor approach with attention weights to classify new examples. This makes them particularly powerful for financial applications where market patterns may not form simple clusters.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture Components](#architecture-components)
4. [Full Context Embeddings (FCE)](#full-context-embeddings-fce)
5. [Application to Financial Markets](#application-to-financial-markets)
6. [Few-Shot Market Pattern Recognition](#few-shot-market-pattern-recognition)
7. [Implementation Strategy](#implementation-strategy)
8. [Bybit Integration](#bybit-integration)
9. [Risk Management](#risk-management)
10. [Performance Metrics](#performance-metrics)
11. [Comparison with Prototypical Networks](#comparison-with-prototypical-networks)
12. [References](#references)

---

## Introduction

Traditional machine learning approaches for trading require large amounts of labeled data for each market pattern or regime. However, financial markets present unique challenges:

- **Pattern scarcity**: Some trading patterns (cup-and-handle, head-and-shoulders) occur rarely
- **Context sensitivity**: The same pattern may have different meanings in different market contexts
- **Rapid adaptation**: Need to recognize new patterns with few examples

### Why Matching Networks for Trading?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Attention-Based Trading Problem                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Prototypical Networks:              Matching Networks:                 │
│   ─────────────────────               ─────────────────                  │
│   Compare to class centroids          Compare to ALL support examples   │
│   Simple averaging                    Attention-weighted voting         │
│                                                                          │
│   Problem: Patterns may not           Solution: Learn similarity to     │
│   cluster nicely around a             individual examples, using        │
│   single centroid                     context-aware embeddings          │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                                                            │        │
│   │   Prototypical:     [●] ← centroid     Matching:          │        │
│   │                      ▲                  Query → [?]        │        │
│   │                    / | \                        ↓          │        │
│   │                  ○   ○   ○               [a₁○ a₂○ a₃○ ...]│        │
│   │                (support)                 attention weights │        │
│   │                                                            │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Advantages

| Aspect | Prototypical Networks | Matching Networks |
|--------|----------------------|-------------------|
| Classification method | Distance to centroids | Attention over all examples |
| Context awareness | Limited | Full via FCE |
| Embedding | Fixed | Context-dependent |
| Pattern flexibility | Assumes cluster structure | Handles complex distributions |
| Computational cost | Lower | Higher (attention computation) |
| Interpretability | High (prototype distances) | High (attention weights) |

## Theoretical Foundation

### The Matching Networks Framework

Matching Networks learn a neural network that maps a small labeled support set S and an unlabeled example x̂ to its label ŷ, without the need for fine-tuning:

$$P(ŷ | x̂, S) = \sum_{i=1}^{k} a(x̂, x_i) y_i$$

where $a(x̂, x_i)$ is an attention mechanism that computes how similar the query $x̂$ is to each support example $x_i$.

### Mathematical Formulation

**Attention Kernel**: The attention is computed as a softmax over cosine similarities:

$$a(x̂, x_i) = \frac{\exp(c(f(x̂), g(x_i)))}{\sum_{j=1}^{k} \exp(c(f(x̂), g(x_j)))}$$

where:
- $f$ is the embedding function for the query
- $g$ is the embedding function for support examples
- $c$ is the cosine similarity: $c(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$

### Key Components

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Matching Networks Architecture                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SUPPORT SET S = {(x₁,y₁), (x₂,y₂), ..., (xₖ,yₖ)}                    │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  g(xᵢ, S) - Support Embedding          │                           │
│   │  ──────────────────────────────        │                           │
│   │  Uses bidirectional LSTM to encode     │                           │
│   │  each support example with context     │                           │
│   │  from the entire support set           │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
│   QUERY x̂                                                              │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  f(x̂, S) - Query Embedding (FCE)       │                           │
│   │  ──────────────────────────────        │                           │
│   │  LSTM with attention that reads over   │                           │
│   │  the support set embeddings            │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
│   CLASSIFICATION                                                        │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  Attention-weighted voting:            │                           │
│   │  P(y|x̂,S) = Σᵢ a(x̂,xᵢ)yᵢ             │                           │
│   │                                        │                           │
│   │  where a = softmax(cosine_similarity)  │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Episodic Training

Training mirrors the test-time task structure:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Episodic Training Process                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Each episode simulates a few-shot classification task:               │
│                                                                         │
│   Step 1: Sample N classes (e.g., 5 market patterns)                  │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │ Double Top │ Double Bottom │ Breakout │ Reversal │ Trend │         │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 2: For each class, sample K support + Q query examples         │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Double Top:    [s1, s2, s3, s4, s5] | [q1, q2, q3]        │      │
│   │  Double Bottom: [s1, s2, s3, s4, s5] | [q1, q2, q3]        │      │
│   │  ...                                                        │      │
│   └─────────────────────────────────────────────────────────────┘      │
│        Support Set (5-shot)       Query Set                             │
│                                                                         │
│   Step 3: Embed support examples using g(·, S)                        │
│   Step 4: Embed query examples using f(·, S)                          │
│   Step 5: Compute attention weights and predictions                   │
│   Step 6: Compute cross-entropy loss and backpropagate                │
│                                                                         │
│   Key Insight: Train and test conditions must match!                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Architecture Components

### Embedding Functions

**Basic Embedding (without FCE)**:

When using simple embeddings without Full Context Embeddings, both f and g are the same neural network:

```python
# Simple embedding: f = g = neural_network
embedding = neural_network(x)  # Same for query and support
```

**Full Context Embeddings (FCE)**:

FCE uses context-dependent embeddings that condition on the entire support set:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Full Context Embeddings (FCE)                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SUPPORT EMBEDDING g(xᵢ, S):                                          │
│   ─────────────────────────────                                        │
│   Step 1: Compute basic embeddings g'(xⱼ) for all xⱼ ∈ S              │
│   Step 2: Run bidirectional LSTM over all g'(xⱼ)                      │
│   Step 3: g(xᵢ, S) = h→ᵢ + h←ᵢ + g'(xᵢ)  (forward + backward + skip) │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  g'(x₁) → [LSTM→] → h→₁                                 │          │
│   │  g'(x₂) → [LSTM→] → h→₂                                 │          │
│   │  g'(x₃) → [LSTM→] → h→₃                                 │          │
│   │          ← [LSTM←]                                       │          │
│   │  g(xᵢ,S) = h→ᵢ + h←ᵢ + g'(xᵢ)                          │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│   QUERY EMBEDDING f(x̂, S):                                             │
│   ─────────────────────────                                            │
│   Uses LSTM with attention over support embeddings                     │
│                                                                         │
│   for k = 1 to K processing steps:                                     │
│       h, c = LSTM(f'(x̂), [h, r], c)                                   │
│       h = h + f'(x̂)              # Skip connection                    │
│       r = attention_readout(h, g(S))  # Attend over support           │
│   f(x̂, S) = h                                                         │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  f'(x̂) → [LSTM] → h₁ → attention → r₁                  │          │
│   │          → [LSTM] → h₂ → attention → r₂                 │          │
│   │          → ...                                          │          │
│   │          → [LSTM] → hₖ = f(x̂, S)                       │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Attention Readout Mechanism

The query embedding uses attention to gather information from support examples:

```python
def attention_readout(h, g_support):
    """
    Compute attention-weighted readout from support embeddings.

    Args:
        h: Current query hidden state
        g_support: Support embeddings (n_support, embed_dim)

    Returns:
        r: Readout vector (weighted sum of support embeddings)
    """
    # Compute attention weights
    scores = h @ g_support.T  # (embed_dim,) @ (n_support, embed_dim).T
    attention = softmax(scores)  # (n_support,)

    # Compute weighted sum
    r = attention @ g_support  # (embed_dim,)

    return r
```

### Embedding Network for Trading

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Base Embedding Network for Trading                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: Market features [OHLCV, indicators, order book, ...]         │
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
│   │  Bidirectional LSTM Block                                   │      │
│   │  ─────────────────────────                                  │      │
│   │  BiLSTM(hidden=128, layers=2)                               │      │
│   │  Output: concatenate(forward, backward) = 256 dims          │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Projection Head                                             │      │
│   │  ───────────────                                            │      │
│   │  Linear(in=256, out=128) → ReLU                             │      │
│   │  Linear(in=128, out=embedding_dim)                          │      │
│   │  L2 Normalize                                                │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   Output: g'(x) - Base embedding vector (embedding_dim,)               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Full Context Embeddings (FCE)

### Why FCE Matters for Trading

Financial markets are highly contextual. The same pattern may have different meanings depending on:

- **Market regime**: Bull vs. bear market
- **Volatility environment**: High vs. low volatility
- **Other assets**: Correlated movements
- **Support set composition**: Which examples are available

FCE allows the network to:
1. Condition support embeddings on the entire support set
2. Condition query embeddings on the available support examples
3. Create context-aware representations

### Support Set Encoding with BiLSTM

```rust
/// Full Context Embedding for Support Set
pub struct SupportSetEncoder {
    /// Base embedding network
    base_embedding: BaseEmbedding,
    /// Bidirectional LSTM for context
    bi_lstm: BiLSTM,
}

impl SupportSetEncoder {
    /// Encode support set with full context
    pub fn encode(&self, support_set: &[MarketWindow]) -> Vec<Embedding> {
        // Step 1: Get base embeddings for all support examples
        let base_embeddings: Vec<Embedding> = support_set
            .iter()
            .map(|x| self.base_embedding.forward(x))
            .collect();

        // Step 2: Run bidirectional LSTM
        let forward_states = self.bi_lstm.forward(&base_embeddings);
        let backward_states = self.bi_lstm.backward(&base_embeddings);

        // Step 3: Combine with skip connection
        base_embeddings
            .iter()
            .enumerate()
            .map(|(i, base)| {
                &forward_states[i] + &backward_states[i] + base
            })
            .collect()
    }
}
```

### Query Encoding with Attention

```rust
/// Full Context Embedding for Query
pub struct QueryEncoder {
    /// Base embedding network
    base_embedding: BaseEmbedding,
    /// LSTM for iterative refinement
    lstm: LSTM,
    /// Number of processing steps
    num_steps: usize,
}

impl QueryEncoder {
    /// Encode query with attention over support set
    pub fn encode(
        &self,
        query: &MarketWindow,
        support_embeddings: &[Embedding],
    ) -> Embedding {
        // Get base query embedding
        let query_base = self.base_embedding.forward(query);

        // Initialize LSTM state
        let mut h = query_base.clone();
        let mut c = Embedding::zeros(self.lstm.hidden_dim);

        // Iterative refinement with attention
        for _ in 0..self.num_steps {
            // Concatenate h with readout for LSTM input
            let r = self.attention_readout(&h, support_embeddings);
            let input = concat(&[&query_base, &r]);

            // LSTM step
            (h, c) = self.lstm.step(&input, (&h, &c));

            // Skip connection
            h = h + &query_base;
        }

        h
    }

    /// Compute attention-weighted readout from support embeddings
    fn attention_readout(
        &self,
        h: &Embedding,
        support_embeddings: &[Embedding],
    ) -> Embedding {
        // Compute attention scores (cosine similarity)
        let scores: Vec<f32> = support_embeddings
            .iter()
            .map(|s| cosine_similarity(h, s))
            .collect();

        // Softmax
        let attention = softmax(&scores);

        // Weighted sum
        support_embeddings
            .iter()
            .zip(attention.iter())
            .fold(Embedding::zeros(h.dim()), |acc, (s, &a)| {
                acc + s * a
            })
    }
}
```

## Application to Financial Markets

### Market Pattern Classification

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Market Pattern Classes                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Class 0: TREND_CONTINUATION                                          │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Price moving in established direction                   │        │
│   │  • Pullbacks within trend structure                        │        │
│   │  • Volume confirming moves                                 │        │
│   │  • Momentum indicators aligned                             │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 1: TREND_REVERSAL                                              │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Divergence in momentum indicators                       │        │
│   │  • Volume spike on reversal candle                         │        │
│   │  • Break of key support/resistance                         │        │
│   │  • Change in market structure                              │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 2: BREAKOUT                                                    │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Price breaking consolidation range                      │        │
│   │  • Volume expansion                                        │        │
│   │  • Volatility increase                                     │        │
│   │  • Strong directional move                                 │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 3: FALSE_BREAKOUT                                              │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Initial break of key level                             │        │
│   │  • Quick reversal back into range                         │        │
│   │  • Volume doesn't confirm                                 │        │
│   │  • Trapping traders                                       │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Class 4: CONSOLIDATION                                               │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Characteristics:                                          │        │
│   │  • Range-bound price action                                │        │
│   │  • Decreasing volatility                                   │        │
│   │  • Volume contraction                                      │        │
│   │  • Coiling for next move                                   │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Trading Strategy Based on Pattern Recognition

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Pattern-Based Trading Signals                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Pattern Detection Pipeline:                                           │
│   ──────────────────────────                                           │
│   1. Maintain support set of labeled pattern examples                  │
│   2. Extract features for current market window                        │
│   3. Embed support set with FCE (g function)                          │
│   4. Embed query with attention over support (f function)             │
│   5. Compute attention weights to all support examples                │
│   6. Generate prediction via weighted voting                          │
│   7. Execute trading signal based on pattern and confidence           │
│                                                                         │
│   Pattern → Signal Mapping:                                            │
│   ┌───────────────────────┬──────────────────────────────────────┐    │
│   │ Pattern               │ Action                                │    │
│   ├───────────────────────┼──────────────────────────────────────┤    │
│   │ TREND_CONTINUATION    │ Enter/hold with trend direction     │    │
│   │ TREND_REVERSAL        │ Close positions, consider reversal  │    │
│   │ BREAKOUT              │ Enter in breakout direction         │    │
│   │ FALSE_BREAKOUT        │ Fade the move, tight stop           │    │
│   │ CONSOLIDATION         │ Wait, reduce position size          │    │
│   └───────────────────────┴──────────────────────────────────────┘    │
│                                                                         │
│   Confidence-Based Position Sizing:                                     │
│   ─────────────────────────────────                                    │
│   position_size = base_size × max_attention_weight                     │
│                                                                         │
│   Interpretability via Attention:                                      │
│   ────────────────────────────────                                     │
│   • Highest attention weights reveal which support examples           │
│     are most similar to the current market                            │
│   • Provides explainable predictions                                  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Few-Shot Market Pattern Recognition

### Episode Generation for Training

```python
def generate_matching_episode(
    dataset,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 10
):
    """
    Generate one training episode for matching network.

    Args:
        dataset: Historical market data with pattern labels
        n_way: Number of pattern classes per episode
        k_shot: Number of support examples per class
        n_query: Number of query examples per class

    Returns:
        support_set: List of (features, label) tuples
        query_set: List of (features, label) tuples
    """
    # Sample n_way classes from available pattern classes
    available_classes = dataset.get_pattern_classes()
    sampled_classes = random.sample(available_classes, n_way)

    support_set = []
    query_set = []

    for new_label, original_class in enumerate(sampled_classes):
        # Get all samples for this pattern
        class_samples = dataset.get_samples_for_pattern(original_class)

        # Sample k_shot + n_query examples
        sampled_indices = random.sample(
            range(len(class_samples)),
            k_shot + n_query
        )

        # Split into support and query
        for i, idx in enumerate(sampled_indices):
            if i < k_shot:
                support_set.append((class_samples[idx], new_label))
            else:
                query_set.append((class_samples[idx], new_label))

    return support_set, query_set
```

### Matching Network Forward Pass

```python
def matching_network_forward(
    support_set,
    query,
    base_embedding_fn,
    support_encoder,
    query_encoder,
):
    """
    Forward pass of matching network.

    Args:
        support_set: List of (features, label) tuples
        query: Query features
        base_embedding_fn: Base embedding function g'
        support_encoder: FCE support encoder
        query_encoder: FCE query encoder

    Returns:
        class_probabilities: Probability distribution over classes
        attention_weights: Attention weights to each support example
    """
    # Separate features and labels
    support_features = [s[0] for s in support_set]
    support_labels = [s[1] for s in support_set]
    n_classes = len(set(support_labels))

    # Encode support set with FCE
    support_embeddings = support_encoder.encode(support_features)

    # Encode query with attention over support
    query_embedding = query_encoder.encode(query, support_embeddings)

    # Compute attention weights (cosine similarity + softmax)
    similarities = [
        cosine_similarity(query_embedding, s_emb)
        for s_emb in support_embeddings
    ]
    attention_weights = softmax(similarities)

    # Weighted voting: P(y|x) = Σ a(x, x_i) * y_i
    class_probabilities = np.zeros(n_classes)
    for i, (weight, label) in enumerate(zip(attention_weights, support_labels)):
        class_probabilities[label] += weight

    return class_probabilities, attention_weights
```

## Implementation Strategy

### Python Implementation (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNetwork(nn.Module):
    """
    Matching Network for few-shot market pattern classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        lstm_layers: int = 1,
        fce_steps: int = 5,
        use_fce: bool = True,
    ):
        super().__init__()

        self.use_fce = use_fce
        self.fce_steps = fce_steps

        # Base embedding network
        self.base_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        if use_fce:
            # Support set encoder (bidirectional LSTM)
            self.support_lstm = nn.LSTM(
                embedding_dim,
                embedding_dim,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
            )
            self.support_proj = nn.Linear(embedding_dim * 2, embedding_dim)

            # Query encoder (LSTM with attention)
            self.query_lstm = nn.LSTMCell(
                embedding_dim * 2,  # Query embedding + readout
                embedding_dim,
            )

    def encode_support(self, support: torch.Tensor) -> torch.Tensor:
        """
        Encode support set with FCE.

        Args:
            support: (n_support, input_dim)

        Returns:
            embeddings: (n_support, embedding_dim)
        """
        # Base embeddings
        base_emb = self.base_embedding(support)  # (n_support, embedding_dim)

        if not self.use_fce:
            return F.normalize(base_emb, dim=-1)

        # Apply bidirectional LSTM
        base_emb_seq = base_emb.unsqueeze(0)  # (1, n_support, embedding_dim)
        lstm_out, _ = self.support_lstm(base_emb_seq)  # (1, n_support, 2*embedding_dim)
        lstm_out = lstm_out.squeeze(0)  # (n_support, 2*embedding_dim)

        # Project and add skip connection
        context_emb = self.support_proj(lstm_out)  # (n_support, embedding_dim)
        full_emb = context_emb + base_emb  # Skip connection

        return F.normalize(full_emb, dim=-1)

    def encode_query(
        self,
        query: torch.Tensor,
        support_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode query with attention over support.

        Args:
            query: (batch_size, input_dim)
            support_embeddings: (n_support, embedding_dim)

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        batch_size = query.shape[0]
        embed_dim = support_embeddings.shape[-1]

        # Base embedding
        query_base = self.base_embedding(query)  # (batch_size, embedding_dim)

        if not self.use_fce:
            return F.normalize(query_base, dim=-1)

        # Initialize LSTM state
        h = query_base
        c = torch.zeros_like(h)

        # Iterative refinement with attention
        for _ in range(self.fce_steps):
            # Attention readout
            r = self._attention_readout(h, support_embeddings)  # (batch_size, embedding_dim)

            # LSTM input: concatenate query base and readout
            lstm_input = torch.cat([query_base, r], dim=-1)  # (batch_size, 2*embedding_dim)

            # LSTM step
            h, c = self.query_lstm(lstm_input, (h, c))

            # Skip connection
            h = h + query_base

        return F.normalize(h, dim=-1)

    def _attention_readout(
        self,
        query: torch.Tensor,
        support_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted readout from support.

        Args:
            query: (batch_size, embedding_dim)
            support_embeddings: (n_support, embedding_dim)

        Returns:
            readout: (batch_size, embedding_dim)
        """
        # Cosine similarity
        query_norm = F.normalize(query, dim=-1)
        support_norm = F.normalize(support_embeddings, dim=-1)

        # (batch_size, n_support)
        similarities = torch.mm(query_norm, support_norm.t())
        attention = F.softmax(similarities, dim=-1)

        # Weighted sum
        readout = torch.mm(attention, support_embeddings)  # (batch_size, embedding_dim)

        return readout

    def forward(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            support: (n_support, input_dim)
            support_labels: (n_support,)
            query: (n_query, input_dim)

        Returns:
            class_probs: (n_query, n_classes)
            attention: (n_query, n_support)
        """
        # Encode support and query
        support_emb = self.encode_support(support)
        query_emb = self.encode_query(query, support_emb)

        # Compute attention weights
        similarities = torch.mm(query_emb, support_emb.t())  # (n_query, n_support)
        attention = F.softmax(similarities, dim=-1)

        # Aggregate by class
        n_classes = support_labels.max().item() + 1
        class_probs = torch.zeros(query.shape[0], n_classes, device=query.device)

        for i in range(support.shape[0]):
            class_probs[:, support_labels[i]] += attention[:, i]

        return class_probs, attention
```

### Rust Implementation

See the `src/` directory for the complete Rust implementation with:
- `network/` - Matching network architecture
- `data/` - Bybit and stock data integration
- `strategy/` - Trading strategy implementation
- `training/` - Episodic training utilities

## Bybit Integration

### Real-Time Pattern Detection

```rust
use crate::data::bybit::{BybitClient, KlineData};
use crate::network::MatchingNetwork;
use crate::strategy::PatternStrategy;

/// Real-time pattern detector using Matching Networks
pub struct BybitPatternDetector {
    client: BybitClient,
    network: MatchingNetwork,
    support_set: SupportSet,
    strategy: PatternStrategy,
}

impl BybitPatternDetector {
    /// Detect pattern in current market
    pub async fn detect_pattern(&self, symbol: &str) -> Result<PatternDetection> {
        // Fetch recent klines
        let klines = self.client
            .get_klines(symbol, "15m", 100)
            .await?;

        // Extract features
        let features = self.extract_features(&klines);

        // Classify using matching network
        let (class_probs, attention) = self.network.forward(
            &self.support_set,
            &features,
        );

        // Get prediction and confidence
        let predicted_class = class_probs.argmax();
        let confidence = class_probs[predicted_class];

        // Find most similar support examples
        let similar_examples = self.get_top_attention(attention, 3);

        Ok(PatternDetection {
            pattern: predicted_class.into(),
            confidence,
            similar_examples,
            trading_signal: self.strategy.generate_signal(predicted_class, confidence),
        })
    }
}
```

### Feature Extraction

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Feature Extraction for Matching                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Price Features:                                                       │
│   • Returns (multiple timeframes)                                       │
│   • Price position relative to moving averages                         │
│   • Higher highs / lower lows pattern                                  │
│   • Support/resistance proximity                                        │
│                                                                         │
│   Volume Features:                                                      │
│   • Volume relative to average                                          │
│   • Volume profile (distribution across price levels)                  │
│   • Buy/Sell volume ratio (from Bybit trades)                         │
│                                                                         │
│   Volatility Features:                                                  │
│   • ATR (Average True Range)                                           │
│   • Bollinger Band width                                               │
│   • Volatility regime indicator                                        │
│                                                                         │
│   Momentum Features:                                                    │
│   • RSI and RSI divergence                                             │
│   • MACD and histogram                                                 │
│   • Stochastic oscillator                                              │
│                                                                         │
│   Crypto-Specific Features:                                            │
│   • Funding rate                                                        │
│   • Open interest change                                               │
│   • Long/Short ratio                                                   │
│   • Liquidation levels                                                 │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Risk Management

### Position Sizing Based on Confidence

```rust
impl PatternStrategy {
    /// Calculate position size based on pattern confidence
    pub fn calculate_position_size(
        &self,
        confidence: f32,
        attention_entropy: f32,
        account_balance: f32,
    ) -> f32 {
        // Base position as percentage of account
        let base_position = account_balance * self.base_risk_pct;

        // Confidence scaling (higher confidence = larger position)
        let confidence_factor = if confidence > 0.8 {
            1.0
        } else if confidence > 0.6 {
            0.7
        } else if confidence > 0.4 {
            0.4
        } else {
            0.0  // Don't trade if confidence is too low
        };

        // Attention entropy scaling (lower entropy = more decisive)
        // High entropy means attention is spread across many examples (uncertain)
        let entropy_factor = (1.0 - attention_entropy / self.max_entropy).max(0.5);

        base_position * confidence_factor * entropy_factor
    }
}
```

### Stop Loss and Take Profit

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Risk Management Rules by Pattern                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TREND_CONTINUATION:                                                   │
│   • Stop: Below recent swing low (long) / above swing high (short)    │
│   • Take Profit: Trail with ATR-based stop                            │
│   • Risk/Reward: 1:2 minimum                                          │
│                                                                         │
│   TREND_REVERSAL:                                                       │
│   • Stop: Beyond the reversal level                                    │
│   • Take Profit: Previous support/resistance                          │
│   • Risk/Reward: 1:3 minimum (compensate for lower win rate)          │
│                                                                         │
│   BREAKOUT:                                                             │
│   • Stop: Below/above the breakout level                               │
│   • Take Profit: Measured move equal to range                         │
│   • Risk/Reward: 1:2 minimum                                          │
│                                                                         │
│   FALSE_BREAKOUT:                                                       │
│   • Stop: Beyond the false breakout high/low                          │
│   • Take Profit: Opposite side of range                               │
│   • Risk/Reward: 1:2 minimum                                          │
│                                                                         │
│   CONSOLIDATION:                                                        │
│   • Action: Reduce position size or stay flat                         │
│   • Wait for breakout signal                                          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Performance Metrics

### Classification Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct predictions | > 60% |
| F1-Score | Harmonic mean of precision/recall | > 0.55 |
| Top-2 Accuracy | Correct class in top 2 predictions | > 80% |
| Confidence Calibration | Predicted confidence matches accuracy | ECE < 0.1 |

### Trading Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return | > 1.5 |
| Sortino Ratio | Downside risk-adjusted return | > 2.0 |
| Maximum Drawdown | Largest peak-to-trough decline | < 15% |
| Win Rate | Percentage of winning trades | > 50% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |

### Few-Shot Specific Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| 5-shot Accuracy | Accuracy with 5 examples per class | Standard metric |
| 1-shot Accuracy | Accuracy with 1 example per class | Harder metric |
| Attention Entropy | How spread out attention is | $-\sum a_i \log a_i$ |
| Prototype Consistency | Stability across episodes | Variance of predictions |

## Comparison with Prototypical Networks

### Key Differences

```
┌────────────────────────────────────────────────────────────────────────┐
│           Matching Networks vs Prototypical Networks                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Classification Mechanism:                                             │
│   ─────────────────────────                                            │
│   Prototypical: Compare to class centroid (prototype)                  │
│   Matching:     Attention-weighted voting over ALL examples            │
│                                                                         │
│   Embedding:                                                            │
│   ──────────                                                           │
│   Prototypical: Fixed embedding f(x)                                   │
│   Matching:     Context-dependent via FCE: f(x, S), g(x, S)           │
│                                                                         │
│   Computation:                                                          │
│   ───────────                                                          │
│   Prototypical: O(n_classes) distances                                 │
│   Matching:     O(n_support) attention weights                         │
│                                                                         │
│   When to Use Each:                                                     │
│   ─────────────────                                                    │
│   Prototypical: Patterns form tight clusters                           │
│   Matching:     Patterns are more spread out / context-dependent       │
│                                                                         │
│   Interpretability:                                                     │
│   ────────────────                                                     │
│   Prototypical: "This looks like the typical X pattern"                │
│   Matching:     "This looks like examples 3, 7, 12 (attention)"       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### When to Choose Matching Networks

1. **Complex pattern distributions**: When patterns don't cluster nicely
2. **Context matters**: When the same features mean different things in different contexts
3. **Interpretability via examples**: When you want to know which specific examples are similar
4. **Small support sets**: Works well even with very few examples per class

## References

1. **Matching Networks for One Shot Learning**
   - Authors: Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., Wierstra, D.
   - URL: https://arxiv.org/abs/1606.04080
   - Year: 2016
   - Conference: NeurIPS 2016

2. **Prototypical Networks for Few-shot Learning**
   - Authors: Snell, J., Swersky, K., Zemel, R.
   - URL: https://arxiv.org/abs/1703.05175
   - Year: 2017

3. **Meta-Learning for Semi-Supervised Few-Shot Classification**
   - Authors: Ren, M., et al.
   - URL: https://arxiv.org/abs/1803.00676
   - Year: 2018

4. **Few-Shot Learning: A Survey**
   - Authors: Wang, Y., et al.
   - URL: https://arxiv.org/abs/1904.05046
   - Year: 2019

---

## Quick Start

### Python

```python
from matching_network import MatchingNetwork, MarketPatternClassifier

# Create the network
network = MatchingNetwork(
    input_dim=15,
    hidden_dim=64,
    embedding_dim=64,
    use_fce=True,
    fce_steps=5,
)

# Create classifier
classifier = MarketPatternClassifier(network)

# Fit with support set
classifier.fit(support_features, support_labels)

# Classify new patterns
predictions, attention = classifier.predict(query_features)

# Interpret predictions
for i, (pred, att) in enumerate(zip(predictions, attention)):
    print(f"Query {i}: Pattern {pred}")
    print(f"  Most similar support examples: {att.argsort()[-3:][::-1]}")
```

### Rust

```rust
use matching_networks_finance::{
    MatchingNetwork, Config, BybitClient, PatternStrategy
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize network
    let config = Config::default();
    let network = MatchingNetwork::new(config)?;

    // Load support set
    let support_set = load_pattern_examples("data/support_set.json")?;

    // Connect to Bybit
    let client = BybitClient::new()?;

    // Real-time pattern detection
    let detection = network.detect_pattern(
        &support_set,
        &client.get_klines("BTCUSDT", "15m", 100).await?,
    )?;

    println!("Pattern: {:?}", detection.pattern);
    println!("Confidence: {:.2}", detection.confidence);
    println!("Trading Signal: {:?}", detection.trading_signal);

    Ok(())
}
```

---

## File Structure

```
84_matching_networks_finance/
├── README.md                    # This file (English)
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simple explanation (English)
├── readme.simple.ru.md          # Simple explanation (Russian)
├── README.specify.md            # Technical specification
├── Cargo.toml                   # Rust dependencies
├── python/
│   ├── matching_network.py      # Core implementation
│   ├── data_loader.py           # Data loading utilities
│   ├── trainer.py               # Training utilities
│   └── backtest.py              # Backtesting framework
├── src/
│   ├── lib.rs                   # Rust library root
│   ├── network/
│   │   ├── mod.rs
│   │   ├── embedding.rs         # Embedding networks
│   │   ├── attention.rs         # Attention mechanisms
│   │   └── fce.rs               # Full Context Embeddings
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs             # Bybit API integration
│   │   ├── stock.rs             # Stock data (yfinance)
│   │   └── features.rs          # Feature extraction
│   ├── strategy/
│   │   ├── mod.rs
│   │   └── pattern_strategy.rs  # Trading strategy
│   ├── training/
│   │   ├── mod.rs
│   │   └── episodic.rs          # Episodic training
│   └── utils/
│       ├── mod.rs
│       └── metrics.rs           # Performance metrics
├── examples/
│   ├── basic_matching.rs        # Basic usage example
│   ├── pattern_detection.rs     # Pattern detection example
│   └── backtest.rs              # Backtesting example
└── tests/
    └── integration_tests.rs     # Integration tests
```
