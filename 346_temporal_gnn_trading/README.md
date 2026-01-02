# Chapter 346: Temporal Graph Neural Networks for Trading

## Overview

Temporal Graph Neural Networks (Temporal GNNs / TGN) represent the cutting-edge of deep learning on dynamic graphs, specifically designed to learn from continuous-time event sequences. Unlike static GNNs that treat graphs as fixed structures, Temporal GNNs maintain memory of past interactions and encode temporal patterns to make predictions about future events in financial markets.

This chapter explores the implementation of Temporal Graph Networks for cryptocurrency trading on Bybit exchange, focusing on capturing the evolving relationships between assets over time.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [TGN Architecture](#tgn-architecture)
4. [Memory Module](#memory-module)
5. [Message Function](#message-function)
6. [Time Encoding](#time-encoding)
7. [Training Approach](#training-approach)
8. [Application to Cryptocurrency Trading](#application-to-cryptocurrency-trading)
9. [Implementation Strategy](#implementation-strategy)
10. [Risk Management](#risk-management)
11. [Performance Metrics](#performance-metrics)
12. [References](#references)

---

## Introduction

Financial markets are fundamentally temporal systems where events occur in continuous time. Traditional approaches discretize time into fixed intervals (1-minute, 1-hour candles), losing valuable information about the exact timing and sequence of events. Temporal Graph Neural Networks address this limitation by:

- **Learning from event streams**: Process trades, order book updates, and price changes as they occur
- **Maintaining temporal memory**: Remember past interactions to inform current predictions
- **Encoding time dynamics**: Capture complex temporal patterns like periodicity and decay
- **Handling irregular sampling**: Work with events that occur at arbitrary times

### Why Temporal GNNs for Trading?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Continuous-Time Financial Events                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Time ──────────────────────────────────────────────────────────────→    │
│                                                                          │
│  BTC:  ●──────●───●────────────●─●──●───────────●────────────●───→      │
│           trade  trade         trades            trade         trade     │
│                                                                          │
│  ETH:  ───●────────●──●────────────────●──●──●───────●────────────→      │
│          trade     trades               trades        trade              │
│                                                                          │
│  SOL:  ────────●──────────●────●─────────────●───────────●──●──→        │
│                 trade       trades            trade         trades       │
│                                                                          │
│  TGN: Learns from the TIMING and SEQUENCE of all these events!          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Traditional vs Temporal Approach

| Aspect | Traditional GNN | Temporal GNN |
|--------|----------------|--------------|
| Time handling | Discrete snapshots | Continuous events |
| Memory | Stateless per snapshot | Persistent memory |
| Event sequence | Lost between snapshots | Preserved |
| Irregularity | Requires resampling | Native support |
| Latency | Batch-oriented | Event-driven |

## Theoretical Foundation

### Continuous-Time Dynamic Graphs

A continuous-time dynamic graph (CTDG) is defined as a sequence of timestamped events:

$$\mathcal{G} = \{(u_1, v_1, t_1, e_1), (u_2, v_2, t_2, e_2), ..., (u_n, v_n, t_n, e_n)\}$$

Where:
- $(u_i, v_i)$ are the source and destination nodes
- $t_i$ is the timestamp (continuous)
- $e_i$ is the edge feature (trade size, price, etc.)

### The TGN Framework

The core insight of TGN is to maintain a **memory state** for each node that gets updated with each interaction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TGN Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Event: (u, v, t, e)                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────┐                                             │
│  │  Raw Messages  │ ← Compute messages from event               │
│  └───────┬────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌────────────────┐                                             │
│  │   Aggregate    │ ← Combine messages for each node            │
│  └───────┬────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌────────────────┐                                             │
│  │ Update Memory  │ ← s(t) = GRU(s(t⁻), aggregated_msg)        │
│  └───────┬────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌────────────────┐                                             │
│  │   Embedding    │ ← z(t) = f(s(t), temporal_neighbors)        │
│  └───────┬────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌────────────────┐                                             │
│  │  Prediction    │ ← Link/node/graph prediction task           │
│  └────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

For each node $i$, TGN maintains:

1. **Memory state** $s_i(t)$: Compressed representation of past interactions
2. **Temporal embedding** $z_i(t)$: Current representation incorporating neighbors

The memory update follows:

$$s_i(t) = \text{mem}(s_i(t^-), m_i(t))$$

Where $m_i(t)$ is the aggregated message and $\text{mem}$ is a learnable function (typically GRU or LSTM).

## TGN Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TGN Architecture                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │   Memory    │   │   Message   │   │    Time     │               │
│  │   Module    │   │   Function  │   │   Encoder   │               │
│  │             │   │             │   │             │               │
│  │ Per-node    │   │ Creates msg │   │ Encodes Δt  │               │
│  │ state s(t)  │   │ from events │   │ features    │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         │                 │                 │                        │
│         └────────────┬────┴────────────────┘                        │
│                      │                                               │
│                      ▼                                               │
│              ┌───────────────┐                                      │
│              │   Embedding   │                                      │
│              │    Module     │                                      │
│              │               │                                      │
│              │ Temporal GNN  │                                      │
│              │ aggregation   │                                      │
│              └───────┬───────┘                                      │
│                      │                                               │
│                      ▼                                               │
│              ┌───────────────┐                                      │
│              │  Prediction   │                                      │
│              │    Heads      │                                      │
│              │               │                                      │
│              │ • Link pred   │                                      │
│              │ • Node class  │                                      │
│              │ • Price pred  │                                      │
│              └───────────────┘                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Node Types in Financial Graphs

```
Node Categories:
├── Assets
│   ├── BTCUSDT (Bitcoin perpetual)
│   ├── ETHUSDT (Ethereum perpetual)
│   └── ... (other trading pairs)
│
├── Market Events
│   ├── Large trades (whale activity)
│   ├── Liquidations
│   └── Funding rate changes
│
├── Order Book States
│   ├── Bid depth levels
│   └── Ask depth levels
│
└── External Signals
    ├── Sentiment indicators
    └── On-chain metrics
```

### Edge Types

```
Edge Categories:
├── Price Correlation
│   └── Weight: Rolling correlation coefficient
│
├── Order Flow
│   └── Weight: Trade volume between assets
│
├── Liquidity Links
│   └── Weight: Shared market makers / depth
│
├── Causal Relations
│   └── Weight: Granger causality strength
│
└── Temporal Proximity
    └── Weight: 1 / (1 + Δt)
```

## Memory Module

The memory module is the heart of TGN, enabling the network to maintain state across events.

### Memory State Structure

For each node $i$, the memory contains:

```
Memory State s_i(t):
├── Raw Memory Vector: [d_memory dimensions]
│   └── Learned representation of interaction history
│
├── Last Update Time: t_last
│   └── When this node was last involved in an event
│
├── Interaction Count: n_interactions
│   └── Number of events involving this node
│
└── Decay Factor: λ_i
    └── How quickly old information fades
```

### Memory Update Mechanisms

#### 1. GRU-based Update (Default)

$$s_i(t) = \text{GRU}(s_i(t^-), m_i(t))$$

Where:
- $z = \sigma(W_z [s_i(t^-), m_i(t)])$ (update gate)
- $r = \sigma(W_r [s_i(t^-), m_i(t)])$ (reset gate)
- $\tilde{s} = \tanh(W [r \odot s_i(t^-), m_i(t)])$ (candidate)
- $s_i(t) = (1-z) \odot s_i(t^-) + z \odot \tilde{s}$ (new state)

#### 2. Attention-based Update

$$s_i(t) = s_i(t^-) + \text{Attn}(s_i(t^-), [m_1, ..., m_k])$$

#### 3. Decay-based Update

$$s_i(t) = \exp(-\lambda \cdot \Delta t) \cdot s_i(t^-) + (1 - \exp(-\lambda \cdot \Delta t)) \cdot m_i(t)$$

### Memory Staleness

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Staleness Problem                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Time ──────────────────────────────────────────────────────→    │
│                                                                  │
│  Node A:  ●───────────────────────────────────────●──────→      │
│           ↑                                       ↑              │
│        Active                                  Active            │
│        Fresh memory                            Stale memory!     │
│                                                                  │
│  Solution: Apply time-decay to stale memories before use         │
│                                                                  │
│  s_effective(t) = exp(-λ · (t - t_last)) · s(t_last)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Message Function

The message function computes information to be passed between nodes during an event.

### Message Types

#### 1. Source Message (from initiator)

$$m_{src}(t) = \text{MLP}([s_u(t^-) \| s_v(t^-) \| e(t) \| \phi(t)])$$

#### 2. Destination Message (to receiver)

$$m_{dst}(t) = \text{MLP}([s_v(t^-) \| s_u(t^-) \| e(t) \| \phi(t)])$$

Where:
- $s_u, s_v$ are memory states of source and destination
- $e(t)$ is the edge feature (trade info)
- $\phi(t)$ is the time encoding

### Message Aggregation

When multiple events involve the same node in a batch:

```
Aggregation Options:
├── Most Recent: Use only the latest message
├── Mean: Average all messages
├── Sum: Sum all messages
├── Attention: Weighted combination
└── LSTM: Sequential processing by time
```

### Financial Message Features

```
Edge Features for Crypto Trading:
├── Trade Information
│   ├── price: Execution price
│   ├── size: Trade volume
│   ├── side: Buy/Sell
│   └── value: USD value
│
├── Order Book Context
│   ├── bid_depth_ratio: Bid vs Ask volume
│   ├── spread: Current spread
│   └── impact: Price impact of trade
│
├── Market Context
│   ├── volatility: Recent volatility
│   ├── trend: Recent price trend
│   └── volume_percentile: Volume vs average
│
└── Temporal Context
    ├── time_since_last: Δt since last trade
    ├── trade_rate: Recent trade frequency
    └── time_of_day: Cyclical encoding
```

## Time Encoding

Time encoding is critical for capturing temporal patterns in financial data.

### Functional Time Encoding

Using learnable periodic functions:

$$\phi(t) = \sqrt{\frac{1}{d}}[\cos(\omega_1 t + \phi_1), \sin(\omega_1 t + \phi_1), ..., \cos(\omega_d t + \phi_d), \sin(\omega_d t + \phi_d)]$$

### Time2Vec Encoding

A learnable approach that captures both periodic and non-periodic patterns:

$$\text{Time2Vec}(t)[i] = \begin{cases} \omega_i t + \phi_i & \text{if } i = 0 \\ \sin(\omega_i t + \phi_i) & \text{if } 1 \leq i \leq k \end{cases}$$

### Multi-Scale Time Encoding

```
┌─────────────────────────────────────────────────────────────────┐
│                 Multi-Scale Time Encoding                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Scale 1: Milliseconds (Market Microstructure)                  │
│  └── Captures: Order arrival, HFT patterns                      │
│                                                                  │
│  Scale 2: Seconds (Trade Flow)                                  │
│  └── Captures: Trade clustering, momentum                       │
│                                                                  │
│  Scale 3: Minutes (Short-term Trends)                           │
│  └── Captures: Intraday patterns, news reaction                 │
│                                                                  │
│  Scale 4: Hours (Session Patterns)                              │
│  └── Captures: Market open/close, funding times                 │
│                                                                  │
│  Scale 5: Days (Macro Trends)                                   │
│  └── Captures: Weekly cycles, weekend effects                   │
│                                                                  │
│  Combined: φ(t) = Concat([φ_1(t), φ_2(t), ..., φ_5(t)])        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Relative vs Absolute Time

For financial markets, we typically encode:

1. **Relative time (Δt)**: Time since last interaction (most important)
2. **Absolute time features**: Hour of day, day of week (for periodicity)
3. **Event-based time**: Time since last funding, since last large trade

## Training Approach

### Self-Supervised Learning

TGN uses temporal link prediction as a self-supervised objective:

$$\mathcal{L} = -\sum_{(u,v,t) \in \mathcal{E}^+} \log p(u, v, t) - \sum_{(u,v',t) \in \mathcal{E}^-} \log(1 - p(u, v', t))$$

### Training Strategies

#### 1. Batch Training with Temporal Ordering

```
Training Batch Processing:
1. Sort events by timestamp
2. Split into temporal batches
3. For each batch:
   a. Compute messages for all events
   b. Aggregate messages per node
   c. Update memories
   d. Compute embeddings
   e. Calculate loss
   f. Backpropagate (not through memory updates)
```

#### 2. Memory Module Training

The memory is trained end-to-end but updated without gradient:

```python
# Pseudo-code for memory training
for batch in temporal_batches:
    # Forward pass
    embeddings = model.compute_embeddings(batch, memory)
    loss = compute_loss(embeddings, labels)

    # Backward pass (gradients only through embedding, not memory update)
    loss.backward()
    optimizer.step()

    # Update memory (no gradient)
    with torch.no_grad():
        memory = model.update_memory(batch, memory)
```

### Negative Sampling Strategies

```
Negative Sampling for Link Prediction:
├── Random: Random node pairs (baseline)
├── Historical: Past interactions (harder)
├── Temporal: Events from different times
└── Inductive: Nodes not in training
```

## Application to Cryptocurrency Trading

### Bybit Market Graph Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Bybit Temporal Trading Graph                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │                     Asset Layer                              │      │
│    │                                                              │      │
│    │  BTCUSDT ←─────────────→ ETHUSDT ←──────────→ SOLUSDT       │      │
│    │     ↑                       ↑                    ↑           │      │
│    │     │ funding               │ correlation        │ volume    │      │
│    │     ↓                       ↓                    ↓           │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │                     Event Layer                              │      │
│    │                                                              │      │
│    │  Trade ──→ Asset   (Creates temporal edge)                  │      │
│    │  Liquidation ──→ Asset (Impacts all correlated)             │      │
│    │  FundingRate ──→ Asset (8-hour cycle)                       │      │
│    │                                                              │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                                                                          │
│    Events flow continuously, updating node memories and edges            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Real-Time Event Processing

```
┌────────────────────────────────────────────────────────────────────┐
│                 Bybit WebSocket Event Stream                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WebSocket Feed:                                                    │
│  ├── Trade stream     → Create trade events                       │
│  ├── Order book       → Update depth features                     │
│  ├── Ticker           → Update price features                     │
│  └── Liquidation      → Create liquidation events                 │
│                                                                     │
│  TGN Processing:                                                    │
│  ├── Buffer events in time window (e.g., 100ms)                   │
│  ├── Sort by timestamp                                             │
│  ├── Batch process through TGN                                    │
│  ├── Update node memories                                          │
│  ├── Generate embeddings                                           │
│  └── Produce trading signals                                       │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering

| Feature Category | Features | Update Trigger |
|-----------------|----------|----------------|
| Trade Features | price, size, side, impact | Each trade |
| Order Book | spread, imbalance, depth | Order book update |
| OHLCV | open, high, low, close, volume | Candlestick close |
| Funding | rate, countdown, predicted | Every 8 hours |
| Open Interest | OI, OI change, long/short | Position change |
| Temporal | time_since_trade, trade_rate | Each event |

### Signal Generation Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                    TGN Trading Signal Pipeline                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Event Ingestion                                                     │
│     └── Receive trade/orderbook/liquidation events from Bybit          │
│                                                                         │
│  2. Message Computation                                                 │
│     └── For each event, compute source and destination messages        │
│                                                                         │
│  3. Memory Update                                                       │
│     └── Update memory states for involved nodes                        │
│                                                                         │
│  4. Embedding Generation                                                │
│     └── Compute temporal embeddings using updated memories             │
│                                                                         │
│  5. Prediction Heads                                                    │
│     ├── Direction: P(price_up | embedding)                             │
│     ├── Magnitude: E[return | embedding]                               │
│     ├── Volatility: E[volatility | embedding]                          │
│     └── Correlation: P(asset_i moves with asset_j)                     │
│                                                                         │
│  6. Signal Aggregation                                                  │
│     └── Combine predictions into actionable signals                    │
│                                                                         │
│  7. Position Sizing                                                     │
│     └── Kelly criterion with confidence weighting                      │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Trading Strategy Example

```rust
// Pseudo-code for TGN-based trading strategy

fn process_event(event: MarketEvent, tgn: &mut TemporalGNN) -> Option<Signal> {
    // Update memory for source and destination nodes
    let messages = tgn.compute_messages(&event);
    tgn.update_memories(messages);

    // Get current embeddings
    let embeddings = tgn.get_embeddings(&[event.symbol]);

    // Predict direction and confidence
    let (prob_up, prob_down, confidence) = tgn.predict_direction(&embeddings);

    // Generate signal if confidence is high enough
    if confidence > 0.7 {
        if prob_up > 0.6 {
            Some(Signal::Long {
                symbol: event.symbol,
                strength: prob_up,
                confidence
            })
        } else if prob_down > 0.6 {
            Some(Signal::Short {
                symbol: event.symbol,
                strength: prob_down,
                confidence
            })
        } else {
            None
        }
    } else {
        None
    }
}
```

## Implementation Strategy

### Module Architecture

```
346_temporal_gnn_trading/
├── src/
│   ├── lib.rs              # Library root
│   ├── memory/
│   │   ├── mod.rs          # Memory module
│   │   ├── state.rs        # Memory state management
│   │   └── updater.rs      # Memory update mechanisms
│   ├── message/
│   │   ├── mod.rs          # Message module
│   │   ├── function.rs     # Message computation
│   │   └── aggregator.rs   # Message aggregation
│   ├── temporal/
│   │   ├── mod.rs          # Temporal module
│   │   ├── encoder.rs      # Time encoding
│   │   └── attention.rs    # Temporal attention
│   ├── embedding/
│   │   ├── mod.rs          # Embedding module
│   │   └── graph_conv.rs   # Graph convolution
│   ├── data/
│   │   ├── mod.rs          # Data module
│   │   ├── bybit.rs        # Bybit API client
│   │   ├── events.rs       # Event definitions
│   │   └── features.rs     # Feature engineering
│   ├── strategy/
│   │   ├── mod.rs          # Strategy module
│   │   ├── signals.rs      # Signal generation
│   │   └── execution.rs    # Order execution
│   └── utils/
│       ├── mod.rs          # Utilities
│       └── metrics.rs      # Performance metrics
├── examples/
│   ├── basic_tgn.rs        # Basic TGN example
│   ├── live_trading.rs     # Live trading demo
│   └── backtest.rs         # Backtesting example
└── tests/
    └── integration.rs      # Integration tests
```

### Key Design Principles

1. **Event-Driven**: Process events as they arrive, not in fixed batches
2. **Memory Efficient**: Incremental updates, not full recomputation
3. **Low Latency**: Optimized for real-time signal generation
4. **Type Safe**: Leverage Rust's type system for correctness
5. **Modular**: Independent components for easy testing and extension

## Risk Management

### Temporal-Aware Risk Metrics

```
┌────────────────────────────────────────────────────────────────┐
│              Temporal Risk Assessment                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Memory-Based Confidence:                                       │
│  └── Higher confidence when memories are fresh                 │
│  └── Lower confidence when many nodes have stale memories      │
│                                                                 │
│  Event Rate Risk:                                               │
│  └── High event rate → Higher volatility risk                  │
│  └── Low event rate → Possible liquidity risk                  │
│                                                                 │
│  Correlation Stability:                                         │
│  └── Track how quickly correlation edges change                │
│  └── High instability → Reduce position sizes                  │
│                                                                 │
│  Memory Divergence:                                             │
│  └── Large difference between recent and older memory          │
│  └── Indicates regime change → Extra caution                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Position Limits

```
Risk Constraints:
├── Max Position Size: 5% of portfolio
├── Max Correlated Exposure: 15%
├── Max Memory Staleness: 1 hour
├── Min Confidence Threshold: 0.6
├── Max Drawdown Trigger: 10%
└── Leverage Limit: 3x
```

### Circuit Breakers

1. **Memory Staleness**: Pause if critical node memories are too old
2. **Event Storm**: Reduce activity during abnormal event rates
3. **Correlation Breakdown**: Exit if edge structure changes rapidly
4. **Model Uncertainty**: Stop if prediction confidence drops

## Performance Metrics

### Model Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Link Prediction AUC | Predicting future interactions | > 0.80 |
| Direction Accuracy | Price direction prediction | > 56% |
| Temporal Precision | Timing of predictions | > 0.70 |
| Sharpe Ratio | Risk-adjusted returns | > 2.0 |
| Max Drawdown | Largest peak-to-trough | < 12% |
| Calmar Ratio | Return / Max Drawdown | > 1.5 |

### Latency Requirements

```
┌─────────────────────────────────────────────────────────┐
│                  Latency Budget                          │
├─────────────────────────────────────────────────────────┤
│ Event Reception:        < 5ms                            │
│ Message Computation:    < 20ms                           │
│ Memory Update:          < 10ms                           │
│ Embedding Generation:   < 50ms                           │
│ Signal Production:      < 15ms                           │
│ Order Submission:       < 30ms                           │
├─────────────────────────────────────────────────────────┤
│ Total Round Trip:       < 130ms                          │
└─────────────────────────────────────────────────────────┘
```

### Memory Efficiency

```
Memory Requirements per Node:
├── Memory state: 128 floats × 4 bytes = 512 bytes
├── Last update time: 8 bytes
├── Interaction counter: 4 bytes
├── Feature cache: 64 floats × 4 bytes = 256 bytes
└── Total per node: ~800 bytes

For 1000 assets: ~800 KB base memory
Plus event history buffer: ~10 MB
Total system requirement: < 50 MB
```

## References

1. Rossi, E., et al. (2020). "Temporal Graph Networks for Deep Learning on Dynamic Graphs." *ICML Workshop on Graph Representation Learning*. https://arxiv.org/abs/2006.10637

2. Xu, D., et al. (2020). "Inductive Representation Learning on Temporal Graphs." *ICLR*

3. Kazemi, S.M., et al. (2020). "Representation Learning for Dynamic Graphs: A Survey." *JMLR*

4. Kumar, S., et al. (2019). "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks." *KDD*

5. Trivedi, R., et al. (2019). "DyRep: Learning Representations over Dynamic Graphs." *ICLR*

6. Wang, Y., et al. (2021). "Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks." *ICLR*

7. Poursafaei, F., et al. (2022). "Towards Better Evaluation for Dynamic Link Prediction." *NeurIPS*

---

## Next Steps

- [View Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Русская версия
- [Run Examples](examples/) - Working Rust code

---

*Chapter 346 of Machine Learning for Trading*
