# Chapter 273: LIT Transformer for Limit Order Book Modeling

## 1. Introduction

The Limit Order Book (LOB) is the central data structure of modern electronic exchanges. It records outstanding buy and sell orders at every price level, providing the most granular view of supply and demand available to market participants. Predicting short-term price movements from LOB data is one of the most competitive problems in quantitative finance, because whoever decodes the LOB's signals first can capture fleeting alpha.

Traditional approaches to LOB modeling treat the book as a flat feature vector: stack the top-N bid and ask levels, maybe append recent trade features, and feed everything into an LSTM or a fully connected network. This flattening discards the rich relational structure that exists *within* each side of the book and *between* the book and the trade stream.

The **LIT (Limit-order-book Informed Transformer)** architecture addresses these limitations by introducing three dedicated components:

1. **LOB Encoder** -- a self-attention module that operates over price levels, learning which levels carry the most predictive information.
2. **Trade Flow Encoder** -- a temporal attention module that processes the stream of recent trades, capturing momentum, aggressor patterns, and clustering effects.
3. **Fusion Transformer** -- a cross-attention mechanism that allows the LOB representation and the trade-flow representation to interact, producing a joint embedding from which a prediction head forecasts the mid-price direction.

This chapter develops the mathematical foundations of LIT, implements a complete Rust prototype with real Bybit market data, and demonstrates how the architecture processes live BTCUSDT order books and trade streams.

## 2. Mathematical Foundations

### 2.1 Notation

Let the LOB snapshot at time $t$ consist of $L$ price levels on each side. For level $i$ on the bid side we have $(p_i^b, q_i^b)$ (price, quantity) and similarly $(p_i^a, q_i^a)$ on the ask side. We stack these into a matrix:

$$\mathbf{X}^{\text{LOB}} \in \mathbb{R}^{2L \times d_{\text{lob}}}$$

where each row is an embedding of a single price level. The raw features per level include price distance from mid-price, volume, number of orders, and cumulative volume.

For the trade stream, let the $K$ most recent trades be represented as:

$$\mathbf{X}^{\text{trade}} \in \mathbb{R}^{K \times d_{\text{trade}}}$$

where each row encodes trade price, size, aggressor side, and inter-arrival time.

### 2.2 LOB-Informed Self-Attention

Standard multi-head self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

For the LOB encoder, queries, keys, and values are derived from the level embeddings:

$$Q = \mathbf{X}^{\text{LOB}} W_Q, \quad K = \mathbf{X}^{\text{LOB}} W_K, \quad V = \mathbf{X}^{\text{LOB}} W_V$$

We augment the attention scores with a **positional bias** that encodes the relative distance between price levels:

$$A_{ij} = \frac{(Q_i)(K_j)^\top}{\sqrt{d_k}} + b(|i - j|)$$

where $b(\cdot)$ is a learned relative positional bias. This bias allows the model to understand that adjacent price levels have a stronger structural relationship than distant ones. Additionally, we add a **side indicator** to distinguish bid levels from ask levels, enabling the model to learn asymmetric patterns such as bid-side absorption or ask-side iceberg orders.

### 2.3 Trade Flow Temporal Attention

The trade flow encoder applies causal self-attention over the sequence of recent trades:

$$Q^t = \mathbf{X}^{\text{trade}} W_Q^t, \quad K^t = \mathbf{X}^{\text{trade}} W_K^t, \quad V^t = \mathbf{X}^{\text{trade}} W_V^t$$

A causal mask ensures that each trade can only attend to itself and earlier trades:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

The temporal encoding uses sinusoidal functions of inter-arrival times rather than fixed positional indices, reflecting the irregularly spaced nature of trade arrivals:

$$\text{TE}(t_k) = \left[\sin(\omega_1 \Delta t_k), \cos(\omega_1 \Delta t_k), \ldots, \sin(\omega_{d/2} \Delta t_k), \cos(\omega_{d/2} \Delta t_k)\right]$$

where $\Delta t_k = t_k - t_{k-1}$ is the inter-arrival time and $\omega_j$ are learnable frequencies.

### 2.4 Fusion Cross-Attention

The fusion transformer allows the LOB and trade representations to exchange information via cross-attention. Let $\mathbf{H}^{\text{LOB}}$ and $\mathbf{H}^{\text{trade}}$ denote the outputs of their respective encoders. We compute two cross-attention operations:

**Trade-to-LOB attention** (the trade stream queries the LOB):

$$\text{CrossAttn}_1 = \text{softmax}\left(\frac{(\mathbf{H}^{\text{trade}} W_Q^c)(\mathbf{H}^{\text{LOB}} W_K^c)^\top}{\sqrt{d_k}}\right) \mathbf{H}^{\text{LOB}} W_V^c$$

**LOB-to-Trade attention** (the LOB queries the trade stream):

$$\text{CrossAttn}_2 = \text{softmax}\left(\frac{(\mathbf{H}^{\text{LOB}} \hat{W}_Q^c)(\mathbf{H}^{\text{trade}} \hat{W}_K^c)^\top}{\sqrt{d_k}}\right) \mathbf{H}^{\text{trade}} \hat{W}_V^c$$

The fused representation is formed by concatenating and projecting:

$$\mathbf{H}^{\text{fused}} = W_{\text{proj}} \cdot [\text{pool}(\text{CrossAttn}_1) \| \text{pool}(\text{CrossAttn}_2)]$$

where $\text{pool}(\cdot)$ denotes mean-pooling across the sequence dimension.

### 2.5 Prediction Head

The prediction head maps the fused representation to a probability distribution over three classes: price up, price down, and price stationary.

$$\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{H}^{\text{fused}} + b_1) + b_2)$$

The model is trained with cross-entropy loss:

$$\mathcal{L} = -\sum_{c \in \{\text{up}, \text{down}, \text{stay}\}} y_c \log(\hat{y}_c)$$

## 3. Architecture Overview

```
                    ┌─────────────────────┐
                    │   Price Prediction   │
                    │   Head (3-class)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Fusion Transformer  │
                    │  (Cross-Attention)   │
                    └──┬───────────────┬──┘
                       │               │
            ┌──────────┴──┐     ┌──────┴──────────┐
            │ LOB Encoder │     │ Trade Flow Enc.  │
            │ (Self-Attn) │     │ (Causal Attn)    │
            └──────┬──────┘     └──────┬───────────┘
                   │                   │
         ┌─────────┴───────┐   ┌───────┴────────┐
         │ LOB Embedding   │   │ Trade Embedding│
         │ + Level Pos Enc │   │ + Temporal Enc │
         └─────────┬───────┘   └───────┬────────┘
                   │                   │
         ┌─────────┴───────┐   ┌───────┴────────┐
         │ Bybit Orderbook │   │ Bybit Recent   │
         │ Snapshot        │   │ Trades         │
         └─────────────────┘   └────────────────┘
```

### 3.1 LOB Encoder

The LOB encoder processes $2L$ level embeddings (L bid levels + L ask levels). Each level is represented by features: distance from mid-price, volume, order count, and cumulative volume. A learnable level-position embedding is added, along with a side indicator (bid=0, ask=1). The encoder applies $N_{\text{LOB}}$ layers of multi-head self-attention with the relative positional bias described in Section 2.2.

### 3.2 Trade Flow Encoder

The trade flow encoder processes $K$ recent trades. Each trade is embedded with features: price relative to mid-price, size, aggressor side, and log inter-arrival time. Temporal positional encoding based on inter-arrival times replaces standard positional encoding. The encoder applies $N_{\text{trade}}$ layers of causal multi-head self-attention.

### 3.3 Fusion Transformer

The fusion transformer takes the encoded LOB and trade representations, performs bidirectional cross-attention, and produces a single fused vector via mean-pooling and projection. This vector captures the joint state of the order book microstructure and recent trading activity.

## 4. Rust Implementation

The implementation in `rust/src/lib.rs` provides:

- **`LobEncoder`**: Self-attention over order book levels with relative positional bias.
- **`TradeFlowEncoder`**: Causal self-attention over recent trade events.
- **`FusionTransformer`**: Cross-attention between LOB and trade encodings.
- **`PredictionHead`**: Two-layer feedforward network for 3-class price direction prediction.
- **`LitModel`**: End-to-end wrapper combining all components.
- **`BybitClient`**: HTTP client for fetching live orderbook snapshots and recent trades from Bybit's public API.

Key design decisions in the Rust implementation:

1. **ndarray for tensor operations**: We use `ndarray` for matrix operations, which provides efficient BLAS-backed computation without the overhead of a full deep learning framework.
2. **Explicit attention computation**: All attention mechanisms are implemented from scratch, making the architecture transparent and easy to modify.
3. **Blocking HTTP for simplicity**: The library uses `reqwest::blocking` for API calls, with an async example demonstrating `tokio` integration.

## 5. Bybit Data Integration

The implementation connects to Bybit's public REST API to fetch:

1. **Order Book Snapshots**: `GET /v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=50` returns up to 50 levels on each side.
2. **Recent Trades**: `GET /v5/market/recent-trade?category=linear&symbol=BTCUSDT&limit=100` returns up to 1000 recent trades.

Data preprocessing steps:

- **Price normalization**: All prices are expressed relative to the mid-price, scaled by the tick size.
- **Volume normalization**: Volumes are log-transformed and standardized.
- **Time encoding**: Trade timestamps are converted to inter-arrival times in milliseconds, then encoded with learnable sinusoidal functions.
- **Level ordering**: Bid levels are ordered best-to-worst (descending price), ask levels best-to-worst (ascending price), then concatenated.

## 6. Key Takeaways

1. **Structure-preserving encoding matters.** Treating the LOB as a sequence of price levels with positional encoding preserves the inherent structure that flat feature vectors destroy. The relative positional bias allows the model to learn that adjacent levels interact more strongly.

2. **Separate encoders for heterogeneous data.** The LOB and trade stream have fundamentally different structures: the LOB is a spatial snapshot while trades are a temporal sequence. Using dedicated encoders with appropriate attention patterns (full attention for LOB, causal attention for trades) respects these differences.

3. **Cross-attention enables information fusion.** The fusion transformer allows the model to discover interactions between book state and trade flow, such as large trades absorbing specific price levels or imbalance signals that predict aggressive order flow.

4. **Temporal encoding for irregular timestamps.** Financial data arrives at irregular intervals. Sinusoidal temporal encoding of inter-arrival times is more appropriate than fixed positional encoding, as it captures the time-varying nature of market activity.

5. **Real-time inference is feasible in Rust.** The Rust implementation achieves microsecond-level inference latency for single snapshots, making it suitable for live trading systems where every microsecond of latency can impact profitability.

6. **The architecture is modular.** Each component (LOB encoder, trade encoder, fusion, prediction head) can be independently improved or replaced. For example, one could substitute the LOB encoder with a CNN-based approach or add a third encoder for funding rate data.

7. **Class imbalance in price prediction.** The "stationary" class typically dominates at short horizons. Production systems should use weighted cross-entropy or focal loss, and evaluate with metrics like balanced accuracy or the Matthews correlation coefficient rather than raw accuracy.
