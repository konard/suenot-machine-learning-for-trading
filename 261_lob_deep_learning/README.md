# Chapter 261: LOB Deep Learning

## 1. Introduction

The Limit Order Book (LOB) is the fundamental data structure that drives price discovery on modern electronic exchanges. It maintains a real-time record of all outstanding buy (bid) and sell (ask) orders at various price levels. Understanding the dynamics of the LOB provides a significant edge in predicting short-term price movements, market microstructure events, and liquidity conditions.

**LOB Deep Learning** applies neural network architectures to raw order book data — the full depth of bids and asks across multiple price levels — to forecast mid-price movements, detect liquidity shifts, and inform execution strategies. Unlike traditional approaches that rely on hand-crafted features from OHLCV data, LOB deep learning operates directly on the high-dimensional, time-varying structure of the order book itself.

The key insight is that the shape of the order book — the distribution of volume across price levels, the imbalance between bids and asks, and how these quantities evolve over time — contains rich predictive information about future price movements. Deep learning models can learn to extract and combine these signals in ways that far exceed what manual feature engineering can achieve.

In this chapter, we build a complete LOB deep learning system in Rust, demonstrate it with both stock market and cryptocurrency (Bybit) data, and explore how different neural network architectures capture order book dynamics for trading decisions.

## 2. Mathematical Foundations

### 2.1 Order Book Representation

At any time `t`, the LOB can be represented as a snapshot vector containing `L` levels on each side:

```
x(t) = [p_a^1, v_a^1, p_b^1, v_b^1, ..., p_a^L, v_a^L, p_b^L, v_b^L]
```

where `p_a^i` and `v_a^i` are the price and volume at the i-th ask level, and `p_b^i` and `v_b^i` are the price and volume at the i-th bid level. For `L = 10` levels, this gives a 40-dimensional snapshot vector.

### 2.2 Mid-Price and Label Definition

The mid-price is defined as:

```
p_mid(t) = (p_a^1(t) + p_b^1(t)) / 2
```

The prediction target is the smoothed future mid-price movement:

```
m(t) = (1/k) * sum_{i=1}^{k} p_mid(t+i) - p_mid(t)
```

This is classified into three categories using threshold `theta`:
- **Up** (label = 2): `m(t) > theta`
- **Down** (label = 0): `m(t) < -theta`
- **Stationary** (label = 1): `-theta <= m(t) <= theta`

### 2.3 Order Flow Imbalance (OFI)

A key derived feature is the Order Flow Imbalance:

```
OFI(t) = (v_b^1(t) - v_b^1(t-1)) * I(p_b^1(t) >= p_b^1(t-1))
        - (v_a^1(t) - v_a^1(t-1)) * I(p_a^1(t) <= p_a^1(t-1))
```

where `I(·)` is the indicator function. Positive OFI suggests buying pressure; negative OFI suggests selling pressure.

### 2.4 Volume Imbalance

The volume imbalance at each level captures the relative strength of bids vs asks:

```
VI^i(t) = (v_b^i(t) - v_a^i(t)) / (v_b^i(t) + v_a^i(t))
```

This normalized measure ranges from -1 (all ask volume) to +1 (all bid volume).

### 2.5 Deep Learning Architectures for LOB

#### Convolutional Approach (DeepLOB-style)

The convolutional approach treats the LOB snapshot sequence as a 2D structure:

```
Input: [T, 4L] time-series of LOB snapshots
  -> Conv1D layers (extract spatial patterns across levels)
  -> Inception modules (multi-scale feature extraction)
  -> LSTM layer (capture temporal dependencies)
  -> Fully connected -> Softmax (3-class prediction)
```

#### MLP Approach

A simpler but effective approach using fully-connected layers:

```
Input: [T * 4L] flattened LOB features
  -> Dense(256, ReLU) -> Dropout
  -> Dense(128, ReLU) -> Dropout
  -> Dense(64, ReLU)
  -> Dense(3, Softmax) (Up/Down/Stationary)
```

### 2.6 Cross-Entropy Loss

The model is trained with categorical cross-entropy:

```
L = -sum_{c=0}^{2} y_c * log(p_c)
```

where `y_c` is the one-hot encoded true label and `p_c` is the predicted probability for class `c`.

## 3. Why LOB Deep Learning Works for Trading

### 3.1 Information Advantage

The LOB contains information not visible in price charts alone. Large resting orders, thin liquidity zones, and order flow patterns provide signals about upcoming price movements before they appear in OHLCV data.

### 3.2 Non-Linear Patterns

The relationship between order book features and price movements is highly non-linear. For example, a large bid order might support the price — or it might be a "spoofing" signal that will be withdrawn. Deep learning models can learn to distinguish these contexts.

### 3.3 Multi-Scale Dependencies

Price movements are influenced by order book dynamics at multiple time scales: tick-by-tick microstructure effects, short-term momentum (seconds to minutes), and longer-term order flow trends. Architectures combining CNNs and RNNs naturally capture these multi-scale patterns.

### 3.4 Adaptive Feature Learning

Rather than relying on hand-crafted features like VWAP or book imbalance ratios, deep learning automatically discovers relevant features from raw data. This is particularly valuable because the most informative features may change across market regimes.

### 3.5 High-Frequency Edge

LOB deep learning is especially powerful in the high-frequency domain where:
- Traditional indicators are too slow
- The signal-to-noise ratio in raw order book data is higher than in aggregated data
- Execution quality depends critically on understanding microstructure dynamics

## 4. Rust Implementation

Our Rust implementation consists of several key components:

### 4.1 Network Architecture

The LOB neural network processes order book snapshots to predict mid-price direction:

```
Input: LOB features (40 features for 10 levels × 4 values)
  -> Hidden Layer 1 (256 units, ReLU)
  -> Hidden Layer 2 (128 units, ReLU)
  -> Hidden Layer 3 (64 units, ReLU)
  -> Output: 3 classes (Down, Stationary, Up) via Softmax
```

### 4.2 Core Components

- **`LOBNetwork`**: Multi-layer neural network for LOB-based mid-price direction prediction.
- **`LOBSnapshot`**: Structured representation of an order book snapshot with bid/ask prices and volumes.
- **`LOBEnvironment`**: Simulates a trading environment driven by LOB snapshots with position management and transaction costs.
- **`LOBAgent`**: Agent that uses the LOB network for epsilon-greedy action selection and trains via experience replay.
- **`BybitClient`**: Fetches OHLCV data from Bybit API and synthesizes LOB-style features.

### 4.3 Training Loop

```
for each episode:
    state = env.reset()
    while not done:
        features = extract_lob_features(state)
        action = epsilon_greedy(features)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        if buffer.len() >= batch_size:
            batch = buffer.sample(batch_size)
            predictions = network.forward(batch.states)
            loss = cross_entropy(predictions, batch.targets)
            update_weights(loss)
        periodically: copy weights to target network
```

### 4.4 Feature Engineering

The state vector includes features derived from LOB snapshots:

- Bid and ask prices at 10 depth levels
- Bid and ask volumes at 10 depth levels
- Volume imbalance at each level
- Bid-ask spread normalized by mid-price
- Order flow imbalance (OFI)
- Cumulative volume delta

## 5. Bybit Data Integration

The implementation fetches real market data from the Bybit exchange API:

```rust
// Fetch BTCUSDT 1-minute klines
let client = BybitClient::new();
let klines = client.fetch_klines_blocking("BTCUSDT", "1", 1000)?;
```

### 5.1 API Endpoints

We use the Bybit V5 API endpoint for kline (candlestick) data:
```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=1&limit=1000
```

### 5.2 LOB Reconstruction from OHLCV

Since full LOB data requires direct exchange feeds, we reconstruct LOB-style features from OHLCV data:

1. **Fetch**: Raw OHLCV data from Bybit for multiple symbols (BTCUSDT, ETHUSDT)
2. **Synthesize LOB**: Generate realistic bid/ask levels from high/low/close prices and volume
3. **Compute Features**: Calculate order flow imbalance, volume imbalance, spread dynamics
4. **Normalize**: Scale features to suitable ranges for neural network input
5. **Label**: Compute smoothed mid-price movements and classify as Up/Down/Stationary

### 5.3 Environment Simulation

The trading environment simulates order execution based on LOB dynamics:
- **Actions**: Buy (0), Sell (1), Hold (2)
- **Reward**: Log return of the position, adjusted for spread and transaction costs
- **Position tracking**: Maintains current position state (-1 short, 0 flat, 1 long)
- **Transaction costs**: Configurable fee model (default 0.075% for Bybit futures) plus half-spread slippage

## 6. Key Takeaways

1. **The order book is the source of truth**: While OHLCV data aggregates market activity into summary statistics, the LOB preserves the full picture of supply and demand. Deep learning on LOB data accesses information that is fundamentally unavailable in candlestick charts.

2. **Volume imbalance is the strongest single predictor**: Across multiple studies and our experiments, the ratio of bid to ask volume at the best levels is consistently the most informative feature for short-term price direction.

3. **Multi-scale architectures capture microstructure dynamics**: Combining convolutional layers (for cross-level patterns) with recurrent layers (for temporal dynamics) mirrors the multi-scale nature of order book evolution.

4. **Normalization is critical**: Raw LOB data varies enormously across instruments and time periods. Proper normalization — using log prices, volume ratios, and z-scoring — is essential for stable training.

5. **Labels matter more than architecture**: The choice of prediction horizon `k` and threshold `theta` has a larger impact on practical trading performance than the specific neural network architecture.

6. **Latency is the ultimate constraint**: In production LOB-based trading, model inference must complete in microseconds. Rust's zero-cost abstractions and lack of garbage collection make it an ideal choice for deploying LOB deep learning models.

7. **Rust enables production-grade LOB processing**: The combination of memory safety, zero-cost abstractions, and the ndarray crate enables efficient matrix operations at the speed required for real-time order book analysis.

8. **Synthetic LOB features from OHLCV are a practical starting point**: While full LOB data provides the richest signals, LOB-inspired features derived from OHLCV data still capture meaningful microstructure information and can be applied to any exchange.

## References

- Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing.
- Sirignano, J. A. (2019). "Deep Learning for Limit Order Books." Quantitative Finance.
- Briola, A., Turiel, J., Marcaccioli, R., & Aste, T. (2024). "Deep Limit Order Book Forecasting: A Microstructural Guide." arXiv:2403.09267.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events." Journal of Financial Econometrics.
- Bybit API Documentation: https://bybit-exchange.github.io/docs/v5/intro
