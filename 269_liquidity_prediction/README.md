# Chapter 269: Liquidity Prediction for Trading

## 1. Introduction

Liquidity -- the ability to execute a trade of a given size at a given price with minimal market impact -- is one of the most critical yet difficult-to-quantify features of any financial market. For algorithmic traders, liquidity is not merely an abstract quality: it directly determines the cost of entering and exiting positions, the feasibility of strategy capacity, and the risk of adverse selection during execution. A market that appears liquid on the surface (tight quoted spreads, deep order books) can become illiquid in milliseconds during a volatility event, leaving traders stranded with partially filled orders and unfavorable slippage.

Predicting liquidity -- forecasting how deep, tight, and resilient a market will be over the next seconds, minutes, or hours -- is therefore a natural and high-value application of machine learning. If a model can reliably anticipate periods of high and low liquidity, a trader can:

- **Time large orders** to execute during predictable high-liquidity windows, reducing market impact.
- **Size orders dynamically**, sending larger child orders when the book is expected to be deep and smaller orders when it thins.
- **Avoid adverse execution** by detecting impending liquidity droughts before they materialize.
- **Improve spread-capture strategies** by predicting when bid-ask spreads will tighten or widen.

In this chapter we develop a complete liquidity prediction pipeline in Rust. We formalize the three classical dimensions of liquidity (depth, tightness, resilience), engineer features from Limit Order Book (LOB) snapshots, build an LSTM-based liquidity forecaster, train an XGBoost-style gradient-boosted classifier for liquidity regime detection, and demonstrate optimal order sizing conditioned on predicted liquidity. All data is sourced from the Bybit cryptocurrency exchange via its public REST API.

---

## 2. Mathematical Foundations

### 2.1 The Three Dimensions of Liquidity

Market microstructure theory characterizes liquidity along three orthogonal dimensions:

**Tightness** measures the cost of a round-trip transaction. The most common proxy is the quoted bid-ask spread:

$$S_t = P_t^{ask} - P_t^{bid}$$

The relative spread normalizes by the mid-price $m_t = (P_t^{ask} + P_t^{bid}) / 2$:

$$s_t = \frac{S_t}{m_t}$$

A tight market has small $s_t$, meaning a trader can buy and immediately sell with minimal loss to the spread.

**Depth** measures the volume available at or near the best quotes. We define cumulative depth at level $N$ on each side:

$$D_t^{bid}(N) = \sum_{i=1}^{N} q_t^{bid,i}, \quad D_t^{ask}(N) = \sum_{i=1}^{N} q_t^{ask,i}$$

where $q_t^{bid,i}$ is the quantity at the $i$-th best bid level. Total near-touch depth is:

$$D_t(N) = D_t^{bid}(N) + D_t^{ask}(N)$$

The **depth profile** $\{D_t(1), D_t(2), \ldots, D_t(N)\}$ provides a multi-scale view of available liquidity. In deep markets, $D_t(N)$ grows steeply with $N$; in thin markets, depth accumulates slowly.

**Resilience** measures how quickly the order book recovers after a liquidity shock (e.g., a large market order consuming multiple levels). We define a resilience proxy as the ratio of depth recovery over a window $\Delta$:

$$R_t(\Delta) = \frac{D_{t+\Delta}(N)}{D_{t}(N)}$$

where $D_t(N)$ is measured immediately after a large trade event. A resilient market has $R_t(\Delta) \approx 1$ or greater within a short $\Delta$, indicating that new limit orders rapidly replenish consumed levels.

### 2.2 LOB Depth Profile Model

We model the cumulative depth as a function of price distance from the mid-price. Let $\delta_i$ be the price offset of the $i$-th level from the mid:

$$\delta_i^{bid} = m_t - P_t^{bid,i}, \quad \delta_i^{ask} = P_t^{ask,i} - m_t$$

Empirical studies show that cumulative depth often follows a power-law or log-linear relationship:

$$\ln D_t(\delta) \approx \alpha + \beta \ln \delta$$

The slope $\beta$ characterizes the concentration of liquidity. A high $\beta$ indicates liquidity is concentrated near the touch; a low $\beta$ means liquidity is dispersed across many levels.

### 2.3 Order Flow Imbalance

Order flow imbalance (OFI) captures the directional pressure on the book. At each snapshot:

$$\text{OFI}_t = \frac{D_t^{bid}(N) - D_t^{ask}(N)}{D_t^{bid}(N) + D_t^{ask}(N)}$$

OFI ranges from $-1$ (all depth on ask side) to $+1$ (all depth on bid side). Persistent positive OFI suggests buying pressure and potential upward price movement; negative OFI suggests selling pressure.

### 2.4 LSTM for Liquidity Time Series Forecasting

We use an LSTM network to forecast future liquidity metrics from historical sequences. Given a feature vector $x_t \in \mathbb{R}^d$ at time $t$ containing spread, depth at multiple levels, OFI, and volume features, the LSTM processes a lookback window $\{x_{t-L+1}, \ldots, x_t\}$ and produces a prediction:

$$\hat{y}_{t+1} = W_{out} \cdot h_t + b_{out}$$

where $h_t$ is the final hidden state. The LSTM cell equations are the standard four-gate formulation:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

For liquidity forecasting, the target $y_{t+1}$ can be the next-period total depth $D_{t+1}(N)$, the spread $s_{t+1}$, or a composite liquidity score.

### 2.5 XGBoost for Liquidity Regime Classification

We classify the market into liquidity regimes -- **High**, **Medium**, **Low** -- using a gradient-boosted decision tree ensemble. The regime labels are assigned from historical depth percentiles:

$$\text{Regime}_t = \begin{cases} \text{High} & \text{if } D_t(N) > P_{67}(D) \\ \text{Medium} & \text{if } P_{33}(D) \le D_t(N) \le P_{67}(D) \\ \text{Low} & \text{if } D_t(N) < P_{33}(D) \end{cases}$$

We implement a simplified gradient-boosted tree ensemble (stumps) to demonstrate the concept in pure Rust without external ML framework dependencies.

### 2.6 Optimal Order Sizing Under Predicted Liquidity

Given a predicted depth $\hat{D}_{t+1}(N)$ and a target execution quantity $Q$, the optimal child order size $q^*$ minimizes expected market impact:

$$q^* = \min\left(Q, \gamma \cdot \hat{D}_{t+1}(N)\right)$$

where $\gamma \in (0, 1)$ is a participation rate cap (e.g., 10% of predicted available depth). This ensures the order does not consume more than a fraction of the expected book, limiting adverse price impact.

The expected slippage from walking the book is approximately:

$$\text{Slippage}(q) \approx \frac{q^2}{2 \cdot D_t(N)} \cdot \bar{\delta}$$

where $\bar{\delta}$ is the average price gap per level. This quadratic relationship underscores the importance of accurate depth prediction: underestimating depth leads to unnecessarily small orders (opportunity cost), while overestimating leads to excessive market impact.

---

## 3. ML Models

### 3.1 LSTM Liquidity Forecaster

The LSTM model takes as input a sequence of feature vectors over a lookback window. Each feature vector includes:

| Feature | Description |
|---------|-------------|
| `spread` | Relative bid-ask spread $s_t$ |
| `depth_1` | Total depth at top 1 level |
| `depth_5` | Total depth at top 5 levels |
| `depth_10` | Total depth at top 10 levels |
| `ofi` | Order flow imbalance |
| `volume` | Recent trade volume |
| `volatility` | Rolling price standard deviation |

The output is the predicted total depth at 5 levels for the next period. The model is trained by minimizing MSE loss with backpropagation through time. In our Rust implementation, we use a from-scratch LSTM cell with Xavier-initialized weights.

### 3.2 XGBoost Liquidity Regime Classifier

The gradient-boosted classifier uses the same feature set but targets discrete regime labels. Each boosting round fits a decision stump (single-split tree) to the pseudo-residuals of a cross-entropy loss. The ensemble output is a weighted sum of stump predictions, passed through softmax for class probabilities.

This approach is valuable for execution algorithms that switch between aggressive and passive strategies depending on the liquidity regime.

---

## 4. Applications

### 4.1 Optimal Order Sizing

The primary application of liquidity prediction is dynamic order sizing. A TWAP or VWAP algorithm typically splits a parent order into equal-sized child orders. By conditioning child order size on predicted liquidity:

- During **high liquidity** periods: increase child order size to accelerate completion
- During **medium liquidity** periods: maintain baseline size
- During **low liquidity** periods: reduce child order size or pause execution

This adaptive approach reduces total market impact cost compared to naive uniform splitting.

### 4.2 Timing of Large Trades

Institutional traders executing large block orders benefit from scheduling execution during predictable liquidity windows. The LSTM forecaster identifies intraday liquidity patterns (e.g., higher depth during overlap of trading sessions, lower depth during off-hours) and adapts the execution schedule accordingly.

### 4.3 Spread Prediction for Market Making

Market makers can use liquidity predictions to adjust their quoting strategy. When the model predicts widening spreads (low liquidity), a market maker should widen their quotes to avoid adverse selection. When tight spreads are predicted, narrower quotes capture more flow.

---

## 5. Rust Implementation

The Rust implementation in `rust/src/lib.rs` provides:

1. **`LiquidityMetrics`** -- Computes depth at N levels, relative spread, order flow imbalance, and a composite liquidity score from raw orderbook data.

2. **`LstmCell` and `LstmForecaster`** -- A from-scratch LSTM implementation for time-series liquidity forecasting. The forecaster processes a sequence of feature vectors and outputs predicted next-period depth.

3. **`LiquidityRegimeClassifier`** -- A gradient-boosted stump ensemble that classifies the market into High, Medium, or Low liquidity regimes.

4. **`OptimalOrderSizer`** -- Computes the recommended child order size given predicted depth and a participation rate constraint.

5. **`BybitClient`** -- Fetches live orderbook snapshots from the Bybit REST API.

The implementation is dependency-light, relying only on `ndarray` for linear algebra, `reqwest` for HTTP, and `serde` for JSON deserialization.

---

## 6. Bybit Data Integration

We use the Bybit V5 public REST API to fetch real-time orderbook data:

```
GET https://api.bybit.com/v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=50
```

The response contains arrays of `[price, quantity]` pairs for bids and asks. Our `BybitClient` struct parses this into a structured `OrderbookSnapshot` that feeds directly into the liquidity metrics computation pipeline.

For historical analysis, multiple snapshots are collected at regular intervals (e.g., every 1 second) and stored as time-series feature vectors for LSTM training.

---

## 7. Key Takeaways

1. **Liquidity is multi-dimensional**: depth, tightness, and resilience each capture different aspects. Effective prediction requires modeling all three.

2. **LSTM networks** are well-suited for liquidity forecasting because they capture both fast intraday patterns and slower regime shifts in order book dynamics.

3. **Regime classification** (High/Medium/Low liquidity) provides a discrete signal that execution algorithms can directly act on, switching between aggressive and passive strategies.

4. **Optimal order sizing** scales linearly with predicted depth, and market impact scales quadratically with order size relative to depth -- making accurate depth prediction highly valuable.

5. **Real-time orderbook data** from exchanges like Bybit provides the raw material for liquidity prediction. Even simple features (spread, depth at N levels, OFI) contain substantial predictive signal.

6. **Rust implementation** enables low-latency inference suitable for live trading systems, with the entire prediction pipeline running in microseconds on modern hardware.

7. **Participation rate constraints** ($\gamma$) provide a safety mechanism that prevents execution algorithms from consuming too large a fraction of available liquidity, even when predictions are overconfident.
