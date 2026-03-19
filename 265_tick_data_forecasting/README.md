# Chapter 265: Tick Data Forecasting with Machine Learning

## Introduction

Tick data represents the finest granularity of market information -- every individual trade and quote update as it occurs in real time. Unlike OHLCV bars that aggregate price action over fixed intervals, tick data preserves the exact microstructure of how prices evolve trade by trade. Forecasting the next tick (or sequence of ticks) is one of the most challenging problems in quantitative finance, sitting at the intersection of time series analysis, point processes, and deep learning.

This chapter explores machine learning approaches to tick-level forecasting for cryptocurrency markets. We build models that predict the direction, magnitude, and timing of the next price change using features derived from the raw tick stream. The ability to forecast even a few ticks ahead with above-random accuracy can translate into substantial alpha for high-frequency trading strategies.

## Mathematical Foundations

### Tick Data as a Marked Point Process

Let $\{(t_i, p_i, v_i)\}_{i=1}^{N}$ denote a sequence of ticks where $t_i$ is the timestamp, $p_i$ is the price, and $v_i$ is the volume. This is a marked point process where:

- The **inter-arrival times** $\delta_i = t_i - t_{i-1}$ follow some distribution $f(\delta | \mathcal{H}_{t_{i-1}})$ conditioned on the history $\mathcal{H}_t$
- The **marks** $(p_i, v_i)$ are drawn from a conditional distribution $g(p, v | t_i, \mathcal{H}_{t_i^{-}})$

### Hawkes Process for Tick Arrivals

The intensity function of a Hawkes process models self-exciting tick arrivals:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$$

where $\mu$ is the baseline intensity, $\alpha$ controls the excitation magnitude, and $\beta$ is the decay rate. The branching ratio $\alpha / \beta < 1$ ensures stationarity.

For multivariate Hawkes processes modeling buy and sell ticks separately:

$$\lambda_k(t) = \mu_k + \sum_{j \in \{b, s\}} \sum_{t_i^{(j)} < t} \alpha_{kj} \cdot e^{-\beta_{kj}(t - t_i^{(j)})}$$

### Tick-Level Feature Engineering

Given the tick sequence up to time $t$, we construct features:

**Trade Imbalance (TI):**
$$TI_n(t) = \frac{\sum_{i=1}^{n} v_i \cdot \text{sign}(\Delta p_i)}{\sum_{i=1}^{n} v_i}$$

where $\Delta p_i = p_i - p_{i-1}$ and sign classification uses the Lee-Ready algorithm or tick rule.

**Weighted Mid-Price Velocity:**
$$\text{WMPV}(t) = \frac{1}{n} \sum_{i=1}^{n} w_i \cdot \frac{\Delta p_i}{\delta_i}$$

with exponential weights $w_i = e^{-\lambda(t - t_i)} / \sum_j e^{-\lambda(t - t_j)}$.

**Volume-Weighted Average Price (VWAP) Deviation:**
$$d_{\text{VWAP}}(t) = p(t) - \frac{\sum_{i=1}^{n} p_i \cdot v_i}{\sum_{i=1}^{n} v_i}$$

**Tick Intensity Features:**
$$\hat{\lambda}(t, \Delta) = \frac{N(t - \Delta, t)}{\Delta}$$

computed over multiple windows $\Delta \in \{1s, 5s, 30s, 60s\}$.

### Autoregressive Tick Model

We model the next tick direction as a classification problem:

$$P(y_{n+1} = +1 | \mathbf{x}_n) = \sigma(\mathbf{w}^T \mathbf{x}_n + b)$$

where $\mathbf{x}_n$ is the feature vector at tick $n$ and $y_{n+1} \in \{-1, 0, +1\}$ indicates down-tick, no change, or up-tick.

For the multi-step forecast, we use a sequence-to-sequence approach:

$$\hat{y}_{n+1:n+h} = f_\theta(\mathbf{x}_{n-L+1:n})$$

where $L$ is the lookback window and $h$ is the forecast horizon.

### Loss Functions for Tick Forecasting

**Directional Accuracy Loss:**
$$\mathcal{L}_{\text{dir}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Magnitude-Weighted Loss:**
$$\mathcal{L}_{\text{mag}} = \frac{1}{N} \sum_{i=1}^{N} |\Delta p_i| \cdot (y_i - \hat{y}_i)^2$$

This penalizes errors on large price moves more heavily, aligning the loss with trading PnL.

**Quantile Regression Loss for Tick Size:**
$$\mathcal{L}_{\tau}(u) = u \cdot (\tau - \mathbb{1}_{u < 0})$$

where $\tau$ is the target quantile and $u = y - \hat{y}$.

## Architecture: Tick Forecasting Network

### Feature Extraction Layer

The raw tick sequence is processed through multiple parallel feature extractors:

1. **Temporal Convolution** over price changes with different kernel sizes (3, 5, 10 ticks)
2. **Exponential Moving Statistics** (mean, variance, skewness) of inter-arrival times
3. **Order Flow Features** including signed volume, trade imbalance, and intensity

### Sequence Encoder

An LSTM or Transformer encoder processes the feature sequence:

$$\mathbf{h}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1})$$

or with self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Prediction Head

The final hidden state feeds into multiple heads:
- **Direction Head**: softmax over $\{-1, 0, +1\}$
- **Magnitude Head**: linear output for $|\Delta p|$
- **Timing Head**: exponential output for $\delta_{n+1}$

## Applications in Trading

### High-Frequency Market Making

Tick forecasts inform quote placement:
- If the model predicts an up-tick with high confidence, the market maker can:
  - Tighten the ask (lean the spread)
  - Widen the bid to avoid adverse selection
  - Adjust inventory targets

### Aggressive Order Execution

For execution algorithms, tick forecasts determine:
- **When** to send limit orders vs. market orders
- **Where** to place limit orders relative to the mid-price
- **How much** to trade at each opportunity

### Short-Term Alpha Signals

Tick-level predictions aggregated over short horizons (100ms - 1s) generate alpha signals:
$$\alpha_t = \sum_{k=1}^{h} \hat{y}_{t+k} \cdot \gamma^k$$

where $\gamma < 1$ is a discount factor for distant predictions.

### Anomaly Detection

Deviations between predicted and actual tick behavior flag unusual market conditions:
$$z_t = \frac{|p_t - \hat{p}_t|}{\hat{\sigma}_t}$$

Large $z_t$ values may indicate news events, manipulation, or liquidity crises.

## Rust Implementation Walkthrough

Our Rust implementation provides a complete tick data forecasting system:

### Core Components

1. **`TickData` struct**: Represents individual ticks with timestamp, price, volume, and side
2. **`TickFeatureExtractor`**: Computes rolling features from the tick stream including trade imbalance, VWAP deviation, intensity, and volatility
3. **`HawkesProcess`**: Models self-exciting tick arrival intensity with MLE parameter estimation
4. **`TickForecaster`**: Neural network that predicts tick direction using extracted features
5. **`TickForecastBacktester`**: Evaluates forecast accuracy and simulates trading PnL

### Feature Pipeline

The `TickFeatureExtractor` maintains a rolling window of recent ticks and computes:
- Trade imbalance over configurable windows
- VWAP and deviation from current price
- Tick arrival intensity at multiple time scales
- Realized volatility from tick returns

### Hawkes Process Estimation

The `HawkesProcess` struct estimates parameters $(\mu, \alpha, \beta)$ via maximum likelihood:

```
log L = sum_i log(lambda(t_i)) - integral_0^T lambda(t) dt
```

The integral has a closed-form solution for exponential kernels, making estimation efficient.

### Neural Forecasting Model

The `TickForecaster` uses a two-layer feedforward network with ReLU activations:
- Input: feature vector from `TickFeatureExtractor`
- Hidden layers with configurable dimensions
- Output: 3-class softmax (down, flat, up) for direction prediction

Training uses cross-entropy loss with gradient descent approximation.

## Bybit Integration

The implementation fetches real-time tick-level data from Bybit's API:

1. **Kline Proxy**: Since the public REST API provides OHLCV at 1-minute minimum, we use 1-minute bars and synthesize tick-like data by decomposing each bar into simulated ticks based on volume distribution
2. **Real Tick Data**: For true tick data, the WebSocket API (`wss://stream.bybit.com/v5/public/linear`) provides real-time trade stream
3. **Feature Computation**: Features are computed incrementally as new ticks arrive, suitable for live trading

The `BybitClient` in our implementation fetches kline data and converts it to a tick-like format for demonstration purposes. In production, the WebSocket feed would provide actual trade-by-trade data.

## Performance Considerations

### Latency Optimization

Tick-level forecasting requires extremely low latency:
- Feature computation must be O(1) amortized (using rolling accumulators)
- Model inference should complete in microseconds
- Rust's zero-cost abstractions make it ideal for this workload

### Data Volume

A liquid cryptocurrency pair generates 10,000-100,000 ticks per day. Key strategies:
- Use circular buffers for rolling windows
- Pre-allocate memory for feature vectors
- Avoid unnecessary heap allocations during the hot path

### Model Staleness

Tick-level models degrade quickly as market conditions change:
- Retrain models every 1-4 hours
- Monitor prediction accuracy in real-time
- Fall back to simpler models when accuracy drops below threshold

## Key Takeaways

1. **Tick data is a marked point process** -- both the timing and marks (price, volume) carry predictive information. The Hawkes process captures the self-exciting nature of tick arrivals.

2. **Feature engineering dominates model choice** -- trade imbalance, VWAP deviation, and tick intensity are the most predictive features. Even simple linear models with good features can outperform complex models with poor features.

3. **Directional accuracy of 52-55% is meaningful** -- due to the high frequency of trading opportunities, even small edge compounds rapidly. A 53% directional accuracy on 10,000 ticks/day can be highly profitable after transaction costs.

4. **Latency is critical** -- tick-level strategies compete on speed. Rust's performance characteristics (no garbage collection, zero-cost abstractions, predictable memory layout) make it the natural choice for production tick forecasting systems.

5. **Model decay is rapid** -- tick-level patterns have half-lives measured in hours, not days. Continuous monitoring and frequent retraining are essential.

6. **Risk management at tick level requires distributional forecasts** -- point predictions are insufficient. Quantile regression and distributional forecasting help manage the tail risks inherent in high-frequency trading.

7. **Bybit's API ecosystem supports tick-level strategies** -- the WebSocket trade stream provides real-time tick data, while the REST API enables backtesting with historical candle data decomposed into synthetic ticks.

## References

- Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity*.
- Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*.
- Cont, R. (2011). Statistical modeling of high-frequency financial data. *IEEE Signal Processing Magazine*.
- Sirignano, J., & Cont, R. (2019). Universal features of price formation in financial markets. *PLoS ONE*.
- Dat, T. T., et al. (2023). Tick-level deep learning for high-frequency trading. *Quantitative Finance*.
