# Chapter 266: Trade Classification with Machine Learning

## Introduction

Trade classification is one of the most fundamental problems in market microstructure. When a trade occurs on an exchange, it results from the interaction of a buy order and a sell order. The question of **who initiated the trade** -- the buyer or the seller -- is critical for computing many downstream quantities: order flow imbalance, the probability of informed trading (PIN), effective spread decomposition, and price impact estimation.

Most exchanges do not report the initiator of a trade in their public market data feeds (though some, like Bybit, do). Even when the exchange does report the aggressor side, researchers often need to classify historical trades from datasets that lack this information. This chapter surveys the classical rule-based methods for trade classification, introduces the Bulk Volume Classification approach, and then builds an ML ensemble that combines all of these signals for improved accuracy.

## Mathematical Foundations

### The Trade Initiation Problem

Consider a trade at time $t$ with price $P_t$ and quantity $Q_t$. The prevailing bid-ask quote at time $t$ is $(B_t, A_t)$ with midpoint:

$$M_t = \frac{B_t + A_t}{2}$$

The spread is $S_t = A_t - B_t$. We wish to classify each trade as **buyer-initiated** (the buyer was the aggressor who crossed the spread) or **seller-initiated**.

### Tick Test

The tick test (or tick rule) is the simplest classification method. It uses the sign of the price change:

$$\text{Side}_t = \begin{cases} \text{Buy} & \text{if } P_t > P_{t-1} \quad (\text{uptick}) \\ \text{Sell} & \text{if } P_t < P_{t-1} \quad (\text{downtick}) \\ \text{Side}_{t-1} & \text{if } P_t = P_{t-1} \quad (\text{zero-tick rule}) \end{cases}$$

The intuition is that buyer-initiated trades push the price up, and seller-initiated trades push it down. The zero-tick rule carries forward the previous classification when the price is unchanged.

**Accuracy**: The tick test typically achieves 70-85% accuracy depending on the market and asset class. It performs worse in markets with large tick sizes relative to the spread.

### Quote Rule

The quote rule uses the position of the trade price relative to the midpoint:

$$\text{Side}_t = \begin{cases} \text{Buy} & \text{if } P_t > M_t \\ \text{Sell} & \text{if } P_t < M_t \\ \text{Indeterminate} & \text{if } P_t = M_t \end{cases}$$

The logic is that a buyer crossing the spread pays a price above the midpoint (closer to the ask), while a seller receives a price below the midpoint (closer to the bid).

### Lee-Ready Algorithm

Lee and Ready (1991) combined the two approaches. The Lee-Ready algorithm uses the quote rule as the primary classifier and falls back to the tick test when the trade price equals the midpoint:

$$\text{LR}_t = \begin{cases} \text{QuoteRule}(P_t, M_t) & \text{if } P_t \neq M_t \\ \text{TickTest}(P_t, P_{t-1}) & \text{if } P_t = M_t \end{cases}$$

Lee and Ready also recommended a 5-second delay in quote matching (using the quote from 5 seconds before the trade) to account for reporting delays. In modern electronic markets, this delay is typically unnecessary.

**Accuracy**: Lee-Ready typically achieves 85-93% accuracy and remains the gold standard among rule-based methods.

### Bulk Volume Classification (BVC)

Easley, Lopez de Prado, and O'Hara (2012) proposed Bulk Volume Classification, which works with aggregated volume bars rather than individual trades. For each bar $i$ with closing price $P_i$:

$$V^B_i = V_i \cdot \Phi\left(\frac{P_i - P_{i-1}}{\sigma_{\Delta P}}\right)$$

where $\Phi$ is the standard normal CDF and $\sigma_{\Delta P}$ is the standard deviation of price changes. The buy-volume fraction is:

$$f^B_i = \Phi\left(\frac{\Delta P_i}{\sigma_{\Delta P}}\right)$$

BVC is particularly useful when individual trade data is unavailable and only bar-level data (OHLCV) exists.

### ML-Based Classification

Machine learning approaches treat trade classification as a binary classification problem. For each trade, we extract a feature vector:

$$\mathbf{x}_t = \left[\frac{P_t - M_t}{S_t}, \frac{Q_t}{\bar{Q}}, \text{tick}_t, S_t, \frac{B^s_t - A^s_t}{B^s_t + A^s_t}, \Delta t\right]$$

where:
- $(P_t - M_t) / S_t$ is the relative price position within the spread
- $Q_t / \bar{Q}$ is the relative trade size
- $\text{tick}_t \in \{-1, 0, +1\}$ is the tick direction
- $S_t$ is the spread
- $(B^s_t - A^s_t) / (B^s_t + A^s_t)$ is the quote imbalance (bid size vs ask size)
- $\Delta t$ is the time since the last trade

A logistic regression model predicts the probability of a buy:

$$P(\text{Buy} | \mathbf{x}_t) = \sigma(\mathbf{w}^T \mathbf{x}_t + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_t + b)}}$$

### Ensemble Approach

Our ensemble combines the ML prediction with the classical methods using a weighted vote:

$$P_{\text{ensemble}} = \alpha_1 \cdot P_{\text{ML}} + \alpha_2 \cdot V_{\text{LR}} + \alpha_3 \cdot V_{\text{tick}} + \alpha_4 \cdot f^B_{\text{BVC}}$$

where $V_{\text{LR}}, V_{\text{tick}} \in \{0, 0.5, 1\}$ are the votes from Lee-Ready and tick test (0.5 for indeterminate), and $\alpha_1 + \alpha_2 + \alpha_3 + \alpha_4 = 1$.

## Applications in Trading

### 1. Order Flow Imbalance

Trade classification is the first step in computing order flow imbalance (OFI), a powerful predictor of short-term price movements:

$$\text{OFI}_t = \sum_{i \in \text{window}} Q_i \cdot \mathbb{1}[\text{Side}_i = \text{Buy}] - \sum_{i \in \text{window}} Q_i \cdot \mathbb{1}[\text{Side}_i = \text{Sell}]$$

### 2. Probability of Informed Trading (PIN)

The PIN model requires counts of buyer-initiated and seller-initiated trades per period. Misclassification directly biases PIN estimates.

### 3. Effective Spread Decomposition

The effective half-spread can be decomposed into adverse selection and realized spread components, both of which require knowledge of the trade initiator:

$$\text{EffectiveSpread}_t = 2 \cdot d_t \cdot (P_t - M_t)$$

where $d_t = +1$ for buys and $d_t = -1$ for sells.

### 4. Market Making Strategies

A market maker that correctly identifies the aggressor can estimate its adverse selection costs and optimize its quoting strategy accordingly.

## Rust Implementation Walkthrough

Our implementation in `rust/src/lib.rs` provides five main components:

### Data Structures

We define `Trade`, `Quote`, `TradeSide`, and supporting structures. Each trade has a timestamp, price, quantity, and an optional side (which may be unknown and needs classification).

### TickTestClassifier

The tick test iterates through trades sequentially, comparing each price to the previous one. It maintains a `last_side` variable for the zero-tick rule:

```rust
for i in 1..trades.len() {
    let diff = trades[i].price - trades[i - 1].price;
    if diff > 0.0 { last_side = TradeSide::Buy; }
    else if diff < 0.0 { last_side = TradeSide::Sell; }
    results.push(Some(last_side));
}
```

### QuoteRuleClassifier and LeeReadyClassifier

The quote rule compares each trade price to the contemporaneous midpoint. Lee-Ready combines both, using `quote_class.or(tick_class)` to fall back to the tick test when the quote rule is indeterminate.

### BulkVolumeClassifier

BVC uses a logistic approximation to the normal CDF for efficiency:

$$\Phi(x) \approx \frac{1}{1 + e^{-1.7x}}$$

This avoids requiring a full normal CDF implementation while maintaining good accuracy.

### EnsembleClassifier

The ensemble runs all four methods in parallel and combines their outputs with weights: ML 40%, Lee-Ready 30%, Tick Test 15%, BVC 15%. The ML component uses a logistic regression model with pre-defined weights for the six extracted features.

### Feature Extraction

The `extract_features` function computes the six-dimensional feature vector for each trade, handling edge cases like the first trade (no tick direction, no time delta) and zero-spread quotes.

## Bybit Integration

The implementation includes three Bybit API endpoints:

1. **`fetch_bybit_trades`**: Retrieves recent trades from `/v5/market/recent-trading-records`. Bybit conveniently reports the aggressor side, allowing us to validate our classification algorithms.

2. **`fetch_bybit_orderbook`**: Retrieves the current best bid/ask from `/v5/market/orderbook`, providing quote data for the quote rule and Lee-Ready classifiers.

3. **`fetch_bybit_klines`**: Retrieves OHLCV candles from `/v5/market/kline` for BVC analysis.

The trading example fetches live BTCUSDT data from Bybit, classifies each trade with all methods, and then compares the predictions against the exchange-reported labels to compute accuracy.

## Running the Code

```bash
cd 266_trade_classification/rust

# Run tests
cargo test

# Run the trading example (fetches live Bybit data)
cargo run --example trading_example
```

## Key Takeaways

1. **Trade classification is foundational**: Many microstructure metrics (OFI, PIN, effective spread) depend on correctly identifying the trade initiator.

2. **Classical methods are surprisingly strong**: Lee-Ready achieves 85-93% accuracy with simple rules. The tick test alone gets 70-85%.

3. **ML improves at the margins**: Machine learning can capture non-linear relationships and additional features (quote imbalance, trade size patterns) that rule-based methods miss.

4. **Ensemble methods work best**: Combining ML predictions with classical rules produces more robust results than any single method, especially in edge cases.

5. **BVC enables bar-level analysis**: When individual trade data is unavailable, BVC provides a principled way to estimate buy/sell volume from OHLCV bars.

6. **Exchange labels are valuable but not always available**: Bybit reports the aggressor side, making it an excellent platform for validating classification algorithms. Many other data sources lack this information.

7. **Feature engineering matters**: The relative price position within the spread is by far the most informative feature, followed by tick direction and quote imbalance. Trade size and time-between-trades provide marginal improvements.

## References

- Lee, C.M.C. and Ready, M.J. (1991). "Inferring Trade Direction from Intraday Data." *Journal of Finance*, 46(2), 733-746.
- Easley, D., Lopez de Prado, M., and O'Hara, M. (2012). "Bulk Classification of Trading Activity." *Johnson School Research Paper Series*, No. 8-2012.
- Ellis, K., Michaely, R., and O'Hara, M. (2000). "The Accuracy of Trade Classification Rules: Evidence from Nasdaq." *Journal of Financial and Quantitative Analysis*, 35(4), 529-551.
- Chakrabarty, B., Li, B., Nguyen, V., and Van Ness, R.A. (2007). "Trade Classification Algorithms for Electronic Communications Network Trades." *Journal of Banking & Finance*, 31(12), 3806-3821.
- Rosenthal, D. (2012). "Performance Metrics for Algorithmic Traders." PhD thesis, Imperial College London.
