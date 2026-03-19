# Chapter 268: Spread Modeling with ML for Trading

## Introduction

The bid-ask spread is one of the most fundamental quantities in market microstructure. It represents the cost of immediacy: the premium a trader pays to execute a transaction right now rather than waiting for a counterparty to arrive. For market makers, the spread is revenue; for institutional investors, it is an execution cost that erodes alpha. Understanding, predicting, and decomposing the spread is therefore critical for both sides of the liquidity provision equation.

Traditional microstructure theory provides elegant models for spread behavior. The Roll (1984) model infers the effective spread from the serial covariance of trade-price changes. The Glosten-Milgrom (1985) and Kyle (1985) models decompose the spread into adverse selection, inventory risk, and order processing components. However, these models rely on stylized assumptions — normally distributed arrivals, constant information asymmetry, single-asset settings — that rarely hold in modern electronic markets.

Machine learning offers a powerful toolkit for extending these classical ideas. ML models can capture nonlinear relationships between spread determinants, adapt to regime changes in volatility and liquidity, and incorporate high-dimensional features from the limit order book. In this chapter, we develop a complete framework for spread modeling with ML, covering spread calculation, decomposition, prediction, and application to trading strategy evaluation.

We implement everything in Rust for performance, and demonstrate integration with Bybit's API for real-time cryptocurrency orderbook data. The techniques apply broadly to equities, futures, FX, and any market with a visible order book.

## Mathematical Foundations

### Spread Definitions

The **quoted spread** is the difference between the best ask and best bid prices at time $t$:

$$S_t^{quoted} = P_t^{ask} - P_t^{bid}$$

The **relative spread** (or proportional spread) normalizes by the midpoint:

$$S_t^{relative} = \frac{P_t^{ask} - P_t^{bid}}{M_t}, \quad M_t = \frac{P_t^{ask} + P_t^{bid}}{2}$$

The **effective spread** measures actual execution costs using trade prices:

$$S_t^{effective} = 2 \cdot D_t \cdot (P_t^{trade} - M_t)$$

where $D_t \in \{+1, -1\}$ is the trade direction indicator (buy or sell).

### Roll Spread Estimator

Roll (1984) showed that under the assumption of an efficient market with a constant spread, the effective spread can be estimated from the autocovariance of price changes. Let $\Delta P_t = P_t - P_{t-1}$ be the price change. Then:

$$\text{Cov}(\Delta P_t, \Delta P_{t-1}) = -c^2$$

where $c$ is the half-spread. The Roll estimator is:

$$\hat{S}_{Roll} = 2\hat{c} = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}$$

When the autocovariance is positive (which happens in trending markets), the Roll estimator is undefined. In practice, we set $\hat{S}_{Roll} = 0$ in that case or use the absolute value with a sign adjustment.

### Spread Decomposition

The spread can be decomposed into three economic components:

1. **Adverse selection component ($\alpha$)**: Compensation for trading with informed traders who possess private information. When a market maker transacts with an informed trader, the price moves against them. This component is larger when information asymmetry is high — around earnings announcements, during news events, or for illiquid securities.

2. **Inventory component ($\beta$)**: Compensation for the risk of holding an unbalanced inventory. Market makers who accumulate a large position face price risk. This component increases with volatility and position size.

3. **Order processing component ($\gamma$)**: The fixed costs of providing liquidity — exchange fees, technology costs, opportunity costs of capital. This is the "base" spread that exists even without adverse selection or inventory risk.

The total spread is:

$$S = \alpha + \beta + \gamma$$

The Huang-Stoll (1997) decomposition estimates these components from the joint behavior of trade prices and quote revisions. Let $Q_t$ be the trade direction and $\Delta M_t$ be the change in the midpoint after a trade. Then:

$$\Delta M_t = \frac{S}{2}(\alpha + \beta) Q_t - \frac{S}{2} \beta Q_{t-1} + \epsilon_t$$

By regressing midpoint changes on current and lagged trade directions, we can estimate $\alpha$, $\beta$, and $\gamma = 1 - \alpha - \beta$.

### Linear Regression for Spread Prediction

We model the spread as a linear function of observable features:

$$S_t = \beta_0 + \beta_1 \sigma_t + \beta_2 V_t + \beta_3 D_t + \beta_4 OI_t + \epsilon_t$$

where:
- $\sigma_t$ is recent realized volatility
- $V_t$ is recent trading volume
- $D_t$ is order book depth (total visible liquidity)
- $OI_t$ is order imbalance (bid size minus ask size, normalized)

The parameters are estimated by ordinary least squares:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

where $\mathbf{X}$ is the feature matrix and $\mathbf{y}$ is the vector of observed spreads.

## ML Models for Spread Analysis

### Regression for Spread Prediction

The simplest approach is linear regression, but spreads exhibit several properties that make naive OLS suboptimal:

- **Heteroskedasticity**: Spread variance increases during volatile periods
- **Non-negativity**: Spreads cannot be negative, but OLS can predict negative values
- **Fat tails**: Spread distributions are right-skewed with occasional spikes
- **Regime dependence**: The relationship between features and spreads changes across market conditions

To address these issues, we can use:

1. **Log-transformed regression**: Model $\log(S_t)$ instead of $S_t$ to enforce positivity and reduce skewness.
2. **Quantile regression**: Predict specific quantiles of the spread distribution rather than the mean, which is useful for worst-case execution cost analysis.
3. **Ridge/LASSO regression**: Add regularization when using many correlated features from the order book.

In our implementation, we use standard linear regression as the foundation, which can be extended to these variants.

### Classification for Spread Regime

Markets alternate between tight-spread (liquid) and wide-spread (illiquid) regimes. Identifying the current regime is valuable for:

- **Market making**: Adjust quote aggressiveness based on the regime
- **Execution**: Choose between aggressive and passive strategies
- **Risk management**: Widen stop-loss bands during illiquid regimes

We implement a threshold-based regime classifier that labels spread observations as "tight" or "wide" based on a configurable threshold (e.g., the historical median spread). A logistic regression or decision tree can then predict the regime from features.

The regime classification problem uses the same features as spread prediction but maps them to a binary outcome:

$$P(\text{wide spread}_t | \mathbf{x}_t) = \sigma(\mathbf{w}^T \mathbf{x}_t + b)$$

where $\sigma$ is the sigmoid function.

## Applications

### Market Making Profitability

A market maker who quotes at the best bid and ask captures the spread on each round-trip trade. The expected profit per unit time is:

$$\Pi = \lambda \cdot S - \lambda_{informed} \cdot \alpha \cdot S - \text{inventory risk cost}$$

where $\lambda$ is the total trade arrival rate and $\lambda_{informed}$ is the informed trade arrival rate. ML spread prediction helps the market maker:

1. **Set optimal quotes**: Quote wider when the predicted spread (and thus adverse selection) is high
2. **Manage inventory**: Skew quotes toward reducing inventory when the inventory component is large
3. **Select instruments**: Focus on assets where the spread decomposition favors the order processing component

### Execution Cost Estimation

For institutional traders, accurate spread prediction is essential for:

1. **Pre-trade analysis**: Estimate the expected execution cost of a planned order
2. **Venue selection**: Route orders to the venue with the lowest predicted spread
3. **Algorithm selection**: Use aggressive algorithms when spreads are tight and passive algorithms when spreads are wide

The total transaction cost for an order of size $Q$ can be estimated as:

$$TC(Q) = \frac{S_{predicted}}{2} + \text{market impact}(Q) + \text{timing risk}$$

Our implementation provides a transaction cost estimator that combines spread prediction with simple market impact models.

### Cryptocurrency-Specific Considerations

Cryptocurrency markets have unique spread characteristics:

- **24/7 trading**: No opening/closing auctions; spread patterns follow global timezone cycles
- **Cross-exchange arbitrage**: Spreads on one exchange are influenced by prices on others
- **High volatility**: Spreads widen dramatically during large price moves
- **Fragmented liquidity**: Different stablecoin pairs (USDT, USDC, BUSD) for the same asset

Bybit provides both spot and perpetual futures orderbooks. Perpetual futures typically have tighter spreads due to higher liquidity and the funding rate mechanism.

## Rust Implementation

Our Rust implementation provides the following components:

### Core Spread Calculations
- `absolute_spread()`: Computes the raw bid-ask spread from best bid and ask prices
- `relative_spread()`: Normalizes the spread by the midpoint for cross-asset comparison
- `effective_spread()`: Computes the actual execution cost using trade prices and directions

### Roll Spread Estimator
- `roll_spread_estimator()`: Implements the Roll (1984) model using the autocovariance of price changes. Handles the positive-covariance edge case by returning zero.

### Linear Regression
- `LinearRegression` struct with `fit()` and `predict()` methods
- Uses the normal equation $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ for parameter estimation
- Supports multi-feature prediction with bias term

### Spread Regime Classifier
- `SpreadRegimeClassifier` with configurable threshold
- Labels observations as `Tight` or `Wide` regimes
- Computes regime transition probabilities

### Transaction Cost Estimator
- Combines spread prediction with market impact estimation
- Uses square-root impact model: $\text{impact} = k \cdot \sigma \cdot \sqrt{Q/V}$
- Provides total cost estimates for order planning

### Bybit Integration
- `fetch_bybit_orderbook()`: Fetches real-time orderbook data via the Bybit v5 API
- Parses bid and ask levels with prices and quantities
- Supports any trading pair (default: BTCUSDT)

All components include comprehensive unit tests verifying correctness.

## Bybit Data Integration

The implementation connects to Bybit's v5 REST API to fetch real-time orderbook data:

```
GET https://api.bybit.com/v5/market/orderbook?category=spot&symbol=BTCUSDT&limit=25
```

The response contains arrays of bid and ask levels, each with a price and quantity. We parse this into structured `OrderBookLevel` objects and compute spreads directly from the top-of-book quotes.

For time-series analysis, the example application polls the orderbook at regular intervals to build a spread history, then applies the regression model and regime classifier to the resulting dataset.

Key implementation details:
- Uses `reqwest` with blocking client for simplicity in examples
- Handles API errors gracefully with `anyhow` error types
- Supports configurable depth (number of orderbook levels)

## Key Takeaways

1. **The bid-ask spread is not a single number** but a composite of adverse selection, inventory risk, and order processing costs. Understanding the decomposition is essential for both market makers and execution traders.

2. **The Roll estimator** provides a simple, model-free estimate of the effective spread from price data alone, without requiring quote data. It is a valuable baseline for any spread analysis.

3. **ML models can predict future spreads** from observable features like volatility, volume, order book depth, and order imbalance. Even linear regression captures a significant portion of spread variation.

4. **Spread regime classification** identifies periods of tight versus wide spreads. This is directly actionable: market makers should be more cautious in wide-spread regimes, while execution algorithms should be more aggressive in tight-spread regimes.

5. **Transaction cost estimation** combines spread prediction with market impact models to provide pre-trade cost forecasts. This is critical for portfolio managers evaluating whether a trade's expected alpha exceeds its expected cost.

6. **Cryptocurrency markets** present unique challenges for spread modeling, including 24/7 trading, high volatility, and cross-exchange effects. Bybit's API provides the orderbook data needed to build real-time spread models.

7. **Rust implementation** provides the performance needed for real-time spread computation and prediction, with sub-microsecond spread calculations and efficient matrix operations for regression.

## References

- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *The Journal of Finance*, 39(4), 1127-1139.
- Glosten, L., & Milgrom, P. (1985). Bid, ask, and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71-100.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
- Huang, R., & Stoll, H. (1997). The components of the bid-ask spread: A general approach. *Review of Financial Studies*, 10(4), 995-1034.
- Hasbrouck, J. (2009). Trading costs and returns for US equities: Estimating effective costs from daily data. *The Journal of Finance*, 64(3), 1445-1477.
