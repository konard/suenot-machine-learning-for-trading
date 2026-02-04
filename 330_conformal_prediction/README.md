# Chapter 330: Conformal Prediction for Trading

## Overview

Conformal Prediction is a powerful framework for generating **distribution-free prediction intervals** with guaranteed coverage. Unlike traditional machine learning methods that output point predictions, conformal prediction provides uncertainty quantification that holds regardless of the underlying data distribution. This makes it exceptionally valuable for trading, where understanding the uncertainty around predictions is crucial for risk management and position sizing.

## Why Conformal Prediction for Trading?

### The Problem with Traditional Approaches

Traditional ML models for trading provide point predictions without reliable uncertainty estimates:

- **Overconfident predictions**: Neural networks often output high-confidence predictions even when wrong
- **Miscalibrated probabilities**: Predicted probabilities don't match actual frequencies
- **Distribution assumptions**: Many methods assume Gaussian errors, which rarely hold in financial markets
- **No coverage guarantees**: No theoretical guarantee that prediction intervals contain true values

### Conformal Prediction Solution

Conformal prediction provides:

```
Traditional: y_pred = f(X) → single point estimate

Conformal Prediction: [y_lower, y_upper] = CP(X)
where:
  P(y_true ∈ [y_lower, y_upper]) ≥ 1 - α

This guarantee holds for ANY data distribution!
```

## Technical Foundation

### 1. Core Concept: Non-Conformity Scores

The key idea is measuring how "strange" or "non-conforming" a new example is compared to training data:

```
Non-conformity score: A(x, y) measures how unusual (x, y) pair is

Common choices:
- Regression: A(x, y) = |y - f(x)| (absolute residual)
- Classification: A(x, y) = 1 - p(y|x) (one minus probability)
- Quantile: A(x, y) = max(q_low(x) - y, y - q_high(x))
```

### 2. Coverage Guarantee Theorem

For exchangeable data and miscoverage level α:

```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α

where C(X) is the conformal prediction set/interval
```

This guarantee is **marginal** (averaged over all possible test points) and **distribution-free**.

## Types of Conformal Prediction

### 1. Full Conformal Prediction

The original method - computationally expensive but optimal:

```
Algorithm:
1. For each possible label y:
   a. Add (x_new, y) to calibration set
   b. Compute all non-conformity scores
   c. Compute p-value: proportion of scores ≥ score of (x_new, y)
2. Return all y with p-value > α

Complexity: O(n) model retrainings per prediction
```

### 2. Split Conformal Prediction (Inductive)

Practical approach used in production:

```
Algorithm:
1. Split data into training and calibration sets
2. Train model on training set
3. Compute non-conformity scores on calibration set
4. Find quantile q = (1-α)(1 + 1/n_cal) of calibration scores
5. For new x: C(x) = {y : A(x, y) ≤ q}

Complexity: O(1) per prediction after initial setup
```

### 3. Conformal Quantile Regression (CQR)

Combines quantile regression with conformal calibration:

```python
# Train two quantile regressors
q_low = QuantileRegressor(quantile=α/2)
q_high = QuantileRegressor(quantile=1-α/2)

# Calibration: compute conformity scores
scores = max(q_low(X_cal) - Y_cal, Y_cal - q_high(X_cal))

# Find quantile of scores
Q = quantile(scores, (1-α)(1 + 1/n_cal))

# Prediction interval for new x
interval = [q_low(x) - Q, q_high(x) + Q]
```

## Model Architecture for Trading

```
┌─────────────────────────────────────────────────────────────────┐
│                 CONFORMAL PREDICTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA LAYER                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Market Data (OHLCV, Volume, Order Book)                   │   │
│  │   - Price returns (multiple timeframes)                   │   │
│  │   - Volume profile and momentum                           │   │
│  │   - Technical indicators (RSI, MACD, Bollinger)           │   │
│  │   - Volatility measures (ATR, realized vol)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  FEATURE ENGINEERING                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Time-series features, lagged values, rolling statistics   │   │
│  │ Cross-asset correlations, regime indicators               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  BASE MODEL TRAINING                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Point Predictor: Neural Network / Gradient Boosting │   │   │
│  │ │   - LSTM, Transformer, or ensemble methods          │   │   │
│  │ │   - Predicts E[Y|X] or quantiles                    │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  CONFORMAL CALIBRATION                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Non-conformity Score Calculation                    │   │   │
│  │ │   - Compute scores on held-out calibration set      │   │   │
│  │ │   - Adaptive scores for heteroscedastic data        │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Quantile Estimation                                 │   │   │
│  │ │   - Find (1-α) quantile of calibration scores       │   │   │
│  │ │   - Store for inference time                        │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  PREDICTION INTERVAL GENERATION                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ For new observation x:                                    │   │
│  │   - Point prediction: ŷ = f(x)                           │   │
│  │   - Lower bound: ŷ - Q                                   │   │
│  │   - Upper bound: ŷ + Q                                   │   │
│  │   - Interval width reflects uncertainty                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  TRADING DECISION                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Position sizing based on:                                 │   │
│  │   - Point prediction direction/magnitude                  │   │
│  │   - Interval width (uncertainty)                          │   │
│  │   - Coverage level confidence                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Adaptive Conformal Prediction for Time Series

Financial time series are non-stationary - standard conformal prediction assumes exchangeability. We use **Adaptive Conformal Inference (ACI)** to handle this:

### Sliding Window Conformal

```python
def adaptive_conformal(data, model, window_size, alpha):
    """
    Recalibrate conformal predictor using sliding window
    """
    predictions = []
    coverages = []

    for t in range(window_size, len(data)):
        # Use recent window for calibration
        cal_window = data[t-window_size:t]

        # Compute non-conformity scores on recent data
        scores = compute_scores(model, cal_window)

        # Find adaptive quantile
        q = np.quantile(scores, (1-alpha) * (1 + 1/window_size))

        # Make prediction with calibrated interval
        pred = model.predict(data[t].X)
        interval = [pred - q, pred + q]

        predictions.append(interval)
        coverages.append(data[t].y in interval)

    return predictions, np.mean(coverages)
```

### ACI with Online Updates

```python
class AdaptiveConformalInference:
    """
    Online conformal prediction with coverage tracking
    """
    def __init__(self, alpha, gamma=0.01):
        self.alpha_t = alpha  # Time-varying miscoverage level
        self.gamma = gamma     # Learning rate
        self.target_alpha = alpha

    def update(self, covered: bool):
        """Update alpha based on realized coverage"""
        # If covered, decrease alpha (tighter intervals)
        # If not covered, increase alpha (wider intervals)
        error = (1 - int(covered)) - self.target_alpha
        self.alpha_t = np.clip(
            self.alpha_t + self.gamma * error,
            0.01, 0.5
        )

    def get_quantile_level(self):
        return 1 - self.alpha_t
```

## Non-Conformity Score Functions

### 1. Absolute Residual (Basic)

```python
def absolute_residual(y_true, y_pred):
    return np.abs(y_true - y_pred)
```

### 2. Normalized Residual (Heteroscedastic)

```python
def normalized_residual(y_true, y_pred, sigma_pred):
    """For models that predict both mean and variance"""
    return np.abs(y_true - y_pred) / sigma_pred
```

### 3. CQR Score (Quantile-based)

```python
def cqr_score(y_true, q_low, q_high):
    """Conformalized Quantile Regression score"""
    return np.maximum(q_low - y_true, y_true - q_high)
```

### 4. MAPIE Score (Adaptive)

```python
def mapie_score(y_true, y_pred, residuals_std):
    """Locally-weighted score based on residual MAD"""
    return np.abs(y_true - y_pred) / residuals_std
```

## Trading Strategy with Conformal Prediction

### Position Sizing Based on Uncertainty

```python
def calculate_position_size(
    prediction_interval,
    point_prediction,
    max_position,
    risk_budget
):
    """
    Size positions inversely proportional to uncertainty
    """
    interval_width = prediction_interval[1] - prediction_interval[0]

    # Normalize by typical volatility
    normalized_width = interval_width / historical_volatility

    # Higher confidence (narrower interval) = larger position
    confidence_factor = 1.0 / (1.0 + normalized_width)

    # Expected return based on point prediction
    expected_return = point_prediction

    # Kelly-inspired sizing with uncertainty adjustment
    if expected_return > 0:  # Long signal
        position = max_position * confidence_factor * (expected_return / risk_budget)
    elif expected_return < 0:  # Short signal
        position = -max_position * confidence_factor * (abs(expected_return) / risk_budget)
    else:
        position = 0

    return np.clip(position, -max_position, max_position)
```

### Signal Generation

```python
def generate_trading_signals(
    model,
    conformal_predictor,
    market_data,
    alpha=0.1
):
    """Generate signals with uncertainty-aware position sizing"""

    signals = []

    for observation in market_data:
        # Get point prediction and conformal interval
        point_pred = model.predict(observation.features)
        interval = conformal_predictor.predict_interval(
            observation.features,
            alpha=alpha
        )

        interval_width = interval[1] - interval[0]

        # Only trade when confident (narrow interval)
        if interval_width < volatility_threshold:
            if interval[0] > 0:  # Entire interval positive
                signal = Signal(
                    direction="LONG",
                    confidence=1.0 - (interval_width / max_width),
                    expected_return=point_pred,
                    interval=interval
                )
            elif interval[1] < 0:  # Entire interval negative
                signal = Signal(
                    direction="SHORT",
                    confidence=1.0 - (interval_width / max_width),
                    expected_return=point_pred,
                    interval=interval
                )
            else:
                signal = Signal(direction="HOLD", confidence=0)
        else:
            signal = Signal(direction="HOLD", confidence=0)

        signals.append(signal)

    return signals
```

## Key Metrics

### Coverage Metrics

- **Empirical Coverage**: Fraction of true values in prediction intervals
- **Conditional Coverage**: Coverage conditioned on features
- **Interval Width**: Average width of prediction intervals
- **Coverage Gap**: |Empirical Coverage - Target Coverage|

### Trading Metrics

- **Sharpe Ratio**: Risk-adjusted returns (target > 2.0)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Coverage-Adjusted Sharpe**: Sharpe accounting for interval reliability

## Advantages of Conformal Prediction for Trading

| Aspect | Traditional ML | Conformal Prediction |
|--------|---------------|---------------------|
| Uncertainty quantification | Heuristic or none | Guaranteed coverage |
| Distribution assumptions | Often Gaussian | Distribution-free |
| Calibration | Requires separate validation | Built-in calibration |
| Position sizing | Subjective | Principled uncertainty-based |
| Risk management | Ad-hoc | Coverage guarantees |
| Adaptivity | Static | Online adaptation possible |

## Comparison with Other Uncertainty Methods

### vs. Bayesian Neural Networks

- **BNN**: Requires prior specification, computationally expensive
- **CP**: Distribution-free, finite-sample guarantee

### vs. Monte Carlo Dropout

- **MC Dropout**: Approximate, no coverage guarantee
- **CP**: Exact coverage guarantee

### vs. Ensemble Methods

- **Ensembles**: Uncertainty from disagreement, uncalibrated
- **CP**: Calibrated intervals with provable coverage

### vs. Quantile Regression

- **QR**: Asymptotically valid, miscalibrated in finite samples
- **CQR**: Conformalized QR with finite-sample guarantees

## Implementation Details

### Data Requirements

```
Market Data Pipeline:
├── OHLCV data (1-minute to daily resolution)
│   └── Multiple assets for portfolio strategies
├── Technical indicators computed from OHLCV
│   └── Momentum, volatility, trend indicators
├── Volume and order flow metrics
│   └── Volume profile, trade imbalance
└── Regime indicators (optional)
    └── Volatility regime, trend regime

Data Split for Conformal Prediction:
├── Training set: 60% (model training)
├── Calibration set: 20% (conformal calibration)
└── Test set: 20% (final evaluation)
```

### Feature Engineering

```python
features = {
    # Price-based features
    'returns_1m': log_return(close, 1),
    'returns_5m': log_return(close, 5),
    'returns_15m': log_return(close, 15),
    'returns_1h': log_return(close, 60),

    # Volatility features
    'volatility_1h': rolling_std(returns, 60),
    'atr_14': average_true_range(high, low, close, 14),
    'realized_vol': realized_volatility(returns, 20),

    # Momentum indicators
    'rsi_14': rsi(close, 14),
    'macd': macd(close, 12, 26, 9),
    'momentum_10': momentum(close, 10),

    # Volume features
    'volume_ratio': volume / volume_ma_20,
    'obv': on_balance_volume(close, volume),

    # Trend indicators
    'sma_cross': sma(close, 10) - sma(close, 30),
    'ema_trend': ema(close, 20) - ema(close, 50),
}
```

### Configuration

```yaml
model:
  base_predictor: "gradient_boosting"  # or "neural_network"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1

conformal:
  method: "split"  # "split", "cv", or "jackknife"
  alpha: 0.1       # Target miscoverage rate (10%)
  calibration_size: 0.2
  adaptive: true
  window_size: 100  # For adaptive conformal

trading:
  max_position: 1.0
  risk_budget: 0.02  # 2% risk per trade
  min_confidence: 0.6
  rebalance_frequency: "1H"

data:
  train_split: 0.6
  calibration_split: 0.2
  test_split: 0.2
  lookback_window: 60
  prediction_horizon: 5  # 5 periods ahead
```

## Production Considerations

```
Inference Pipeline:
├── Data Collection (Exchange WebSocket/REST)
│   └── Real-time OHLCV updates
├── Feature Computation
│   └── Rolling statistics, indicators
├── Model Prediction
│   └── Point prediction from base model
├── Conformal Interval
│   └── Apply calibrated quantile
├── Adaptive Update
│   └── Track coverage, adjust alpha
└── Signal Generation
    └── Position sizing with uncertainty

Latency Budget:
├── Data collection: ~10ms
├── Feature computation: ~5ms
├── Model inference: ~10ms
├── Conformal calibration: ~1ms
├── Signal generation: ~1ms
└── Total: ~30ms (excluding execution)

Recalibration Schedule:
├── Full recalibration: Daily (end of day)
├── Adaptive alpha update: Per prediction
├── Model retraining: Weekly/Monthly
└── Feature review: Quarterly
```

## Directory Structure

```
330_conformal_prediction/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt         # Python dependencies
│   ├── conformal_predictor.py   # Core conformal prediction
│   ├── data_fetcher.py          # CCXT data fetching (Bybit)
│   ├── features.py              # Feature engineering
│   ├── models.py                # Base predictors
│   ├── trading_strategy.py      # Trading logic
│   ├── backtest.py              # Backtesting engine
│   └── main.py                  # Example usage
└── rust_conformal/              # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Library entry point
    │   ├── api/                 # Bybit API client
    │   │   ├── mod.rs
    │   │   ├── client.rs        # HTTP client for Bybit
    │   │   └── types.rs         # API response types
    │   ├── conformal/           # Conformal prediction core
    │   │   ├── mod.rs
    │   │   ├── predictor.rs     # Split conformal predictor
    │   │   ├── scores.rs        # Non-conformity scores
    │   │   └── adaptive.rs      # Adaptive conformal inference
    │   ├── features/            # Feature engineering
    │   │   ├── mod.rs
    │   │   ├── engine.rs        # Feature computation
    │   │   └── indicators.rs    # Technical indicators
    │   ├── strategy/            # Trading strategy
    │   │   ├── mod.rs
    │   │   └── signal.rs        # Signal generation
    │   └── backtest/            # Backtesting engine
    │       ├── mod.rs
    │       ├── engine.rs        # Backtest execution
    │       └── metrics.rs       # Performance metrics
    └── examples/
        ├── fetch_data.rs        # Data fetching example
        ├── conformal_intervals.rs  # Interval generation
        ├── trading_signals.rs   # Signal generation
        └── backtest.rs          # Full backtest example
```

## References

1. **A Tutorial on Conformal Prediction** (Shafer & Vovk, 2008)
   - https://jmlr.org/papers/v9/shafer08a.html

2. **Conformal Prediction Under Covariate Shift** (Tibshirani et al., 2019)
   - https://arxiv.org/abs/1904.06019

3. **Conformalized Quantile Regression** (Romano et al., 2019)
   - https://arxiv.org/abs/1905.03222

4. **Adaptive Conformal Inference Under Distribution Shift** (Gibbs & Candes, 2021)
   - https://arxiv.org/abs/2106.00170

5. **Conformal Prediction: A Gentle Introduction** (Angelopoulos & Bates, 2021)
   - https://arxiv.org/abs/2107.07511

6. **MAPIE: Model Agnostic Prediction Interval Estimator**
   - https://github.com/scikit-learn-contrib/MAPIE

## Difficulty Level

**Intermediate to Advanced** - Requires understanding of:
- Statistical inference and confidence intervals
- Machine learning fundamentals
- Time series analysis
- Python/Rust programming
- Financial risk management concepts

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results.
