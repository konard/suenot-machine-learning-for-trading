# Chapter 39: Conformal Prediction — Trading with Calibrated Uncertainty

## Overview

Conformal Prediction (CP) is a powerful framework for uncertainty quantification that provides **calibrated prediction intervals with guaranteed coverage**. Unlike standard machine learning models that often produce overconfident predictions, conformal prediction gives honest estimates of uncertainty that hold under minimal assumptions.

The key insight for trading: **trade only when the model is confident, and size positions inversely proportional to uncertainty**. This approach naturally avoids trades during high-uncertainty regimes (market stress, regime changes) and concentrates capital when predictions are reliable.

### Why Conformal Prediction for Trading?

1. **Guaranteed Coverage**: If you target 90% coverage, approximately 90% of your prediction intervals will contain the true value
2. **Distribution-Free**: Works with any underlying model without parametric assumptions
3. **Finite-Sample Valid**: Guarantees hold for any sample size, not just asymptotically
4. **Adaptive**: Intervals naturally widen during volatile periods and narrow during stable periods
5. **Model-Agnostic**: Wrap any ML model (neural networks, gradient boosting, etc.) with conformal prediction

## Contents

1. [Theoretical Foundations](#theoretical-foundations)
    * [Exchangeability and Coverage Guarantees](#exchangeability-and-coverage-guarantees)
    * [Nonconformity Scores](#nonconformity-scores)
2. [Conformal Prediction Methods](#conformal-prediction-methods)
    * [Split Conformal Prediction](#split-conformal-prediction)
    * [Conformalized Quantile Regression (CQR)](#conformalized-quantile-regression-cqr)
    * [Adaptive Conformal Inference for Time Series](#adaptive-conformal-inference-for-time-series)
3. [Trading Strategy Design](#trading-strategy-design)
    * [Signal Generation with Uncertainty](#signal-generation-with-uncertainty)
    * [Position Sizing with Calibrated Confidence](#position-sizing-with-calibrated-confidence)
    * [Kelly Criterion with Prediction Intervals](#kelly-criterion-with-prediction-intervals)
4. [Implementation](#implementation)
    * [Code Examples](#code-examples)
    * [Notebooks](#notebooks)
5. [Backtesting and Evaluation](#backtesting-and-evaluation)
6. [Resources and References](#resources-and-references)

---

## Theoretical Foundations

### Exchangeability and Coverage Guarantees

Conformal prediction relies on the assumption of **exchangeability**: the joint distribution of data points is invariant to permutations. This is weaker than the i.i.d. assumption and allows for dependent data under certain conditions.

**Key Theorem (Vovk et al., 2005)**: For exchangeable data $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$, a conformal prediction set $C(X_{n+1})$ constructed at level $1-\alpha$ satisfies:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This guarantee is **marginal** (averaged over all test points) and holds exactly in finite samples.

### Nonconformity Scores

The core of conformal prediction is the **nonconformity score** — a function measuring how "unusual" a data point is relative to others. Common choices:

- **Absolute residual**: $s(x, y) = |y - \hat{f}(x)|$
- **Normalized residual**: $s(x, y) = \frac{|y - \hat{f}(x)|}{\hat{\sigma}(x)}$
- **Quantile-based**: $s(x, y) = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$

The choice of score function affects the shape and adaptivity of prediction intervals.

---

## Conformal Prediction Methods

### Split Conformal Prediction

The simplest and most practical method:

1. **Split** data into training set and calibration set
2. **Train** underlying model on training set
3. **Calibrate** by computing nonconformity scores on calibration set
4. **Predict** by finding the $1-\alpha$ quantile of calibration scores

```python
import numpy as np
from sklearn.model_selection import train_test_split

class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression with guaranteed coverage.

    Coverage guarantee: P(Y ∈ [lower, upper]) ≥ 1 - alpha
    """
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha  # Miscoverage rate (1 - alpha = coverage)
        self.calibration_scores = None
        self.q_hat = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Train model and calibrate on held-out calibration set.

        Parameters:
        -----------
        X_train : array-like, Training features
        y_train : array-like, Training targets
        X_calib : array-like, Calibration features (held out from training)
        y_calib : array-like, Calibration targets
        """
        # Step 1: Train underlying model
        self.model.fit(X_train, y_train)

        # Step 2: Get predictions on calibration set
        y_pred_calib = self.model.predict(X_calib)

        # Step 3: Compute nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_calib - y_pred_calib)

        # Step 4: Compute quantile for prediction intervals
        # The (1-alpha)(1 + 1/n) quantile ensures finite-sample coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1
        self.q_hat = np.quantile(self.calibration_scores, q_level)

        return self

    def predict(self, X):
        """
        Return point prediction and prediction interval.

        Returns:
        --------
        dict with keys: 'prediction', 'lower', 'upper', 'interval_width'
        """
        y_pred = self.model.predict(X)

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        return {
            'prediction': y_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': np.full_like(y_pred, 2 * self.q_hat)
        }

    def coverage(self, X_test, y_test):
        """Compute empirical coverage on test set."""
        pred = self.predict(X_test)
        covered = (y_test >= pred['lower']) & (y_test <= pred['upper'])
        return covered.mean()
```

### Conformalized Quantile Regression (CQR)

CQR produces **adaptive intervals** that vary in width based on input features. This is crucial for financial data where uncertainty varies significantly across market regimes.

```python
from sklearn.ensemble import GradientBoostingRegressor

class ConformizedQuantileRegression:
    """
    CQR: Conformalized Quantile Regression

    Produces heteroscedastic intervals that adapt to local uncertainty.
    More informative than split conformal for financial data.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        # Fit quantile models for lower and upper bounds
        self.lower_model = GradientBoostingRegressor(
            loss='quantile', alpha=alpha/2, n_estimators=100
        )
        self.upper_model = GradientBoostingRegressor(
            loss='quantile', alpha=1-alpha/2, n_estimators=100
        )
        self.q_hat = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        """Fit quantile models and calibrate."""
        # Fit quantile models
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        # Get initial intervals on calibration set
        lower_calib = self.lower_model.predict(X_calib)
        upper_calib = self.upper_model.predict(X_calib)

        # Compute conformity scores
        # Score = how much interval needs to expand to cover true value
        scores = np.maximum(
            lower_calib - y_calib,  # Lower bound too high
            y_calib - upper_calib   # Upper bound too low
        )

        # Quantile for guaranteed coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(scores, q_level)

        return self

    def predict(self, X):
        """Return adaptive prediction intervals."""
        lower = self.lower_model.predict(X) - self.q_hat
        upper = self.upper_model.predict(X) + self.q_hat

        return {
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'midpoint': (upper + lower) / 2
        }
```

### Adaptive Conformal Inference for Time Series

Standard conformal prediction assumes exchangeability, which is violated in time series. **Adaptive Conformal Inference (ACI)** addresses this by dynamically adjusting the coverage level based on recent performance.

```python
class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference (ACI) for time series.

    Dynamically adjusts coverage level based on recent errors,
    maintaining approximate coverage under distribution shift.

    Reference: Gibbs & Candès (2021) "Adaptive Conformal Inference
               Under Distribution Shift"
    """
    def __init__(self, model, target_coverage=0.9, gamma=0.05):
        self.model = model
        self.target_coverage = target_coverage
        self.gamma = gamma  # Learning rate for adaptation
        self.alpha_t = 1 - target_coverage  # Current miscoverage rate
        self.history = []  # Track coverage over time

    def update(self, y_true, lower, upper):
        """
        Update alpha based on whether y_true was covered.

        This implements online learning of the coverage level:
        - If covered more than target: decrease intervals (lower alpha)
        - If covered less than target: increase intervals (higher alpha)
        """
        covered = (lower <= y_true) and (y_true <= upper)
        self.history.append(covered)

        # Gradient update: move alpha toward achieving target coverage
        if covered:
            # Covered -> can afford narrower intervals
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha_t - 0)
        else:
            # Not covered -> need wider intervals
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha_t - 1)

        # Clip to valid range
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.5)

        return covered

    def predict(self, X, calibration_scores):
        """Generate prediction with adaptive interval."""
        y_pred = self.model.predict(X)

        # Compute interval width based on current alpha
        q_level = 1 - self.alpha_t
        q_hat = np.quantile(calibration_scores, min(q_level, 1.0))

        return {
            'prediction': y_pred,
            'lower': y_pred - q_hat,
            'upper': y_pred + q_hat,
            'interval_width': 2 * q_hat,
            'current_alpha': self.alpha_t,
            'recent_coverage': np.mean(self.history[-100:]) if self.history else None
        }
```

---

## Trading Strategy Design

### Signal Generation with Uncertainty

The core idea: **only trade when the prediction interval is narrow (high confidence) AND the direction is clear**.

```python
class ConformalTradingStrategy:
    """
    Trading strategy using calibrated prediction intervals.

    Key principles:
    1. Trade only when interval is narrow (high confidence)
    2. Direction must be clear (interval doesn't cross zero, or is far from it)
    3. Position size inversely proportional to uncertainty
    """
    def __init__(self, predictor, width_threshold=0.02, min_edge=0.005):
        """
        Parameters:
        -----------
        predictor : Conformal predictor with predict() method
        width_threshold : Maximum interval width to take a trade (e.g., 2% for returns)
        min_edge : Minimum expected edge to trade (e.g., 0.5% expected return)
        """
        self.predictor = predictor
        self.width_threshold = width_threshold
        self.min_edge = min_edge

    def generate_signal(self, X):
        """
        Generate trading signal based on prediction interval.

        Returns:
        --------
        dict with: prediction, interval_width, confidence, trade, direction, size
        """
        pred = self.predictor.predict(X)

        # Handle both scalar and array inputs
        if hasattr(pred['interval_width'], '__len__'):
            interval_width = pred['interval_width'][0]
            lower = pred['lower'][0]
            upper = pred['upper'][0]
            midpoint = pred.get('midpoint', (lower + upper) / 2)
            if hasattr(midpoint, '__len__'):
                midpoint = midpoint[0]
        else:
            interval_width = pred['interval_width']
            lower = pred['lower']
            upper = pred['upper']
            midpoint = pred.get('midpoint', (lower + upper) / 2)

        signal = {
            'prediction': midpoint,
            'interval_width': interval_width,
            'lower': lower,
            'upper': upper,
            'confidence': 1 / (1 + interval_width * 10),  # Transform to 0-1 scale
            'trade': False,
            'direction': 0,
            'size': 0.0
        }

        # Condition 1: Interval must be narrow enough
        if interval_width >= self.width_threshold:
            signal['skip_reason'] = 'interval_too_wide'
            return signal

        # Condition 2: Direction must be clear with sufficient edge
        if lower > self.min_edge:
            # Entire interval is positive with sufficient magnitude
            signal['direction'] = 1  # Long
            signal['trade'] = True
            signal['edge'] = lower  # Worst-case expected return
        elif upper < -self.min_edge:
            # Entire interval is negative with sufficient magnitude
            signal['direction'] = -1  # Short
            signal['trade'] = True
            signal['edge'] = -upper  # Worst-case expected return
        else:
            signal['skip_reason'] = 'unclear_direction'
            return signal

        # Position size inversely proportional to interval width
        # Narrower interval -> higher confidence -> larger position
        signal['size'] = self._compute_size(interval_width, signal['edge'])

        return signal

    def _compute_size(self, interval_width, edge):
        """
        Compute position size based on uncertainty and edge.

        Uses a simplified approach: size = edge / interval_width
        Capped at 1.0 (100% of capital)
        """
        if interval_width <= 0:
            return 0.0

        # Size proportional to edge/uncertainty ratio
        raw_size = edge / interval_width

        # Apply caps
        size = min(raw_size, 1.0)
        size = max(size, 0.0)

        return size
```

### Position Sizing with Calibrated Confidence

```python
def kelly_with_conformal(prediction, lower, upper, risk_free_rate=0):
    """
    Kelly criterion adapted for conformal prediction intervals.

    Key insight: The interval width provides a calibrated estimate of
    uncertainty that can be used to adjust the Kelly fraction.

    Parameters:
    -----------
    prediction : Point prediction (expected return)
    lower : Lower bound of prediction interval
    upper : Upper bound of prediction interval
    risk_free_rate : Risk-free rate for excess return calculation

    Returns:
    --------
    kelly_fraction : Recommended position size as fraction of capital
    """
    interval_width = upper - lower
    expected_excess = prediction - risk_free_rate

    # Edge case: no expected edge
    if expected_excess <= 0:
        return 0.0

    # Edge case: degenerate interval
    if interval_width <= 0:
        return 0.0

    # The interval width serves as a volatility proxy
    # Kelly fraction = expected_return / variance
    # We use interval_width as proxy for standard deviation
    implied_variance = (interval_width / 2) ** 2  # Half-width as std proxy

    kelly_fraction = expected_excess / implied_variance

    # Apply half-Kelly for safety (common practice)
    kelly_fraction = kelly_fraction / 2

    # Cap at reasonable levels
    kelly_fraction = min(kelly_fraction, 2.0)  # Max 200% (with leverage)
    kelly_fraction = max(kelly_fraction, -2.0)  # Max -200% (short)

    return kelly_fraction


class ConfidenceBasedSizing:
    """
    Position sizing based on prediction interval confidence.

    Maps interval width to position size using various schemes.
    """
    def __init__(self, method='inverse', max_size=1.0, min_size=0.0):
        self.method = method
        self.max_size = max_size
        self.min_size = min_size

    def compute_size(self, interval_width, baseline_width=None):
        """
        Compute position size based on interval width.

        Parameters:
        -----------
        interval_width : Current prediction interval width
        baseline_width : Reference width for normalization
        """
        if baseline_width is None:
            baseline_width = interval_width

        if self.method == 'inverse':
            # Size = baseline / current (narrower = larger)
            size = baseline_width / max(interval_width, 1e-6)

        elif self.method == 'linear':
            # Linear decrease: size = 1 - width/baseline
            size = 1 - interval_width / max(baseline_width, 1e-6)

        elif self.method == 'exponential':
            # Exponential decay based on width
            size = np.exp(-interval_width / baseline_width)

        elif self.method == 'threshold':
            # Binary: full size if below threshold, zero otherwise
            size = self.max_size if interval_width < baseline_width else 0

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Clip to bounds
        return np.clip(size, self.min_size, self.max_size)
```

---

## Implementation

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_conformal_theory.ipynb` | Theory: exchangeability, coverage guarantees, nonconformity scores |
| 2 | `02_split_conformal.ipynb` | Split conformal prediction implementation and analysis |
| 3 | `03_conformalized_quantile.ipynb` | Conformalized Quantile Regression for adaptive intervals |
| 4 | `04_adaptive_conformal.ipynb` | Adaptive Conformal Inference for time series |
| 5 | `05_financial_application.ipynb` | Application to return prediction with real market data |
| 6 | `06_interval_analysis.ipynb` | Analysis of interval width patterns and market regimes |
| 7 | `07_trading_rules.ipynb` | Trading rules based on prediction intervals |
| 8 | `08_position_sizing.ipynb` | Kelly-like sizing with calibrated uncertainty |
| 9 | `09_backtesting.ipynb` | Full backtest of conformal trading strategy |
| 10 | `10_comparison.ipynb` | Comparison vs. standard ML without uncertainty quantification |

### Code Examples

See the `rust_examples/` directory for production-ready Rust implementations featuring:

- **Bybit API client** for real-time cryptocurrency data
- **Modular conformal prediction algorithms** (Split CP, CQR, ACI)
- **Trading strategy framework** with interval-based signals
- **Backtesting engine** with proper time-series handling

---

## Backtesting and Evaluation

### Key Metrics

**Coverage Metrics:**
- **Empirical Coverage**: Fraction of true values within prediction intervals
- **Conditional Coverage**: Coverage stratified by interval width, volatility regime, etc.
- **Coverage Stability**: How consistent is coverage over time?

**Interval Quality:**
- **Average Width**: Mean prediction interval width
- **Width Variability**: Standard deviation of interval widths
- **Sharpness**: Inverse of average width (narrower is sharper)
- **Winkler Score**: Combined measure of coverage and sharpness

**Trading Performance:**
- **Sharpe Ratio**: Risk-adjusted return
- **Win Rate**: Fraction of profitable trades
- **Average Trade**: Mean return per trade
- **Trade Selectivity**: Fraction of periods with a trade signal

```python
def evaluate_conformal_strategy(results_df):
    """
    Comprehensive evaluation of conformal trading strategy.

    Parameters:
    -----------
    results_df : DataFrame with columns:
        - prediction, lower, upper, actual, direction, size, pnl
    """
    metrics = {}

    # Coverage metrics
    covered = (results_df['actual'] >= results_df['lower']) & \
              (results_df['actual'] <= results_df['upper'])
    metrics['coverage'] = covered.mean()

    # Interval metrics
    widths = results_df['upper'] - results_df['lower']
    metrics['avg_width'] = widths.mean()
    metrics['width_std'] = widths.std()
    metrics['sharpness'] = 1 / widths.mean()

    # Trading metrics (only for actual trades)
    trades = results_df[results_df['direction'] != 0]
    if len(trades) > 0:
        metrics['n_trades'] = len(trades)
        metrics['trade_frequency'] = len(trades) / len(results_df)
        metrics['avg_pnl'] = trades['pnl'].mean()
        metrics['sharpe'] = trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252)
        metrics['win_rate'] = (trades['pnl'] > 0).mean()
        metrics['total_return'] = trades['pnl'].sum()

        # Conditional coverage for trades
        trades_covered = covered[results_df['direction'] != 0]
        metrics['coverage_on_trades'] = trades_covered.mean()

    return metrics
```

---

## Resources and References

### Academic Papers

- **Vovk, Gammerman, Shafer (2005)**: "Algorithmic Learning in a Random World" — The foundational textbook on conformal prediction
- **Romano, Patterson, Candès (2019)**: "Conformalized Quantile Regression" — CQR for adaptive intervals
- **Gibbs & Candès (2021)**: "Adaptive Conformal Inference Under Distribution Shift" — ACI for non-exchangeable data
- **Barber et al. (2022)**: "Conformal Prediction Beyond Exchangeability" — Extensions for dependent data

### Software Libraries

- [MAPIE](https://mapie.readthedocs.io/): Comprehensive conformal prediction library for Python
- [Crepes](https://github.com/henrikbostrom/crepes): Conformal Regressors and Predictive Systems
- [ConformalPrediction.jl](https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl): Julia implementation

### Tutorials and Courses

- [A Tutorial on Conformal Prediction](https://www.jmlr.org/papers/v9/shafer08a.html) — Shafer & Vovk, JMLR 2008
- [Conformal Prediction in 2020](https://arxiv.org/abs/2107.07511) — Recent survey paper

---

## Difficulty Level

**Intermediate** (3/5)

### Prerequisites

- Statistical inference and hypothesis testing
- Prediction intervals vs. confidence intervals
- Quantile regression basics
- Time series analysis fundamentals
- Risk management and position sizing

### Learning Outcomes

After completing this chapter, you will be able to:

1. Implement conformal prediction for return forecasting
2. Construct prediction intervals with guaranteed coverage
3. Design trading strategies that leverage uncertainty quantification
4. Size positions based on calibrated confidence
5. Evaluate strategies using coverage and performance metrics
6. Adapt conformal methods for non-stationary financial time series
