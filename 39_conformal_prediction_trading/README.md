# Chapter 39: Conformal Prediction — Trading with Calibrated Uncertainty

## Overview

Conformal Prediction предоставляет calibrated prediction intervals с гарантированным coverage. В отличие от обычных ML моделей, которые могут быть overconfident, conformal prediction дает честные оценки неопределенности.

## Trading Strategy

**Суть стратегии:** Торгуем только когда prediction interval узкий (высокая уверенность). Position sizing обратно пропорционален ширине интервала.

**Сигнал на вход:**
- Trade: Prediction interval width < threshold И direction clear
- Size: Inversely proportional to interval width
- Skip: Wide interval (высокая неопределенность)

**Edge:** Избегаем сделок когда модель не уверена; размер позиции отражает confidence

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_conformal_theory.ipynb` | Теория: exchangeability, coverage guarantees |
| 2 | `02_split_conformal.ipynb` | Split conformal prediction |
| 3 | `03_conformalized_quantile.ipynb` | Conformalized Quantile Regression |
| 4 | `04_adaptive_conformal.ipynb` | Adaptive Conformal для time series |
| 5 | `05_financial_application.ipynb` | Применение к return prediction |
| 6 | `06_interval_analysis.ipynb` | Анализ ширины интервалов |
| 7 | `07_trading_rules.ipynb` | Правила торговли на основе intervals |
| 8 | `08_position_sizing.ipynb` | Kelly-like sizing с calibrated uncertainty |
| 9 | `09_backtesting.ipynb` | Backtest стратегии |
| 10 | `10_comparison.ipynb` | vs standard ML без uncertainty |

### Conformal Prediction Basics

```python
import numpy as np
from sklearn.model_selection import train_test_split

class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression
    """
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha  # Miscoverage rate (1 - alpha = coverage)
        self.calibration_scores = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Train model and calibrate on held-out calibration set
        """
        # Train underlying model
        self.model.fit(X_train, y_train)

        # Get predictions on calibration set
        y_pred_calib = self.model.predict(X_calib)

        # Compute nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_calib - y_pred_calib)

        # Compute quantile for prediction intervals
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(self.calibration_scores, q_level)

    def predict(self, X):
        """
        Return point prediction and prediction interval
        """
        y_pred = self.model.predict(X)

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        return {
            'prediction': y_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': 2 * self.q_hat
        }
```

### Conformalized Quantile Regression

```python
from sklearn.ensemble import GradientBoostingRegressor

class ConformizedQuantileRegression:
    """
    CQR: Conformalized Quantile Regression
    More adaptive intervals than split conformal
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.lower_model = GradientBoostingRegressor(loss='quantile', alpha=alpha/2)
        self.upper_model = GradientBoostingRegressor(loss='quantile', alpha=1-alpha/2)

    def fit(self, X_train, y_train, X_calib, y_calib):
        # Fit quantile models
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        # Get initial intervals on calibration set
        lower_calib = self.lower_model.predict(X_calib)
        upper_calib = self.upper_model.predict(X_calib)

        # Compute conformity scores
        # Score = max(lower - y, y - upper) (how much interval needs to expand)
        scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)

        # Quantile for coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level)

    def predict(self, X):
        lower = self.lower_model.predict(X) - self.q_hat
        upper = self.upper_model.predict(X) + self.q_hat

        return {
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'midpoint': (upper + lower) / 2
        }
```

### Adaptive Conformal for Time Series

```python
class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference (ACI) for time series
    Adjusts to changing conditions over time
    """
    def __init__(self, model, target_coverage=0.9, gamma=0.1):
        self.model = model
        self.target_coverage = target_coverage
        self.gamma = gamma  # Learning rate for adaptation
        self.alpha_t = 1 - target_coverage  # Current miscoverage rate

    def update(self, y_true, y_pred, interval_width):
        """
        Update alpha based on whether y_true was covered
        """
        covered = (y_pred - interval_width/2 <= y_true <= y_pred + interval_width/2)

        # If covered, decrease alpha (narrower intervals)
        # If not covered, increase alpha (wider intervals)
        if covered:
            self.alpha_t = self.alpha_t - self.gamma * (1 - self.target_coverage)
        else:
            self.alpha_t = self.alpha_t + self.gamma * self.target_coverage

        # Keep alpha in valid range
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)

    def predict(self, X, calibration_scores):
        y_pred = self.model.predict(X)

        # Compute interval width based on current alpha
        q_level = 1 - self.alpha_t
        q_hat = np.quantile(calibration_scores, q_level)

        return {
            'prediction': y_pred,
            'lower': y_pred - q_hat,
            'upper': y_pred + q_hat,
            'interval_width': 2 * q_hat,
            'current_alpha': self.alpha_t
        }
```

### Trading Strategy with Uncertainty

```python
class ConformalTradingStrategy:
    """
    Trading strategy using calibrated prediction intervals
    """
    def __init__(self, predictor, width_threshold=0.02, min_edge=0.005):
        self.predictor = predictor
        self.width_threshold = width_threshold
        self.min_edge = min_edge

    def generate_signal(self, X):
        """
        Generate trading signal based on prediction interval
        """
        pred = self.predictor.predict(X)

        signal = {
            'prediction': pred['midpoint'],
            'interval_width': pred['interval_width'],
            'confidence': 1 / (1 + pred['interval_width']),
            'trade': False,
            'direction': 0,
            'size': 0
        }

        # Only trade if interval is narrow enough
        if pred['interval_width'] < self.width_threshold:
            # Direction based on whether interval is above/below zero
            if pred['lower'] > self.min_edge:
                signal['direction'] = 1  # Long
                signal['trade'] = True
            elif pred['upper'] < -self.min_edge:
                signal['direction'] = -1  # Short
                signal['trade'] = True

        # Position size inversely proportional to interval width
        if signal['trade']:
            signal['size'] = 1 / pred['interval_width']
            signal['size'] = min(signal['size'], 1.0)  # Cap at 100%

        return signal

    def kelly_sizing(self, prediction, interval_width, risk_free_rate=0):
        """
        Kelly criterion with calibrated uncertainty
        """
        # Expected return (midpoint of interval)
        expected_return = prediction

        # Uncertainty as proxy for outcome probability spread
        # Narrower interval -> higher confidence -> larger Kelly fraction
        confidence = 1 / (1 + interval_width * 10)

        # Simplified Kelly: f* = (p*b - q) / b
        # where p = win prob, b = win/loss ratio
        if expected_return > 0:
            kelly_fraction = confidence * expected_return / interval_width
        else:
            kelly_fraction = 0

        # Half Kelly for safety
        return kelly_fraction / 2
```

### Interval Width Analysis

```python
class IntervalAnalyzer:
    """
    Analyze prediction interval patterns
    """
    def analyze_interval_width(self, predictions, actuals, intervals):
        """
        Analyze when intervals are wide vs narrow
        """
        widths = intervals['upper'] - intervals['lower']

        analysis = {
            'mean_width': widths.mean(),
            'std_width': widths.std(),
            'coverage': ((actuals >= intervals['lower']) &
                        (actuals <= intervals['upper'])).mean(),

            # When are intervals wide?
            'width_vs_volatility': np.corrcoef(widths, actuals.rolling(20).std())[0,1],
            'width_vs_regime': self._regime_analysis(widths),

            # Conditional coverage
            'coverage_narrow': ((actuals >= intervals['lower']) &
                               (actuals <= intervals['upper']))[widths < widths.median()].mean(),
            'coverage_wide': ((actuals >= intervals['lower']) &
                             (actuals <= intervals['upper']))[widths >= widths.median()].mean()
        }

        return analysis

    def optimal_threshold(self, predictions, actuals, intervals):
        """
        Find optimal interval width threshold for trading
        """
        widths = intervals['upper'] - intervals['lower']

        results = []
        for threshold in np.percentile(widths, range(10, 100, 10)):
            mask = widths < threshold
            if mask.sum() > 10:
                trades = predictions[mask]
                returns = actuals[mask]

                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                hit_rate = (np.sign(trades) == np.sign(returns)).mean()
                n_trades = mask.sum()

                results.append({
                    'threshold': threshold,
                    'sharpe': sharpe,
                    'hit_rate': hit_rate,
                    'n_trades': n_trades,
                    'pct_traded': mask.mean()
                })

        return pd.DataFrame(results)
```

### Backtesting Framework

```python
def backtest_conformal_strategy(data, predictor, strategy, train_size=252, calib_size=126):
    """
    Rolling backtest with conformal prediction
    """
    results = []

    for t in range(train_size + calib_size, len(data)):
        # Training window
        train_data = data.iloc[t-train_size-calib_size:t-calib_size]
        calib_data = data.iloc[t-calib_size:t]
        test_point = data.iloc[t:t+1]

        # Fit predictor
        predictor.fit(
            train_data[features], train_data['target'],
            calib_data[features], calib_data['target']
        )

        # Generate signal
        signal = strategy.generate_signal(test_point[features])

        # Record results
        actual_return = data.iloc[t+1]['target'] if t+1 < len(data) else 0

        results.append({
            'date': data.index[t],
            'prediction': signal['prediction'],
            'interval_width': signal['interval_width'],
            'direction': signal['direction'],
            'size': signal['size'],
            'actual': actual_return,
            'pnl': signal['direction'] * signal['size'] * actual_return
        })

    return pd.DataFrame(results)
```

### Key Metrics

- **Coverage:** Empirical vs target coverage, Conditional coverage
- **Interval Quality:** Average width, Width stability
- **Strategy:** Sharpe, Win rate, Avg trade, Turnover
- **Comparison:** vs trading without uncertainty, vs fixed position sizing

### Dependencies

```python
mapie>=0.6.0          # Conformal prediction library
crepes>=0.3.0         # Alternative CP library
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
```

## Expected Outcomes

1. **Conformal prediction framework** для return forecasting
2. **Calibrated intervals** с guaranteed coverage
3. **Trading strategy** using interval width filtering
4. **Position sizing** based on uncertainty
5. **Results:** Better risk-adjusted returns by avoiding low-confidence trades

## References

- [A Tutorial on Conformal Prediction](https://www.jmlr.org/papers/v9/shafer08a.html)
- [Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)
- [Adaptive Conformal Inference](https://arxiv.org/abs/2106.00170)
- [MAPIE Documentation](https://mapie.readthedocs.io/)

## Difficulty Level

⭐⭐⭐☆☆ (Intermediate)

Требуется понимание: Statistical inference, Prediction intervals, Quantile regression, Risk management
