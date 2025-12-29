# Chapter 34: Online Learning — Adaptive Momentum with Continuous Retraining

## Overview

Традиционные ML модели обучаются на batch данных и деградируют со временем (concept drift). Online learning позволяет модели непрерывно адаптироваться к новым данным без полного переобучения. В этой главе мы строим адаптивную momentum стратегию, которая эволюционирует с рынком.

## Trading Strategy

**Суть стратегии:** Online gradient descent для непрерывной адаптации весов momentum сигналов. Модель автоматически увеличивает вес факторов, которые работают, и уменьшает вес тех, что перестали работать.

**Сигнал на вход:**
- Long: Взвешенный momentum score > threshold
- Short: Взвешенный momentum score < -threshold
- Weights: Адаптируются в реальном времени

**Edge:** Быстрая адаптация к смене режимов рынка

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_concept_drift.ipynb` | Теория concept drift, типы изменений |
| 2 | `02_online_learning_basics.ipynb` | SGD, regret bounds, convergence |
| 3 | `03_momentum_features.ipynb` | Набор momentum features для адаптации |
| 4 | `04_river_library.ipynb` | Использование River для online ML |
| 5 | `05_online_linear.ipynb` | Online linear regression для весов |
| 6 | `06_online_tree.ipynb` | Hoeffding Trees для non-linear |
| 7 | `07_drift_detection.ipynb` | ADWIN, DDM для обнаружения drift |
| 8 | `08_adaptive_windows.ipynb` | Динамический размер обучающего окна |
| 9 | `09_ensemble_online.ipynb` | Ensemble of online learners |
| 10 | `10_backtesting.ipynb` | Simulation с streaming data |
| 11 | `11_comparison.ipynb` | vs batch retraining, static model |

### Data Requirements

```
Streaming Data Simulation:
├── Daily stock returns (10+ years)
├── Multiple momentum factors
├── Simulated as streaming: one day at a time
└── No lookahead allowed

Momentum Factors:
├── Price momentum (1m, 3m, 6m, 12m)
├── Volume momentum
├── Earnings momentum
├── Analyst revision momentum
└── Industry momentum
```

### Online Learning Setup

```python
from river import linear_model, preprocessing, optim, drift

# Online linear model with adaptive learning rate
model = preprocessing.StandardScaler() | linear_model.LinearRegression(
    optimizer=optim.Adam(lr=0.01),
    l2=0.001
)

# Streaming prediction loop
for day in trading_days:
    # Get features for today (known at market open)
    x = get_features(day)

    # Predict return
    y_pred = model.predict_one(x)

    # Generate trading signal
    signal = 'long' if y_pred > threshold else 'short' if y_pred < -threshold else 'flat'

    # At end of day, observe actual return
    y_true = get_actual_return(day)

    # Update model with new observation
    model.learn_one(x, y_true)
```

### Concept Drift Detection

```python
from river import drift

# ADWIN: Adaptive Windowing
adwin = drift.ADWIN(delta=0.002)

# DDM: Drift Detection Method
ddm = drift.DDM(min_num_instances=30)

# Detecting drift
for t, (x, y_true) in enumerate(stream):
    y_pred = model.predict_one(x)
    error = abs(y_pred - y_true)

    adwin.update(error)
    if adwin.drift_detected:
        print(f"Drift detected at time {t}")
        # Option 1: Reset model
        model = create_fresh_model()
        # Option 2: Reduce learning rate history
        # Option 3: Switch to different model
```

### Adaptive Window Approach

```python
class AdaptiveWindowModel:
    """
    Dynamically adjusts training window based on recent performance
    """
    def __init__(self, min_window=20, max_window=252):
        self.min_window = min_window
        self.max_window = max_window
        self.current_window = 60
        self.buffer = []

    def update(self, x, y):
        self.buffer.append((x, y))

        # Keep buffer at max size
        if len(self.buffer) > self.max_window:
            self.buffer.pop(0)

        # Retrain on adaptive window
        window_data = self.buffer[-self.current_window:]
        self.model.fit(window_data)

        # Evaluate recent performance
        recent_error = self.evaluate_recent(window=20)

        # Adjust window size
        if recent_error > self.error_threshold:
            # Performance degrading: shrink window (adapt faster)
            self.current_window = max(self.min_window, self.current_window - 10)
        else:
            # Performance good: expand window (more stable)
            self.current_window = min(self.max_window, self.current_window + 5)
```

### Ensemble of Online Learners

```python
from river import ensemble

# Adaptive Random Forest
arf = ensemble.AdaptiveRandomForestRegressor(
    n_models=10,
    max_depth=6,
    drift_detector=drift.ADWIN()
)

# Bagging with online base learners
bagging = ensemble.BaggingRegressor(
    model=linear_model.LinearRegression(),
    n_models=10
)

# Stacking online learners
class OnlineStacking:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict_one(self, x):
        base_preds = [m.predict_one(x) for m in self.base_models]
        return self.meta_model.predict_one(base_preds)

    def learn_one(self, x, y):
        base_preds = [m.predict_one(x) for m in self.base_models]
        for m in self.base_models:
            m.learn_one(x, y)
        self.meta_model.learn_one(base_preds, y)
```

### Momentum Factor Weighting

```python
class AdaptiveMomentumWeights:
    """
    Online learning of optimal momentum factor weights
    """
    def __init__(self, n_factors, learning_rate=0.01):
        self.weights = np.ones(n_factors) / n_factors
        self.lr = learning_rate

    def predict(self, factor_signals):
        return np.dot(self.weights, factor_signals)

    def update(self, factor_signals, actual_return):
        prediction = self.predict(factor_signals)
        error = actual_return - prediction

        # Gradient update
        gradient = -2 * error * factor_signals
        self.weights -= self.lr * gradient

        # Normalize weights (optional)
        self.weights = self.weights / np.sum(np.abs(self.weights))

    def get_factor_importance(self):
        return dict(zip(self.factor_names, self.weights))
```

### Backtesting Framework

```python
def online_backtest(data, model, initial_train=252):
    """
    Backtest with streaming simulation
    """
    results = []

    # Initial training period (no trading)
    for t in range(initial_train):
        x, y = data.iloc[t]
        model.learn_one(x, y)

    # Trading period
    for t in range(initial_train, len(data)):
        x, y = data.iloc[t]

        # Predict before observing
        y_pred = model.predict_one(x)
        signal = generate_signal(y_pred)

        # Trade
        pnl = signal * y

        # Learn after observing
        model.learn_one(x, y)

        results.append({
            'date': data.index[t],
            'prediction': y_pred,
            'actual': y,
            'signal': signal,
            'pnl': pnl
        })

    return pd.DataFrame(results)
```

### Key Metrics

- **Prediction:** Rolling IC, MSE, Directional accuracy
- **Adaptation:** Drift detection frequency, Weight evolution
- **Strategy:** Sharpe, Max DD, Win rate
- **Comparison:** vs static model, vs monthly retrain

### Dependencies

```python
river>=0.18.0          # Main online learning library
scikit-multiflow>=0.5  # Alternative library
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **Streaming simulation framework** для daily data
2. **Online learning models** (linear, trees, ensemble)
3. **Drift detection pipeline** с ADWIN/DDM
4. **Adaptive momentum weights** эволюционирующие со временем
5. **Results:** Higher Sharpe в non-stationary periods vs static model

## References

- [Online Machine Learning](https://www.amazon.com/Online-Machine-Learning-Foundations-Applications/dp/0262046113)
- [River Documentation](https://riverml.xyz/)
- [A Survey on Concept Drift Adaptation](https://dl.acm.org/doi/10.1145/2523813)
- [Adaptive Learning Rate Methods](https://arxiv.org/abs/1412.6980) (Adam)

## Difficulty Level

⭐⭐⭐☆☆ (Intermediate)

Требуется понимание: Online optimization, Concept drift, Streaming algorithms, Momentum factors
