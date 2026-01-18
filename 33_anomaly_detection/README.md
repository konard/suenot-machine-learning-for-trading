# Chapter 33: Anomaly Detection — Market Regime Detection, Unusual Pattern Recognition & Tail Risk Hedging

## Overview

Anomaly Detection в трейдинге применяется для выявления необычного поведения рынка: аномальных движений цен, манипуляций, flash crashes, внезапных изменений режима рынка. Это критически важно как для risk management, так и для поиска торговых возможностей — аномалии часто предшествуют крупным движениям.

## Trading Strategy

**Суть стратегии:** Multi-layer anomaly detection:
1. **Statistical anomalies:** Z-score, IQR-based outliers в returns и volume
2. **Pattern anomalies:** Отклонения от типичных паттернов (Isolation Forest, Autoencoder)
3. **Regime anomalies:** Hidden Markov Models для обнаружения смены режима

**Сигнал на вход:**
- **Protective exit:** Закрытие позиций при обнаружении аномальной волатильности
- **Contrarian entry:** Вход после flash crash, когда аномалия завершается
- **Regime switch:** Адаптация стратегии при смене рыночного режима

**Risk Management:** Автоматическое снижение позиций при повышенном anomaly score

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_statistical_anomalies.ipynb` | Z-score, IQR, rolling statistics |
| 2 | `02_isolation_forest.ipynb` | Unsupervised anomaly detection |
| 3 | `03_autoencoder_anomaly.ipynb` | Neural network reconstruction error |
| 4 | `04_variational_autoencoder.ipynb` | VAE для probabilistic anomaly detection |
| 5 | `05_hidden_markov_model.ipynb` | Regime detection with HMM |
| 6 | `06_mahalanobis_distance.ipynb` | Multivariate outlier detection |
| 7 | `07_local_outlier_factor.ipynb` | LOF для density-based detection |
| 8 | `08_dbscan_clustering.ipynb` | Cluster-based anomaly detection |
| 9 | `09_lstm_autoencoder.ipynb` | Sequence-aware anomaly detection |
| 10 | `10_real_time_detection.ipynb` | Online anomaly scoring |
| 11 | `11_trading_signals.ipynb` | Converting anomalies to signals |
| 12 | `12_backtesting.ipynb` | Full strategy backtest |

### Anomaly Detection Methods

```
Statistical Methods:
├── Z-Score              - Simple univariate detection
├── Modified Z-Score     - MAD-based robust detection
├── IQR Method           - Quartile-based outliers
├── Grubbs Test          - Single outlier test
└── Rolling Statistics   - Adaptive thresholds

Machine Learning Methods:
├── Isolation Forest     - Tree-based isolation
├── One-Class SVM        - Boundary-based detection
├── Local Outlier Factor - Density-based detection
├── DBSCAN              - Clustering-based detection
└── Elliptic Envelope   - Gaussian assumption

Deep Learning Methods:
├── Autoencoder         - Reconstruction error
├── Variational AE      - Probabilistic detection
├── LSTM Autoencoder    - Temporal patterns
├── Transformer AE      - Attention-based
└── GAN-based           - Generative anomaly scoring

Probabilistic Methods:
├── Hidden Markov Model - Regime detection
├── Gaussian Mixture    - Multi-modal detection
├── Bayesian Detection  - Uncertainty-aware
└── CUSUM              - Change point detection
```

### Feature Engineering for Anomaly Detection

```python
def compute_anomaly_features(df):
    """
    Create features optimized for anomaly detection
    """
    features = {}

    # Price-based features
    features['return'] = df['close'].pct_change()
    features['log_return'] = np.log(df['close']).diff()
    features['return_abs'] = features['return'].abs()

    # Volatility features
    features['volatility_5'] = features['return'].rolling(5).std()
    features['volatility_20'] = features['return'].rolling(20).std()
    features['vol_ratio'] = features['volatility_5'] / features['volatility_20']

    # Volume anomalies
    features['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(20).mean()) /
        df['volume'].rolling(20).std()
    )
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Price-Volume relationship
    features['pv_correlation'] = (
        features['return'].rolling(20).corr(df['volume'].pct_change())
    )

    # Spread and range
    features['range'] = (df['high'] - df['low']) / df['close']
    features['range_ratio'] = features['range'] / features['range'].rolling(20).mean()

    # Microstructure features
    features['close_position'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'])
    )

    # Return distribution moments
    features['skewness'] = features['return'].rolling(20).skew()
    features['kurtosis'] = features['return'].rolling(20).kurt()

    return pd.DataFrame(features)
```

### Z-Score Anomaly Detection

```python
def zscore_anomaly(series, window=20, threshold=3.0):
    """
    Detect anomalies using rolling Z-score
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    zscore = (series - rolling_mean) / rolling_std

    anomaly = np.abs(zscore) > threshold
    anomaly_score = np.abs(zscore) / threshold  # Normalized score

    return anomaly, anomaly_score, zscore

def modified_zscore_anomaly(series, threshold=3.5):
    """
    MAD-based robust Z-score (less sensitive to outliers)
    """
    median = series.median()
    mad = np.median(np.abs(series - median))

    # Scale factor for consistency with standard deviation
    modified_zscore = 0.6745 * (series - median) / mad

    anomaly = np.abs(modified_zscore) > threshold
    return anomaly, modified_zscore
```

### Isolation Forest Implementation

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_anomaly(features, contamination=0.01):
    """
    Isolation Forest for multivariate anomaly detection

    Key insight: Anomalies are easier to isolate,
    requiring fewer random splits
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples='auto',
        random_state=42
    )

    # Fit and predict (-1 = anomaly, 1 = normal)
    labels = model.fit_predict(features)

    # Anomaly score (lower = more anomalous)
    scores = model.decision_function(features)

    # Convert to positive anomaly score (higher = more anomalous)
    anomaly_score = -scores

    return labels == -1, anomaly_score
```

### Autoencoder Anomaly Detection

```python
import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection via reconstruction error
    """
    def __init__(self, input_dim, encoding_dim=8):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def anomaly_score(self, x):
        """
        Compute reconstruction error as anomaly score
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=1)
        return mse.numpy()

def train_autoencoder(model, normal_data, epochs=100, lr=0.001):
    """
    Train autoencoder on normal data only
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(normal_data)
        loss = criterion(output, normal_data)

        loss.backward()
        optimizer.step()

    return model
```

### Hidden Markov Model for Regime Detection

```python
from hmmlearn import hmm

def fit_regime_hmm(returns, n_regimes=3):
    """
    Fit Hidden Markov Model to detect market regimes

    Typical regimes:
    - Low volatility (calm market)
    - Normal volatility (trending)
    - High volatility (crisis/opportunity)
    """
    # Prepare features
    features = np.column_stack([
        returns,
        returns.rolling(5).std(),
        returns.rolling(20).mean()
    ]).dropna()

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100
    )

    model.fit(features)

    # Predict regimes
    regimes = model.predict(features)

    # Compute regime probabilities
    probs = model.predict_proba(features)

    # Anomaly: low probability of being in any regime
    max_prob = probs.max(axis=1)
    regime_anomaly = max_prob < 0.5  # Uncertain regime = anomalous

    return regimes, probs, regime_anomaly
```

### Real-time Anomaly Detection

```python
class OnlineAnomalyDetector:
    """
    Online anomaly detection for real-time trading
    """
    def __init__(self, lookback=100, threshold=3.0):
        self.lookback = lookback
        self.threshold = threshold
        self.buffer = []

    def update(self, value):
        """
        Add new observation and return anomaly score
        """
        self.buffer.append(value)

        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)

        if len(self.buffer) < self.lookback // 2:
            return 0.0, False  # Not enough data

        # Compute statistics
        data = np.array(self.buffer)
        mean = np.mean(data[:-1])  # Exclude current
        std = np.std(data[:-1])

        if std == 0:
            return 0.0, False

        zscore = (value - mean) / std
        anomaly_score = abs(zscore)
        is_anomaly = anomaly_score > self.threshold

        return anomaly_score, is_anomaly

    def get_adaptive_threshold(self, base_threshold=3.0):
        """
        Adapt threshold based on recent volatility
        """
        if len(self.buffer) < self.lookback:
            return base_threshold

        recent_vol = np.std(self.buffer[-20:])
        historical_vol = np.std(self.buffer)

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Increase threshold in high volatility
        return base_threshold * max(1.0, vol_ratio)
```

### Trading Signals from Anomalies

```python
def anomaly_trading_signals(prices, anomaly_scores, config):
    """
    Convert anomaly scores to trading signals
    """
    signals = pd.DataFrame(index=prices.index)

    # Base signals
    signals['anomaly_score'] = anomaly_scores
    signals['return'] = prices.pct_change()

    # Risk reduction on high anomaly
    signals['reduce_position'] = (
        anomaly_scores > config['reduce_threshold']
    )

    # Exit on extreme anomaly
    signals['emergency_exit'] = (
        anomaly_scores > config['exit_threshold']
    )

    # Contrarian entry after anomaly resolves
    signals['anomaly_resolved'] = (
        (anomaly_scores.shift(1) > config['entry_threshold']) &
        (anomaly_scores < config['entry_threshold'] * 0.5)
    )

    # Entry direction based on prior anomaly type
    signals['contrarian_long'] = (
        signals['anomaly_resolved'] &
        (signals['return'].shift(1) < 0)  # Buy after down anomaly
    )

    signals['contrarian_short'] = (
        signals['anomaly_resolved'] &
        (signals['return'].shift(1) > 0)  # Sell after up anomaly
    )

    return signals
```

### Ensemble Anomaly Detection

```python
class EnsembleAnomalyDetector:
    """
    Combine multiple anomaly detection methods
    """
    def __init__(self):
        self.detectors = {
            'zscore': self._zscore_score,
            'isolation_forest': self._iforest_score,
            'autoencoder': self._autoencoder_score,
            'lof': self._lof_score
        }
        self.weights = {
            'zscore': 0.2,
            'isolation_forest': 0.3,
            'autoencoder': 0.3,
            'lof': 0.2
        }

    def fit(self, normal_data):
        """
        Fit all detectors on normal data
        """
        self.iforest = IsolationForest(contamination=0.01)
        self.iforest.fit(normal_data)

        self.autoencoder = train_autoencoder(
            AnomalyAutoencoder(normal_data.shape[1]),
            torch.FloatTensor(normal_data)
        )

        from sklearn.neighbors import LocalOutlierFactor
        self.lof = LocalOutlierFactor(novelty=True)
        self.lof.fit(normal_data)

        # Store statistics for z-score
        self.mean = normal_data.mean(axis=0)
        self.std = normal_data.std(axis=0)

        return self

    def score(self, data):
        """
        Compute weighted ensemble anomaly score
        """
        scores = {}

        for name, scorer in self.detectors.items():
            scores[name] = scorer(data)

        # Normalize each score to [0, 1]
        for name in scores:
            s = scores[name]
            scores[name] = (s - s.min()) / (s.max() - s.min() + 1e-8)

        # Weighted combination
        ensemble_score = sum(
            scores[name] * self.weights[name]
            for name in scores
        )

        return ensemble_score, scores
```

### Architecture Diagram

```
                    Market Data Input
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Statistical │ │  ML-Based   │ │ Deep        │
    │ Methods     │ │  Methods    │ │ Learning    │
    ├─────────────┤ ├─────────────┤ ├─────────────┤
    │ - Z-Score   │ │ - Isolation │ │ - Auto-     │
    │ - IQR       │ │   Forest    │ │   encoder   │
    │ - MAD       │ │ - One-Class │ │ - LSTM-AE   │
    │ - Grubbs    │ │   SVM       │ │ - VAE       │
    │             │ │ - LOF       │ │             │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                 ┌─────────────────┐
                 │ Ensemble Layer  │
                 │ (Weighted Vote) │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Anomaly Score   │
                 │ (0-1 scale)     │
                 └────────┬────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Risk        │ │ Trading     │ │ Alert       │
    │ Management  │ │ Signals     │ │ System      │
    │ - Position  │ │ - Exit      │ │ - Telegram  │
    │   sizing    │ │ - Entry     │ │ - Email     │
    │ - Hedging   │ │ - Regime    │ │ - Dashboard │
    └─────────────┘ └─────────────┘ └─────────────┘
```

### Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of hourly data
├── Recommended: 3+ years for regime detection
├── Frequency: 1-minute to daily
└── Source: Bybit, Binance, or other exchange APIs

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: trades count, funding rate

Feature Requirements:
├── Returns (raw and log)
├── Volatility measures
├── Volume ratios
├── Technical indicators
└── Cross-asset correlations (optional)
```

### Key Metrics

- **Detection Rate:** True positive rate for known anomalies
- **False Positive Rate:** Critical for trading (false alarms = unnecessary trades)
- **Precision@K:** Precision for top-K anomaly scores
- **AUROC:** Area under ROC curve
- **Time-to-Detection:** Latency from anomaly start to detection
- **Strategy Metrics:** Sharpe, Max DD with anomaly-based risk management

### Dependencies

```python
# Core
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.2.0
hmmlearn>=0.3.0

# Deep Learning
torch>=2.0.0
pytorch-lightning>=2.0.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0

# Market Data
ccxt>=4.0.0  # For exchange APIs
websocket-client>=1.4.0

# Optional
pyod>=1.0.0  # Python Outlier Detection library
alibi-detect>=0.11.0  # Outlier and drift detection
```

## Expected Outcomes

1. **Multi-method anomaly detection** с ensemble scoring
2. **Regime detection** с Hidden Markov Models
3. **Real-time anomaly scoring** для live trading
4. **Trading signals** с risk-adjusted position sizing
5. **Backtest results:** Улучшение risk-adjusted returns на 15-30% через anomaly-based risk management

## References

- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (Liu, Ting, Zhou)
- [Anomaly Detection in Financial Markets](https://arxiv.org/abs/1911.04107)
- [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407)
- [Hidden Markov Models in Finance](https://www.sciencedirect.com/science/article/pii/S0378426608001987)
- [PyOD: Python Outlier Detection Library](https://pyod.readthedocs.io/)

## Rust Implementations

This chapter includes two Rust implementations:

### 1. Anomaly Detection (`rust_anomaly_crypto/`)
High-performance anomaly detection on cryptocurrency data from Bybit.
- Real-time data fetching from Bybit
- Statistical anomaly detection (Z-score, MAD, IQR)
- Isolation Forest implementation
- Online anomaly detection
- Modular and extensible design

### 2. Risk Hedging (`rust_risk_hedging/`)
Automated tail risk hedging system.
- Multi-model anomaly scoring
- Automatic hedge sizing
- Portfolio protection logic

---

# Part 2: Tail Risk Hedging Strategy

## Overview

Аномальные рыночные условия часто предшествуют кризисам или резким движениям. Anomaly detection позволяет идентифицировать "необычные" состояния рынка до того, как они станут очевидными. В этой части мы строим систему раннего предупреждения и автоматического хеджирования.

## Hedging Strategy

**Суть стратегии:** Мониторинг multivariate anomaly score. При превышении порога — автоматическая покупка защитных инструментов (VIX calls, put spreads, treasuries).

**Сигнал на хедж:**
- Anomaly score > 95th percentile → Light hedge (5% portfolio)
- Anomaly score > 99th percentile → Heavy hedge (15% portfolio)

**Instruments:** VIX calls, SPY puts, TLT longs, Gold

### Hedging Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_market_features.ipynb` | Feature engineering для market state |
| 2 | `02_historical_crises.ipynb` | Анализ исторических кризисов как ground truth |
| 3 | `03_threshold_calibration.ipynb` | Калибровка порогов на исторических кризисах |
| 4 | `04_hedging_instruments.ipynb` | Выбор и pricing защитных инструментов |
| 5 | `05_hedging_strategy.ipynb` | Автоматическая логика хеджирования |
| 6 | `06_hedging_backtesting.ipynb` | Backtest с cost-benefit analysis |

### Market Stress Indicators

```
Market Indicators:
├── VIX and VIX term structure (VIX, VIX3M, VIX6M)
├── Credit spreads (HY-IG, TED spread, LIBOR-OIS)
├── Equity indices (SPY, sector ETFs)
├── Bond markets (TLT, LQD, HYG)
├── Currency stress (USD index, JPY, CHF)
├── Commodity signals (Gold, Oil)
└── Intermarket correlations

Historical Crisis Dates (Ground Truth):
├── 2008 Financial Crisis
├── 2010 Flash Crash
├── 2011 European Debt Crisis
├── 2015 China Devaluation
├── 2018 Volmageddon
├── 2020 COVID Crash
└── 2022 Rate Shock
```

### Hedging Decision Logic

```python
def hedging_decision(anomaly_score, thresholds, portfolio_value):
    """
    Determine hedge size based on anomaly score
    """
    if anomaly_score > thresholds['extreme']:  # 99th percentile
        hedge_pct = 0.15
        instruments = {
            'VIX_calls': 0.05,
            'SPY_puts': 0.05,
            'TLT': 0.03,
            'GLD': 0.02
        }
    elif anomaly_score > thresholds['high']:  # 95th percentile
        hedge_pct = 0.05
        instruments = {
            'VIX_calls': 0.02,
            'SPY_puts': 0.02,
            'TLT': 0.01
        }
    else:
        hedge_pct = 0
        instruments = {}

    return {k: v * portfolio_value for k, v in instruments.items()}
```

### Cost-Benefit Analysis

```
Hedging Costs:
├── VIX calls: ~3-5% decay per month in contango
├── SPY puts: theta decay + spread
├── Opportunity cost: hedge $ not invested

Benefits:
├── Drawdown reduction during crises
├── Faster recovery (less capital loss)
├── Psychological benefit (stay invested)

Optimization:
├── Minimize: E[hedge_cost] - λ * E[tail_loss_reduction]
├── Backtest across multiple crises
└── Find optimal threshold calibration
```

### Hedging Metrics

- **Detection:** True Positive Rate, False Positive Rate, AUC
- **Timing:** Days of warning before crisis, False alarms per year
- **Portfolio:** Max Drawdown with/without hedge, Recovery time
- **Cost:** Annual hedge cost, Hedge efficiency ratio

## Expected Outcomes (Hedging)

1. **Feature set** из 20+ market stress indicators
2. **Ensemble model** с calibrated thresholds
3. **Hedging strategy** с автоматическим execution
4. **Results:** 30-50% drawdown reduction с < 3% annual cost

## References (Hedging)

- [Tail Risk Hedging](https://www.amazon.com/Tail-Risk-Hedging-Portfolio-Management/dp/0071791752)
- [Predicting Stock Market Crashes](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635170)

---

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: Statistics, Machine Learning, Time Series Analysis, Risk Management, Options Pricing, Crisis Analysis
