# Chapter 33: Anomaly Detection — Tail Risk Hedging Strategy

## Overview

Аномальные рыночные условия часто предшествуют кризисам или резким движениям. Anomaly detection позволяет идентифицировать "необычные" состояния рынка до того, как они станут очевидными. В этой главе мы строим систему раннего предупреждения и автоматического хеджирования.

## Trading Strategy

**Суть стратегии:** Мониторинг multivariate anomaly score. При превышении порога — автоматическая покупка защитных инструментов (VIX calls, put spreads, treasuries).

**Сигнал на хедж:**
- Anomaly score > 95th percentile → Light hedge (5% portfolio)
- Anomaly score > 99th percentile → Heavy hedge (15% portfolio)

**Instruments:** VIX calls, SPY puts, TLT longs, Gold

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_market_features.ipynb` | Feature engineering для market state |
| 2 | `02_historical_crises.ipynb` | Анализ исторических кризисов как ground truth |
| 3 | `03_isolation_forest.ipynb` | Isolation Forest для anomaly detection |
| 4 | `04_one_class_svm.ipynb` | One-Class SVM подход |
| 5 | `05_autoencoder_anomaly.ipynb` | Autoencoder reconstruction error |
| 6 | `06_mahalanobis_distance.ipynb` | Statistical approach |
| 7 | `07_ensemble_detector.ipynb` | Комбинация методов |
| 8 | `08_threshold_calibration.ipynb` | Калибровка порогов на исторических кризисах |
| 9 | `09_hedging_instruments.ipynb` | Выбор и pricing защитных инструментов |
| 10 | `10_hedging_strategy.ipynb` | Автоматическая логика хеджирования |
| 11 | `11_backtesting.ipynb` | Backtest с cost-benefit analysis |
| 12 | `12_real_time_monitoring.ipynb` | Dashboard для мониторинга |

### Data Requirements

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

### Feature Engineering

```python
features = {
    # Volatility
    'vix_level': VIX,
    'vix_percentile': rolling_percentile(VIX, 252),
    'vix_term_structure': (VIX3M - VIX) / VIX,  # Contango/Backwardation
    'realized_vs_implied': realized_vol_20d / VIX,
    'vol_of_vol': rolling_std(VIX, 20),

    # Credit
    'hy_spread': HYG_yield - treasury_yield,
    'hy_spread_change_5d': hy_spread.pct_change(5),
    'ted_spread': LIBOR_3m - treasury_3m,
    'credit_momentum': LQD.pct_change(20),

    # Equity
    'spy_drawdown': SPY / SPY.rolling(252).max() - 1,
    'spy_momentum_20d': SPY.pct_change(20),
    'breadth': pct_stocks_above_200ma,
    'sector_dispersion': std(sector_returns),

    # Cross-asset
    'stock_bond_corr': rolling_corr(SPY, TLT, 60),
    'gold_correlation': rolling_corr(SPY, GLD, 60),
    'currency_stress': USD_index_change_20d,

    # Market internals
    'put_call_ratio': equity_put_volume / equity_call_volume,
    'margin_debt_change': margin_debt.pct_change(63),
    'fund_flows': equity_fund_flows_4w
}
```

### Anomaly Detection Models

```python
# 1. Isolation Forest
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expected 5% anomalies
    random_state=42
)
anomaly_score_if = -iso_forest.decision_function(features)

# 2. One-Class SVM
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05
)
anomaly_score_svm = -oc_svm.decision_function(features)

# 3. Autoencoder
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Anomaly score = reconstruction error
reconstruction_error = torch.mean((x - autoencoder(x))**2, dim=1)

# 4. Mahalanobis Distance
from scipy.spatial.distance import mahalanobis

def mahalanobis_score(x, mean, cov_inv):
    return mahalanobis(x, mean, cov_inv)
```

### Ensemble Anomaly Score

```python
def ensemble_anomaly_score(features, models, weights=None):
    """
    Combine multiple anomaly detectors
    """
    if weights is None:
        weights = [1/len(models)] * len(models)

    scores = []
    for model in models:
        # Normalize scores to [0, 1]
        raw_score = model.score(features)
        normalized = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
        scores.append(normalized)

    # Weighted average
    ensemble_score = sum(w * s for w, s in zip(weights, scores))

    return ensemble_score
```

### Hedging Strategy

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

### Key Metrics

- **Detection:** True Positive Rate, False Positive Rate, AUC
- **Timing:** Days of warning before crisis, False alarms per year
- **Portfolio:** Max Drawdown with/without hedge, Recovery time
- **Cost:** Annual hedge cost, Hedge efficiency ratio

### Dependencies

```python
scikit-learn>=1.2.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **Feature set** из 20+ market stress indicators
2. **Multiple anomaly detectors** (IF, OCSVM, AE, Mahalanobis)
3. **Ensemble model** с calibrated thresholds
4. **Hedging strategy** с автоматическим execution
5. **Results:** 30-50% drawdown reduction с < 3% annual cost

## References

- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [A Survey of Anomaly Detection Techniques](https://link.springer.com/article/10.1023/A:1006412129858)
- [Tail Risk Hedging](https://www.amazon.com/Tail-Risk-Hedging-Portfolio-Management/dp/0071791752)
- [Predicting Stock Market Crashes](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635170)

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: Anomaly detection, Options pricing, Risk management, Crisis analysis
