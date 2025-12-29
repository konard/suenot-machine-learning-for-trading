# Chapter 32: Cross-Asset Momentum — Global Tactical Asset Allocation

## Overview

Cross-asset momentum применяет momentum стратегии через различные классы активов: акции, облигации, commodities, валюты, REITs. Это позволяет диверсифицировать источники alpha и снизить корреляцию с традиционными equity momentum стратегиями.

## Trading Strategy

**Суть стратегии:** Dual momentum подход:
1. **Time-series momentum:** Long актив если его return > 0 (или > T-bills)
2. **Cross-sectional momentum:** Rank активов по momentum, long top, short bottom

**Сигнал на вход:**
- Long: Positive absolute momentum + top relative momentum
- Cash: Negative absolute momentum (защита от drawdowns)

**Position Sizing:** Risk parity — volatility targeting для каждой позиции

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_asset_universe.ipynb` | Выбор ETFs для каждого asset class |
| 2 | `02_data_collection.ipynb` | Загрузка long history (15+ лет) |
| 3 | `03_momentum_signals.ipynb` | 1/3/6/12 month momentum calculation |
| 4 | `04_time_series_momentum.ipynb` | Absolute momentum filter |
| 5 | `05_cross_sectional_momentum.ipynb` | Relative momentum ranking |
| 6 | `06_dual_momentum.ipynb` | Комбинация time-series + cross-sectional |
| 7 | `07_volatility_targeting.ipynb` | Position sizing по volatility |
| 8 | `08_risk_parity_weights.ipynb` | Risk parity allocation |
| 9 | `09_rebalancing.ipynb` | Monthly rebalancing logic |
| 10 | `10_backtesting.ipynb` | Full backtest с transaction costs |
| 11 | `11_regime_analysis.ipynb` | Performance по market regimes |
| 12 | `12_ml_enhancement.ipynb` | ML для timing rebalancing |

### Asset Universe

```
Equities:
├── SPY  - US Large Cap
├── IWM  - US Small Cap
├── EFA  - Developed International
├── EEM  - Emerging Markets
└── VNQ  - US REITs

Fixed Income:
├── IEF  - 7-10 Year Treasury
├── TLT  - 20+ Year Treasury
├── LQD  - Investment Grade Corporate
├── HYG  - High Yield Corporate
└── TIP  - TIPS

Commodities:
├── GLD  - Gold
├── SLV  - Silver
├── USO  - Oil
├── DBA  - Agriculture
└── GSG  - Broad Commodities

Currencies:
├── UUP  - US Dollar Index
├── FXE  - Euro
├── FXY  - Yen
└── FXB  - British Pound

Alternatives:
├── VNQ  - REITs
├── VNQI - International REITs
└── MLPA - MLPs
```

### Momentum Calculation

```python
def calculate_momentum(prices, lookback_months=[1, 3, 6, 12]):
    """
    Calculate momentum signals for multiple lookbacks
    """
    momentum = {}
    for lb in lookback_months:
        # Standard momentum: return over lookback period
        mom = prices.pct_change(periods=lb * 21)  # ~21 trading days per month

        # Skip last month (mean reversion effect)
        mom_skip = prices.shift(21).pct_change(periods=(lb-1) * 21)

        momentum[f'mom_{lb}m'] = mom
        momentum[f'mom_{lb}m_skip1m'] = mom_skip

    return pd.DataFrame(momentum)

def composite_momentum(momentum_df, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    Combine multiple lookbacks into single signal
    """
    return (momentum_df['mom_1m_skip1m'] * weights[0] +
            momentum_df['mom_3m_skip1m'] * weights[1] +
            momentum_df['mom_6m_skip1m'] * weights[2] +
            momentum_df['mom_12m_skip1m'] * weights[3])
```

### Dual Momentum Implementation

```python
def dual_momentum_signal(returns, risk_free_rate):
    """
    Antonacci's Dual Momentum approach
    """
    signals = {}

    for asset in returns.columns:
        # Time-series momentum (absolute)
        excess_return = returns[asset].rolling(252).mean() - risk_free_rate
        ts_signal = 1 if excess_return > 0 else 0

        # Cross-sectional momentum (relative)
        all_momentum = returns.rolling(252).mean()
        rank = all_momentum[asset].rank(pct=True)
        cs_signal = 1 if rank > 0.5 else -1

        # Dual momentum: only long if both positive
        signals[asset] = ts_signal * max(cs_signal, 0)

    return pd.DataFrame(signals)
```

### Volatility Targeting

```python
def volatility_target_weights(returns, target_vol=0.10):
    """
    Scale positions to target volatility
    """
    realized_vol = returns.rolling(63).std() * np.sqrt(252)  # 3-month vol

    # Position size inversely proportional to vol
    raw_weights = target_vol / realized_vol

    # Cap individual position at 100%
    capped_weights = raw_weights.clip(upper=1.0)

    return capped_weights

def risk_parity_weights(returns):
    """
    Equal risk contribution from each asset
    """
    cov_matrix = returns.rolling(252).cov()
    vol = returns.rolling(252).std()

    # Inverse volatility weights (simplified risk parity)
    inv_vol_weights = 1 / vol
    weights = inv_vol_weights / inv_vol_weights.sum()

    return weights
```

### Rebalancing Rules

```
Rebalancing Schedule:
├── Monthly rebalancing (end of month)
├── Optional: weekly for faster signals
└── Transaction cost consideration

Rebalancing Bands:
├── Only trade if weight drift > 5%
├── Reduces turnover significantly
└── Maintains approximate target allocation

Momentum Decay:
├── Fresh signal = full weight
├── Aging signal = reduced weight
└── Prevents whipsaws at signal boundary
```

### Key Metrics

- **Returns:** CAGR, Total Return
- **Risk:** Volatility, Max Drawdown, VaR
- **Risk-Adjusted:** Sharpe, Sortino, Calmar
- **Momentum-Specific:** Hit Rate, Average Win/Loss, Turnover
- **Comparison:** vs 60/40, Buy&Hold SPY, AQR Momentum

### Dependencies

```python
pandas>=1.5.0
numpy>=1.23.0
yfinance>=0.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
empyrical>=0.5.5  # For performance metrics
pyfolio>=0.9.2    # For tearsheets
```

## Expected Outcomes

1. **Asset universe** из 20+ ETFs across asset classes
2. **Momentum signals** с multiple lookbacks
3. **Dual momentum strategy** с time-series + cross-sectional
4. **Risk parity implementation** для position sizing
5. **Backtest results:** Sharpe > 0.8, Max DD < 20% (vs 50%+ for equities)

## References

- [Dual Momentum Investing](https://www.amazon.com/Dual-Momentum-Investing-Innovative-Strategy/dp/0071849440) (Gary Antonacci)
- [Time Series Momentum](https://pages.stern.nyu.edu/~lpedMDL1/papers/TimeSeriesMomentum.pdf) (Moskowitz, Ooi, Pedersen)
- [Value and Momentum Everywhere](https://pages.stern.nyu.edu/~lpedMDL1/papers/ValMomEverywhere.pdf) (Asness, Moskowitz, Pedersen)
- [AQR Momentum Indices](https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly)

## Difficulty Level

⭐⭐⭐☆☆ (Intermediate)

Требуется понимание: Momentum factors, Asset allocation, Risk parity, Portfolio construction
