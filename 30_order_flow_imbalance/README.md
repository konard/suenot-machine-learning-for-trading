# Chapter 30: Order Flow Imbalance — Intraday Microstructure Strategy

## Overview

Order Flow Imbalance (OFI) — мера дисбаланса между давлением покупателей и продавцов на уровне order book. Экстремальный дисбаланс часто предшествует краткосрочному движению цены. В этой главе мы используем ML для прогнозирования краткосрочных movements на основе микроструктуры рынка.

## Trading Strategy

**Суть стратегии:** Анализ order flow imbalance из L2 order book data. ML предсказывает вероятность reversal или continuation на горизонте 1-5 минут.

**Сигнал на вход:**
- Long: Сильный buy imbalance + ML predicts continuation
- Short: Сильный sell imbalance + ML predicts continuation
- Mean-reversion: Extreme imbalance + ML predicts reversal

**Edge:** Информация из order book опережает ценовые движения

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_order_book_data.ipynb` | Загрузка и парсинг L2 order book data |
| 2 | `02_order_flow_basics.ipynb` | Теория: OFI, VPIN, Kyle's Lambda |
| 3 | `03_ofi_calculation.ipynb` | Расчет Order Flow Imbalance метрик |
| 4 | `04_feature_engineering.ipynb` | Features из order book snapshots |
| 5 | `05_labeling.ipynb` | Создание labels (future price move) |
| 6 | `06_model_training.ipynb` | LightGBM для предсказания направления |
| 7 | `07_signal_generation.ipynb` | Генерация торговых сигналов |
| 8 | `08_execution_simulation.ipynb` | Симуляция исполнения с slippage |
| 9 | `09_backtesting.ipynb` | Intraday backtesting framework |
| 10 | `10_latency_analysis.ipynb` | Влияние latency на P&L |
| 11 | `11_production_considerations.ipynb` | Real-time implementation notes |

### Data Requirements

```
Order Book Data (L2):
├── Bid/Ask prices (10+ levels)
├── Bid/Ask sizes at each level
├── Timestamps (millisecond or better)
├── Trade prints (time & sales)
└── At least 1 year of data

Recommended Sources:
├── Lobster Data (academic, NASDAQ)
├── Algoseek (commercial)
├── Crypto: Binance/Coinbase API
├── Futures: CME DataMine
└── Simulation: Generate synthetic

Instruments:
├── Liquid stocks (AAPL, MSFT, SPY)
├── E-mini S&P 500 futures (ES)
├── Bitcoin (BTC-USD)
└── High volume = better signal
```

### Order Flow Imbalance Calculation

```python
def calculate_ofi(book_t, book_t1):
    """
    Order Flow Imbalance based on Cont et al. (2014)
    """
    # Bid side changes
    if book_t1['bid_price'] > book_t['bid_price']:
        delta_bid = book_t1['bid_size']
    elif book_t1['bid_price'] == book_t['bid_price']:
        delta_bid = book_t1['bid_size'] - book_t['bid_size']
    else:
        delta_bid = -book_t['bid_size']

    # Ask side changes
    if book_t1['ask_price'] < book_t['ask_price']:
        delta_ask = -book_t1['ask_size']
    elif book_t1['ask_price'] == book_t['ask_price']:
        delta_ask = -(book_t1['ask_size'] - book_t['ask_size'])
    else:
        delta_ask = book_t['ask_size']

    ofi = delta_bid + delta_ask
    return ofi

# Aggregate OFI over time windows
ofi_1min = ofi_series.rolling('1min').sum()
ofi_5min = ofi_series.rolling('5min').sum()
```

### Feature Engineering from Order Book

```python
features = {
    # Imbalance metrics
    'ofi_1min': sum(ofi, 1min),
    'ofi_5min': sum(ofi, 5min),
    'volume_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
    'depth_imbalance_l1': (bid_size_l1 - ask_size_l1) / (bid_size_l1 + ask_size_l1),
    'depth_imbalance_l5': same for top 5 levels,

    # Spread and liquidity
    'spread_bps': (ask - bid) / mid * 10000,
    'spread_z_score': (spread - mean_spread) / std_spread,
    'total_depth': sum(bid_sizes) + sum(ask_sizes),
    'depth_ratio': sum(bid_sizes) / sum(ask_sizes),

    # Trade flow
    'buy_volume_1min': volume of buys (trade >= ask),
    'sell_volume_1min': volume of sells (trade <= bid),
    'trade_imbalance': (buys - sells) / (buys + sells),
    'large_trade_indicator': any trade > 2 * avg_trade_size,

    # Order book shape
    'bid_slope': price_impact per 100 shares (bid side),
    'ask_slope': price_impact per 100 shares (ask side),
    'resilience': how fast depth recovers after large trade,

    # Momentum/Mean-reversion indicators
    'price_momentum_1min': return over last 1 min,
    'volatility_1min': realized vol over 1 min,
    'vwap_distance': (price - vwap) / price,
}
```

### VPIN (Volume-Synchronized Probability of Informed Trading)

```python
def calculate_vpin(trades, bucket_size=50):
    """
    VPIN indicator for toxicity detection
    """
    # Classify trades as buy/sell (Lee-Ready or similar)
    trades['side'] = classify_trades(trades)

    # Create volume buckets
    buckets = create_volume_buckets(trades, bucket_size)

    # Calculate buy/sell imbalance per bucket
    vpin = []
    for bucket in buckets:
        buy_vol = bucket[bucket['side'] == 'buy']['volume'].sum()
        sell_vol = bucket[bucket['side'] == 'sell']['volume'].sum()
        vpin.append(abs(buy_vol - sell_vol) / (buy_vol + sell_vol))

    # Rolling VPIN
    return pd.Series(vpin).rolling(50).mean()
```

### Model Architecture

```
Input: 50+ microstructure features
├── OFI features (10)
├── Depth features (10)
├── Trade flow features (10)
├── Order book shape (8)
├── Technical (7)
└── Time features (5)

Model: LightGBM (speed critical for HFT)
├── max_depth: 6
├── num_leaves: 31
├── learning_rate: 0.05
├── Feature importance for interpretability

Output:
├── P(price_up_1min)
├── P(price_up_5min)
├── Expected magnitude (regression)

Trading Rule:
├── Long if P(up) > 0.55 AND ofi > threshold
├── Short if P(up) < 0.45 AND ofi < -threshold
├── Hold: 1-5 minutes
├── Stop-loss: 2-3 ticks
```

### Execution Considerations

```
Latency Requirements:
├── Feature calculation: <1ms
├── Model prediction: <1ms
├── Order submission: <10ms
├── Total latency: <50ms target

Execution Costs:
├── Commission: ~$0.001 per share
├── Spread: 1 tick minimum
├── Market impact: depends on size
├── Slippage model: queue position based

Position Management:
├── Max position: 100-1000 shares
├── Max holding: 5 minutes
├── Flat by end of day
└── Max daily loss: stop trading
```

### Key Metrics

- **Signal Quality:** IC (Information Coefficient), Hit rate per horizon
- **Execution:** Slippage vs expected, Fill rate
- **P&L:** Gross P&L, Net P&L after costs, Sharpe (intraday)
- **Risk:** Max drawdown (intraday), # losing days

### Dependencies

```python
pandas>=1.5.0
numpy>=1.23.0
lightgbm>=4.0.0
numba>=0.57.0       # For fast calculations
polars>=0.18.0      # For large data processing
matplotlib>=3.6.0
lobsterdata>=0.1.0  # If using LOBSTER format
```

## Expected Outcomes

1. **Order book parser** для L2 data (LOBSTER, Algoseek format)
2. **OFI calculation pipeline** с multiple timeframes
3. **Feature library (50+ features)** из order book
4. **LightGBM model** с >52% accuracy на 1-min direction
5. **Execution simulator** с realistic slippage
6. **Intraday backtest** с gross Sharpe > 2.0 (before costs)
7. **Analysis** влияния latency на profitability

## References

- [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) (Cont et al., 2014)
- [VPIN and the Flash Crash](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596)
- [Machine Learning for Market Microstructure](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/)
- [LOBSTER: Limit Order Book System](https://lobsterdata.com/)
- [High-Frequency Trading: A Practical Guide](https://www.wiley.com/en-us/High+Frequency+Trading)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Требуется понимание: Market microstructure, Order book mechanics, Low-latency systems, HFT concepts

## Important Disclaimers

- HFT стратегии требуют significant infrastructure investment
- Latency critical — колокация может быть необходима
- Регуляторные требования (SEC, MiFID II)
- Paper trading обязателен перед live
- Эта глава — educational, не production-ready система
