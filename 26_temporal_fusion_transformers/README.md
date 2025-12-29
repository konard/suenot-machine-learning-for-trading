# Chapter 26: Temporal Fusion Transformers — Multi-horizon Portfolio Allocation

## Overview

Temporal Fusion Transformer (TFT) — это state-of-the-art архитектура для интерпретируемого multi-horizon прогнозирования временных рядов. В этой главе мы применяем TFT для прогнозирования доходности активов на разных горизонтах и построения динамической стратегии аллокации портфеля.

## Trading Strategy

**Суть стратегии:** Прогнозирование доходности на 1/5/20 дней вперед с quantile forecasts. Динамическая аллокация между акциями, облигациями и кэшем на основе предсказанных распределений и соотношения risk/reward.

**Сигнал на вход:**
- Long: Прогноз положительной доходности с узким confidence interval
- Short: Прогноз отрицательной доходности с узким confidence interval
- Cash: Широкий confidence interval (высокая неопределенность)

**Position Sizing:** Обратно пропорционально ширине prediction interval

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_preparation.ipynb` | Загрузка данных (акции, облигации, macro), feature engineering |
| 2 | `02_tft_architecture.ipynb` | Разбор архитектуры TFT: Variable Selection, Gating, Attention |
| 3 | `03_model_training.ipynb` | Обучение TFT с PyTorch Forecasting / Darts |
| 4 | `04_multi_horizon_forecasting.ipynb` | Прогнозы на 1/5/20 дней с quantile regression |
| 5 | `05_interpretability.ipynb` | Attention weights, Variable Importance, Temporal patterns |
| 6 | `06_allocation_strategy.ipynb` | Правила аллокации на основе прогнозов |
| 7 | `07_backtesting.ipynb` | Backtest стратегии с transaction costs |
| 8 | `08_comparison_baselines.ipynb` | Сравнение с LSTM, ARIMA, Buy&Hold |

### Data Requirements

```
Primary Data:
├── S&P 500 daily OHLCV (10+ лет)
├── US Treasury yields (2Y, 10Y, 30Y)
├── VIX index
├── Sector ETFs (XLF, XLK, XLE, etc.)
└── Macro indicators (GDP, CPI, Unemployment)

Features:
├── Static: Sector, Market Cap category
├Known future: Trading days calendar, earnings dates
└── Observed: Price, Volume, Technical indicators, Macro
```

### Model Architecture

```
Temporal Fusion Transformer:
├── Input Embedding
│   ├── Static covariate encoders
│   ├── Known future input encoders
│   └── Observed input encoders
├── Variable Selection Networks (per time step)
├── LSTM Encoder-Decoder
├── Temporal Self-Attention (interpretable)
├── Position-wise Feed-Forward
└── Quantile Output (10%, 50%, 90%)
```

### Key Metrics

- **Forecasting:** MAE, RMSE, Quantile Loss, Coverage
- **Strategy:** Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio
- **Comparison:** vs Buy&Hold, 60/40, Risk Parity

### Dependencies

```python
pytorch-forecasting>=0.10.0
pytorch-lightning>=2.0.0
darts>=0.25.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **TFT модель** с multi-horizon forecasting для портфеля активов
2. **Interpretability analysis** — какие факторы важны на каких горизонтах
3. **Allocation strategy** с динамическим переключением акции/облигации/кэш
4. **Backtesting results** с Sharpe > 1.0 (цель), сравнение с baselines

## References

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) (Google, 2021)
- [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)
- [Darts Library](https://unit8co.github.io/darts/)

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: Attention механизмов, LSTM, Quantile regression, Portfolio management
