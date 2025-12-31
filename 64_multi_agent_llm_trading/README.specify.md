# Chapter 64: Multi-Agent LLM Trading

## Описание

Мульти-агентные системы на базе LLM для симуляции торговых решений.

## Техническое задание

### Цели
1. Изучить теоретические основы метода
2. Реализовать базовую версию на Python
3. Создать оптимизированную версию на Rust
4. Протестировать на финансовых данных
5. Провести бэктестинг торговой стратегии

### Ключевые компоненты
- Теоретическое описание метода
- Python реализация с PyTorch
- Rust реализация для production
- Jupyter notebooks с примерами
- Бэктестинг framework

### Метрики
- Accuracy / F1-score для классификации
- MSE / MAE для регрессии
- Sharpe Ratio / Sortino Ratio для стратегий
- Maximum Drawdown
- Сравнение с baseline моделями

## Научные работы

1. **TradingAgents: Multi-Agents LLM Financial Trading Framework**
   - URL: https://arxiv.org/abs/2412.20138
   - Год: 2024

2. **FinCon: A Synthesized LLM Multi-Agent System for Financial Agents**
   - URL: https://arxiv.org/abs/2407.06567
   - Год: 2024

## Данные
- Yahoo Finance / yfinance
- Binance API для криптовалют
- LOBSTER для order book data
- Kaggle финансовые датасеты

## Реализация

### Python
- PyTorch / TensorFlow
- NumPy, Pandas
- scikit-learn
- Backtrader / Zipline

### Rust
- ndarray
- polars
- burn / candle

## Структура
```
64_multi_agent_llm_trading/
├── README.specify.md
├── README.md
├── docs/
│   └── ru/
│       └── theory.md
├── python/
│   ├── model.py
│   ├── train.py
│   ├── backtest.py
│   └── notebooks/
│       └── example.ipynb
└── rust/
    └── src/
        └── lib.rs
```
