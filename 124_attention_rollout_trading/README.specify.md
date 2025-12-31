# Chapter 124: Attention Rollout для Trading

## Описание

Агрегация attention через слои для глобального понимания модели.

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

1. **Quantifying Attention Flow in Transformers**
   - URL: https://arxiv.org/abs/2005.00928
   - Год: 2020

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
124_attention_rollout_trading/
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
