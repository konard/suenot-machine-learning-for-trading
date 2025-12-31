# Chapter 103: Causal Forests для Finance

## Описание

Каузальные леса для оценки гетерогенных эффектов в торговых стратегиях.

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

1. **Generalized Random Forests**
   - URL: https://arxiv.org/abs/1610.01271
   - Год: 2019

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
103_causal_forests_finance/
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
