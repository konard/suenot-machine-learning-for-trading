# Chapter 186: VQE Portfolio Optimization

## Описание

Variational Quantum Eigensolver для оптимизации портфеля.

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

1. **Quantum Portfolio Optimization with Expert Analysis**
   - URL: https://arxiv.org/abs/2507.20532
   - Год: 2025

2. **Applications of Quantum Machine Learning for Quantitative Finance**
   - URL: https://arxiv.org/abs/2405.10119
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

### Rust
- ndarray, polars
- burn / candle

## Структура
```
186_vqe_portfolio_optimization/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
