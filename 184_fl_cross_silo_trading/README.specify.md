# Chapter 184: FL Cross-Silo Trading

## Описание

Cross-silo федеративное обучение между финансовыми институтами.

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

1. **Advances and Open Problems in Federated Learning**
   - URL: https://arxiv.org/abs/1912.04977
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

### Rust
- ndarray, polars
- burn / candle

## Структура
```
184_fl_cross_silo_trading/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
