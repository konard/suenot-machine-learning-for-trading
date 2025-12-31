# Chapter 211: NAS для Trading

## Описание

Neural Architecture Search для автоматического дизайна торговых моделей.

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

1. **Neural Architecture Search: A Survey**
   - URL: https://arxiv.org/abs/1808.05377
   - Год: 2018

2. **Neuroevolution NAS for Evolving RNNs in Stock Prediction**
   - URL: https://arxiv.org/abs/2410.17212
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
211_nas_for_trading/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
