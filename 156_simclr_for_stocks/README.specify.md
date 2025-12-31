# Chapter 156: SimCLR для Stocks

## Описание

Simple Contrastive Learning для представлений финансовых временных рядов.

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

1. **A Simple Framework for Contrastive Learning of Visual Representations**
   - URL: https://arxiv.org/abs/2002.05709
   - Год: 2020

2. **Contrastive Learning Framework for Bitcoin Crash Prediction**
   - URL: https://www.mdpi.com/2571-905X/7/2/25
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
156_simclr_for_stocks/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
