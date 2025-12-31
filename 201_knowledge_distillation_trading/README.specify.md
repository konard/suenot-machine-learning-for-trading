# Chapter 201: Knowledge Distillation для Trading

## Описание

Дистилляция знаний из больших моделей в компактные для trading.

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

1. **Distilling the Knowledge in a Neural Network**
   - URL: https://arxiv.org/abs/1503.02531
   - Год: 2015

2. **Long-Short Dual-Mode Knowledge Distillation for Asset Pricing**
   - URL: https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2306970
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
201_knowledge_distillation_trading/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
