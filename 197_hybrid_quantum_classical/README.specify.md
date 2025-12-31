# Chapter 197: Hybrid Quantum-Classical

## Описание

Гибридные квантово-классические алгоритмы для NISQ устройств.

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

1. **Quantum Machine Learning and Optimisation in Finance**
   - URL: https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2375260
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
197_hybrid_quantum_classical/
├── README.specify.md
├── docs/ru/
├── python/
└── rust/src/
```
