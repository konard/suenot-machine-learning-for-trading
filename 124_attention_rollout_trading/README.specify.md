# Chapter 124: Attention Rollout Trading

## Описание

Attention Rollout — метод интерпретируемости для моделей трансформеров, который отслеживает поток внимания через слои для объяснения торговых решений.

## Техническое задание

### Цели

1. Изучить теоретические основы Attention Rollout
2. Реализовать базовую версию на Python
3. Создать оптимизированную версию на Rust
4. Протестировать на финансовых данных (акции и криптовалюты)
5. Провести бэктестинг торговых стратегий с анализом внимания

### Ключевые компоненты

- Теоретическое описание метода Attention Rollout
- Реализация на Python с PyTorch
- Реализация на Rust для высокой производительности
- Jupyter notebooks с примерами
- Фреймворк бэктестинга с анализом внимания

### Метрики

- Корректность вычисления rollout матриц
- Sharpe Ratio / Sortino Ratio для торговых стратегий
- Maximum Drawdown
- Сравнение паттернов внимания прибыльных и убыточных сделок
- Время выполнения (Python vs Rust)

## Научные работы

- Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." ACL 2020. https://arxiv.org/abs/2005.00928
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Chefer, H., et al. (2021). "Transformer Interpretability Beyond Attention Visualization." CVPR 2021. https://arxiv.org/abs/2012.09838

## Данные

- Yahoo Finance (yfinance) — данные фондового рынка
- Bybit API — данные криптовалютного рынка
- Исторические данные OHLCV

## Реализация

### Python
- PyTorch — модели трансформеров
- NumPy — численные вычисления
- Pandas — обработка данных
- yfinance — загрузка данных акций
- requests — API запросы к Bybit
- matplotlib/seaborn — визуализация

### Rust
- ndarray — многомерные массивы
- reqwest — HTTP клиент
- serde — сериализация
- tokio — асинхронный runtime

## Структура

```
124_attention_rollout_trading/
├── README.md
├── README.ru.md
├── readme.simple.md
├── readme.simple.ru.md
├── README.specify.md
├── python/
│   ├── requirements.txt
│   ├── __init__.py
│   ├── attention_rollout.py
│   ├── model.py
│   ├── data_loader.py
│   ├── backtest.py
│   ├── visualization.py
│   └── notebooks/
│       └── 01_attention_rollout_trading.ipynb
└── rust/
    ├── Cargo.toml
    ├── src/
    │   └── lib.rs
    └── examples/
        └── trading_example.rs
```
