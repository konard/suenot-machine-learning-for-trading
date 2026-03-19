# Chapter 137: Gated SSM

## Описание

Gating mechanisms в state space models для selective processing.

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
- Примеры на Rust (examples/)
- Бэктестинг framework

### Метрики
- Accuracy / F1-score для классификации
- MSE / MAE для регрессии
- Sharpe Ratio / Sortino Ratio для стратегий
- Maximum Drawdown
- Сравнение с baseline моделями

## Научные работы

1. **Gated State Spaces**
   - URL: https://arxiv.org/abs/2206.13947
   - Год: 2022

## Данные
- Yahoo Finance / yfinance (акции)
- Bybit API (криптовалюты: BTCUSDT, ETHUSDT)

## Реализация

### Python
- PyTorch (DiagonalSSMLayer, GatedSSMBlock, GatedSSMModel)
- NumPy, Pandas
- requests (Bybit API)
- yfinance (фондовый рынок)

### Rust
- tokio (async runtime)
- reqwest (HTTP client для Bybit API)
- serde / serde_json
- statrs (статистика)

## Структура
```
137_gated_ssm/
├── README.md                  # Полная документация (English)
├── README.ru.md               # Русский перевод
├── README.specify.md          # Спецификация (этот файл)
├── readme.simple.md           # Упрощённое объяснение (English)
├── readme.simple.ru.md        # Упрощённое объяснение (Russian)
├── Cargo.toml                 # Конфигурация Rust
├── python/
│   ├── __init__.py
│   ├── model.py               # PyTorch модель Gated SSM
│   ├── data_loader.py         # Загрузка данных (Bybit + yfinance)
│   ├── backtest.py            # Бэктестинг
│   └── requirements.txt       # Python зависимости
├── src/
│   ├── lib.rs                 # Rust библиотека
│   ├── model/
│   │   ├── mod.rs
│   │   ├── ssm.rs             # Diagonal SSM
│   │   └── gated_ssm.rs       # Gated SSM block + model
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs           # Bybit API client
│   │   └── features.rs        # Feature engineering
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── signals.rs         # Торговые сигналы
│   │   └── strategy.rs        # Торговая стратегия
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs          # Движок бэктестинга
└── examples/
    ├── basic_gated_ssm.rs     # Базовый пример
    ├── crypto_trading.rs      # Торговля криптовалютами
    └── backtest_strategy.rs   # Бэктестинг стратегии
```
