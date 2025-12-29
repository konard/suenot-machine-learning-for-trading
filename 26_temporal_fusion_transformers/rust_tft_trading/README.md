# TFT Trading (Rust)

Реализация Temporal Fusion Transformer на Rust для прогнозирования криптовалют с данными биржи Bybit.

## Особенности

- Полная реализация архитектуры TFT
  - Gated Residual Networks (GRN)
  - Variable Selection Networks (VSN)
  - Interpretable Multi-Head Attention
  - Quantile Forecasting
- Интеграция с Bybit API
  - Получение исторических свечей (OHLCV)
  - Стакан заявок и тикеры
  - Поддержка пагинации для больших объемов данных
- Feature Engineering
  - Технические индикаторы (RSI, MACD, Bollinger Bands, ATR)
  - Временные признаки (час, день недели)
  - Автоматическая нормализация
- Торговая стратегия
  - Генерация сигналов на основе quantile прогнозов
  - Бэктестинг с комиссиями и slippage
  - Расчет метрик (Sharpe, Sortino, Max Drawdown)

## Структура проекта

```
rust_tft_trading/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный модуль
│   ├── api/                # Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   └── types.rs        # Типы данных
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Загрузчик данных
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset для TFT
│   ├── model/              # Архитектура TFT
│   │   ├── mod.rs
│   │   ├── config.rs       # Конфигурация
│   │   ├── grn.rs          # Gated Residual Network
│   │   ├── vsn.rs          # Variable Selection
│   │   ├── attention.rs    # Temporal Attention
│   │   ├── losses.rs       # Quantile Loss
│   │   └── tft.rs          # Полная модель
│   ├── training/           # Обучение
│   │   ├── mod.rs
│   │   └── trainer.rs      # Training loop
│   └── strategy/           # Торговля
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Бэктестинг
└── examples/
    ├── fetch_data.rs       # Загрузка данных
    ├── train_model.rs      # Обучение модели
    ├── predict.rs          # Прогнозирование
    └── backtest.rs         # Бэктестинг
```

## Установка

```bash
cd rust_tft_trading
cargo build --release
```

## Примеры использования

### Загрузка данных с Bybit

```bash
cargo run --example fetch_data
```

Этот пример демонстрирует:
- Получение топовых торговых пар по объему
- Загрузку часовых свечей
- Анализ стакана заявок
- Расчет базовой статистики

### Обучение модели

```bash
cargo run --example train_model
```

Полный пайплайн обучения TFT:
- Загрузка данных с Bybit
- Подготовка признаков
- Создание dataset
- Обучение модели
- Оценка на тестовых данных

### Прогнозирование

```bash
cargo run --example predict
```

Генерация прогнозов:
- Загрузка последних данных
- Quantile прогнозы на 24 часа
- Интерпретация важности признаков
- Генерация торгового сигнала

### Бэктестинг

```bash
cargo run --example backtest
```

Бэктестинг стратегии:
- Обучение на исторических данных
- Генерация сигналов
- Симуляция торговли
- Расчет метрик (Sharpe, Max Drawdown и др.)

## API Reference

### BybitClient

```rust
use tft_trading::BybitClient;

let client = BybitClient::new();

// Получить свечи
let klines = client.get_klines("BTCUSDT", "1h", 100).await?;

// Получить стакан
let orderbook = client.get_orderbook("BTCUSDT", 10).await?;

// Получить тикер
let ticker = client.get_ticker("BTCUSDT").await?;
```

### DataLoader

```rust
use tft_trading::{DataLoader, DataLoaderConfig};

let config = DataLoaderConfig {
    symbol: "BTCUSDT".to_string(),
    interval: "1h".to_string(),
    num_candles: 1000,
    encoder_length: 168,
    prediction_length: 24,
    ..Default::default()
};

let loader = DataLoader::with_config(config);
let dataset = loader.load_from_bybit().await?;
```

### TFTModel

```rust
use tft_trading::{TFTModel, TFTConfig};

let config = TFTConfig {
    hidden_size: 64,
    num_attention_heads: 4,
    encoder_length: 168,
    prediction_length: 24,
    quantiles: vec![0.1, 0.5, 0.9],
    ..Default::default()
};

let mut model = TFTModel::new(config);

// Forward pass
let prediction = model.forward(&sample);

// Получить медиану
let median = prediction.median();

// Получить интервал
let lower = prediction.lower();
let upper = prediction.upper();
```

### Backtester

```rust
use tft_trading::{Backtester, BacktestConfig, SignalGenerator};

let config = BacktestConfig {
    initial_capital: 10000.0,
    commission: 0.001,
    slippage: 0.0005,
    ..Default::default()
};

let signal_gen = SignalGenerator::default();
let mut backtester = Backtester::new(config, signal_gen);

let result = backtester.run(&predictions, &prices, &timestamps);
println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
```

## Технические индикаторы

Модуль `FeatureExtractor` вычисляет следующие индикаторы:

| Индикатор | Описание |
|-----------|----------|
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | Upper, Middle, Lower bands |
| ATR | Average True Range |
| Returns | Процентное изменение |
| Log Returns | Логарифмические returns |
| Volatility | Realized volatility |

## Зависимости

- `tokio` — Async runtime
- `reqwest` — HTTP клиент
- `ndarray` — Численные вычисления
- `serde` — Сериализация
- `chrono` — Работа с временем
- `thiserror` — Error handling

## Ограничения

Эта реализация предназначена для образовательных целей и демонстрации концепций. Для production использования рекомендуется:

1. Использовать `tch-rs` (PyTorch bindings) для GPU ускорения
2. Добавить полноценный градиентный спуск
3. Реализовать сохранение/загрузку весов модели
4. Добавить более сложные стратегии управления рисками

## Лицензия

MIT
