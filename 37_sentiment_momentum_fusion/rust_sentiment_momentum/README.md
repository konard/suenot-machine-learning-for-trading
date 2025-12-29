# Rust Sentiment-Momentum Fusion

Модульная реализация стратегии Sentiment-Momentum Fusion на Rust с использованием данных криптовалютной биржи Bybit.

## Описание

Этот проект демонстрирует объединение ценового моментума с анализом настроений для генерации торговых сигналов на криптовалютном рынке.

## Структура проекта

```
rust_sentiment_momentum/
├── Cargo.toml              # Конфигурация проекта
├── README.md               # Этот файл
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── main.rs             # CLI приложение
│   ├── api/                # API клиенты
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API клиент
│   │   └── types.rs        # Типы данных API
│   ├── sentiment/          # Анализ настроений
│   │   ├── mod.rs
│   │   ├── analyzer.rs     # Анализатор настроений
│   │   ├── aggregator.rs   # Агрегатор настроений
│   │   └── mock_data.rs    # Генерация тестовых данных
│   ├── momentum/           # Расчёт моментума
│   │   ├── mod.rs
│   │   ├── price.rs        # Ценовой моментум
│   │   └── sentiment.rs    # Sentiment моментум
│   ├── signals/            # Генерация сигналов
│   │   ├── mod.rs
│   │   ├── fusion.rs       # Fusion стратегия
│   │   └── divergence.rs   # Обнаружение дивергенции
│   └── models/             # Модели данных
│       ├── mod.rs
│       └── types.rs        # Общие типы данных
└── examples/               # Примеры использования
    ├── fetch_bybit_data.rs     # Получение данных с Bybit
    ├── calculate_momentum.rs   # Расчёт моментума
    ├── sentiment_signals.rs    # Генерация sentiment сигналов
    └── full_strategy.rs        # Полная стратегия
```

## Установка и запуск

### Требования

- Rust 1.70+
- Cargo

### Сборка

```bash
cd rust_sentiment_momentum
cargo build --release
```

### Запуск примеров

```bash
# Получение данных с Bybit
cargo run --example fetch_bybit_data

# Расчёт моментума
cargo run --example calculate_momentum

# Sentiment сигналы
cargo run --example sentiment_signals

# Полная стратегия
cargo run --example full_strategy
```

### Запуск CLI

```bash
cargo run -- --help
cargo run -- --symbol BTCUSDT --interval 60
```

## Модули

### API (`src/api/`)

Клиент для работы с Bybit API v5:
- Получение OHLCV (kline) данных
- Получение списка торговых пар
- Поддержка пагинации для исторических данных

### Sentiment (`src/sentiment/`)

Анализ и агрегация настроений:
- Симуляция sentiment-данных (для демонстрации)
- Временное взвешивание (свежие сообщения важнее)
- Взвешивание по вовлечённости (engagement)

### Momentum (`src/momentum/`)

Расчёт различных видов моментума:
- Ценовой моментум (ROC, RSI-like)
- Sentiment моментум
- Множественные временные окна

### Signals (`src/signals/`)

Генерация торговых сигналов:
- Fusion стратегия (объединение цены и sentiment)
- Обнаружение дивергенции
- Настраиваемые веса и пороги

## Пример использования

```rust
use rust_sentiment_momentum::{
    api::BybitClient,
    momentum::{calculate_price_momentum, calculate_sentiment_momentum},
    signals::FusionStrategy,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Получаем данные с Bybit
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "60", None, None, Some(100)).await?;

    // Рассчитываем ценовой моментум
    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let price_momentum = calculate_price_momentum(&prices, 20);

    // Создаём стратегию
    let strategy = FusionStrategy::new(0.6, 0.4);

    // Генерируем сигналы
    // ...

    Ok(())
}
```

## Лицензия

MIT License
