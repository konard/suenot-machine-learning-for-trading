# Rust Gradient Boosting Machine for Cryptocurrency Trading

Реализация градиентного бустинга на Rust для торговли криптовалютой с использованием данных биржи Bybit.

## Структура проекта

```
rust_gbm/
├── Cargo.toml              # Зависимости и конфигурация
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── main.rs             # Точка входа с демонстрацией
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── types.rs        # Типы данных (Candle, OrderBook, etc.)
│   ├── features/
│   │   ├── mod.rs
│   │   ├── technical.rs    # Технические индикаторы
│   │   └── engineering.rs  # Инженерия признаков
│   ├── models/
│   │   ├── mod.rs
│   │   └── gbm.rs          # Gradient Boosting Machine
│   └── strategies/
│       ├── mod.rs
│       └── long_short.rs   # Long-Short стратегия
└── examples/
    ├── fetch_data.rs       # Пример загрузки данных
    ├── train_model.rs      # Пример обучения модели
    └── backtest.rs         # Пример бэктестинга
```

## Модули

### Data (`src/data/`)

Загрузка рыночных данных с биржи Bybit:

```rust
use rust_gbm::data::{BybitClient, Interval};

let client = BybitClient::new();

// Получить свечи
let candles = client
    .get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)
    .await?;

// Получить книгу ордеров
let orderbook = client.get_orderbook("BTCUSDT", Some(10)).await?;

// Получить последние сделки
let trades = client.get_recent_trades("BTCUSDT", Some(100)).await?;
```

### Features (`src/features/`)

Технические индикаторы и инженерия признаков:

```rust
use rust_gbm::features::{FeatureEngineer, FeatureConfig};
use rust_gbm::features::technical::{sma, rsi, macd, bollinger_bands};

// Создать инженер признаков с настройками по умолчанию
let engineer = FeatureEngineer::new();

// Или с пользовательскими настройками
let config = FeatureConfig {
    ma_periods: vec![5, 10, 20, 50, 100],
    rsi_period: 14,
    target_period: 1,  // Предсказываем 1 свечу вперёд
    ..Default::default()
};
let engineer = FeatureEngineer::with_config(config);

// Построить датасет
let dataset = engineer.build_clean_features(&candles);
```

Доступные индикаторы:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- OBV (On-Balance Volume)
- ROC (Rate of Change)
- CCI (Commodity Channel Index)
- Williams %R
- MFI (Money Flow Index)

### Models (`src/models/`)

Gradient Boosting Machine:

```rust
use rust_gbm::models::{GbmRegressor, GbmClassifier, GbmParams, time_series_cv};

// Настройка параметров
let params = GbmParams {
    n_estimators: 100,
    max_depth: 4,
    learning_rate: 0.1,
    min_samples_split: 10,
    min_samples_leaf: 5,
    subsample: 0.8,
};

// Регрессор (предсказывает доходность)
let mut regressor = GbmRegressor::with_params(params.clone());
regressor.fit(&train_dataset)?;
let metrics = regressor.evaluate(&test_dataset)?;

// Классификатор (предсказывает направление)
let mut classifier = GbmClassifier::with_params(params);
classifier.fit(&train_dataset)?;

// Кросс-валидация временных рядов
let cv_results = time_series_cv(&dataset, &params, 5)?;
```

### Strategies (`src/strategies/`)

Long-Short торговая стратегия:

```rust
use rust_gbm::strategies::{LongShortStrategy, StrategyConfig, print_backtest_summary};

let config = StrategyConfig {
    long_threshold: 0.1,      // Порог для покупки (%)
    short_threshold: 0.1,     // Порог для продажи (%)
    initial_capital: 10000.0, // Начальный капитал
    position_size: 0.5,       // Размер позиции (50% капитала)
    fee_rate: 0.001,          // Комиссия (0.1%)
    max_positions: 1,
};

let mut strategy = LongShortStrategy::with_config(config);
strategy.set_model(trained_model);

// Запустить бэктест
let metrics = strategy.backtest(&test_dataset, &prices)?;
print_backtest_summary(&metrics);
```

## Запуск

### Основная программа

```bash
cargo run --release
```

### Примеры

```bash
# Загрузка данных
cargo run --example fetch_data

# Обучение модели
cargo run --example train_model

# Бэктестинг
cargo run --example backtest
```

### Тесты

```bash
cargo test
```

## Метрики

### Метрики модели
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (коэффициент детерминации)
- MAE (Mean Absolute Error)
- Directional Accuracy (точность направления)

### Метрики стратегии
- Total Return (общая доходность)
- Annualized Return (годовая доходность)
- Maximum Drawdown (максимальная просадка)
- Sharpe Ratio (коэффициент Шарпа)
- Win Rate (процент прибыльных сделок)
- Profit Factor (фактор прибыли)

## Зависимости

- `tokio` - асинхронный рантайм
- `reqwest` - HTTP клиент
- `serde` - сериализация
- `smartcore` - машинное обучение (GBM)
- `ndarray` - работа с массивами
- `chrono` - дата/время
- `tracing` - логирование

## Примечания

1. **API Bybit**: Используется публичный API без аутентификации (только чтение данных)
2. **Тестнет**: Для тестирования можно использовать `BybitClient::testnet()`
3. **Rate Limits**: Соблюдайте лимиты API (встроена задержка при пагинации)

## Лицензия

MIT
