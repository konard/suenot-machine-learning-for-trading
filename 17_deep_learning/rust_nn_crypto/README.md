# Rust Neural Network for Cryptocurrency Trading

Модульная реализация нейронных сетей прямого распространения на Rust для торговли криптовалютами с данными биржи Bybit.

## Структура проекта

```
rust_nn_crypto/
├── Cargo.toml              # Зависимости и конфигурация проекта
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── nn/                 # Нейронная сеть
│   │   ├── mod.rs
│   │   ├── activation.rs   # Функции активации (ReLU, Sigmoid, Tanh, LeakyReLU)
│   │   ├── layer.rs        # Полносвязные слои с dropout
│   │   ├── network.rs      # Полная сеть с обучением
│   │   └── optimizer.rs    # Оптимизаторы (SGD, Adam)
│   ├── data/               # Работа с данными
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   ├── ohlcv.rs        # Структуры OHLCV данных
│   │   └── normalize.rs    # Нормализация данных
│   ├── features/           # Извлечение признаков
│   │   ├── mod.rs
│   │   ├── indicators.rs   # Технические индикаторы
│   │   └── engine.rs       # Движок генерации признаков
│   ├── strategy/           # Торговые стратегии
│   │   ├── mod.rs
│   │   ├── signals.rs      # Генерация торговых сигналов
│   │   ├── position.rs     # Управление позициями
│   │   └── trading.rs      # Торговая стратегия
│   ├── backtest/           # Бэктестинг
│   │   ├── mod.rs
│   │   ├── engine.rs       # Движок бэктестинга
│   │   └── metrics.rs      # Метрики производительности
│   └── bin/                # Исполняемые файлы
│       ├── fetch_data.rs   # Загрузка данных с Bybit
│       ├── train.rs        # Обучение модели
│       └── backtest.rs     # Запуск бэктеста
```

## Установка и сборка

```bash
# Клонирование репозитория
cd 17_deep_learning/rust_nn_crypto

# Сборка проекта
cargo build --release

# Запуск тестов
cargo test
```

## Использование

### 1. Загрузка данных с Bybit

```bash
# Загрузить часовые данные BTC за последние 30 дней
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 30

# Загрузить дневные данные ETH за год
cargo run --bin fetch_data -- --symbol ETHUSDT --interval D --days 365 --output eth_daily.csv

# Доступные интервалы: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
```

### 2. Обучение модели

```bash
# Обучить модель на загруженных данных
cargo run --bin train -- --data BTCUSDT_60.csv --epochs 100 --batch 32

# С дополнительными параметрами
cargo run --bin train -- \
    --data BTCUSDT_60.csv \
    --model btc_model.json \
    --epochs 200 \
    --batch 64 \
    --lr 0.001
```

### 3. Бэктестинг

```bash
# Запустить бэктест
cargo run --bin backtest -- --data BTCUSDT_60.csv --model model.json

# С настройками стратегии
cargo run --bin backtest -- \
    --data BTCUSDT_60.csv \
    --model btc_model.json \
    --capital 10000 \
    --stop-loss 0.02 \
    --take-profit 0.04 \
    --no-short
```

## Модули

### Neural Network (`nn`)

Реализация нейронной сети с нуля на Rust:

```rust
use rust_nn_crypto::nn::{NeuralNetwork, NetworkConfig, activation::ActivationType};

// Создание сети для регрессии
let mut model = NeuralNetwork::regression(
    input_size,
    &[64, 32, 16],  // Скрытые слои
    1               // Выход
);

// Или через конфигурацию
let config = NetworkConfig::new(30)
    .add_layer(64, ActivationType::ReLU)
    .add_layer_with_dropout(32, ActivationType::ReLU, 0.2)
    .output_layer(1, ActivationType::Linear);

let mut model = NeuralNetwork::from_config(config);

// Обучение
model.train(&x_train, &y_train, epochs, batch_size, verbose);

// Предсказание
let predictions = model.predict(&x_test);
```

### Data (`data`)

Работа с данными криптовалют:

```rust
use rust_nn_crypto::data::{BybitClient, BybitConfig, OHLCVSeries};

// Создание клиента
let client = BybitClient::public();

// Получение данных
let klines = client.get_klines("BTCUSDT", "60", 1000, None, None)?;

// Сохранение в CSV
klines.save_csv("btc_hourly.csv")?;

// Загрузка из CSV
let series = OHLCVSeries::load_csv("btc_hourly.csv", "BTCUSDT".into(), "60".into())?;
```

### Features (`features`)

Технические индикаторы и инженерия признаков:

```rust
use rust_nn_crypto::features::{FeatureEngine, sma, rsi, macd, bollinger_bands};

// Отдельные индикаторы
let sma_20 = sma(&close_prices, 20);
let rsi_14 = rsi(&close_prices, 14);
let macd_result = macd(&close_prices, 12, 26, 9);

// Автоматическая генерация признаков
let mut engine = FeatureEngine::default_config();
let (features, targets, indices) = engine.extract_features(&series, 1);
```

### Strategy (`strategy`)

Торговые стратегии:

```rust
use rust_nn_crypto::strategy::{TradingStrategy, StrategyConfig, Signal};

let mut strategy = TradingStrategy::new(
    StrategyConfig::default(),
    10000.0  // Начальный капитал
);

// Обработка сигнала
let action = strategy.process_signal(
    Signal::Buy,
    price,
    timestamp,
    "BTCUSDT",
    None
);

// Статистика
println!("Win rate: {:.2}%", strategy.win_rate() * 100.0);
println!("Total PnL: ${:.2}", strategy.total_pnl());
```

### Backtest (`backtest`)

Бэктестинг стратегий:

```rust
use rust_nn_crypto::backtest::{Backtester, BacktestConfig};

let mut backtester = Backtester::new(BacktestConfig::default());

let result = backtester.train_and_backtest(
    &mut model,
    &series,
    strategy_config,
    epochs,
    batch_size
);

result.metrics.print_report();
```

## Технические индикаторы

| Индикатор | Функция | Описание |
|-----------|---------|----------|
| SMA | `sma(data, period)` | Простая скользящая средняя |
| EMA | `ema(data, period)` | Экспоненциальная скользящая средняя |
| RSI | `rsi(data, period)` | Индекс относительной силы |
| MACD | `macd(data, fast, slow, signal)` | Схождение/расхождение скользящих средних |
| Bollinger Bands | `bollinger_bands(data, period, std)` | Полосы Боллинджера |
| ATR | `atr(series, period)` | Средний истинный диапазон |
| Stochastic | `stochastic(series, k, d)` | Стохастик |
| OBV | `obv(series)` | On-Balance Volume |
| MFI | `mfi(series, period)` | Money Flow Index |

## Метрики производительности

- **Total Return** — Общая доходность
- **Annual Return** — Годовая доходность
- **Sharpe Ratio** — Коэффициент Шарпа
- **Sortino Ratio** — Коэффициент Сортино
- **Max Drawdown** — Максимальная просадка
- **Calmar Ratio** — Коэффициент Кальмара
- **Win Rate** — Процент прибыльных сделок
- **Profit Factor** — Фактор прибыли

## Пример полного пайплайна

```rust
use rust_nn_crypto::{
    data::{BybitClient, BybitConfig},
    features::FeatureEngine,
    nn::NeuralNetwork,
    backtest::{Backtester, BacktestConfig},
    strategy::StrategyConfig,
};

fn main() -> anyhow::Result<()> {
    // 1. Загрузка данных
    let client = BybitClient::public();
    let series = client.get_historical_klines(
        "BTCUSDT",
        "60",
        start_time,
        end_time
    )?;

    // 2. Создание модели
    let mut model = NeuralNetwork::regression(30, &[64, 32], 1);

    // 3. Бэктестинг с обучением
    let mut backtester = Backtester::new(BacktestConfig::default());
    let result = backtester.train_and_backtest(
        &mut model,
        &series,
        StrategyConfig::default(),
        100,
        32
    );

    // 4. Вывод результатов
    result.metrics.print_report();

    Ok(())
}
```

## Зависимости

- `ndarray` — Линейная алгебра
- `reqwest` — HTTP клиент для Bybit API
- `serde` — Сериализация/десериализация
- `chrono` — Работа с датами
- `tokio` — Асинхронный runtime

## Лицензия

MIT
