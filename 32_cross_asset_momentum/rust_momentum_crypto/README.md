# Momentum Crypto

Библиотека на Rust для реализации кросс-активного моментума на криптовалютном рынке с использованием данных биржи Bybit.

## Возможности

- Загрузка исторических данных с Bybit API
- Time-series momentum (абсолютный моментум)
- Cross-sectional momentum (относительный моментум)
- Dual momentum (комбинация обоих подходов)
- Расчёт весов портфеля (equal weight, risk parity, volatility targeting)
- Бэктестинг с учётом комиссий и slippage
- Полный набор метрик производительности (Sharpe, Sortino, Max Drawdown, etc.)

## Структура проекта

```
rust_momentum_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── main.rs             # CLI интерфейс
│   ├── data/
│   │   ├── mod.rs          # Модуль данных
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── types.rs        # Типы данных (Candle, PriceSeries, etc.)
│   ├── momentum/
│   │   ├── mod.rs          # Модуль моментума
│   │   ├── timeseries.rs   # Time-series momentum
│   │   ├── crosssection.rs # Cross-sectional momentum
│   │   └── dual.rs         # Dual momentum
│   ├── strategy/
│   │   ├── mod.rs          # Модуль стратегии
│   │   ├── signals.rs      # Генерация сигналов
│   │   └── weights.rs      # Расчёт весов
│   ├── backtest/
│   │   ├── mod.rs          # Модуль бэктестинга
│   │   ├── engine.rs       # Движок бэктеста
│   │   └── metrics.rs      # Метрики производительности
│   └── utils/
│       ├── mod.rs          # Утилиты
│       └── config.rs       # Конфигурация
└── examples/
    ├── fetch_prices.rs     # Загрузка данных с Bybit
    ├── calc_momentum.rs    # Расчёт моментума
    ├── run_strategy.rs     # Запуск стратегии
    └── backtest.rs         # Полный бэктест
```

## Установка

Добавьте в `Cargo.toml`:

```toml
[dependencies]
momentum-crypto = { path = "path/to/rust_momentum_crypto" }
```

## Быстрый старт

### CLI

```bash
# Получить текущие цены
cargo run -- prices

# Рассчитать моментум
cargo run -- momentum --lookback 30 --top 5

# Сгенерировать сигналы
cargo run -- signals --top-n 3

# Запустить бэктест
cargo run -- backtest --days 90 --capital 10000

# Создать конфигурацию
cargo run -- config --output config.json --preset default
```

### Примеры

```bash
# Загрузка цен
cargo run --example fetch_prices

# Расчёт моментума
cargo run --example calc_momentum

# Запуск стратегии
cargo run --example run_strategy

# Бэктестинг
cargo run --example backtest
```

### Как библиотека

```rust
use momentum_crypto::{
    data::{BybitClient, get_momentum_universe},
    momentum::{DualMomentum, DualMomentumConfig},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Создаём клиент Bybit
    let client = BybitClient::new();

    // Загружаем данные
    let mut price_data = HashMap::new();
    for symbol in get_momentum_universe().iter().take(5) {
        let series = client.get_klines(symbol, "D", None, None, Some(60)).await?;
        price_data.insert(symbol.to_string(), series);
    }

    // Создаём стратегию dual momentum
    let config = DualMomentumConfig::crypto();
    let strategy = DualMomentum::new(config);

    // Анализируем активы
    let analysis = strategy.analyze(&price_data)?;

    for result in &analysis {
        if result.selected {
            println!(
                "Выбран {} с моментумом {:.2}%, вес {:.1}%",
                result.symbol,
                result.ts_momentum * 100.0,
                result.weight * 100.0
            );
        }
    }

    Ok(())
}
```

## Модули

### data

Модуль для работы с данными:

- `BybitClient` - клиент для Bybit API v5
- `Candle` - структура OHLCV свечи
- `PriceSeries` - временной ряд цен
- `Portfolio` - портфель с весами
- `Signal` - торговый сигнал (Long, Short, Neutral, Cash)

### momentum

Модуль для расчёта моментума:

- `TimeSeriesMomentum` - абсолютный моментум (актив vs сам себя)
- `CrossSectionalMomentum` - относительный моментум (актив vs другие)
- `DualMomentum` - комбинация TSM + CSM

### strategy

Модуль стратегии:

- `SignalGenerator` - генерация торговых сигналов
- `WeightCalculator` - расчёт весов портфеля

### backtest

Модуль бэктестинга:

- `BacktestEngine` - движок симуляции
- Метрики: CAGR, Sharpe, Sortino, Max Drawdown, Calmar, VaR, CVaR

### utils

Утилиты:

- `StrategyConfig` - конфигурация стратегии (JSON сериализация)

## Bybit API

Библиотека использует публичный API Bybit v5:

- `GET /v5/market/kline` - исторические свечи
- `GET /v5/market/tickers` - текущие цены

Ограничения:
- Rate limit: 120 запросов в минуту
- Максимум 1000 свечей за запрос
- Доступ без аутентификации (только публичные данные)

## Конфигурация

Пример файла конфигурации:

```json
{
  "name": "Crypto Dual Momentum",
  "description": "Cross-asset momentum strategy for cryptocurrencies",
  "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
  "momentum": {
    "ts_lookback": 30,
    "cs_lookback": 30,
    "top_n": 3,
    "skip_period": 1,
    "multi_period": false
  },
  "portfolio": {
    "initial_capital": 10000.0,
    "target_volatility": 0.30,
    "max_weight": 0.40,
    "use_risk_parity": true,
    "max_leverage": 1.0
  },
  "trading": {
    "rebalance_period": 7,
    "rebalance_threshold": 0.05,
    "commission": 0.001,
    "slippage": 0.0005
  }
}
```

## Тестирование

```bash
# Запуск тестов
cargo test

# Тесты с выводом
cargo test -- --nocapture

# Конкретный тест
cargo test test_dual_momentum
```

## Предупреждение

Эта библиотека предназначена для образовательных целей. Криптовалюты являются высокорисковыми активами. Не инвестируйте больше, чем готовы потерять.

## Лицензия

MIT
