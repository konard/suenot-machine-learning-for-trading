# ML4T Bybit Examples (Rust)

Модульные примеры работы с криптовалютной биржей Bybit на языке Rust для машинного обучения в трейдинге.

## Структура проекта

```
rust_examples/
├── Cargo.toml              # Конфигурация проекта и зависимости
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── client/             # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── rest.rs         # REST API клиент
│   │   ├── websocket.rs    # WebSocket клиент
│   │   └── types.rs        # Типы данных API
│   ├── data/               # Работа с рыночными данными
│   │   ├── mod.rs
│   │   ├── kline.rs        # Свечные данные (OHLCV)
│   │   └── orderbook.rs    # Стакан заявок
│   ├── indicators/         # Технические индикаторы
│   │   ├── mod.rs
│   │   ├── sma.rs          # Simple Moving Average
│   │   ├── ema.rs          # Exponential Moving Average
│   │   ├── rsi.rs          # Relative Strength Index
│   │   ├── macd.rs         # MACD
│   │   └── bollinger.rs    # Bollinger Bands
│   ├── strategies/         # Торговые стратегии
│   │   ├── mod.rs
│   │   ├── base.rs         # Базовый трейт стратегии
│   │   ├── sma_cross.rs    # Пересечение SMA
│   │   └── rsi_oversold.rs # RSI перепроданность
│   └── backtest/           # Бэктестинг
│       ├── mod.rs
│       ├── engine.rs       # Движок бэктеста
│       └── metrics.rs      # Метрики производительности
└── examples/               # Примеры использования
    ├── fetch_klines.rs     # Получение исторических данных
    ├── simple_sma_strategy.rs  # Простая SMA стратегия
    ├── rsi_strategy.rs     # RSI стратегия
    ├── backtest_example.rs # Пример бэктеста
    └── live_ticker.rs      # Подписка на live данные
```

## Установка и запуск

### Предварительные требования

- Rust 1.70+ (установите через [rustup](https://rustup.rs/))
- Аккаунт на Bybit (опционально, для API ключей)

### Установка

```bash
cd rust_examples
cargo build
```

### Запуск примеров

```bash
# Получение исторических свечей
cargo run --example fetch_klines

# Простая SMA стратегия
cargo run --example simple_sma_strategy

# RSI стратегия
cargo run --example rsi_strategy

# Бэктест стратегии
cargo run --example backtest_example

# Live данные через WebSocket
cargo run --example live_ticker
```

## Настройка API ключей

Для работы с приватными эндпоинтами (ордера, баланс) создайте файл `.env`:

```env
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=true
```

**ВАЖНО:** Для обучения используйте тестовую сеть (testnet)!

## Модули

### Client (`src/client/`)

Клиент для работы с Bybit API:

```rust
use ml4t_bybit::client::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BybitClient::new_testnet();

    // Получить текущую цену
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("BTC: ${}", ticker.last_price);

    Ok(())
}
```

### Data (`src/data/`)

Структуры для работы с рыночными данными:

```rust
use ml4t_bybit::data::{Kline, Interval};

// Свеча (OHLCV)
let kline = Kline {
    timestamp: 1704067200000,
    open: 42000.0,
    high: 42500.0,
    low: 41800.0,
    close: 42300.0,
    volume: 1000.0,
};
```

### Indicators (`src/indicators/`)

Технические индикаторы:

```rust
use ml4t_bybit::indicators::{SMA, RSI, MACD};

let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];

// Simple Moving Average
let sma = SMA::new(3);
let sma_values = sma.calculate(&prices);

// RSI
let rsi = RSI::new(14);
let rsi_values = rsi.calculate(&prices);

// MACD
let macd = MACD::new(12, 26, 9);
let (macd_line, signal_line, histogram) = macd.calculate(&prices);
```

### Strategies (`src/strategies/`)

Торговые стратегии:

```rust
use ml4t_bybit::strategies::{Strategy, SmaCrossStrategy, Signal};

let strategy = SmaCrossStrategy::new(10, 20);
let signal = strategy.generate_signal(&klines);

match signal {
    Signal::Buy => println!("Покупаем!"),
    Signal::Sell => println!("Продаём!"),
    Signal::Hold => println!("Ждём..."),
}
```

### Backtest (`src/backtest/`)

Бэктестинг стратегий:

```rust
use ml4t_bybit::backtest::{BacktestEngine, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10000.0,
    commission: 0.001, // 0.1%
    slippage: 0.0005,  // 0.05%
};

let engine = BacktestEngine::new(config);
let results = engine.run(&strategy, &historical_data);

println!("Общая доходность: {:.2}%", results.total_return * 100.0);
println!("Коэффициент Шарпа: {:.2}", results.sharpe_ratio);
println!("Максимальная просадка: {:.2}%", results.max_drawdown * 100.0);
```

## API Reference

### Публичные эндпоинты (без аутентификации)

| Метод | Описание |
|-------|----------|
| `get_ticker(symbol)` | Текущая цена и статистика |
| `get_klines(symbol, interval, limit)` | Исторические свечи |
| `get_orderbook(symbol, limit)` | Стакан заявок |
| `get_trades(symbol, limit)` | Последние сделки |

### Приватные эндпоинты (требуют API ключ)

| Метод | Описание |
|-------|----------|
| `get_balance()` | Баланс аккаунта |
| `place_order(...)` | Создать ордер |
| `cancel_order(order_id)` | Отменить ордер |
| `get_positions()` | Открытые позиции |

## Примеры стратегий

### 1. SMA Crossover

Классическая стратегия пересечения скользящих средних:
- **Покупка:** когда быстрая SMA пересекает медленную снизу вверх
- **Продажа:** когда быстрая SMA пересекает медленную сверху вниз

### 2. RSI Oversold/Overbought

Стратегия на основе индекса относительной силы:
- **Покупка:** RSI < 30 (перепроданность)
- **Продажа:** RSI > 70 (перекупленность)

## Важные замечания

1. **Используйте тестовую сеть!** Не торгуйте на реальные деньги без тщательного тестирования.

2. **Это учебные примеры.** Стратегии приведены для демонстрации и не гарантируют прибыль.

3. **Управление рисками.** Всегда используйте стоп-лоссы и не рискуйте больше, чем можете потерять.

4. **Latency.** Для высокочастотной торговли требуются дополнительные оптимизации.

## Полезные ссылки

- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)
- [Bybit Testnet](https://testnet.bybit.com/)
- [Rust Book](https://doc.rust-lang.org/book/)

## Лицензия

MIT License
