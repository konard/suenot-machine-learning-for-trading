# Alpha Factors Library (Rust)

Библиотека для расчёта альфа-факторов и технических индикаторов на криптовалютных данных биржи Bybit.

## Возможности

- **API клиент Bybit** — получение OHLCV данных, тикеров, стакана заявок
- **Трендовые индикаторы** — SMA, EMA, MACD, Bollinger Bands, Keltner Channel
- **Индикаторы моментума** — RSI, Stochastic, ROC, CCI, ADX, Williams %R
- **Индикаторы объёма** — OBV, VWAP, MFI, CMF, A/D Line
- **Индикаторы волатильности** — ATR, Historical Volatility, Donchian Channel
- **Альфа-факторы** — реализации из статьи WorldQuant "101 Formulaic Alphas"

## Установка

Добавьте в `Cargo.toml`:

```toml
[dependencies]
alpha_factors = { path = "path/to/alpha_factors" }
tokio = { version = "1.0", features = ["full"] }
```

## Быстрый старт

### Получение данных с Bybit

```rust
use alpha_factors::{BybitClient, api::Interval};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BybitClient::new();

    // Получаем последние 100 часовых свечей BTC
    let klines = client
        .get_klines_with_interval("BTCUSDT", Interval::Hour1, 100)
        .await?;

    println!("Последняя цена: {}", klines.last().unwrap().close);
    Ok(())
}
```

### Расчёт индикаторов

```rust
use alpha_factors::{factors, data::kline::KlineVec};

// Извлекаем цены закрытия
let closes = klines.closes();

// Рассчитываем индикаторы
let sma_20 = factors::sma(&closes, 20);
let rsi_14 = factors::rsi(&closes, 14);
let macd = factors::macd(&closes, 12, 26, 9);

println!("SMA(20): {:.2}", sma_20.last().unwrap());
println!("RSI(14): {:.2}", rsi_14.last().unwrap());
println!("MACD: {:.4}", macd.macd_line.last().unwrap());
```

### Генерация сигналов

```rust
use alpha_factors::factors::{FactorCalculator, Signal};

// Создаём калькулятор факторов
let calc = FactorCalculator::from_klines(&klines);
let factors = calc.calculate_all();

// Получаем комплексный сигнал
if let Some(snapshot) = factors.last_values() {
    let current_price = klines.last().unwrap().close;
    let signal = snapshot.generate_signal(current_price);

    match signal {
        Signal::StrongBuy => println!("Сильная покупка!"),
        Signal::Buy => println!("Покупка"),
        Signal::Neutral => println!("Держать"),
        Signal::Sell => println!("Продажа"),
        Signal::StrongSell => println!("Сильная продажа!"),
    }
}
```

## Структура проекта

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Точка входа библиотеки
│   ├── api/                # API клиент Bybit
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   ├── error.rs        # Типы ошибок
│   │   └── response.rs     # Структуры ответов
│   ├── data/               # Структуры данных
│   │   ├── mod.rs
│   │   ├── kline.rs        # OHLCV свечи
│   │   ├── ticker.rs       # Тикеры
│   │   └── orderbook.rs    # Стакан заявок
│   └── factors/            # Индикаторы и факторы
│       ├── mod.rs
│       ├── trend.rs        # Трендовые индикаторы
│       ├── momentum.rs     # Индикаторы моментума
│       ├── volume.rs       # Индикаторы объёма
│       ├── volatility.rs   # Индикаторы волатильности
│       ├── alpha.rs        # Альфа-факторы
│       └── utils.rs        # Вспомогательные функции
└── examples/
    ├── fetch_klines.rs         # Получение данных
    ├── calculate_factors.rs    # Расчёт индикаторов
    └── momentum_strategy.rs    # Пример стратегии
```

## API Bybit

### Поддерживаемые категории

- `Category::Linear` — бессрочные контракты (USDT)
- `Category::Inverse` — инверсные контракты
- `Category::Spot` — спотовый рынок
- `Category::Option` — опционы

### Поддерживаемые интервалы

| Интервал | Код |
|----------|-----|
| 1 минута | `Interval::Min1` |
| 5 минут | `Interval::Min5` |
| 15 минут | `Interval::Min15` |
| 1 час | `Interval::Hour1` |
| 4 часа | `Interval::Hour4` |
| 1 день | `Interval::Day1` |
| 1 неделя | `Interval::Week1` |

### Примеры запросов

```rust
// Тикер
let ticker = client.get_ticker("ETHUSDT").await?;
println!("Спред: {:.4}%", ticker.spread_percent());

// Стакан заявок
let orderbook = client.get_orderbook("BTCUSDT", 25).await?;
println!("Дисбаланс: {:.2}", orderbook.imbalance(10));

// Все тикеры
let tickers = client.get_all_tickers().await?;
```

## Индикаторы

### Трендовые

```rust
// Скользящие средние
let sma = factors::sma(&closes, 20);
let ema = factors::ema(&closes, 20);
let wma = factors::wma(&closes, 20);

// MACD
let macd = factors::macd(&closes, 12, 26, 9);

// Bollinger Bands
let bb = factors::bollinger_bands(&closes, 20, 2.0);
```

### Моментум

```rust
// RSI
let rsi = factors::rsi(&closes, 14);

// Stochastic
let stoch = factors::stochastic(&highs, &lows, &closes, 14, 3);

// ROC
let roc = factors::roc(&closes, 10);

// CCI
let cci = factors::cci(&highs, &lows, &closes, 20);

// ADX
let adx = factors::adx(&highs, &lows, &closes, 14);
```

### Объём

```rust
// OBV
let obv = factors::obv(&closes, &volumes);

// VWAP
let vwap = factors::vwap(&highs, &lows, &closes, &volumes);

// MFI
let mfi = factors::mfi(&highs, &lows, &closes, &volumes, 14);

// CMF
let cmf = factors::cmf(&highs, &lows, &closes, &volumes, 20);
```

### Волатильность

```rust
// ATR
let atr = factors::atr(&highs, &lows, &closes, 14);

// Historical Volatility
let hv = factors::historical_volatility(&closes, 20, 252.0);

// Donchian Channel
let (upper, middle, lower) = factors::donchian_channel(&highs, &lows, 20);
```

### Альфа-факторы

```rust
// WorldQuant 101 Alphas
let alpha_003 = factors::alpha_003(&opens, &volumes, 10);
let alpha_012 = factors::alpha_012(&closes, &volumes);

// Пользовательские факторы
let momentum = factors::momentum_factor(&closes, 20);
let mean_rev = factors::mean_reversion_factor(&closes, 20);
let vol_spike = factors::volume_spike_factor(&volumes, 20);
```

## Запуск примеров

```bash
# Получение данных
cargo run --example fetch_klines

# Расчёт индикаторов
cargo run --example calculate_factors

# Бэктестинг стратегии
cargo run --example momentum_strategy
```

## Тестирование

```bash
cargo test
```

## Лицензия

MIT

## Ссылки

- [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/intro)
- [WorldQuant 101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf)
- [TA-Lib](https://ta-lib.org/)
