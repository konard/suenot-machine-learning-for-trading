# Options Greeks ML - Rust Implementation

Библиотека на Rust для торговли волатильностью опционов на криптовалютном рынке через Bybit.

## Возможности

- Расчёт цен опционов и греков по модели Блэка-Шоулза
- Расчёт и предсказание реализованной волатильности
- Анализ Volatility Risk Premium (IV vs RV)
- Стратегия торговли страддлами
- Дельта-хеджирование и гамма-скальпинг
- Интеграция с Bybit API

## Структура модулей

```
src/
├── lib.rs              # Главный модуль библиотеки
├── main.rs             # CLI утилита
├── greeks/             # Расчёт греков
│   ├── mod.rs
│   └── black_scholes.rs
├── volatility/         # Волатильность
│   ├── mod.rs
│   ├── realized.rs     # Реализованная волатильность
│   ├── implied.rs      # Подразумеваемая волатильность
│   ├── predictor.rs    # Предсказание волатильности
│   └── vrp.rs          # Volatility Risk Premium
├── strategy/           # Торговые стратегии
│   ├── mod.rs
│   ├── delta_hedger.rs # Дельта-хеджирование
│   ├── straddle.rs     # Стратегия страддлов
│   └── gamma_scalper.rs # Гамма-скальпинг
├── api/                # Интеграция с биржами
│   ├── mod.rs
│   └── bybit.rs        # Bybit API
└── models/             # Модели данных
    └── mod.rs
```

## Быстрый старт

### Установка

```bash
cd rust
cargo build --release
```

### Запуск демо

```bash
# Полная демонстрация
cargo run -- demo

# Только греки
cargo run -- greeks

# Только волатильность
cargo run -- volatility

# Только стратегия
cargo run -- strategy

# Живые данные с Bybit
cargo run -- live
```

### Запуск примеров

```bash
# Расчёт греков
cargo run --example calculate_greeks

# Предсказание волатильности
cargo run --example volatility_forecast

# Бэктест страддлов
cargo run --example straddle_backtest

# Живые данные Bybit
cargo run --example bybit_live
```

## Использование как библиотеки

Добавьте в `Cargo.toml`:

```toml
[dependencies]
options-greeks-ml = { path = "../40_options_ml_greeks/rust" }
```

### Пример: Расчёт греков

```rust
use options_greeks_ml::greeks::BlackScholes;

// Создаём модель для криптоопциона
let bs = BlackScholes::crypto(
    42000.0,  // spot price
    42000.0,  // strike (ATM)
    7.0,      // days to expiry
    0.55,     // IV (55%)
);

// Цены
let call_price = bs.call_price();
let straddle_price = bs.straddle_price();

// Греки
let greeks = bs.straddle_greeks();
println!("Delta: {}", greeks.delta);  // ~0 для ATM страддла
println!("Vega:  {}", greeks.vega);
println!("Theta: {}", greeks.theta);
```

### Пример: Предсказание волатильности

```rust
use options_greeks_ml::volatility::{
    RealizedVolatility,
    VolatilityFeatures,
    VolatilityPredictor,
    VolatilityRiskPremium,
};

// Расчёт RV из цен
let rv = RealizedVolatility::crypto();
let rv_20d = rv.calculate(&prices, Some(20)).unwrap();

// Предсказание
let features = VolatilityFeatures::from_prices(&prices, None, Some(current_iv));
let predictor = VolatilityPredictor::default_weights(7);
let predicted_rv = predictor.predict(&features);

// Торговый сигнал
let vrp = VolatilityRiskPremium::default_crypto();
let signal = vrp.trading_signal(current_iv, predicted_rv, None);

match signal.action {
    VrpAction::SellVolatility => println!("SELL vol - edge: {:.2}%", signal.edge * 100.0),
    VrpAction::BuyVolatility => println!("BUY vol - edge: {:.2}%", signal.edge * 100.0),
    VrpAction::NoTrade => println!("No trade"),
}
```

### Пример: Получение данных с Bybit

```rust
use options_greeks_ml::api::bybit::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new_public();

    // Текущая цена
    let ticker = client.get_ticker("BTCUSDT").await.unwrap();
    println!("BTC: ${}", ticker.last_price);

    // Исторические свечи
    let candles = client.get_klines("BTCUSDT", "D", 30).await.unwrap();

    // Опционная цепочка (если доступна)
    let options = client.get_options_chain("BTC").await.unwrap();
}
```

## Конфигурация

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
# Отредактируйте .env и добавьте ваши API ключи
```

## Тестирование

```bash
# Все тесты
cargo test

# С выводом
cargo test -- --nocapture

# Конкретный модуль
cargo test greeks
cargo test volatility
```

## Предупреждения

1. **Торговля опционами** связана с высоким риском потери капитала
2. **Тестируйте** на testnet перед реальной торговлей
3. **Никогда не рискуйте** больше, чем можете позволить себе потерять
4. Этот код предоставляется **только в образовательных целях**

## Зависимости

- `tokio` - асинхронный рантайм
- `reqwest` - HTTP клиент
- `serde` - сериализация
- `statrs` - статистические функции
- `chrono` - работа с датами
- `tracing` - логирование

## Лицензия

MIT
