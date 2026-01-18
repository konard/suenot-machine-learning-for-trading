# HOT Crypto

Библиотека на Rust для реализации Higher Order Transformer для торговли криптовалютами с использованием данных биржи Bybit.

## Возможности

- Higher Order Attention механизм с CP разложением
- Standard и Kernel Attention для сравнения
- Тензорные операции и CP/Tucker разложение
- Загрузка данных с Bybit API v5
- Генерация торговых сигналов
- Управление рисками
- Бэктестинг стратегий

## Структура проекта

```
rust_hot_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Главный модуль библиотеки
│   ├── main.rs                 # CLI интерфейс
│   ├── data/
│   │   ├── mod.rs              # Модуль данных
│   │   ├── bybit.rs            # Клиент Bybit API
│   │   └── types.rs            # Типы данных (Candle, PriceSeries)
│   ├── attention/
│   │   ├── mod.rs              # Модуль внимания
│   │   ├── standard.rs         # Стандартный attention
│   │   ├── higher_order.rs     # Higher order attention
│   │   └── kernel.rs           # Kernel attention
│   ├── tensor/
│   │   ├── mod.rs              # Модуль тензоров
│   │   ├── tensor3d.rs         # 3D тензор
│   │   ├── decomposition.rs    # CP разложение
│   │   └── operations.rs       # Тензорные операции
│   ├── model/
│   │   ├── mod.rs              # Модуль модели
│   │   ├── transformer.rs      # HOT трансформер
│   │   └── predictor.rs        # Предсказатель цен
│   ├── strategy/
│   │   ├── mod.rs              # Модуль стратегии
│   │   └── signals.rs          # Торговые сигналы
│   └── utils/
│       ├── mod.rs              # Утилиты
│       └── config.rs           # Конфигурация
└── examples/
    ├── fetch_data.rs           # Загрузка данных с Bybit
    ├── attention_demo.rs       # Демонстрация attention
    ├── predict_movement.rs     # Прогноз движения цены
    └── backtest.rs             # Бэктест стратегии
```

## Установка

Добавьте в `Cargo.toml`:

```toml
[dependencies]
hot-crypto = { path = "path/to/rust_hot_crypto" }
```

## Быстрый старт

### CLI

```bash
# Получить текущие цены
cargo run -- prices

# Сгенерировать сигналы
cargo run -- signals --top-n 5 --min-confidence 0.6

# Прогноз для конкретного символа
cargo run -- predict --symbol BTCUSDT

# Показать конфигурацию
cargo run -- config --preset default

# Информация о библиотеке
cargo run -- info
```

### Примеры

```bash
# Загрузка данных с Bybit
cargo run --example fetch_data

# Демонстрация attention механизмов
cargo run --example attention_demo

# Прогноз движения цены
cargo run --example predict_movement

# Бэктестинг стратегии
cargo run --example backtest
```

### Как библиотека

```rust
use hot_crypto::{
    data::BybitClient,
    model::HOTPredictor,
    strategy::SignalGenerator,
    get_crypto_universe,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Создаём клиент Bybit
    let client = BybitClient::new();

    // Загружаем данные
    let series = client.get_klines("BTCUSDT", "D", None, None, Some(90)).await?;

    // Создаём предсказатель
    let predictor = HOTPredictor::default_crypto();

    // Получаем прогноз
    let prediction = predictor.predict_from_series(&series);

    println!(
        "{}: {} (confidence: {:.1}%)",
        prediction.symbol,
        prediction.prediction.as_str(),
        prediction.confidence * 100.0
    );

    // Генерируем торговый сигнал
    let generator = SignalGenerator::new();
    let signal = generator.generate(&prediction);

    println!("Signal: {} (position: {:.1}%)",
             signal.signal.as_str(),
             signal.position_size * 100.0);

    Ok(())
}
```

## Модули

### data

Модуль для работы с рыночными данными:

- `BybitClient` - клиент для Bybit API v5
- `Candle` - структура OHLCV свечи
- `PriceSeries` - временной ряд цен
- `Features` - признаки для модели

### attention

Механизмы внимания:

- `StandardAttention` - стандартный scaled dot-product attention
- `HigherOrderAttention` - attention высшего порядка с CP разложением
- `KernelAttention` - kernel attention для линейной сложности

### tensor

Тензорные операции:

- `Tensor3D` - трёхмерный тензор
- `CPDecomposition` - CP (CANDECOMP/PARAFAC) разложение
- Операции: outer product, mode-n product, нормы и т.д.

### model

Модель HOT:

- `HOTModel` - полная архитектура трансформера
- `HOTPredictor` - предсказатель с классификационной головой
- `MovementClass` - классы движения (Up, Down, Neutral)

### strategy

Торговая стратегия:

- `Signal` - торговый сигнал (Buy, Sell, Hold)
- `SignalGenerator` - генератор сигналов
- `RiskManager` - управление рисками

### utils

Утилиты:

- `Config` - конфигурация стратегии (JSON сериализация)

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
  "name": "HOT Crypto Strategy",
  "description": "Higher Order Transformer for crypto prediction",
  "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
  "model": {
    "input_dim": 7,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "rank": 16,
    "lookback": 60
  },
  "portfolio": {
    "initial_capital": 10000.0,
    "max_position": 0.25,
    "max_exposure": 0.80,
    "target_volatility": 0.30
  },
  "trading": {
    "min_confidence": 0.60,
    "rebalance_period": 1,
    "commission": 0.001,
    "slippage": 0.0005,
    "stop_loss": 0.03,
    "take_profit": 0.05
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
cargo test test_higher_order_attention
```

## Предупреждение

Эта библиотека предназначена для образовательных целей. Криптовалюты являются высокорисковыми активами. Не инвестируйте больше, чем готовы потерять.

## Лицензия

MIT
