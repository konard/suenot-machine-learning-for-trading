# Crypto Autoencoders (Rust)

Реализация автоэнкодеров для анализа риск-факторов криптовалют на Rust.
Получает данные с биржи Bybit и применяет различные модели автоэнкодеров.

## Возможности

- **Bybit API клиент**: получение OHLCV данных, стакана заявок, тикеров
- **Предобработка данных**: извлечение технических индикаторов (RSI, MACD, Bollinger Bands и др.)
- **Автоэнкодеры**:
  - Простой (Vanilla) автоэнкодер
  - Глубокий (Deep) автоэнкодер
  - Шумоподавляющий (Denoising) автоэнкодер
  - Вариационный (VAE) автоэнкодер
  - Условный (Conditional) автоэнкодер
- **Анализ риск-факторов**: корреляции, важность признаков, кластеризация режимов

## Установка

```bash
cd rust_autoencoders
cargo build --release
```

## Использование

### 1. Загрузка данных с Bybit

```bash
# Загрузить данные для BTCUSDT (1000 часовых свечей)
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000

# Загрузить несколько символов
cargo run --example fetch_data -- --multi-symbol

# Параметры:
#   -s, --symbol     Торговая пара (по умолчанию: BTCUSDT)
#   -i, --interval   Интервал свечей: 1m, 5m, 15m, 1h, 4h, 1d
#   -l, --limit      Количество свечей (макс. 1000)
#   -o, --output-dir Папка для данных (по умолчанию: data)
```

### 2. Обучение моделей

```bash
# Обучить глубокий автоэнкодер
cargo run --example train_model -- --model deep --epochs 100

# Обучить VAE
cargo run --example train_model -- --model vae --latent-size 4

# Обучить шумоподавляющий автоэнкодер
cargo run --example train_model -- --model denoising --noise-std 0.1

# Параметры:
#   -m, --model       Тип модели: simple, deep, denoising, vae, conditional
#   -l, --latent-size Размер скрытого представления (по умолчанию: 8)
#   -e, --epochs      Количество эпох (по умолчанию: 100)
#   --lr              Скорость обучения (по умолчанию: 0.001)
```

### 3. Анализ риск-факторов

```bash
# Анализировать факторы
cargo run --example analyze_factors

# Параметры:
#   -n, --n-clusters  Количество рыночных режимов (по умолчанию: 3)
```

## Структура проекта

```
rust_autoencoders/
├── Cargo.toml              # Зависимости и настройки
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── bybit_client.rs     # Клиент Bybit API
│   ├── data_processor.rs   # Предобработка и признаки
│   ├── autoencoder.rs      # Модели автоэнкодеров
│   ├── risk_factors.rs     # Анализ риск-факторов
│   └── utils.rs            # Вспомогательные функции
├── examples/
│   ├── fetch_data.rs       # Загрузка данных
│   ├── train_model.rs      # Обучение моделей
│   └── analyze_factors.rs  # Анализ факторов
└── data/                   # Данные (создается автоматически)
```

## Примеры кода

### Получение данных

```rust
use crypto_autoencoders::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();

    // Получаем 1000 часовых свечей BTCUSDT
    let klines = client.get_klines("BTCUSDT", "1h", 1000).await.unwrap();

    for kline in klines.iter().take(5) {
        println!(
            "{}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            kline.datetime(),
            kline.open, kline.high, kline.low, kline.close, kline.volume
        );
    }
}
```

### Извлечение признаков

```rust
use crypto_autoencoders::{DataProcessor, NormalizationMethod};

let processor = DataProcessor::new()
    .with_lookback(20)
    .with_normalization(NormalizationMethod::MinMax);

let features = processor.extract_features(&klines);
println!("Признаки: {:?}", features.names);
```

### Обучение автоэнкодера

```rust
use crypto_autoencoders::Autoencoder;

// Создаем глубокий автоэнкодер: 16 -> [64, 32] -> 8 -> [32, 64] -> 16
let mut ae = Autoencoder::deep(16, &[64, 32], 8);

// Обучаем
ae.fit(&data, 100, 0.001);

// Получаем латентное представление
let latent = ae.transform(&data);

// Восстанавливаем
let reconstructed = ae.inverse_transform(&latent);
```

### Анализ риск-факторов

```rust
use crypto_autoencoders::RiskFactorAnalyzer;

let analyzer = RiskFactorAnalyzer::new();
let risk_factors = analyzer.analyze(&mut ae, &features);

for factor in &risk_factors {
    println!("{}: explained_variance = {:.2}%",
        factor.name,
        factor.explained_variance * 100.0
    );
}
```

## Извлекаемые признаки

Из OHLCV данных извлекаются 16 технических индикаторов:

| # | Признак | Описание |
|---|---------|----------|
| 1 | return | Процентное изменение цены |
| 2 | log_return | Логарифмическая доходность |
| 3 | volatility | Стандартное отклонение доходностей |
| 4 | rsi | Relative Strength Index |
| 5 | sma_ratio | Отношение цены к SMA |
| 6 | ema_ratio | Отношение цены к EMA |
| 7 | bb_position | Позиция в полосах Боллинджера |
| 8 | atr_normalized | Нормализованный ATR |
| 9 | volume_ratio | Отношение объема к среднему |
| 10 | price_range_ratio | Размах свечи к цене |
| 11 | macd_signal_diff | MACD - сигнальная линия |
| 12 | momentum | Изменение цены за N периодов |
| 13 | high_low_position | Позиция в диапазоне high-low |
| 14 | body_ratio | Тело свечи к размаху |
| 15 | upper_shadow_ratio | Верхняя тень к размаху |
| 16 | lower_shadow_ratio | Нижняя тень к размаху |

## Зависимости

- `reqwest` - HTTP клиент
- `tokio` - Асинхронный runtime
- `ndarray` - Многомерные массивы
- `serde` - Сериализация
- `chrono` - Работа с датами
- `clap` - CLI аргументы

## Лицензия

MIT
