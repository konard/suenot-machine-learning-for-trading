# ML Crypto - Machine Learning for Cryptocurrency Trading

Rust-реализация концепций машинного обучения для криптовалютного трейдинга с использованием данных биржи Bybit.

## Структура проекта

```
rust_ml_crypto/
├── Cargo.toml              # Зависимости и конфигурация
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── main.rs             # CLI приложение
│   ├── api/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── error.rs        # Типы ошибок
│   ├── data/
│   │   ├── mod.rs
│   │   ├── types.rs        # Candle, OrderBook, Trade, Dataset
│   │   └── loader.rs       # CSV/JSON загрузка и сохранение
│   ├── features/
│   │   ├── mod.rs
│   │   ├── engineering.rs  # Генерация признаков
│   │   ├── indicators.rs   # Технические индикаторы
│   │   └── mutual_info.rs  # Взаимная информация
│   └── ml/
│       ├── mod.rs
│       ├── knn.rs          # K-ближайших соседей
│       ├── metrics.rs      # Метрики качества
│       ├── cross_validation.rs  # Кросс-валидация
│       └── bias_variance.rs     # Анализ смещения-дисперсии
└── examples/
    ├── fetch_data.rs       # Загрузка данных с Bybit
    ├── knn_workflow.rs     # Полный ML workflow
    ├── mutual_information.rs # Отбор признаков
    ├── bias_variance.rs    # Компромисс смещения-дисперсии
    └── cross_validation.rs # Кросс-валидация
```

## Установка

```bash
cd rust_ml_crypto
cargo build --release
```

## Использование

### CLI приложение

```bash
# Загрузить данные с Bybit
cargo run -- fetch -s BTCUSDT -i 1h -l 500

# Запустить ML workflow
cargo run -- workflow

# Анализ признаков
cargo run -- features

# Анализ bias-variance
cargo run -- bias-variance

# Кросс-валидация
cargo run -- cross-val --time-series
```

### Запуск примеров

```bash
# Загрузка данных
cargo run --example fetch_data

# Полный ML workflow с KNN
cargo run --example knn_workflow

# Взаимная информация для отбора признаков
cargo run --example mutual_information

# Компромисс смещения и дисперсии
cargo run --example bias_variance

# Кросс-валидация
cargo run --example cross_validation
```

## Модули

### API (src/api/)

Клиент для работы с Bybit API:

```rust
use ml_crypto::api::BybitClient;

let client = BybitClient::new();
let candles = client.get_klines("BTCUSDT", "1h", 100).await?;
let orderbook = client.get_orderbook("BTCUSDT", 10).await?;
let trades = client.get_recent_trades("BTCUSDT", 50).await?;
```

### Data (src/data/)

Структуры данных и утилиты:

```rust
use ml_crypto::data::{Candle, Dataset, DataLoader};

// Загрузка из файла
let candles = DataLoader::load_candles("data.csv")?;

// Сохранение
DataLoader::save_candles(&candles, "output.csv")?;

// Dataset для ML
let dataset = Dataset::new(x, y, feature_names, target_name);
let (train, test) = dataset.train_test_split(0.2);
```

### Features (src/features/)

Инженерия признаков и технические индикаторы:

```rust
use ml_crypto::features::{FeatureEngine, TechnicalIndicators, MutualInformation};

// Генерация признаков
let engine = FeatureEngine::new();
let dataset = engine.generate_features(&candles)?;

// Технические индикаторы
let sma = TechnicalIndicators::sma(&prices, 20);
let rsi = TechnicalIndicators::rsi(&prices, 14);
let (macd, signal, hist) = TechnicalIndicators::macd(&prices, 12, 26, 9);

// Взаимная информация
let mi_scores = MutualInformation::feature_mutual_info(&x, &y, 20);
```

### ML (src/ml/)

Алгоритмы машинного обучения:

```rust
use ml_crypto::ml::{KNNClassifier, Metrics, CrossValidator};

// KNN классификатор
let mut knn = KNNClassifier::new(5);
knn.fit(&x_train, &y_train);
let predictions = knn.predict(&x_test);

// Метрики
let accuracy = Metrics::accuracy(&y_test, &predictions);
let f1 = Metrics::f1_score(&y_test, &predictions, 1.0);

// Кросс-валидация
let splits = CrossValidator::time_series_split(n_samples, 5, None);
let purged = CrossValidator::purged_k_fold(n_samples, 5, 2, 2);
```

## Признаки (Features)

Генерируемые признаки включают:

- **Доходности**: простые и логарифмические
- **Скользящие средние**: SMA, EMA (различные периоды)
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: позиция и ширина
- **ATR**: Average True Range (волатильность)
- **Volume**: отношение к среднему
- **Свечные паттерны**: тело, тени

## Кросс-валидация для финансовых данных

Библиотека поддерживает специализированные методы CV:

1. **K-Fold** - стандартная кросс-валидация
2. **Time Series Split** - для временных рядов (расширяющееся окно)
3. **Purged K-Fold** - с очисткой данных для предотвращения утечки
4. **Combinatorial Purged CV** - комбинаторная очищенная CV

## Примеры вывода

```
=== KNN Workflow Example ===

Step 1: Fetching data from Bybit...
  Fetched 500 candles

Step 2: Generating features...
  Generated 25 features for 380 samples

Step 3: Splitting data...
  Training samples: 304
  Testing samples: 76

Step 4: Training KNN classifiers...

   K   Accuracy  Precision     Recall         F1
------------------------------------------------
   1     0.5132     0.5200     0.5306     0.5253
   3     0.5395     0.5476     0.5510     0.5493
   5     0.5526     0.5610     0.5612     0.5611
...
```

## Лицензия

MIT
