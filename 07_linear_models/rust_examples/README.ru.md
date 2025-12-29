# Линейные модели для криптотрейдинга (Rust)

Реализация линейных моделей на Rust для прогнозирования цен криптовалют с использованием данных биржи **Bybit**.

## Возможности

- 📊 **Клиент Bybit API** - Получение OHLCV данных для любой криптовалютной пары
- 🔧 **Feature Engineering** - Технические индикаторы (SMA, EMA, RSI, MACD, полосы Боллинджера, ATR и др.)
- 📈 **Линейная регрессия** - Реализации МНК и градиентного спуска
- 🎯 **Регуляризация** - Ridge (L2), Lasso (L1) и Elastic Net
- 🔀 **Логистическая регрессия** - Бинарная классификация направления цены
- 📉 **Метрики** - Полный набор метрик для регрессии и классификации

## Структура проекта

```
rust_examples/
├── Cargo.toml
├── README.md
├── README.ru.md
├── src/
│   ├── lib.rs              # Точка входа библиотеки
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs        # Клиент API Bybit
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs    # Утилиты предобработки данных
│   │   └── features.rs     # Расчёт технических индикаторов
│   ├── models/
│   │   ├── mod.rs
│   │   ├── linear.rs       # Линейная регрессия (OLS, GD, SGD)
│   │   ├── regularization.rs  # Ridge, Lasso, Elastic Net
│   │   └── logistic.rs     # Логистическая регрессия
│   └── metrics/
│       ├── mod.rs
│       ├── regression.rs   # MSE, RMSE, R², IC и др.
│       └── classification.rs  # Accuracy, F1, AUC-ROC и др.
└── examples/
    ├── fetch_data.rs       # Пример получения данных
    ├── linear_regression.rs    # Пример линейной регрессии
    ├── ridge_lasso.rs      # Пример регуляризации
    ├── logistic_regression.rs  # Пример классификации
    └── full_pipeline.rs    # Полный ML-пайплайн
```

## Установка

### Требования

- Rust 1.70+ (установка через [rustup](https://rustup.rs/))
- OpenBLAS (для операций линейной алгебры)

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# macOS
brew install openblas

# Fedora
sudo dnf install openblas-devel
```

### Сборка

```bash
cd rust_examples
cargo build --release
```

## Использование

### Запуск примеров

```bash
# Получение данных криптовалют с Bybit
cargo run --example fetch_data

# Линейная регрессия для прогноза доходности
cargo run --example linear_regression

# Ridge и Lasso регуляризация
cargo run --example ridge_lasso

# Логистическая регрессия для прогноза направления
cargo run --example logistic_regression

# Полный ML-пайплайн для трейдинга
cargo run --example full_pipeline
```

### Использование как библиотеки

```rust
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval},
    data::features::FeatureEngineering,
    models::linear::LinearRegression,
    metrics::regression::RegressionMetrics,
};

fn main() -> anyhow::Result<()> {
    // Получение данных
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(500), None, None)?;

    // Генерация признаков
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);
    let target = FeatureEngineering::create_target(&klines, 1);

    // Обучение модели
    let mut model = LinearRegression::new(true);
    model.fit(&features, &target)?;

    // Оценка
    let predictions = model.predict(&features)?;
    let metrics = RegressionMetrics::calculate(&target, &predictions);

    println!("R²: {:.4}", metrics.r2);
    println!("IC: {:.4}", metrics.ic);

    Ok(())
}
```

## Модули

### API (Клиент Bybit)

```rust
use linear_models_crypto::api::bybit::{BybitClient, Interval};

let client = BybitClient::new();

// Получить 100 часовых свечей
let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

// Получить исторические данные с пагинацией
let history = client.get_klines_history("ETHUSDT", Interval::Day1, start_time, end_time)?;

// Получить текущий тикер
let ticker = client.get_ticker("SOLUSDT")?;
```

### Технические индикаторы

Поддерживаемые индикаторы:
- **Скользящие средние**: SMA, EMA
- **Моментум**: RSI, MACD, Momentum, ROC
- **Волатильность**: Полосы Боллинджера, ATR, скользящая волатильность
- **Объём**: OBV, коэффициент объёма
- **Ценовые признаки**: Лагированные доходности, диапазон High-Low

```rust
use linear_models_crypto::data::features::FeatureEngineering;

// Генерация всех признаков
let (features, names) = FeatureEngineering::generate_features(&klines);

// Отдельные индикаторы
let sma_20 = FeatureEngineering::sma(&prices, 20);
let rsi_14 = FeatureEngineering::rsi(&prices, 14);
let (macd, signal, hist) = FeatureEngineering::macd(&prices, 12, 26, 9);
```

### Модели

#### Линейная регрессия

```rust
use linear_models_crypto::models::linear::{LinearRegression, LinearRegressionGD};

// МНК (OLS)
let mut ols = LinearRegression::new(true);
ols.fit(&x_train, &y_train)?;
let predictions = ols.predict(&x_test)?;

// Градиентный спуск
let mut gd = LinearRegressionGD::new(0.01, 1000, 1e-6, true);
gd.fit(&x_train, &y_train)?;
```

#### Регуляризованная регрессия

```rust
use linear_models_crypto::models::regularization::{RidgeRegression, LassoRegression, ElasticNet};

// Ridge (L2) - сжатие коэффициентов
let mut ridge = RidgeRegression::new(1.0, true, false);
ridge.fit(&x_train, &y_train)?;

// Lasso (L1) - отбор признаков
let mut lasso = LassoRegression::new(0.01, true, 1000, 1e-6);
lasso.fit(&x_train, &y_train)?;
println!("Отобранные признаки: {:?}", lasso.selected_features());

// Elastic Net - комбинация L1 и L2
let mut enet = ElasticNet::new(0.1, 0.5, true, 1000, 1e-6);
enet.fit(&x_train, &y_train)?;
```

#### Логистическая регрессия

```rust
use linear_models_crypto::models::logistic::{LogisticRegression, Regularization};

// Базовая модель
let mut lr = LogisticRegression::default();
lr.fit(&x_train, &y_train)?;

// С L2 регуляризацией
let mut lr_l2 = LogisticRegression::with_l2(1.0);
lr_l2.fit(&x_train, &y_train)?;

// Прогнозы
let probabilities = lr.predict_proba(&x_test)?;  // Вероятности
let classes = lr.predict(&x_test)?;               // Классы (0 или 1)
```

### Метрики

```rust
use linear_models_crypto::metrics::{
    regression::RegressionMetrics,
    classification::ClassificationMetrics,
};

// Метрики регрессии
let reg_metrics = RegressionMetrics::calculate(&y_true, &y_pred);
println!("{}", reg_metrics.report());

// Метрики классификации
let clf_metrics = ClassificationMetrics::calculate_with_proba(&y_true, &y_pred, Some(&y_proba));
println!("{}", clf_metrics.report());
```

## Ключевые метрики

### Регрессия
- **MSE/RMSE**: Среднеквадратичная ошибка
- **MAE**: Средняя абсолютная ошибка
- **R²**: Коэффициент детерминации
- **IC**: Информационный коэффициент (корреляция Пирсона)
- **Hit Rate**: Точность направления

### Классификация
- **Accuracy, Precision, Recall, F1**
- **AUC-ROC**: Площадь под ROC-кривой
- **MCC**: Коэффициент корреляции Мэтьюса
- **Log Loss**: Логарифмическая функция потерь

## Советы по производительности

1. **Используйте release-режим** для ускорения в 10-100 раз:
   ```bash
   cargo run --release --example full_pipeline
   ```

2. **Стандартизируйте признаки** перед обучением регуляризованных моделей

3. **Используйте кросс-валидацию временных рядов** для избежания lookahead bias

4. **Начинайте с простых моделей** (OLS) перед добавлением регуляризации

## Пример вывода

```
╔══════════════════════════════════════════════════════╗
║     Cryptocurrency ML Trading Pipeline               ║
╚══════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1: Fetching Market Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fetching 90 days of 4-hour data for multiple assets...

  BTCUSDT - 540 candles fetched
  ETHUSDT - 540 candles fetched
  SOLUSDT - 540 candles fetched

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 6: Backtesting Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy             Return       Sharpe       Max DD   Win Rate   Trades
--------------------------------------------------------------------------------
Buy & Hold           12.34%         0.85       -15.23%      55.0%      108
Regression Signal    18.56%         1.23       -10.12%      58.0%       95
Classification       15.78%         1.05       -12.45%      56.5%       72
```

## Лицензия

MIT License - см. основной репозиторий для подробностей.
