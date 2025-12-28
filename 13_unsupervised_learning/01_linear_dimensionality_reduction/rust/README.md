# PCA Crypto - Principal Component Analysis for Cryptocurrency Trading

Реализация методов линейного снижения размерности (PCA) на Rust для анализа криптовалютных рынков с использованием данных биржи Bybit.

## Структура проекта

```
rust/
├── Cargo.toml              # Конфигурация проекта и зависимости
├── src/
│   ├── lib.rs              # Корневой модуль библиотеки
│   ├── main.rs             # CLI приложение
│   ├── api/                # Модуль работы с Bybit API
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── types.rs        # Типы данных API
│   ├── data/               # Модуль работы с данными
│   │   ├── mod.rs
│   │   ├── market_data.rs  # Структуры рыночных данных
│   │   ├── returns.rs      # Расчёт доходностей
│   │   └── preprocessing.rs # Предобработка данных
│   ├── pca/                # Модуль PCA
│   │   ├── mod.rs
│   │   ├── analysis.rs     # Основной анализ PCA
│   │   └── decomposition.rs # Матричные разложения
│   ├── portfolio/          # Модуль портфелей
│   │   ├── mod.rs
│   │   ├── eigenportfolio.rs # Собственные портфели
│   │   └── metrics.rs      # Метрики производительности
│   └── utils/              # Вспомогательные функции
│       ├── mod.rs
│       ├── statistics.rs   # Статистические функции
│       └── visualization.rs # Текстовая визуализация
└── examples/               # Примеры использования
    ├── curse_of_dimensionality.rs
    ├── pca_basic.rs
    ├── eigenportfolios.rs
    └── bybit_pca.rs
```

## Установка

### Требования

- Rust 1.70+ (установите через [rustup](https://rustup.rs/))
- OpenBLAS для линейной алгебры (опционально)

### Сборка

```bash
cd rust
cargo build --release
```

## Использование

### CLI команды

```bash
# Загрузить данные с Bybit
cargo run -- fetch -n 10 -l 100 -o data/prices.csv

# Запустить PCA анализ
cargo run -- analyze -i data/prices.csv -n 5

# Создать собственные портфели
cargo run -- portfolio -i data/prices.csv -n 4

# Анализ факторов риска
cargo run -- risk-factors -i data/prices.csv -n 5

# Демонстрация проклятия размерности
cargo run -- curse-dimensionality -n 100 -m 100
```

### Примеры

```bash
# Проклятие размерности
cargo run --example curse_of_dimensionality

# Основы PCA
cargo run --example pca_basic

# Собственные портфели
cargo run --example eigenportfolios

# PCA с данными Bybit
cargo run --example bybit_pca
```

## Модули

### API (`src/api/`)

Клиент для работы с Bybit API v5:

```rust
use pca_crypto::api::{BybitClient, Timeframe};

let client = BybitClient::new();

// Получить топ-10 символов по обороту
let symbols = client.get_top_symbols_by_turnover("linear", 10)?;

// Получить исторические данные
let klines = client.get_klines(
    "linear",
    "BTCUSDT",
    Timeframe::Day1,
    Some(100),
    None,
    None,
)?;
```

### Data (`src/data/`)

Работа с рыночными данными и доходностями:

```rust
use pca_crypto::data::{MarketData, Returns};

// Загрузить данные
let market_data = MarketData::from_csv("data/prices.csv")?;

// Рассчитать доходности
let returns = Returns::from_market_data(&market_data);

// Статистики
let cov = returns.covariance_matrix();
let corr = returns.correlation_matrix();
```

### PCA (`src/pca/`)

Анализ главных компонент:

```rust
use pca_crypto::pca::PCAAnalysis;

// Обучить PCA
let pca = PCAAnalysis::fit(&returns, Some(5));

// Получить объяснённую дисперсию
println!("Explained variance: {:?}", pca.explained_variance_ratio);

// Трансформировать данные
let transformed = pca.transform_returns(&returns);

// Обратная трансформация
let reconstructed = pca.inverse_transform(&transformed);
```

### Portfolio (`src/portfolio/`)

Построение и анализ портфелей:

```rust
use pca_crypto::portfolio::{EigenportfolioSet, PortfolioMetrics};

// Создать собственные портфели
let portfolios = EigenportfolioSet::from_returns(&returns, 4);

// Сравнить производительность
portfolios.compare_performance(&returns, 365.0);

// Метрики отдельного портфеля
let metrics = PortfolioMetrics::from_returns(&port_returns, 365.0);
metrics.summary();
```

## Примеры вывода

### PCA Summary

```
=== PCA Summary ===
Number of components: 5
Number of features: 10

Explained Variance Ratio:
--------------------------------------------------
   PC    Variance        Ratio   Cumulative
--------------------------------------------------
    1     0.004523      45.23%       45.23%
    2     0.001234      12.34%       57.57%
    3     0.000987       9.87%       67.44%
    4     0.000654       6.54%       73.98%
    5     0.000543       5.43%       79.41%
```

### Eigenportfolio Performance

```
=== Portfolio Performance Comparison ===
Portfolio                 Return (%)  Volatility (%)       Sharpe
-----------------------------------------------------------------
Market (Equal Weight)        12.34%        45.67%          0.27
Eigenportfolio 1             11.89%        44.32%          0.27
Eigenportfolio 2              5.43%        32.10%          0.17
Eigenportfolio 3             -2.10%        28.45%         -0.07
Eigenportfolio 4              8.76%        35.21%          0.25
```

## Теория

### Проклятие размерности

С ростом числа измерений:
- Среднее расстояние между точками растёт как √(d/6)
- Данные становятся разреженными
- Требуется экспоненциально больше данных

### PCA

Находит ортогональные направления максимальной дисперсии:
1. Вычисляем ковариационную матрицу
2. Находим собственные векторы и значения
3. Проецируем на главные компоненты

### Собственные портфели

- PC1 ≈ рыночный фактор (все активы движутся вместе)
- PC2+ ≈ отраслевые/стилевые факторы
- Портфели некоррелированы по построению

## API Bybit

Проект использует публичный API Bybit v5:
- Endpoint: `https://api.bybit.com/v5/`
- Не требует аутентификации для рыночных данных
- Rate limit: 120 запросов/минуту

Документация: https://bybit-exchange.github.io/docs/v5/intro

## Лицензия

MIT License

## См. также

- [README.ru.md](../README.ru.md) - Подробное описание PCA на русском
- [README.simple.ru.md](../README.simple.ru.md) - Объяснение для начинающих
- Python notebooks в родительской директории
