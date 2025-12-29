# Конформное предсказание для торговли криптовалютами (Rust)

Реализация методов конформного предсказания на Rust для торговли криптовалютами с использованием данных биржи **Bybit**.

## Возможности

- **API клиент Bybit** - Получение OHLCV данных для любой криптовалютной пары
- **Инжиниринг признаков** - Технические индикаторы (SMA, EMA, RSI, MACD, полосы Боллинджера, ATR и др.)
- **Раздельное конформное предсказание** - Интервалы предсказания с гарантированным покрытием
- **Адаптивный конформный вывод** - Динамическая подстройка для временных рядов
- **Торговая стратегия** - Торговля только при высокой уверенности, размер позиции на основе неопределённости
- **Комплексные метрики** - Покрытие, коэффициент Шарпа, процент выигрышей, просадка

## Структура проекта

```
rust_examples/
├── Cargo.toml
├── README.md
├── README.ru.md
├── src/
│   ├── lib.rs                  # Точка входа библиотеки
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs            # API клиент Bybit
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs        # Утилиты предобработки данных
│   │   └── features.rs         # Расчёт технических индикаторов
│   ├── conformal/
│   │   ├── mod.rs
│   │   ├── model.rs            # Простые модели предсказания
│   │   ├── split.rs            # Раздельное конформное предсказание
│   │   └── adaptive.rs         # Адаптивный конформный вывод
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── trading.rs          # Торговая стратегия с интервалами
│   │   └── sizing.rs           # Методы определения размера позиции
│   └── metrics/
│       ├── mod.rs
│       ├── coverage.rs         # Метрики покрытия и интервалов
│       └── trading.rs          # Метрики торговой результативности
└── examples/
    ├── fetch_data.rs           # Пример получения данных
    ├── split_conformal.rs      # Пример раздельного CP
    ├── adaptive_conformal.rs   # Пример адаптивного CP
    ├── trading_strategy.rs     # Пример торговой стратегии
    └── full_pipeline.rs        # Полный ML торговый пайплайн
```

## Установка

### Требования

- Rust 1.70+ (установка через [rustup](https://rustup.rs/))

```bash
# Установите Rust, если ещё не установлен
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Сборка

```bash
cd rust_examples
cargo build --release
```

## Использование

### Запуск примеров

```bash
# Получение криптовалютных данных с Bybit
cargo run --example fetch_data

# Раздельное конформное предсказание для прогнозирования доходностей
cargo run --example split_conformal

# Адаптивный конформный вывод для временных рядов
cargo run --example adaptive_conformal

# Торговая стратегия с интервалами предсказания
cargo run --example trading_strategy

# Полный ML торговый пайплайн
cargo run --example full_pipeline
```

### Использование как библиотеки

```rust
use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    data::features::FeatureEngineering,
    conformal::split::SplitConformalPredictor,
    conformal::model::LinearModel,
    strategy::trading::ConformalTradingStrategy,
    metrics::coverage::CoverageMetrics,
};

fn main() -> anyhow::Result<()> {
    // Получение данных с Bybit
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour4, Some(500), None, None)?;

    // Генерация признаков и целей
    let (features, _) = FeatureEngineering::generate_features(&klines);
    let targets = FeatureEngineering::create_returns(&klines, 1);

    // Разделение данных
    // ... обучение, калибровка, тест ...

    // Обучение конформного предсказателя (90% покрытие)
    let model = LinearModel::new(true);
    let mut cp = SplitConformalPredictor::new(model, 0.1);
    cp.fit(&x_train, &y_train, &x_calib, &y_calib);

    // Генерация предсказаний с интервалами
    let intervals = cp.predict(&x_test);

    // Создание торговой стратегии
    let strategy = ConformalTradingStrategy::new(0.02, 0.005);
    let signals = strategy.generate_signals(&intervals);

    // Оценка
    let coverage = CoverageMetrics::calculate(&intervals, &y_test, 0.1);
    println!("Покрытие: {:.1}%", coverage.coverage * 100.0);

    Ok(())
}
```

## Ключевые концепции

### Гарантия покрытия

Раздельное конформное предсказание обеспечивает:

```
P(Y ∈ [нижняя, верхняя]) ≥ 1 - α
```

Для α = 0.1 примерно 90% ваших интервалов будут содержать истинное значение.

### Логика торговой стратегии

1. **Торгуем только при уверенности**: Пропускаем если ширина_интервала > порог
2. **Требуется ясное направление**: Пропускаем если интервал пересекает ноль
3. **Размер на основе уверенности**: Уже интервал → больше позиция

```
если ширина_интервала < порог И нижняя > мин_преимущество:
    ЛОНГ с размером = f(1/ширина_интервала)
иначе если ширина_интервала < порог И верхняя < -мин_преимущество:
    ШОРТ с размером = f(1/ширина_интервала)
иначе:
    НЕ ТОРГУЕМ
```

## Советы по производительности

1. **Используйте release режим** для ускорения в 10-100 раз:
   ```bash
   cargo run --release --example full_pipeline
   ```

2. **Стандартизируйте признаки** перед обучением

3. **Используйте временные разбиения** для избежания заглядывания вперёд

4. **Начните с более широких порогов** и настройте на основе результатов

## Лицензия

MIT License - Смотрите основной репозиторий для деталей.
