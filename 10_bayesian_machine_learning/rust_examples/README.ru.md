# Байесовское машинное обучение для криптотрейдинга (Rust)

Реализация методов байесовского машинного обучения на Rust для торговли криптовалютами с использованием данных биржи Bybit в реальном времени.

## Возможности

- **Данные Bybit в реальном времени** — получение OHLCV-данных для любой поддерживаемой торговой пары
- **Модульная архитектура** — чёткое разделение между данными, выводом и примерами
- **MCMC-сэмплирование** — реализации Metropolis-Hastings и адаптивного MCMC
- **Фокус на трейдинг** — примеры специально разработаны для криптоторговли

## Структура проекта

```
rust_examples/
├── Cargo.toml
├── README.md
├── README.ru.md
└── src/
    ├── lib.rs                    # Корень библиотеки
    ├── data/
    │   ├── mod.rs
    │   ├── bybit.rs              # Клиент API Bybit
    │   └── returns.rs            # Утилиты расчёта доходности
    ├── bayesian/
    │   ├── mod.rs
    │   ├── distributions.rs      # Beta, Normal, Student-t распределения
    │   ├── inference.rs          # MCMC сэмплеры
    │   ├── linear_regression.rs  # Байесовская линейная/скользящая регрессия
    │   ├── sharpe.rs             # Байесовский коэффициент Шарпа
    │   └── volatility.rs         # Модели стохастической волатильности
    └── bin/
        ├── conjugate_priors.rs   # Оценка вероятности движения цены
        ├── bayesian_sharpe.rs    # Сравнение коэффициентов Шарпа
        ├── pairs_trading.rs      # Скользящая регрессия для парной торговли
        └── stochastic_volatility.rs  # Изменяющаяся во времени волатильность
```

## Установка

```bash
cd rust_examples
cargo build --release
```

## Использование

### 1. Сопряжённые априорные распределения: вероятность движения цены

Оценка вероятности роста цены с использованием Beta-Binomial сопряжённых априорных распределений:

```bash
# По умолчанию: BTCUSDT, часовой интервал
cargo run --release --bin conjugate_priors

# Пользовательский символ и интервал
cargo run --release --bin conjugate_priors -- --symbol ETHUSDT --interval 15 --limit 500

# С информативным априорным распределением (например, ожидание 60% вероятности роста)
cargo run --release --bin conjugate_priors -- --prior-alpha 6 --prior-beta 4
```

**Вывод:**
- Последовательные байесовские обновления
- Статистика апостериорного распределения
- Доверительные интервалы
- Вероятностные утверждения

### 2. Байесовский коэффициент Шарпа

Сравнение риск-скорректированной доходности двух криптовалют:

```bash
# Сравнение BTC и ETH
cargo run --release --bin bayesian_sharpe

# Сравнение любых двух символов
cargo run --release --bin bayesian_sharpe -- -1 SOLUSDT -2 AVAXUSDT --samples 10000

# Различные таймфреймы
cargo run --release --bin bayesian_sharpe -- --interval D --limit 365
```

**Вывод:**
- Полные апостериорные распределения
- Вероятность превосходства
- Анализ размера эффекта
- Вероятности пороговых значений риска

### 3. Парная торговля: скользящая байесовская регрессия

Оценка изменяющихся во времени коэффициентов хеджирования:

```bash
# Пара ETH/BTC (по умолчанию)
cargo run --release --bin pairs_trading

# Пользовательская пара
cargo run --release --bin pairs_trading -- -1 SOLUSDT -2 ETHUSDT --window 100

# Дневные данные
cargo run --release --bin pairs_trading -- --interval D --limit 365
```

**Вывод:**
- Скользящие оценки коэффициента хеджирования с неопределённостью
- Z-оценка спреда и торговые сигналы
- Обнаружение смены режима
- Предложения по размеру позиции

### 4. Модель стохастической волатильности

Моделирование изменяющейся во времени волатильности на крипторынках:

```bash
# Анализ волатильности BTC
cargo run --release --bin stochastic_volatility

# Другой актив
cargo run --release --bin stochastic_volatility -- --symbol ETHUSDT

# Больше MCMC-сэмплов для лучших оценок
cargo run --release --bin stochastic_volatility -- --samples 5000
```

**Вывод:**
- Параметры персистентности и возврата к среднему
- Оценки волатильности во времени с доверительными интервалами
- Классификация текущего режима волатильности
- Метрики риска на основе VaR
- Прогнозы волатильности

## Поддерживаемые символы

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)
- XRPUSDT (Ripple)
- DOGEUSDT (Dogecoin)
- ADAUSDT (Cardano)
- AVAXUSDT (Avalanche)
- DOTUSDT (Polkadot)
- LINKUSDT (Chainlink)
- MATICUSDT (Polygon)

## Поддерживаемые интервалы

| Интервал | Описание |
|----------|----------|
| 1 | 1 минута |
| 5 | 5 минут |
| 15 | 15 минут |
| 30 | 30 минут |
| 60 | 1 час |
| 240 | 4 часа |
| D | 1 день |
| W | 1 неделя |
| M | 1 месяц |

## Использование библиотеки

Используйте библиотеку в собственных Rust-проектах:

```rust
use bayesian_crypto::data::{BybitClient, Symbol, Returns};
use bayesian_crypto::bayesian::distributions::Beta;
use bayesian_crypto::bayesian::sharpe::BayesianSharpe;
use bayesian_crypto::bayesian::inference::MCMCConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Получение данных
    let client = BybitClient::new();
    let klines = client.get_klines(Symbol::BTCUSDT, "60", 500).await?;

    // Расчёт доходности
    let returns = Returns::from_klines(&klines);

    // Байесовский коэффициент Шарпа
    let estimator = BayesianSharpe::new(8760.0); // Часовая аннуализация
    let config = MCMCConfig::new(5000).with_warmup(1000);
    let result = estimator.estimate(&returns.values, &config);

    println!("Коэффициент Шарпа: {:.4} (95% ДИ: [{:.4}, {:.4}])",
        result.sharpe_mean(),
        result.sharpe_ci(0.95).0,
        result.sharpe_ci(0.95).1
    );

    Ok(())
}
```

## Детали алгоритмов

### Сопряжённые априорные распределения
Использует Beta-Binomial сопряжённость для точных апостериорных обновлений:
- Априорное: Beta(α, β)
- Правдоподобие: Binomial(n, p)
- Апостериорное: Beta(α + успехи, β + неудачи)

### Байесовский коэффициент Шарпа
Использует Student-t правдоподобие для устойчивости к выбросам:
- Доходности ~ StudentT(ν, μ, σ)
- Априорные: μ ~ Normal, σ ~ HalfCauchy, ν ~ Exponential
- Вывод через адаптивный Metropolis-Hastings MCMC

### Скользящая байесовская регрессия
Normal-Inverse-Gamma сопряжённое априорное для линейной регрессии:
- β | σ² ~ Normal(μ₀, σ²V₀)
- σ² ~ InverseGamma(a₀, b₀)
- Обеспечивает апостериорные обновления в замкнутой форме

### Стохастическая волатильность
AR(1) модель для лог-волатильности:
- rₜ = exp(hₜ/2) × εₜ, εₜ ~ N(0,1)
- hₜ = μ + φ(hₜ₋₁ - μ) + σηηₜ
- Вывод через particle MCMC

## Советы по производительности

1. **Используйте release-режим**: `cargo run --release` значительно быстрее
2. **Настройте количество сэмплов**: начните с меньшего для исследования, увеличьте для финального анализа
3. **Установите random seed**: используйте `--seed` для воспроизводимых результатов
4. **Кэшируйте данные**: рассмотрите сохранение полученных данных для повторного анализа

## Лицензия

MIT
