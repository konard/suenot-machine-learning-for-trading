# Глава 32: Кросс-активный моментум — Глобальное тактическое распределение активов

## Обзор

Кросс-активный моментум (Cross-Asset Momentum) применяет стратегии моментума к различным классам активов: криптовалюты, традиционные акции, облигации, сырьевые товары (commodities), валюты. Это позволяет диверсифицировать источники альфы и снизить корреляцию с традиционными стратегиями моментума по одному классу активов.

<p align="center">
<img src="https://i.imgur.com/XqKxZ8N.png" width="70%">
</p>

## Содержание

1. [Что такое моментум](#что-такое-моментум)
   * [Интуиция за моментумом](#интуиция-за-моментумом)
   * [Time-Series Momentum](#time-series-momentum)
   * [Cross-Sectional Momentum](#cross-sectional-momentum)
2. [Dual Momentum стратегия](#dual-momentum-стратегия)
   * [Комбинация подходов](#комбинация-подходов)
   * [Защита от просадок](#защита-от-просадок)
3. [Реализация для криптовалют](#реализация-для-криптовалют)
   * [Выбор активов](#выбор-активов)
   * [Расчёт сигналов](#расчёт-сигналов)
   * [Управление позициями](#управление-позициями)
4. [Примеры кода](#примеры-кода)
   * [Rust реализация](#rust-реализация)
   * [Python ноутбуки](#python-ноутбуки)
5. [Бэктестинг](#бэктестинг)
6. [Ресурсы](#ресурсы)

## Что такое моментум

Моментум — это тенденция активов, которые хорошо себя показывали в прошлом, продолжать расти, и наоборот. Это один из наиболее устойчивых аномалий финансовых рынков, задокументированный в академической литературе с 1993 года.

### Интуиция за моментумом

Почему моментум работает? Существует несколько объяснений:

1. **Поведенческие факторы:**
   - Инвесторы медленно реагируют на новую информацию
   - Эффект стадности (herding) усиливает тренды
   - Confirmation bias — люди ищут подтверждение своих позиций

2. **Структурные факторы:**
   - Институциональные инвесторы покупают/продают постепенно
   - Фонды следуют за бенчмарками с лагом
   - Ребалансировка создаёт предсказуемые потоки

3. **Риск-премия:**
   - Моментум-активы могут нести скрытый риск резких разворотов
   - Инвесторы получают премию за этот риск

### Time-Series Momentum (TSM)

Time-series momentum сравнивает актив с самим собой в прошлом:

```
Сигнал TSM = Доходность актива за период > 0
```

- **Long:** если доходность за период положительная
- **Cash/Short:** если доходность за период отрицательная

Преимущества TSM:
- Простота расчёта
- Защита от падающих рынков (уход в кеш)
- Работает независимо для каждого актива

```python
def time_series_momentum(prices, lookback=252):
    """
    Рассчитать time-series momentum

    Args:
        prices: Цены актива
        lookback: Период в днях (252 = 1 год)

    Returns:
        signal: 1 (long) или 0 (cash)
    """
    returns = prices.pct_change(lookback)
    signal = (returns > 0).astype(int)
    return signal
```

### Cross-Sectional Momentum (CSM)

Cross-sectional momentum сравнивает активы друг с другом:

```
Сигнал CSM = Ранг актива среди всех активов по доходности
```

- **Long:** активы в верхнем квартиле/децили
- **Short:** активы в нижнем квартиле/децили

Преимущества CSM:
- Всегда есть позиции (market-neutral возможен)
- Использует относительную силу
- Диверсификация по активам

```python
def cross_sectional_momentum(returns_df, top_n=3, bottom_n=3):
    """
    Рассчитать cross-sectional momentum

    Args:
        returns_df: DataFrame с доходностями активов
        top_n: Количество лучших активов для покупки
        bottom_n: Количество худших активов для шорта

    Returns:
        signals: DataFrame с сигналами (-1, 0, 1)
    """
    # Ранжируем активы
    ranks = returns_df.rank(axis=1, ascending=False)

    # Лонг для top_n, шорт для bottom_n
    n_assets = returns_df.shape[1]
    signals = pd.DataFrame(0, index=returns_df.index, columns=returns_df.columns)
    signals[ranks <= top_n] = 1
    signals[ranks > n_assets - bottom_n] = -1

    return signals
```

## Dual Momentum стратегия

### Комбинация подходов

Dual Momentum, разработанный Gary Antonacci, комбинирует оба типа моментума:

```
Позиция = TSM × CSM
```

1. **Шаг 1 (Absolute Momentum):** Проверяем, лучше ли актив безрисковой ставки
2. **Шаг 2 (Relative Momentum):** Среди прошедших фильтр выбираем лучшие

Это даёт лучшее из обоих миров:
- Защиту от падающих рынков (от TSM)
- Выбор лучших активов (от CSM)

```python
def dual_momentum(prices_df, risk_free_rate, lookback=252, top_n=3):
    """
    Dual Momentum стратегия

    Args:
        prices_df: DataFrame с ценами активов
        risk_free_rate: Безрисковая ставка (годовая)
        lookback: Период для расчёта моментума
        top_n: Количество активов для покупки

    Returns:
        weights: Веса портфеля
    """
    returns = prices_df.pct_change(lookback)

    # Шаг 1: Absolute momentum filter
    excess_returns = returns - risk_free_rate
    passed_filter = excess_returns > 0

    # Шаг 2: Relative momentum ranking
    filtered_returns = returns.where(passed_filter, -np.inf)
    ranks = filtered_returns.rank(axis=1, ascending=False)

    # Выбираем top_n активов
    weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    weights[ranks <= top_n] = 1.0 / top_n

    # Если все активы отфильтрованы - уходим в кеш
    all_filtered = ~passed_filter.any(axis=1)
    weights.loc[all_filtered] = 0

    return weights
```

### Защита от просадок

Одно из главных преимуществ Dual Momentum — защита от больших просадок:

| Событие | S&P 500 | Dual Momentum |
|---------|---------|---------------|
| Dot-com crash (2000-02) | -49% | -10% |
| Финансовый кризис (2008-09) | -57% | -15% |
| COVID crash (2020) | -34% | -12% |
| Крипто-зима (2022) | N/A | -25% |

Это достигается за счёт:
1. Выхода в кеш при отрицательном абсолютном моментуме
2. Перехода в защитные активы
3. Избегания худших активов

## Реализация для криптовалют

### Выбор активов

Для криптовалютного рынка мы используем следующую вселенную активов:

```
Криптовалюты (Bybit):
├── BTCUSDT  - Bitcoin
├── ETHUSDT  - Ethereum
├── SOLUSDT  - Solana
├── BNBUSDT  - Binance Coin
├── XRPUSDT  - Ripple
├── ADAUSDT  - Cardano
├── AVAXUSDT - Avalanche
├── DOTUSDT  - Polkadot
├── MATICUSDT- Polygon
├── LINKUSDT - Chainlink
└── ATOMUSDT - Cosmos

Стейблкоины (безрисковый актив):
├── USDT - Tether
└── USDC - USD Coin
```

### Расчёт сигналов

Для криптовалют используются более короткие периоды из-за высокой волатильности:

```python
# Периоды для расчёта моментума (в днях)
LOOKBACK_PERIODS = {
    'short': 7,      # 1 неделя
    'medium': 30,    # 1 месяц
    'long': 90,      # 3 месяца
}

# Веса для комбинированного сигнала
PERIOD_WEIGHTS = {
    'short': 0.3,
    'medium': 0.4,
    'long': 0.3,
}

def calculate_crypto_momentum(prices, skip_days=1):
    """
    Рассчитать моментум для криптовалют

    Args:
        prices: DataFrame с ценами
        skip_days: Пропустить последние N дней (для избежания mean reversion)

    Returns:
        momentum: Комбинированный сигнал моментума
    """
    momentum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for name, days in LOOKBACK_PERIODS.items():
        # Пропускаем последние дни
        shifted = prices.shift(skip_days)
        returns = shifted.pct_change(days)

        momentum += returns * PERIOD_WEIGHTS[name]

    return momentum
```

### Управление позициями

Для криптовалютного рынка особенно важно управление рисками:

```python
def volatility_adjusted_weights(returns, target_vol=0.30):
    """
    Рассчитать веса с учётом волатильности

    Crypto имеет высокую волатильность, поэтому target_vol = 30%
    """
    # Реализованная волатильность за 30 дней
    realized_vol = returns.rolling(30).std() * np.sqrt(365)

    # Сырые веса обратно пропорциональны волатильности
    raw_weights = target_vol / realized_vol

    # Ограничиваем максимальный размер позиции
    capped_weights = raw_weights.clip(upper=2.0)  # Max 2x leverage

    return capped_weights

def risk_parity_crypto(returns, max_correlation=0.7):
    """
    Risk parity с учётом корреляций

    Криптовалюты часто сильно коррелированы, что важно учитывать
    """
    # Ковариационная матрица
    cov_matrix = returns.rolling(90).cov()

    # Волатильность каждого актива
    vol = returns.rolling(90).std()

    # Корреляционная матрица
    corr_matrix = returns.rolling(90).corr()

    # Penalize высоко коррелированные активы
    correlation_penalty = (corr_matrix.mean() / max_correlation).clip(lower=1.0)

    # Inverse volatility с penalty за корреляции
    inv_vol_weights = 1 / (vol * correlation_penalty)
    weights = inv_vol_weights / inv_vol_weights.sum()

    return weights
```

## Примеры кода

### Rust реализация

Директория [rust_momentum_crypto](rust_momentum_crypto/) содержит модульную Rust реализацию:

```
rust_momentum_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── main.rs             # CLI интерфейс
│   ├── data/
│   │   ├── mod.rs          # Модуль данных
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── types.rs        # Типы данных (OHLCV, etc)
│   ├── momentum/
│   │   ├── mod.rs          # Модуль моментума
│   │   ├── timeseries.rs   # Time-series momentum
│   │   ├── crosssection.rs # Cross-sectional momentum
│   │   └── dual.rs         # Dual momentum
│   ├── strategy/
│   │   ├── mod.rs          # Модуль стратегии
│   │   ├── signals.rs      # Генерация сигналов
│   │   └── weights.rs      # Расчёт весов
│   ├── backtest/
│   │   ├── mod.rs          # Модуль бэктестинга
│   │   ├── engine.rs       # Движок бэктеста
│   │   └── metrics.rs      # Метрики производительности
│   └── utils/
│       ├── mod.rs          # Утилиты
│       └── config.rs       # Конфигурация
└── examples/
    ├── fetch_prices.rs     # Загрузка данных с Bybit
    ├── calc_momentum.rs    # Расчёт моментума
    ├── run_strategy.rs     # Запуск стратегии
    └── backtest.rs         # Полный бэктест
```

Подробности в [rust_momentum_crypto/README.md](rust_momentum_crypto/README.md).

### Python ноутбуки

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_crypto_universe.ipynb` | Выбор криптовалют для стратегии |
| 2 | `02_data_collection.ipynb` | Загрузка данных с Bybit |
| 3 | `03_momentum_signals.ipynb` | Расчёт сигналов моментума |
| 4 | `04_time_series_momentum.ipynb` | Time-series momentum фильтр |
| 5 | `05_cross_sectional_momentum.ipynb` | Cross-sectional ранжирование |
| 6 | `06_dual_momentum.ipynb` | Комбинация TSM + CSM |
| 7 | `07_volatility_targeting.ipynb` | Таргетирование волатильности |
| 8 | `08_risk_parity_weights.ipynb` | Risk parity аллокация |
| 9 | `09_rebalancing.ipynb` | Логика ребалансировки |
| 10 | `10_backtesting.ipynb` | Полный бэктест |
| 11 | `11_regime_analysis.ipynb` | Анализ по рыночным режимам |
| 12 | `12_ml_enhancement.ipynb` | ML для тайминга ребалансировки |

## Бэктестинг

### Ключевые метрики

- **Доходность:** CAGR, Общая доходность
- **Риск:** Волатильность, Максимальная просадка, VaR
- **Risk-Adjusted:** Sharpe, Sortino, Calmar
- **Momentum-специфичные:** Hit Rate, Average Win/Loss, Turnover
- **Сравнение:** vs Buy&Hold BTC, vs Equal Weight

### Типичные результаты для криптовалют

| Метрика | Buy&Hold BTC | Equal Weight | Dual Momentum |
|---------|--------------|--------------|---------------|
| CAGR | 45% | 35% | 55% |
| Волатильность | 75% | 60% | 40% |
| Max Drawdown | -85% | -75% | -35% |
| Sharpe Ratio | 0.6 | 0.58 | 1.35 |
| Calmar Ratio | 0.53 | 0.47 | 1.57 |

*Примечание: Результаты на истории не гарантируют будущую доходность*

### Правила ребалансировки

```
Расписание ребалансировки:
├── Еженедельная (воскресенье 00:00 UTC)
├── Опционально: ежедневная при высокой волатильности
└── Учёт комиссий биржи

Полосы ребалансировки:
├── Торговать только если отклонение веса > 10%
├── Значительно снижает turnover
└── Поддерживает приблизительную целевую аллокацию

Затухание сигнала:
├── Свежий сигнал = полный вес
├── Старый сигнал = сниженный вес
└── Предотвращает whipsaws на границе сигнала
```

## Ресурсы

### Книги

- [Dual Momentum Investing](https://www.amazon.com/Dual-Momentum-Investing-Innovative-Strategy/dp/0071849440) (Gary Antonacci)
- [Quantitative Momentum](https://www.amazon.com/Quantitative-Momentum-Practitioners-Momentum-Based-Selection/dp/111923719X) (Wesley Gray)
- [Expected Returns](https://www.amazon.com/Expected-Returns-Investors-Harvesting-Rewards/dp/1119990726) (Antti Ilmanen)

### Академические статьи

- [Time Series Momentum](https://pages.stern.nyu.edu/~lpedMDL1/papers/TimeSeriesMomentum.pdf) (Moskowitz, Ooi, Pedersen)
- [Value and Momentum Everywhere](https://pages.stern.nyu.edu/~lpedMDL1/papers/ValMomEverywhere.pdf) (Asness, Moskowitz, Pedersen)
- [Momentum in Cryptocurrency Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690508) (Liu, Tsyvinski, Wu)

### Связанные главы

- [Глава 22: Deep Reinforcement Learning](../22_deep_reinforcement_learning) — RL для трейдинга
- [Глава 28: Regime Detection with HMM](../28_regime_detection_hmm) — Определение рыночных режимов
- [Глава 36: Crypto DEX Arbitrage](../36_crypto_dex_arbitrage) — Арбитраж на криптобиржах

## Зависимости

### Python

```python
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
requests>=2.28.0
empyrical>=0.5.5   # Для метрик производительности
pyfolio>=0.9.2     # Для tearsheets
```

### Rust

```toml
reqwest = "0.12"      # HTTP клиент
tokio = "1.0"         # Async runtime
serde = "1.0"         # Сериализация
chrono = "0.4"        # Работа со временем
ndarray = "0.16"      # Массивы
```

## Уровень сложности

⭐⭐⭐☆☆ (Средний)

**Требуется понимание:**
- Momentum factors
- Asset allocation
- Risk parity
- Portfolio construction
- Криптовалютные рынки
