# Temporal Fusion Transformers для многогоризонтного прогнозирования

Эта глава посвящена **Temporal Fusion Transformers (TFT)** — современной архитектуре для интерпретируемого многогоризонтного прогнозирования временных рядов. Мы применяем TFT для прогнозирования доходности криптовалютных активов и построения динамической торговой стратегии.

<p align="center">
<img src="https://i.imgur.com/8QZxK9L.png" width="70%">
</p>

## Содержание

1. [Введение в Temporal Fusion Transformers](#введение-в-temporal-fusion-transformers)
    * [Зачем нужен TFT?](#зачем-нужен-tft)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими моделями](#сравнение-с-другими-моделями)
2. [Архитектура TFT](#архитектура-tft)
    * [Обработка входных данных](#обработка-входных-данных)
    * [Variable Selection Networks](#variable-selection-networks)
    * [Gated Residual Networks (GRN)](#gated-residual-networks-grn)
    * [Temporal Self-Attention](#temporal-self-attention)
    * [Quantile Forecasting](#quantile-forecasting)
3. [Типы входных данных](#типы-входных-данных)
    * [Статические ковариаты](#статические-ковариаты)
    * [Известные будущие переменные](#известные-будущие-переменные)
    * [Наблюдаемые переменные](#наблюдаемые-переменные)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Архитектура TFT](#02-архитектура-tft)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Многогоризонтные прогнозы](#04-многогоризонтные-прогнозы)
    * [05: Интерпретируемость](#05-интерпретируемость)
    * [06: Торговая стратегия](#06-торговая-стратегия)
    * [07: Бэктестинг](#07-бэктестинг)
    * [08: Сравнение с baseline](#08-сравнение-с-baseline)
5. [Реализация на Rust](#реализация-на-rust)
6. [Практические рекомендации](#практические-рекомендации)
7. [Ресурсы](#ресурсы)

## Введение в Temporal Fusion Transformers

Temporal Fusion Transformer (TFT) — это архитектура глубокого обучения, разработанная Google в 2019 году специально для временных рядов. Она объединяет лучшие элементы рекуррентных сетей, механизма внимания и специализированных компонентов для работы с разнородными входными данными.

### Зачем нужен TFT?

Традиционные модели временных рядов имеют ограничения:

| Модель | Проблема |
|--------|----------|
| **ARIMA** | Только линейные зависимости, один горизонт |
| **LSTM** | Сложно интерпретировать, нет механизма внимания |
| **Prophet** | Не учитывает экзогенные переменные хорошо |
| **Transformer** | Не различает типы входных данных |

TFT решает все эти проблемы:
- **Многогоризонтное прогнозирование**: одна модель для прогнозов на 1, 5, 20 дней
- **Интерпретируемость**: веса важности переменных и временных паттернов
- **Гибкость**: работа со статическими, известными и наблюдаемыми переменными
- **Quantile forecasts**: вероятностные прогнозы с доверительными интервалами

### Ключевые преимущества

1. **Обработка разнородных данных**
   - Статические признаки (тип актива, сектор)
   - Известные будущие (календарь, праздники)
   - Наблюдаемые временные ряды (цены, объемы)

2. **Интерпретируемость**
   - Variable Selection: какие признаки важны
   - Attention weights: какие временные точки важны
   - Можно объяснить каждый прогноз

3. **Quantile Regression**
   - Прогнозирование нескольких квантилей (10%, 50%, 90%)
   - Оценка неопределенности прогноза
   - Основа для управления рисками

### Сравнение с другими моделями

| Характеристика | LSTM | Transformer | N-BEATS | TFT |
|----------------|------|-------------|---------|-----|
| Multi-horizon | ✓ | ✓ | ✓ | ✓ |
| Интерпретируемость | ✗ | Частично | ✓ | ✓ |
| Разные типы данных | ✗ | ✗ | ✗ | ✓ |
| Quantile forecasts | Отдельно | Отдельно | ✗ | ✓ |
| Attention для времени | ✗ | ✓ | ✗ | ✓ |

## Архитектура TFT

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL FUSION TRANSFORMER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Статические │  │   Известные │  │ Наблюдаемые │                 │
│  │  ковариаты   │  │   будущие   │  │  переменные │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         ▼                ▼                ▼                         │
│  ┌─────────────────────────────────────────────────┐               │
│  │          Variable Selection Networks            │               │
│  │     (Выбор важных переменных для каждого шага)  │               │
│  └───────────────────────┬─────────────────────────┘               │
│                          │                                          │
│         ┌────────────────┴────────────────┐                        │
│         ▼                                 ▼                        │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │   LSTM Encoder  │              │   LSTM Decoder  │              │
│  │   (Прошлое)     │───────────▶  │   (Будущее)     │              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                │                        │
│           └────────────┬───────────────────┘                       │
│                        ▼                                            │
│  ┌─────────────────────────────────────────────────┐               │
│  │         Temporal Self-Attention Layer           │               │
│  │    (Взвешивание временных зависимостей)         │               │
│  └───────────────────────┬─────────────────────────┘               │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────┐               │
│  │            Quantile Output Layer                │               │
│  │         (Прогнозы: 10%, 50%, 90%)               │               │
│  └─────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### Обработка входных данных

TFT использует три категории входных данных:

```python
# Пример структуры данных для TFT
{
    "static": {
        "asset_type": "crypto",        # Категориальная
        "exchange": "bybit",           # Категориальная
        "market_cap_tier": 1           # Числовая
    },
    "known_future": {
        "day_of_week": [0, 1, 2, ...], # Известно заранее
        "is_weekend": [0, 0, 0, ...],  # Известно заранее
        "hours_to_funding": [8, 7, ...]# Время до фандинга
    },
    "observed": {
        "close": [45000, 45100, ...],  # Только прошлое
        "volume": [1000, 1200, ...],   # Только прошлое
        "rsi": [55, 58, ...]           # Только прошлое
    }
}
```

### Variable Selection Networks

Variable Selection Network (VSN) автоматически определяет важность каждой переменной:

```
Вход: [x₁, x₂, x₃, ..., xₙ] — все признаки

       ↓
┌─────────────────────┐
│   GRN для каждого   │
│      признака       │
└─────────────────────┘
       ↓
┌─────────────────────┐
│     Softmax         │  → Веса важности v = [v₁, v₂, ..., vₙ]
└─────────────────────┘
       ↓
Взвешенная сумма: Σ vᵢ * GRN(xᵢ)
```

**Интерпретация весов:**
- Высокий вес → признак важен для прогноза
- Веса меняются во времени → адаптивность модели
- Можно визуализировать для объяснения решений

### Gated Residual Networks (GRN)

GRN — базовый строительный блок TFT:

```python
def GRN(x, context=None):
    # Основная трансформация
    η₁ = Dense(x)
    η₂ = Dense(concat(x, context)) if context else Dense(x)

    # ELU активация
    η = ELU(η₁ + η₂)

    # Gating механизм (GLU)
    gate = Dense(η) → Sigmoid
    output = Dense(η) * gate  # Элементное умножение

    # Residual connection
    return LayerNorm(x + output)
```

Gating позволяет модели:
- Пропускать нерелевантную информацию
- Адаптивно комбинировать признаки
- Избегать vanishing gradients

### Temporal Self-Attention

Interpretable Multi-Head Attention (IMHA) — модифицированный self-attention:

```
Q = Dense(h)     # Query из скрытых состояний
K = Dense(h)     # Key
V = Dense(h)     # Value (без трансформации для интерпретируемости!)

Attention(Q, K, V) = softmax(QK^T / √d) * V
```

**Особенности:**
- Values не трансформируются — можно напрямую интерпретировать веса
- Отдельный attention для каждой головы
- Веса показывают, какие прошлые моменты важны

### Quantile Forecasting

Вместо одного прогноза TFT предсказывает несколько квантилей:

```
Выход модели: [q₁₀, q₅₀, q₉₀] для каждого горизонта

Где:
- q₁₀ = 10-й перцентиль (пессимистичный прогноз)
- q₅₀ = медиана (основной прогноз)
- q₉₀ = 90-й перцентиль (оптимистичный прогноз)
```

**Quantile Loss:**

$$L_q(\hat{y}, y) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$$

где q — целевой квантиль (0.1, 0.5, 0.9).

## Типы входных данных

### Статические ковариаты

Признаки, которые не меняются во времени:

| Признак | Тип | Пример |
|---------|-----|--------|
| `symbol` | Категориальный | BTCUSDT, ETHUSDT |
| `exchange` | Категориальный | bybit, binance |
| `asset_class` | Категориальный | layer1, defi, meme |
| `launch_year` | Числовой | 2009, 2015, 2021 |
| `max_supply` | Числовой | 21M, 100M, unlimited |

### Известные будущие переменные

Признаки, которые известны заранее:

| Признак | Описание |
|---------|----------|
| `day_of_week` | День недели (0-6) |
| `hour_of_day` | Час дня (0-23) |
| `is_weekend` | Выходной день |
| `hours_to_funding` | Часы до ставки фандинга |
| `days_to_expiry` | Дни до экспирации опционов |
| `is_month_end` | Конец месяца |

### Наблюдаемые переменные

Признаки, которые известны только для прошлого:

| Признак | Описание |
|---------|----------|
| `close` | Цена закрытия |
| `volume` | Объем торгов |
| `returns` | Доходность |
| `volatility` | Реализованная волатильность |
| `rsi` | Relative Strength Index |
| `funding_rate` | Ставка фандинга |
| `open_interest` | Открытый интерес |
| `bid_ask_spread` | Спред стакана |

## Практические примеры

### 01: Подготовка данных

Ноутбук [01_data_preparation.ipynb](01_data_preparation.ipynb):
- Загрузка данных с Bybit API
- Расчет технических индикаторов
- Создание временных признаков
- Разделение на train/val/test

```python
# Структура данных для PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet

dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="returns",
    group_ids=["symbol"],
    static_categoricals=["asset_class"],
    time_varying_known_reals=["day_of_week", "hour"],
    time_varying_unknown_reals=["close", "volume", "rsi"],
    max_encoder_length=168,  # 7 дней для часовых данных
    max_prediction_length=24,  # Прогноз на 24 часа
)
```

### 02: Архитектура TFT

Ноутбук [02_tft_architecture.ipynb](02_tft_architecture.ipynb):
- Детальный разбор каждого компонента
- Реализация GRN с нуля
- Variable Selection Network
- Temporal Self-Attention

### 03: Обучение модели

Ноутбук [03_model_training.ipynb](03_model_training.ipynb):
- Конфигурация гиперпараметров
- Обучение с PyTorch Lightning
- Early stopping и LR scheduling
- Мониторинг метрик

```python
from pytorch_forecasting import TemporalFusionTransformer

model = TemporalFusionTransformer.from_dataset(
    dataset,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # 7 квантилей
    loss=QuantileLoss(),
    learning_rate=0.001,
    reduce_on_plateau_patience=4,
)
```

### 04: Многогоризонтные прогнозы

Ноутбук [04_multi_horizon_forecasting.ipynb](04_multi_horizon_forecasting.ipynb):
- Прогнозы на 1h, 4h, 1d, 7d
- Визуализация доверительных интервалов
- Анализ точности по горизонтам
- Калибровка квантилей

### 05: Интерпретируемость

Ноутбук [05_interpretability.ipynb](05_interpretability.ipynb):
- Веса важности переменных
- Attention patterns во времени
- SHAP values для TFT
- Объяснение конкретных прогнозов

```python
# Получение весов важности
interpretation = model.interpret_output(predictions, reduction="sum")

# Variable importance
var_importance = interpretation["encoder_importance"]
# {"close": 0.25, "volume": 0.20, "rsi": 0.15, ...}

# Temporal attention
attention_weights = interpretation["attention"]
# Какие прошлые шаги важны для прогноза
```

### 06: Торговая стратегия

Ноутбук [06_allocation_strategy.ipynb](06_allocation_strategy.ipynb):
- Правила входа/выхода на основе прогнозов
- Размер позиции по ширине prediction interval
- Динамическая аллокация между активами
- Риск-менеджмент

**Логика стратегии:**

```python
def generate_signal(forecast):
    q10, q50, q90 = forecast['q10'], forecast['q50'], forecast['q90']
    interval_width = q90 - q10

    # Уверенный сигнал = узкий интервал + сильный прогноз
    if q10 > 0 and interval_width < threshold:
        return "LONG", confidence_to_size(interval_width)
    elif q90 < 0 and interval_width < threshold:
        return "SHORT", confidence_to_size(interval_width)
    else:
        return "HOLD", 0

def confidence_to_size(interval_width):
    # Обратно пропорционально неопределенности
    return min(1.0, base_size / interval_width)
```

### 07: Бэктестинг

Ноутбук [07_backtesting.ipynb](07_backtesting.ipynb):
- Walk-forward validation
- Реалистичные комиссии и slippage
- Sharpe, Sortino, Calmar ratios
- Drawdown analysis

### 08: Сравнение с baseline

Ноутбук [08_comparison_baselines.ipynb](08_comparison_baselines.ipynb):
- LSTM baseline
- ARIMA baseline
- Buy & Hold
- Статистические тесты значимости

## Реализация на Rust

Директория [rust_tft_trading](rust_tft_trading/) содержит реализацию на Rust с использованием данных Bybit:

```
rust_tft_trading/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный модуль
│   ├── main.rs             # CLI приложение
│   ├── api/                # Работа с Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   └── types.rs        # Типы данных API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Загрузка данных
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset для TFT
│   ├── model/              # Архитектура TFT
│   │   ├── mod.rs
│   │   ├── embedding.rs    # Embeddings
│   │   ├── grn.rs          # Gated Residual Network
│   │   ├── vsn.rs          # Variable Selection
│   │   ├── attention.rs    # Temporal attention
│   │   └── tft.rs          # Полная модель
│   ├── training/           # Обучение
│   │   ├── mod.rs
│   │   ├── trainer.rs      # Training loop
│   │   └── losses.rs       # Quantile loss
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Бэктестинг
└── examples/
    ├── fetch_data.rs       # Загрузка данных
    ├── train_model.rs      # Обучение модели
    ├── predict.rs          # Прогнозирование
    └── backtest.rs         # Бэктестинг стратегии
```

Подробности в [rust_tft_trading/README.md](rust_tft_trading/README.md).

## Практические рекомендации

### Когда использовать TFT

**Хорошие use cases:**
- Прогнозирование на несколько горизонтов одновременно
- Когда нужна интерпретируемость
- Есть разные типы признаков (статические, будущие, наблюдаемые)
- Важна оценка неопределенности

**Не подходит для:**
- Высокочастотный трейдинг (медленный инференс)
- Очень короткие ряды (<100 точек)
- Когда достаточно простых моделей

### Оптимизация гиперпараметров

| Параметр | Рекомендуемый диапазон | Влияние |
|----------|----------------------|---------|
| `hidden_size` | 16-256 | Capacity модели |
| `attention_head_size` | 1-4 | Сложность attention |
| `dropout` | 0.1-0.3 | Регуляризация |
| `learning_rate` | 1e-4 - 1e-2 | Скорость сходимости |
| `max_encoder_length` | 48-336 | Контекст (часы) |

### Вычислительные требования

| Задача | GPU память | Время обучения | Время инференса |
|--------|------------|----------------|-----------------|
| Маленький TFT | 4GB | 1-2 часа | 10ms/sample |
| Средний TFT | 8GB | 4-8 часов | 30ms/sample |
| Большой TFT | 16GB | 12-24 часа | 100ms/sample |

## Ресурсы

### Статьи

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) — Lim et al., Google, 2019
- [Deep Learning for Time Series Forecasting: Tutorial and Literature Survey](https://arxiv.org/abs/2004.10240) — обзор DL методов
- [Probabilistic Forecasting with Temporal Convolutional Neural Network](https://arxiv.org/abs/1906.04397) — альтернативный подход

### Реализации

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — официальная библиотека с TFT
- [Darts](https://unit8co.github.io/darts/) — альтернативная реализация
- [GluonTS](https://github.com/awslabs/gluon-ts) — реализация от Amazon

### Связанные главы

- [Глава 19: RNN для временных рядов](../19_recurrent_neural_nets) — основы LSTM/GRU
- [Глава 20: Автоэнкодеры](../20_autoencoders_for_conditional_risk_factors) — латентные представления
- [Глава 25: Диффузионные модели](../25_diffusion_models_for_trading) — альтернативный подход к прогнозированию

---

## Уровень сложности

**Продвинутый**

Требуется понимание:
- Механизм внимания (Attention)
- LSTM и рекуррентные сети
- Quantile regression
- Основы портфельного управления
