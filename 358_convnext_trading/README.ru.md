# Глава 358: ConvNeXt для Трейдинга — Современные ConvNets, Конкурирующие с Transformers

## Обзор

ConvNeXt представляет собой эволюцию свёрточных нейронных сетей, включающую принципы проектирования из Vision Transformers (ViT) при сохранении эффективности и простоты свёрток. Эта глава исследует применение архитектуры ConvNeXt для предсказания финансовых временных рядов и генерации торговых сигналов на криптовалютных рынках.

Ключевая идея из оригинальной статьи "A ConvNet for the 2020s" (Liu et al., 2022) заключается в том, что многие архитектурные решения из Transformers могут быть успешно перенесены в ConvNets, создавая модели, которые конкурируют или превосходят производительность Transformer при большей эффективности.

## Торговая Стратегия

**Основной подход:** Использование архитектуры ConvNeXt для обработки многоканальных финансовых временных рядов (OHLCV + технические индикаторы) как 1D последовательностей, генерируя торговые сигналы для криптовалютных пар.

**Ключевые преимущества для трейдинга:**
1. **Эффективное моделирование дальних зависимостей** — Большие размеры ядер (7×1) захватывают паттерны на длительных временных горизонтах
2. **Иерархическое извлечение признаков** — Многоступенчатая архитектура улавливает паттерны на разных временных масштабах
3. **Вычислительная эффективность** — Более быстрый инференс чем у Transformers для торговли в реальном времени
4. **Устойчивость к шуму** — Depthwise separable свёртки снижают переобучение

**Преимущество:** ConvNeXt сочетает индуктивные смещения CNN (трансляционная эквивариантность, локальность) с современными техниками обучения, что делает его особенно подходящим для финансовых временных рядов, где паттерны повторяются в разные временные периоды.

## Основы Архитектуры ConvNeXt

### Ключевые Принципы Проектирования

1. **Макро-дизайн** — Соотношение стадий (3:3:9:3) и дизайн stem-ячейки
2. **ResNeXt-ификация** — Группированные свёртки с depthwise separable свёртками
3. **Инвертированный Bottleneck** — Расширение каналов, depthwise conv, сжатие
4. **Большие размеры ядер** — Использование ядер 7×7 (адаптировано к 7×1 для 1D временных рядов)
5. **Layer Normalization** — Замена BatchNorm на LayerNorm
6. **Меньше функций активации** — Один GELU на блок
7. **Отдельные слои даунсэмплинга** — Явный даунсэмплинг между стадиями

### Структура Блока ConvNeXt

```
Вход
  │
  ├─→ Depthwise Conv 7×1 (groups=C)
  │
  ├─→ LayerNorm
  │
  ├─→ Pointwise Conv 1×1 (расширение 4×)
  │
  ├─→ GELU
  │
  ├─→ Pointwise Conv 1×1 (сжатие)
  │
  └─→ Остаточное соединение → Выход
```

## Техническая Реализация

### Архитектура для Трейдинга

```
Вход: [batch, channels, sequence_length]
      [B, C, T] где C = OHLCV + индикаторы

Стадия 1: Stem + ConvNeXt блоки ×3
  - Patchify stem: Conv 4×1, stride 4
  - Каналы: 96 → 96

Стадия 2: Downsample + ConvNeXt блоки ×3
  - Downsample: LayerNorm + Conv 2×1, stride 2
  - Каналы: 96 → 192

Стадия 3: Downsample + ConvNeXt блоки ×9
  - Каналы: 192 → 384

Стадия 4: Downsample + ConvNeXt блоки ×3
  - Каналы: 384 → 768

Голова: Global Average Pool → LayerNorm → FC → Softmax/Sigmoid
  - Классификация: [Long, Short, Hold] или
  - Регрессия: Предсказание изменения цены
```

### Реализация на Rust

Реализация на Rust обеспечивает:
- Высокопроизводительный инференс для продакшн торговых систем
- Эффективность памяти для обработки больших исторических наборов данных
- Интеграцию с биржей Bybit для криптовалютных данных
- Модульный дизайн для лёгкой настройки

### Структура Проекта

```
358_convnext_trading/
├── README.md
├── README.ru.md
├── readme.simple.md
├── readme.simple.ru.md
└── rust/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── main.rs
    │   ├── convnext/
    │   │   ├── mod.rs
    │   │   ├── block.rs
    │   │   ├── model.rs
    │   │   └── layers.rs
    │   ├── data/
    │   │   ├── mod.rs
    │   │   ├── bybit.rs
    │   │   ├── features.rs
    │   │   └── dataset.rs
    │   ├── trading/
    │   │   ├── mod.rs
    │   │   ├── signals.rs
    │   │   ├── strategy.rs
    │   │   └── backtest.rs
    │   └── utils/
    │       ├── mod.rs
    │       └── metrics.rs
    └── examples/
        ├── fetch_data.rs
        ├── train_model.rs
        └── live_signals.rs
```

## Конвейер Данных

### Получение Данных с Bybit

```rust
// Получение OHLCV данных с Bybit
let client = BybitClient::new();
let candles = client.get_klines(
    "BTCUSDT",
    Interval::H1,
    start_time,
    end_time
).await?;
```

### Feature Engineering

| Группа признаков | Индикаторы |
|-----------------|------------|
| Цена | Open, High, Low, Close (нормализованные) |
| Объём | Volume, VWAP, Volume SMA |
| Моментум | RSI, MACD, Stochastic |
| Волатильность | ATR, Bollinger Bands, Keltner Channels |
| Тренд | EMA (9, 21, 50, 200), ADX |

### Построение Входного Тензора

```rust
// Форма: [batch, channels, sequence_length]
// Пример: [32, 20, 256] - 32 образца, 20 признаков, 256 временных шагов
let input = Tensor::zeros(&[batch_size, num_features, seq_length]);
```

## Обучение Модели

### Функции Потерь

1. **Классификация (Предсказание направления)**
   - CrossEntropyLoss для [Long, Short, Hold]
   - Взвешивание по частоте классов для балансировки

2. **Регрессия (Предсказание доходности)**
   - MSE для непрерывного предсказания доходности
   - Huber Loss для устойчивости к выбросам

### Конфигурация Обучения

```rust
let config = TrainingConfig {
    learning_rate: 4e-4,
    batch_size: 32,
    epochs: 100,
    weight_decay: 0.05,
    warmup_epochs: 5,
    label_smoothing: 0.1,
    drop_path_rate: 0.1,
    layer_scale_init: 1e-6,
};
```

### Аугментация Данных для Временных Рядов

1. **Time Warping** — Небольшое растяжение/сжатие временной оси
2. **Magnitude Scaling** — Случайное масштабирование значений
3. **Jittering** — Добавление небольшого гауссовского шума
4. **Window Slicing** — Случайная обрезка с паддингом

## Торговая Стратегия

### Генерация Сигналов

```rust
pub fn generate_signal(model: &ConvNeXt, features: &Tensor) -> Signal {
    let logits = model.forward(features);
    let probs = softmax(&logits, -1);

    let long_prob = probs[0];
    let short_prob = probs[1];
    let hold_prob = probs[2];

    if long_prob > CONFIDENCE_THRESHOLD && long_prob > short_prob {
        Signal::Long { confidence: long_prob }
    } else if short_prob > CONFIDENCE_THRESHOLD && short_prob > long_prob {
        Signal::Short { confidence: short_prob }
    } else {
        Signal::Hold
    }
}
```

### Размер Позиции

Критерий Келли с риск-менеджментом:

```rust
pub fn calculate_position_size(
    signal: &Signal,
    portfolio_value: f64,
    max_risk_per_trade: f64,  // например, 0.02 (2%)
) -> f64 {
    let edge = signal.confidence - 0.5;  // Преимущество над случайным
    let win_rate = signal.confidence;
    let win_loss_ratio = 1.5;  // Целевое соотношение риск/прибыль

    // Доля Келли
    let kelly_f = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio;

    // Половинный Келли для безопасности
    let position_fraction = kelly_f * 0.5;

    // Применение ограничения максимального риска
    let max_position = portfolio_value * max_risk_per_trade;

    (portfolio_value * position_fraction).min(max_position)
}
```

## Фреймворк Бэктестинга

### Метрики Производительности

```rust
pub struct BacktestMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_duration: Duration,
    pub total_trades: usize,
}
```

### Пример Результатов Бэктестинга (BTC/USDT, 1H)

| Метрика | Значение |
|---------|----------|
| Общая доходность | +47.3% |
| Sharpe Ratio | 1.82 |
| Sortino Ratio | 2.41 |
| Максимальная просадка | -12.4% |
| Винрейт | 58.7% |
| Profit Factor | 1.65 |
| Всего сделок | 342 |

*Примечание: Результаты приведены для иллюстрации. Прошлые результаты не гарантируют будущую доходность.*

## Варианты Модели

### ConvNeXt-Tiny (Рекомендуется для Трейдинга)
- Параметры: ~28M
- Каналы: [96, 192, 384, 768]
- Блоки: [3, 3, 9, 3]
- Лучше всего для: Инференса в реальном времени

### ConvNeXt-Small
- Параметры: ~50M
- Каналы: [96, 192, 384, 768]
- Блоки: [3, 3, 27, 3]
- Лучше всего для: Более высокой точности

### ConvNeXt-Base
- Параметры: ~89M
- Каналы: [128, 256, 512, 1024]
- Блоки: [3, 3, 27, 3]
- Лучше всего для: Исследований/ансамблей

## Ключевые Метрики

| Метрика | Описание | Цель |
|---------|----------|------|
| Точность направления | Правильное предсказание направления цены | >55% |
| Sharpe Ratio | Доходность с поправкой на риск | >1.5 |
| Sortino Ratio | Доходность с поправкой на нисходящий риск | >2.0 |
| Макс. просадка | Наибольшее снижение от пика до впадины | <15% |
| Profit Factor | Валовая прибыль / Валовый убыток | >1.5 |
| Винрейт | Процент прибыльных сделок | >55% |

## Зависимости (Rust)

```toml
[dependencies]
ndarray = "0.15"
ndarray-rand = "0.14"
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
hmac = "0.12"
sha2 = "0.10"
hex = "0.4"
anyhow = "1.0"
```

## Примеры Использования

### Получение Данных

```bash
cd 358_convnext_trading/rust
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --days 365
```

### Обучение Модели

```bash
cargo run --example train_model -- --data data/btcusdt_1h.json --epochs 100
```

### Генерация Сигналов в Реальном Времени

```bash
cargo run --example live_signals -- --symbol BTCUSDT --interval 1h
```

## Ожидаемые Результаты

1. **Реализация ConvNeXt** оптимизированная для 1D временных рядов
2. **Конвейер данных Bybit** для криптовалютных OHLCV данных
3. **Модуль feature engineering** с техническими индикаторами
4. **Фреймворк обучения** с правильной валидацией
5. **Движок бэктестинга** с комплексными метриками
6. **Генератор торговых сигналов** для работы в реальном времени

## Литература

1. **A ConvNet for the 2020s**
   - Liu, Z., et al. (2022)
   - URL: https://arxiv.org/abs/2201.03545

2. **Deep Residual Learning for Image Recognition**
   - He, K., et al. (2015)
   - URL: https://arxiv.org/abs/1512.03385

3. **An Image is Worth 16x16 Words: Transformers for Image Recognition**
   - Dosovitskiy, A., et al. (2020)
   - URL: https://arxiv.org/abs/2010.11929

4. **Financial Machine Learning**
   - López de Prado, M. (2018)
   - Advances in Financial Machine Learning

## Уровень Сложности

Продвинутый

Требования: Основы глубокого обучения, Архитектуры CNN, Анализ временных рядов, Программирование на Rust, Основы трейдинга
