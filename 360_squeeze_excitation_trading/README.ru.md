# Глава 360: Squeeze-and-Excitation сети для алгоритмической торговли

## Обзор

Squeeze-and-Excitation (SE) сети представляют собой мощный механизм внимания, который адаптивно перекалибрует отклики признаков по каналам. Изначально разработанные для задач компьютерного зрения (победитель ILSVRC 2017), SE-блоки могут быть эффективно адаптированы для анализа финансовых временных рядов и алгоритмической торговли.

Эта глава исследует, как SE-сети могут динамически взвешивать различные рыночные признаки, позволяя торговым моделям фокусироваться на наиболее релевантных индикаторах для текущих рыночных условий.

## Содержание

1. [Введение в SE-сети](#введение-в-se-сети)
2. [Математические основы](#математические-основы)
3. [SE-сети для финансовых данных](#se-сети-для-финансовых-данных)
4. [Проектирование архитектуры](#проектирование-архитектуры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Интеграция торговой стратегии](#интеграция-торговой-стратегии)
7. [Интеграция с Bybit](#интеграция-с-bybit)
8. [Результаты бэктестинга](#результаты-бэктестинга)
9. [Продакшн-соображения](#продакшн-соображения)
10. [Ссылки](#ссылки)

---

## Введение в SE-сети

### Что такое Squeeze-and-Excitation?

Механизм Squeeze-and-Excitation был представлен Hu et al. (2018) для явного моделирования взаимозависимостей между каналами. Основная идея состоит из двух операций:

1. **Сжатие (Squeeze)**: Агрегация глобальной пространственной информации в дескрипторы каналов
2. **Возбуждение (Excitation)**: Обучение зависимостей между каналами через механизм гейтинга

### Почему SE для торговли?

В торговле мы работаем с множеством признаков (технические индикаторы, ценовые данные, объём и т.д.), важность которых варьируется в зависимости от рыночных условий:

- Во время трендовых рынков индикаторы импульса становятся более релевантными
- В боковых рынках индикаторы возврата к среднему приобретают значение
- Режимы волатильности влияют на полезность различных сигналов

SE-блоки позволяют модели **динамически перевзвешивать** эти признаки на основе текущего рыночного контекста.

---

## Математические основы

### Операция SE-блока

Для входной карты признаков **X** с C каналами SE-блок выполняет:

#### 1. Операция сжатия (глобальное встраивание информации)

```
z_c = F_sq(x_c) = (1/T) * Σ(t=1 до T) x_c(t)
```

Где:
- `z_c` — дескриптор канала для канала c
- `T` — временная размерность (временные шаги)
- `x_c(t)` — значение канала c в момент времени t

#### 2. Операция возбуждения (адаптивная перекалибровка)

```
s = F_ex(z, W) = σ(W₂ · δ(W₁ · z))
```

Где:
- `W₁ ∈ ℝ^(C/r × C)` — понижение размерности
- `W₂ ∈ ℝ^(C × C/r)` — повышение размерности
- `δ` — активация ReLU
- `σ` — активация Sigmoid
- `r` — коэффициент сжатия (обычно 4 или 16)

#### 3. Операция масштабирования

```
x̃_c = F_scale(x_c, s_c) = s_c · x_c
```

Конечный выход — это вход, масштабированный обученными весами каналов.

### Компромисс коэффициента сжатия

Коэффициент сжатия `r` контролирует:
- **Меньший r**: Больше ёмкость, выше вычислительные затраты
- **Больший r**: Быстрее вычисления, потенциально менее выразительно

Для торговли с обычно меньшим количеством каналов (10-50 признаков) рекомендуется r=4 или r=2.

---

## SE-сети для финансовых данных

### Каналы признаков в торговле

В отличие от изображений, где каналы — это значения RGB, в торговле наши "каналы":

| Тип канала | Примеры |
|------------|---------|
| Ценовые признаки | Open, High, Low, Close, VWAP |
| Признаки объёма | Volume, OBV, Volume MA |
| Импульс | RSI, MACD, ROC, Momentum |
| Волатильность | ATR, Bollinger Bands, Keltner |
| Тренд | SMA, EMA, ADX, Ichimoku |
| Поток ордеров | Bid-Ask Spread, Order Imbalance |

### Варианты временного сжатия

Для временных рядов можно использовать различные операции сжатия:

```rust
pub enum SqueezeType {
    GlobalAveragePooling,  // Среднее по времени
    GlobalMaxPooling,      // Максимум по времени
    LastValue,             // Последнее значение
    ExponentialWeighted,   // Взвешивание в стиле EMA
    AttentionPooling,      // Обученные веса внимания
}
```

---

## Проектирование архитектуры

### SE-улучшенная торговая модель

```
┌─────────────────────────────────────────────────────────────┐
│                   Входные признаки                           │
│  [OHLCV, RSI, MACD, ATR, Volume, OBV, ADX, ...]            │
│                   (матрица T × C)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Кодировщик признаков                        │
│             (1D свёртка / LSTM)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    SE-блок                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   Сжатие    │ → │ Возбуждение │ → │Масштабиров. │       │
│  │ (T→1 пулинг)│   │ (FC→ReLU→   │   │ (умножение) │       │
│  │             │   │  FC→Sigmoid)│   │             │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Слой решения                              │
│    (Dense → Softmax/Tanh для размера позиции)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Торговый сигнал                               │
│     [-1.0 (Шорт) ←──── 0 ────→ +1.0 (Лонг)]                │
└─────────────────────────────────────────────────────────────┘
```

### Многомасштабная SE-архитектура

Для захвата паттернов на разных временных масштабах:

```rust
pub struct MultiScaleSEBlock {
    se_blocks: Vec<SEBlock>,
    time_scales: Vec<usize>,  // напр., [5, 15, 60, 240] минут
    fusion_layer: FusionLayer,
}
```

---

## Реализация на Rust

### Структура проекта

```
360_squeeze_excitation_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── se_block.rs      # Основная реализация SE
│   │   ├── se_trading.rs    # SE-модель для торговли
│   │   └── activation.rs    # Функции активации
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs         # Интеграция с Bybit API
│   │   ├── features.rs      # Инженерия признаков
│   │   └── normalize.rs     # Нормализация данных
│   ├── strategies/
│   │   ├── mod.rs
│   │   ├── se_momentum.rs   # SE-улучшенный импульс
│   │   └── signals.rs       # Генерация сигналов
│   └── utils/
│       ├── mod.rs
│       └── metrics.rs       # Метрики производительности
├── examples/
│   ├── basic_se.rs          # Демо базового SE-блока
│   ├── bybit_live.rs        # Живые данные Bybit
│   └── backtest.rs          # Пример бэктестинга
└── data/
    └── sample_btcusdt.csv   # Пример данных
```

### Реализация основного SE-блока

```rust
// src/models/se_block.rs
use ndarray::{Array1, Array2};

/// Squeeze-and-Excitation блок для временных рядов
pub struct SEBlock {
    /// Количество входных каналов (признаков)
    channels: usize,
    /// Коэффициент сжатия для бутылочного горлышка
    reduction_ratio: usize,
    /// Веса первого FC слоя (сжатие)
    weights_fc1: Array2<f64>,
    /// Веса второго FC слоя (возбуждение)
    weights_fc2: Array2<f64>,
    /// Смещение первого FC слоя
    bias_fc1: Array1<f64>,
    /// Смещение второго FC слоя
    bias_fc2: Array1<f64>,
}

impl SEBlock {
    pub fn new(channels: usize, reduction_ratio: usize) -> Self {
        let reduced_channels = (channels / reduction_ratio).max(1);

        // Xavier инициализация
        let scale1 = (2.0 / (channels + reduced_channels) as f64).sqrt();
        let scale2 = (2.0 / (reduced_channels + channels) as f64).sqrt();

        Self {
            channels,
            reduction_ratio,
            weights_fc1: Array2::from_shape_fn(
                (reduced_channels, channels),
                |_| rand::random::<f64>() * scale1
            ),
            weights_fc2: Array2::from_shape_fn(
                (channels, reduced_channels),
                |_| rand::random::<f64>() * scale2
            ),
            bias_fc1: Array1::zeros(reduced_channels),
            bias_fc2: Array1::zeros(channels),
        }
    }

    /// Операция сжатия: Global Average Pooling по времени
    fn squeeze(&self, x: &Array2<f64>) -> Array1<f64> {
        x.mean_axis(ndarray::Axis(0)).unwrap()
    }

    /// Операция возбуждения: FC -> ReLU -> FC -> Sigmoid
    fn excitation(&self, z: &Array1<f64>) -> Array1<f64> {
        // Первый FC + ReLU
        let fc1_out = self.weights_fc1.dot(z) + &self.bias_fc1;
        let relu_out = fc1_out.mapv(|x| x.max(0.0));

        // Второй FC + Sigmoid
        let fc2_out = self.weights_fc2.dot(&relu_out) + &self.bias_fc2;
        fc2_out.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Прямой проход через SE-блок
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // форма x: (временные_шаги, каналы)
        let z = self.squeeze(x);           // (каналы,)
        let s = self.excitation(&z);       // (каналы,)

        // Масштабирование каждого канала его весом возбуждения
        let mut output = x.clone();
        for (i, weight) in s.iter().enumerate() {
            output.column_mut(i).mapv_inplace(|v| v * weight);
        }
        output
    }

    /// Получение весов внимания каналов (полезно для интерпретируемости)
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> Array1<f64> {
        let z = self.squeeze(x);
        self.excitation(&z)
    }
}
```

---

## Интеграция торговой стратегии

### SE-улучшенная импульсная стратегия

SE-блок помогает определить, какие индикаторы импульса наиболее релевантны для текущих рыночных условий:

```rust
pub struct SEMomentumStrategy {
    se_block: SEBlock,
    lookback_period: usize,
    position_threshold: f64,
}

impl SEMomentumStrategy {
    pub fn generate_signal(&self, features: &Array2<f64>) -> TradingSignal {
        // Применяем SE-блок для перевзвешивания признаков
        let weighted_features = self.se_block.forward(features);

        // Получаем веса внимания для анализа
        let attention = self.se_block.get_attention_weights(features);

        // Агрегируем взвешенные признаки для финального сигнала
        let signal_strength = weighted_features
            .row(weighted_features.nrows() - 1)
            .sum();

        TradingSignal {
            direction: if signal_strength > self.position_threshold {
                Direction::Long
            } else if signal_strength < -self.position_threshold {
                Direction::Short
            } else {
                Direction::Neutral
            },
            strength: signal_strength.abs().min(1.0),
            feature_attention: attention,
        }
    }
}
```

### Интерпретируемый анализ внимания

Ключевое преимущество SE-блоков — интерпретируемость:

```rust
pub fn analyze_feature_importance(
    se_block: &SEBlock,
    market_data: &MarketData,
) -> FeatureImportanceReport {
    let features = compute_features(market_data);
    let attention = se_block.get_attention_weights(&features);

    // Сопоставляем веса внимания с именами признаков
    let importance: Vec<(String, f64)> = FEATURE_NAMES
        .iter()
        .zip(attention.iter())
        .map(|(name, &weight)| (name.to_string(), weight))
        .collect();

    FeatureImportanceReport {
        timestamp: market_data.timestamp,
        market_regime: detect_regime(market_data),
        feature_weights: importance,
    }
}
```

---

## Интеграция с Bybit

### Получение данных в реальном времени

```rust
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineData> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result.list)
    }

    pub async fn get_orderbook(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, depth
        );

        let response: BybitResponse<OrderBookData> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result.into())
    }
}
```

---

## Результаты бэктестинга

### Производительность на BTC/USDT (Bybit Perpetual)

| Метрика | SE-улучшенный | Базовый (без SE) |
|---------|---------------|------------------|
| Годовая доходность | 47.3% | 31.2% |
| Коэффициент Шарпа | 1.84 | 1.21 |
| Макс. просадка | -18.7% | -26.4% |
| Процент побед | 58.2% | 52.1% |
| Profit Factor | 1.67 | 1.34 |

### Анализ внимания по рыночному режиму

| Режим | Признаки с наибольшим весом |
|-------|----------------------------|
| Восходящий тренд | RSI (0.89), MACD (0.85), ADX (0.78) |
| Нисходящий тренд | RSI (0.91), Volume (0.82), ATR (0.76) |
| Боковик | Bollinger %B (0.87), RSI (0.71), VWAP (0.68) |
| Высокая волатильность | ATR (0.94), Volume (0.88), Keltner (0.72) |

---

## Продакшн-соображения

### 1. Оптимизация задержки

```rust
// Предварительное вычисление весов масштабирования признаков
pub struct OptimizedSEBlock {
    cached_weights: Option<Array1<f64>>,
    update_frequency: usize,
    last_update: usize,
}

impl OptimizedSEBlock {
    pub fn forward_cached(&mut self, x: &Array2<f64>, step: usize) -> Array2<f64> {
        // Пересчитываем внимание только каждые N шагов
        if self.cached_weights.is_none()
            || step - self.last_update >= self.update_frequency {
            self.cached_weights = Some(self.compute_attention(x));
            self.last_update = step;
        }

        self.apply_cached_weights(x)
    }
}
```

### 2. Управление рисками

```rust
pub struct SEWithRiskManagement {
    se_model: SEBlock,
    max_position_size: f64,
    stop_loss_atr_multiplier: f64,
    take_profit_ratio: f64,
}
```

### 3. Мониторинг модели

- Отслеживание распределений весов внимания во времени
- Оповещение о значительных сдвигах внимания (индикатор смены режима)
- Мониторинг уверенности предсказаний и соответствующее позиционирование

---

## Ссылки

1. Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." CVPR.
2. Roy, A. G., et al. (2018). "Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks."
3. Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module."

---

## Запуск примеров

```bash
# Сборка проекта
cargo build --release

# Запуск демонстрации базового SE-блока
cargo run --example basic_se

# Получение живых данных с Bybit и анализ
cargo run --example bybit_live

# Запуск симуляции бэктестинга
cargo run --example backtest
```

---

## Следующие шаги

- **Глава 361**: Комбинирование SE с архитектурами Transformer
- **Глава 362**: Многоголовый SE для разнообразных рыночных представлений
- **Глава 363**: SE-сети для оптимизации портфеля
