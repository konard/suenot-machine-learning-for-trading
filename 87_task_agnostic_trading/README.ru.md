# Глава 87: Задаче-агностичная торговля

## Обзор

Задаче-агностичная торговля решает фундаментальное ограничение торговых систем на основе машинного обучения: традиционные подходы требуют отдельных моделей для каждой торговой задачи — одну для предсказания трендов, другую для прогнозирования волатильности, ещё одну для определения режима рынка и так далее. Каждая модель обучает свои собственные признаки с нуля, что приводит к избыточным вычислениям, фрагментации знаний и плохой обобщающей способности.

Задаче-агностичное обучение представлений решает эту проблему путём обучения единого **универсального кодировщика**, который отображает сырые рыночные данные в общее пространство представлений, полезное для *всех* нижестоящих торговых задач одновременно. Лёгкие задаче-специфичные головы затем декодируют эти представления для каждой задачи, а гармонизация градиентов обеспечивает сбалансированное мультизадачное обучение.

## Содержание

1. [Введение](#введение)
2. [Теоретические основы](#теоретические-основы)
3. [Архитектура](#архитектура)
4. [Мультизадачное обучение для торговли](#мультизадачное-обучение-для-торговли)
5. [Гармонизация градиентов](#гармонизация-градиентов)
6. [Слияние решений](#слияние-решений)
7. [Стратегия реализации](#стратегия-реализации)
8. [Интеграция с Bybit](#интеграция-с-bybit)
9. [Фреймворк бэктестирования](#фреймворк-бэктестирования)
10. [Метрики производительности](#метрики-производительности)
11. [Литература](#литература)

---

## Введение

В количественном трейдинге модель, обученная для предсказания тренда, изучает такие признаки, как индикаторы моментума и пересечения скользящих средних. Модель волатильности изучает признаки вроде реализованной дисперсии и ATR. Модель режимов изучает признаки вроде затухания автокорреляции и формы распределения. Но эти признаки существенно пересекаются — все они описывают одну и ту же базовую динамику рынка с разных точек зрения.

### Проблема задаче-специфичных моделей

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Traditional Approach: Separate Models                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Market Data                                                            │
│        │                                                                     │
│        ├──────────────────┐                                                  │
│        │                  │                                                  │
│   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐       ┌────▼────┐        │
│   │ Trend   │       │ Vol     │       │ Regime  │       │ Risk    │        │
│   │ Model   │       │ Model   │       │ Model   │       │ Model   │        │
│   │ (Full)  │       │ (Full)  │       │ (Full)  │       │ (Full)  │        │
│   └────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘        │
│        │                  │                  │                  │            │
│   Up/Down/Side       σ forecast        Trending/MR        Low/Med/High     │
│                                                                              │
│   Problem: 4x redundant feature learning, no shared knowledge               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Задаче-агностичное решение

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Task-Agnostic Approach: Shared Encoder                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Market Data                                                            │
│        │                                                                     │
│   ┌────▼──────────────────────────────────────────────────┐                 │
│   │           Universal Encoder (Shared)                   │                 │
│   │   Input → [64] → BN → ReLU → [32] → BN → ReLU → [16] │                 │
│   └────┬──────────────────────────────────────────────────┘                 │
│        │  Shared Representations                                             │
│        ├──────────────┬──────────────┬──────────────┐                       │
│   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐                   │
│   │ Trend   │   │ Vol     │   │ Regime  │   │ Risk    │                   │
│   │ Head    │   │ Head    │   │ Head    │   │ Head    │                   │
│   │ (Light) │   │ (Light) │   │ (Light) │   │ (Light) │                   │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                   │
│        │              │              │              │                        │
│        └──────────────┴──────┬───────┴──────────────┘                       │
│                              │                                               │
│                     Decision Fusion                                          │
│                              │                                               │
│                    Unified Trading Signal                                    │
│                                                                              │
│   Advantage: Shared features, cross-task knowledge transfer                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Ключевые преимущества

| Аспект | Задаче-специфичный | Задаче-агностичный |
|--------|---------------|---------------|
| Параметры | 4 x полная модель | 1 кодировщик + 4 лёгкие головы |
| Обучение признаков | Избыточное | Общее |
| Межзадачный перенос | Отсутствует | Автоматический |
| Время инференса | 4 x прямой проход | 1 кодирование + 4 прохода голов |
| Согласованность | Независимые сигналы | Согласованные решения |
| Новые задачи | Полное переобучение | Добавить голову |

---

## Теоретические основы

### Фреймворк мультизадачного обучения

Для набора из $T$ торговых задач $\{\mathcal{T}_1, \ldots, \mathcal{T}_T\}$ мы обучаем:

1. Общий кодировщик $f_\theta: \mathbb{R}^d \to \mathbb{R}^k$, отображающий входные признаки в представления
2. Задаче-специфичные головы $g_{\phi_t}: \mathbb{R}^k \to \mathbb{R}^{c_t}$ для каждой задачи $t$

Мультизадачная целевая функция:

$$\min_{\theta, \{\phi_t\}} \sum_{t=1}^{T} w_t \cdot \mathcal{L}_t(g_{\phi_t}(f_\theta(X)), Y_t)$$

где $w_t$ — веса задач, а $\mathcal{L}_t$ — функция потерь для задачи $t$.

### Задаче-агностичные представления

Представление является **задаче-агностичным**, если оно отражает фундаментальную структуру данных без смещения в сторону какой-либо конкретной нижестоящей задачи. Формально:

$$I(Z; Y_t) \approx I(Z; Y_{t'}) \quad \forall t, t' \in \{1, \ldots, T\}$$

где $I(Z; Y_t)$ — взаимная информация между представлением $Z$ и метками задачи $Y_t$.

Это означает, что кодировщик извлекает признаки, одинаково полезные для предсказания тренда, прогнозирования волатильности, определения режима и оценки рисков.

### Торговые задачи

Мы определяем четыре основные торговые задачи:

1. **Предсказание тренда** (Классификация: 3 класса)
   - Вверх / Боковик / Вниз
   - Функция потерь: перекрёстная энтропия

2. **Прогноз волатильности** (Регрессия: 1 выход)
   - Предсказание реализованной волатильности следующего периода
   - Функция потерь: среднеквадратичная ошибка (MSE)

3. **Определение режима** (Классификация: 4 класса)
   - Трендовый / Возврат к среднему / Волатильный / Спокойный
   - Функция потерь: перекрёстная энтропия

4. **Оценка рисков** (Классификация: 3 класса)
   - Низкий / Средний / Высокий риск
   - Функция потерь: перекрёстная энтропия

---

## Архитектура

### Универсальный кодировщик

Кодировщик представляет собой нейронную сеть прямого распространения с:

- **Пакетной нормализацией** после каждого скрытого слоя для стабильности обучения
- **Активациями ReLU** для нелинейности
- **Остаточными связями** при совпадении размерностей
- **L2-нормализацией** выходных представлений для предотвращения коллапса

```
Input (d=20 features)
    │
    ▼
┌─────────────────────┐
│  Linear(20 → 64)    │
│  BatchNorm1d(64)     │
│  ReLU                │
│  Dropout(0.1)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linear(64 → 32)    │
│  BatchNorm1d(32)     │
│  ReLU                │
│  Dropout(0.1)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linear(32 → 16)    │
│  L2 Normalize        │
└─────────┬───────────┘
          │
          ▼
  Representation (k=16)
```

### Головы задач

Каждая голова задачи — это двухслойная сеть:

```python
class TaskHead(nn.Module):
    def __init__(self, repr_dim, hidden_dim, output_dim):
        self.hidden = Linear(repr_dim, hidden_dim)
        self.output = Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = relu(self.hidden(z))
        out = self.output(h)
        return softmax(out)  # or sigmoid for regression
```

### Извлечение признаков

Мы извлекаем 19 задаче-агностичных признаков из данных OHLCV:

| # | Признак | Категория | Описание |
|---|---------|----------|-------------|
| 1 | return_1d | Доходность | Дневная доходность цены |
| 2 | return_5d | Доходность | 5-дневная кумулятивная доходность |
| 3 | rsi | Моментум | Индекс относительной силы (нормализованный) |
| 4 | macd_signal | Моментум | Гистограмма MACD |
| 5 | ma_crossover | Моментум | Сигнал пересечения скользящих средних |
| 6 | realized_vol | Волатильность | Реализованная волатильность (стд. логарифмических доходностей) |
| 7 | bb_position | Волатильность | Положение внутри полос Боллинджера |
| 8 | atr_ratio | Волатильность | Средний истинный диапазон / Цена |
| 9 | volume_ratio | Объём | Текущий объём / 20-дневное среднее |
| 10 | volume_trend | Объём | Краткосрочный vs долгосрочный объём |
| 11 | body_ratio | Свеча | Тело свечи / общий диапазон |
| 12 | upper_shadow | Свеча | Верхняя тень / общий диапазон |
| 13 | lower_shadow | Свеча | Нижняя тень / общий диапазон |
| 14 | range_pct | Диапазон | Диапазон максимум-минимум в процентах |
| 15 | close_position | Диапазон | Положение цены закрытия в диапазоне |
| 16 | trend_strength | Тренд | Наклон линейной регрессии |
| 17 | trend_consistency | Тренд | Доля дней роста |
| 18 | return_skewness | Распределение | Асимметрия недавних доходностей |
| 19 | return_kurtosis | Распределение | Эксцесс доходностей |

---

## Мультизадачное обучение для торговли

### Процесс обучения

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Task Training Loop                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each epoch:                                                         │
│    For each mini-batch:                                                  │
│                                                                          │
│    1. Forward pass through encoder                                       │
│       features → representations                                        │
│                                                                          │
│    2. Forward pass through each task head                                │
│       representations → predictions[task]                                │
│                                                                          │
│    3. Compute losses for each task                                       │
│       L_trend = CrossEntropy(pred_trend, label_trend)                   │
│       L_vol   = MSE(pred_vol, label_vol)                                │
│       L_regime = CrossEntropy(pred_regime, label_regime)                │
│       L_risk  = CrossEntropy(pred_risk, label_risk)                     │
│                                                                          │
│    4. Harmonize gradients                                                │
│       w_t = GradNorm(L_1, ..., L_T)                                    │
│                                                                          │
│    5. Compute total loss                                                 │
│       L_total = Σ w_t * L_t                                             │
│                                                                          │
│    6. Backpropagate and update                                           │
│       θ ← θ - lr * ∇_θ L_total                                         │
│       φ_t ← φ_t - lr * ∇_{φ_t} L_total                                │
│                                                                          │
│  Early stopping on validation loss                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Гармонизация градиентов

Ключевая проблема мультизадачного обучения — **конфликт градиентов**: градиенты от разных задач могут указывать в противоположных направлениях, заставляя кодировщик осциллировать или сходиться к решению, отдающему предпочтение одной задаче в ущерб остальным.

### Алгоритм GradNorm

GradNorm динамически корректирует веса задач так, чтобы все задачи обучались с одинаковой скоростью:

1. Отслеживает отношение потерь $r_t = L_t / L_t^{(0)}$ (текущая / начальная)
2. Вычисляет относительную скорость обучения: $\tilde{r}_t = r_t / \bar{r}$
3. Обновляет веса: $w_t \propto \tilde{r}_t^{-\alpha}$

Задачи, обучающиеся медленнее, получают более высокие веса, обеспечивая сбалансированный прогресс.

### PCGrad (Проекция конфликтующих градиентов)

Для конфликтующих пар градиентов:

$$g_i^{PC} = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j \quad \text{if } g_i \cdot g_j < 0$$

Этот метод проецирует конфликтующую компоненту, сохраняя кооперативную компоненту.

### Взвешивание по неопределённости

Взвешивание задач по обратной неопределённости предсказания:

$$w_t = \frac{1}{2\sigma_t^2}$$

где $\sigma_t$ — обученная неопределённость для задачи $t$. Задачи с большей неопределённостью получают меньший вес.

---

## Слияние решений

После получения предсказаний от всех голов задач мы объединяем их в единое торговое решение.

### Слияние по взвешенному среднему

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Decision Fusion Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trend Head (w=0.35):                                                │
│    [Up: 0.6, Side: 0.3, Down: 0.1]                                 │
│    → signal += 0.35 * (0.6 - 0.1) = +0.175                        │
│                                                                      │
│  Volatility Head (w=0.20):                                           │
│    [Predicted: 0.3]                                                  │
│    → confidence *= (1 - 0.3*0.3) = 0.91                            │
│    → risk_level = 0.3                                                │
│                                                                      │
│  Regime Head (w=0.25):                                               │
│    [Trend: 0.5, MR: 0.2, Vol: 0.2, Calm: 0.1]                     │
│    → regime = "Trending"                                             │
│    → signal *= 1.2 (boost for trending regime)                      │
│                                                                      │
│  Risk Head (w=0.20):                                                 │
│    [Low: 0.6, Med: 0.3, High: 0.1]                                 │
│    → signal *= (1 - 0.1 * 0.5) = 0.95                              │
│                                                                      │
│  Final: signal=+0.20, confidence=0.72, regime=Trending, risk=0.30   │
│  → Signal Type: BUY                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Классификация сигналов

| Значение сигнала | Тип | Позиция |
|-------------|------|----------|
| >= 0.40 | Сильная покупка | 100% длинная |
| >= 0.15 | Покупка | 50% длинная |
| -0.15 до 0.15 | Удержание | Нет позиции |
| <= -0.15 | Продажа | 50% короткая |
| <= -0.40 | Сильная продажа | 100% короткая |

---

## Стратегия реализации

### Python (PyTorch)

```python
from task_agnostic_trading import (
    TaskAgnosticModel, EncoderConfig, MultiTaskTrainer,
    TrainerConfig, FeatureExtractor, DecisionFusion,
    SignalGenerator, BacktestEngine, BybitClient,
)
import asyncio

# 1. Fetch data
client = BybitClient()
klines = asyncio.run(client.fetch_klines("BTCUSDT", "60", 250))

# 2. Extract features
extractor = FeatureExtractor()
features = extractor.extract_all(klines)

# 3. Create model
config = EncoderConfig(input_dim=19, hidden_dims=[64, 32], repr_dim=16)
model = TaskAgnosticModel(config)

# 4. Train
trainer = MultiTaskTrainer(TrainerConfig(epochs=100))
result = trainer.train(model, features, labels)

# 5. Predict and fuse
outputs = model.predict_single(features)
fusion = DecisionFusion()
fused = fusion.fuse(outputs)

# 6. Generate signals and backtest
signals = SignalGenerator().generate(fused)
bt = BacktestEngine().run(signals, prices)
```

### Rust

```rust
use task_agnostic_trading::prelude::*;

// Create model
let config = TaskAgnosticConfig::default()
    .with_input_dim(19)
    .with_repr_dim(16);
let model = TaskAgnosticModel::new(config);

// Predict
let fused = model.predict(&features);

// Generate signals
let gen = SignalGenerator::new(SignalConfig::default());
let signals = gen.generate(&fused);

// Backtest
let engine = BacktestEngine::new(BacktestConfig::default());
let result = engine.run(&signals, &prices);
println!("Sharpe: {:.2}", result.metrics.sharpe_ratio);
```

---

## Интеграция с Bybit

Система поддерживает получение данных криптовалют через API биржи Bybit:

```python
client = BybitClient()

# Single symbol
klines = await client.fetch_klines("BTCUSDT", interval="60", limit=200)

# Multiple symbols concurrently
multi_data = await client.fetch_multi_klines(
    ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    interval="60", limit=200
)
```

### Поддерживаемые типы данных

- **Свечи (OHLCV)**: Данные свечного графика с различными интервалами
- **Тикеры**: 24-часовые рыночные сводки
- **Стаканы заявок**: Глубина спроса и предложения
- **Ставки финансирования**: Финансирование бессрочных контрактов

---

## Фреймворк бэктестирования

Движок бэктестирования моделирует торговлю с учётом:

- **Транзакционных издержек**: Настраиваемая стоимость за сделку (по умолчанию 0.1%)
- **Проскальзывания**: Отклонение цены исполнения (по умолчанию 0.05%)
- **Размера позиции**: На основе силы сигнала, уверенности и риска

### Метрики производительности

| Метрика | Описание |
|--------|-------------|
| Общая доходность | Кумулятивная доходность портфеля |
| Годовая доходность | Геометрическая годовая доходность |
| Коэффициент Шарпа | Доходность с поправкой на риск (годовая) |
| Коэффициент Сортино | Доходность с поправкой на нижний риск |
| Максимальная просадка | Наибольшее снижение от пика до дна |
| Коэффициент Кальмара | Доходность / Максимальная просадка |
| Доля выигрышных сделок | Процент прибыльных сделок |
| Профит-фактор | Валовая прибыль / Валовый убыток |

---

## Метрики производительности

### Оценка модели

- **Задачи классификации**: Перекрёстная энтропия, точность, F1-мера
- **Задачи регрессии**: MSE, MAE, R²
- **Качество представлений**: Отношение межклассового / внутриклассового расстояния
- **Баланс задач**: Дисперсия весов между задачами (меньше = более сбалансированно)

### Оценка стратегии

- **Доходность с поправкой на риск**: Коэффициенты Шарпа, Сортино, Кальмара
- **Анализ просадок**: Максимальная просадка, время восстановления
- **Статистика сделок**: Доля выигрышных, профит-фактор, средняя сделка
- **Сравнение с бенчмарком**: Стратегия vs купи и держи

---

## Литература

1. **Task-Agnostic Representation Learning**
   - Lan et al., 2019
   - URL: https://arxiv.org/abs/1907.12157

2. **GradNorm: Gradient Normalization for Adaptive Loss Balancing**
   - Chen et al., 2018
   - URL: https://arxiv.org/abs/1711.02257

3. **Multi-Task Learning Using Uncertainty to Weigh Losses**
   - Kendall et al., 2018
   - URL: https://arxiv.org/abs/1705.07115

4. **Gradient Surgery for Multi-Task Learning (PCGrad)**
   - Yu et al., 2020
   - URL: https://arxiv.org/abs/2001.06782

5. **An Overview of Multi-Task Learning in Deep Neural Networks**
   - Ruder, 2017
   - URL: https://arxiv.org/abs/1706.05098

---

## Структура каталога

```
87_task_agnostic_trading/
├── README.md                          # This file
├── README.ru.md                       # Russian translation
├── readme.simple.md                   # Simplified explanation (English)
├── readme.simple.ru.md                # Simplified explanation (Russian)
├── README.specify.md                  # Task specification
├── Cargo.toml                         # Rust project configuration
├── Cargo.lock                         # Dependency lock file
├── src/                               # Rust source code
│   ├── lib.rs                         # Library root
│   ├── model/                         # Model components
│   │   ├── mod.rs                     # Module root
│   │   ├── encoder.rs                 # Universal encoder
│   │   ├── task_heads.rs              # Task-specific heads
│   │   └── fusion.rs                  # Decision fusion
│   ├── data/                          # Data handling
│   │   ├── mod.rs                     # Module root
│   │   ├── bybit.rs                   # Bybit API client
│   │   ├── features.rs                # Feature extraction
│   │   └── types.rs                   # Data types
│   ├── training/                      # Training logic
│   │   ├── mod.rs                     # Module root
│   │   ├── multi_task.rs              # Multi-task trainer
│   │   └── gradient.rs                # Gradient harmonization
│   ├── strategy/                      # Trading strategy
│   │   ├── mod.rs                     # Module root
│   │   ├── regime.rs                  # Regime classification
│   │   └── signals.rs                 # Signal generation
│   └── backtest/                      # Backtesting
│       ├── mod.rs                     # Module root
│       └── engine.rs                  # Backtest engine
├── examples/                          # Rust examples
│   ├── basic_task_agnostic.rs         # Basic usage
│   ├── multi_task_strategy.rs         # Full strategy pipeline
│   └── universal_encoder.rs           # Encoder analysis
└── python/                            # Python implementation
    └── task_agnostic_trading.py       # Complete Python module
```

## Быстрый старт

### Python

```bash
cd 87_task_agnostic_trading/python
pip install torch numpy aiohttp
python task_agnostic_trading.py
```

### Rust

```bash
cd 87_task_agnostic_trading
cargo test
cargo run --example basic_task_agnostic
cargo run --example multi_task_strategy
cargo run --example universal_encoder
```
