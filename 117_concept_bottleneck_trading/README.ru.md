# Глава 117: Концептуальные Бутылочные Модели в Трейдинге

## Обзор

Концептуальные бутылочные модели (Concept Bottleneck Models, CBM) — это интерпретируемые архитектуры машинного обучения, которые делают предсказания через понятные человеку промежуточные концепции. В трейдинге CBM обеспечивают прозрачность, сначала предсказывая интерпретируемые рыночные концепции (такие как сила тренда, режим волатильности или моментум), а затем принимая торговые решения.

Этот подход решает критическое ограничение "черных ящиков" в финансах: невозможность понять, *почему* модель принимает конкретные торговые решения.

## Ключевые Концепции

### Что такое Concept Bottleneck Model?

CBM состоит из двух этапов:
1. **Предиктор концепций**: Преобразует входные данные X в интерпретируемые концепции C
2. **Предиктор меток**: Преобразует концепции C в финальные предсказания Y

```
Сырые данные (X) → [Модель концепций] → Концепции (C) → [Модель задачи] → Предсказания (Y)
```

### Преимущества для Трейдинга

- **Интерпретируемость**: Понимание, какие рыночные условия влияют на торговые сигналы
- **Интервенция**: Возможность исправить неверные концепции во время инференса
- **Отладка**: Определение когда и почему модель ошибается
- **Регуляторное соответствие**: Объяснимые решения для аудита
- **Управление рисками**: Человеческий контроль над автоматической торговлей

## Архитектура

### Примеры Торговых Концепций

| Концепция | Описание | Вычисление |
|-----------|----------|------------|
| `trend_strength` | Сила текущего тренда | Наклон линейной регрессии за период |
| `volatility_regime` | Режим волатильности | Скользящее std vs исторические перцентили |
| `momentum_signal` | Сигнал моментума | RSI, пересечения MACD |
| `volume_profile` | Объем относительно среднего | Текущий объем / SMA(объем) |
| `support_resistance` | Близость к уровням | Расстояние до пивот-точек |
| `market_regime` | Бычий/Медвежий/Боковой | Комбинация тренда и волатильности |

### Архитектура Модели

```python
class ConceptBottleneckTrader:
    def __init__(self):
        self.concept_encoder = ConceptEncoder(input_dim, concept_dim)
        self.task_predictor = TaskPredictor(concept_dim, output_dim)

    def forward(self, x):
        concepts = self.concept_encoder(x)  # Интерпретируемое узкое место
        prediction = self.task_predictor(concepts)
        return prediction, concepts
```

## Реализация

### Структура Проекта

```
117_concept_bottleneck_trading/
├── README.md                    # Документация на английском
├── README.ru.md                 # Этот файл
├── readme.simple.md             # Упрощенный английский
├── readme.simple.ru.md          # Упрощенный русский
├── python/
│   ├── __init__.py
│   ├── concepts.py              # Модуль вычисления концепций
│   ├── model.py                 # Реализация CBM
│   ├── data_loader.py           # Загрузка данных с Bybit
│   ├── backtest.py              # Фреймворк бэктестинга
│   └── train.py                 # Скрипт обучения
└── rust_examples/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # Клиент API Bybit
    │   ├── data/                # Обработка данных
    │   ├── models/              # Реализация CBM
    │   └── utils/               # Технические индикаторы
    └── examples/
        ├── basic_cbm.rs
        └── trading_cbm.rs
```

## Источники Данных

### Интеграция с Bybit API

Реализация использует биржу Bybit для криптовалютных данных:

```python
# Пример на Python
from data_loader import BybitDataLoader

loader = BybitDataLoader()
df = loader.fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    limit=1000
)
```

```rust
// Пример на Rust
use concept_bottleneck_trading::api::BybitClient;

let client = BybitClient::new();
let klines = client.fetch_klines("BTCUSDT", "1h", 1000).await?;
```

### Поддерживаемые Данные

- **Криптовалюта**: Бессрочные фьючерсы Bybit (BTCUSDT, ETHUSDT и др.)
- **Фондовый рынок**: Интеграция с Yahoo Finance для акций
- **Таймфреймы**: 1m, 5m, 15m, 1h, 4h, 1D

## Процесс Обучения

### 1. Генерация Меток Концепций

Концепции могут быть сгенерированы через:
- **Правила**: Технические индикаторы с заданными порогами
- **Экспертная разметка**: Ручная разметка трейдерами
- **Кластеризация**: Автоматическое обнаружение рыночных режимов

### 2. Совместное Обучение

```python
# Loss = Потеря концепций + Потеря задачи
loss = concept_loss(pred_concepts, true_concepts) + \
       alpha * task_loss(pred_signal, true_signal)
```

### 3. Интервенция при Инференсе

```python
# Переопределение концепции на основе экспертного знания
concepts = model.predict_concepts(market_data)
if expert_override:
    concepts['volatility_regime'] = 'high'  # Ручная коррекция
signal = model.predict_signal(concepts)
```

## Метрики Оценки

### Качество Концепций
- Точность предсказания концепций
- Корреляция концепция-цель

### Торговая Эффективность
- Коэффициент Шарпа
- Максимальная просадка
- Процент выигрышных сделок
- Профит-фактор

## Научные Работы

1. Koh, P. W., et al. (2020). "Concept Bottleneck Models." *ICML 2020*.
2. Chen, Z., et al. (2020). "Concept Whitening for Interpretable Image Recognition."
3. Losch, M., et al. (2019). "Interpretability Beyond Feature Attribution."

## Быстрый Старт

### Python

```bash
cd 117_concept_bottleneck_trading/python
pip install -r requirements.txt
python train.py --symbol BTCUSDT --interval 1h
```

### Rust

```bash
cd 117_concept_bottleneck_trading/rust_examples
cargo run --example trading_cbm
```

## Лицензия

Этот проект является частью репозитория Machine Learning for Trading.
