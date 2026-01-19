# Глава 62: BloombergGPT для трейдинга — Применение финансовых LLM

В этой главе рассматривается **BloombergGPT** — большая языковая модель Bloomberg с 50 миллиардами параметров, специально разработанная для финансовой сферы. Мы изучим, как доменно-специфичные LLM могут использоваться для трейдинга, включая анализ тональности, распознавание сущностей и ответы на финансовые вопросы.

<p align="center">
<img src="https://i.imgur.com/YZB9kWp.png" width="70%">
</p>

## Содержание

1. [Введение в BloombergGPT](#введение-в-bloomberggpt)
    * [Зачем нужны доменно-специфичные LLM?](#зачем-нужны-доменно-специфичные-llm)
    * [Ключевые инновации](#ключевые-инновации)
    * [Сравнение с другими моделями](#сравнение-с-другими-моделями)
2. [Архитектура BloombergGPT](#архитектура-bloomberggpt)
    * [Спецификации модели](#спецификации-модели)
    * [Состав обучающих данных](#состав-обучающих-данных)
    * [Методология обучения](#методология-обучения)
3. [Применение в трейдинге](#применение-в-трейдинге)
    * [Анализ тональности для трейдинга](#анализ-тональности-для-трейдинга)
    * [Распознавание именованных сущностей](#распознавание-именованных-сущностей)
    * [Ответы на финансовые вопросы](#ответы-на-финансовые-вопросы)
    * [Классификация новостей](#классификация-новостей)
4. [Практические примеры](#практические-примеры)
    * [01: Анализ финансовой тональности](#01-анализ-финансовой-тональности)
    * [02: Генерация торговых сигналов](#02-генерация-торговых-сигналов)
    * [03: Прогнозирование влияния новостей](#03-прогнозирование-влияния-новостей)
    * [04: Бэктестинг LLM-сигналов](#04-бэктестинг-llm-сигналов)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в BloombergGPT

BloombergGPT представляет парадигмальный сдвиг в финансовой обработке естественного языка. В то время как универсальные LLM, такие как GPT-4 или BLOOM, могут обрабатывать финансовые задачи, BloombergGPT была специально обучена на финансовых данных, достигая превосходной производительности на доменно-специфичных задачах без ущерба для общего понимания языка.

### Зачем нужны доменно-специфичные LLM?

Универсальные LLM сталкиваются с проблемами при работе с финансовым языком:

```
ПРОБЛЕМЫ УНИВЕРСАЛЬНЫХ LLM:
┌──────────────────────────────────────────────────────────────────┐
│  1. ДОМЕННЫЙ ЖАРГОН                                              │
│     "Акция торгуется на уровне 15x форвардного P/E               │
│      с дивидендной доходностью 2%"                               │
│     Универсальная LLM: Может неверно интерпретировать термины    │
│     BloombergGPT: Нативно понимает метрики оценки                │
├──────────────────────────────────────────────────────────────────┤
│  2. РАЗРЕШЕНИЕ НЕОДНОЗНАЧНОСТИ СУЩНОСТЕЙ                         │
│     "Apple объявила квартальную прибыль"                         │
│     Универсальная LLM: Это фрукт или компания?                   │
│     BloombergGPT: Чётко определяет контекст AAPL                 │
├──────────────────────────────────────────────────────────────────┤
│  3. ВРЕМЕННЫЕ РАССУЖДЕНИЯ                                        │
│     "Результаты Q3 превзошли консенсус на 200 б.п."              │
│     Универсальная LLM: Может не связать Q3 с конкретным периодом │
│     BloombergGPT: Обучена на временных финансовых паттернах      │
├──────────────────────────────────────────────────────────────────┤
│  4. НЮАНСЫ ТОНАЛЬНОСТИ                                           │
│     "Компания сохранила прогноз несмотря на сложности"           │
│     Универсальная LLM: Нейтрально или слегка негативно?          │
│     BloombergGPT: Распознаёт как умеренно позитивно              │
└──────────────────────────────────────────────────────────────────┘
```

### Ключевые инновации

1. **Массивный финансовый датасет (FinPile)**
   - 363 миллиарда токенов проприетарных финансовых данных Bloomberg
   - 40 лет финансовых документов, новостей, отчётов и транскриптов
   - Крупнейший доменно-специфичный датасет для обучения финансовых LLM

2. **Смешанная стратегия обучения**
   - Комбинация финансовых данных (51.27%) с общими данными (48.73%)
   - Сохраняет общие языковые способности
   - Достигает лучших результатов в обоих направлениях

3. **Аспектно-специфичная тональность**
   - Выходит за рамки бинарной позитивной/негативной оценки
   - Определяет тональность по отношению к конкретным сущностям в тексте
   - Критически важно для торговых сигналов из многотемных новостей

4. **Разрешение неоднозначности финансовых сущностей**
   - Связывает упоминания с идентификаторами сущностей Bloomberg
   - Различает компании с похожими названиями
   - Критично для точной генерации торговых сигналов

### Сравнение с другими моделями

| Характеристика | GPT-4 | BLOOM-176B | FinBERT | BloombergGPT |
|----------------|-------|------------|---------|--------------|
| Параметры | ~1.7T | 176B | 110M | 50.6B |
| Финансовое предобучение | Ограничено | Нет | Да | Обширное |
| Общий NLP | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| Финансовая тональность | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Разрешение сущностей | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ |
| Публичная доступность | Только API | Да | Да | Нет |

## Архитектура BloombergGPT

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         АРХИТЕКТУРА BloombergGPT                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Входной текст ──────────────────────────────────────────────────────────┐   │
│  "Apple отчиталась о прибыли Q3 выше ожиданий, акции выросли на 5%"      │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                         │                                     │
│                                         ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                    Токен-эмбеддинг + ALiBi позиции                     │   │
│  │    Словарь: 131,072 токена | Скрытая размерность: 7,680                │   │
│  └─────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                     Нормализация слоя                                  │   │
│  └─────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │              БЛОК ДЕКОДЕРА ТРАНСФОРМЕРА (×70 слоёв)                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Многоголовое самовнимание (40 голов)                           │  │   │
│  │  │  • Каузальная маскировка для авторегрессивной генерации         │  │   │
│  │  │  • ALiBi позиционное кодирование (без позиционных эмбеддингов)  │  │   │
│  │  │  • Размерность головы: 7,680 / 40 = 192                         │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Полносвязная сеть                                              │  │   │
│  │  │  • Скрытые: 7,680 → 30,720 → 7,680                              │  │   │
│  │  │  • Активация GELU                                               │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                         │   │
│  │  + Остаточные соединения + Нормализация слоя                          │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                Финальная нормализация → Линейный → Softmax             │   │
│  │                      (связан с входными эмбеддингами)                  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  Выход: Вероятности следующего токена / Специфичные для задачи предсказания  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Спецификации модели

| Спецификация | Значение |
|--------------|----------|
| Всего параметров | 50.6 миллиарда |
| Слоёв | 70 блоков декодера трансформера |
| Голов внимания | 40 |
| Скрытая размерность | 7,680 |
| Скрытая размерность FFN | 30,720 (4× скрытая) |
| Размер словаря | 131,072 токена |
| Длина контекста | 2,048 токенов |
| Позиционное кодирование | ALiBi (Внимание с линейными смещениями) |
| Функция активации | GELU |

### Состав обучающих данных

```
СОСТАВ ОБУЧАЮЩИХ ДАННЫХ (708B токенов, использовано ~569B)
═══════════════════════════════════════════════════════════════════════════════

FINPILE (363B токенов, 51.27%)                  ПУБЛИЧНЫЕ ДАННЫЕ (345B токенов, 48.73%)
┌─────────────────────────────────────────┐     ┌──────────────────────────────┐
│  Bloomberg Web     (298.0B)  81.98%     │     │  The Pile    (184.6B) 53.50% │
│  Bloomberg News     (38.1B)  10.48%     │     │  C4          (138.5B) 40.14% │
│  Bloomberg Filings  (14.1B)   3.88%     │     │  Wikipedia    (21.9B)  6.36% │
│  Bloomberg Press    (13.3B)   3.66%     │     └──────────────────────────────┘
└─────────────────────────────────────────┘

ИСТОЧНИКИ FINPILE:
• Web: Курированные финансовые веб-сайты и порталы
• News: Новостные статьи Bloomberg и информационные ленты
• Filings: Отчёты SEC, регуляторные документы, финансовые отчёты
• Press: Пресс-релизы и объявления компаний
```

### Методология обучения

```python
# Конфигурация обучения, использованная для BloombergGPT
training_config = {
    # Оптимизация
    "optimizer": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "weight_decay": 0.1,

    # Расписание learning rate
    "max_learning_rate": 6e-5,
    "min_learning_rate": 6e-6,
    "lr_schedule": "cosine_decay",
    "warmup_steps": 1600,

    # Размер батча
    "batch_size": 1024,  # Начальный
    "batch_size_final": 2048,  # После разогрева
    "sequence_length": 2048,

    # Продолжительность обучения
    "total_steps": 139200,
    "training_days": 53,

    # Оборудование
    "gpus": 512,  # A100 40GB
    "gpu_instances": 64,
    "throughput_tflops": 102,

    # Техники эффективности
    "zero_stage": 3,
    "activation_checkpointing": True,
    "mixed_precision": "bf16",  # Прямой/обратный проход
    "param_precision": "fp32",  # Состояния оптимизатора
}
```

## Применение в трейдинге

### Анализ тональности для трейдинга

BloombergGPT превосходно справляется с аспектно-специфичным анализом тональности, что критически важно для генерации действенных торговых сигналов:

```python
# Пример: аспектно-специфичная тональность для трейдинга
text = """
Microsoft отчиталась о сильном росте облачных сервисов, выручка Azure
выросла на 29% г/г. Однако компания снизила прогноз для сегмента ПК
из-за слабого потребительского спроса. CEO Наделла подчеркнул
инвестиции в ИИ как ключевой приоритет.
"""

# BloombergGPT может определить:
aspects = {
    "Microsoft_Cloud": "ПОЗИТИВНО",     # Сильный рост, выше ожиданий
    "Microsoft_PC": "НЕГАТИВНО",        # Снижен прогноз
    "Microsoft_AI": "НЕЙТРАЛЬНО/ПОЗИТИВНО", # Стратегический приоритет
    "Microsoft_Overall": "ПОЗИТИВНО"     # Чистый позитивный нарратив
}

# Генерация торгового сигнала
def generate_signal(aspects, weights):
    """
    Генерация торгового сигнала из аспектных тональностей.

    Args:
        aspects: Словарь аспект -> тональность
        weights: Словарь аспект -> вес важности

    Returns:
        Сила сигнала (-1 до 1)
    """
    sentiment_scores = {"ПОЗИТИВНО": 1, "НЕЙТРАЛЬНО": 0, "НЕГАТИВНО": -1}

    total_weight = sum(weights.values())
    signal = sum(
        weights[aspect] * sentiment_scores[sentiment]
        for aspect, sentiment in aspects.items()
    ) / total_weight

    return signal  # напр., 0.4 -> Умеренный сигнал на покупку
```

**Результаты бенчмарков (F1 Score):**

| Задача | BloombergGPT | BLOOM-176B | GPT-NeoX |
|--------|-------------|------------|----------|
| Тональность новостей об акциях | **79.63%** | 19.96% | 14.72% |
| Тональность соцсетей | **63.96%** | 21.63% | 17.22% |
| Тональность транскриптов | **52.70%** | 14.51% | 13.46% |
| ES тональность новостей | **58.36%** | 42.86% | 16.39% |
| Среднее | **62.47%** | 24.24% | 15.45% |

### Распознавание именованных сущностей

Точное определение финансовых сущностей критически важно для трейдинга:

```python
# Пример: NER для трейдинга
text = "Акции Apple взлетели после того, как Тим Кук объявил о рекордных продажах iPhone."

# Вывод BloombergGPT NER:
entities = [
    {"text": "Apple", "type": "ORG", "bloomberg_id": "AAPL US Equity"},
    {"text": "Тим Кук", "type": "PER", "role": "CEO"},
    {"text": "iPhone", "type": "PRODUCT", "company": "AAPL"}
]

# Пример разрешения неоднозначности
ambiguous_text = "Apple отчиталась о прибыли, пока начинается сезон сбора яблок."

# BloombergGPT корректно определяет:
# "Apple" (первое) -> AAPL US Equity (компания)
# "яблок" (второе) -> Не финансовая сущность (фрукт)
```

**Результаты NER + разрешение неоднозначности:**

| Датасет | BloombergGPT | BLOOM-176B | GPT-NeoX |
|---------|-------------|------------|----------|
| Новостные ленты | **68.15%** | 45.87% | 39.45% |
| Отчётность | **62.34%** | 48.21% | 42.31% |
| Заголовки | **65.92%** | 44.76% | 38.94% |
| Транскрипты | **61.48%** | 43.89% | 37.82% |
| Среднее | **64.83%** | 45.43% | 39.26% |

### Ответы на финансовые вопросы

BloombergGPT может отвечать на сложные финансовые вопросы:

```python
# Пример: задача ConvFinQA (диалоговые финансовые вопросы)

context = """
КОНСОЛИДИРОВАННЫЕ ОТЧЁТЫ О РЕЗУЛЬТАТАХ ДЕЯТЕЛЬНОСТИ
(В миллионах, кроме данных на акцию)

                          2023        2022        2021
Выручка                 $98,456     $87,321     $76,845
Себестоимость           $42,187     $38,542     $34,521
Валовая прибыль         $56,269     $48,779     $42,324
Операционные расходы    $28,456     $25,321     $22,187
Операционная прибыль    $27,813     $23,458     $20,137
"""

question = "Каков был рост операционной прибыли год к году с 2022 по 2023?"

# BloombergGPT может:
# 1. Извлечь релевантные числа ($27,813 и $23,458)
# 2. Рассчитать: (27,813 - 23,458) / 23,458 = 18.55%
# 3. Дать ответ: "Операционная прибыль выросла на 18.55% г/г"
```

### Классификация новостей

Классификация новостей по релевантности для трейдинга:

```python
# Пример: классификация новостей для трейдинга
news_items = [
    {
        "headline": "ФРС сигнализирует о возможном снижении ставок в Q2 2024",
        "classification": {
            "topic": "МОНЕТАРНАЯ_ПОЛИТИКА",
            "market_impact": "ВЫСОКИЙ",
            "direction": "RISK_ON",
            "assets_affected": ["SPY", "QQQ", "TLT", "GLD"]
        }
    },
    {
        "headline": "Tesla отзывает 2М автомобилей из-за проблем с автопилотом",
        "classification": {
            "topic": "РЕГУЛЯТОРНОЕ",
            "market_impact": "СРЕДНИЙ",
            "direction": "НЕГАТИВНО",
            "assets_affected": ["TSLA"]
        }
    }
]
```

## Практические примеры

### 01: Анализ финансовой тональности

```python
# python/01_sentiment_analysis.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple

class FinancialSentimentAnalyzer:
    """
    Анализатор финансовой тональности на основе LLM.

    Поскольку BloombergGPT не является публично доступной,
    мы демонстрируем подход с использованием open-source альтернатив
    (FinBERT, FinGPT) с тем же интерфейсом.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.label_map = {0: "ПОЗИТИВНО", 1: "НЕГАТИВНО", 2: "НЕЙТРАЛЬНО"}

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Анализ тональности финансового текста.

        Args:
            text: Финансовые новости или документ

        Returns:
            Словарь с вероятностями тональности
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            "positive": probs[0].item(),
            "negative": probs[1].item(),
            "neutral": probs[2].item(),
            "label": self.label_map[probs.argmax().item()],
            "confidence": probs.max().item()
        }

    def analyze_aspects(
        self,
        text: str,
        entities: List[str]
    ) -> Dict[str, Dict]:
        """
        Анализ тональности по отношению к конкретным сущностям (аспектный).

        Имитирует возможности BloombergGPT по аспектно-специфичной тональности.
        """
        results = {}

        for entity in entities:
            # Извлечение предложений, упоминающих сущность
            sentences = [s for s in text.split('.') if entity.lower() in s.lower()]

            if sentences:
                entity_text = '. '.join(sentences)
                results[entity] = self.analyze(entity_text)
            else:
                results[entity] = {"label": "НЕ_УПОМЯНУТО", "confidence": 0.0}

        return results
```

### 02: Генерация торговых сигналов

```python
# python/02_trading_signals.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class TradingSignal:
    """Торговый сигнал, сгенерированный из анализа LLM."""
    timestamp: datetime
    symbol: str
    signal: float  # -1 (сильная продажа) до 1 (сильная покупка)
    confidence: float
    reasoning: str
    source_type: str  # "news", "earnings", "filing"

class LLMSignalGenerator:
    """
    Генерация торговых сигналов из анализа тональности LLM.

    Этот класс демонстрирует, как анализ в стиле BloombergGPT
    может быть преобразован в действенные торговые сигналы.
    """

    def __init__(
        self,
        sentiment_analyzer,
        signal_threshold: float = 0.6,
        confidence_threshold: float = 0.7
    ):
        self.analyzer = sentiment_analyzer
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

        # Маппинг тональности в сигнал
        self.sentiment_weights = {
            "ПОЗИТИВНО": 1.0,
            "НЕГАТИВНО": -1.0,
            "НЕЙТРАЛЬНО": 0.0
        }

        # Веса важности источников
        self.source_weights = {
            "earnings": 1.0,  # Максимальное влияние
            "filing": 0.8,
            "news": 0.6,
            "social": 0.3  # Шумные, низкий вес
        }

    def generate_signal(
        self,
        text: str,
        symbol: str,
        source_type: str = "news",
        timestamp: Optional[datetime] = None
    ) -> Optional[TradingSignal]:
        """
        Генерация торгового сигнала из финансового текста.
        """
        timestamp = timestamp or datetime.now()

        # Анализ тональности
        sentiment = self.analyzer.analyze(text)

        # Проверка порога уверенности
        if sentiment['confidence'] < self.confidence_threshold:
            return None

        # Расчёт силы сигнала
        base_signal = self.sentiment_weights.get(sentiment['label'], 0)
        source_weight = self.source_weights.get(source_type, 0.5)
        signal_strength = base_signal * sentiment['confidence'] * source_weight

        # Применение порога
        if abs(signal_strength) < self.signal_threshold * source_weight:
            return None

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal_strength,
            confidence=sentiment['confidence'],
            reasoning=f"Тональность: {sentiment['label']}, Источник: {source_type}",
            source_type=source_type
        )
```

### 03: Прогнозирование влияния новостей

```python
# python/03_news_impact.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class NewsImpactPredictor(nn.Module):
    """
    Прогнозирование рыночного влияния финансовых новостей.

    Модель комбинирует эмбеддинги LLM с рыночными данными
    для прогнозирования величины и направления движения цен
    после новостных событий.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        market_features: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Ветка кодирования текста
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Ветка рыночных признаков (объём, волатильность и т.д.)
        self.market_encoder = nn.Sequential(
            nn.Linear(market_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Комбинированная голова прогнозирования
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # [направление, величина, уверенность]
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        market_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Прогнозирование влияния новости.
        """
        text_encoded = self.text_encoder(text_embedding)
        market_encoded = self.market_encoder(market_features)

        combined = torch.cat([text_encoded, market_encoded], dim=-1)
        output = self.predictor(combined)

        return {
            "direction": torch.tanh(output[:, 0]),     # -1 до 1
            "magnitude": torch.exp(output[:, 1]),     # Положительная, лог-шкала
            "confidence": torch.sigmoid(output[:, 2]) # 0 до 1
        }
```

### 04: Бэктестинг LLM-сигналов

```python
# python/04_backtest.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    """Конфигурация для бэктестинга LLM сигналов."""
    initial_capital: float = 100000
    max_position_size: float = 0.1  # Макс 10% на позицию
    transaction_cost_bps: float = 10  # 10 базисных пунктов
    slippage_bps: float = 5
    signal_decay_hours: float = 24
    rebalance_frequency: str = "daily"

@dataclass
class BacktestResult:
    """Результаты бэктестинга LLM сигналов."""
    returns: pd.Series
    positions: pd.DataFrame
    trades: List[Dict]
    metrics: Dict[str, float]

class LLMSignalBacktester:
    """
    Бэктестинг торговых сигналов, сгенерированных из анализа LLM.

    Этот бэктестер специально обрабатывает уникальные характеристики
    сигналов, полученных от LLM:
    - Нерегулярное время сигналов (управляемое новостями)
    - Затухание сигнала со временем
    - Различные уровни уверенности
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Запуск бэктеста на LLM сигналах.

        Args:
            signals: DataFrame с колонками [timestamp, symbol, signal, confidence]
            prices: DataFrame с OHLCV данными, индексированный по timestamp
            start_date: Дата начала бэктеста
            end_date: Дата окончания бэктеста

        Returns:
            BacktestResult с метриками производительности
        """
        # ... (полная реализация как в английской версии)
        pass

    def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Расчёт метрик производительности бэктеста."""
        if returns.empty:
            return {}

        # Фактор аннуализации (предполагая дневные доходности)
        ann_factor = 252

        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)

        sharpe = ann_return / volatility if volatility > 0 else 0

        # Максимальная просадка
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": len(trades)
        }
```

## Реализация на Rust

Поскольку BloombergGPT не является публично доступной, мы реализуем обёртку финансового LLM в стиле BloombergGPT на Rust, которая может работать с open-source альтернативами.

```
rust_bloomberggpt/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиенты внешних API
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Криптоданные Bybit
│   │   └── yahoo.rs        # Данные Yahoo Finance
│   ├── llm/                # Интерфейс LLM
│   │   ├── mod.rs
│   │   ├── client.rs       # Клиент API LLM
│   │   ├── prompts.rs      # Финансовые промпты
│   │   └── embeddings.rs   # Текстовые эмбеддинги
│   ├── analysis/           # Финансовый анализ
│   │   ├── mod.rs
│   │   ├── sentiment.rs    # Анализ тональности
│   │   ├── ner.rs          # Распознавание сущностей
│   │   └── qa.rs           # Ответы на вопросы
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── sentiment_analysis.rs
    ├── generate_signals.rs
    └── backtest.rs
```

Смотрите [rust_bloomberggpt](rust_bloomberggpt/) для полной реализации на Rust.

### Быстрый старт (Rust)

```bash
cd rust_bloomberggpt

# Запуск примера анализа тональности
cargo run --example sentiment_analysis

# Генерация торговых сигналов из новостей
cargo run --example generate_signals -- --symbol BTCUSDT

# Запуск бэктеста
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── sentiment_analysis.py   # Финансовая тональность
├── trading_signals.py      # Генерация сигналов
├── news_impact.py          # Прогноз влияния
├── backtest.py             # Бэктестинг
├── data_loader.py          # Утилиты загрузки данных
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_sentiment_demo.py
    ├── 02_signal_generation.py
    ├── 03_impact_prediction.py
    └── 04_full_backtest.py
```

### Быстрый старт (Python)

```bash
cd python

# Установка зависимостей
pip install -r requirements.txt

# Запуск анализа тональности
python examples/01_sentiment_demo.py

# Генерация сигналов
python examples/02_signal_generation.py --symbol AAPL

# Запуск бэктеста
python examples/04_full_backtest.py --capital 100000
```

## Лучшие практики

### Когда использовать финансовые LLM для трейдинга

**Хорошие варианты использования:**
- Анализ тональности звонков по прибыли и новостей
- Торговые сигналы, управляемые событиями
- Извлечение сущностей из финансовых документов
- Суммаризация отчётов SEC
- Классификация новостей по влиянию

**Не идеально для:**
- Высокочастотный трейдинг (слишком высокая задержка)
- Чисто ценовое прогнозирование (используйте количественные модели)
- Полная замена фундаментального анализа

### Рекомендации по генерации сигналов

1. **Фильтрация по уверенности**
   ```python
   # Действуйте только на сигналы с высокой уверенностью
   if signal.confidence < 0.7:
       continue  # Пропуск сигналов с низкой уверенностью
   ```

2. **Затухание сигнала**
   ```python
   # Влияние новостей затухает со временем
   signal_strength *= np.exp(-hours_since_news / 24)
   ```

3. **Взвешивание источников**
   ```python
   source_weights = {
       "earnings": 1.0,  # Максимальное влияние
       "sec_filing": 0.8,
       "news": 0.6,
       "social": 0.3  # Шумные, низкий вес
   }
   ```

4. **Размер позиции**
   ```python
   # Масштабирование позиции по уверенности
   position_size = base_size * confidence * signal_strength
   ```

### Распространённые ошибки

1. **Переобучение на тональности** - Не торгуйте исключительно на тональности; комбинируйте с ценой/объёмом
2. **Проблемы с задержкой** - Инференс LLM медленный; не подходит для HFT
3. **Риск галлюцинаций** - Всегда проверяйте извлечение сущностей через базу данных
4. **Управление затратами** - Вызовы API LLM дорогие; группируйте когда возможно

## Ресурсы

### Статьи

- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) — Оригинальная статья BloombergGPT (2023)
- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063) — Статья FinBERT
- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) — Open-source финансовая LLM
- [Large Language Models in Finance: A Survey](https://arxiv.org/abs/2311.10723) — Обзорная статья

### Open-Source альтернативы

Поскольку BloombergGPT проприетарна, рассмотрите альтернативы:

| Модель | Размер | Доступность | Лучше всего для |
|--------|--------|-------------|-----------------|
| [FinBERT](https://huggingface.co/ProsusAI/finbert) | 110M | Открытая | Анализ тональности |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Разные | Открытая | Общий финансовый NLP |
| [FinMA](https://huggingface.co/ChanceFocus/finma-7b-nlp) | 7B | Открытая | Финансовые задачи |
| GPT-4 | ~1.7T | API | Общий + финансовый |

### Связанные главы

- [Глава 61: FinGPT Financial LLM](../61_fingpt_financial_llm) — Open-source альтернатива
- [Глава 67: LLM Sentiment Analysis](../67_llm_sentiment_analysis) — Глубокое погружение в тональность
- [Глава 241: FinBERT Sentiment](../241_finbert_sentiment) — Меньшая, более быстрая модель
- [Глава 37: Sentiment Momentum Fusion](../37_sentiment_momentum_fusion) — Комбинирование сигналов

---

## Уровень сложности

**Продвинутый**

Предварительные требования:
- Понимание архитектуры трансформеров и LLM
- Знания финансовых рынков (тональность, торговые сигналы)
- Опыт программирования на Python/Rust
- Опыт работы с NLP задачами (анализ тональности, NER)

## Ссылки

1. Wu, S., et al. (2023). "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564
2. Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models."
3. Yang, H., et al. (2023). "FinGPT: Open-Source Financial Large Language Models."
4. Liu, X., et al. (2023). "Large Language Models in Finance: A Survey."
