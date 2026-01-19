# Глава 69: LLM Анализ Earnings Calls — Извлечение Торговых Сигналов из Корпоративных Коммуникаций

## Обзор

Earnings calls (звонки о прибыли) — это ежеквартальные конференц-звонки, на которых публичные компании обсуждают финансовые результаты с аналитиками и инвесторами. Эти звонки содержат богатую информацию о результатах компании, настроении руководства, прогнозах на будущее и рыночных ожиданиях. Большие языковые модели (LLM) могут анализировать эти транскрипты для извлечения торговых сигналов, которые ранее было сложно количественно оценить.

В этой главе мы рассмотрим применение LLM для анализа earnings calls в торговле криптовалютами и акциями, используя продвинутые методы NLP для извлечения настроения, определения уровня уверенности руководства, выявления ключевых тем и генерации вероятностных торговых сигналов.

## Основные Концепции

### Что такое Анализ Earnings Calls?

Earnings calls состоят из двух основных частей:
1. **Подготовленные Замечания**: Скриптованная презентация руководства о финансовых результатах
2. **Сессия Вопросов и Ответов**: Аналитики задают вопросы, руководство отвечает спонтанно

```
Структура Earnings Call:
├── Вступление (CFO/CEO представление)
├── Финансовые Результаты (по сценарию, обычно позитивная подача)
├── Прогнозы (ожидания на будущее)
├── Сессия Вопросов и Ответов (без сценария, более показательна)
│   ├── Вопросы аналитиков (выявляют проблемы)
│   └── Ответы руководства (индикаторы уверенности)
└── Заключительные замечания

Фокус Анализа LLM:
├── Полярность настроения (позитивное/негативное/нейтральное)
├── Маркеры уверенности (уклончивый язык vs уверенный)
├── Изменения прогнозов (превысили/соответствуют/не достигли ожиданий)
├── Раскрытие рисков (упомянуты новые риски)
└── Сдвиги тона (сравнение с предыдущими звонками)
```

### Почему LLM для Анализа Earnings Calls?

1. **Контекстное Понимание**: LLM понимают финансовый жаргон и контекст
2. **Обнаружение Нюансов**: Улавливают тонкие сдвиги настроения, которые могут пропустить аналитики
3. **Масштаб**: Обрабатывают сотни earnings calls эффективно
4. **Согласованность**: Применяют единообразный анализ ко всем транскриптам
5. **Многофакторное Извлечение**: Извлекают несколько сигналов одновременно
6. **Анализ Q&A**: Оценивают отзывчивость и прозрачность руководства

### Ключевые Лингвистические Признаки

```
Индикаторы Уверенности:
├── Сильные: "Мы уверены", "явно", "определённо"
├── Умеренные: "Мы считаем", "мы ожидаем", "вероятно"
├── Слабые: "Мы надеемся", "потенциально", "возможно"
└── Уклончивые: "При условии", "неопределённо", "сложно"

Маркеры Настроения:
├── Позитивные: "Сильные результаты", "превысили ожидания"
├── Негативные: "Встречные ветры", "сложная среда"
├── Нейтральные: "В соответствии с", "как ожидалось"
└── Смешанные: "Несмотря на трудности, нам удалось..."

Прогнозные Сигналы:
├── Бычьи: "Повышаем прогноз", "ускоряющийся рост"
├── Медвежьи: "Понижаем ожидания", "осторожный взгляд"
└── Нейтральные: "Сохраняем прогноз", "стабильные перспективы"
```

## Торговая Стратегия

**Обзор Стратегии:** Используем LLM для анализа транскриптов earnings calls и генерации торговых сигналов на основе:
1. Общего показателя настроения
2. Уровня уверенности руководства
3. Направления прогноза (повышение/сохранение/понижение)
4. Оценки прозрачности Q&A
5. Сравнения с предыдущими кварталами

### Генерация Сигналов

```
1. Обработка Транскрипта:
   - Разбор транскрипта на секции (подготовленные замечания, Q&A)
   - Очистка и нормализация текста
   - Идентификация спикеров и их ролей

2. Анализ LLM:
   - Извлечение оценок настроения по секциям
   - Идентификация маркеров уверенности
   - Обнаружение изменений прогнозов
   - Анализ качества Q&A

3. Агрегация Сигналов:
   - Взвешивание сигналов по важности
   - Сравнение с консенсусом аналитиков
   - Генерация итогового торгового сигнала

4. Оценка Рисков:
   - Оценка неопределённости в анализе
   - Проверка на конфликтующие сигналы
   - Корректировка размера позиции
```

### Сигналы Входа

- **Сигнал на Покупку**: Высокое позитивное настроение + Сильная уверенность + Повышенный прогноз + Хорошее Q&A
- **Сигнал на Продажу**: Негативное настроение + Уклончивый язык + Пониженный прогноз + Уклончивое Q&A
- **Сигнал Удержания**: Смешанные сигналы или нейтральное настроение

### Управление Рисками

- **Порог Уверенности**: Торговать только когда уверенность LLM > порога
- **Сила Сигнала**: Масштабировать размер позиции с силой сигнала
- **Волатильность Earnings**: Учитывать движения цены после отчётности
- **Стоп-лосс**: Использовать историческую реакцию на earnings для размещения стопов

## Техническая Спецификация

### Математические Основы

#### Расчёт Настроения

```
Расчёт Показателя Настроения:
├── Настроение по секциям: S_i ∈ [-1, 1]
├── Веса секций: w_i (подготовленные замечания, Q&A, прогнозы)
├── Агрегированное настроение: S = Σ(w_i × S_i) / Σ(w_i)
│
├── Корректировка уверенности:
│   S_adj = S × коэффициент_уверенности
│   где коэффициент_уверенности ∈ [0.5, 1.5]
│
└── Историческая нормализация:
    S_normalized = (S_adj - μ_historical) / σ_historical
```

#### Сила Сигнала

```
Компоненты Силы Сигнала:
├── Величина настроения: |S_normalized|
├── Изменение прогноза: G ∈ {-1, 0, 1}
├── Уровень уверенности: C ∈ [0, 1]
├── Качество Q&A: Q ∈ [0, 1]
│
Сигнал = w_s × S_normalized + w_g × G + w_c × C + w_q × Q

Размер позиции:
Позиция = базовый_размер × tanh(Сигнал × масштаб)
```

### Диаграмма Архитектуры

```
                    Транскрипт Earnings Call
                           │
                           ▼
            ┌─────────────────────────────┐
            │      Парсер Транскрипта     │
            │  ├── Определение секций     │
            │  ├── Идентификация спикеров │
            │  ├── Парное Q&A             │
            │  └── Нормализация текста    │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │   Движок Анализа LLM        │
            │                             │
            │  ┌───────────────────────┐  │
            │  │ Извлечение Настроения │  │
            │  │ - Настроение секций   │  │
            │  │ - Настроение сущностей│  │
            │  │ - Временные изменения │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │  Анализ Уверенности   │  │
            │  │ - Обнаружение хеджинга│  │
            │  │ - Маркеры уверенности │  │
            │  │ - Анализ тона         │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │ Извлечение Прогнозов  │  │
            │  │ - Прогноз выручки     │  │
            │  │ - Прогноз EPS         │  │
            │  │ - Изменение направления│  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │  Оценка Качества Q&A  │  │
            │  │ - Отзывчивость        │  │
            │  │ - Прозрачность        │  │
            │  │ - Обнаружение уклонения│  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Показатель │ │   Уровень   │ │ Направление │
     │  Настроения │ │ Уверенности │ │   Прогноза  │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Агрегация Сигналов      │
            │  ├── Комбинация весов       │
            │  ├── Исторические сравнение │
            │  ├── Корректировка консенсуса│
            │  └── Масштабирование довер. │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │     Торговое Решение        │
            │  ├── Направление сигнала    │
            │  ├── Размер позиции         │
            │  ├── Время входа            │
            │  └── Размещение стоп-лосса  │
            └─────────────────────────────┘
```

### Промпт-Инжиниринг для Анализа Earnings

```python
import json
from openai import OpenAI

EARNINGS_ANALYSIS_PROMPT = """
Вы эксперт-финансовый аналитик, специализирующийся на анализе earnings calls.
Проанализируйте следующий транскрипт earnings call и предоставьте структурированную оценку.

Транскрипт:
{transcript}

Предоставьте ваш анализ в следующем JSON формате:
{{
    "overall_sentiment": {{
        "score": <float от -1 до 1>,
        "explanation": "<краткое объяснение>"
    }},
    "management_confidence": {{
        "score": <float от 0 до 1>,
        "hedging_examples": ["<примеры фраз>"],
        "confidence_examples": ["<примеры фраз>"]
    }},
    "guidance_assessment": {{
        "direction": "<raised|maintained|lowered|not_provided>",
        "revenue_guidance": "<конкретный прогноз если упомянут>",
        "eps_guidance": "<конкретный прогноз если упомянут>",
        "key_drivers": ["<основные факторы>"]
    }},
    "qa_quality": {{
        "score": <float от 0 до 1>,
        "transparency_level": "<high|medium|low>",
        "evasive_responses": ["<вопросы которых избегали>"]
    }},
    "key_themes": ["<основные обсуждаемые темы>"],
    "risk_factors": ["<новые или подчёркнутые риски>"],
    "trading_signal": {{
        "direction": "<bullish|neutral|bearish>",
        "strength": <float от 0 до 1>,
        "reasoning": "<краткое объяснение>"
    }}
}}
"""

def analyze_earnings_call(transcript: str, client: OpenAI) -> dict:
    """
    Анализ транскрипта earnings call с использованием LLM

    Args:
        transcript: Текст транскрипта earnings call
        client: Экземпляр клиента OpenAI

    Returns:
        Структурированный словарь анализа
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Вы эксперт-финансовый аналитик. Предоставляйте точный, практичный анализ."
            },
            {
                "role": "user",
                "content": EARNINGS_ANALYSIS_PROMPT.format(transcript=transcript)
            }
        ],
        temperature=0.1,  # Низкая температура для согласованного анализа
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### Парсинг Транскрипта

```python
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class SpeakerRole(Enum):
    CEO = "ceo"
    CFO = "cfo"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OTHER = "other"

@dataclass
class TranscriptSegment:
    """Сегмент транскрипта earnings call"""
    speaker: str
    role: SpeakerRole
    text: str
    section: str  # 'prepared_remarks' или 'qa'
    timestamp: Optional[str] = None

class EarningsTranscriptParser:
    """
    Парсер транскриптов earnings calls

    Обрабатывает различные форматы транскриптов и извлекает структурированные данные
    """

    # Общие паттерны для идентификации спикеров
    SPEAKER_PATTERNS = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*(.+)$',  # "John Smith - CEO"
        r'^([A-Z][a-z]+ [A-Z][a-z]+):',  # "John Smith:"
        r'^\[([A-Z][a-z]+ [A-Z][a-z]+)\]',  # "[John Smith]"
    ]

    CEO_KEYWORDS = ['ceo', 'chief executive', 'president', 'генеральный директор']
    CFO_KEYWORDS = ['cfo', 'chief financial', 'finance', 'финансовый директор']
    ANALYST_KEYWORDS = ['analyst', 'research', 'capital', 'securities', 'bank', 'аналитик']

    def __init__(self):
        self.segments: List[TranscriptSegment] = []

    def parse(self, transcript: str) -> List[TranscriptSegment]:
        """
        Парсинг транскрипта в сегменты

        Args:
            transcript: Сырой текст транскрипта

        Returns:
            Список объектов TranscriptSegment
        """
        self.segments = []

        # Разделение на строки
        lines = transcript.split('\n')

        current_speaker = None
        current_role = SpeakerRole.OTHER
        current_text = []
        current_section = self._detect_initial_section(transcript)

        for line in lines:
            # Проверка маркеров секций
            if self._is_qa_start(line):
                # Сохранение текущего сегмента
                if current_speaker and current_text:
                    self._add_segment(current_speaker, current_role,
                                    ' '.join(current_text), current_section)
                current_section = 'qa'
                current_text = []
                continue

            # Проверка нового спикера
            speaker_match = self._extract_speaker(line)
            if speaker_match:
                # Сохранение предыдущего сегмента
                if current_speaker and current_text:
                    self._add_segment(current_speaker, current_role,
                                    ' '.join(current_text), current_section)

                current_speaker, role_hint = speaker_match
                current_role = self._identify_role(current_speaker, role_hint)
                current_text = [self._clean_speaker_line(line)]
            else:
                # Продолжение текущего сегмента
                if line.strip():
                    current_text.append(line.strip())

        # Добавление финального сегмента
        if current_speaker and current_text:
            self._add_segment(current_speaker, current_role,
                            ' '.join(current_text), current_section)

        return self.segments

    def get_prepared_remarks(self) -> List[TranscriptSegment]:
        """Получить только сегменты подготовленных замечаний"""
        return [s for s in self.segments if s.section == 'prepared_remarks']

    def get_qa_segments(self) -> List[TranscriptSegment]:
        """Получить только сегменты Q&A"""
        return [s for s in self.segments if s.section == 'qa']

    def get_management_segments(self) -> List[TranscriptSegment]:
        """Получить сегменты от CEO/CFO"""
        return [s for s in self.segments
                if s.role in [SpeakerRole.CEO, SpeakerRole.CFO]]
```

### Генератор Торговых Сигналов

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

class SignalDirection(Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"

@dataclass
class TradingSignal:
    """Торговый сигнал из анализа earnings"""
    direction: SignalDirection
    strength: float  # от 0 до 1
    confidence: float  # от 0 до 1
    sentiment_score: float
    confidence_level: float
    guidance_direction: str
    qa_quality: float
    reasoning: str

class EarningsSignalGenerator:
    """
    Генерация торговых сигналов из анализа earnings calls
    """

    def __init__(self,
                 sentiment_weight: float = 0.35,
                 confidence_weight: float = 0.20,
                 guidance_weight: float = 0.30,
                 qa_weight: float = 0.15,
                 signal_threshold: float = 0.3):
        """
        Инициализация генератора сигналов

        Args:
            sentiment_weight: Вес для показателя настроения
            confidence_weight: Вес для уверенности руководства
            guidance_weight: Вес для направления прогноза
            qa_weight: Вес для качества Q&A
            signal_threshold: Минимальный показатель для ненейтрального сигнала
        """
        self.weights = {
            'sentiment': sentiment_weight,
            'confidence': confidence_weight,
            'guidance': guidance_weight,
            'qa': qa_weight
        }
        self.signal_threshold = signal_threshold

    def generate_signal(self,
                       analysis: Dict,
                       historical_sentiment: Optional[float] = None) -> TradingSignal:
        """
        Генерация торгового сигнала из анализа LLM

        Args:
            analysis: Словарь из анализа LLM
            historical_sentiment: Среднее настроение из предыдущих звонков

        Returns:
            TradingSignal с направлением и силой
        """
        # Извлечение компонентов
        sentiment = analysis['overall_sentiment']['score']
        confidence = analysis['management_confidence']['score']
        guidance = self._guidance_to_score(analysis['guidance_assessment']['direction'])
        qa_quality = analysis['qa_quality']['score']

        # Корректировка для исторического базового уровня
        if historical_sentiment is not None:
            sentiment_delta = sentiment - historical_sentiment
            sentiment = sentiment + 0.5 * sentiment_delta
            sentiment = max(-1, min(1, sentiment))

        # Расчёт композитного показателя
        composite = (
            self.weights['sentiment'] * sentiment +
            self.weights['confidence'] * (confidence - 0.5) * 2 +
            self.weights['guidance'] * guidance +
            self.weights['qa'] * (qa_quality - 0.5) * 2
        )

        # Определение направления
        if composite > self.signal_threshold:
            direction = SignalDirection.BULLISH
        elif composite < -self.signal_threshold:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Расчёт силы
        strength = min(abs(composite), 1.0)

        return TradingSignal(
            direction=direction,
            strength=strength,
            confidence=self._calculate_confidence(sentiment, confidence, qa_quality, analysis),
            sentiment_score=sentiment,
            confidence_level=confidence,
            guidance_direction=analysis['guidance_assessment']['direction'],
            qa_quality=qa_quality,
            reasoning=self._generate_reasoning(sentiment, confidence, guidance, qa_quality, direction)
        )
```

## Требования к Данным

```
Данные Earnings Calls:
├── Источник: Сайты IR компаний, SEC EDGAR, поставщики данных
├── Формат: Текстовые транскрипты, аудио файлы
├── Частота: Ежеквартально (4 раза в год на компанию)
├── История: Рекомендуется 2+ года для бэктестинга
│
Обязательные Поля:
├── Идентификатор компании (тикер, CUSIP)
├── Дата и время earnings
├── Полный текст транскрипта
├── Идентификация спикеров
├── Маркеры секций (подготовленные/Q&A)
│
Данные Цен:
├── Источник: Yahoo Finance, Polygon.io, Alpha Vantage
├── Частота: Дневные OHLCV
├── Цены до/после earnings
└── Данные объёма для проверки ликвидности
```

## Применение к Криптовалютам

### Применение к Крипто-Рынкам

Хотя традиционные earnings calls не существуют для криптовалют, аналогичный анализ может применяться к:

1. **Обновления Проектов**: Ежеквартальные/ежемесячные обновления разработки
2. **AMA Сессии**: Сессии вопросов и ответов с командами проектов
3. **Звонки по Управлению**: Обсуждения управления DAO
4. **Объявления о Партнёрствах**: Крупные объявления о сотрудничестве
5. **Обсуждения Обновлений Протокола**: Объяснения технических обновлений

### Интеграция с Bybit для Крипто-Данных

```python
import requests
from typing import List, Dict
from datetime import datetime

class BybitClient:
    """
    Клиент для получения рыночных данных Bybit
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(self,
                   symbol: str,
                   interval: str,
                   limit: int = 200) -> List[Dict]:
        """
        Получение данных свечей с Bybit

        Args:
            symbol: Торговая пара (напр., "BTCUSDT")
            interval: Интервал свечей (напр., "1h", "4h", "1d")
            limit: Количество свечей для получения

        Returns:
            Список словарей свечей
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Ошибка Bybit API: {data['retMsg']}")

        candles = []
        for item in data['result']['list']:
            candles.append({
                'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })

        return candles[::-1]  # Обратный порядок в хронологический
```

## Ключевые Метрики

- **Точность Настроения**: Корреляция между предсказанным настроением и движением цены
- **Точность Сигнала**: Процент правильных предсказаний направления
- **Коэффициент Шарпа**: Доходность с поправкой на риск стратегии earnings
- **Информационный Коэффициент**: Корреляция между силой сигнала и доходностью
- **Доля Выигрыша**: Процент прибыльных сделок
- **Максимальная Просадка**: Наибольшее падение от пика до минимума

## Зависимости

```python
# Основные
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# LLM API
openai>=1.0.0
anthropic>=0.7.0
tiktoken>=0.5.0

# NLP
transformers>=4.30.0
spacy>=3.5.0

# Рыночные Данные
yfinance>=0.2.0
ccxt>=4.0.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0

# Утилиты
python-dotenv>=1.0.0
tqdm>=4.65.0
```

## Ожидаемые Результаты

1. **Извлечение Настроения**: Точное извлечение общего тона и специфических маркеров настроения
2. **Обнаружение Уверенности**: Идентификация уклончивости vs уверенности руководства
3. **Анализ Прогнозов**: Правильная классификация направления прогноза
4. **Оценка Q&A**: Оценка прозрачности руководства
5. **Торговые Сигналы**: Практичные сигналы с калиброванной уверенностью
6. **Результаты Бэктеста**: Ожидаемый коэффициент Шарпа 0.8-1.5 на объявлениях earnings

## Ссылки

1. **Large Language Models in Equity Markets: Applications, Techniques, and Insights** (2025)
   - URL: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365

2. **MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents** (2025)
   - URL: https://arxiv.org/abs/2502.00415

3. **Can ChatGPT Forecast Stock Price Movements?** (Lopez-Lira & Tang, 2024)
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788

4. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** (Araci, 2019)
   - URL: https://arxiv.org/abs/1908.10063

5. **Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Alpha** (2025)
   - URL: https://arxiv.org/abs/2508.04975

6. **Can Large Language Models Forecast Time Series of Earnings per Share?** (2025)
   - URL: https://www.tandfonline.com/doi/full/10.1080/00128775.2025.2534144

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительного анализа earnings calls для криптовалютных данных с Bybit. Смотрите директорию `rust/`.

### Возможности:
- Парсинг транскриптов и определение секций
- Анализ настроения с финансовым лексиконом
- Интеграция LLM API для продвинутого анализа
- Генерация торговых сигналов
- Фреймворк бэктестинга с полными метриками
- Получение данных в реальном времени с Bybit API
- Модульный и расширяемый дизайн

## Уровень Сложности

Продвинутый

Требуется понимание: Обработка Естественного Языка, Анализ Настроения, LLM API, Финансовый Анализ, Торговые Системы, Событийные Стратегии
