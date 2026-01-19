# Глава 73: Оценка рисков с помощью LLM для трейдинга

## Обзор

Большие языковые модели (LLM) могут анализировать неструктурированные текстовые данные для оценки различных типов рисков на финансовых рынках. В этой главе мы исследуем, как использовать LLM для оценки рисков в торговых стратегиях, комбинируя анализ новостей, извлечение настроений и скоринг рисков для принятия обоснованных торговых решений.

## Торговая стратегия

**Основная концепция:** LLM обрабатывают финансовые новости, отчёты о доходах, документы SEC и социальные сети для генерации оценок риска, которые влияют на размер позиций и торговые решения.

**Сигналы на вход:**
- Long: Низкая оценка риска + положительный импульс настроений
- Short: Высокая оценка риска + негативные индикаторы настроений
- Выход: Оценка риска пересекает пороговое значение или разворот настроений

**Преимущество:** LLM могут обрабатывать огромные объёмы неструктурированного текста быстрее, чем аналитики-люди, выявляя тонкие сигналы риска в языковых паттернах, которые коррелируют с будущими движениями цены.

## Техническая спецификация

### Ключевые компоненты

1. **Конвейер текстовых данных** — Сбор новостей, документов, социальных сетей
2. **LLM-скоринг рисков** — Генерация оценок риска из текста
3. **Генерация сигналов** — Преобразование оценок в торговые сигналы
4. **Управление позициями** — Размер позиций с учётом риска
5. **Фреймворк бэктестинга** — Валидация эффективности стратегии

### Архитектура

```
                    ┌─────────────────────┐
                    │   Источники данных  │
                    │   (Новости, Файлы)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Обработчик текста  │
                    │  (Очистка, Чанки)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM-движок риска  │
                    │  (Промпт + Модель)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Агрегатор рисков   │
                    │  (Оценка + История) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Торговый движок   │
                    │  (Сигналы + Ордера) │
                    └─────────────────────┘
```

### Требования к данным

```
Источники новостей:
├── API финансовых новостей (Alpha Vantage, NewsAPI)
├── Документы SEC EDGAR (10-K, 10-Q, 8-K)
├── Транскрипты звонков по доходам
├── Социальные сети (Twitter/X, Reddit, StockTwits)
└── Криптовалютные новости (CoinDesk, CryptoNews)

Рыночные данные:
├── OHLCV ценовые данные (Bybit для крипты, Yahoo для акций)
├── Снимки стакана ордеров
├── Аналитика торгового объёма
└── Метрики волатильности
```

### Категории рисков

LLM оценивает несколько измерений риска:

| Тип риска | Описание | Индикаторы |
|-----------|----------|------------|
| **Рыночный риск** | Подверженность движениям рынка | Упоминания волатильности, макро-беспокойства |
| **Кредитный риск** | Вероятность дефолта контрагента | Уровни долга, изменения рейтинга |
| **Риск ликвидности** | Возможность торговать без влияния | Беспокойства по объёму, расширение спреда |
| **Операционный риск** | Сбои систем/процессов | Технические проблемы, смена руководства |
| **Регуляторный риск** | Правовые/комплаенс риски | Судебные иски, регуляторные действия |
| **Сентимент-риск** | Изменения восприятия рынка | Тон соцсетей, покрытие аналитиками |

### Инженерия промптов

```python
RISK_ASSESSMENT_PROMPT = """
Проанализируй следующий финансовый текст и предоставь оценку риска.

Текст: {text}

Оцени следующие измерения риска по шкале 1-10 (1=минимальный риск, 10=максимальный риск):

1. Рыночный риск: Подверженность волатильности рынка и системному риску
2. Кредитный риск: Индикаторы контрагентского риска или риска дефолта
3. Риск ликвидности: Беспокойства по торговле и глубине рынка
4. Операционный риск: Риски исполнения бизнеса и управления
5. Регуляторный риск: Правовые и комплаенс риски
6. Сентимент-риск: Восприятие рынка и репутация

Для каждого измерения предоставь:
- Оценку (1-10)
- Выявленные ключевые факторы
- Уровень уверенности (низкий/средний/высокий)

Также предоставь:
- Общую оценку риска (взвешенное среднее)
- Направление риска (растущий/стабильный/снижающийся)
- Временной горизонт (краткосрочный/среднесрочный/долгосрочный)

Вывод в формате JSON.
"""
```

### Ключевые метрики

**Качество оценки рисков:**
- Точность предсказания (оценка риска vs реализованная волатильность)
- Информационный коэффициент (IC) с будущими доходностями
- Уровень ложноположительных/ложноотрицательных срабатываний для рисковых событий

**Эффективность стратегии:**
- Коэффициент Шарпа
- Максимальная просадка
- Доходность с учётом риска
- Процент успешных предсказаний риска

### Зависимости

```python
# Python зависимости
openai>=1.0.0           # OpenAI API клиент
anthropic>=0.5.0        # Claude API клиент
transformers>=4.30.0    # HuggingFace модели
torch>=2.0.0            # PyTorch
pandas>=2.0.0           # Манипуляция данными
numpy>=1.24.0           # Численные вычисления
yfinance>=0.2.0         # Данные акций
requests>=2.28.0        # HTTP клиент
beautifulsoup4>=4.12.0  # Парсинг HTML
```

```rust
// Rust зависимости
reqwest = "0.12"        // HTTP клиент
serde = "1.0"           // Сериализация
tokio = "1.0"           // Async runtime
ndarray = "0.16"        // Массивы
polars = "0.46"         // DataFrames
```

## Реализация на Python

### Базовая оценка риска

```python
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import openai

class RiskLevel(Enum):
    LOW = 1        # Низкий
    MODERATE = 2   # Умеренный
    HIGH = 3       # Высокий
    SEVERE = 4     # Критический

@dataclass
class RiskScore:
    """Оценка риска по различным измерениям."""
    market_risk: float       # Рыночный риск
    credit_risk: float       # Кредитный риск
    liquidity_risk: float    # Риск ликвидности
    operational_risk: float  # Операционный риск
    regulatory_risk: float   # Регуляторный риск
    sentiment_risk: float    # Сентимент-риск
    overall_score: float     # Общая оценка
    confidence: str          # Уверенность
    direction: str           # Направление

    @property
    def risk_level(self) -> RiskLevel:
        """Определить уровень риска по общей оценке."""
        if self.overall_score <= 3:
            return RiskLevel.LOW
        elif self.overall_score <= 5:
            return RiskLevel.MODERATE
        elif self.overall_score <= 7:
            return RiskLevel.HIGH
        return RiskLevel.SEVERE

class LLMRiskAssessor:
    """Движок оценки рисков на основе LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def assess_risk(self, text: str) -> RiskScore:
        """Проанализировать текст и вернуть оценки риска."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Вы финансовый аналитик рисков."},
                {"role": "user", "content": self._build_prompt(text)}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_response(result)

    def _build_prompt(self, text: str) -> str:
        return f"""Проанализируй этот финансовый текст на предмет риска:

{text}

Верни JSON с оценками (1-10) для: market_risk, credit_risk,
liquidity_risk, operational_risk, regulatory_risk, sentiment_risk,
overall_score, confidence (low/medium/high), direction (increasing/stable/decreasing)"""

    def _parse_response(self, data: dict) -> RiskScore:
        return RiskScore(
            market_risk=float(data.get("market_risk", 5)),
            credit_risk=float(data.get("credit_risk", 5)),
            liquidity_risk=float(data.get("liquidity_risk", 5)),
            operational_risk=float(data.get("operational_risk", 5)),
            regulatory_risk=float(data.get("regulatory_risk", 5)),
            sentiment_risk=float(data.get("sentiment_risk", 5)),
            overall_score=float(data.get("overall_score", 5)),
            confidence=data.get("confidence", "medium"),
            direction=data.get("direction", "stable")
        )
```

### Интеграция с торговой стратегией

```python
import pandas as pd
import numpy as np
from typing import Tuple, List

class RiskBasedTrader:
    """Торговая стратегия на основе оценки рисков LLM."""

    def __init__(
        self,
        risk_assessor: LLMRiskAssessor,
        risk_threshold_long: float = 4.0,   # Порог для лонга
        risk_threshold_short: float = 7.0,  # Порог для шорта
        max_position_size: float = 1.0      # Максимальный размер позиции
    ):
        self.assessor = risk_assessor
        self.risk_threshold_long = risk_threshold_long
        self.risk_threshold_short = risk_threshold_short
        self.max_position_size = max_position_size
        self.risk_history: List[RiskScore] = []

    def generate_signal(self, text: str, current_price: float) -> Tuple[str, float]:
        """Сгенерировать торговый сигнал из анализа текста.

        Returns:
            Кортеж (сигнал, размер_позиции)
            сигнал: 'long', 'short', или 'neutral'
        """
        risk_score = self.assessor.assess_risk(text)
        self.risk_history.append(risk_score)

        # Рассчитать размер позиции обратно пропорционально риску
        position_size = self._calculate_position_size(risk_score)

        # Сгенерировать сигнал на основе уровня и направления риска
        if risk_score.overall_score <= self.risk_threshold_long:
            if risk_score.direction in ['stable', 'decreasing']:
                return ('long', position_size)

        elif risk_score.overall_score >= self.risk_threshold_short:
            if risk_score.direction in ['stable', 'increasing']:
                return ('short', position_size * 0.5)  # Меньшие позиции для шорта

        return ('neutral', 0.0)

    def _calculate_position_size(self, risk: RiskScore) -> float:
        """Рассчитать размер позиции на основе оценки риска."""
        # Ниже риск = больше позиция (обратная зависимость)
        risk_factor = 1 - (risk.overall_score / 10)

        # Корректировка по уверенности
        confidence_multiplier = {
            'high': 1.0,    # Высокая
            'medium': 0.75, # Средняя
            'low': 0.5      # Низкая
        }.get(risk.confidence, 0.5)

        return self.max_position_size * risk_factor * confidence_multiplier

    def get_risk_trend(self, window: int = 5) -> str:
        """Проанализировать тренд риска за последнее время."""
        if len(self.risk_history) < window:
            return 'insufficient_data'  # Недостаточно данных

        recent_scores = [r.overall_score for r in self.risk_history[-window:]]
        slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if slope > 0.5:
            return 'increasing'   # Растущий
        elif slope < -0.5:
            return 'decreasing'   # Снижающийся
        return 'stable'           # Стабильный
```

## Реализация на Rust

Смотрите директорию `rust_llm_risk/` для полной реализации на Rust, которая включает:

- **Получение данных** из Bybit и новостных API
- **Обработка текста** и разбиение на чанки
- **Интеграция с LLM API** (совместимо с OpenAI)
- **Движок скоринга** рисков
- **Фреймворк бэктестинга**
- **Реализация торговой** стратегии

### Быстрый старт (Rust)

```bash
cd rust_llm_risk

# Сборка проекта
cargo build --release

# Получение рыночных данных
cargo run --example fetch_data

# Запуск оценки рисков
cargo run --example assess_risk -- --symbol BTCUSDT --days 30

# Бэктест стратегии
cargo run --example backtest -- --start 2024-01-01 --end 2024-06-01
```

## Ожидаемые результаты

1. **Конвейер оценки рисков** — Система end-to-end для LLM-скоринга рисков
2. **Торговые сигналы** — Сигналы входа/выхода с учётом риска
3. **Размер позиций** — Динамический расчёт на основе уровней риска
4. **Метрики эффективности** — Улучшение коэффициента Шарпа vs базовая модель
5. **Мониторинг в реальном времени** — Дашборд для отслеживания рисков

## Сценарии использования

### Торговля криптовалютами
- Мониторинг соцсетей на предмет изменения настроений
- Анализ объявлений бирж на предмет рисков
- Отслеживание регуляторных новостей по юрисдикциям

### Торговля акциями
- Анализ звонков по доходам на предмет специфических рисков компании
- Парсинг документов SEC для скрытых факторов риска
- Агрегация настроений новостей для секторных рисков

### Торговля опционами
- Предсказание событий волатильности из новостей
- Скоринг рисков для торговли на отчётности
- Оценка рисков слияний и поглощений

## Лучшие практики

1. **Инженерия промптов** — Тестируйте и уточняйте промпты для стабильной оценки
2. **Выбор модели** — Используйте подходящую модель для сложности задачи
3. **Ограничение частоты** — Внедрите кэширование для управления стоимостью API
4. **Валидация** — Проводите обширный бэктестинг перед реальной торговлей
5. **Человеческий контроль** — Проверяйте высокорисковые оценки вручную

## Ссылки

- [Risk Assessment with Large Language Models](https://arxiv.org/abs/2310.01926)
- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031)
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
- [Sentiment Analysis in Financial Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4135500)

## Уровень сложности

Эксперт

Требуемые знания: промптинг LLM, NLP, управление финансовыми рисками, торговые системы, интеграция API
