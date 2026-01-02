# Глава 340: Ассоциативная Память в Трейдинге — Предсказания на Основе Поиска Паттернов в Плотных Ассоциативных Сетях

## Обзор

Ассоциативная память представляет собой мощную парадигму для распознавания и извлечения паттернов в трейдинге. В отличие от традиционных нейронных сетей прямого распространения, которые учатся отображать входы на выходы, ассоциативная память хранит паттерны и извлекает наиболее похожий при получении частичного или зашумлённого входа. Это делает её особенно подходящей для распознавания рыночных паттернов и прогнозирования на основе исторического сходства.

## Ключевые Концепции

### Что такое Ассоциативная Память?

Ассоциативная память — это система памяти с адресацией по содержимому, которая:
1. **Хранит** набор паттернов во время обучения
2. **Извлекает** наиболее похожий сохранённый паттерн при получении запроса
3. **Дополняет** частичные паттерны, заполняя недостающую информацию
4. **Корректирует** зашумлённые входы, сходясь к ближайшему сохранённому паттерну

### От Сетей Хопфилда к Современной Ассоциативной Памяти

```
Классическая сеть Хопфилда (1982):
├── Бинарные паттерны: {-1, +1}
├── Энергетическая динамика
├── Ограниченная ёмкость: ~0.14N паттернов (N = нейроны)
└── Сходится к сохранённым аттракторам

Современная Плотная Ассоциативная Память (2016):
├── Непрерывные паттерны: ℝ^d
├── Экспоненциальная ёмкость хранения
├── Механизм извлечения подобный вниманию
└── Дифференцируема для интеграции с глубоким обучением
```

### Почему Ассоциативная Память для Трейдинга?

1. **Сопоставление паттернов**: Рынки демонстрируют повторяющиеся паттерны; АП находит наиболее похожий исторический режим
2. **Устойчивость к шуму**: Реальные рыночные данные зашумлены; АП естественно фильтрует шум
3. **Интерпретируемость**: Извлечённые паттерны дают контекст для прогнозов
4. **Эффективность памяти**: Хранение репрезентативных паттернов вместо всех данных
5. **Обучение на одном примере**: Обучение на редких, но важных рыночных событиях

## Торговая Стратегия

**Обзор стратегии:** Использование Плотной Ассоциативной Памяти для идентификации похожих исторических рыночных паттернов и прогнозирования будущих движений цен на основе того, что происходило после этих паттернов.

### Генерация Сигналов

```
1. Извлечение признаков:
   - Вычисление рыночных признаков: доходности, волатильность, паттерны объёма
   - Нормализация для создания вектора паттерна

2. Извлечение паттерна:
   - Запрос к ассоциативной памяти с текущим паттерном
   - Извлечение K наиболее похожих исторических паттернов
   - Взвешивание по оценке сходства

3. Прогнозирование:
   - Агрегация исходов из извлечённых паттернов
   - Генерация направленного сигнала с уверенностью

4. Размер позиции:
   - Масштабирование позиции по уверенности извлечения
   - Высокое сходство = большая позиция
```

### Сигналы на Вход

- **Лонг-сигнал**: Извлечённые паттерны преимущественно сопровождались положительными доходностями
- **Шорт-сигнал**: Извлечённые паттерны преимущественно сопровождались отрицательными доходностями
- **Порог уверенности**: Торговать только когда оценка сходства превышает порог

### Управление Рисками

- **Детекция новизны**: Низкое сходство указывает на новые рыночные условия → снижение экспозиции
- **Проверка консенсуса**: Несколько извлечённых паттернов должны согласоваться по направлению
- **Масштабирование по волатильности**: Корректировка размера позиции на основе ожидаемой волатильности из извлечённых паттернов

## Техническая Спецификация

### Математические Основы

#### Классическая Энергия Хопфилда

Для бинарных паттернов x ∈ {-1, +1}^N с матрицей весов W:

```
E(x) = -½ x^T W x

Правило обновления (асинхронное):
x_i ← sign(Σ_j W_ij x_j)

Обучение весов (по Хеббу):
W_ij = (1/P) Σ_μ ξ_i^μ ξ_j^μ
```

#### Современная Плотная Ассоциативная Память

Для паттернов {ξ^μ} ∈ ℝ^d функция энергии становится:

```
E(x) = -log Σ_μ exp(β x · ξ^μ)

Динамика извлечения:
x_new = Σ_μ softmax(β x · ξ^μ) ξ^μ

Это эквивалентно механизму внимания!
```

#### Связь с Механизмом Внимания

```
Запрос:    q = W_q x
Ключи:     K = [ξ^1, ξ^2, ..., ξ^M]
Значения:  V = [v^1, v^2, ..., v^M]

Выход внимания:
output = Σ_μ softmax(q · ξ^μ / √d) v^μ
```

### Архитектурная Диаграмма

```
                    Поток Рыночных Данных
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Инженерия Признаков      │
            │  ├── Доходности и Волатил.  │
            │  ├── Технические Индикаторы │
            │  ├── Паттерны Объёма        │
            │  └── Кросс-активные Признаки│
            └──────────────┬──────────────┘
                           │
                           ▼ Паттерн запроса x
            ┌─────────────────────────────┐
            │ Плотная Ассоциативная Память│
            │                             │
            │  ┌───────────────────────┐  │
            │  │   Память Паттернов    │  │
            │  │   ξ^1, ξ^2, ..., ξ^M  │  │
            │  │ (Исторические сост.)  │  │
            │  └───────────────────────┘  │
            │            ↓ ↓ ↓            │
            │  ┌───────────────────────┐  │
            │  │  Оценка Сходства      │  │
            │  │  softmax(β x · ξ^μ)   │  │
            │  └───────────────────────┘  │
            │            ↓ ↓ ↓            │
            │  ┌───────────────────────┐  │
            │  │  Взвешенное Извлечение│  │
            │  │  Σ α_μ v^μ            │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │ Прогнозир.  │ │ Уверенность │ │  Похожие    │
     │ Доходности  │ │ Извлечения  │ │  Паттерны   │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Торговое Решение        │
            │  ├── Направление Сигнала    │
            │  ├── Размер Позиции         │
            │  └── Параметры Риска        │
            └─────────────────────────────┘
```

### Инженерия Признаков для Хранения Паттернов

```python
def compute_market_pattern(df, lookback=20):
    """
    Создание вектора рыночного паттерна для ассоциативной памяти
    """
    pattern = {}

    # Признаки на основе доходностей
    returns = df['close'].pct_change()
    pattern['return_mean'] = returns.rolling(lookback).mean().iloc[-1]
    pattern['return_std'] = returns.rolling(lookback).std().iloc[-1]
    pattern['return_skew'] = returns.rolling(lookback).skew().iloc[-1]
    pattern['return_kurt'] = returns.rolling(lookback).kurt().iloc[-1]

    # Признаки тренда
    pattern['trend_5'] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
    pattern['trend_20'] = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)

    # Признаки волатильности
    pattern['volatility'] = returns.rolling(lookback).std().iloc[-1] * np.sqrt(252)
    pattern['volatility_change'] = (
        returns.rolling(5).std().iloc[-1] /
        returns.rolling(20).std().iloc[-1]
    )

    # Признаки объёма
    volume_ma = df['volume'].rolling(lookback).mean()
    pattern['volume_ratio'] = df['volume'].iloc[-1] / volume_ma.iloc[-1]

    # Признаки диапазона
    atr = (df['high'] - df['low']).rolling(lookback).mean()
    pattern['atr_ratio'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / atr.iloc[-1]

    # Позиция закрытия в диапазоне
    pattern['close_position'] = (
        (df['close'].iloc[-1] - df['low'].iloc[-lookback:].min()) /
        (df['high'].iloc[-lookback:].max() - df['low'].iloc[-lookback:].min())
    )

    return np.array(list(pattern.values()))
```

### Реализация Плотной Ассоциативной Памяти

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseAssociativeMemory(nn.Module):
    """
    Плотная Ассоциативная Память для извлечения паттернов

    Хранит паттерны и извлекает на основе сходства,
    используя softmax-внимание для гладкого дифференцируемого извлечения.
    """

    def __init__(self, pattern_dim: int, memory_size: int,
                 beta: float = 1.0, n_heads: int = 4):
        super().__init__()

        self.pattern_dim = pattern_dim
        self.memory_size = memory_size
        self.beta = beta
        self.n_heads = n_heads

        # Обучаемая память паттернов
        self.patterns = nn.Parameter(torch.randn(memory_size, pattern_dim))
        self.values = nn.Parameter(torch.randn(memory_size, pattern_dim))

        # Компоненты многоголового внимания
        self.head_dim = pattern_dim // n_heads
        self.W_q = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_k = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_v = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_o = nn.Linear(pattern_dim, pattern_dim, bias=False)

        # Голова предсказания
        self.predictor = nn.Sequential(
            nn.Linear(pattern_dim, pattern_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(pattern_dim // 2, 1),
            nn.Tanh()
        )

    def store_patterns(self, patterns: torch.Tensor, values: torch.Tensor):
        """
        Сохранение паттернов и их связанных значений (меток/исходов)
        """
        with torch.no_grad():
            # Нормализация паттернов
            patterns_norm = F.normalize(patterns, dim=-1)
            self.patterns.data = patterns_norm
            self.values.data = values

    def retrieve(self, query: torch.Tensor, return_attention: bool = False):
        """
        Извлечение из памяти с использованием механизма внимания

        Args:
            query: (batch, pattern_dim) паттерн запроса
            return_attention: возвращать ли веса внимания

        Returns:
            retrieved: (batch, pattern_dim) извлечённый паттерн
            attention: (batch, memory_size) веса внимания (опционально)
        """
        batch_size = query.shape[0]

        # Проекция запроса
        q = self.W_q(query)  # (batch, pattern_dim)

        # Изменение формы для многоголового внимания
        q = q.view(batch_size, self.n_heads, self.head_dim)

        # Проекция памяти
        k = self.W_k(self.patterns)  # (memory_size, pattern_dim)
        v = self.W_v(self.values)    # (memory_size, pattern_dim)

        k = k.view(1, self.memory_size, self.n_heads, self.head_dim)
        v = v.view(1, self.memory_size, self.n_heads, self.head_dim)

        # Вычисление оценок внимания
        scores = torch.einsum('bhd,bmhd->bhm', q, k)
        scores = scores * self.beta / (self.head_dim ** 0.5)

        # Softmax внимание
        attention = F.softmax(scores, dim=-1)

        # Извлечение значений
        retrieved = torch.einsum('bhm,bmhd->bhd', attention, v)
        retrieved = retrieved.view(batch_size, self.pattern_dim)

        # Выходная проекция
        retrieved = self.W_o(retrieved)

        if return_attention:
            attention_avg = attention.mean(dim=1)
            return retrieved, attention_avg

        return retrieved

    def forward(self, query: torch.Tensor):
        """
        Извлечение и предсказание

        Args:
            query: (batch, pattern_dim) текущий рыночный паттерн

        Returns:
            prediction: (batch, 1) предсказанное направление
            confidence: (batch, 1) уверенность извлечения
        """
        retrieved, attention = self.retrieve(query, return_attention=True)

        # Предсказание из извлечённого паттерна
        prediction = self.predictor(retrieved)

        # Уверенность на основе концентрации внимания
        entropy = -(attention * (attention + 1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.memory_size, dtype=torch.float32))
        confidence = 1 - (entropy / max_entropy)

        return prediction, confidence
```

### Непрерывная Сеть Хопфилда (Современная Версия)

```python
class ContinuousHopfieldNetwork(nn.Module):
    """
    Непрерывная Сеть Хопфилда с экспоненциальной ёмкостью хранения

    На основе "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, beta: float = None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.beta = beta if beta else (1.0 / (hidden_dim ** 0.5))

        # Проекция паттерна
        self.W_pattern = nn.Linear(input_dim, hidden_dim)
        self.W_query = nn.Linear(input_dim, hidden_dim)

        # Выходная проекция
        self.W_out = nn.Linear(hidden_dim, input_dim)

        # Память паттернов (устанавливается при обучении)
        self.register_buffer('stored_patterns', None)

    def store(self, patterns: torch.Tensor):
        """
        Сохранение паттернов в памяти
        """
        with torch.no_grad():
            projected = self.W_pattern(patterns)
            self.stored_patterns = F.normalize(projected, dim=-1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Вычисление энергии текущего состояния

        E(x) = -log Σ exp(β ξ^T x)

        Меньшая энергия = ближе к сохранённым паттернам
        """
        if self.stored_patterns is None:
            raise ValueError("Паттерны не сохранены")

        state_proj = self.W_query(state)
        state_proj = F.normalize(state_proj, dim=-1)

        similarities = state_proj @ self.stored_patterns.T
        energy = -torch.logsumexp(self.beta * similarities, dim=-1)

        return energy

    def update(self, state: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """
        Обновление состояния в направлении сохранённых паттернов
        """
        if self.stored_patterns is None:
            raise ValueError("Паттерны не сохранены")

        for _ in range(n_steps):
            state_proj = self.W_query(state)
            state_proj = F.normalize(state_proj, dim=-1)

            similarities = state_proj @ self.stored_patterns.T
            attention = F.softmax(self.beta * similarities, dim=-1)

            new_state_proj = attention @ self.stored_patterns
            state = self.W_out(new_state_proj)

        return state

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Извлечение ближайшего паттерна к запросу
        """
        return self.update(query, n_steps=3)
```

### Управление Памятью Паттернов

```python
class PatternMemoryManager:
    """
    Управление хранением паттернов с ограничениями ёмкости
    и оценкой релевантности
    """

    def __init__(self, max_patterns: int, pattern_dim: int,
                 similarity_threshold: float = 0.95):
        self.max_patterns = max_patterns
        self.pattern_dim = pattern_dim
        self.similarity_threshold = similarity_threshold

        self.patterns = []
        self.outcomes = []
        self.timestamps = []
        self.retrieval_counts = []

    def add_pattern(self, pattern: np.ndarray, outcome: float, timestamp):
        """
        Добавление паттерна в память с возможной заменой старых/похожих
        """
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)

        # Проверка на похожие существующие паттерны
        if len(self.patterns) > 0:
            patterns_arr = np.array(self.patterns)
            similarities = patterns_arr @ pattern

            # Если очень похожий паттерн существует, обновляем его
            if np.max(similarities) > self.similarity_threshold:
                idx = np.argmax(similarities)
                alpha = 0.3
                self.patterns[idx] = alpha * pattern + (1 - alpha) * self.patterns[idx]
                self.patterns[idx] /= np.linalg.norm(self.patterns[idx])
                self.outcomes[idx] = alpha * outcome + (1 - alpha) * self.outcomes[idx]
                return

        # Добавление нового паттерна
        if len(self.patterns) >= self.max_patterns:
            self._evict_pattern()

        self.patterns.append(pattern)
        self.outcomes.append(outcome)
        self.timestamps.append(timestamp)
        self.retrieval_counts.append(0)

    def _evict_pattern(self):
        """
        Удаление наименее полезного паттерна на основе давности и использования
        """
        if len(self.patterns) == 0:
            return

        n = len(self.patterns)
        recency_scores = np.arange(n) / n
        usage_scores = np.array(self.retrieval_counts)
        usage_scores = usage_scores / (usage_scores.max() + 1)

        scores = 0.5 * recency_scores + 0.5 * usage_scores

        remove_idx = np.argmin(scores)
        self.patterns.pop(remove_idx)
        self.outcomes.pop(remove_idx)
        self.timestamps.pop(remove_idx)
        self.retrieval_counts.pop(remove_idx)

    def predict(self, query: np.ndarray, k: int = 5) -> tuple:
        """
        Предсказание исхода на основе похожих паттернов
        """
        patterns, outcomes, similarities = self.query(query, k)

        if len(patterns) == 0:
            return 0.0, 0.0

        similarities = np.array(similarities)
        weights = similarities / (similarities.sum() + 1e-8)
        prediction = np.sum(weights * np.array(outcomes))
        confidence = np.mean(similarities)

        return prediction, confidence
```

## Требования к Данным

```
Исторические OHLCV Данные:
├── Минимум: 1 год часовых данных
├── Рекомендуется: 3+ года для разнообразных паттернов
├── Частота: рекомендуется 1 час - 1 день
└── Источник: Bybit, Binance или другие биржи

Обязательные поля:
├── timestamp (временная метка)
├── open, high, low, close (OHLC)
├── volume (объём)
└── Опционально: оборот, количество сделок

Построение паттернов:
├── Глубина: 20-60 периодов для паттерна
├── Признаки: 10-50 измерений
├── Нормализация: Z-оценка или min-max
└── Обновление: скользящее окно
```

## Ключевые Метрики

- **Точность извлечения**: Процент случаев, когда правильный паттерн в топ-K
- **Покрытие паттернов**: Процент рыночных условий, покрытых сохранёнными паттернами
- **Калибровка уверенности**: Корреляция между уверенностью и точностью
- **Коэффициент Шарпа**: Риск-скорректированная доходность
- **Максимальная просадка**: Наибольшее снижение от пика до минимума
- **Доля выигрышей**: Процент прибыльных сделок

## Зависимости

```python
# Базовые
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Глубокое обучение
torch>=2.0.0
pytorch-lightning>=2.0.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0

# Рыночные данные
ccxt>=4.0.0
websocket-client>=1.4.0

# Утилиты
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Ожидаемые Результаты

1. **Система распознавания паттернов** с экспоненциальной ёмкостью хранения
2. **Предсказания на основе извлечения** с использованием механизмов внимания
3. **Торговля с калиброванной уверенностью** и детекцией новизны
4. **Интерпретируемые решения** через анализ похожих паттернов
5. **Результаты бэктеста**: Ожидаемый коэффициент Шарпа 1.0-2.0 при правильной настройке

## Ссылки

1. **Dense Associative Memory for Pattern Recognition** (Krotov & Hopfield, 2016)
   - URL: https://arxiv.org/abs/1606.01164

2. **Hopfield Networks is All You Need** (Ramsauer et al., 2020)
   - URL: https://arxiv.org/abs/2008.02217

3. **Modern Hopfield Networks and Attention for Immune Repertoire Classification** (Widrich et al., 2020)
   - URL: https://arxiv.org/abs/2007.13505

4. **Associative Memory in Machine Learning** - Обзор и применения

5. **Neural Networks and Deep Learning** (Michael Nielsen) - Глава о сетях Хопфилда

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительной торговли с ассоциативной памятью на данных криптовалют с Bybit. Смотрите директорию `rust/`.

### Возможности:
- Получение данных с Bybit в реальном времени
- Реализация Плотной Ассоциативной Памяти
- Хранение и извлечение паттернов
- Торговые сигналы на основе уверенности
- Фреймворк бэктестинга
- Модульный и расширяемый дизайн

## Уровень Сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание: Нейронные Сети, Механизмы Внимания, Энергетические Модели, Распознавание Паттернов, Торговые Системы
