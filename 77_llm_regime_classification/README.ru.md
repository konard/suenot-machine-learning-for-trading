# Глава 77: Классификация рыночных режимов с помощью LLM

В этой главе рассматривается **классификация рыночных режимов на основе больших языковых моделей (LLM)** для трейдинга и инвестиционных стратегий. Мы демонстрируем, как LLM могут идентифицировать различные рыночные условия (бычий, медвежий, боковой, волатильный) с использованием как числовых данных, так и текстовой информации из новостей, социальных сетей и экономических индикаторов.

<p align="center">
<img src="https://i.imgur.com/8XdE3Wz.png" width="70%">
</p>

## Содержание

1. [Введение в классификацию рыночных режимов](#введение-в-классификацию-рыночных-режимов)
    * [Почему важна классификация режимов](#почему-важна-классификация-режимов)
    * [Традиционные методы vs методы на основе LLM](#традиционные-методы-vs-методы-на-основе-llm)
    * [Определение рыночных режимов](#определение-рыночных-режимов)
2. [Теоретические основы](#теоретические-основы)
    * [Скрытые марковские модели как базовый метод](#скрытые-марковские-модели-как-базовый-метод)
    * [Обучение представлений с помощью LLM](#обучение-представлений-с-помощью-llm)
    * [Мультимодальное определение режимов](#мультимодальное-определение-режимов)
3. [Методы классификации](#методы-классификации)
    * [Классификация на основе текста](#классификация-на-основе-текста)
    * [Определение режимов по временным рядам](#определение-режимов-по-временным-рядам)
    * [Гибридные LLM-статистические подходы](#гибридные-llm-статистические-подходы)
4. [Практические примеры](#практические-примеры)
    * [01: Классификация режимов фондового рынка](#01-классификация-режимов-фондового-рынка)
    * [02: Режимы криптовалютного рынка (Bybit)](#02-режимы-криптовалютного-рынка-bybit)
    * [03: Согласованность режимов для нескольких активов](#03-согласованность-режимов-для-нескольких-активов)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в классификацию рыночных режимов

Классификация рыночных режимов — это задача определения текущего состояния или "режима" финансовых рынков. Различные режимы требуют различных торговых стратегий — то, что работает на бычьем рынке, может привести к катастрофическим убыткам на медвежьем. LLM предлагают мощный подход к этой задаче благодаря способности понимать контекст из множества источников данных.

### Почему важна классификация режимов

```
ВАЖНОСТЬ ТОРГОВЛИ С УЧЁТОМ РЕЖИМОВ:
======================================================================

+------------------------------------------------------------------+
|  БЕЗ УЧЁТА РЕЖИМОВ:                                               |
|                                                                    |
|  Стратегия: Покупать при RSI < 30                                  |
|  Бычий рынок: Отлично работает! Покупаем просадки, цены растут     |
|  Медвежий рынок: КАТАСТРОФА! Продолжаем покупать падающий рынок    |
|                                                                    |
|  Результат: Стратегия работает иногда, иногда приводит к убыткам   |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|  С УЧЁТОМ РЕЖИМОВ:                                                |
|                                                                    |
|  Бычий режим: Используем momentum-стратегии, покупаем просадки     |
|  Медвежий режим: Используем короткие позиции, хеджируем            |
|  Боковой режим: Используем mean-reversion, продаём опционы         |
|  Высокая волатильность: Уменьшаем позиции, используем стопы        |
|                                                                    |
|  Результат: Адаптивная стратегия для всех рыночных условий         |
+------------------------------------------------------------------+
```

### Традиционные методы vs методы на основе LLM

| Аспект | Традиционные методы | Классификация на основе LLM |
|--------|---------------------|--------------------------|
| Типы данных | Только числовые (цены, волатильность) | Текст, числа, мультимодальные |
| Контекст | Технические индикаторы | Экономические нарративы, настроения новостей |
| Типы режимов | Предопределённые (2-4 состояния) | Динамические, описание новых режимов |
| Определение переходов | Статистическое (HMM) | Семантическое понимание |
| Объяснение | Отсутствует | Описания на естественном языке |
| Адаптация | Требует переобучения | Адаптация через промпты |

### Определение рыночных режимов

```
ОСНОВНЫЕ РЫНОЧНЫЕ РЕЖИМЫ:
======================================================================

1. БЫЧИЙ РЕЖИМ (BULL)
   +----------------------------------------------------------------+
   | Характеристики:                                                  |
   | - Цены растут                                                    |
   | - Позитивные настроения и оптимизм                               |
   | - Низкая волатильность, стабильный рост                          |
   | - Поведение "risk-on"                                            |
   |                                                                  |
   | Лучшие стратегии: Momentum, следование за трендом, buy-and-hold  |
   +----------------------------------------------------------------+

2. МЕДВЕЖИЙ РЕЖИМ (BEAR)
   +----------------------------------------------------------------+
   | Характеристики:                                                  |
   | - Цены снижаются                                                 |
   | - Негативные настроения и страх                                  |
   | - Обычно повышенная волатильность                                |
   | - Поведение "risk-off"                                           |
   |                                                                  |
   | Лучшие стратегии: Короткие позиции, хеджирование, cash           |
   +----------------------------------------------------------------+

3. БОКОВОЙ РЕЖИМ (SIDEWAYS)
   +----------------------------------------------------------------+
   | Характеристики:                                                  |
   | - Цены движутся в диапазоне                                      |
   | - Смешанные настроения, неопределённость                         |
   | - Низкая или умеренная волатильность                             |
   | - Mean-reverting поведение                                       |
   |                                                                  |
   | Лучшие стратегии: Range trading, продажа опционов, mean-reversion|
   +----------------------------------------------------------------+

4. РЕЖИМ ВЫСОКОЙ ВОЛАТИЛЬНОСТИ (HIGH VOLATILITY)
   +----------------------------------------------------------------+
   | Характеристики:                                                  |
   | - Большие колебания цен в обе стороны                            |
   | - Неопределённость и смешанные новости                           |
   | - Повышенный VIX (индекс страха)                                 |
   | - Частые развороты тренда                                        |
   |                                                                  |
   | Лучшие стратегии: Уменьшение позиций, торговля волатильностью    |
   +----------------------------------------------------------------+

5. КРИЗИСНЫЙ РЕЖИМ (CRISIS)
   +----------------------------------------------------------------+
   | Характеристики:                                                  |
   | - Быстрое падение цен                                            |
   | - Экстремальный страх и паника                                   |
   | - Очень высокая волатильность                                    |
   | - Проблемы с ликвидностью, рост корреляций                       |
   |                                                                  |
   | Лучшие стратегии: Сохранение капитала, хеджирование хвостовых рисков|
   +----------------------------------------------------------------+
```

## Теоретические основы

### Скрытые марковские модели как базовый метод

Традиционное определение режимов использует скрытые марковские модели (HMM) для моделирования переходов между режимами:

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Перечисление рыночных режимов."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    """Результат классификации режима."""
    regime: MarketRegime
    probability: float
    confidence: float
    explanation: str
    supporting_factors: List[str]


class HMMRegimeDetector:
    """
    Базовый детектор режимов на скрытых марковских моделях.

    Использует доходности и волатильность для классификации
    рыночных режимов статистическим методом.
    """

    def __init__(self, n_regimes: int = 4, lookback: int = 60):
        """
        Инициализация детектора HMM.

        Args:
            n_regimes: Количество скрытых состояний (режимов)
            lookback: Период ретроспективного анализа
        """
        self.n_regimes = n_regimes
        self.lookback = lookback

        # Параметры режимов (средняя доходность, волатильность)
        self.regime_params = {
            MarketRegime.BULL: {'mean': 0.001, 'vol': 0.01},
            MarketRegime.BEAR: {'mean': -0.001, 'vol': 0.02},
            MarketRegime.SIDEWAYS: {'mean': 0.0, 'vol': 0.008},
            MarketRegime.HIGH_VOLATILITY: {'mean': 0.0, 'vol': 0.03}
        }

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None
    ) -> RegimeResult:
        """
        Определение текущего рыночного режима на основе данных.

        Args:
            returns: Массив недавних доходностей
            volatility: Опционально массив волатильности

        Returns:
            RegimeResult с классификацией
        """
        if len(returns) < self.lookback:
            returns = np.pad(returns, (self.lookback - len(returns), 0), 'constant')

        recent_returns = returns[-self.lookback:]
        mean_return = np.mean(recent_returns)

        if volatility is None:
            volatility = np.std(recent_returns) * np.sqrt(252)
        else:
            volatility = np.mean(volatility[-self.lookback:])

        # Простая классификация на основе правил
        if volatility > 0.35:  # Годовая волатильность > 35%
            regime = MarketRegime.HIGH_VOLATILITY
            prob = min(0.95, volatility / 0.5)
        elif mean_return > 0.0005 and volatility < 0.20:
            regime = MarketRegime.BULL
            prob = min(0.9, mean_return / 0.002 + 0.5)
        elif mean_return < -0.0005:
            regime = MarketRegime.BEAR
            prob = min(0.9, abs(mean_return) / 0.002 + 0.5)
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.7

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=0.7,
            explanation=f"Статистическое определение: средняя доходность={mean_return:.4f}, волатильность={volatility:.2%}",
            supporting_factors=[
                f"Средняя доходность: {mean_return:.4f}",
                f"Волатильность: {volatility:.2%}",
                f"Период анализа: {self.lookback} периодов"
            ]
        )


# Пример использования
detector = HMMRegimeDetector()
returns = np.random.randn(100) * 0.02  # Тестовые доходности
result = detector.detect_regime(returns)

print(f"Режим: {result.regime.value}")
print(f"Вероятность: {result.probability:.2%}")
print(f"Объяснение: {result.explanation}")
```

### Обучение представлений с помощью LLM

LLM могут обучаться богатым представлениям, отражающим характеристики режимов:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np


class LLMRegimeEncoder(nn.Module):
    """
    Кодирование рыночных данных (текст + числа) в представления,
    учитывающие режим, с использованием LLM backbone.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        numerical_features: int = 20,
        embedding_dim: int = 256,
        num_regimes: int = 5
    ):
        super().__init__()

        self.num_regimes = num_regimes

        # Текстовый энкодер (LLM backbone)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Числовой энкодер для данных цен/объёмов
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # Энкодер временных рядов (для OHLCV последовательностей)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.ts_proj = nn.Linear(5, 64)  # OHLCV -> d_model

        # Слой слияния
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + 128 + 64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Голова классификации режимов
        self.regime_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_regimes)
        )

    def forward(
        self,
        texts: List[str],
        numerical_features: torch.Tensor,
        ohlcv: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход для классификации режимов.

        Args:
            texts: Описания рыночного контекста
            numerical_features: Технические индикаторы (batch, num_features)
            ohlcv: Опционально OHLCV данные (batch, seq_len, 5)

        Returns:
            Словарь с эмбеддингами и вероятностями режимов
        """
        batch_size = numerical_features.size(0)

        # Кодирование каждой модальности
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            text_outputs = self.text_encoder(**inputs)
            text_emb = text_outputs.last_hidden_state[:, 0]

        num_emb = self.numerical_encoder(numerical_features)

        if ohlcv is not None:
            x = self.ts_proj(ohlcv)
            ts_emb = self.ts_encoder(x).mean(dim=1)
        else:
            ts_emb = torch.zeros(batch_size, 64)

        # Слияние модальностей
        combined = torch.cat([text_emb, num_emb, ts_emb], dim=-1)
        embeddings = self.fusion(combined)

        # Классификация режима
        logits = self.regime_classifier(embeddings)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            'embeddings': embeddings,
            'logits': logits,
            'probabilities': probabilities
        }


# Пример использования
encoder = LLMRegimeEncoder()

texts = [
    "Рынки выросли сегодня на сильных отчётах о прибылях. Оптимизм инвесторов растёт.",
    "Акции обвалились на фоне опасений рецессии. ФРС сигнализирует о дальнейшем повышении ставок.",
    "Рынки торговались в узком диапазоне. Трейдеры ожидают экономических данных следующей недели."
]

numerical = torch.randn(3, 20)
ohlcv = torch.randn(3, 60, 5)

outputs = encoder(texts, numerical, ohlcv)
print(f"Вероятности режимов: {outputs['probabilities']}")
```

## Методы классификации

### Классификация на основе текста

Используем новости и социальные сети для классификации рыночных режимов:

```python
class TextRegimeClassifier:
    """
    Классификация рыночного режима по текстовым данным.

    Использует LLM эмбеддинги и анализ финансовых настроений
    для определения рыночных условий.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name

        # Ключевые слова режимов для zero-shot классификации
        self.regime_keywords = {
            MarketRegime.BULL: [
                'ралли', 'рост', 'бычий', 'оптимизм',
                'рекордный максимум', 'прорыв', 'покупка',
                'сильные отчёты', 'рост экономики'
            ],
            MarketRegime.BEAR: [
                'обвал', 'падение', 'медвежий', 'распродажа',
                'снижение', 'страх', 'коррекция',
                'рецессия', 'слабые отчёты'
            ],
            MarketRegime.SIDEWAYS: [
                'консолидация', 'диапазон', 'боковой', 'стабильный',
                'без изменений', 'смешанный', 'нейтральный',
                'ожидание', 'неопределённость'
            ],
            MarketRegime.HIGH_VOLATILITY: [
                'волатильность', 'колебания', 'турбулентность',
                'нестабильность', 'хеджирование', 'VIX',
                'риск', 'неопределённость'
            ],
            MarketRegime.CRISIS: [
                'паника', 'кризис', 'крах', 'коллапс',
                'чёрный лебедь', 'системный риск',
                'экстренный', 'flash crash'
            ]
        }

    def classify_text(self, texts: List[str]) -> List[RegimeResult]:
        """Классификация режима на основе текстового содержания."""
        results = []

        for text in texts:
            text_lower = text.lower()
            regime_scores = {}

            for regime, keywords in self.regime_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                regime_scores[regime] = score / len(keywords)

            if max(regime_scores.values()) == 0:
                best_regime = MarketRegime.SIDEWAYS
                prob = 0.5
            else:
                best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                total = sum(regime_scores.values()) + 1e-8
                prob = regime_scores[best_regime] / total

            results.append(RegimeResult(
                regime=best_regime,
                probability=prob,
                confidence=min(0.9, prob + 0.2),
                explanation="Классификация на основе текста",
                supporting_factors=[f"Текст: {text[:100]}..."]
            ))

        return results


# Пример использования
classifier = TextRegimeClassifier()

headlines = [
    "Акции устремились к новым максимумам на сильных отчётах компаний",
    "Технологический сектор показал рост на оптимизме вокруг ИИ",
    "Инвесторы настроены бычьи после паузы ФРС"
]

results = classifier.classify_text(headlines)
for result in results:
    print(f"Режим: {result.regime.value}, Вероятность: {result.probability:.1%}")
```

### Определение режимов по временным рядам

```python
class TransformerRegimeDetector(nn.Module):
    """
    Детектор режимов на базе Transformer для финансовых временных рядов.

    Использует механизмы внимания для идентификации паттернов,
    специфичных для режимов, в OHLCV данных.
    """

    def __init__(
        self,
        input_dim: int = 5,  # OHLCV
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        num_regimes: int = 5,
        seq_length: int = 60
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_regimes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1)]
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
```

### Гибридные LLM-статистические подходы

```python
class HybridRegimeClassifier:
    """
    Гибридный классификатор режимов, объединяющий:
    - Статистические методы (HMM, кластеризация волатильности)
    - Анализ текста на основе LLM
    - Экономические индикаторы на основе правил
    """

    def __init__(
        self,
        lookback: int = 60,
        text_weight: float = 0.3,
        stats_weight: float = 0.5,
        econ_weight: float = 0.2
    ):
        self.lookback = lookback
        self.weights = {
            'text': text_weight,
            'stats': stats_weight,
            'econ': econ_weight
        }
        self.hmm_detector = HMMRegimeDetector(lookback=lookback)
        self.text_classifier = TextRegimeClassifier()

    def classify(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        texts: List[str],
        economic_data: Optional[Dict[str, float]] = None
    ) -> RegimeResult:
        """Классификация режима с использованием всей доступной информации."""

        # Статистическая классификация
        stats_result = self.hmm_detector.detect_regime(returns, volatility)

        # Текстовая классификация
        text_results = self.text_classifier.classify_text(texts)
        text_regime = max(text_results, key=lambda x: x.probability).regime

        # Объединение результатов
        regime_scores = {r: 0.0 for r in MarketRegime}
        regime_scores[stats_result.regime] += self.weights['stats'] * stats_result.probability
        regime_scores[text_regime] += self.weights['text'] * 0.7

        total = sum(regime_scores.values()) + 1e-8
        regime_probs = {r: s / total for r, s in regime_scores.items()}

        best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
        prob = regime_probs[best_regime]

        return RegimeResult(
            regime=best_regime,
            probability=prob,
            confidence=min(0.95, prob + 0.1),
            explanation="Гибридная классификация: статистика + текст + экономика",
            supporting_factors=[
                f"Статистика: {stats_result.regime.value}",
                f"Текст: {text_regime.value}"
            ]
        )
```

## Практические примеры

### 01: Классификация режимов фондового рынка

См. `python/examples/01_stock_regime.py` для полной реализации.

```python
# Быстрый старт: Классификация режима фондового рынка
from python.classifier import RegimeClassifier
from python.data_loader import YahooFinanceLoader

# Загрузка рыночных данных
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="1y")

# Инициализация классификатора
classifier = RegimeClassifier(lookback_window=60)

# Обучение на исторических данных
classifier.fit(spy_data)

# Классификация текущего режима
result = classifier.classify_current(spy_data)

print(f"Текущий рыночный режим: {result.regime.value}")
print(f"Уверенность: {result.confidence:.1%}")
print(f"Объяснение: {result.explanation}")
```

### 02: Режимы криптовалютного рынка (Bybit)

См. `python/examples/02_crypto_regime.py` для полной реализации.

```python
# Классификация режимов криптовалют на Bybit
from python.data_loader import BybitDataLoader
from python.classifier import CryptoRegimeClassifier

# Инициализация загрузчика Bybit
bybit = BybitDataLoader()

# Получение данных BTC
btc_data = bybit.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30
)

# Инициализация классификатора для криптовалют
classifier = CryptoRegimeClassifier(
    volatility_threshold=0.5,  # Выше для крипты
    trend_threshold=0.02
)

# Обучение и классификация
classifier.fit(btc_data)
result = classifier.classify_current(btc_data)

print(f"\nРежим рынка BTC: {result.regime.value}")
print(f"Вероятность: {result.probability:.1%}")
print(f"Уверенность: {result.confidence:.1%}")
```

### 03: Согласованность режимов для нескольких активов

```python
# Анализ согласованности режимов нескольких активов
from python.classifier import MultiAssetRegimeClassifier
from python.data_loader import YahooFinanceLoader, BybitDataLoader

# Загрузка нескольких активов
yahoo = YahooFinanceLoader()
bybit = BybitDataLoader()

assets = {
    'SPY': yahoo.get_daily("SPY", period="6mo"),
    'QQQ': yahoo.get_daily("QQQ", period="6mo"),
    'TLT': yahoo.get_daily("TLT", period="6mo"),
    'GLD': yahoo.get_daily("GLD", period="6mo"),
    'BTC': bybit.get_historical_klines("BTCUSDT", "1d", days=180)
}

# Мультиактивный анализ режимов
classifier = MultiAssetRegimeClassifier()

for symbol, data in assets.items():
    result = classifier.classify(data)
    print(f"{symbol}: {result.regime.value} ({result.probability:.0%})")

# Проверка согласованности режимов
alignment = classifier.compute_alignment(assets)
print(f"\nОценка согласованности режимов: {alignment['score']:.2f}")
print(f"Доминирующий режим: {alignment['dominant_regime']}")
```

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительную классификацию режимов для продакшн-систем. См. директорию `rust/` для полного кода.

```rust
//! Классификация режимов LLM - Реализация на Rust
//!
//! Высокопроизводительная классификация рыночных режимов для торговых систем.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Перечисление рыночных режимов
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    Crisis,
}

/// Результат классификации режима
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeResult {
    pub regime: MarketRegime,
    pub probability: f64,
    pub confidence: f64,
    pub explanation: String,
    pub supporting_factors: Vec<String>,
}

/// Статистический классификатор режимов
pub struct StatisticalRegimeClassifier {
    lookback_window: usize,
    returns_history: Vec<f64>,
    volatility_history: Vec<f64>,
}

impl StatisticalRegimeClassifier {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            returns_history: Vec::with_capacity(lookback_window),
            volatility_history: Vec::with_capacity(lookback_window),
        }
    }

    pub fn update(&mut self, returns: f64, volatility: f64) {
        self.returns_history.push(returns);
        self.volatility_history.push(volatility);

        if self.returns_history.len() > self.lookback_window {
            self.returns_history.remove(0);
            self.volatility_history.remove(0);
        }
    }

    pub fn classify(&self) -> RegimeResult {
        let mean_return = self.compute_mean(&self.returns_history);
        let mean_vol = self.compute_mean(&self.volatility_history);
        let annualized_vol = mean_vol * (252.0_f64).sqrt();

        let (regime, prob, explanation) = if annualized_vol > 0.40 {
            (MarketRegime::Crisis, 0.85, "Экстремальная волатильность".to_string())
        } else if annualized_vol > 0.25 {
            (MarketRegime::HighVolatility, 0.75, "Повышенная волатильность".to_string())
        } else if mean_return > 0.0005 {
            (MarketRegime::Bull, 0.80, "Позитивный тренд".to_string())
        } else if mean_return < -0.0005 {
            (MarketRegime::Bear, 0.80, "Негативный тренд".to_string())
        } else {
            (MarketRegime::Sideways, 0.70, "Боковое движение".to_string())
        };

        RegimeResult {
            regime,
            probability: prob,
            confidence: prob * 0.9 + 0.1,
            explanation,
            supporting_factors: vec![
                format!("Средняя доходность: {:.4}%", mean_return * 100.0),
                format!("Волатильность: {:.1}%", annualized_vol * 100.0),
            ],
        }
    }

    fn compute_mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() { return 0.0; }
        data.iter().sum::<f64>() / data.len() as f64
    }
}
```

## Реализация на Python

Реализация на Python включает комплексные модули для исследований и разработки. См. директорию `python/` для полного кода.

**Основные модули:**

| Модуль | Описание |
|--------|----------|
| `classifier.py` | Основные алгоритмы классификации режимов |
| `data_loader.py` | Загрузчики данных Yahoo Finance и Bybit |
| `embeddings.py` | Генерация LLM эмбеддингов для текста и временных рядов |
| `signals.py` | Генерация торговых сигналов на основе режимов |
| `backtest.py` | Фреймворк для бэктестинга режимных стратегий |
| `evaluate.py` | Метрики оценки и визуализация |

## Фреймворк для бэктестинга

Тестирование торговых стратегий на основе режимов на исторических данных:

```python
from python.backtest import RegimeBacktester
from python.classifier import HybridRegimeClassifier
from python.data_loader import YahooFinanceLoader

# Загрузка исторических данных
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="5y")

# Инициализация бэктестера
backtester = RegimeBacktester(
    initial_capital=100000,
    commission=0.001
)

# Определение стратегии на основе режимов
strategy = {
    'bull': {'position': 1.5, 'stop_loss': -0.05},      # 150% лонг
    'bear': {'position': -0.5, 'stop_loss': -0.03},     # 50% шорт
    'sideways': {'position': 0.5, 'stop_loss': -0.02},  # 50% лонг
    'high_volatility': {'position': 0.3, 'stop_loss': -0.02},
    'crisis': {'position': 0.0, 'stop_loss': None}       # Всё в кэше
}

# Запуск бэктеста
results = backtester.run(
    data=spy_data,
    classifier=HybridRegimeClassifier(),
    strategy=strategy
)

print(f"Результаты стратегии:")
print(f"  Общая доходность: {results['total_return']:.2%}")
print(f"  Годовая доходность: {results['annual_return']:.2%}")
print(f"  Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
print(f"  Максимальная просадка: {results['max_drawdown']:.2%}")
```

## Лучшие практики

### Рекомендации по классификации

```
ЛУЧШИЕ ПРАКТИКИ КЛАССИФИКАЦИИ РЕЖИМОВ LLM:
======================================================================

1. ПОДГОТОВКА ДАННЫХ
   +----------------------------------------------------------------+
   | - Нормализуйте признаки перед классификацией                     |
   | - Явно обрабатывайте пропущенные данные                          |
   | - Используйте скользящие окна для избежания look-ahead bias      |
   | - Разделяйте данные для обучения по временным периодам           |
   +----------------------------------------------------------------+

2. ВЫБОР МОДЕЛИ
   +----------------------------------------------------------------+
   | - Начните с простых статистических базовых моделей (HMM)         |
   | - Добавьте текстовые сигналы для понимания контекста             |
   | - Используйте специализированные LLM (FinBERT) для финансов      |
   | - Ансамблируйте несколько методов для надёжности                 |
   +----------------------------------------------------------------+

3. ПЕРЕХОДЫ МЕЖДУ РЕЖИМАМИ
   +----------------------------------------------------------------+
   | - Добавьте гистерезис для предотвращения частых переключений     |
   | - Требуйте подтверждения (несколько сигналов) для перехода       |
   | - Отслеживайте длительность режима для тайминга стратегии        |
   | - Учитывайте вероятности переходов в управлении рисками          |
   +----------------------------------------------------------------+

4. АДАПТАЦИЯ СТРАТЕГИИ
   +----------------------------------------------------------------+
   | - Сопоставьте каждый режим с конкретными параметрами стратегии   |
   | - Корректируйте размер позиций на основе уверенности в режиме    |
   | - Используйте специфичные для режима стопы и цели                |
   | - Снижайте экспозицию в периоды неопределённых переходов         |
   +----------------------------------------------------------------+
```

### Типичные ошибки

```
ТИПИЧНЫЕ ОШИБКИ, КОТОРЫХ СЛЕДУЕТ ИЗБЕГАТЬ:
======================================================================

X Использование будущих данных для разметки режимов
  -> Всегда используйте каузальную (только прошлое) разметку

X Переобучение на исторических режимах
  -> Тестируйте на out-of-sample переходах режимов

X Игнорирование периодов перехода режимов
  -> Добавляйте неопределённость в переходных фазах

X Определение режима по одному сигналу
  -> Объединяйте несколько источников данных

X Фиксированные пороги режимов
  -> Адаптируйте пороги к рыночным условиям
```

## Ресурсы

### Научные работы

1. **Market Regime Detection with LLMs** (2024)
   - https://arxiv.org/abs/2401.10586

2. **Hidden Markov Models for Regime Detection** (Hamilton, 1989)
   - Классическая работа по моделям переключения режимов

3. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** (2019)
   - https://arxiv.org/abs/1908.10063

### Наборы данных

| Набор данных | Описание | Размер |
|--------------|----------|--------|
| Yahoo Finance | Исторические данные акций | Варьируется |
| Bybit API | Данные криптовалют | В реальном времени |
| FRED | Экономические индикаторы | Различные ряды |

### Инструменты и библиотеки

- [hmmlearn](https://github.com/hmmlearn/hmmlearn) - Скрытые марковские модели
- [PyTorch](https://pytorch.org/) - Фреймворк глубокого обучения
- [Transformers](https://huggingface.co/transformers/) - Библиотека LLM
- [Candle](https://github.com/huggingface/candle) - ML фреймворк на Rust

### Структура директории

```
77_llm_regime_classification/
+-- README.md              # Основной файл (English)
+-- README.ru.md           # Русский перевод
+-- readme.simple.md       # Упрощённое объяснение (English)
+-- readme.simple.ru.md    # Упрощённое объяснение (Russian)
+-- python/
|   +-- __init__.py
|   +-- classifier.py      # Основная классификация режимов
|   +-- embeddings.py      # Генерация LLM эмбеддингов
|   +-- data_loader.py     # Загрузчики Yahoo Finance и Bybit
|   +-- signals.py         # Генерация торговых сигналов
|   +-- backtest.py        # Фреймворк бэктестинга
|   +-- evaluate.py        # Метрики оценки
|   +-- requirements.txt   # Зависимости Python
|   +-- examples/
|       +-- 01_stock_regime.py
|       +-- 02_crypto_regime.py
|       +-- 03_multi_asset_regime.py
+-- rust/
    +-- Cargo.toml
    +-- src/
        +-- lib.rs
        +-- classifier.rs
        +-- data_loader.rs
        +-- signals.rs
        +-- backtest.rs
    +-- examples/
        +-- basic_classification.rs
        +-- bybit_monitor.rs
        +-- backtest.rs
```
