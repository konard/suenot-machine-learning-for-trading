# Глава 76: Обнаружение аномалий с помощью LLM на финансовых рынках

В этой главе рассматривается **обнаружение аномалий на основе больших языковых моделей (LLM)** для анализа финансовых данных. Мы демонстрируем, как LLM могут выявлять необычные паттерны, подозрительные транзакции, манипуляции рынком и другие аномалии как на традиционных фондовых рынках, так и на криптовалютных биржах, таких как Bybit.

<p align="center">
<img src="https://i.imgur.com/Zx8KQPL.png" width="70%">
</p>

## Содержание

1. [Введение в обнаружение аномалий с LLM](#введение-в-обнаружение-аномалий-с-llm)
    * [Почему LLM для обнаружения аномалий?](#почему-llm-для-обнаружения-аномалий)
    * [Традиционные vs LLM-подходы](#традиционные-vs-llm-подходы)
    * [Применение в финансах](#применение-в-финансах)
2. [Теоретические основы](#теоретические-основы)
    * [Типы аномалий в финансовых данных](#типы-аномалий-в-финансовых-данных)
    * [Обучение представлений с LLM](#обучение-представлений-с-llm)
    * [Zero-shot и Few-shot обнаружение](#zero-shot-и-few-shot-обнаружение)
3. [Методы обнаружения](#методы-обнаружения)
    * [Обнаружение аномалий в тексте](#обнаружение-аномалий-в-тексте)
    * [Аномалии во временных рядах](#аномалии-во-временных-рядах)
    * [Мультимодальное обнаружение](#мультимодальное-обнаружение)
4. [Практические примеры](#практические-примеры)
    * [01: Обнаружение необычных торговых паттернов](#01-обнаружение-необычных-торговых-паттернов)
    * [02: Обнаружение рыночных аномалий по новостям](#02-обнаружение-рыночных-аномалий-по-новостям)
    * [03: Обнаружение манипуляций на крипторынке (Bybit)](#03-обнаружение-манипуляций-на-крипторынке-bybit)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в обнаружение аномалий с LLM

Обнаружение аномалий критически важно на финансовых рынках для выявления мошенничества, манипуляций, необычных торговых паттернов и других нарушений. Традиционные статистические методы часто не способны охватить сложную многомерную природу финансовых аномалий. LLM предлагают мощную альтернативу благодаря глубокому пониманию паттернов и контекста.

### Почему LLM для обнаружения аномалий?

```
ПРЕИМУЩЕСТВА LLM ДЛЯ ОБНАРУЖЕНИЯ ФИНАНСОВЫХ АНОМАЛИЙ:
======================================================================

+------------------------------------------------------------------+
|  1. КОНТЕКСТНОЕ ПОНИМАНИЕ                                         |
|     Традиционно: Z-score отмечает "рост цены на 5%" как аномалию  |
|     LLM: Учитывает контекст - "5% рост после отчетности = норма"  |
|          vs "5% рост без новостей = подозрительно"                |
+------------------------------------------------------------------+
|  2. МУЛЬТИМОДАЛЬНЫЙ АНАЛИЗ                                        |
|     Традиционно: Анализ цены ИЛИ текста отдельно                  |
|     LLM: Объединяет цену + новости + настроения + поток ордеров   |
|          для комплексной оценки аномалии                          |
+------------------------------------------------------------------+
|  3. ZERO-SHOT СПОСОБНОСТЬ                                         |
|     Традиционно: Требуются размеченные данные для каждого типа    |
|     LLM: Может обнаруживать новые типы аномалий, понимая          |
|          что выглядит "нормально"                                 |
+------------------------------------------------------------------+
|  4. ГЕНЕРАЦИЯ ОБЪЯСНЕНИЙ                                          |
|     Традиционно: Отмечает аномалию числовым score                 |
|     LLM: "Этот паттерн похож на pump-and-dump: резкий рост        |
|          объема с координированной активностью в соцсетях"        |
+------------------------------------------------------------------+
```

### Традиционные vs LLM-подходы

| Аспект | Традиционные методы | LLM-обнаружение |
|--------|---------------------|-----------------|
| Типы данных | Только числовые | Текст, числа, мультимодальные |
| Данные для обучения | Большие размеченные датасеты | Few-shot или zero-shot |
| Новые аномалии | Слабое обнаружение | Сильная генерализация |
| Объяснимость | Ограничена (только scores) | Объяснения на естественном языке |
| Понимание контекста | На основе правил | Семантическое понимание |
| Адаптация | Требует переобучения | Адаптация через промпты |
| Вычисления | Легковесные | Более ресурсоемкие |

### Применение в финансах

```
СЦЕНАРИИ ИСПОЛЬЗОВАНИЯ ОБНАРУЖЕНИЯ ФИНАНСОВЫХ АНОМАЛИЙ:
======================================================================

МОНИТОРИНГ РЫНКА
+------------------------------------------------------------------+
| - Обнаружение схем pump-and-dump                                  |
| - Идентификация wash trading                                      |
| - Распознавание паттернов front-running                           |
| - Обнаружение spoofing и layering                                 |
+------------------------------------------------------------------+

УПРАВЛЕНИЕ РИСКАМИ
+------------------------------------------------------------------+
| - Раннее предупреждение о flash crash                             |
| - Обнаружение кризиса ликвидности                                |
| - Оповещения о нарушении корреляций                              |
| - Аномалии режима волатильности                                   |
+------------------------------------------------------------------+

ОБНАРУЖЕНИЕ МОШЕННИЧЕСТВА
+------------------------------------------------------------------+
| - Распознавание паттернов инсайдерской торговли                  |
| - Попытки захвата аккаунтов                                       |
| - Необычные паттерны транзакций                                  |
| - Fake news и манипуляции рынком                                  |
+------------------------------------------------------------------+

КРИПТОВАЛЮТЫ (Bybit и др.)
+------------------------------------------------------------------+
| - Отслеживание движений китов                                     |
| - Аномалии потоков на биржах                                      |
| - Обнаружение DeFi эксплойтов                                    |
| - Сигналы предупреждения о rug pull                               |
+------------------------------------------------------------------+
```

## Теоретические основы

### Типы аномалий в финансовых данных

Финансовые аномалии можно разделить на несколько типов:

```
ТАКСОНОМИЯ АНОМАЛИЙ:
======================================================================

1. ТОЧЕЧНЫЕ АНОМАЛИИ (Единичные случаи)
   +---------------------------------------------------------------+
   |  - Отдельная точка данных значительно отличается от других    |
   |  - Пример: Одна крупная сделка на неликвидном рынке           |
   |  - Обнаружение: Расстояние эмбеддинга от центроида кластера   |
   +---------------------------------------------------------------+

2. КОНТЕКСТНЫЕ АНОМАЛИИ (Зависят от контекста)
   +---------------------------------------------------------------+
   |  - Нормально в одном контексте, аномально в другом            |
   |  - Пример: Высокий объем нормален при отчетности,             |
   |            подозрителен в обычный вторник                     |
   |  - Обнаружение: Понимание контекста LLM + условный скоринг    |
   +---------------------------------------------------------------+

3. КОЛЛЕКТИВНЫЕ АНОМАЛИИ (На основе паттернов)
   +---------------------------------------------------------------+
   |  - Последовательность событий вместе указывает на аномалию    |
   |  - Пример: Серия малых сделок перед большим движением         |
   |  - Обнаружение: Моделирование последовательностей с attention |
   +---------------------------------------------------------------+

4. СЕМАНТИЧЕСКИЕ АНОМАЛИИ (На основе смысла)
   +---------------------------------------------------------------+
   |  - Аномалии в текстовом/семантическом содержимом              |
   |  - Пример: Пресс-релиз с необычными языковыми паттернами      |
   |  - Обнаружение: Семантический анализ LLM и скоринг отклонений |
   +---------------------------------------------------------------+
```

### Обучение представлений с LLM

LLM создают богатые представления, захватывающие семантический смысл:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FinancialAnomalyEncoder(nn.Module):
    """
    Кодирование финансовых данных (текст + числа) в представления,
    удобные для обнаружения аномалий, используя LLM backbone.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        numerical_features: int = 20,
        embedding_dim: int = 256
    ):
        super().__init__()

        # Текстовый энкодер (LLM backbone)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Энкодер числовых признаков
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # Объединение и проекция в пространство аномалий
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + 128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Голова для скоринга аномалий
        self.anomaly_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(
        self,
        texts: list,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Кодирование мультимодальных входов в эмбеддинги.

        Args:
            texts: Список текстовых описаний
            numerical_features: Тензор формы (batch, num_features)

        Returns:
            Эмбеддинги формы (batch, embedding_dim)
        """
        # Токенизация и кодирование текста
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_outputs = self.text_encoder(**inputs)
            # Используем представление CLS токена
            text_embeddings = text_outputs.last_hidden_state[:, 0]

        # Кодирование числовых признаков
        num_embeddings = self.numerical_encoder(numerical_features)

        # Объединение модальностей
        combined = torch.cat([text_embeddings, num_embeddings], dim=-1)
        embeddings = self.fusion(combined)

        return embeddings

    def compute_anomaly_score(
        self,
        embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Вычисление score аномалии для эмбеддингов.

        Args:
            embeddings: Запросные эмбеддинги (batch, embedding_dim)
            reference_embeddings: Эталонное нормальное распределение

        Returns:
            Scores аномалии в [0, 1]
        """
        if reference_embeddings is not None:
            # Скоринг на основе расстояния
            centroid = reference_embeddings.mean(dim=0, keepdim=True)
            distances = torch.norm(embeddings - centroid, dim=-1)
            ref_distances = torch.norm(reference_embeddings - centroid, dim=-1)

            # Нормализация по эталонному распределению
            mean_dist = ref_distances.mean()
            std_dist = ref_distances.std()
            z_scores = (distances - mean_dist) / (std_dist + 1e-8)

            # Преобразование в вероятность
            scores = torch.sigmoid(z_scores)
        else:
            # Использование обученной головы
            scores = self.anomaly_head(embeddings).squeeze(-1)

        return scores


# Пример использования
encoder = FinancialAnomalyEncoder()

# Примеры финансовых событий
texts = [
    "Apple сообщает о квартальной выручке, превышающей ожидания аналитиков",
    "Неизвестная компания видит рост объема на 500% без новостей",
    "ФРС объявляет решение по процентной ставке согласно ожиданиям"
]

numerical = torch.randn(3, 20)  # Числовые признаки

embeddings = encoder.encode(texts, numerical)
scores = encoder.compute_anomaly_score(embeddings)

print(f"Scores аномалий: {scores.tolist()}")
```

### Zero-shot и Few-shot обнаружение

LLM превосходно обнаруживают аномалии без обширных размеченных данных:

```python
class ZeroShotAnomalyDetector:
    """
    Обнаружение аномалий с использованием zero-shot возможностей LLM.

    Использует prompt engineering для использования понимания LLM
    того, что составляет "нормальное" финансовое поведение.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name

        self.system_prompt = """Вы эксперт по обнаружению финансовых аномалий.
Проанализируйте следующие рыночные данные и определите, представляют ли они
аномальное поведение. Учитывайте:
- Исторический контекст и типичные паттерны
- Рыночные условия и новости
- Статистическую вероятность
- Индикаторы потенциальной манипуляции

Ответьте:
1. ANOMALY_SCORE: Оценка от 0.0 (норма) до 1.0 (высокая аномалия)
2. ANOMALY_TYPE: Категория аномалии (если есть)
3. EXPLANATION: Краткое обоснование вашей оценки
4. CONFIDENCE: Уровень уверенности (низкий/средний/высокий)
"""

    def analyze(self, market_data: dict) -> dict:
        """
        Анализ рыночных данных на аномалии.

        Args:
            market_data: Словарь содержащий:
                - symbol: Торговый символ
                - price_change: Изменение цены %
                - volume_ratio: Объем vs средний
                - news: Последние новости
                - context: Дополнительный контекст

        Returns:
            Словарь с оценкой аномалии
        """
        prompt = self._format_prompt(market_data)

        # В продакшене вызов реального LLM API
        # response = openai.ChatCompletion.create(...)

        # Демонстрационный ответ
        response = self._mock_analysis(market_data)

        return response

    def _format_prompt(self, data: dict) -> str:
        """Форматирование данных как промпта для анализа."""
        return f"""
ЗАПРОС НА АНАЛИЗ РЫНОЧНЫХ ДАННЫХ
================================

Символ: {data.get('symbol', 'НЕИЗВЕСТНО')}
Временная метка: {data.get('timestamp', 'Н/Д')}

ЦЕНОВОЕ ДВИЖЕНИЕ:
- Текущая цена: ${data.get('price', 0):.2f}
- Изменение цены (24ч): {data.get('price_change', 0):.2f}%
- Соотношение объема (vs 20-дн. сред.): {data.get('volume_ratio', 1):.2f}x

РЫНОЧНЫЙ КОНТЕКСТ:
- Общий рынок: {data.get('market_trend', 'Н/Д')}
- Показатели сектора: {data.get('sector_trend', 'Н/Д')}
- Уровень VIX: {data.get('vix', 'Н/Д')}

ПОСЛЕДНИЕ НОВОСТИ:
{data.get('news', 'Нет последних новостей')}

ПОТОК ОРДЕРОВ:
- Дисбаланс Bid/Ask: {data.get('order_imbalance', 0):.2f}
- Количество крупных сделок: {data.get('large_trades', 0)}

Пожалуйста, проанализируйте эти данные на потенциальные аномалии.
"""

    def _mock_analysis(self, data: dict) -> dict:
        """Демонстрационный анализ."""
        volume_ratio = data.get('volume_ratio', 1)
        price_change = abs(data.get('price_change', 0))

        # Простая эвристика для демонстрации
        if volume_ratio > 5 and price_change > 10:
            return {
                'anomaly_score': 0.85,
                'anomaly_type': 'НЕОБЫЧНАЯ_АКТИВНОСТЬ',
                'explanation': 'Значительный всплеск объема с большим движением '
                              'цены без соответствующего новостного катализатора. '
                              'Паттерн предполагает потенциальную манипуляцию или '
                              'нераскрытую существенную информацию.',
                'confidence': 'высокий'
            }
        elif volume_ratio > 3:
            return {
                'anomaly_score': 0.5,
                'anomaly_type': 'ПОВЫШЕННЫЙ_ОБЪЕМ',
                'explanation': 'Объем повышен, но в пределах, которые могут '
                              'объясняться нормальной рыночной активностью.',
                'confidence': 'средний'
            }
        else:
            return {
                'anomaly_score': 0.1,
                'anomaly_type': 'НОРМА',
                'explanation': 'Активность в пределах нормальных параметров.',
                'confidence': 'высокий'
            }


# Пример использования
detector = ZeroShotAnomalyDetector()

# Нормальный случай
normal_data = {
    'symbol': 'AAPL',
    'price': 175.50,
    'price_change': 2.1,
    'volume_ratio': 1.2,
    'news': 'Apple анонсирует новое мероприятие по запуску продукта',
    'market_trend': 'Бычий',
    'vix': 15.2
}

# Подозрительный случай
suspicious_data = {
    'symbol': 'XYZ',
    'price': 3.50,
    'price_change': 45.0,
    'volume_ratio': 12.5,
    'news': 'Нет последних новостей',
    'market_trend': 'Нейтральный',
    'vix': 15.2,
    'large_trades': 50
}

print("Нормальный случай:", detector.analyze(normal_data))
print("Подозрительный случай:", detector.analyze(suspicious_data))
```

## Методы обнаружения

### Обнаружение аномалий в тексте

Обнаружение аномалий в финансовом тексте: пресс-релизы, социальные сети, новости:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.covariance import EllipticEnvelope
from typing import List, Tuple

class TextAnomalyDetector:
    """
    Обнаружение аномальных текстовых паттернов в финансовых коммуникациях.

    Использует LLM эмбеддинги для создания пространства представлений,
    где аномальные тексты идентифицируются по расстоянию от
    нормального распределения.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        contamination: float = 0.05
    ):
        """
        Инициализация детектора.

        Args:
            model_name: Предобученная модель для кодирования текста
            contamination: Ожидаемая доля аномалий
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.contamination = contamination
        self.detector = None
        self.reference_mean = None
        self.reference_std = None

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Кодирование текстов в эмбеддинги."""
        embeddings = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy().flatten())

        return np.array(embeddings)

    def fit(self, normal_texts: List[str]):
        """
        Обучение детектора на нормальных текстах.

        Args:
            normal_texts: Список известных нормальных финансовых текстов
        """
        embeddings = self._encode_texts(normal_texts)

        # Обучение детектора выбросов
        self.detector = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        self.detector.fit(embeddings)

        # Сохранение статистик для скоринга
        self.reference_mean = embeddings.mean(axis=0)
        self.reference_std = embeddings.std(axis=0) + 1e-8

    def detect(self, texts: List[str]) -> List[dict]:
        """
        Обнаружение аномалий в новых текстах.

        Args:
            texts: Тексты для анализа

        Returns:
            Список результатов обнаружения
        """
        if self.detector is None:
            raise ValueError("Детектор не обучен. Сначала вызовите fit().")

        embeddings = self._encode_texts(texts)

        # Получение предсказаний (-1 = аномалия, 1 = норма)
        predictions = self.detector.predict(embeddings)

        # Получение scores аномалии (на основе расстояния Махаланобиса)
        scores = -self.detector.score_samples(embeddings)

        # Нормализация scores в [0, 1]
        scores_normalized = 1 / (1 + np.exp(-scores))

        results = []
        for i, (text, pred, score) in enumerate(zip(texts, predictions, scores_normalized)):
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'is_anomaly': pred == -1,
                'anomaly_score': float(score),
                'z_scores': self._compute_feature_zscore(embeddings[i])
            })

        return results

    def _compute_feature_zscore(self, embedding: np.ndarray) -> float:
        """Вычисление агрегированного z-score для эмбеддинга."""
        z_scores = (embedding - self.reference_mean) / self.reference_std
        return float(np.abs(z_scores).mean())


# Пример использования
detector = TextAnomalyDetector()

# Нормальные финансовые тексты
normal_texts = [
    "Компания сообщает о квартальной выручке в $5.2 миллиарда",
    "Совет объявляет об увеличении дивидендов на 10%",
    "CEO обсуждает планы расширения на earnings call",
    "Аналитик повышает рейтинг до buy с целевой ценой $150",
    "Компания приобретает конкурента за $2 миллиарда",
    "Q3 прибыль превысила консенсус на 5 центов",
    "Руководство подтверждает прогноз на год",
    "Запуск нового продукта стимулирует сильный спрос"
]

# Обучение на нормальных текстах
detector.fit(normal_texts)

# Тестовые тексты (смесь нормальных и аномальных)
test_texts = [
    "Квартальные результаты соответствуют ожиданиям",  # Норма
    "!!! СРОЧНО - АКЦИЯ ВЫРАСТЕТ В 10 РАЗ - ПОКУПАЙ СЕЙЧАС !!!",  # Pump scheme
    "Компания подает заявку о банкротстве по Chapter 11",  # Новость (необычная, но не манипуляция)
    "инсайдер: скоро огромная сделка, успей до взлета",  # Манипуляция
    "Сильные результаты благодаря росту основного бизнеса"  # Норма
]

results = detector.detect(test_texts)
for r in results:
    print(f"Аномалия: {r['is_anomaly']}, Score: {r['anomaly_score']:.3f} - {r['text']}")
```

### Аномалии во временных рядах

Обнаружение аномалий в ценовых и объемных временных рядах с использованием LLM-подобных эмбеддингов:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class TimeSeriesAnomalyEncoder(nn.Module):
    """
    Энкодер на основе трансформера для обнаружения аномалий
    в финансовых временных рядах.

    Преобразует OHLCV данные в эмбеддинги для обнаружения
    аномалий через метрики расстояния или ошибку реконструкции.
    """

    def __init__(
        self,
        input_dim: int = 5,  # OHLCV
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        seq_length: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()

        self.seq_length = seq_length
        self.d_model = d_model

        # Проекция входа
        self.input_proj = nn.Linear(input_dim, d_model)

        # Позиционное кодирование
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_length, d_model) * 0.1
        )

        # Энкодер трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Выходные проекции
        self.embedding_proj = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Кодирование временного ряда в эмбеддинг.

        Args:
            x: Входной тензор формы (batch, seq_length, input_dim)

        Returns:
            Эмбеддинги формы (batch, d_model)
        """
        # Проекция в размерность модели
        x = self.input_proj(x)

        # Добавление позиционного кодирования
        x = x + self.pos_encoding[:, :x.size(1)]

        # Кодирование трансформером
        encoded = self.transformer(x)

        # Глобальный pooling
        embedding = encoded.mean(dim=1)
        embedding = self.embedding_proj(embedding)

        return embedding

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Реконструкция входа через автоэнкодер.

        Args:
            x: Входной тензор

        Returns:
            Реконструированный тензор
        """
        # Кодирование
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1)]
        encoded = self.transformer(x)

        # Декодирование
        reconstructed = self.reconstruction_head(encoded)

        return reconstructed

    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        reference_embeddings: Optional[torch.Tensor] = None,
        method: str = 'reconstruction'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление scores аномалии.

        Args:
            x: Входной временной ряд
            reference_embeddings: Нормальные эталонные эмбеддинги для метода расстояния
            method: 'reconstruction' или 'distance'

        Returns:
            Кортеж (anomaly_scores, embeddings)
        """
        embeddings = self.encode(x)

        if method == 'reconstruction':
            reconstructed = self.reconstruct(x)
            # Ошибка реконструкции как score аномалии
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
            scores = mse

        elif method == 'distance' and reference_embeddings is not None:
            # Расстояние до центроида нормального кластера
            centroid = reference_embeddings.mean(dim=0, keepdim=True)
            distances = torch.norm(embeddings - centroid, dim=-1)

            # Z-score нормализация
            ref_distances = torch.norm(reference_embeddings - centroid, dim=-1)
            mean_dist = ref_distances.mean()
            std_dist = ref_distances.std() + 1e-8
            scores = (distances - mean_dist) / std_dist

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return scores, embeddings


class TimeSeriesAnomalyDetector:
    """
    Полная система обнаружения аномалий для финансовых временных рядов.
    """

    def __init__(
        self,
        seq_length: int = 60,
        threshold_percentile: float = 95
    ):
        self.seq_length = seq_length
        self.threshold_percentile = threshold_percentile

        self.encoder = TimeSeriesAnomalyEncoder(seq_length=seq_length)
        self.threshold = None
        self.reference_embeddings = None

    def fit(self, normal_data: np.ndarray, epochs: int = 50):
        """
        Обучение детектора на нормальных данных.

        Args:
            normal_data: Массив формы (num_samples, seq_length, features)
            epochs: Эпохи обучения
        """
        self.encoder.train()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

        tensor_data = torch.FloatTensor(normal_data)

        for epoch in range(epochs):
            optimizer.zero_grad()

            reconstructed = self.encoder.reconstruct(tensor_data)
            loss = nn.MSELoss()(reconstructed, tensor_data)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # Вычисление эталонных эмбеддингов и порога
        self.encoder.eval()
        with torch.no_grad():
            scores, embeddings = self.encoder.compute_anomaly_score(
                tensor_data, method='reconstruction'
            )
            self.reference_embeddings = embeddings
            self.threshold = np.percentile(
                scores.numpy(),
                self.threshold_percentile
            )

        print(f"Порог установлен на {self.threshold_percentile}-м перцентиле: {self.threshold:.6f}")

    def detect(
        self,
        data: np.ndarray
    ) -> list:
        """
        Обнаружение аномалий в новых данных.

        Args:
            data: Массив формы (num_samples, seq_length, features)

        Returns:
            Список результатов обнаружения
        """
        self.encoder.eval()
        tensor_data = torch.FloatTensor(data)

        with torch.no_grad():
            scores, embeddings = self.encoder.compute_anomaly_score(
                tensor_data,
                reference_embeddings=self.reference_embeddings,
                method='reconstruction'
            )

        results = []
        for i, score in enumerate(scores.numpy()):
            results.append({
                'index': i,
                'anomaly_score': float(score),
                'is_anomaly': score > self.threshold,
                'threshold': self.threshold
            })

        return results
```

## Практические примеры

### 01: Обнаружение необычных торговых паттернов

См. `python/examples/01_unusual_trading.py` для полной реализации.

```python
# Быстрый старт: Обнаружение необычных торговых паттернов
from python.detector import TradingPatternDetector
from python.data_loader import YahooFinanceLoader

# Загрузка рыночных данных
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="1y")

# Инициализация детектора
detector = TradingPatternDetector(
    lookback_window=60,
    threshold_percentile=95
)

# Обучение на исторических данных
detector.fit(spy_data)

# Обнаружение аномалий в недавних данных
anomalies = detector.detect(spy_data[-30:])

print("Обнаруженные аномалии:")
for a in anomalies:
    if a['is_anomaly']:
        print(f"  Дата: {a['date']}, Score: {a['score']:.3f}")
        print(f"  Причина: {a['explanation']}")
```

### 02: Обнаружение рыночных аномалий по новостям

См. `python/examples/02_news_anomaly.py` для полной реализации.

```python
# Обнаружение аномалий на основе новостей
from python.detector import NewsAnomalyDetector

detector = NewsAnomalyDetector()

# Анализ заголовков новостей
headlines = [
    "Apple сообщает о сильных результатах Q4, акции растут на 3%",
    "!!СРОЧНО!! Эта акция вырастет в 100 раз завтра - инсайдерская информация",
    "ФРС сохраняет процентные ставки согласно ожиданиям",
    "CEO компании X арестован за мошенничество, торги остановлены"
]

results = detector.analyze_batch(headlines)

for headline, result in zip(headlines, results):
    print(f"\n{headline[:50]}...")
    print(f"  Score аномалии: {result['score']:.3f}")
    print(f"  Тип: {result['anomaly_type']}")
    print(f"  Действие: {result['recommended_action']}")
```

### 03: Обнаружение манипуляций на крипторынке (Bybit)

См. `python/examples/03_crypto_manipulation.py` для полной реализации.

```python
# Обнаружение манипуляций на Bybit
from python.data_loader import BybitDataLoader
from python.detector import CryptoManipulationDetector

# Инициализация загрузчика Bybit
bybit = BybitDataLoader()

# Получение недавних данных BTC
btc_data = bybit.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30
)

# Инициализация детектора манипуляций
detector = CryptoManipulationDetector(
    volume_spike_threshold=5.0,
    price_spike_threshold=3.0
)

# Обучение на данных
detector.fit(btc_data)

# Мониторинг в реальном времени (симуляция)
for i in range(-10, 0):
    window = btc_data.iloc[i-60:i]
    result = detector.analyze_window(window)

    if result['is_manipulation_suspected']:
        print(f"\n!!! ALERT в {window.index[-1]} !!!")
        print(f"  Тип манипуляции: {result['manipulation_type']}")
        print(f"  Уверенность: {result['confidence']:.1%}")
        print(f"  Индикаторы: {result['indicators']}")
```

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительное обнаружение аномалий для production сред. См. директорию `rust/` для полного кода.

```rust
//! LLM Обнаружение аномалий - Реализация на Rust
//!
//! Высокопроизводительное обнаружение аномалий для финансовых данных,
//! разработанное для production сред с низкой латентностью.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Результат обнаружения аномалии
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub timestamp: i64,
    pub symbol: String,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
    pub anomaly_type: AnomalyType,
    pub confidence: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    Normal,
    VolumeSurge,
    PriceSpike,
    PatternAnomaly,
    ManipulationSuspected,
    Unknown,
}

/// Статистический детектор аномалий с z-scores и rolling статистиками
pub struct StatisticalAnomalyDetector {
    lookback_window: usize,
    volume_threshold: f64,
    price_threshold: f64,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
}

impl StatisticalAnomalyDetector {
    pub fn new(lookback_window: usize, volume_threshold: f64, price_threshold: f64) -> Self {
        Self {
            lookback_window,
            volume_threshold,
            price_threshold,
            price_history: Vec::with_capacity(lookback_window),
            volume_history: Vec::with_capacity(lookback_window),
        }
    }

    pub fn update(&mut self, price: f64, volume: f64) {
        self.price_history.push(price);
        self.volume_history.push(volume);

        // Сохраняем только окно lookback
        if self.price_history.len() > self.lookback_window {
            self.price_history.remove(0);
            self.volume_history.remove(0);
        }
    }

    pub fn detect(&self, price: f64, volume: f64) -> AnomalyResult {
        let price_z = self.compute_z_score(price, &self.price_history);
        let volume_z = self.compute_z_score(volume, &self.volume_history);

        let mut anomaly_type = AnomalyType::Normal;
        let mut score = 0.0;

        if volume_z.abs() > self.volume_threshold {
            anomaly_type = AnomalyType::VolumeSurge;
            score = score.max(volume_z.abs() / 10.0);
        }

        if price_z.abs() > self.price_threshold {
            anomaly_type = if anomaly_type == AnomalyType::VolumeSurge {
                AnomalyType::ManipulationSuspected
            } else {
                AnomalyType::PriceSpike
            };
            score = score.max(price_z.abs() / 10.0);
        }

        score = score.min(1.0);

        AnomalyResult {
            timestamp: chrono::Utc::now().timestamp(),
            symbol: String::new(),
            anomaly_score: score,
            is_anomaly: anomaly_type != AnomalyType::Normal,
            anomaly_type,
            confidence: self.compute_confidence(price_z, volume_z),
            explanation: self.generate_explanation(price_z, volume_z, anomaly_type),
        }
    }

    fn compute_z_score(&self, value: f64, history: &[f64]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let variance: f64 = history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / history.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        (value - mean) / std
    }

    fn compute_confidence(&self, price_z: f64, volume_z: f64) -> f64 {
        let max_z = price_z.abs().max(volume_z.abs());
        1.0 - (-max_z * 0.5).exp()
    }

    fn generate_explanation(&self, price_z: f64, volume_z: f64, anomaly_type: AnomalyType) -> String {
        match anomaly_type {
            AnomalyType::Normal => "Аномалия не обнаружена".to_string(),
            AnomalyType::VolumeSurge => {
                format!("Обнаружен всплеск объема (z-score: {:.2})", volume_z)
            }
            AnomalyType::PriceSpike => {
                format!("Обнаружен скачок цены (z-score: {:.2})", price_z)
            }
            AnomalyType::ManipulationSuspected => {
                format!(
                    "Возможная манипуляция: аномальны и цена (z={:.2}) и объем (z={:.2})",
                    price_z, volume_z
                )
            }
            _ => "Неизвестный паттерн аномалии".to_string(),
        }
    }
}
```

## Реализация на Python

Реализация на Python включает полные модули для исследований и разработки. См. директорию `python/` для полного кода.

**Основные модули:**

| Модуль | Описание |
|--------|----------|
| `detector.py` | Основные алгоритмы обнаружения аномалий |
| `data_loader.py` | Загрузчики данных Yahoo Finance и Bybit |
| `embeddings.py` | Генерация эмбеддингов на основе LLM |
| `signals.py` | Генерация торговых сигналов из scores аномалий |
| `backtest.py` | Фреймворк бэктестинга |
| `evaluate.py` | Метрики оценки (precision, recall, F1 и др.) |

## Фреймворк бэктестинга

Тестирование стратегий обнаружения аномалий на исторических данных:

```python
from python.backtest import AnomalyBacktester
from python.detector import MultiModalAnomalyDetector
from python.data_loader import BybitDataLoader

# Загрузка исторических данных
bybit = BybitDataLoader()
btc_data = bybit.get_historical_klines("BTCUSDT", "1h", days=90)

# Инициализация бэктестера
backtester = AnomalyBacktester(
    initial_capital=100000,
    anomaly_threshold=0.7,
    position_size=0.1
)

# Запуск бэктеста
results = backtester.run(
    data=btc_data,
    detector=MultiModalAnomalyDetector(),
    strategy="avoid_anomalies"  # Снижение экспозиции во время аномалий
)

print(f"Общая доходность: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Максимальная просадка: {results['max_drawdown']:.2%}")
print(f"Избежано аномалий: {results['anomalies_detected']}")
```

## Лучшие практики

### Руководства по обнаружению

```
ЛУЧШИЕ ПРАКТИКИ ОБНАРУЖЕНИЯ АНОМАЛИЙ С LLM:
======================================================================

1. ПОДГОТОВКА ДАННЫХ
   +----------------------------------------------------------------+
   | - Нормализация признаков перед embedding                        |
   | - Явная обработка пропущенных данных                           |
   | - Сохранение временного порядка для временных рядов            |
   | - Разделение обучающих данных по времени (без утечки)          |
   +----------------------------------------------------------------+

2. ВЫБОР МОДЕЛИ
   +----------------------------------------------------------------+
   | - Использование domain-specific моделей (FinBERT для финансов) |
   | - Учет ограничений вычислений для real-time обнаружения        |
   | - Баланс precision vs recall в зависимости от use case         |
   | - Ансамблирование нескольких методов обнаружения               |
   +----------------------------------------------------------------+

3. НАСТРОЙКА ПОРОГОВ
   +----------------------------------------------------------------+
   | - Установка порогов на основе допустимой FP rate               |
   | - Разные пороги для разных типов аномалий                      |
   | - Адаптивные пороги для изменяющихся рыночных условий          |
   | - Регулярная рекалибровка при эволюции динамики рынка          |
   +----------------------------------------------------------------+

4. ОЦЕНКА
   +----------------------------------------------------------------+
   | - Использование time-based train/test splits                    |
   | - Отчет precision, recall и F1 при разных порогах              |
   | - Оценка на разных рыночных режимах                            |
   | - Учет операционных метрик (latency, throughput)               |
   +----------------------------------------------------------------+

5. PRODUCTION ДЕПЛОЙ
   +----------------------------------------------------------------+
   | - Реализация circuit breakers для стабильности системы          |
   | - Логирование всех обнаружений для аудита                       |
   | - Human-in-the-loop для high-confidence аномалий               |
   | - Регулярный мониторинг производительности модели              |
   +----------------------------------------------------------------+
```

## Ресурсы

### Научные статьи

1. **Are Large Language Models Anomaly Detectors?** (Chen et al., 2023)
   - https://arxiv.org/abs/2306.04069

2. **Deep Learning for Anomaly Detection: A Review** (Pang et al., 2021)
   - https://arxiv.org/abs/2007.02500

3. **Time-Series Anomaly Detection Service at Microsoft** (Ren et al., 2019)
   - https://arxiv.org/abs/1906.03821

### Датасеты

| Датасет | Описание | Размер |
|---------|----------|--------|
| KDD Cup 1999 | Данные сетевых вторжений | 4.9M примеров |
| Credit Card Fraud | Анонимизированные транзакции | 284K примеров |
| Yahoo S5 | Бенчмарк аномалий временных рядов | 367 рядов |
| NAB | Numenta Anomaly Benchmark | 58 файлов |

### Инструменты и библиотеки

- [PyOD](https://github.com/yzhao062/pyod) - Обнаружение выбросов на Python
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) - Библиотека обнаружения аномалий
- [ADTK](https://github.com/arundo/adtk) - Toolkit обнаружения аномалий
- [Candle](https://github.com/huggingface/candle) - Rust ML фреймворк

### Структура директории

```
76_llm_anomaly_detection/
+-- README.md              # Этот файл (English)
+-- README.ru.md           # Перевод на русский
+-- readme.simple.md       # Упрощенное объяснение
+-- readme.simple.ru.md    # Упрощенное объяснение (русский)
+-- python/
|   +-- __init__.py
|   +-- detector.py        # Основное обнаружение аномалий
|   +-- embeddings.py      # Генерация LLM эмбеддингов
|   +-- data_loader.py     # Загрузчики Yahoo Finance и Bybit
|   +-- signals.py         # Генерация торговых сигналов
|   +-- backtest.py        # Фреймворк бэктестинга
|   +-- evaluate.py        # Метрики оценки
|   +-- requirements.txt   # Python зависимости
|   +-- examples/
|       +-- 01_unusual_trading.py
|       +-- 02_news_anomaly.py
|       +-- 03_crypto_manipulation.py
+-- rust/
    +-- Cargo.toml
    +-- src/
        +-- lib.rs         # Корень библиотеки
        +-- detector.rs    # Обнаружение аномалий
        +-- embeddings.rs  # Генерация эмбеддингов
        +-- data_loader.rs # Загрузка данных
        +-- signals.rs     # Генерация сигналов
        +-- backtest.rs    # Бэктестинг
    +-- examples/
        +-- detect_anomalies.rs
        +-- monitor_crypto.rs
        +-- backtest.rs
```
