# Глава 37: Слияние Настроений и Моментума — Улучшенный Моментум на основе Социальных Сетей

## Обзор

Price momentum (ценовой моментум) и sentiment momentum (моментум настроений) из социальных сетей могут взаимно усиливать друг друга. В этой главе мы комбинируем традиционный ценовой моментум с сигналами настроений из Twitter, Reddit и StockTwits для построения улучшенной стратегии моментума.

### Что такое Sentiment-Momentum Fusion?

Это торговая стратегия, которая объединяет:
- **Ценовой моментум** — тенденция активов продолжать движение в том же направлении
- **Sentiment моментум** — изменение настроений людей в социальных сетях
- **Сигналы дивергенции** — расхождение между ценой и настроениями (предвестник разворота)

## Торговая Стратегия

**Суть стратегии:** Покупаем активы с положительным ценовым моментумом И положительным sentiment-моментумом. Продаём при дивергенции между ценой и настроениями.

### Сигналы входа

| Сигнал | Ценовой моментум | Sentiment моментум | Действие |
|--------|------------------|-------------------|----------|
| **Strong Long** | Положительный | Положительный | Покупка с полным размером |
| **Long** | Положительный | Нейтральный | Покупка с уменьшенным размером |
| **Short** | Отрицательный | Отрицательный | Продажа |
| **Exit** | Любой | Дивергенция | Закрытие позиции |

### Преимущество (Edge)

- **Sentiment предсказывает моментум:** Изменение настроений часто предшествует изменению цены
- **Дивергенция предсказывает разворот:** Когда цена и настроения расходятся — жди разворота

## Техническая Спецификация

### Ноутбуки для создания

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_data_collection.ipynb` | Сбор данных из Twitter, Reddit, StockTwits |
| 2 | `02_text_preprocessing.ipynb` | Очистка и подготовка текста |
| 3 | `03_finbert_sentiment.ipynb` | FinBERT для финансового sentiment-анализа |
| 4 | `04_sentiment_aggregation.ipynb` | Агрегация настроений по тикерам |
| 5 | `05_sentiment_momentum.ipynb` | Расчёт sentiment-моментума |
| 6 | `06_price_momentum.ipynb` | Традиционный ценовой моментум |
| 7 | `07_fusion_features.ipynb` | Комбинация ценовых и sentiment-признаков |
| 8 | `08_divergence_detection.ipynb` | Обнаружение сигналов дивергенции |
| 9 | `09_ml_model.ipynb` | ML-модель для объединённого сигнала |
| 10 | `10_backtesting.ipynb` | Бэктест с учётом комиссий |
| 11 | `11_real_time_pipeline.ipynb` | Pipeline для real-time скоринга |

### Источники данных

```
Twitter/X:
├── Кэштеги ($BTC, $ETH, $SOL)
├── Финансовые инфлюенсеры
├── Упоминания криптопроектов
└── API: Twitter API v2 / Scrapers

Reddit:
├── r/CryptoCurrency
├── r/Bitcoin
├── r/ethereum
├── r/altcoin
└── API: PRAW (Python Reddit API Wrapper)

Telegram:
├── Крипто-каналы
├── Торговые группы
└── Новостные боты

Криптовалютные форумы:
├── Bitcointalk
├── CryptoCompare
└── TradingView Ideas
```

### Извлечение настроений с помощью FinBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTSentiment:
    """
    Класс для анализа финансовых настроений с использованием FinBERT.
    FinBERT - это модель BERT, предобученная на финансовых текстах.
    """

    def __init__(self):
        # Загружаем предобученную модель FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.labels = ['negative', 'neutral', 'positive']

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Оценка настроений для списка текстов.

        Args:
            texts: Список текстов для анализа

        Returns:
            Массив оценок в диапазоне [-1, 1]
            -1 = негативное настроение
             0 = нейтральное
            +1 = позитивное
        """
        # Токенизация текстов
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Получаем предсказания модели
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Преобразуем в непрерывную оценку: -1 (негатив) до +1 (позитив)
        scores = probs[:, 2] - probs[:, 0]  # positive - negative

        return scores.numpy()

    def predict_with_confidence(self, texts: list[str]) -> tuple:
        """
        Возвращает оценку настроений и уверенность модели.

        Returns:
            (scores, confidence): кортеж из оценок и уверенности
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        scores = probs[:, 2] - probs[:, 0]
        confidence = probs.max(dim=1).values  # Максимальная вероятность

        return scores.numpy(), confidence.numpy()
```

### Агрегация настроений

```python
class SentimentAggregator:
    """
    Агрегация настроений из нескольких источников для каждого тикера.

    Используется взвешивание:
    - По времени: свежие сообщения важнее
    - По вовлечённости: сообщения с большим engagement важнее
    """

    def __init__(self, decay_halflife_hours: float = 24):
        """
        Args:
            decay_halflife_hours: Период полураспада для временного веса.
                                  24 часа = вчерашнее сообщение весит в 2 раза меньше.
        """
        self.decay_halflife = decay_halflife_hours

    def aggregate_daily(self, ticker: str, date: pd.Timestamp,
                       messages: pd.DataFrame) -> dict:
        """
        Агрегация настроений для тикера за определённую дату.

        Args:
            ticker: Символ актива (BTC, ETH, etc.)
            date: Дата агрегации
            messages: DataFrame с колонками:
                - timestamp: время сообщения
                - sentiment: оценка настроения
                - likes, retweets, replies: метрики вовлечённости

        Returns:
            Словарь с агрегированными метриками
        """
        if len(messages) == 0:
            return {
                'sentiment': 0,
                'volume': 0,
                'confidence': 0
            }

        # Временное взвешивание (экспоненциальный распад)
        hours_ago = (date - messages['timestamp']).dt.total_seconds() / 3600
        time_weights = 0.5 ** (hours_ago / self.decay_halflife)

        # Взвешивание по вовлечённости
        engagement = (
            messages['likes'] +
            messages['retweets'] +
            messages['replies']
        )
        engagement_weights = np.log1p(engagement)  # log(1 + x) для сглаживания

        # Комбинируем веса
        weights = time_weights * engagement_weights
        weights = weights / weights.sum()  # Нормализация

        # Взвешенное среднее настроение
        weighted_sentiment = (messages['sentiment'] * weights).sum()

        # Уверенность на основе согласованности
        sentiment_std = messages['sentiment'].std()
        confidence = 1 - sentiment_std  # Выше согласие = выше уверенность

        return {
            'sentiment': weighted_sentiment,
            'volume': len(messages),
            'confidence': confidence,
            'bullish_ratio': (messages['sentiment'] > 0).mean()
        }
```

### Расчёт Sentiment-Моментума

```python
def calculate_sentiment_momentum(sentiment_series: pd.Series,
                                 windows: list = [1, 5, 20]) -> pd.DataFrame:
    """
    Расчёт sentiment-моментума за несколько периодов.

    Sentiment-моментум показывает, как изменяются настроения:
    - Растущий моментум = настроения улучшаются
    - Падающий моментум = настроения ухудшаются

    Args:
        sentiment_series: Временной ряд агрегированных настроений
        windows: Список периодов для расчёта

    Returns:
        DataFrame с различными метриками моментума
    """
    momentum = {}

    for window in windows:
        # Моментум уровня: текущее значение vs скользящее среднее
        sma = sentiment_series.rolling(window).mean()
        momentum[f'sent_mom_{window}d'] = sentiment_series - sma.shift(1)

        # Моментум изменения: ускорение настроений
        momentum[f'sent_change_{window}d'] = sentiment_series.diff(window)

        # Взвешенный по объёму моментум (если есть данные об объёме)
        if 'volume' in sentiment_series.columns:
            vw_sent = (
                sentiment_series['sentiment'] * sentiment_series['volume']
            ).rolling(window).sum()
            vw_sent /= sentiment_series['volume'].rolling(window).sum()
            momentum[f'sent_vw_mom_{window}d'] = vw_sent - vw_sent.shift(window)

    return pd.DataFrame(momentum)
```

### Слияние цены и настроений

```python
class SentimentMomentumFusion:
    """
    Комбинация ценового моментума с sentiment-моментумом.

    Логика:
    - Когда оба моментума положительные = сильный сигнал покупки
    - Когда оба моментума отрицательные = сильный сигнал продажи
    - При дивергенции = осторожность, возможен разворот
    """

    def __init__(self, price_weight: float = 0.6, sentiment_weight: float = 0.4):
        """
        Args:
            price_weight: Вес ценового моментума (по умолчанию 60%)
            sentiment_weight: Вес sentiment-моментума (по умолчанию 40%)
        """
        self.price_weight = price_weight
        self.sentiment_weight = sentiment_weight

    def calculate_combined_score(self, price_data: pd.DataFrame,
                                 sentiment_data: pd.DataFrame) -> pd.Series:
        """
        Расчёт объединённой оценки моментума.

        Args:
            price_data: DataFrame с ценовыми данными и return_20d
            sentiment_data: DataFrame с sentiment и sent_mom_5d

        Returns:
            Series с объединённой оценкой
        """
        # Нормализация ценового моментума (z-score)
        price_mom = price_data['return_20d']
        price_mom_z = (price_mom - price_mom.mean()) / price_mom.std()

        # Нормализация sentiment-моментума (z-score)
        sent_mom = sentiment_data['sent_mom_5d']
        sent_mom_z = (sent_mom - sent_mom.mean()) / sent_mom.std()

        # Объединённая оценка
        combined = (
            self.price_weight * price_mom_z +
            self.sentiment_weight * sent_mom_z
        )

        return combined

    def detect_divergence(self, price_data: pd.DataFrame,
                         sentiment_data: pd.DataFrame,
                         threshold: float = 1.5) -> dict:
        """
        Обнаружение дивергенции между ценой и настроениями.

        Дивергенция = цена и настроения движутся в разных направлениях.
        Это часто предвещает разворот тренда.

        Args:
            price_data: DataFrame с return_5d
            sentiment_data: DataFrame с sent_change_5d
            threshold: Порог для определения сильной дивергенции

        Returns:
            Словарь с сигналами дивергенции
        """
        price_direction = np.sign(price_data['return_5d'])
        sent_direction = np.sign(sentiment_data['sent_change_5d'])

        # Дивергенция: цена и настроения идут в разных направлениях
        divergence = price_direction != sent_direction

        # Сильная дивергенция: ещё и значительная величина
        price_magnitude = abs(price_data['return_5d']) > price_data['return_5d'].std()
        sent_magnitude = abs(sentiment_data['sent_change_5d']) > sentiment_data['sent_change_5d'].std()

        strong_divergence = divergence & price_magnitude & sent_magnitude

        # Тип дивергенции
        divergence_type = np.where(
            price_direction > sent_direction,
            'bearish',  # Цена растёт, настроения падают
            'bullish'   # Цена падает, настроения растут
        )

        return {
            'divergence': divergence,
            'strong_divergence': strong_divergence,
            'type': divergence_type
        }
```

### ML-модель для генерации сигналов

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

class SentimentMomentumModel:
    """
    ML-модель, объединяющая ценовые и sentiment-признаки
    для генерации торговых сигналов.
    """

    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )

        # Список признаков для модели
        self.features = [
            # Ценовые признаки
            'return_5d', 'return_20d', 'return_60d',
            'volatility_20d', 'volume_ratio',

            # Sentiment-признаки
            'sentiment_score', 'sentiment_volume',
            'sent_mom_5d', 'sent_mom_20d',
            'bullish_ratio',

            # Признаки взаимодействия
            'price_sent_correlation',  # Корреляция цена-sentiment
            'divergence_flag',          # Флаг дивергенции
            'sentiment_leads_price'     # Sentiment опережает цену
        ]

    def create_labels(self, returns: pd.Series,
                     forward_days: int = 5,
                     threshold: float = 0.02) -> np.ndarray:
        """
        Создание меток классов на основе будущих доходностей.

        Args:
            returns: Серия дневных доходностей
            forward_days: Горизонт прогноза (дней)
            threshold: Порог для определения класса (2% по умолчанию)

        Returns:
            Массив меток: 1 (покупка), 0 (нейтрально), -1 (продажа)
        """
        forward_returns = returns.shift(-forward_days)

        labels = np.where(
            forward_returns > threshold, 1,      # Покупка
            np.where(forward_returns < -threshold, -1, 0)  # Продажа / Нейтрально
        )

        return labels

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучение модели с time-series кросс-валидацией.

        Важно: используем TimeSeriesSplit чтобы не было утечки данных из будущего!
        """
        tscv = TimeSeriesSplit(n_splits=5)

        scores = []
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_pred = self.model.predict(X.iloc[val_idx])
            score = accuracy_score(y.iloc[val_idx], val_pred)
            scores.append(score)
            print(f"Валидационная точность: {score:.4f}")

        print(f"Средняя точность: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    def predict_signal(self, X: pd.DataFrame) -> np.ndarray:
        """
        Генерация торговых сигналов.

        Returns:
            Массив сигналов в диапазоне [-1, 1]
        """
        proba = self.model.predict_proba(X)
        # Вероятность положительного класса минус вероятность отрицательного
        return proba[:, 2] - proba[:, 0]
```

## Ключевые Метрики

### Качество Sentiment-анализа
- **Корреляция с будущими доходностями:** IC (Information Coefficient)
- **Точность классификации:** Accuracy, F1-score
- **Калибровка:** Brier Score

### Качество сигналов
- **Hit Rate:** Процент прибыльных сигналов
- **Average Return per Signal:** Средняя доходность на сигнал
- **Signal IC:** Корреляция сигнала с будущими доходностями

### Стратегия
- **Sharpe Ratio:** Отношение доходности к риску
- **Maximum Drawdown:** Максимальная просадка
- **Turnover:** Оборачиваемость портфеля
- **Сравнение:** vs только ценовой моментум, vs только sentiment

## Зависимости

```python
# NLP и машинное обучение
transformers>=4.30.0     # Для FinBERT
torch>=2.0.0             # Бэкенд для transformers

# API социальных сетей
tweepy>=4.14.0           # Twitter API
praw>=7.7.0              # Reddit API

# Обработка данных
pandas>=1.5.0
numpy>=1.23.0

# ML-модели
lightgbm>=4.0.0
scikit-learn>=1.3.0

# Криптовалютные данные
ccxt>=4.0.0              # Для бирж (Bybit и др.)
```

## Ожидаемые Результаты

1. **Data Pipeline** для сбора данных из Twitter, Reddit, Telegram
2. **FinBERT Sentiment Scoring** для финансовых и крипто-текстов
3. **Sentiment Aggregation** по тикерам и временным периодам
4. **Fusion Strategy** объединяющая price + sentiment
5. **Divergence Signals** для предсказания разворотов
6. **Backtest Results:** Улучшенный Sharpe Ratio по сравнению с чистым price-моментумом

## Rust-реализация

В директории `rust_sentiment_momentum/` находится модульная реализация на Rust:
- Подключение к Bybit API для получения данных о ценах
- Модуль анализа настроений
- Расчёт моментума
- Генерация торговых сигналов

Подробности см. в [rust_sentiment_momentum/README.md](rust_sentiment_momentum/README.md).

## Ссылки

### Научные работы
- [FinBERT: A Pretrained Language Model for Financial Communications](https://arxiv.org/abs/1908.10063)
- [Predicting Stock Movements with Social Media](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2756815)
- [Twitter Mood Predicts the Stock Market](https://arxiv.org/abs/1010.3003)
- [Retail Investor Sentiment and Behavior](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3776874)

### Криптовалютные исследования
- [Cryptocurrency Sentiment Analysis](https://arxiv.org/abs/2103.00549)
- [Social Media and Cryptocurrency Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3807831)

## Уровень Сложности

⭐⭐⭐⭐☆ (Продвинутый)

**Требуется понимание:**
- NLP и Transformers (FinBERT)
- API социальных сетей (Twitter, Reddit)
- Анализ настроений (Sentiment Analysis)
- Стратегии моментума (Momentum Strategies)
- Машинное обучение для финансов

**Рекомендуемый опыт:**
- Уверенное владение Python
- Опыт работы с API
- Понимание финансовых рынков
- Базовые знания NLP
