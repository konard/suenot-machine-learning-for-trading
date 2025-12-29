# Глава 33: Обнаружение Аномалий — Детекция Рыночных Режимов и Распознавание Необычных Паттернов

## Обзор

Обнаружение аномалий в трейдинге применяется для выявления необычного поведения рынка: аномальных движений цен, манипуляций, мгновенных обвалов (flash crashes), внезапных изменений рыночного режима. Это критически важно как для управления рисками, так и для поиска торговых возможностей — аномалии часто предшествуют крупным движениям.

## Торговая Стратегия

**Суть стратегии:** Многоуровневое обнаружение аномалий:
1. **Статистические аномалии:** Z-оценка, IQR-выбросы в доходностях и объёмах
2. **Паттерн-аномалии:** Отклонения от типичных паттернов (Isolation Forest, Автоэнкодер)
3. **Режимные аномалии:** Скрытые Марковские Модели для обнаружения смены режима

**Сигнал на вход:**
- **Защитный выход:** Закрытие позиций при обнаружении аномальной волатильности
- **Контр-трендовый вход:** Вход после flash crash, когда аномалия завершается
- **Смена режима:** Адаптация стратегии при изменении рыночного режима

**Управление рисками:** Автоматическое снижение позиций при повышенном показателе аномальности

## Техническая Спецификация

### Создаваемые Ноутбуки

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_statistical_anomalies.ipynb` | Z-оценка, IQR, скользящие статистики |
| 2 | `02_isolation_forest.ipynb` | Обнаружение аномалий без учителя |
| 3 | `03_autoencoder_anomaly.ipynb` | Ошибка восстановления нейросети |
| 4 | `04_variational_autoencoder.ipynb` | VAE для вероятностного обнаружения |
| 5 | `05_hidden_markov_model.ipynb` | Детекция режимов через HMM |
| 6 | `06_mahalanobis_distance.ipynb` | Многомерное обнаружение выбросов |
| 7 | `07_local_outlier_factor.ipynb` | LOF для плотностной детекции |
| 8 | `08_dbscan_clustering.ipynb` | Кластерное обнаружение аномалий |
| 9 | `09_lstm_autoencoder.ipynb` | Последовательностное обнаружение |
| 10 | `10_real_time_detection.ipynb` | Онлайн-оценка аномальности |
| 11 | `11_trading_signals.ipynb` | Преобразование аномалий в сигналы |
| 12 | `12_backtesting.ipynb` | Полный бэктест стратегии |

### Методы Обнаружения Аномалий

```
Статистические методы:
├── Z-оценка           - Простое одномерное обнаружение
├── Модифицированная   - MAD-устойчивое обнаружение
│   Z-оценка
├── IQR метод          - Выбросы на основе квартилей
├── Тест Граббса       - Тест на единичный выброс
└── Скользящие         - Адаптивные пороги
    статистики

Методы машинного обучения:
├── Isolation Forest   - Изоляция на деревьях
├── One-Class SVM      - Граничное обнаружение
├── Local Outlier      - Плотностное обнаружение
│   Factor (LOF)
├── DBSCAN             - Кластерное обнаружение
└── Elliptic Envelope  - Гауссово предположение

Методы глубокого обучения:
├── Автоэнкодер        - Ошибка восстановления
├── Вариационный       - Вероятностное обнаружение
│   автоэнкодер
├── LSTM-Автоэнкодер   - Временные паттерны
├── Transformer AE     - На основе внимания
└── GAN-методы         - Генеративная оценка

Вероятностные методы:
├── Скрытая Марковская - Детекция режимов
│   Модель (HMM)
├── Смесь Гауссиан     - Мультимодальная детекция
├── Байесовская        - С учётом неопределённости
│   детекция
└── CUSUM              - Обнаружение точек смены
```

### Инженерия Признаков для Обнаружения Аномалий

```python
def compute_anomaly_features(df):
    """
    Создание признаков, оптимизированных для обнаружения аномалий
    """
    features = {}

    # Ценовые признаки
    features['return'] = df['close'].pct_change()
    features['log_return'] = np.log(df['close']).diff()
    features['return_abs'] = features['return'].abs()

    # Признаки волатильности
    features['volatility_5'] = features['return'].rolling(5).std()
    features['volatility_20'] = features['return'].rolling(20).std()
    features['vol_ratio'] = features['volatility_5'] / features['volatility_20']

    # Объёмные аномалии
    features['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(20).mean()) /
        df['volume'].rolling(20).std()
    )
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Корреляция цены и объёма
    features['pv_correlation'] = (
        features['return'].rolling(20).corr(df['volume'].pct_change())
    )

    # Спред и диапазон
    features['range'] = (df['high'] - df['low']) / df['close']
    features['range_ratio'] = features['range'] / features['range'].rolling(20).mean()

    # Микроструктурные признаки
    features['close_position'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'])
    )

    # Моменты распределения доходностей
    features['skewness'] = features['return'].rolling(20).skew()
    features['kurtosis'] = features['return'].rolling(20).kurt()

    return pd.DataFrame(features)
```

### Обнаружение Аномалий по Z-оценке

```python
def zscore_anomaly(series, window=20, threshold=3.0):
    """
    Обнаружение аномалий с использованием скользящей Z-оценки
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    zscore = (series - rolling_mean) / rolling_std

    anomaly = np.abs(zscore) > threshold
    anomaly_score = np.abs(zscore) / threshold  # Нормализованная оценка

    return anomaly, anomaly_score, zscore

def modified_zscore_anomaly(series, threshold=3.5):
    """
    Робастная Z-оценка на основе MAD (менее чувствительна к выбросам)
    """
    median = series.median()
    mad = np.median(np.abs(series - median))

    # Масштабный коэффициент для согласованности со стандартным отклонением
    modified_zscore = 0.6745 * (series - median) / mad

    anomaly = np.abs(modified_zscore) > threshold
    return anomaly, modified_zscore
```

### Реализация Isolation Forest

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_anomaly(features, contamination=0.01):
    """
    Isolation Forest для многомерного обнаружения аномалий

    Ключевая идея: Аномалии легче изолировать,
    требуется меньше случайных разбиений
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples='auto',
        random_state=42
    )

    # Обучение и предсказание (-1 = аномалия, 1 = норма)
    labels = model.fit_predict(features)

    # Оценка аномальности (меньше = более аномально)
    scores = model.decision_function(features)

    # Конвертация в положительную оценку (больше = более аномально)
    anomaly_score = -scores

    return labels == -1, anomaly_score
```

### Автоэнкодер для Обнаружения Аномалий

```python
import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    """
    Автоэнкодер для обнаружения аномалий через ошибку восстановления
    """
    def __init__(self, input_dim, encoding_dim=8):
        super().__init__()

        # Кодировщик
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
        )

        # Декодировщик
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def anomaly_score(self, x):
        """
        Вычисление ошибки восстановления как оценки аномальности
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=1)
        return mse.numpy()

def train_autoencoder(model, normal_data, epochs=100, lr=0.001):
    """
    Обучение автоэнкодера только на нормальных данных
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(normal_data)
        loss = criterion(output, normal_data)

        loss.backward()
        optimizer.step()

    return model
```

### Скрытая Марковская Модель для Детекции Режимов

```python
from hmmlearn import hmm

def fit_regime_hmm(returns, n_regimes=3):
    """
    Обучение HMM для обнаружения рыночных режимов

    Типичные режимы:
    - Низкая волатильность (спокойный рынок)
    - Нормальная волатильность (трендовый)
    - Высокая волатильность (кризис/возможность)
    """
    # Подготовка признаков
    features = np.column_stack([
        returns,
        returns.rolling(5).std(),
        returns.rolling(20).mean()
    ]).dropna()

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100
    )

    model.fit(features)

    # Предсказание режимов
    regimes = model.predict(features)

    # Вычисление вероятностей режимов
    probs = model.predict_proba(features)

    # Аномалия: низкая вероятность нахождения в любом режиме
    max_prob = probs.max(axis=1)
    regime_anomaly = max_prob < 0.5  # Неопределённый режим = аномалия

    return regimes, probs, regime_anomaly
```

### Обнаружение Аномалий в Реальном Времени

```python
class OnlineAnomalyDetector:
    """
    Онлайн-детектор аномалий для торговли в реальном времени
    """
    def __init__(self, lookback=100, threshold=3.0):
        self.lookback = lookback
        self.threshold = threshold
        self.buffer = []

    def update(self, value):
        """
        Добавить новое наблюдение и вернуть оценку аномальности
        """
        self.buffer.append(value)

        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)

        if len(self.buffer) < self.lookback // 2:
            return 0.0, False  # Недостаточно данных

        # Вычисление статистик
        data = np.array(self.buffer)
        mean = np.mean(data[:-1])  # Исключая текущее
        std = np.std(data[:-1])

        if std == 0:
            return 0.0, False

        zscore = (value - mean) / std
        anomaly_score = abs(zscore)
        is_anomaly = anomaly_score > self.threshold

        return anomaly_score, is_anomaly

    def get_adaptive_threshold(self, base_threshold=3.0):
        """
        Адаптация порога на основе недавней волатильности
        """
        if len(self.buffer) < self.lookback:
            return base_threshold

        recent_vol = np.std(self.buffer[-20:])
        historical_vol = np.std(self.buffer)

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Увеличение порога при высокой волатильности
        return base_threshold * max(1.0, vol_ratio)
```

### Торговые Сигналы из Аномалий

```python
def anomaly_trading_signals(prices, anomaly_scores, config):
    """
    Преобразование оценок аномальности в торговые сигналы
    """
    signals = pd.DataFrame(index=prices.index)

    # Базовые сигналы
    signals['anomaly_score'] = anomaly_scores
    signals['return'] = prices.pct_change()

    # Снижение позиции при высокой аномальности
    signals['reduce_position'] = (
        anomaly_scores > config['reduce_threshold']
    )

    # Выход при экстремальной аномалии
    signals['emergency_exit'] = (
        anomaly_scores > config['exit_threshold']
    )

    # Контр-трендовый вход после разрешения аномалии
    signals['anomaly_resolved'] = (
        (anomaly_scores.shift(1) > config['entry_threshold']) &
        (anomaly_scores < config['entry_threshold'] * 0.5)
    )

    # Направление входа на основе типа предыдущей аномалии
    signals['contrarian_long'] = (
        signals['anomaly_resolved'] &
        (signals['return'].shift(1) < 0)  # Покупка после падающей аномалии
    )

    signals['contrarian_short'] = (
        signals['anomaly_resolved'] &
        (signals['return'].shift(1) > 0)  # Продажа после растущей аномалии
    )

    return signals
```

### Ансамблевое Обнаружение Аномалий

```python
class EnsembleAnomalyDetector:
    """
    Комбинирование нескольких методов обнаружения аномалий
    """
    def __init__(self):
        self.detectors = {
            'zscore': self._zscore_score,
            'isolation_forest': self._iforest_score,
            'autoencoder': self._autoencoder_score,
            'lof': self._lof_score
        }
        self.weights = {
            'zscore': 0.2,
            'isolation_forest': 0.3,
            'autoencoder': 0.3,
            'lof': 0.2
        }

    def fit(self, normal_data):
        """
        Обучение всех детекторов на нормальных данных
        """
        self.iforest = IsolationForest(contamination=0.01)
        self.iforest.fit(normal_data)

        self.autoencoder = train_autoencoder(
            AnomalyAutoencoder(normal_data.shape[1]),
            torch.FloatTensor(normal_data)
        )

        from sklearn.neighbors import LocalOutlierFactor
        self.lof = LocalOutlierFactor(novelty=True)
        self.lof.fit(normal_data)

        # Сохранение статистик для z-оценки
        self.mean = normal_data.mean(axis=0)
        self.std = normal_data.std(axis=0)

        return self

    def score(self, data):
        """
        Вычисление взвешенной ансамблевой оценки аномальности
        """
        scores = {}

        for name, scorer in self.detectors.items():
            scores[name] = scorer(data)

        # Нормализация каждой оценки к [0, 1]
        for name in scores:
            s = scores[name]
            scores[name] = (s - s.min()) / (s.max() - s.min() + 1e-8)

        # Взвешенная комбинация
        ensemble_score = sum(
            scores[name] * self.weights[name]
            for name in scores
        )

        return ensemble_score, scores
```

### Архитектурная Диаграмма

```
                 Входные Рыночные Данные
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │Статистичес- │ │ Методы      │ │ Глубокое    │
    │ кие методы  │ │ МО          │ │ Обучение    │
    ├─────────────┤ ├─────────────┤ ├─────────────┤
    │ - Z-оценка  │ │ - Isolation │ │ - Авто-     │
    │ - IQR       │ │   Forest    │ │   энкодер   │
    │ - MAD       │ │ - One-Class │ │ - LSTM-AE   │
    │ - Граббс    │ │   SVM       │ │ - VAE       │
    │             │ │ - LOF       │ │             │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                 ┌─────────────────┐
                 │ Ансамблевый     │
                 │ Слой (Голосов.) │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Оценка          │
                 │ Аномальности    │
                 │ (шкала 0-1)     │
                 └────────┬────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Управление  │ │ Торговые    │ │ Система     │
    │ Рисками     │ │ Сигналы     │ │ Оповещений  │
    │ - Размер    │ │ - Выход     │ │ - Telegram  │
    │   позиции   │ │ - Вход      │ │ - Email     │
    │ - Хеджиров. │ │ - Режим     │ │ - Dashboard │
    └─────────────┘ └─────────────┘ └─────────────┘
```

### Требования к Данным

```
Исторические OHLCV Данные:
├── Минимум: 1 год часовых данных
├── Рекомендуется: 3+ года для детекции режимов
├── Частота: от 1 минуты до дневных
└── Источник: Bybit, Binance или другие API бирж

Обязательные поля:
├── timestamp (временная метка)
├── open, high, low, close (OHLC)
├── volume (объём)
└── Опционально: количество сделок, ставка финансирования

Требования к признакам:
├── Доходности (простые и логарифмические)
├── Меры волатильности
├── Коэффициенты объёма
├── Технические индикаторы
└── Кросс-активные корреляции (опционально)
```

### Ключевые Метрики

- **Коэффициент Обнаружения:** Доля правильно обнаруженных аномалий
- **Доля Ложных Срабатываний:** Критично для торговли (ложные тревоги = лишние сделки)
- **Precision@K:** Точность для топ-K оценок аномальности
- **AUROC:** Площадь под ROC-кривой
- **Время до Обнаружения:** Задержка от начала аномалии до детекции
- **Метрики Стратегии:** Шарп, максимальная просадка с риск-менеджментом на аномалиях

### Зависимости

```python
# Базовые
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0

# Машинное обучение
scikit-learn>=1.2.0
hmmlearn>=0.3.0

# Глубокое обучение
torch>=2.0.0
pytorch-lightning>=2.0.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0

# Рыночные данные
ccxt>=4.0.0  # Для API бирж
websocket-client>=1.4.0

# Опционально
pyod>=1.0.0  # Библиотека обнаружения выбросов Python
alibi-detect>=0.11.0  # Детекция выбросов и дрифта
```

## Ожидаемые Результаты

1. **Многометодное обнаружение аномалий** с ансамблевой оценкой
2. **Детекция режимов** с помощью Скрытых Марковских Моделей
3. **Оценка аномальности в реальном времени** для live-торговли
4. **Торговые сигналы** с риск-адаптированным размером позиции
5. **Результаты бэктеста:** Улучшение риск-скорректированной доходности на 15-30% через управление рисками на основе аномалий

## Ссылки

- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (Liu, Ting, Zhou)
- [Обнаружение Аномалий на Финансовых Рынках](https://arxiv.org/abs/1911.04107)
- [Глубокое Обучение для Обнаружения Аномалий: Обзор](https://arxiv.org/abs/1901.03407)
- [Скрытые Марковские Модели в Финансах](https://www.sciencedirect.com/science/article/pii/S0378426608001987)
- [PyOD: Библиотека Обнаружения Выбросов Python](https://pyod.readthedocs.io/)

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительного обнаружения аномалий на данных криптовалют с Bybit. Смотрите директорию `rust_anomaly_crypto/`.

### Возможности:
- Получение данных с Bybit в реальном времени
- Статистическое обнаружение аномалий (Z-оценка, MAD, IQR)
- Реализация Isolation Forest
- Онлайн-обнаружение аномалий
- Модульный и расширяемый дизайн

## Уровень Сложности

⭐⭐⭐⭐☆ (Продвинутый)

Требуется понимание: Статистика, Машинное обучение, Анализ временных рядов, Управление рисками
