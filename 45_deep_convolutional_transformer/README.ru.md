# Глава 45: Deep Convolutional Transformer (DCT) для прогнозирования движения акций

В этой главе рассматривается **Deep Convolutional Transformer (DCT)** — гибридная архитектура, объединяющая сверточные нейронные сети (CNN) с механизмом multi-head attention на основе Transformer для извлечения как локальных паттернов, так и глобальных зависимостей из финансовых временных рядов.

<p align="center">
<img src="https://i.imgur.com/zKqMvBN.png" width="70%">
</p>

## Содержание

1. [Введение в DCT](#введение-в-dct)
    * [Зачем объединять CNN и Transformer?](#зачем-объединять-cnn-и-transformer)
    * [Ключевые инновации](#ключевые-инновации)
    * [Сравнение с другими моделями](#сравнение-с-другими-моделями)
2. [Архитектура DCT](#архитектура-dct)
    * [Inception сверточное представление](#inception-сверточное-представление)
    * [Multi-Head Self-Attention](#multi-head-self-attention)
    * [Разделяемые полносвязные слои](#разделяемые-полносвязные-слои)
    * [Классификационная голова](#классификационная-голова)
3. [Предобработка данных](#предобработка-данных)
    * [Инженерия признаков](#инженерия-признаков)
    * [Техники нормализации](#техники-нормализации)
    * [Окно ретроспективного анализа](#окно-ретроспективного-анализа)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Архитектура модели DCT](#02-архитектура-модели-dct)
    * [03: Пайплайн обучения](#03-пайплайн-обучения)
    * [04: Прогнозирование движения акций](#04-прогнозирование-движения-акций)
    * [05: Бэктестинг стратегии](#05-бэктестинг-стратегии)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в DCT

Deep Convolutional Transformer (DCT) решает фундаментальную задачу прогнозирования цен акций: одновременный захват **локальных временных паттернов** (краткосрочные движения цен) и **глобальных зависимостей** (долгосрочные рыночные тренды).

Традиционные подходы часто фокусируются на одном аспекте:
- **CNN** отлично извлекают локальные признаки, но плохо справляются с дальними зависимостями
- **Transformer** захватывают глобальный контекст, но могут пропустить детальные локальные паттерны

DCT объединяет сильные стороны обеих архитектур.

### Зачем объединять CNN и Transformer?

```
Задачи финансовых временных рядов:
├── Локальные паттерны (сила CNN)
│   ├── Свечные паттерны (2-5 баров)
│   ├── Краткосрочный моментум
│   └── Всплески объема
│
└── Глобальные зависимости (сила Transformer)
    ├── Направление тренда
    ├── Смена рыночного режима
    └── Сезонные паттерны
```

**Решение DCT**: Использование inception-style сверточных слоев для извлечения многомасштабных локальных признаков, затем применение Transformer attention для захвата глобальных зависимостей.

### Ключевые инновации

1. **Inception Convolutional Token Embedding**
   - Несколько параллельных сверточных ядер разного размера
   - Захват паттернов на различных временных масштабах (1 день, 3 дня, 5 дней и т.д.)
   - Аналогично inception-модулю GoogLeNet, но адаптировано для временных рядов

2. **Разделяемые полносвязные слои**
   - Уменьшение количества параметров и вычислительной сложности
   - Применение depthwise separable convolutions к полносвязным операциям
   - Улучшение обобщающей способности на ограниченных финансовых данных

3. **Multi-Head Self-Attention**
   - Стандартный Transformer attention для захвата глобальных зависимостей
   - Каждая голова может фокусироваться на разных аспектах (тренд, волатильность, моментум)
   - Интерпретируемые веса attention показывают, какие временные шаги важны

4. **Классификация движения**
   - Прогнозирование направления цены: Вверх, Вниз или Стабильно
   - Возможна также бинарная классификация (только Вверх/Вниз)
   - Пороговая маркировка для определения движения

### Сравнение с другими моделями

| Характеристика | LSTM | CNN | Transformer | TFT | DCT |
|----------------|------|-----|-------------|-----|-----|
| Локальные паттерны | ✓ | ✓✓ | ✗ | ✓ | ✓✓ |
| Глобальные зависимости | ✓ | ✗ | ✓✓ | ✓✓ | ✓✓ |
| Многомасштабные признаки | ✗ | ✗ | ✗ | ✓ | ✓✓ |
| Эффективность параметров | ✗ | ✓ | ✗ | ✗ | ✓ |
| Интерпретируемость | ✗ | ✗ | ✓ | ✓✓ | ✓ |
| Стабильность обучения | ✗ | ✓ | ✓ | ✓ | ✓ |

## Архитектура DCT

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DEEP CONVOLUTIONAL TRANSFORMER                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    ВХОД: [batch, seq_len, features]                  │ │
│  │         (Open, High, Low, Close, Volume, Технические индикаторы)     │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              INCEPTION СВЕРТОЧНОЕ ПРЕДСТАВЛЕНИЕ                      │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │ │
│  │  │ Conv1D  │  │ Conv1D  │  │ Conv1D  │  │MaxPool1D│               │ │
│  │  │  k=1    │  │  k=3    │  │  k=5    │  │  k=3    │               │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │ │
│  │       │            │            │            │                       │ │
│  │       └────────────┴──────┬─────┴────────────┘                       │ │
│  │                           │ Конкатенация                             │ │
│  │                           ▼                                          │ │
│  │                    ┌────────────┐                                    │ │
│  │                    │  Conv1D    │                                    │ │
│  │                    │  k=1       │  Уменьшение каналов                │ │
│  │                    └────────────┘                                    │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                           │
│                               ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                   ПОЗИЦИОННОЕ КОДИРОВАНИЕ                            │ │
│  │         Добавление синусоидальной позиционной информации             │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                           │
│                               ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              TRANSFORMER ENCODER (N слоев)                           │ │
│  │  ┌───────────────────────────────────────────────────────────────┐  │ │
│  │  │  Multi-Head Self-Attention                                    │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │  Q  K  V  ──→ Attention Scores ──→ Взвешенная сумма      │  │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │ │
│  │  │                           │                                   │  │ │
│  │  │                    Add & LayerNorm                            │  │ │
│  │  │                           │                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │  Разделяемая Feed-Forward сеть                          │  │  │ │
│  │  │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐          │  │  │ │
│  │  │  │  │ Depthwise│ ──→│   ReLU   │ ──→│ Pointwise│          │  │  │ │
│  │  │  │  │  Conv    │    │          │    │   Conv   │          │  │  │ │
│  │  │  │  └──────────┘    └──────────┘    └──────────┘          │  │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │ │
│  │  │                           │                                   │  │ │
│  │  │                    Add & LayerNorm                            │  │ │
│  │  └───────────────────────────┬───────────────────────────────────┘  │ │
│  │                              │ × N слоев                            │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        ГЛОБАЛЬНЫЙ ПУЛИНГ                             │ │
│  │              Усреднение по временному измерению                      │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    КЛАССИФИКАЦИОННАЯ ГОЛОВА                          │ │
│  │           Linear → Dropout → Linear → Softmax                        │ │
│  │              Выход: вероятности [Вверх, Вниз, Стабильно]             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Inception сверточное представление

Модуль inception использует параллельные свертки с разными размерами ядер для захвата паттернов на нескольких временных масштабах:

```python
class InceptionConvEmbedding(nn.Module):
    """
    Inception-style сверточное представление для временных рядов.

    Захватывает локальные паттерны на нескольких временных масштабах одновременно:
    - kernel_size=1: Точечные признаки (мгновенные изменения цены)
    - kernel_size=3: Краткосрочные паттерны (движения за 2-3 дня)
    - kernel_size=5: Среднесрочные паттерны (недельные тренды)
    - max_pool: Наиболее выраженные признаки в локальном окне
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Параллельные ветви с разными рецептивными полями
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        )

        # Уменьшение объединенных каналов
        self.reduce = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)

        # Конкатенация по измерению каналов
        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.reduce(out)
        return self.activation(out)
```

**Почему Inception?**
- Разные размеры ядер захватывают паттерны на разных временных масштабах
- MaxPooling ветвь извлекает наиболее выраженные признаки
- Свертки 1x1 эффективно уменьшают размерность
- Параллельные ветви вычисляются одновременно (эффективно на GPU)

### Multi-Head Self-Attention

Стандартный Transformer attention, адаптированный для временных рядов:

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention для временных зависимостей.

    Каждая голова attention может научиться фокусироваться на:
    - Недавних временных шагах (краткосрочный моментум)
    - Периодических паттернах (дневная/недельная сезонность)
    - Ключевых событиях (отчеты, объявления)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: [batch, seq_len, d_model]
        B, L, D = x.shape

        # Вычисление Q, K, V в одной проекции
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Масштабированное скалярное произведение attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Агрегация значений
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        if return_attention:
            return out, attn
        return out
```

### Разделяемые полносвязные слои

Вдохновлены depthwise separable convolutions, уменьшают количество параметров:

```python
class SeparableFFN(nn.Module):
    """
    Разделяемая feed-forward сеть.

    Декомпозирует стандартную FFN на:
    1. Depthwise операция: обрабатывает каждый канал независимо
    2. Pointwise операция: смешивает информацию между каналами

    Это уменьшает параметры с O(d_model * d_ff) до O(d_model + d_ff)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # Depthwise: расширение размерности с сохранением раздельных каналов
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=1, groups=d_model
        )

        # Pointwise: смешивание по расширенным каналам
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Применение depthwise conv
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.depthwise(x)
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]

        # Применение pointwise преобразования
        return self.pointwise(x)
```

### Классификационная голова

Финальный слой предсказывает направление движения акции:

```python
class MovementClassifier(nn.Module):
    """
    Классификационная голова для прогнозирования движения.

    Классы выхода:
    - Вверх: Рост цены > порога
    - Вниз: Падение цены > порога
    - Стабильно: Изменение цены в пределах порога
    """

    def __init__(self, d_model, num_classes=3, dropout=0.2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [batch, d_model] (после глобального пулинга)
        return self.classifier(x)
```

## Предобработка данных

### Инженерия признаков

Стандартные OHLCV признаки плюс технические индикаторы:

```python
def prepare_features(df):
    """
    Подготовка признаков для модели DCT.

    Args:
        df: DataFrame с колонками [open, high, low, close, volume]

    Returns:
        DataFrame с нормализованными признаками
    """
    features = pd.DataFrame()

    # Ценовые признаки (нормализованные)
    features['open'] = df['open'] / df['close']
    features['high'] = df['high'] / df['close']
    features['low'] = df['low'] / df['close']
    features['close'] = df['close'].pct_change()

    # Объем (логарифмически нормализованный)
    features['volume'] = np.log1p(df['volume'] / df['volume'].rolling(20).mean())

    # Технические индикаторы
    features['ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
    features['volatility'] = df['close'].pct_change().rolling(20).std()
    features['rsi'] = compute_rsi(df['close'], 14)
    features['macd'] = compute_macd(df['close'])

    return features.dropna()
```

### Техники нормализации

DCT использует несколько подходов к нормализации:

1. **Z-score нормализация**: Для большинства признаков
2. **Min-max масштабирование**: Для ограниченных индикаторов (RSI)
3. **Логарифмическое преобразование**: Для данных объема
4. **Нормализация отношений**: Цена относительно скользящего среднего

### Окно ретроспективного анализа

Статья использует 30-дневное окно ретроспективного анализа:

```python
def create_sequences(data, lookback=30, horizon=1):
    """
    Создание последовательностей для обучения.

    Args:
        data: Матрица признаков [n_samples, n_features]
        lookback: Количество прошлых дней для использования
        horizon: Количество дней вперед для прогноза

    Returns:
        X: [n_sequences, lookback, n_features]
        y: [n_sequences] метки движения
    """
    X, y = [], []

    for i in range(lookback, len(data) - horizon):
        X.append(data[i-lookback:i])

        # Вычисление метки движения
        future_return = (data[i + horizon, 3] - data[i, 3]) / data[i, 3]
        if future_return > 0.005:  # Порог роста
            y.append(0)  # Вверх
        elif future_return < -0.005:  # Порог падения
            y.append(1)  # Вниз
        else:
            y.append(2)  # Стабильно

    return np.array(X), np.array(y)
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import Tuple, List
import yfinance as yf

def download_stock_data(
    symbol: str,
    start_date: str = "2013-01-01",
    end_date: str = "2024-08-31"
) -> pd.DataFrame:
    """
    Загрузка данных акций с Yahoo Finance.

    Args:
        symbol: Тикер акции (например, 'AAPL', 'MSFT')
        start_date: Начальная дата
        end_date: Конечная дата

    Returns:
        DataFrame с OHLCV данными
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    # Стандартизация имен колонок
    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление технических индикаторов для модели DCT.
    """
    result = df.copy()

    # Ценовые соотношения
    result['hl_ratio'] = (df['high'] - df['low']) / df['close']
    result['oc_ratio'] = (df['close'] - df['open']) / df['open']

    # Скользящие средние
    for window in [5, 10, 20]:
        result[f'ma_{window}'] = df['close'].rolling(window).mean()
        result[f'ma_ratio_{window}'] = df['close'] / result[f'ma_{window}']

    # Волатильность
    result['volatility'] = df['close'].pct_change().rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    result['macd'] = ema12 - ema26
    result['macd_signal'] = result['macd'].ewm(span=9).mean()

    # Признаки объема
    result['volume_ma'] = df['volume'].rolling(20).mean()
    result['volume_ratio'] = df['volume'] / result['volume_ma']

    return result.dropna()
```

### 02-05: Архитектура модели, Обучение, Прогнозирование, Бэктестинг

Смотрите полные реализации в [python/](python/).

## Реализация на Rust

Смотрите [rust_dct](rust_dct/) для полной реализации на Rust с использованием данных Bybit.

```
rust_dct/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Bybit API клиент
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура DCT
│   │   ├── mod.rs
│   │   ├── inception.rs    # Inception conv представление
│   │   ├── attention.rs    # Multi-head attention
│   │   ├── encoder.rs      # Transformer encoder
│   │   └── dct.rs          # Полная модель
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка данных Bybit
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Переход в проект Rust
cd rust_dct

# Загрузка данных с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучение модели
cargo run --example train -- --epochs 100 --batch-size 32

# Запуск бэктеста
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── model.py                # Основная реализация DCT модели
├── data_loader.py          # Загрузка данных (yfinance, Bybit)
├── features.py             # Инженерия признаков
├── train.py                # Скрипт обучения
├── backtest.py             # Утилиты бэктестинга
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_architecture.py
    ├── 03_training.py
    ├── 04_prediction.py
    └── 05_backtesting.py
```

### Быстрый старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка данных
python data_loader.py --symbols AAPL,MSFT,GOOGL --source yfinance
python data_loader.py --symbols BTCUSDT,ETHUSDT --source bybit

# Обучение модели
python train.py --config configs/default.yaml

# Запуск бэктеста
python backtest.py --model checkpoints/best_model.pt
```

## Лучшие практики

### Когда использовать DCT

**Хорошие случаи использования:**
- Прогнозирование направления движения акций
- Бинарная (Вверх/Вниз) или тернарная (Вверх/Вниз/Стабильно) классификация
- Среднесрочные прогнозы (от дня до недели)
- Рынки с выраженным трендовым поведением

**Не идеально для:**
- Высокочастотной торговли (проблемы с задержкой)
- Точного прогноза цены (используйте регрессионные модели)
- Очень шумных/неликвидных рынков
- Коротких последовательностей (<10 временных шагов)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендовано | Примечания |
|----------|---------------|------------|
| `d_model` | 64-128 | Больше для сложных паттернов |
| `num_heads` | 4-8 | Должно делить d_model |
| `num_layers` | 2-4 | Больше может переобучить |
| `lookback` | 30 | Согласно статье |
| `dropout` | 0.1-0.3 | Больше для малых датасетов |
| `threshold` | 0.5% | Порог классификации движения |

### Распространенные ошибки

1. **Дисбаланс классов**: Рынок часто трендовый, создавая несбалансированные метки
   - Решение: Использовать взвешенную функцию потерь или ресэмплинг

2. **Заглядывание в будущее**: Использование будущей информации в признаках
   - Решение: Тщательная инженерия признаков и разбиение с учетом дат

3. **Переобучение**: Финансовые данные имеют низкое соотношение сигнал/шум
   - Решение: Сильная регуляризация, ранняя остановка, кросс-валидация

4. **Нестационарность**: Рыночные режимы меняются со временем
   - Решение: Скользящие окна обучения, детекция режимов

## Ресурсы

### Научные работы

- [Deep Convolutional Transformer Network for Stock Movement Prediction](https://www.mdpi.com/2079-9292/13/21/4225) - Оригинальная статья DCT (2024)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Оригинальный Transformer
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) - Архитектура Inception

### Реализации

- [PyTorch](https://pytorch.org/) - Фреймворк глубокого обучения
- [Burn](https://burn.dev/) - Rust фреймворк глубокого обучения
- [yfinance](https://github.com/ranaroussi/yfinance) - Данные Yahoo Finance

### Связанные главы

- [Глава 18: Сверточные нейронные сети](../18_convolutional_neural_nets) - Основы CNN
- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) - TFT для прогнозирования
- [Глава 41: Higher Order Transformers](../41_higher_order_transformers) - Продвинутый attention
- [Глава 43: Stockformer](../43_stockformer_multivariate) - Мультивариативное прогнозирование

---

## Уровень сложности

**Средний до Продвинутого**

Предварительные требования:
- Понимание архитектур CNN
- Transformer и механизмы attention
- Предобработка временных рядов
- PyTorch или аналогичный фреймворк глубокого обучения
