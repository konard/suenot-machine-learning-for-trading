# Глава 53: BigBird — Разреженное внимание для длинных последовательностей в трейдинге

Эта глава исследует **BigBird**, механизм разреженного внимания, который комбинирует случайные, оконные и глобальные паттерны внимания для обработки длинных последовательностей с линейной сложностью. BigBird позволяет трансформерам обрабатывать значительно более длинные контекстные окна, что особенно ценно для финансовых временных рядов, требующих захвата дальних зависимостей.

<p align="center">
<img src="https://i.imgur.com/JQW8k9M.png" width="70%">
</p>

## Содержание

1. [Введение в BigBird](#введение-в-bigbird)
    * [Узкое место внимания](#узкое-место-внимания)
    * [Решение BigBird](#решение-bigbird)
    * [Ключевые преимущества](#ключевые-преимущества)
2. [Архитектура BigBird](#архитектура-bigbird)
    * [Случайное внимание](#случайное-внимание)
    * [Оконное (локальное) внимание](#оконное-локальное-внимание)
    * [Глобальное внимание](#глобальное-внимание)
    * [Комбинированный разреженный паттерн](#комбинированный-разреженный-паттерн)
3. [Финансовые применения](#финансовые-применения)
    * [Дальние рыночные зависимости](#дальние-рыночные-зависимости)
    * [Обработка тиковых данных](#обработка-тиковых-данных)
    * [Многотаймфреймовый анализ](#многотаймфреймовый-анализ)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Сравнение с другими методами](#сравнение-с-другими-методами)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в BigBird

### Узкое место внимания

Стандартные трансформеры вычисляют оценки внимания между всеми парами токенов, что приводит к сложности **O(n²)**:

```
Длина последовательности: 512   → Матрица внимания: 262,144 элементов
Длина последовательности: 4096  → Матрица внимания: 16,777,216 элементов
Длина последовательности: 8192  → Матрица внимания: 67,108,864 элементов
```

Для финансовых приложений, требующих длинного исторического контекста (например, год дневных данных = 252 точки, месяц часовых данных = 720 точек, день минутных данных = 1440 точек), это квадратичное масштабирование становится непрактичным.

### Решение BigBird

BigBird вводит **паттерн разреженного внимания**, который достигает сложности **O(n)**, сохраняя при этом:
- **Универсальную аппроксимацию**: может аппроксимировать любую функцию последовательность-к-последовательности
- **Полноту по Тьюрингу**: может симулировать любую машину Тьюринга

Ключевое понимание: не всем парам токенов нужно обращать внимание друг на друга. Тщательно спроектированный разреженный паттерн захватывает как локальные, так и глобальные зависимости.

```
Стандартный Трансформер:       BigBird:
┌─────────────────┐           ┌─────────────────┐
│█████████████████│           │█ ░ █ ░ ░ █ ░ █ │  ← Глобальные токены
│█████████████████│           │░ █ █ █ ░ ░ ░ █ │
│█████████████████│           │█ █ █ █ █ ░ ░ ░ │  ← Оконное внимание
│█████████████████│           │░ █ █ █ █ █ ░ ░ │
│█████████████████│           │░ ░ █ █ █ █ █ ░ │
│█████████████████│           │█ ░ ░ █ █ █ █ █ │
│█████████████████│           │░ ░ ░ ░ █ █ █ █ │
│█████████████████│           │█ █ ░ ░ ░ █ █ █ │  ← Случайное внимание
└─────────────────┘           └─────────────────┘
  O(n²) плотное                 O(n) разреженное
```

### Ключевые преимущества

1. **В 8 раз более длинные последовательности**: обработка последовательностей до 8x длиннее на том же оборудовании
2. **Линейная сложность**: память и вычисления масштабируются линейно с длиной последовательности
3. **Теоретические гарантии**: доказанная универсальная аппроксимация и полнота по Тьюрингу
4. **Гибкость**: можно добавлять глобальные токены для важных позиций конкретной задачи

## Архитектура BigBird

Разреженное внимание BigBird комбинирует три взаимодополняющих паттерна:

### Случайное внимание

Каждый запрос обращает внимание на `r` случайно выбранных ключей, обеспечивая поток информации между далекими позициями:

```python
def random_attention_pattern(seq_len: int, num_random: int) -> torch.Tensor:
    """
    Генерация паттерна случайного внимания.

    Args:
        seq_len: Длина последовательности
        num_random: Количество случайных связей на запрос (r)

    Returns:
        Маска внимания [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Выборка случайных индексов (исключая себя)
        candidates = list(range(seq_len))
        candidates.remove(i)
        random_indices = random.sample(candidates, min(num_random, len(candidates)))
        mask[i, random_indices] = True

    return mask
```

**Интуиция**: Случайные связи создают "ярлыки" в графе внимания, обеспечивая соединение любых двух токенов через небольшое количество переходов (свойство теории графов).

### Оконное (локальное) внимание

Каждый запрос обращает внимание на свою локальную окрестность из `w` токенов:

```python
def window_attention_pattern(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Генерация паттерна скользящего окна внимания.

    Args:
        seq_len: Длина последовательности
        window_size: Размер окна внимания (w)

    Returns:
        Маска внимания [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    half_window = window_size // 2

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, start:end] = True

    return mask
```

**Интуиция**: Финансовые временные ряды имеют сильные локальные зависимости (сегодняшняя цена сильно зависит от вчерашней). Оконное внимание эффективно захватывает эти паттерны.

### Глобальное внимание

Обозначенные "глобальные" токены обращают внимание на все позиции и получают внимание от всех позиций:

```python
def global_attention_pattern(
    seq_len: int,
    global_indices: List[int]
) -> torch.Tensor:
    """
    Генерация паттерна глобального внимания.

    Args:
        seq_len: Длина последовательности
        global_indices: Индексы глобальных токенов (g)

    Returns:
        Маска внимания [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for idx in global_indices:
        mask[idx, :] = True  # Глобальный токен видит всех
        mask[:, idx] = True  # Все видят глобальный токен

    return mask
```

**Интуиция**: Глобальные токены (как [CLS] в BERT) агрегируют информацию со всей последовательности. В трейдинге они могут представлять ключевые даты (отчетность, заседания ФРС), открытие/закрытие рынка или обученные важные позиции.

### Комбинированный разреженный паттерн

BigBird комбинирует все три паттерна:

```python
class BigBirdAttentionPattern:
    """
    Паттерн разреженного внимания BigBird, комбинирующий:
    - Случайное внимание (r случайных связей на запрос)
    - Оконное внимание (w локальных соседей)
    - Глобальное внимание (g глобальных токенов)
    """

    def __init__(
        self,
        seq_len: int,
        num_random: int = 3,
        window_size: int = 3,
        num_global: int = 2,
        global_tokens: str = 'first'  # 'first', 'last', 'both', 'random'
    ):
        self.seq_len = seq_len
        self.num_random = num_random
        self.window_size = window_size
        self.num_global = num_global

        # Определение позиций глобальных токенов
        if global_tokens == 'first':
            self.global_indices = list(range(num_global))
        elif global_tokens == 'last':
            self.global_indices = list(range(seq_len - num_global, seq_len))
        elif global_tokens == 'both':
            half = num_global // 2
            self.global_indices = list(range(half)) + list(range(seq_len - half, seq_len))
        else:  # random
            self.global_indices = random.sample(range(seq_len), num_global)

    def get_attention_mask(self) -> torch.Tensor:
        """Генерация комбинированной маски внимания BigBird."""
        # Начинаем со случайного внимания
        mask = random_attention_pattern(self.seq_len, self.num_random)

        # Добавляем оконное внимание
        mask |= window_attention_pattern(self.seq_len, self.window_size)

        # Добавляем глобальное внимание
        mask |= global_attention_pattern(self.seq_len, self.global_indices)

        # Обеспечиваем диагональ (само-внимание)
        mask.fill_diagonal_(True)

        return mask
```

## Финансовые применения

### Дальние рыночные зависимости

Финансовые рынки демонстрируют зависимости на нескольких временных масштабах:

```
Краткосрочные (минуты-часы):
- Внутридневной моментум
- Дисбаланс потока ордеров
- Эффекты микроструктуры рынка

Среднесрочные (дни-недели):
- Следование тренду
- Возврат к среднему
- Эффекты отчетности

Долгосрочные (месяцы-годы):
- Бизнес-циклы
- Структурные смены режимов
- Сезонные паттерны
```

Разреженное внимание BigBird захватывает все это с линейной сложностью:

```python
# Пример: обработка года дневных данных
seq_len = 252  # Торговых дней в году

# Стандартный трансформер: 252 × 252 = 63,504 оценок внимания
# BigBird с window=5, random=3, global=2:
# На токен: 5 (окно) + 3 (случайных) + 2 (глобальных) ≈ 10 связей
# Всего: 252 × 10 = 2,520 оценок внимания (сокращение в 25 раз!)

pattern = BigBirdAttentionPattern(
    seq_len=252,
    window_size=5,      # Недельный локальный контекст
    num_random=3,       # Случайные дальние связи
    num_global=2        # Первый (начало года) и последний (самый свежий)
)
```

### Обработка тиковых данных

Для высокочастотных приложений BigBird позволяет обрабатывать длинные последовательности тиков:

```python
# Обработка 1 часа тиковых данных (около 10,000 тиков для ликвидных активов)
seq_len = 10000

# Стандартный трансформер: 10000² = 100,000,000 оценок внимания (невозможно!)
# BigBird: 10000 × 15 = 150,000 оценок внимания

config = BigBirdConfig(
    seq_len=10000,
    window_size=11,     # Локальная микроструктура (±5 тиков)
    num_random=3,       # Кросс-сессионные связи
    num_global=3,       # Ключевые временные метки (открытие, значимые события)
    d_model=128
)
```

### Многотаймфреймовый анализ

Используйте глобальные токены BigBird для маркировки важных таймфреймов:

```python
def create_multi_timeframe_globals(
    timestamps: pd.DatetimeIndex,
    mark_opens: bool = True,
    mark_closes: bool = True,
    mark_events: Optional[List[datetime]] = None
) -> List[int]:
    """
    Создание индексов глобальных токенов для многотаймфреймового анализа.

    Args:
        timestamps: Временные метки последовательности
        mark_opens: Отмечать время открытия рынка как глобальные
        mark_closes: Отмечать время закрытия рынка как глобальные
        mark_events: Пользовательские временные метки событий для маркировки

    Returns:
        Список индексов глобальных токенов
    """
    global_indices = []

    if mark_opens:
        # Найти временные метки открытия рынка
        opens = timestamps[timestamps.hour == 9 & timestamps.minute == 30]
        global_indices.extend(timestamps.get_indexer(opens))

    if mark_closes:
        # Найти временные метки закрытия рынка
        closes = timestamps[timestamps.hour == 16 & timestamps.minute == 0]
        global_indices.extend(timestamps.get_indexer(closes))

    if mark_events:
        for event in mark_events:
            idx = timestamps.get_loc(event, method='nearest')
            global_indices.append(idx)

    return sorted(set(global_indices))
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ccxt

def fetch_bybit_data(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    limit: int = 1000
) -> pd.DataFrame:
    """
    Получение OHLCV данных с Bybit.

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм свечей
        limit: Количество свечей

    Returns:
        DataFrame с OHLCV данными
    """
    exchange = ccxt.bybit()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка признаков для модели BigBird.

    Args:
        df: DataFrame с OHLCV данными

    Returns:
        DataFrame с дополнительными признаками
    """
    # Лог-доходности
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Волатильность (скользящее std доходностей)
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['volatility_50'] = df['log_return'].rolling(50).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Признаки объема
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    # Диапазон цен
    df['range'] = (df['high'] - df['low']) / df['close']

    return df.dropna()
```

### 02: Архитектура BigBird

Смотрите [python/model.py](python/model.py) для полной реализации.

### 03: Обучение модели

```python
# python/03_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import BigBirdConfig, BigBirdForTrading

def train_bigbird_model(
    symbols: list = ['BTCUSDT', 'ETHUSDT'],
    seq_len: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Обучение модели BigBird на криптовалютных данных.
    """
    # Подготовка данных
    print("Получение и подготовка данных...")
    # ... код подготовки данных ...

    # Инициализация модели
    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=X.shape[-1],
        d_model=128,
        n_heads=8,
        n_layers=4,
        window_size=7,
        num_random=3,
        num_global=2
    )

    model = BigBirdForTrading(config)
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")

    # Настройка обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['predictions'], batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.6f}")

    return model
```

### 04: Прогнозирование длинных последовательностей

```python
# python/04_long_sequence_prediction.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def predict_and_visualize(model, X: torch.Tensor):
    """
    Построение прогнозов и визуализация паттернов внимания.
    """
    model.eval()
    with torch.no_grad():
        output = model(X, return_attention=True)

    predictions = output['predictions']
    attention = output['attention_weights']

    # Визуализация внимания из последнего слоя
    if attention:
        last_layer_attn = attention['layer_3']
        avg_attn = last_layer_attn[0].mean(dim=0).cpu().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, cmap='Blues', vmax=0.1)
        plt.title('Паттерн разреженного внимания BigBird')
        plt.xlabel('Позиция ключа')
        plt.ylabel('Позиция запроса')
        plt.savefig('attention_pattern.png', dpi=150, bbox_inches='tight')
        plt.close()

    return predictions, attention
```

### 05: Бэктестинг стратегии

```python
# python/05_backtest.py

def backtest_bigbird_strategy(
    model,
    test_data: pd.DataFrame,
    seq_len: int = 256,
    initial_capital: float = 100000,
    position_size: float = 0.1,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Бэктестинг стратегии на основе прогнозов BigBird.

    Args:
        model: Обученная модель BigBird
        test_data: DataFrame с OHLCV и признаками
        seq_len: Длина входной последовательности
        initial_capital: Начальный капитал
        position_size: Доля капитала на сделку
        transaction_cost: Комиссия (0.1% = 0.001)

    Returns:
        Словарь с результатами бэктеста
    """
    # ... реализация бэктеста ...

    # Расчет метрик
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'final_capital': capital,
        'num_trades': num_trades
    }

    return {'results': results_df, 'metrics': metrics}
```

## Реализация на Rust

Смотрите [rust/](rust/) для полной реализации на Rust с использованием фреймворка `burn`.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главные экспорты библиотеки
│   ├── config.rs           # Структуры конфигурации
│   ├── attention.rs        # Разреженное внимание BigBird
│   ├── model.rs            # Полная модель BigBird
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент API Bybit
│   │   ├── features.rs     # Конструирование признаков
│   │   └── dataset.rs      # Реализация датасета
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктеста
└── examples/
    ├── fetch_data.rs       # Загрузка рыночных данных
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
cd rust

# Получение данных с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --timeframe 1h

# Обучение модели
cargo run --example train -- --epochs 100 --seq-len 256

# Запуск бэктеста
cargo run --example backtest -- --model checkpoints/best.safetensors
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── config.py               # Классы конфигурации
├── model.py                # Реализация модели BigBird
├── data.py                 # Загрузка и предобработка данных
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
cd python

# Установка зависимостей
pip install -r requirements.txt

# Запуск примеров
python examples/01_data_preparation.py
python examples/03_training.py --epochs 100
python examples/05_backtesting.py --model checkpoints/best.pt
```

## Сравнение с другими методами

| Метод | Сложность | Макс. последовательность | Глобальный контекст | Локальный контекст |
|-------|-----------|--------------------------|---------------------|-------------------|
| Стандартный Трансформер | O(n²) | ~512 | Полный | Полный |
| Linformer | O(n) | ~4096 | Аппроксимированный | Аппроксимированный |
| Performer | O(n) | ~8192 | Аппроксимированный | Ограниченный |
| Longformer | O(n) | ~4096 | Глобальные токены | Окно |
| **BigBird** | O(n) | ~8192 | Глобальные токены | Окно + Случайное |
| Reformer | O(n log n) | ~64k | На основе LSH | На основе LSH |

### Когда использовать BigBird

**Идеально для:**
- Длинных исторических последовательностей (>500 временных шагов)
- Когда важны и локальные, и глобальные паттерны
- Многодневных или многонедельных горизонтов прогнозирования
- Обработки тиковых данных

**Рассмотрите альтернативы когда:**
- Короткие последовательности (<256) - используйте стандартный трансформер
- Чисто локальные паттерны - используйте сверточные модели
- Real-time инференс с жесткими требованиями к задержке - используйте более простые модели

## Лучшие практики

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуется | Примечания |
|----------|---------------|------------|
| `seq_len` | 256-1024 | Больше для низкочастотных данных |
| `window_size` | 5-11 | Нечетное число, ~1-2% от seq_len |
| `num_random` | 2-5 | Больше для более длинных последовательностей |
| `num_global` | 2-4 | Первые и/или последние позиции |
| `d_model` | 128-256 | Масштабировать со сложностью данных |
| `n_heads` | 4-8 | Должен делить d_model |

### Распространенные ошибки

1. **Кэширование масок**: предварительно вычисляйте маски внимания для эффективности
2. **Размещение глобальных токенов**: размещайте глобальные токены в значимых позициях (открытие рынка, ключевые события)
3. **Несоответствие длины последовательности**: убедитесь, что обучение и инференс используют одинаковый seq_len
4. **Управление памятью**: для очень длинных последовательностей используйте gradient checkpointing

## Ресурсы

### Статьи

- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Оригинальная статья BigBird (NeurIPS 2020)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Связанный подход со скользящим окном
- [ETC: Encoding Long and Structured Inputs](https://arxiv.org/abs/2004.08483) - Глобально-локальное внимание

### Реализации

- [Google Research BigBird](https://github.com/google-research/bigbird) - Официальная реализация
- [Hugging Face BigBird](https://huggingface.co/docs/transformers/model_doc/big_bird) - Реализация на PyTorch

### Связанные главы

- [Глава 51: Linformer Long Sequences](../51_linformer_long_sequences) - Альтернатива с линейной сложностью
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) - Внимание на основе ядер
- [Глава 54: Reformer LSH Attention](../54_reformer_lsh_attention) - Локально-чувствительное хеширование
- [Глава 57: Longformer Financial](../57_longformer_financial) - Внимание со скользящим окном

---

## Уровень сложности

**Средний-Продвинутый**

Предварительные требования:
- Основы архитектуры трансформеров
- Механизмы внимания (само-внимание, многоголовое внимание)
- Основы PyTorch/Rust ML
- Концепции прогнозирования временных рядов
