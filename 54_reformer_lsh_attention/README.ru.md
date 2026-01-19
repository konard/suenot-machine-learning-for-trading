# Глава 54: Reformer - Locality-Sensitive Hashing (LSH) Attention

Эта глава посвящена архитектуре **Reformer**, которая использует **Locality-Sensitive Hashing (LSH) Attention** для достижения сложности O(L log L) вместо стандартной O(L^2). Это делает Reformer особенно подходящим для обработки длинных финансовых временных рядов.

<p align="center">
<img src="https://i.imgur.com/reformer_arch.png" width="70%">
</p>

## Содержание

1. [Введение в Reformer](#введение-в-reformer)
    * [Почему важно эффективное внимание](#почему-важно-эффективное-внимание)
    * [Ключевые инновации](#ключевые-инновации)
    * [Сравнение с другими эффективными Transformer](#сравнение-с-другими-эффективными-transformer)
2. [Механизм LSH Attention](#механизм-lsh-attention)
    * [Объяснение Locality-Sensitive Hashing](#объяснение-locality-sensitive-hashing)
    * [Хеш-корзины и чанкинг](#хеш-корзины-и-чанкинг)
    * [Многораундовое хеширование](#многораундовое-хеширование)
3. [Обратимые слои](#обратимые-слои)
    * [Эффективность памяти](#эффективность-памяти)
    * [Детали реализации](#детали-реализации)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в Reformer

**Reformer** — это модифицированная архитектура Transformer, представленная Kitaev, Kaiser и Levskaya (ICLR 2020), которая решает проблемы памяти и вычислительных ресурсов стандартных Transformer при обработке длинных последовательностей.

### Почему важно эффективное внимание

Стандартное self-attention вычисляет оценки внимания между всеми парами токенов:

```
Стандартное внимание: O(L^2 * d)
- L = длина последовательности
- d = размерность модели

Для L = 65536 (тиковые финансовые данные):
- Память: ~17 миллиардов весов внимания
- Вычисления: непрактично для реальной торговли
```

В финансовых приложениях часто требуется обрабатывать:
- **Высокочастотные тиковые данные**: Тысячи тиков в минуту
- **Длинный исторический контекст**: Дни или недели почасовых данных
- **Несколько активов одновременно**: Прогнозы на уровне портфеля

Reformer делает это возможным, снижая сложность до **O(L log L)**.

### Ключевые инновации

1. **LSH Attention**
   - Использует locality-sensitive hashing для аппроксимации внимания
   - Обращает внимание только на похожие ключи, уменьшая вычисления
   - Сложность: O(L log L) вместо O(L^2)

2. **Обратимые остаточные слои**
   - Позволяют пересчёт во время обратного распространения
   - Уменьшают память с O(N * L * d) до O(L * d)
   - N = количество слоёв

3. **Чанкованные Feed-Forward слои**
   - Обрабатывают feed-forward слои порциями
   - Дополнительно снижают пиковое использование памяти

### Сравнение с другими эффективными Transformer

| Модель | Сложность внимания | Метод | Применение в трейдинге |
|--------|-------------------|-------|------------------------|
| Стандартный | O(L^2) | Полное внимание | Только короткие последовательности |
| Linformer | O(L) | Линейная проекция | Общее прогнозирование |
| Performer | O(L) | Случайные признаки | Быстрый инференс |
| **Reformer** | **O(L log L)** | **LSH хеширование** | **Длинные тиковые последовательности** |
| Longformer | O(L) | Локальное + глобальное | Документоподобные данные |
| BigBird | O(L) | Разреженные паттерны | Смешанные паттерны |

**Почему Reformer для трейдинга?**
- Лучше захватывает точных ближайших соседей (важно для сопоставления паттернов)
- Многораундовое хеширование обеспечивает компромисс точность/скорость
- Хорошо работает как с криптовалютными, так и с фондовыми данными

## Механизм LSH Attention

### Объяснение Locality-Sensitive Hashing

**LSH** — это техника, которая хеширует похожие элементы в одну "корзину" с высокой вероятностью. Для внимания это означает, что запросы и ключи с высокими оценками внимания, скорее всего, будут иметь одинаковый хеш.

```
СТАНДАРТНОЕ ВНИМАНИЕ:
Запрос q обращает внимание на ВСЕ ключи k1, k2, k3, ..., kL
Вычисление: softmax(Q @ K^T / sqrt(d)) @ V
Стоимость: O(L^2)

LSH ВНИМАНИЕ:
1. Хешируем запросы и ключи в корзины
2. Запрос q обращает внимание только на ключи в СВОЕЙ корзине
3. Похожие векторы -> Одна корзина (с высокой вероятностью)
Стоимость: O(L * размер_корзины) ~ O(L log L)
```

Ключевая идея: **Веса внимания часто разреженны**. Большая часть внимания сконцентрирована на нескольких ключах, поэтому остальные можно пропустить.

### Хеш-корзины и чанкинг

Процесс LSH attention:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ПРОЦЕСС LSH ATTENTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ВХОД: Последовательность [x1, x2, x3, ..., xL]                │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │         1. ПРОЕКЦИЯ ХЕШИРОВАНИЯ          │                   │
│  │    h(x) = sign(x @ R) где R ~ N(0,1)    │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         2. НАЗНАЧЕНИЕ КОРЗИН             │                   │
│  │    Корзина 0: [x1, x4, x7]              │                   │
│  │    Корзина 1: [x2, x5, x9]              │                   │
│  │    Корзина 2: [x3, x6, x8]              │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         3. СОРТИРОВКА ПО КОРЗИНАМ        │                   │
│  │    Отсортировано: [x1,x4,x7|x2,x5,x9|..]│                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         4. ЧАНКОВАННОЕ ВНИМАНИЕ          │                   │
│  │    Внимание внутри чанков + lookback     │                   │
│  │    [x1,x4,x7] обращают внимание друг     │                   │
│  │    на друга                              │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ВЫХОД: Результат внимания (обратная сортировка)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Многораундовое хеширование

Один раунд хеширования может пропустить некоторые похожие пары. Reformer использует **несколько раундов хеширования** для повышения точности:

```python
def multi_round_lsh_attention(x, n_rounds=4, n_buckets=64):
    """
    Несколько раундов LSH увеличивают вероятность
    попадания похожих векторов в одну корзину.

    Вероятность коллизии (похожие векторы):
    - 1 раунд: ~70%
    - 4 раунда: ~99%
    """
    outputs = []
    for round in range(n_rounds):
        # Разная случайная проекция для каждого раунда
        hash_vectors = hash_with_random_rotation(x, round)
        buckets = assign_to_buckets(hash_vectors, n_buckets)
        attn_output = attend_within_buckets(x, buckets)
        outputs.append(attn_output)

    # Усреднение по раундам
    return torch.mean(torch.stack(outputs), dim=0)
```

## Обратимые слои

### Эффективность памяти

Стандартный Transformer сохраняет активации для каждого слоя при прямом проходе:

```
Стандартный: Память ~ N * L * d
N = 12 слоёв, L = 65536, d = 512
Память = 12 * 65536 * 512 * 4 байта = 1.5 ГБ (только активации!)

Обратимый: Память ~ L * d
Память = 65536 * 512 * 4 байта = 128 МБ
```

### Детали реализации

Обратимые слои разделяют вход на два потока и применяют функции поочерёдно:

```python
class ReversibleBlock(nn.Module):
    """
    Y1 = X1 + Attention(X2)
    Y2 = X2 + FeedForward(Y1)

    Обратный проход (без сохранённых активаций):
    X2 = Y2 - FeedForward(Y1)
    X1 = Y1 - Attention(X2)
    """

    def __init__(self, attention, feed_forward):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward

    def forward(self, x1, x2):
        y1 = x1 + self.attention(x2)
        y2 = x2 + self.feed_forward(y1)
        return y1, y2

    def reverse(self, y1, y2):
        x2 = y2 - self.feed_forward(y1)
        x1 = y1 - self.attention(x2)
        return x1, x2
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,  # Длинная последовательность для Reformer
    horizon: int = 24,
    interval: str = '1h'
) -> Dict:
    """
    Подготовка данных длинных последовательностей для Reformer.

    Reformer отлично справляется с длинными последовательностями,
    поэтому можем использовать 4096+ временных шагов (недели почасовых данных).

    Аргументы:
        symbols: Торговые пары (например, ['BTCUSDT', 'ETHUSDT'])
        lookback: Исторические временные шаги (могут быть гораздо длиннее)
        horizon: Горизонт прогноза
        interval: Интервал данных

    Возвращает:
        Словарь с массивами X, y для обучения
    """
    all_data = []

    for symbol in symbols:
        # Загрузка данных с Bybit
        df = load_bybit_data(symbol, interval=interval)

        # Расчёт признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(100).mean()) / \
                              df['volume'].rolling(100).std()
        df['price_zscore'] = (df['close'] - df['close'].rolling(100).mean()) / \
                             df['close'].rolling(100).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['trend'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean() - 1

        all_data.append(df)

    # Стекаем все символы
    stacked = np.stack([df.values for df in all_data], axis=1)

    # Создание последовательностей
    X, y = [], []
    for i in range(lookback, len(stacked) - horizon):
        X.append(stacked[i-lookback:i])
        y.append(stacked[i+horizon-1:i+horizon, :, 0])

    return {
        'X': np.array(X),
        'y': np.array(y).squeeze(),
        'symbols': symbols
    }
```

### 02: Обучение модели

```python
# python/03_train_model.py

import torch
import torch.nn as nn
from reformer import ReformerModel, ReformerConfig

def train_reformer(
    symbols: list,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    lookback: int = 4096
):
    """
    Обучение модели Reformer на длинных финансовых последовательностях.
    """

    # Подготовка данных
    print("Загрузка данных...")
    data = prepare_long_sequence_data(symbols, lookback=lookback)

    X = torch.FloatTensor(data['X'])
    y = torch.FloatTensor(data['y'])

    # Разделение train/val
    split = int(0.8 * len(X))
    train_dataset = TensorDataset(X[:split], y[:split])
    val_dataset = TensorDataset(X[split:], y[split:])

    # Конфигурация модели
    config = ReformerConfig(
        input_features=X.shape[-1],
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        n_buckets=64,
        n_rounds=4,
        chunk_size=64,
        dropout=0.1,
        max_seq_len=lookback,
        num_tickers=len(symbols)
    )

    model = ReformerModel(config)
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")

    # Настройка обучения
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['predictions'], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output['predictions'], batch_y)
                val_loss += loss.item()

        print(f"Эпоха {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        scheduler.step()

    return model
```

### 03: Визуализация сопоставления паттернов

```python
# python/04_pattern_analysis.py

def analyze_pattern_matching(
    model,
    sequence: torch.Tensor,
    pattern_length: int = 168  # 1 неделя почасовых данных
) -> dict:
    """
    Анализ того, как Reformer сопоставляет паттерны в длинных последовательностях.

    LSH attention естественно группирует похожие паттерны вместе,
    что эффективно для поиска исторических аналогов.
    """
    model.eval()

    # Получаем хеш-корзины для последовательности
    with torch.no_grad():
        qk = model.encoder_layers[0].attention.qk_proj(sequence)
        qk = torch.nn.functional.normalize(qk, dim=-1)
        buckets = model.encoder_layers[0].attention.hash_vectors(qk, round_idx=0)

    # Находим похожие временные периоды (одинаковые назначения корзин)
    recent_start = len(sequence) - pattern_length
    matches = {}

    for t in range(recent_start, len(sequence)):
        current_bucket = buckets[:, t]
        historical_matches = (buckets[:, :recent_start] == current_bucket.unsqueeze(-1)).all(dim=0)
        match_indices = historical_matches.nonzero().squeeze().tolist()
        matches[t - recent_start] = match_indices

    return {
        'pattern_matches': matches,
        'avg_matches': np.mean([len(v) for v in matches.values()])
    }
```

## Реализация на Rust

См. [rust_reformer](rust_reformer/) для полной реализации на Rust.

```
rust_reformer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Основные экспорты
│   ├── api/                    # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs           # HTTP клиент
│   │   └── types.rs            # Типы ответов
│   ├── data/                   # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs           # Загрузка данных
│   │   ├── features.rs         # Инженерия признаков
│   │   └── dataset.rs          # Датасет для обучения
│   ├── model/                  # Архитектура Reformer
│   │   ├── mod.rs
│   │   ├── lsh_attention.rs    # Реализация LSH attention
│   │   ├── reversible.rs       # Обратимые слои
│   │   ├── embedding.rs        # Token embedding
│   │   └── reformer.rs         # Полная модель
│   └── strategy/               # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs          # Генерация сигналов
│       └── backtest.rs         # Движок бэктестинга
└── examples/
    ├── fetch_data.rs           # Загрузка данных Bybit
    ├── train.rs                # Обучение модели
    └── backtest.rs             # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust_reformer

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель
cargo run --example train -- --epochs 100 --batch-size 16 --seq-len 4096

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

### Быстрый старт (Python)

```bash
# Установить зависимости
pip install -r requirements.txt

# Загрузить данные
python data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h

# Обучить модель
python train.py --epochs 100 --batch-size 16 --seq-len 4096

# Запустить бэктест
python backtest.py --model checkpoints/best_reformer.pt
```

## Лучшие практики

### Когда использовать Reformer

**Хорошие случаи использования:**
- Длинные исторические последовательности (1000+ шагов)
- Анализ высокочастотных данных
- Сопоставление паттернов на длинных временных горизонтах
- Среды с ограниченной памятью

**Не идеально для:**
- Очень коротких последовательностей (<256 токенов)
- Когда требуется точное внимание
- Простых задач прогнозирования

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуется | Примечания |
|----------|---------------|------------|
| `n_buckets` | 64-128 | Больше корзин = больше точность |
| `n_rounds` | 4-8 | Больше раундов = лучше точность |
| `chunk_size` | 64-128 | Согласовать с размером корзины |
| `d_model` | 256-512 | Стандартные рекомендации Transformer |
| `n_layers` | 4-8 | Использовать обратимые слои для глубоких моделей |

### Частые ошибки

1. **Слишком мало раундов хеширования**: Используйте минимум 4 раунда для надёжного внимания
2. **Несоответствие размера корзин**: Держите n_buckets ~= seq_len / chunk_size
3. **Игнорирование каузальной маски**: Обязательно для авторегрессивной генерации
4. **Маленькие последовательности**: Стандартное внимание может быть быстрее для L < 512

### Соображения памяти

```
Стандартный Transformer (L=4096, d=512, N=6):
- Активации: 6 * 4096 * 512 * 4 = 48 МБ
- Внимание: 4096^2 * 6 * 4 = 384 МБ
- Всего: ~432 МБ на сэмпл

Reformer (L=4096, d=512, N=6):
- Активации: 4096 * 512 * 4 = 8 МБ (обратимые)
- Внимание: 4096 * 64 * 6 * 4 * 4 = 25 МБ (LSH)
- Всего: ~33 МБ на сэмпл

13-кратное сокращение памяти!
```

## Ресурсы

### Статьи

- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) - Оригинальная статья Reformer
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Связанное разреженное внимание
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Альтернативное эффективное внимание
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) - Внимание на случайных признаках

### Реализации

- [Reformer PyTorch (Lucidrains)](https://github.com/lucidrains/reformer-pytorch) - Популярная реализация
- [Trax Reformer (Google)](https://github.com/google/trax/tree/master/trax/models/reformer) - Оригинальная реализация
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/reformer) - Production-ready

### Связанные главы

- [Глава 51: Linformer Long Sequences](../51_linformer_long_sequences) - Линейное внимание
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) - Механизм FAVOR+
- [Глава 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) - Разреженные паттерны
- [Глава 55: FNet Fourier Transform](../55_fnet_fourier_transform) - Фурье-смешивание

---

## Уровень сложности

**Продвинутый**

Необходимые знания:
- Архитектура Transformer и механизмы внимания
- Алгоритмы хеширования и locality-sensitive hashing
- Основы прогнозирования временных рядов
- Библиотеки ML для PyTorch/Rust
