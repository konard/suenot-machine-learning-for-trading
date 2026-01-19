# Глава 51: Linformer — Self-Attention с линейной сложностью для длинных последовательностей

Эта глава посвящена **Linformer** — прорывной архитектуре трансформера, которая снижает сложность self-attention с O(n²) до O(n) с помощью низкоранговой аппроксимации матрицы. Это делает её идеальной для эффективной обработки длинных последовательностей финансовых временных рядов.

<p align="center">
<img src="https://i.imgur.com/JK8m3Qf.png" width="70%">
</p>

## Содержание

1. [Введение в Linformer](#введение-в-linformer)
    * [Проблема длинных последовательностей](#проблема-длинных-последовательностей)
    * [Низкоранговая аппроксимация матрицы](#низкоранговая-аппроксимация-матрицы)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими эффективными трансформерами](#сравнение-с-другими-эффективными-трансформерами)
2. [Архитектура Linformer](#архитектура-linformer)
    * [Стандартный Self-Attention](#стандартный-self-attention)
    * [Матрицы линейной проекции](#матрицы-линейной-проекции)
    * [Анализ вычислительной сложности](#анализ-вычислительной-сложности)
    * [Эффективность памяти](#эффективность-памяти)
3. [Математическое обоснование](#математическое-обоснование)
    * [Лемма Джонсона-Линденштрауса](#лемма-джонсона-линденштрауса)
    * [Низкоранговое свойство Self-Attention](#низкоранговое-свойство-self-attention)
    * [Границы ошибки](#границы-ошибки)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в Linformer

Linformer — это вариант трансформера, разработанный Facebook AI, который достигает **линейной сложности** O(n) как по времени, так и по памяти, в сравнении с квадратичной сложностью O(n²) стандартных трансформеров. Это достигается за счёт умной низкоранговой аппроксимации матрицы внимания.

### Проблема длинных последовательностей

Стандартные трансформеры испытывают трудности с длинными последовательностями, потому что сложность внимания масштабируется квадратично:

```
Длина последов. │ Стандартный Attention │ Использование памяти
────────────────┼───────────────────────┼─────────────────────
     512        │   262,144 операций    │   ~0.5 ГБ
    2,048       │ 4,194,304 операций    │   ~8 ГБ
    8,192       │ 67,108,864 операций   │  ~128 ГБ
   32,768       │ 1,073,741,824 операций│ ~2 ТБ (!)
```

В финансовых приложениях нам часто нужно обрабатывать:
- **Тиковые данные**: Тысячи обновлений цен в минуту
- **Многодневные паттерны**: Недели или месяцы часовых данных
- **История книги ордеров**: Глубокие последовательности для анализа микроструктуры рынка

### Низкоранговая аппроксимация матрицы

Ключевое наблюдение: **матрицы self-attention изначально являются низкоранговыми**. Большая часть информации содержится в гораздо меньшем подпространстве.

```
Стандартный Attention:
┌─────────────────────────┐
│  Полная n × n матрица   │  ← O(n²) вычислений
│  attention              │
└─────────────────────────┘

Linformer:
┌─────────────────────────┐
│  Проекция в k × n       │  ← k << n
│  Низкоранговая аппрокс. │  ← O(n × k) вычислений
└─────────────────────────┘

Где k обычно 128-256, независимо от длины последовательности n
```

### Ключевые преимущества

1. **Линейная временная сложность**
   - O(n) вместо O(n²)
   - В 20 раз быстрее для длинных последовательностей
   - Позволяет обрабатывать последовательности из 10,000+ токенов

2. **Линейное использование памяти**
   - Память масштабируется линейно с длиной последовательности
   - Можно обрабатывать в 10 раз более длинные последовательности при той же памяти
   - Критично для сред с ограниченными ресурсами

3. **Сохранённое качество модели**
   - Теоретические гарантии через лемму Джонсона-Линденштрауса
   - Эмпирические результаты соответствуют стандартным трансформерам
   - Хорошо работает для финансовых временных рядов

4. **Лёгкая интеграция**
   - Замена стандартного внимания "на лету"
   - Работает с существующими архитектурами трансформеров
   - Совместим с предобучением и дообучением

### Сравнение с другими эффективными трансформерами

| Модель | Сложность | Метод | Авторегрессивный | Переменная длина |
|--------|-----------|-------|------------------|------------------|
| Стандартный Transformer | O(n²) | Полное внимание | ✓ | ✓ |
| **Linformer** | **O(n)** | **Низкоранговая проекция** | **✗** | **Фиксированная** |
| Performer | O(n) | Случайные признаки | ✓ | ✓ |
| Longformer | O(n) | Локальное + глобальное | ✓ | ✓ |
| BigBird | O(n) | Разреженное + случайное + глобальное | ✓ | ✓ |
| Reformer | O(n log n) | LSH хеширование | ✓ | ✓ |

**Когда использовать Linformer:**
- Неавторегрессивные задачи (классификация, регрессия, кодирование)
- Сценарии с фиксированной длиной последовательности
- Когда нужна максимальная эффективность
- Анализ финансовых временных рядов (часто фиксированные окна)

## Архитектура Linformer

### Стандартный Self-Attention

Стандартный self-attention вычисляет:

```python
# Стандартный Transformer Attention
# Вход: X размерности [batch, seq_len, d_model]

Q = X @ W_Q  # [batch, n, d_k]
K = X @ W_K  # [batch, n, d_k]
V = X @ W_V  # [batch, n, d_v]

# Матрица внимания: O(n²) вычислений!
Attention = softmax(Q @ K.T / sqrt(d_k))  # [batch, n, n]
Output = Attention @ V  # [batch, n, d_v]
```

Узким местом является вычисление матрицы внимания n × n.

### Матрицы линейной проекции

Linformer вводит матрицы проекции E и F:

```python
# Linformer Attention
# E: [k, n] - проецирует ключи в k измерений
# F: [k, n] - проецирует значения в k измерений
# k << n (обычно k = 128 или 256)

Q = X @ W_Q         # [batch, n, d_k]
K_proj = E @ K      # [batch, k, d_k] - сжато!
V_proj = F @ V      # [batch, k, d_v] - сжато!

# Матрица внимания: O(n × k) вычислений
Attention = softmax(Q @ K_proj.T / sqrt(d_k))  # [batch, n, k]
Output = Attention @ V_proj  # [batch, n, d_v]
```

```
┌──────────────────────────────────────────────────────────────────┐
│                         LINFORMER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Вход X: [batch, n, d_model]                                     │
│        │                                                           │
│        ▼                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                          │
│   │   Q     │  │   K     │  │   V     │                          │
│   │ [n,d_k] │  │ [n,d_k] │  │ [n,d_v] │                          │
│   └────┬────┘  └────┬────┘  └────┬────┘                          │
│        │            │            │                                 │
│        │      ┌─────┴─────┐  ┌───┴─────┐                          │
│        │      │ E @ K     │  │ F @ V   │  ← Линейные проекции     │
│        │      │ [k, d_k]  │  │ [k,d_v] │    k << n                │
│        │      └─────┬─────┘  └───┬─────┘                          │
│        │            │            │                                 │
│        ▼            ▼            ▼                                 │
│   ┌────────────────────────────────────────┐                      │
│   │  Attention = softmax(Q @ K_proj.T)     │                      │
│   │         [n, k] вместо [n, n]!          │                      │
│   └────────────────────────────────────────┘                      │
│        │                                                           │
│        ▼                                                           │
│   Выход: Attention @ V_proj → [batch, n, d_v]                     │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Анализ вычислительной сложности

```
Стандартный Self-Attention:
- Q @ K.T: O(n × n × d_k) = O(n² × d_k)
- Attention @ V: O(n × n × d_v) = O(n² × d_v)
- Итого: O(n² × d)

Linformer:
- E @ K: O(k × n × d_k) = O(n × k × d_k)
- F @ V: O(k × n × d_v) = O(n × k × d_v)
- Q @ K_proj.T: O(n × k × d_k) = O(n × k × d_k)
- Attention @ V_proj: O(n × k × d_v) = O(n × k × d_v)
- Итого: O(n × k × d)

Когда k фиксировано (например, 128), сложность становится O(n)!
```

### Эффективность памяти

```python
# Сравнение памяти для batch_size=32, d_model=512

sequence_lengths = [512, 1024, 2048, 4096, 8192]

for n in sequence_lengths:
    # Стандартный Transformer
    standard_memory = n * n * 4  # float32, только матрица внимания

    # Linformer (k=128)
    k = 128
    linformer_memory = n * k * 4

    savings = (1 - linformer_memory / standard_memory) * 100
    print(f"n={n:5d}: Стандарт={standard_memory/1e6:.1f}МБ, "
          f"Linformer={linformer_memory/1e6:.1f}МБ, "
          f"Экономия={savings:.1f}%")

# Вывод:
# n=  512: Стандарт=1.0МБ,  Linformer=0.3МБ,  Экономия=75.0%
# n= 1024: Стандарт=4.2МБ,  Linformer=0.5МБ,  Экономия=87.5%
# n= 2048: Стандарт=16.8МБ, Linformer=1.0МБ,  Экономия=93.8%
# n= 4096: Стандарт=67.1МБ, Linformer=2.1МБ,  Экономия=96.9%
# n= 8192: Стандарт=268.4МБ, Linformer=4.2МБ, Экономия=98.4%
```

## Математическое обоснование

### Лемма Джонсона-Линденштрауса

Теоретическое обоснование Linformer опирается на **лемму Джонсона-Линденштрауса (JL)**:

> Для любого ε > 0 и любого множества из n точек в многомерном пространстве
> существует линейная проекция в пространство размерности k = O(log(n)/ε²)
> такая, что все попарные расстояния сохраняются с точностью до множителя (1 ± ε).

**Применение к Attention:**
```
Если матрица внимания A имеет эффективный ранг r,
то проекция из n измерений в k ≥ r измерений
сохраняет существенную информацию с высокой вероятностью.
```

### Низкоранговое свойство Self-Attention

Эмпирическое наблюдение: матрицы self-attention приблизительно низкоранговые.

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_attention_rank(attention_matrix):
    """
    Анализ эффективного ранга матрицы внимания.
    """
    # Сингулярное разложение
    U, S, Vh = np.linalg.svd(attention_matrix)

    # Кумулятивная энергия (объяснённая дисперсия)
    total_energy = np.sum(S ** 2)
    cumulative_energy = np.cumsum(S ** 2) / total_energy

    # Эффективный ранг (95% энергии)
    effective_rank = np.argmax(cumulative_energy >= 0.95) + 1

    return effective_rank, cumulative_energy

# Типичный результат: Для n=1024 эффективный ранг часто < 128
# Это обосновывает использование k=128 для размерности проекции
```

Матрица внимания может быть разложена:
```
A = softmax(Q @ K.T / sqrt(d_k))

SVD: A = U @ Σ @ V.T

Если Σ имеет быстрое затухание (мало доминирующих сингулярных чисел),
A эффективно низкоранговая и может быть аппроксимирована.
```

### Границы ошибки

Ошибка аппроксимации ограничена:

```
Дано:
- P = softmax(Q @ K.T / sqrt(d_k)) @ V  (стандартное внимание)
- P̂ = softmax(Q @ (E @ K).T / sqrt(d_k)) @ (F @ V)  (Linformer)

Теорема: Для k = O(d/ε²), с высокой вероятностью:
||P - P̂||_F ≤ ε ||P||_F

Смысл: Ошибка ограничена и контролируется через k.
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 2048,  # Длинные последовательности для Linformer!
    horizon: int = 24
) -> Dict:
    """
    Подготовка данных с длинными последовательностями для обучения Linformer.

    Linformer может эффективно обрабатывать гораздо более длинные
    последовательности, чем стандартные трансформеры.

    Args:
        symbols: Список торговых пар (напр., ['BTCUSDT', 'ETHUSDT'])
        lookback: Количество исторических временных шагов (может быть очень большим!)
        horizon: Горизонт прогнозирования

    Returns:
        Словарь с X (признаки) и y (таргеты)
    """
    all_data = []

    for symbol in symbols:
        # Загрузка данных с Bybit
        df = load_bybit_data(symbol, interval='1h', limit=lookback + horizon + 100)

        # Расчёт признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['momentum'] = df['close'] / df['close'].shift(20) - 1

        all_data.append(df)

    # Выравнивание по временной метке
    aligned = pd.concat(all_data, axis=1, keys=symbols)
    aligned = aligned.dropna()

    # Создание последовательностей
    X, y = [], []
    for i in range(lookback, len(aligned) - horizon):
        X.append(aligned.iloc[i-lookback:i].values)
        y.append(aligned.iloc[i+horizon]['log_return'].values)

    return {
        'X': np.array(X),           # [n_samples, lookback, features]
        'y': np.array(y),           # [n_samples, n_assets]
        'symbols': symbols,
        'lookback': lookback,
        'horizon': horizon
    }
```

### 02: Архитектура Linformer

```python
# python/linformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinformerAttention(nn.Module):
    """
    Linformer Self-Attention с линейной сложностью.

    Проецирует ключи и значения в меньшую размерность k,
    снижая сложность с O(n²) до O(n×k).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        k: int = 128,
        dropout: float = 0.1,
        share_kv: bool = True
    ):
        """
        Args:
            d_model: Размерность модели
            n_heads: Количество голов внимания
            seq_len: Фиксированная длина последовательности
            k: Размерность проекции (k << seq_len)
            dropout: Коэффициент dropout
            share_kv: Если True, общая проекция для K и V
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model должен делиться на n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k
        self.scale = math.sqrt(self.d_k)

        # Проекции Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Матрицы линейной проекции E и F
        # Проецируют из seq_len в k измерений
        self.E = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        if share_kv:
            # Общая проекция для K и V (более эффективно)
            self.F = self.E
        else:
            # Раздельные проекции для K и V
            self.F = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass с вниманием линейной сложности.
        """
        batch_size, seq_len, _ = x.shape

        # Линейные проекции для Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Изменение формы для многоголового внимания
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Проекция K и V в меньшую размерность
        K_proj = torch.einsum('hkn,bhnd->bhkd', self.E, K)
        V_proj = torch.einsum('hkn,bhnd->bhkd', self.F, V)

        # Вычисление оценок внимания
        attention_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Применение внимания к спроецированным значениям
        context = torch.matmul(attention_weights, V_proj)

        # Изменение формы обратно
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(context)

        if return_attention:
            return output, attention_weights
        return output, None
```

### 03: Обучение модели

```python
# python/03_train_model.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from linformer import Linformer
import numpy as np


def train_linformer(
    train_data: dict,
    val_data: dict,
    config: dict,
    device: str = 'cuda'
) -> Linformer:
    """
    Обучение модели Linformer на финансовых данных.
    """
    # Инициализация модели
    model = Linformer(
        n_features=train_data['X'].shape[-1],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        k=config['k'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        output_type=config['output_type'],
        n_outputs=config['n_outputs']
    ).to(device)

    # Создание загрузчиков данных
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_data['X']),
            torch.FloatTensor(train_data['y'])
        ),
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Цикл обучения
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        avg_train_loss = np.mean(train_losses)
        print(f"Эпоха {epoch+1}/{config['epochs']}: "
              f"Train Loss={avg_train_loss:.6f}")

    return model


# Пример конфигурации
config = {
    'seq_len': 2048,        # Длинная последовательность - Linformer справляется!
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'k': 128,               # Размерность проекции
    'd_ff': 1024,
    'dropout': 0.1,
    'output_type': 'regression',
    'n_outputs': 1,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 10
}
```

### 04: Прогнозирование на длинных последовательностях

```python
# python/04_long_sequence_forecasting.py

import torch
import numpy as np
import pandas as pd
from typing import List, Dict


def forecast_with_long_context(
    model: torch.nn.Module,
    data: pd.DataFrame,
    lookback: int = 2048,
    device: str = 'cuda'
) -> Dict:
    """
    Прогнозирование с использованием длинного исторического контекста.

    Linformer может эффективно использовать гораздо более длинный контекст,
    чем стандартные трансформеры, потенциально захватывая долгосрочные паттерны.
    """
    model.eval()

    # Подготовка признаков
    features = prepare_features(data)

    # Получение последних lookback временных шагов
    x = torch.FloatTensor(features[-lookback:]).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(x)

    return {
        'prediction': prediction.cpu().numpy(),
        'timestamp': data.index[-1],
        'context_length': lookback
    }
```

### 05: Бэктестинг портфеля

```python
# python/05_backtest.py

import pandas as pd
import numpy as np
from typing import Dict, List


def backtest_linformer_strategy(
    model: torch.nn.Module,
    test_data: Dict,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Бэктестинг торговой стратегии на основе Linformer.
    """
    model.eval()

    X = torch.FloatTensor(test_data['X']).to(device)
    actual_returns = test_data['y']

    capital = initial_capital
    position = 0
    results = []

    for i in range(len(X)):
        with torch.no_grad():
            prediction = model(X[i:i+1]).cpu().numpy().flatten()[0]

        # Генерация сигнала
        if prediction > 0.001:
            target_position = 1
        elif prediction < -0.001:
            target_position = -1
        else:
            target_position = 0

        # Расчёт стоимости сделки
        position_change = abs(target_position - position)
        trade_cost = position_change * transaction_cost * capital

        position = target_position

        # Расчёт доходности
        actual_return = actual_returns[i].item() if hasattr(actual_returns[i], 'item') else actual_returns[i]
        portfolio_return = position * actual_return

        capital = capital * (1 + portfolio_return) - trade_cost

        results.append({
            'step': i,
            'capital': capital,
            'position': position,
            'prediction': prediction,
            'portfolio_return': portfolio_return
        })

    df = pd.DataFrame(results)

    # Расчёт метрик
    total_return = (df['capital'].iloc[-1] / initial_capital - 1) * 100
    sharpe_ratio = calculate_sharpe_ratio(df['portfolio_return'])
    max_drawdown = calculate_max_drawdown(df['capital'])

    print("\nРезультаты бэктестинга:")
    print(f"Общая доходность: {total_return:.2f}%")
    print(f"Коэффициент Шарпа: {sharpe_ratio:.3f}")
    print(f"Максимальная просадка: {max_drawdown:.2f}%")
    print(f"Итоговый капитал: ${df['capital'].iloc[-1]:,.2f}")

    return df
```

## Реализация на Rust

См. [rust/](rust/) для полной реализации на Rust.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент API Bybit
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура Linformer
│   │   ├── mod.rs
│   │   ├── attention.rs    # Реализация линейного внимания
│   │   ├── encoder.rs      # Слои энкодера
│   │   └── linformer.rs    # Полная модель
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
# Перейти в проект Rust
cd rust

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель с длинными последовательностями
cargo run --example train -- --seq-len 2048 --epochs 100

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

```
python/
├── linformer.py            # Основная реализация модели
├── data_loader.py          # Загрузка данных Bybit
├── features.py             # Инженерия признаков
├── train.py                # Скрипт обучения
├── backtest.py             # Утилиты бэктестинга
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_linformer_architecture.ipynb
    ├── 03_training.ipynb
    ├── 04_long_sequence_forecasting.ipynb
    └── 05_backtesting.ipynb
```

### Быстрый старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка данных
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучение модели с длинными последовательностями
python train.py --seq-len 2048 --k 128 --epochs 100

# Запуск бэктеста
python backtest.py --model checkpoints/best_linformer.pt
```

## Лучшие практики

### Когда использовать Linformer

**Хорошие варианты использования:**
- Анализ длинных последовательностей (1000+ токенов)
- Прогнозирование временных рядов фиксированной длины
- Неавторегрессивные задачи (кодирование, классификация)
- Среды с ограниченной памятью
- Высокочастотные данные с длинными окнами просмотра

**Не идеально для:**
- Авторегрессивной генерации (используйте Performer или Longformer)
- Последовательностей переменной длины (требуется паддинг)
- Задач, требующих полных паттернов внимания
- Очень коротких последовательностей (стандартное внимание достаточно)

### Выбор размерности проекции k

```python
# Правило выбора k:
# k должен захватывать эффективный ранг внимания

# Для финансовых временных рядов:
# - Короткие последовательности (n < 512): k = 64
# - Средние последовательности (512 <= n < 2048): k = 128
# - Длинные последовательности (n >= 2048): k = 256
```

### Рекомендации по гиперпараметрам

| Параметр | Рекомендация | Примечания |
|----------|--------------|------------|
| `seq_len` | 2048-8192 | Оптимальный диапазон для Linformer |
| `k` | 128-256 | Размерность проекции |
| `d_model` | 256-512 | Соответствие n_heads |
| `n_heads` | 8-16 | Должен делить d_model |
| `n_layers` | 4-6 | Больше для длинных последовательностей |
| `dropout` | 0.1-0.2 | Выше для малых датасетов |

### Типичные ошибки

1. **Последовательности переменной длины**: Дополните до фиксированной длины или используйте разбиение
2. **Интерпретируемость внимания**: Спроецированное внимание сложнее интерпретировать
3. **Позиционное кодирование**: Критично для длинных последовательностей; рассмотрите обучаемое
4. **Компромисс память-вычисления**: k слишком мало вредит качеству; слишком велико снижает выгоду

## Ресурсы

### Научные работы

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) — Оригинальная статья
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Обзор эффективных механизмов внимания
- [Long Range Arena](https://arxiv.org/abs/2011.04006) — Бенчмарк для задач дальнего действия

### Реализации

- [lucidrains/linformer](https://github.com/lucidrains/linformer) — Реализация на PyTorch
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — Хаб моделей
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Библиотека временных рядов

### Связанные главы

- [Глава 50: Memory-Augmented Transformers](../50_memory_augmented_transformers) — Механизмы внешней памяти
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Внимание на случайных признаках
- [Глава 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) — Разреженные паттерны внимания
- [Глава 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Locality-sensitive hashing

---

## Уровень сложности

**Средний - Продвинутый**

Предварительные требования:
- Архитектура трансформера и self-attention
- Матричная факторизация и низкоранговая аппроксимация
- Основы анализа временных рядов
- Библиотеки ML на PyTorch/Rust
