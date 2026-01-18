# Глава 44: ProbSparse Attention для трейдинга

Эта глава посвящена **ProbSparse Attention** — эффективному механизму самовнимания, который снижает вычислительную сложность с O(L²) до O(L·log(L)). Изначально представленный в статье Informer для прогнозирования длинных временных рядов, ProbSparse Attention особенно ценен для финансовых приложений, обрабатывающих большие объёмы исторических данных.

<p align="center">
<img src="https://i.imgur.com/QR7Zk8v.png" width="70%">
</p>

## Содержание

1. [Введение в ProbSparse Attention](#введение-в-probsparse-attention)
    * [Почему важна эффективность внимания](#почему-важна-эффективность-внимания)
    * [Ключевые инновации](#ключевые-инновации)
    * [Сравнение с другими методами](#сравнение-с-другими-методами)
2. [Математические основы](#математические-основы)
    * [Измерение разреженности запросов](#измерение-разреженности-запросов)
    * [Интуиция KL-дивергенции](#интуиция-kl-дивергенции)
    * [Выбор Top-u запросов](#выбор-top-u-запросов)
3. [Компоненты архитектуры](#компоненты-архитектуры)
    * [ProbSparse Self-Attention](#probsparse-self-attention)
    * [Дистилляция Self-Attention](#дистилляция-self-attention)
    * [Стек энкодеров](#стек-энкодеров)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в ProbSparse Attention

### Почему важна эффективность внимания

Стандартный self-attention в Трансформерах имеет фундаментальное ограничение: **квадратичная сложность O(L²)**, где L — длина последовательности. Для прогнозирования финансовых временных рядов это создаёт значительные проблемы:

```
Использование памяти стандартным Attention:
┌─────────────────┬──────────────────┬────────────────┐
│ Длина последов. │ Стандартный O(L²)│ ProbSparse     │
├─────────────────┼──────────────────┼────────────────┤
│ L = 96          │ 9,216 операций   │ ~640 оп.       │
│ L = 720 (месяц) │ 518,400 оп.      │ ~4,700 оп.     │
│ L = 8,760 (год) │ 76,737,600 оп.   │ ~79,000 оп.    │
└─────────────────┴──────────────────┴────────────────┘
```

Для торговых приложений, требующих 1+ год часовых данных, ProbSparse Attention делает модели Трансформеров практичными.

### Ключевые инновации

1. **Измерение разреженности запросов**: Не все запросы одинаково важны для внимания. ProbSparse идентифицирует "активные" запросы, генерирующие разнообразные паттерны внимания, и концентрирует вычисления на них.

2. **Выбор Top-u запросов**: Только самые информативные запросы (u = c·log(L)) участвуют в полном вычислении внимания.

3. **Дистилляция Self-Attention**: Прогрессивное сокращение длины последовательности через слои энкодера устраняет избыточность.

```
┌──────────────────────────────────────────────────────────────────┐
│                ПОТОК PROBSPARSE ATTENTION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Входная последовательность: [q₁, q₂, q₃, ..., qₗ]  (L запросов)│
│                                                                   │
│         │ Вычисляем измерение разреженности M(qᵢ, K)             │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                 │
│  │ M(qᵢ) = max(qᵢKᵀ/√d) - mean(qᵢKᵀ/√d)       │                 │
│  │                                              │                 │
│  │ "Активные" запросы: Высокий M → Разное      │                 │
│  │                                 внимание     │                 │
│  │ "Ленивые" запросы: Низкий M → Равномерное   │                 │
│  │                                 внимание     │                 │
│  └─────────────────────────────────────────────┘                 │
│         │                                                         │
│         │ Выбираем Top-u запросов (u = c·log(L))                 │
│         ▼                                                         │
│  Q_reduce = [q₃, q₇, q₁₂, ...]  (только u запросов)             │
│                                                                   │
│         │ Вычисляем внимание только для Q_reduce                 │
│         ▼                                                         │
│  Выход: Разреженное внимание со сложностью O(L·log(L))          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Сравнение с другими методами

| Метод | Сложность | Память | Дальний обзор | Применение |
|-------|-----------|--------|---------------|------------|
| Full Attention | O(L²) | O(L²) | ✓ | Только короткие послед. |
| Local Attention | O(L·k) | O(L·k) | Ограниченно | Внутридневные паттерны |
| Linformer | O(L·k) | O(L·k) | ✓ | Общее применение |
| Performer | O(L·d) | O(L·d) | ✓ | Общее применение |
| **ProbSparse** | O(L·logL) | O(L·logL) | ✓ | **Длинные послед.** |
| Flash Attention | O(L²) | O(L) | ✓ | Оптимизация железа |

## Математические основы

### Измерение разреженности запросов

Ключевое наблюдение ProbSparse: оценки внимания следуют **распределению с длинным хвостом**. Большинство запросов производят равномерное внимание (мало информации), в то время как несколько "активных" запросов сильно фокусируются на конкретных ключах.

**Измерение разреженности запроса** количественно определяет, насколько "остро" распределение внимания запроса:

```
M(qᵢ, K) = max_j(qᵢ · kⱼᵀ / √d) - (1/Lₖ) Σⱼ(qᵢ · kⱼᵀ / √d)
```

Где:
- `qᵢ` — i-й вектор запроса
- `kⱼ` — векторы ключей
- `d` — размерность эмбеддинга
- `Lₖ` — длина последовательности ключей

**Интерпретация**:
- **Высокий M(qᵢ)**: Запрос имеет доминирующий ключ → "Активный" запрос
- **Низкий M(qᵢ)**: Запрос уделяет внимание равномерно → "Ленивый" запрос

### Интуиция KL-дивергенции

Измерение разреженности M аппроксимирует KL-дивергенцию между реальным распределением внимания и равномерным распределением:

```
KL(p || q_uniform) ≈ log(Lₖ) + M(qᵢ, K)
```

Активные запросы имеют высокую KL-дивергенцию (далеко от равномерного), а ленивые запросы — низкую KL-дивергенцию (близко к равномерному).

```python
# Интуиция: Активные vs Ленивые запросы
import numpy as np

# Активный запрос: сильно концентрируется на конкретных ключах
active_attention = np.array([0.8, 0.1, 0.05, 0.03, 0.02])  # Острое
M_active = active_attention.max() - active_attention.mean()  # Высокое

# Ленивый запрос: равномерное внимание
lazy_attention = np.array([0.21, 0.20, 0.20, 0.19, 0.20])  # Плоское
M_lazy = lazy_attention.max() - lazy_attention.mean()  # Низкое

print(f"Активный M: {M_active:.3f}")  # ~0.6
print(f"Ленивый M: {M_lazy:.3f}")     # ~0.01
```

### Выбор Top-u запросов

На основе измерений разреженности мы выбираем только top-u запросов для полного вычисления внимания:

```
u = min(c · log(Lq), Lq)
```

Где:
- `c` — фактор выборки (обычно 5)
- `Lq` — длина последовательности запросов

Для последовательности из 720 временных шагов:
```
u = 5 × log(720) ≈ 5 × 6.58 ≈ 33 запроса
```

Это сокращает операции с 720² = 518,400 до примерно 720 × 33 = 23,760 — **сокращение в 22 раза**.

## Компоненты архитектуры

### ProbSparse Self-Attention

```python
class ProbSparseAttention(nn.Module):
    """
    Механизм ProbSparse Self-Attention

    Достигает сложности O(L·log(L)), выбирая только самые
    информативные запросы для полного вычисления внимания.
    """

    def __init__(self, d_model: int, n_heads: int, sampling_factor: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sampling_factor = sampling_factor
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        # Проекция в Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Вычисляем количество top запросов для выбора
        u = max(1, min(seq_len, int(self.sampling_factor * math.log(seq_len + 1))))

        # Вычисляем измерение разреженности запроса M(q, K)
        U_part = min(int(self.sampling_factor * seq_len * math.log(seq_len + 1)), seq_len)
        sample_idx = torch.randint(0, seq_len, (U_part,), device=x.device)
        K_sample = K[:, :, sample_idx, :]

        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / self.scale

        # M(q) = max(scores) - mean(scores)
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)

        # Выбираем top-u запросов
        M_top_indices = M.topk(u, dim=-1)[1]

        # Собираем выбранные запросы
        batch_idx = torch.arange(batch, device=x.device)[:, None, None]
        head_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
        Q_reduce = Q[batch_idx, head_idx, M_top_indices]

        # Полное внимание только для выбранных запросов
        attn_scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)

        # Инициализируем выход средними значениями, затем заполняем разреженные позиции
        output = V.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        output.scatter_(2, M_top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim), context)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(output)
```

### Дистилляция Self-Attention

Операция дистилляции прогрессивно сокращает длину последовательности между слоями энкодера:

```python
class AttentionDistilling(nn.Module):
    """
    Слой дистилляции, сокращающий длину последовательности вдвое.

    Использует Conv1d + ELU + MaxPool для извлечения значимых признаков
    при отбрасывании избыточной информации.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)  # [batch, seq_len//2, d_model]
```

### Стек энкодеров

```
┌─────────────────────────────────────────────────────────────────┐
│                     ЭНКОДЕР INFORMER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Вход: [batch, L, d_model]                                      │
│                                                                  │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Слой энкодера 1       │  ← ProbSparse Attention            │
│  │   [batch, L, d_model]   │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Слой дистилляции 1    │  ← Conv + MaxPool (L → L/2)        │
│  │   [batch, L/2, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Слой энкодера 2       │  ← ProbSparse Attention            │
│  │   [batch, L/2, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Слой дистилляции 2    │  ← Conv + MaxPool (L/2 → L/4)      │
│  │   [batch, L/4, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Слой энкодера 3       │  ← ProbSparse Attention            │
│  │   [batch, L/4, d_model] │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  Выход: [batch, L/4, d_model]                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Tuple
import torch

def prepare_informer_data(
    df: pd.DataFrame,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    features: List[str] = ['close', 'volume', 'high', 'low', 'open']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготовка данных для обучения модели Informer.

    Args:
        df: DataFrame с OHLCV данными
        seq_len: Длина входной последовательности (энкодер)
        label_len: Длина последовательности меток (начало декодера)
        pred_len: Горизонт прогнозирования
        features: Столбцы признаков для использования

    Returns:
        X: Входные последовательности [n_samples, seq_len, n_features]
        y: Целевые последовательности [n_samples, pred_len]
    """

    # Вычисляем доходности и технические индикаторы
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'] / df['volume'].rolling(20).mean()

    # Нормализуем признаки
    for col in features:
        df[f'{col}_norm'] = (df[col] - df[col].rolling(100).mean()) / df[col].rolling(100).std()

    df = df.dropna()
    data = df[[f'{col}_norm' for col in features]].values
    targets = df['returns'].values

    # Создаём последовательности
    X, y = [], []
    for i in range(seq_len, len(data) - pred_len):
        X.append(data[i-seq_len:i])
        y.append(targets[i:i+pred_len])

    return np.array(X), np.array(y)
```

### 02: Обучение модели

```python
# python/03_train.py

import torch
import torch.nn as nn
from model import InformerModel, InformerConfig

def train_informer(
    train_loader,
    val_loader,
    config: InformerConfig,
    epochs: int = 100,
    lr: float = 0.001
):
    """Обучение модели Informer с ProbSparse attention"""

    model = InformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Эпоха {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return model
```

### 03: Бэктестинг стратегии

```python
# python/05_backtest.py

def backtest_informer_strategy(
    model,
    test_data: pd.DataFrame,
    seq_len: int = 96,
    pred_len: int = 24,
    threshold: float = 0.0005,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Бэктестинг торговой стратегии на основе прогнозов Informer.

    Стратегия: Открыть лонг если прогноз > порога,
               Открыть шорт если прогноз < -порога,
               Иначе оставаться вне рынка.
    """

    capital = initial_capital
    position = 0  # -1: шорт, 0: вне рынка, 1: лонг

    results = []

    for i in range(seq_len, len(test_data) - pred_len):
        # Получаем входную последовательность
        X = test_data.iloc[i-seq_len:i][['close_norm', 'volume_norm', 'volatility_norm']].values
        X = torch.FloatTensor(X).unsqueeze(0)

        # Получаем прогноз
        with torch.no_grad():
            pred = model(X)[0, 0].item()

        # Получаем реальную доходность
        actual_return = np.log(
            test_data.iloc[i+1]['close'] / test_data.iloc[i]['close']
        )

        # Логика торговли
        new_position = 0
        if pred > threshold:
            new_position = 1
        elif pred < -threshold:
            new_position = -1

        # Вычисляем транзакционные издержки при смене позиции
        if new_position != position:
            capital *= (1 - transaction_cost)

        # Вычисляем PnL
        pnl = position * actual_return * capital
        capital += pnl

        position = new_position

        results.append({
            'timestamp': test_data.index[i],
            'capital': capital,
            'position': position,
            'predicted_return': pred,
            'actual_return': actual_return,
            'pnl': pnl
        })

    results_df = pd.DataFrame(results)

    # Вычисляем метрики
    returns = results_df['pnl'] / results_df['capital'].shift(1)
    returns = returns.dropna()

    metrics = {
        'total_return': (capital - initial_capital) / initial_capital,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24),
        'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24),
        'max_drawdown': (results_df['capital'].cummax() - results_df['capital']).max() / results_df['capital'].cummax().max(),
        'win_rate': (returns > 0).mean(),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else float('inf'),
        'num_trades': (results_df['position'].diff() != 0).sum()
    }

    return {
        'results': results_df,
        'metrics': metrics
    }
```

## Реализация на Rust

Смотрите [rust/](rust/) для полной реализации на Rust с использованием данных Bybit.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Dataset для обучения
│   ├── model/              # Архитектура Informer
│   │   ├── mod.rs
│   │   ├── attention.rs    # ProbSparse attention
│   │   ├── embedding.rs    # Token embedding
│   │   ├── encoder.rs      # Энкодер с дистилляцией
│   │   └── informer.rs     # Полная модель
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
cd rust

# Загрузка данных с Bybit
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 10000

# Обучение модели
cargo run --example train -- --epochs 100 --batch-size 32

# Запуск бэктеста
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── model.py                # Informer с ProbSparse attention
├── data.py                 # Загрузка и предобработка данных
├── train.py                # Скрипт обучения
├── backtest.py             # Утилиты бэктестинга
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_architecture.py
    ├── 03_training.py
    ├── 04_forecasting.py
    └── 05_backtesting.py
```

### Быстрый старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка данных
python data.py --symbol BTCUSDT --interval 1h --limit 10000

# Обучение модели
python train.py --config configs/default.yaml

# Запуск бэктеста
python backtest.py --model checkpoints/best_model.pt
```

## Лучшие практики

### Когда использовать ProbSparse Attention

**Хорошие случаи использования:**
- Прогнозирование длинных последовательностей (L > 100)
- Мульти-горизонтные предсказания
- Среды с ограниченными ресурсами
- Системы реального времени

**Не идеально для:**
- Очень коротких последовательностей (L < 50) — накладные расходы превышают выгоду
- Задач, требующих полной интерпретируемости внимания
- Когда максимальная точность важнее эффективности

### Рекомендуемые гиперпараметры

| Параметр | Рекомендуемое | Примечания |
|----------|---------------|------------|
| `seq_len` | 96-720 | Длиннее для данных низкой частоты |
| `d_model` | 64-256 | Зависит от сложности данных |
| `n_heads` | 4-8 | Должно делить d_model |
| `sampling_factor` | 5 | Дефолт из статьи, редко нужна настройка |
| `n_encoder_layers` | 2-4 | Больше слоёв — используйте дистилляцию |
| `dropout` | 0.1-0.2 | Выше для малых датасетов |

### Типичные ошибки

1. **Слишком короткая последовательность**: Накладные расходы ProbSparse не оправданы для L < 50
2. **Отсутствие нормализации**: Всегда нормализуйте входы для стабильного обучения
3. **Игнорирование дистилляции**: Для глубоких энкодеров дистилляция необходима
4. **Избыточная выборка запросов**: Не устанавливайте sampling_factor слишком высоко (>10)

## Ресурсы

### Научные статьи

- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Оригинальная статья ProbSparse
- [Autoformer: Decomposition Transformers with Auto-Correlation](https://arxiv.org/abs/2106.13008) — Связанный эффективный Трансформер
- [FEDformer: Frequency Enhanced Decomposed Transformer](https://arxiv.org/abs/2201.12740) — Подход в частотной области
- [Comparing Different Transformer Model Structures for Stock Prediction](https://arxiv.org/abs/2504.16361) — Сравнение для трейдинга

### Реализации

- [Hugging Face Informer](https://huggingface.co/docs/transformers/en/model_doc/informer) — Официальная реализация
- [Informer2020 GitHub](https://github.com/zhouhaoyi/Informer2020) — Код оригинальных авторов
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Библиотека временных рядов

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Мульти-горизонтное прогнозирование
- [Глава 43: Stockformer Multivariate](../43_stockformer_multivariate) — Кросс-активное предсказание
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Линейное внимание
- [Глава 58: Flash Attention Trading](../58_flash_attention_trading) — Аппаратная оптимизация

---

## Уровень сложности

**Средний-Продвинутый**

Предварительные требования:
- Основы архитектуры Трансформера
- Понимание механизма self-attention
- Базовые знания прогнозирования временных рядов
- Опыт работы с PyTorch или Rust ML
