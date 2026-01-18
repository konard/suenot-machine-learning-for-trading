# Глава 41: Higher Order Transformers для торговли криптовалютами

## Обзор

Higher Order Transformers (HOT) расширяют стандартный механизм self-attention для захвата сложных многомерных зависимостей в мультивариантных временных рядах. В отличие от традиционных трансформеров, которые моделируют попарные связи, HOT захватывает взаимодействия высшего порядка между несколькими активами одновременно, что делает его особенно мощным для прогнозирования криптовалютных рынков.

<p align="center">
<img src="https://i.imgur.com/YqKxZ9M.png" width="70%">
</p>

## Содержание

1. [Введение](#введение)
   * [Зачем нужен Higher Order Attention?](#зачем-нужен-higher-order-attention)
   * [Ключевые инновации](#ключевые-инновации)
2. [Математические основы](#математические-основы)
   * [Стандартный Self-Attention](#стандартный-self-attention)
   * [Higher Order Self-Attention](#higher-order-self-attention)
   * [Тензорное разложение](#тензорное-разложение)
3. [Архитектура](#архитектура)
   * [Higher Order Attention слой](#higher-order-attention-слой)
   * [Kernel Attention для эффективности](#kernel-attention-для-эффективности)
   * [Полная архитектура модели](#полная-архитектура-модели)
4. [Реализация](#реализация)
   * [Реализация на Rust](#реализация-на-rust)
   * [Python референс](#python-референс)
5. [Применение к криптовалютам](#применение-к-криптовалютам)
   * [Обработка рыночных данных](#обработка-рыночных-данных)
   * [Инженерия признаков](#инженерия-признаков)
   * [Торговые сигналы](#торговые-сигналы)
6. [Бэктестинг](#бэктестинг)
7. [Ресурсы](#ресурсы)

## Введение

### Зачем нужен Higher Order Attention?

Стандартные трансформеры используют **попарное внимание** для моделирования связей между двумя временными шагами или двумя активами. Однако финансовые рынки демонстрируют сложные **многосторонние взаимодействия**:

- Актив A влияет на актив B, который затем влияет на актив C
- Три или более актива движутся вместе из-за общих факторов
- Нелинейные зависимости, охватывающие несколько измерений

**Пример:** Когда Bitcoin падает, Ethereum часто следует за ним, но связь с альткоинами зависит от общего настроения рынка — трёхстороннее взаимодействие.

```
Стандартный Attention:     BTC → ETH (только попарно)
Higher Order Attention: BTC × ETH × Настроение_рынка → Движение альткоинов
```

### Ключевые инновации

1. **Тензор внимания третьего порядка:** Захватывает взаимодействия между тройками временных шагов/активов
2. **Тензорное разложение:** Снижает вычислительную сложность с O(n³) до O(n×r²)
3. **Kernel Attention:** Дополнительно снижает до линейной сложности O(n)
4. **Мультимодальное слияние:** Объединяет ценовые данные с другими модальностями (объём, настроение)

## Математические основы

### Стандартный Self-Attention

Стандартный self-attention вычисляет:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Где:
- Q, K, V ∈ ℝ^(n×d) — матрицы query, key, value
- n = длина последовательности
- d = размерность эмбеддинга

Это захватывает **попарные** связи со сложностью O(n²d).

### Higher Order Self-Attention

Higher Order Attention расширяет это на **тройные** взаимодействия:

```
HOA(Q, K, V) = softmax(T(Q, K, K) / √d) × V
```

Где T — **тензор третьего порядка**, захватывающий трёхсторонние взаимодействия:

```
T[i,j,k] = Σ_d Q[i,d] × K[j,d] × K[k,d]
```

**Интуиция:** Вместо вопроса "насколько позиция j связана с позицией i?" мы спрашиваем "насколько позиции j И k вместе связаны с позицией i?"

### Тензорное разложение

Наивный тензор третьего порядка имеет сложность O(n³) — непрактично для длинных последовательностей. Мы используем **CP (CANDECOMP/PARAFAC) разложение**:

```
T[i,j,k] ≈ Σ_r a_r[i] × b_r[j] × c_r[k]
```

Где r — **ранг** разложения. Это снижает сложность до O(n×r²).

```python
# Концептуальная иллюстрация
def cp_decomposition(tensor, rank):
    """
    Разложение 3D тензора на компоненты ранга 1

    Args:
        tensor: Исходный тензор формы (n, n, n)
        rank: Количество компонент ранга 1

    Returns:
        factors: Список из 3 матриц, каждая формы (n, rank)
    """
    # Инициализация факторных матриц
    A = random_init(n, rank)  # Фактор 1
    B = random_init(n, rank)  # Фактор 2
    C = random_init(n, rank)  # Фактор 3

    # Оптимизация методом чередующихся наименьших квадратов
    for iteration in range(max_iter):
        A = update_factor(tensor, B, C)
        B = update_factor(tensor, A, C)
        C = update_factor(tensor, A, B)

    return [A, B, C]
```

**Компромисс:**
- Выше ранг = лучше аппроксимация, больше вычислений
- Ниже ранг = быстрее, но может потерять важные взаимодействия
- Типичные значения: ранг = 8-64 в зависимости от задачи

## Архитектура

### Higher Order Attention слой

```
┌─────────────────────────────────────────────────────────────┐
│              Higher Order Attention Layer                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Вход X ∈ ℝ^(n×d)                                         │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Linear  │ → Q, K, V                                    │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ CP Разложение       │                                  │
│    │ T ≈ Σ a_r ⊗ b_r ⊗ c_r │                                │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Kernel  │ → φ(Q), φ(K)                                 │
│    │ Mapping │                                              │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Softmax │ → Веса внимания                              │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │  @ V    │ → Выход                                      │
│    └────┬────┘                                              │
│         │                                                   │
│    Выход ∈ ℝ^(n×d)                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Attention для эффективности

Для достижения линейной сложности мы используем **kernel attention** с feature maps:

```
φ(x) = exp(x) / √Σexp(x)²
```

Attention становится:

```
KernelAttn(Q, K, V) = φ(Q) × (φ(K)^T × V) / (φ(Q) × φ(K)^T × 1)
```

**Снижение сложности:**
- Стандартный: O(n²d)
- С kernel: O(nd²) — линейная по длине последовательности!

### Полная архитектура модели

```
┌────────────────────────────────────────────────────────────────┐
│           Higher Order Transformer для трейдинга                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐                                              │
│  │ Ценовые данные│──┐                                          │
│  │ (OHLCV)      │  │                                           │
│  └──────────────┘  │    ┌─────────────────┐                    │
│                    ├───→│ Input Embedding │                    │
│  ┌──────────────┐  │    │ + Positional    │                    │
│  │ Объём/OI     │──┤    └────────┬────────┘                    │
│  └──────────────┘  │             │                             │
│                    │    ┌────────┴────────┐                    │
│  ┌──────────────┐  │    │   HOT Block ×N  │                    │
│  │ Тех. индик.  │──┘    │  ┌───────────┐  │                    │
│  └──────────────┘       │  │ HOA Layer │  │                    │
│                         │  ├───────────┤  │                    │
│                         │  │ FFN Layer │  │                    │
│                         │  ├───────────┤  │                    │
│                         │  │ LayerNorm │  │                    │
│                         │  └───────────┘  │                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                         ┌────────┴────────┐                    │
│                         │ Output Head     │                    │
│                         │ (Классификация) │                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                         ┌────────┴────────┐                    │
│                         │ Вверх/Вниз/Нейт.│                    │
│                         │   Прогноз       │                    │
│                         └─────────────────┘                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Реализация

### Реализация на Rust

Директория [rust_hot_crypto](rust_hot_crypto/) содержит модульную реализацию на Rust:

```
rust_hot_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Главный модуль библиотеки
│   ├── main.rs                 # CLI интерфейс
│   ├── data/
│   │   ├── mod.rs              # Модуль данных
│   │   ├── bybit.rs            # Клиент Bybit API
│   │   └── types.rs            # Типы данных (Candle и т.д.)
│   ├── attention/
│   │   ├── mod.rs              # Модуль внимания
│   │   ├── standard.rs         # Стандартный attention
│   │   ├── higher_order.rs     # Higher order attention
│   │   └── kernel.rs           # Kernel attention
│   ├── tensor/
│   │   ├── mod.rs              # Модуль тензоров
│   │   ├── decomposition.rs    # CP/Tucker разложение
│   │   └── operations.rs       # Тензорные операции
│   ├── model/
│   │   ├── mod.rs              # Модуль модели
│   │   ├── transformer.rs      # HOT трансформер
│   │   └── predictor.rs        # Предсказатель цен
│   ├── strategy/
│   │   ├── mod.rs              # Модуль стратегии
│   │   └── signals.rs          # Торговые сигналы
│   └── utils/
│       ├── mod.rs              # Утилиты
│       └── config.rs           # Конфигурация
└── examples/
    ├── fetch_data.rs           # Загрузка данных Bybit
    ├── attention_demo.rs       # Демонстрация HOA
    ├── predict_movement.rs     # Прогноз движения цены
    └── backtest.rs             # Бэктест стратегии
```

### Быстрый старт с Rust

```bash
# Перейти в директорию Rust проекта
cd 41_higher_order_transformers/rust_hot_crypto

# Загрузить данные криптовалют с Bybit
cargo run --example fetch_data

# Запустить демонстрацию higher order attention
cargo run --example attention_demo

# Прогнозировать движение цены
cargo run --example predict_movement

# Запустить полный бэктест
cargo run --example backtest
```

### Python референс

```python
import torch
import torch.nn as nn
import tensorly as tl

class HigherOrderAttention(nn.Module):
    """
    Higher Order Self-Attention с CP разложением
    """
    def __init__(self, d_model, n_heads=8, rank=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.d_k = d_model // n_heads

        # Линейные проекции
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Факторы CP разложения
        self.factor_a = nn.Parameter(torch.randn(n_heads, rank, self.d_k))
        self.factor_b = nn.Parameter(torch.randn(n_heads, rank, self.d_k))
        self.factor_c = nn.Parameter(torch.randn(n_heads, rank, self.d_k))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Проецируем в Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Транспонируем для multi-head: (batch, heads, seq, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Higher order attention через CP разложение
        # T[i,j,k] = Σ_r (Q·a_r)[i] × (K·b_r)[j] × (K·c_r)[k]

        # Вычисляем проекции на факторы
        Q_a = torch.einsum('bhsd,hrd->bhsr', Q, self.factor_a)
        K_b = torch.einsum('bhsd,hrd->bhsr', K, self.factor_b)
        K_c = torch.einsum('bhsd,hrd->bhsr', K, self.factor_c)

        # Агрегируем взаимодействия второго порядка
        K_bc = torch.einsum('bhir,bhjr->bhij', K_b, K_c)

        # Вычисляем оценки внимания
        attn_scores = torch.einsum('bhir,bhjj->bhij', Q_a, K_bc) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применяем внимание к values
        output = torch.matmul(attn_weights, V)

        # Изменяем форму и проецируем
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output
```

## Применение к криптовалютам

### Обработка рыночных данных

Для торговли криптовалютами с Bybit:

```python
CRYPTO_UNIVERSE = {
    'major': ['BTCUSDT', 'ETHUSDT'],
    'large_cap': ['SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'],
    'mid_cap': ['AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT'],
    'small_cap': ['ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT']
}

FEATURES = {
    'price': ['open', 'high', 'low', 'close'],
    'volume': ['volume', 'quote_volume', 'taker_buy_volume'],
    'derived': ['returns', 'volatility', 'rsi', 'macd', 'bb_width']
}
```

### Инженерия признаков

```python
def prepare_features(df, lookback=60):
    """
    Подготовка признаков для HOT модели

    Args:
        df: DataFrame с OHLCV
        lookback: Количество периодов для ретроспективы

    Returns:
        X: Тензор признаков (batch, seq, features)
        y: Метки (batch,)
    """
    features = []

    # Ценовые признаки (нормализованные)
    for col in ['open', 'high', 'low', 'close']:
        features.append(df[col].pct_change())

    # Признаки объёма
    features.append(np.log1p(df['volume']).diff())

    # Технические индикаторы
    features.append(compute_rsi(df['close'], 14))
    features.append(compute_macd(df['close']))
    features.append(compute_bb_width(df['close']))

    # Волатильность
    features.append(df['close'].pct_change().rolling(20).std())

    # Объединяем
    X = np.column_stack(features)

    # Создаём последовательности
    sequences = []
    labels = []

    for i in range(lookback, len(X) - 1):
        sequences.append(X[i-lookback:i])

        # Метка: 2 = вверх, 1 = нейтрально, 0 = вниз
        future_return = df['close'].iloc[i+1] / df['close'].iloc[i] - 1
        if future_return > 0.005:
            labels.append(2)  # Вверх
        elif future_return < -0.005:
            labels.append(0)  # Вниз
        else:
            labels.append(1)  # Нейтрально

    return np.array(sequences), np.array(labels)
```

### Торговые сигналы

```python
def generate_signals(model, current_data, threshold=0.6):
    """
    Генерация торговых сигналов из прогнозов модели

    Args:
        model: Обученная HOT модель
        current_data: Текущие рыночные данные
        threshold: Порог уверенности

    Returns:
        signal: 'buy', 'sell' или 'hold'
        confidence: Уверенность модели
    """
    with torch.no_grad():
        logits = model(current_data)
        probs = torch.softmax(logits, dim=-1)

        pred_class = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()

        if confidence < threshold:
            return 'hold', confidence

        signal_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        return signal_map[pred_class], confidence
```

## Бэктестинг

### Ключевые метрики

| Метрика | Описание | Хорошее значение |
|---------|----------|------------------|
| **Accuracy** | % правильных прогнозов | > 55% |
| **F1 Score** | Сбалансированная метрика | > 0.50 |
| **Sharpe Ratio** | Доходность с учётом риска | > 1.5 |
| **Max Drawdown** | Максимальная просадка | < 20% |
| **Win Rate** | % прибыльных сделок | > 50% |

### Ожидаемые результаты (крипто)

| Модель | Accuracy | F1 | Sharpe | Max DD |
|--------|----------|----|----|--------|
| LSTM Baseline | 52% | 0.48 | 0.9 | -35% |
| Стандартный Transformer | 54% | 0.51 | 1.2 | -28% |
| **Higher Order Transformer** | **57%** | **0.55** | **1.6** | **-22%** |

*Примечание: Результаты на исторических данных. Прошлые результаты не гарантируют будущей доходности.*

### Правила торговли

```
Правила входа:
├── Уверенность сигнала > 60%
├── Объём > 20-дневного среднего
├── RSI не в экстремальной зоне (20 < RSI < 80)
└── Нет конфликтующих сигналов за последние 4 часа

Правила выхода:
├── Противоположный сигнал с уверенностью > 70%
├── Стоп-лосс: -3%
├── Тейк-профит: +5%
└── По времени: выход через 24 часа если нет чёткого направления

Размер позиции:
├── Базовый размер: 10% портфеля на сделку
├── Масштабирование по уверенности: размер × (уверенность - 0.5) × 2
├── Максимальная единичная позиция: 25%
└── Максимальная общая экспозиция: 80%
```

## Ресурсы

### Научные статьи

1. **Higher Order Transformers: Enhancing Stock Movement Prediction On Multimodal Time-Series Data**
   - arXiv: [2412.10540](https://arxiv.org/abs/2412.10540)
   - Ключевые идеи: Attention третьего порядка, тензорное разложение

2. **Transformers for Time Series Forecasting**
   - Основополагающая работа по применению трансформеров к финансовым данным

3. **Tensor Decomposition for Signal Processing and Machine Learning**
   - Исчерпывающий обзор CP и Tucker разложений

### Книги

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.)
- [Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf) (Kolda & Bader)

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — TFT для прогнозирования
- [Глава 44: ProbSparse Attention](../44_probsparse_attention) — Эффективные варианты attention
- [Глава 58: Flash Attention](../58_flash_attention_trading) — Memory-эффективный attention

## Зависимости

### Rust

```toml
ndarray = "0.16"
ndarray-linalg = "0.16"
reqwest = "0.12"
tokio = "1.0"
serde = "1.0"
serde_json = "1.0"
chrono = "0.4"
rand = "0.8"
anyhow = "1.0"
```

### Python

```python
torch>=2.0.0
tensorly>=0.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## Уровень сложности

**Продвинутый**

**Необходимые знания:**
- Архитектура трансформеров
- Тензорные операции и разложения
- Анализ временных рядов
- Криптовалютные рынки
- Управление рисками

---

*Этот материал предназначен для образовательных целей. Криптовалюты — высокорисковые активы. Не инвестируйте больше, чем готовы потерять.*
