# Глава 114: Визуализация Attention для трейдинга

## Обзор

Визуализация Attention — это мощная техника интерпретируемости для понимания того, как модели на основе Transformer принимают решения на финансовых рынках. Визуализируя веса внимания, выученные во время обучения, трейдеры и исследователи могут понять, какие исторические временные шаги, признаки или рыночные события модель считает наиболее важными при генерации торговых сигналов.

В этой главе реализуются методы визуализации attention для интерпретации Transformer-моделей, применяемых к торговле криптовалютами на Bybit и прогнозированию фондового рынка, обеспечивая прозрачные и объяснимые стратегии на основе ИИ.

## Содержание

1. [Теоретические основы](#теоретические-основы)
2. [Механизм Attention подробно](#механизм-attention-подробно)
3. [Техники визуализации](#техники-визуализации)
4. [Реализация](#реализация)
5. [Торговая стратегия](#торговая-стратегия)
6. [Результаты и метрики](#результаты-и-метрики)
7. [Литература](#литература)

## Теоретические основы

### Механизм Attention

Механизм self-attention, представленный в "Attention Is All You Need" (Vaswani et al., 2017), вычисляет взвешенную сумму значений на основе совместимости между запросами и ключами:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Где:
- `Q ∈ ℝ^{n×d_k}` — матрица запросов (queries)
- `K ∈ ℝ^{n×d_k}` — матрица ключей (keys)
- `V ∈ ℝ^{n×d_v}` — матрица значений (values)
- `d_k` — размерность ключей (масштабирующий фактор)
- `n` — длина последовательности

Веса внимания `A = softmax(QK^T / sqrt(d_k))` формируют матрицу `n × n`, где `A[i,j]` представляет, насколько позиция `i` обращает внимание на позицию `j`.

### Multi-Head Attention

Multi-head attention запускает `h` параллельных функций внимания, каждая со своими обученными проекциями:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

где head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

Каждая голова может научиться обращать внимание на разные паттерны:
- **Голова 1**: Краткосрочный моментум (недавние цены)
- **Голова 2**: Кластеры волатильности (периоды высокого объёма)
- **Голова 3**: Долгосрочные тренды (далёкий исторический контекст)

### Почему визуализация важна для трейдинга

| Аспект | Преимущество |
|--------|--------------|
| Прозрачность | Понимание *почему* был сгенерирован торговый сигнал |
| Управление рисками | Определение, не полагается ли модель на ложные корреляции |
| Feature Engineering | Обнаружение, какие входы влияют на предсказания |
| Определение режимов | Наблюдение за изменением внимания в разных рыночных условиях |
| Отладка | Выявление сбоев модели (например, коллапс внимания) |

## Механизм Attention подробно

### Паттерны Attention в финансовых данных

При применении attention к финансовым временным рядам появляются характерные паттерны:

1. **Локальное внимание**: Сильные веса на недавних временных шагах (краткосрочный моментум)
2. **Периодическое внимание**: Веса на том же времени дня или дне недели (сезонность)
3. **Событийное внимание**: Пики в точках высокого объёма или волатильности
4. **Возвратное внимание**: Фокус на точках экстремального отклонения

### Вычисление Attention Score

Для Transformer-энкодера, обрабатывающего OHLCV данные:

```python
# Вход: (batch, seq_len, features) где features = [open, high, low, close, volume]
# После эмбеддинга: (batch, seq_len, d_model)

# Attention scores для одной головы
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

# attention_weights[b, i, j] = насколько позиция i обращает внимание на позицию j
```

### Свойства весов Attention

1. **Построчная сумма равна 1**: Внимание каждой позиции на все остальные суммируется в 1
2. **Неотрицательность**: Все веса ≥ 0 (благодаря softmax)
3. **Обучаемые паттерны**: Веса определяются выученными Q, K проекциями

## Техники визуализации

### 1. Тепловые карты Attention

Наиболее прямая визуализация: отображение матрицы attention как 2D тепловой карты.

```
Позиция (Query)
    ↓
    |  t-4   t-3   t-2   t-1   t
----+-----------------------------
t-4 | 0.05  0.10  0.15  0.20  0.50  ← Позиция (Key)
t-3 | 0.10  0.15  0.20  0.25  0.30
t-2 | 0.15  0.20  0.25  0.20  0.20
t-1 | 0.10  0.15  0.30  0.25  0.20
t   | 0.05  0.10  0.20  0.25  0.40
```

### 2. Поток Attention (в стиле BertViz)

Визуализация attention как связей между входными токенами/временными шагами:

```
Время:  t-4    t-3    t-2    t-1    t
        │      │      │      │      │
        │      │      ├──────┼──────┤ (сильная)
        │      │      │      ├──────┤ (средняя)
        │      │      │      │      │
       [P]    [P]    [P]    [P]    [P] ← Предсказания
```

### 3. Анализ по головам

Сравнение паттернов attention по разным головам:

| Голова | Паттерн | Торговая интерпретация |
|--------|---------|------------------------|
| 1 | Диагональный (локальный) | Недавний ценовой моментум |
| 2 | Вертикальные полосы | Фокус на определённом времени дня |
| 3 | Разреженный, высокий объём | Реакция на события/новости |
| 4 | Равномерный | Базовое/усреднение |

### 4. Агрегация по слоям

Объединение attention по слоям с помощью:
- **Attention Rollout**: Умножение матриц attention по слоям
- **Attention Flow**: Учёт остаточных связей

```
A_combined = A_L * A_{L-1} * ... * A_1
```

### 5. Атрибуция признаков через Attention

Взвешивание входных признаков по агрегированному attention:

```python
# attention_weights: (batch, heads, seq_len, seq_len)
# Усреднение по головам
avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)

# Сумма attention, полученного каждой позицией
position_importance = avg_attention.sum(dim=1)  # (batch, seq_len)

# Умножение на входные признаки для атрибуции
feature_attribution = input_features * position_importance.unsqueeze(-1)
```

## Реализация

### Python

Реализация на Python использует PyTorch и включает:

- **`python/model.py`**: Transformer-модель с извлечением весов attention
- **`python/visualization.py`**: Тепловые карты attention, диаграммы потоков и инструменты анализа
- **`python/data_loader.py`**: Загрузка данных фондового рынка (yfinance) и криптовалютного (Bybit)
- **`python/backtest.py`**: Фреймворк бэктестинга с метриками интерпретируемости

```python
from python.model import AttentionTransformer

model = AttentionTransformer(
    input_dim=8,          # OHLCV + технические индикаторы
    d_model=64,
    n_heads=4,
    n_layers=3,
    output_dim=3,         # доходность, направление, волатильность
    dropout=0.1,
    return_attention=True  # Включить извлечение attention
)

# Прямой проход с весами attention
predictions, attention_weights = model(features)
# attention_weights: словарь с ключами 'layer_0', 'layer_1', ...
# Каждое значение: (batch, heads, seq_len, seq_len)
```

### Rust

Реализация на Rust предоставляет продакшн-версию:

- **`src/model/`**: Transformer с отслеживанием attention
- **`src/visualization/`**: Эффективная агрегация attention
- **`src/data/`**: Клиент API Bybit и инженерия признаков
- **`src/trading/`**: Генерация сигналов с уверенностью на основе attention
- **`src/backtest/`**: Движок оценки производительности

```bash
# Запуск базового примера
cargo run --example basic_attention

# Запуск анализа визуализации
cargo run --example attention_analysis

# Запуск торговой стратегии
cargo run --example trading_strategy
```

## Торговая стратегия

### Оценка уверенности на основе Attention

Использование паттернов attention для оценки уверенности предсказания:

```python
def compute_attention_confidence(attention_weights):
    """
    Высокая уверенность: attention сфокусирован (низкая энтропия)
    Низкая уверенность: attention размыт (высокая энтропия)
    """
    # Вычисление энтропии распределения attention
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
    max_entropy = math.log(attention_weights.shape[-1])

    # Нормализация к [0, 1], инверсия так что сфокусированный = высокая уверенность
    confidence = 1 - (entropy / max_entropy)
    return confidence.mean()
```

### Генерация сигналов с интерпретируемостью

```python
class AttentionTradingStrategy:
    def generate_signal(self, model, features):
        pred_return, pred_direction, pred_vol, attention = model(features)

        # Вычисление уверенности на основе attention
        confidence = self.compute_attention_confidence(attention)

        # Торговать только когда attention сфокусирован (уверенность высокая)
        if confidence < self.confidence_threshold:
            return Signal.HOLD

        # Стандартный направленный сигнал
        if pred_direction > 0.6 and pred_return > self.return_threshold:
            return Signal.LONG
        elif pred_direction < 0.4 and pred_return < -self.return_threshold:
            return Signal.SHORT
        return Signal.HOLD
```

### Управление рисками через анализ Attention

- **Обнаружение коллапса Attention**: Если attention становится равномерным, модель может быть неуверена
- **Обнаружение смены режима**: Резкое изменение паттернов attention сигнализирует о новом рыночном режиме
- **Отслеживание важности признаков**: Мониторинг того, какие входы влияют на решения со временем

## Результаты и метрики

### Метрики оценки

| Метрика | Описание |
|---------|----------|
| MSE / MAE | Точность предсказания доходности |
| Accuracy / F1 | Качество классификации направления |
| Sharpe Ratio | Доходность с поправкой на риск |
| Sortino Ratio | Доходность с поправкой на нисходящий риск |
| Maximum Drawdown | Худшее падение от пика до дна |
| Attention Entropy | Мера фокусировки attention |
| Attention Stability | Стабильность паттернов во времени |

### Анализ интерпретируемости

1. **Специализация голов**: Измерение различий в attention каждой головы
2. **Временная локальность**: Количественная оценка фокуса на недавнем vs. далёком прошлом
3. **Корреляция атрибуции**: Сравнение атрибуции на основе attention с SHAP/LIME

### Сравнение с базовыми моделями

Стратегия визуализации attention сравнивается с:
- LSTM без attention
- Transformer без фильтрации на основе уверенности attention
- Случайный baseline
- Стратегия Buy-and-Hold

## Структура проекта

```
114_attention_visualization_trading/
├── README.md                  # Основной README (English)
├── README.ru.md               # Этот файл
├── readme.simple.md           # Упрощённое объяснение (English)
├── readme.simple.ru.md        # Упрощённое объяснение (Russian)
├── Cargo.toml                 # Конфигурация проекта Rust
├── python/
│   ├── __init__.py
│   ├── model.py               # Transformer с извлечением attention
│   ├── visualization.py       # Инструменты визуализации attention
│   ├── data_loader.py         # Загрузка данных акций и крипто
│   ├── backtest.py            # Фреймворк бэктестинга
│   └── requirements.txt       # Зависимости Python
├── src/
│   ├── lib.rs                 # Корень библиотеки Rust
│   ├── model/
│   │   ├── mod.rs             # Модуль модели
│   │   └── transformer.rs     # Реализация Transformer
│   ├── data/
│   │   ├── mod.rs             # Модуль данных
│   │   ├── bybit.rs           # Клиент API Bybit
│   │   └── features.rs        # Инженерия признаков
│   ├── trading/
│   │   ├── mod.rs             # Модуль торговли
│   │   ├── signals.rs         # Генерация сигналов
│   │   └── strategy.rs        # Торговая стратегия
│   └── backtest/
│       ├── mod.rs             # Модуль бэктестинга
│       └── engine.rs          # Движок бэктестинга
└── examples/
    ├── basic_attention.rs     # Базовая визуализация attention
    ├── attention_analysis.rs  # Комплексный анализ
    └── trading_strategy.rs    # Полная торговая стратегия
```

## Литература

1. **Vaswani, A., et al. (2017)**. Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

2. **Kobayashi, G., Kuribayashi, T., Yokoi, S., & Inui, K. (2020)**. Attention is Not Only a Weight: Analyzing Transformers with Vector Norms. *EMNLP 2020*. https://arxiv.org/abs/2004.10102

3. **Abnar, S., & Zuidema, W. (2020)**. Quantifying Attention Flow in Transformers. *ACL 2020*. https://arxiv.org/abs/2005.00928

4. **Vig, J. (2019)**. A Multiscale Visualization of Attention in the Transformer Model. *ACL 2019 System Demonstrations*. https://arxiv.org/abs/1906.05714

5. **Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019)**. What Does BERT Look At? An Analysis of BERT's Attention. *BlackboxNLP 2019*. https://arxiv.org/abs/1906.04341

6. **De Prado, M. L. (2018)**. Advances in Financial Machine Learning. *Wiley*.

## Лицензия

MIT
