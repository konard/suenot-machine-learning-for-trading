# Глава 341: Graph Transformer для трейдинга

## Обзор

Graph Transformers объединяют мощь графовых нейронных сетей (GNN) с механизмами внимания Transformer для моделирования сложных реляционных структур на финансовых рынках. В отличие от традиционных моделей временных рядов, которые рассматривают активы независимо, Graph Transformers фиксируют межактивные зависимости, рыночные корреляции и структурные паттерны, возникающие в криптовалютной экосистеме.

## Почему Graph Transformers для трейдинга?

### Проблема традиционных подходов

Традиционные ML модели для трейдинга (LSTM, GRU, стандартные Transformers) рассматривают каждый актив как независимый временной ряд. Однако финансовые рынки по своей природе **реляционны**:

- **Корреляции**: Движения BTC влияют на ETH, что влияет на другие альткоины
- **Секторные связи**: DeFi токены движутся вместе, как и Layer-2 решения
- **Структура рынка**: Потоки ордеров бирж, кошельки китов, on-chain транзакции образуют граф
- **Распространение информации**: Новости/события распространяются через взаимосвязанные рынки

### Решение Graph Transformer

Graph Transformers моделируют эти отношения явно:

```
Традиционный: X_t = f(X_{t-1}, X_{t-2}, ...) для каждого актива независимо

Graph Transformer: X_t = f(X_{t-1}, A, E)
где:
  A = матрица смежности (какие активы связаны)
  E = признаки рёбер (сила/тип связей)
```

## Техническая архитектура

### 1. Построение графа для криптовалютных рынков

```
Структура рыночного графа:
├── Узлы: Отдельные активы (BTC, ETH, SOL, ...)
│   └── Признаки узлов: Цена, объём, волатильность, метрики order book
├── Рёбра: Отношения между активами
│   ├── Корреляционные рёбра (скользящая корреляция > порог)
│   ├── Секторные рёбра (одна категория: DeFi, L2, Meme, ...)
│   ├── On-chain рёбра (переводы токенов, DEX свапы)
│   └── Order flow рёбра (паттерны кросс-биржевого арбитража)
└── Глобальные признаки: Общий sentiment рынка, суммарный объём, доминация
```

### 2. Слой Graph Transformer

Ключевая инновация объединяет структуру графа с self-attention:

```
Стандартный Transformer:
  Attention(Q, K, V) = softmax(QK^T / √d) V

Graph Transformer:
  Attention(Q, K, V, A, E) = softmax((QK^T + bias(A, E)) / √d) V

где:
  - bias(A, E) кодирует структуру графа в attention scores
  - Не-соседи могут иметь attention = 0 (разреженное внимание)
  - Признаки рёбер E модулируют силу внимания
```

### 3. Позиционное кодирование для графов

В отличие от последовательностей, графы не имеют естественных позиций. Мы используем:

- **Laplacian Positional Encoding (LPE)**: Собственные векторы Лапласиана графа
- **Random Walk Positional Encoding (RWPE)**: Вероятности посадки из случайных блужданий
- **Centrality Encoding**: Важность узла (степень, PageRank, betweenness)

```python
# Laplacian Positional Encoding
L = D - A  # Матрица Лапласиана
eigenvalues, eigenvectors = eig(L)
pos_encoding = eigenvectors[:, 1:k+1]  # Первые k нетривиальных собственных вектора
```

## Архитектура модели

```
┌─────────────────────────────────────────────────────────────────┐
│                    МОДЕЛЬ GRAPH TRANSFORMER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ВХОДНОЙ СЛОЙ                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Признаки узлов (на актив):                                │   │
│  │   - Доходности (1м, 5м, 15м, 1ч, 4ч)                     │   │
│  │   - Профиль объёма (соотношение buy/sell, отклонение VWAP)│   │
│  │   - Order book (bid-ask спред, дисбаланс глубины)         │   │
│  │   - Технические индикаторы (RSI, MACD, Bollinger)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ПОЗИЦИОННОЕ КОДИРОВАНИЕ                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Laplacian PE + Random Walk PE + Centrality Encoding       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  БЛОКИ GRAPH TRANSFORMER (×N)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Multi-Head Graph Attention                          │   │   │
│  │ │   - Query/Key/Value проекции                        │   │   │
│  │ │   - Edge-aware attention bias                       │   │   │
│  │ │   - Разреженное внимание (по графу)                 │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Feed-Forward Network                                │   │   │
│  │ │   - Linear → GELU → Linear                          │   │   │
│  │ │   - Residual connections                            │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Edge Update (опционально)                           │   │   │
│  │ │   - Обновление признаков рёбер из представлений узлов│   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ВЫХОДНЫЕ ГОЛОВЫ                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Per-Node: Предсказание направления (вверх/вниз/нейтрал)   │   │
│  │ Per-Node: Предсказание величины доходности               │   │
│  │ Per-Edge: Предсказание изменения корреляции              │   │
│  │ Global: Классификация режима рынка                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Торговая стратегия

### Генерация сигналов

```python
def generate_signals(model, graph):
    # Forward pass через Graph Transformer
    node_embeddings = model(graph)

    # Предсказания для каждого актива
    direction_probs = model.direction_head(node_embeddings)  # [N, 3]
    return_preds = model.return_head(node_embeddings)        # [N, 1]

    signals = []
    for i, asset in enumerate(graph.nodes):
        prob_up = direction_probs[i, 0]
        prob_down = direction_probs[i, 2]
        expected_return = return_preds[i]

        if prob_up > 0.6 and expected_return > 0.005:
            signals.append(Signal(asset, "LONG", confidence=prob_up))
        elif prob_down > 0.6 and expected_return < -0.005:
            signals.append(Signal(asset, "SHORT", confidence=prob_down))

    return signals
```

### Построение портфеля

Graph Transformers позволяют строить портфель с учётом структуры графа:

```python
def construct_portfolio(signals, graph, node_embeddings):
    # Используем структуру графа для диверсификации
    selected_assets = []

    for signal in sorted(signals, key=lambda s: -s.confidence):
        asset = signal.asset

        # Проверяем, не слишком ли коррелирован с уже выбранными активами
        correlations = get_correlations(asset, selected_assets, graph)
        if max(correlations) < 0.7:  # Ограничение диверсификации
            selected_assets.append(asset)

    # Взвешиваем по confidence и центральности в графе
    weights = calculate_weights(selected_assets, signals, node_embeddings)
    return Portfolio(selected_assets, weights)
```

## Ключевые компоненты

### 1. Multi-Head Graph Attention

```python
class GraphAttention(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        # x: [N, d] признаки узлов
        # edge_index: [2, E] связи рёбер
        # edge_attr: [E, d_e] признаки рёбер

        Q = self.W_q(x)  # [N, d]
        K = self.W_k(x)  # [N, d]
        V = self.W_v(x)  # [N, d]

        # Вычисляем attention scores для связанных узлов
        src, dst = edge_index
        scores = (Q[dst] * K[src]).sum(dim=-1) / sqrt(d)  # [E]

        # Добавляем edge bias
        edge_bias = self.edge_proj(edge_attr).squeeze()  # [E]
        scores = scores + edge_bias

        # Sparse softmax (только по соседям)
        attn_weights = sparse_softmax(scores, dst, num_nodes=N)

        # Агрегация
        out = scatter_add(attn_weights.unsqueeze(-1) * V[src], dst, dim=0)
        return out
```

### 2. Обновление признаков рёбер

```python
class EdgeUpdate(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        # Объединяем признаки исходного и целевого узлов
        edge_features = torch.cat([
            x[src],
            x[dst],
            edge_attr,
            x[src] - x[dst],  # Разность
            x[src] * x[dst],  # Взаимодействие
        ], dim=-1)

        # Обновляем признаки рёбер
        new_edge_attr = self.mlp(edge_features)
        return new_edge_attr
```

### 3. Graph Pooling для глобальных предсказаний

```python
class GraphPooling(nn.Module):
    def forward(self, x, batch):
        # Attention-based pooling
        attn_scores = self.attention(x)  # [N, 1]
        attn_weights = softmax(attn_scores, batch)

        # Взвешенная сумма по графу
        global_repr = scatter_add(attn_weights * x, batch, dim=0)
        return global_repr
```

## Детали реализации

### Требования к данным

```
Данные криптовалютного рынка:
├── OHLCV данные (минимум 1-минутное разрешение)
│   └── Множество активов (BTC, ETH, SOL, AVAX, ...)
├── Снимки order book (L2 данные)
│   └── Уровни Bid/Ask с объёмами
├── Данные о сделках
│   └── Отдельные сделки с временными метками
└── On-chain данные (опционально, но ценны)
    ├── Движения кошельков китов
    ├── Притоки/оттоки с бирж
    └── Объёмы торговли на DEX

Данные для построения графа:
├── Скользящие корреляции (30-дневное окно)
├── Классификации секторов
├── Рейтинги по рыночной капитализации
└── Связи торговых пар
```

### Feature Engineering

```python
features = {
    # Ценовые признаки (на узел)
    'returns_1m': log_return(close, 1),
    'returns_5m': log_return(close, 5),
    'returns_15m': log_return(close, 15),
    'returns_1h': log_return(close, 60),
    'volatility_1h': rolling_std(returns, 60),

    # Признаки объёма
    'volume_ratio': volume / volume_ma_20,
    'buy_sell_ratio': buy_volume / (buy_volume + sell_volume),
    'vwap_deviation': (close - vwap) / vwap,

    # Признаки order book
    'spread_bps': (ask - bid) / mid * 10000,
    'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth),
    'ofi': order_flow_imbalance(book_changes),

    # Технические индикаторы
    'rsi_14': rsi(close, 14),
    'macd_signal': macd(close) - macd_signal(close),
    'bb_position': (close - bb_lower) / (bb_upper - bb_lower),

    # Графовые признаки
    'degree_centrality': graph.degree(node),
    'pagerank': graph.pagerank(node),
    'clustering_coef': graph.clustering(node),
}
```

### Конфигурация обучения

```yaml
model:
  num_layers: 6
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  edge_dim: 64
  use_edge_features: true
  positional_encoding: "laplacian"
  num_pe_dims: 16

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_steps: 1000
  max_epochs: 100
  early_stopping_patience: 10

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  sequence_length: 60  # 1 час 1-минутных данных
  prediction_horizon: 5  # На 5 минут вперёд
```

## Ключевые метрики

### Производительность модели

- **Node-level Accuracy**: Точность классификации на актив
- **Direction Accuracy**: % правильных предсказаний вверх/вниз
- **Information Coefficient (IC)**: Корреляция между предсказанными и реальными доходностями
- **Graph-aware IC**: IC с учётом корреляций активов

### Торговые показатели

- **Sharpe Ratio**: Доходность с поправкой на риск (цель > 2.0)
- **Sortino Ratio**: Доходность с поправкой на downside риск
- **Maximum Drawdown**: Максимальная просадка
- **Win Rate**: % прибыльных сделок
- **Profit Factor**: Валовая прибыль / Валовые убытки

## Преимущества Graph Transformers

| Аспект | Традиционные модели | Graph Transformers |
|--------|--------------------|--------------------|
| Связи активов | Игнорируются или ручной инжиниринг | Обучаются автоматически |
| Распространение информации | Нет | Естественно через message passing |
| Определение режима рынка | Нужна отдельная модель | Встроено через global pooling |
| Изменения корреляций | Статичные предположения | Динамические, обучаемые |
| Масштабируемость | Линейно по активам | Может быть разреженной (эффективно) |
| Интерпретируемость | Ограничена | Веса внимания = объяснения |

## Сравнение с другими подходами

### vs. Стандартные Transformers

- **Стандартные**: Рассматривают активы как "токены" в последовательности
- **Graph Transformer**: Явно кодирует связи между активами

### vs. GCN/GAT

- **GCN/GAT**: Фиксированные паттерны агрегации
- **Graph Transformer**: Гибкое внимание по всему графу + структурный bias

### vs. Temporal Fusion Transformer

- **TFT**: Только временное внимание
- **Graph Transformer**: И временное, и кросс-активное внимание

## Продакшн соображения

```
Pipeline инференса:
├── Сбор данных (Bybit WebSocket)
│   └── Real-time OHLCV + обновления order book
├── Обновление графа (каждые N минут)
│   └── Пересчёт корреляций, обновление рёбер
├── Вычисление признаков
│   └── Векторизованное вычисление признаков
├── Инференс модели
│   └── GPU-ускоренный forward pass
├── Генерация сигналов
│   └── Извлечение сигналов по порогам
└── Исполнение ордеров
    └── API интеграция с риск-менеджментом

Бюджет задержки:
├── Сбор данных: ~10мс (WebSocket)
├── Вычисление признаков: ~5мс (Rust)
├── Построение графа: ~20мс (каждые 5 мин)
├── Инференс модели: ~15мс (GPU)
├── Генерация сигналов: ~1мс
└── Итого: ~50мс (без учёта исполнения)
```

## Структура директории

```
341_graph_transformer_trading/
├── README.md                    # Английская версия
├── README.ru.md                 # Этот файл
├── readme.simple.md             # Объяснение для начинающих
├── readme.simple.ru.md          # Русская версия для начинающих
└── rust_graph_transformer/      # Реализация на Rust
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Точка входа библиотеки
    │   ├── api/                 # Клиент Bybit API
    │   ├── graph/               # Построение графа и операции
    │   ├── transformer/         # Реализация Graph Transformer
    │   ├── features/            # Feature engineering
    │   ├── strategy/            # Торговая стратегия
    │   └── backtest/            # Движок бэктестинга
    └── examples/
        ├── fetch_market_data.rs
        ├── build_market_graph.rs
        ├── train_model.rs
        └── live_trading.rs
```

## Ссылки

1. **A Generalization of Transformer Networks to Graphs** (Dwivedi & Bresson, 2020)
   - https://arxiv.org/abs/2012.09699

2. **Do Transformers Really Perform Bad for Graph Representation?** (Ying et al., 2021)
   - https://arxiv.org/abs/2106.05234 (Graphormer)

3. **Recipe for a General, Powerful, Scalable Graph Transformer** (Rampášek et al., 2022)
   - https://arxiv.org/abs/2205.12454 (GPS)

4. **Graph Neural Networks for Financial Market Prediction** (Различные авторы)
   - Применения к рынкам акций/криптовалют

5. **Temporal Graph Networks** (Rossi et al., 2020)
   - https://arxiv.org/abs/2006.10637

## Уровень сложности

**Эксперт** - Требуется понимание:
- Графовых нейронных сетей
- Архитектуры Transformer
- Микроструктуры финансовых рынков
- PyTorch/тензорных операций
- Распределённого обучения (для больших графов)

## Дисклеймер

Эта глава предназначена **только для образовательных целей**. Торговля криптовалютами несёт существенный риск. Описанные здесь стратегии не были проверены в реальной торговле и должны быть тщательно протестированы перед любым применением в реальном мире. Прошлые результаты не гарантируют будущих результатов.
