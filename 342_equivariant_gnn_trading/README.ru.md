# Глава 342: Эквивариантные графовые нейронные сети для торговли криптовалютами

## Обзор

Эквивариантные графовые нейронные сети (E-GNN) представляют собой парадигмальный сдвиг в геометрическом глубоком обучении, спроектированные для уважения симметрий, присущих данным. В торговле криптовалютами активы формируют сложные реляционные структуры, где движения цен, корреляции и динамика рынка демонстрируют геометрические свойства. E-GNN сохраняют эти симметрии в процессе обучения, обеспечивая более устойчивое распознавание паттернов и улучшенную обобщаемость в различных рыночных режимах.

Ключевое понимание заключается в том, что финансовые рынки имеют присущие им симметрии: инвариантность масштаба (удвоение всех цен не меняет относительную доходность), эквивариантность перестановок (порядок активов не должен влиять на прогнозы), и инвариантность вращения в пространстве признаков (коррелированные активы должны вести себя одинаково независимо от абсолютных значений их характеристик).

## Торговая стратегия

**Основная концепция:** Моделирование рынка криптовалют как динамического графа, где узлы представляют активы, рёбра кодируют корреляции и торговые взаимосвязи, а характеристики узлов/рёбер отражают микроструктуру рынка. E-GNN обучается генерировать торговые сигналы, соблюдая геометрическую структуру этого финансового графа.

**Ключевые преимущества для торговли:**
1. **Сохранение симметрий** — Модель обучается паттернам, инвариантным к нерелевантным трансформациям (масштаб, перестановки)
2. **Захват корреляций** — Структура графа естественно кодирует корреляции между активами и рыночные режимы
3. **Геометрические признаки** — Эмбеддинги сохраняют отношения расстояний в пространстве признаков
4. **Устойчивость** — Эквивариантность обеспечивает встроенную регуляризацию против ложных паттернов

**Преимущество:** E-GNN фильтруют шум от выбора системы координат, фокусируясь на подлинных рыночных паттернах, сохраняющихся в разных представлениях.

## Техническая спецификация

### Эквивариантность в финансах

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    E(n) Эквивариантность в торговле                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Что такое эквивариантность?                                           │
│  ──────────────────────────                                             │
│                                                                         │
│  Функция f является G-эквивариантной, если:                            │
│                                                                         │
│     f(g · x) = g · f(x)  для всех g ∈ G                                │
│                                                                         │
│  Трансформация входа и применение f = Применение f и трансформация     │
│                                                                         │
│                                                                         │
│  Финансовые симметрии:                                                 │
│  ───────────────────                                                   │
│                                                                         │
│  1. Эквивариантность перестановок:                                      │
│     - Переупорядочение активов не должно менять прогнозы               │
│     - BTC, ETH, SOL → SOL, BTC, ETH даёт те же относительные сигналы   │
│                                                                         │
│  2. Инвариантность масштаба:                                           │
│     - Важны доходности, а не абсолютные цены                           │
│     - $100 → $110 тот же паттерн, что и $1000 → $1100                  │
│                                                                         │
│  3. Инвариантность сдвига:                                             │
│     - Важны относительные позиции, не абсолютные                       │
│     - Прогнозы определяются разницами признаков, не уровнями           │
│                                                                         │
│  4. Инвариантность вращения (в пространстве признаков):                │
│     - Кластеры коррелированных активов сохраняют структуру             │
│     - Вращение PCA не должно менять прогнозы                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Обзор архитектуры

```
┌─────────────────────────────────────────────────────────────────────────┐
│         Эквивариантная GNN для торговли криптовалютами                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Вход: Граф активов G = (V, E)                                          │
│  ─────────────────────────────                                          │
│  • Узлы V: Криптовалютные активы (BTC, ETH, SOL, ...)                  │
│  • Рёбра E: Связи корреляции/каузальности между активами               │
│  • Признаки узлов h_i: Цена, объём, волатильность, моментум            │
│  • Координаты x_i: Позиция в пространстве эмбеддингов                  │
│                                                                         │
│                    ┌─────────────────────┐                              │
│                    │  Построение графа   │                              │
│                    │      активов        │                              │
│                    └──────────┬──────────┘                              │
│                               │                                         │
│                               ▼                                         │
│              ┌────────────────────────────────┐                         │
│              │   Слой входного эмбеддинга     │                         │
│              │   h_i → (h_i, x_i)             │                         │
│              └───────────────┬────────────────┘                         │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                      │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │  E-GNN      │     │  E-GNN      │     │  E-GNN      │              │
│   │  Слой 1     │────▶│  Слой 2     │────▶│  Слой L     │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                  │                      │
│                                                  ▼                      │
│                              ┌────────────────────────────┐             │
│                              │   Инвариантная агрегация   │             │
│                              │   (Пулинг на уровне графа) │             │
│                              └───────────────┬────────────┘             │
│                                              │                          │
│                      ┌───────────────────────┼───────────────────────┐  │
│                      │                       │                       │  │
│                      ▼                       ▼                       ▼  │
│               ┌────────────┐         ┌────────────┐         ┌──────────┐│
│               │ Прогноз    │         │ Размер     │         │ Оценка   ││
│               │ направления│         │ позиции    │         │ риска    ││
│               └────────────┘         └────────────┘         └──────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Математика слоя EGNN

Слой эквивариантной графовой нейронной сети обновляет как признаки узлов, так и координаты:

```python
class EGNNLayer:
    """
    E(n) Эквивариантный слой графовой нейронной сети

    Обновления:
    1. Сообщения: m_ij = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)
    2. Координаты: x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)
    3. Признаки: h_i' = φ_h(h_i, Σ_j m_ij)

    Ключ: Обновления координат используют относительные позиции (x_i - x_j),
          обеспечивая эквивариантность сдвига
    """

    def __init__(self, hidden_dim, edge_dim=0, act_fn=SiLU,
                 coords_agg='mean', update_coords=True):
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # MLP для рёбер: вычисляет сообщения
        self.edge_mlp = Sequential(
            Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            act_fn(),
            Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        # MLP для узлов: обновляет признаки узлов
        self.node_mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            act_fn(),
            Linear(hidden_dim, hidden_dim)
        )

        # MLP для координат: обновляет координаты
        if update_coords:
            self.coord_mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                act_fn(),
                Linear(hidden_dim, 1, bias=False)
            )

        self.coords_agg = coords_agg

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: Признаки узлов [N, hidden_dim]
        x: Координаты узлов [N, coord_dim]
        edge_index: Индексы рёбер [2, E]
        edge_attr: Признаки рёбер [E, edge_dim]
        """
        row, col = edge_index

        # Вычисляем квадраты расстояний (инвариантный скаляр)
        coord_diff = x[row] - x[col]  # [E, coord_dim]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Признаки рёбер
        edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Вычисляем сообщения
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Обновляем координаты (эквивариантно)
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # [E, 1]
            coord_update = coord_diff * coord_weights  # [E, coord_dim]

            # Агрегируем обновления координат
            x_new = x + scatter_mean(coord_update, row, dim=0, dim_size=x.size(0))
        else:
            x_new = x

        # Агрегируем сообщения
        m_i = scatter_sum(m_ij, row, dim=0, dim_size=h.size(0))  # [N, hidden_dim]

        # Обновляем признаки узлов
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_new = h + h_new  # Остаточное соединение

        return h_new, x_new
```

### Построение финансового графа

```python
class CryptoMarketGraph:
    """
    Строит динамический граф, представляющий структуру рынка криптовалют.

    Узлы: Отдельные активы (BTC, ETH, SOL и т.д.)
    Рёбра: На основе корреляции, каузальности или секторных связей
    """

    def __init__(self, correlation_threshold=0.3, window_size=168):
        self.corr_threshold = correlation_threshold
        self.window_size = window_size  # 1 неделя часовых данных

    def build_graph(self, returns_df, orderbook_features=None):
        """
        Строит рыночный граф из временных рядов доходностей.

        Args:
            returns_df: DataFrame с доходностями активов [время, активы]
            orderbook_features: Опциональные данные книги ордеров

        Returns:
            Граф с признаками узлов/рёбер и координатами
        """
        n_assets = len(returns_df.columns)

        # Признаки узлов: технические индикаторы для каждого актива
        node_features = self._compute_node_features(returns_df)

        # Начальные координаты: PCA-эмбеддинг доходностей
        coords = self._compute_initial_coords(returns_df)

        # Рёбра: на основе скользящей корреляции
        edge_index, edge_attr = self._build_edges(returns_df)

        return {
            'x': coords,
            'h': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'asset_names': list(returns_df.columns)
        }

    def _compute_node_features(self, returns_df):
        """Вычисляет признаки для каждого актива"""
        features = {}

        for asset in returns_df.columns:
            r = returns_df[asset]

            features[asset] = {
                'return_1h': r.iloc[-1],
                'return_24h': r.iloc[-24:].sum(),
                'return_7d': r.sum(),
                'volatility': r.std() * np.sqrt(24 * 365),
                'skewness': r.skew(),
                'kurtosis': r.kurtosis(),
                'momentum': r.ewm(span=12).mean().iloc[-1],
                'volume_zscore': self._compute_volume_zscore(asset),
                'rsi': self._compute_rsi(r, 14),
                'macd_signal': self._compute_macd_signal(r),
            }

        return self._to_tensor(features)

    def _compute_initial_coords(self, returns_df, dim=3):
        """Размещает активы в пространстве признаков с помощью PCA"""
        from sklearn.decomposition import PCA

        # Вычисляем матрицу корреляций
        corr_matrix = returns_df.corr().values

        # PCA-эмбеддинг
        pca = PCA(n_components=dim)
        coords = pca.fit_transform(corr_matrix)

        return torch.tensor(coords, dtype=torch.float32)

    def _build_edges(self, returns_df):
        """Строит рёбра на основе корреляции"""
        corr_matrix = returns_df.corr().values
        n = len(returns_df.columns)

        edges = []
        edge_features = []

        for i in range(n):
            for j in range(n):
                if i != j and abs(corr_matrix[i, j]) > self.corr_threshold:
                    edges.append([i, j])
                    edge_features.append([
                        corr_matrix[i, j],  # Корреляция
                        abs(corr_matrix[i, j]),  # Абсолютная корреляция
                        1 if corr_matrix[i, j] > 0 else -1  # Знак
                    ])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return edge_index, edge_attr
```

### Полная модель E-GNN для торговли

```python
class EquivariantGNNTrader:
    """
    Полная модель E-GNN для торговли криптовалютами.

    Архитектура:
    1. Построение графа из рыночных данных
    2. Многослойная E-GNN для извлечения признаков
    3. Инвариантный пулинг для представления на уровне графа
    4. Многоголовый выход (направление, размер позиции, риск)
    """

    def __init__(
        self,
        input_dim: int = 10,       # Размерность признаков узла
        hidden_dim: int = 64,      # Размер скрытого слоя
        coord_dim: int = 3,        # Размерность координат
        n_layers: int = 4,         # Количество слоёв E-GNN
        output_classes: int = 3,   # Long, Hold, Short
        dropout: float = 0.1
    ):
        # Входной эмбеддинг
        self.node_embed = Linear(input_dim, hidden_dim)

        # Слои E-GNN
        self.egnn_layers = ModuleList([
            EGNNLayer(hidden_dim, edge_dim=3, update_coords=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        # Нормализация
        self.layer_norms = ModuleList([
            LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.dropout = Dropout(dropout)

        # Инвариантный пулинг
        self.graph_pool = AttentionalPooling(hidden_dim)

        # Выходные головы
        self.direction_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, output_classes)
        )

        self.sizing_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
            Sigmoid()
        )

        self.risk_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 2)  # Волатильность, VaR
        )

    def forward(self, graph):
        """
        Прямой проход через E-GNN трейдер.

        Args:
            graph: Dict с 'h', 'x', 'edge_index', 'edge_attr', 'batch'

        Returns:
            Dict с 'direction', 'position_size', 'risk_metrics'
        """
        h = graph['h']
        x = graph['x']
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_attr', None)
        batch = graph.get('batch', torch.zeros(h.size(0), dtype=torch.long))

        # Эмбеддинг входных признаков
        h = self.node_embed(h)

        # Слои E-GNN
        for i, (egnn, norm) in enumerate(zip(self.egnn_layers, self.layer_norms)):
            h_new, x = egnn(h, x, edge_index, edge_attr)
            h = norm(h_new)
            h = self.dropout(h)

        # Пулинг на уровне графа (инвариантный)
        graph_repr = self.graph_pool(h, batch)

        # Прогнозы
        direction_logits = self.direction_head(graph_repr)
        position_size = self.sizing_head(graph_repr)
        risk_metrics = self.risk_head(graph_repr)

        return {
            'direction': F.softmax(direction_logits, dim=-1),
            'direction_logits': direction_logits,
            'position_size': position_size,
            'volatility_pred': F.softplus(risk_metrics[:, 0:1]),
            'var_pred': risk_metrics[:, 1:2]
        }
```

### Конвейер обучения

```python
class EGNNTrainingPipeline:
    """
    Конвейер обучения для Эквивариантного GNN Трейдера
    """

    def __init__(
        self,
        model: EquivariantGNNTrader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    def train_step(self, graph, labels):
        """Один шаг обучения"""
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(graph)
        loss = self._compute_loss(output, labels)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, output, labels):
        """
        Мультизадачная функция потерь:
        1. Классификация направления (focal loss для дисбаланса)
        2. Прокси коэффициента Шарпа
        3. Прогноз риска
        """
        # Потери направления (focal)
        direction_loss = self._focal_loss(
            output['direction_logits'],
            labels['direction'],
            gamma=2.0
        )

        # Потери в стиле Шарпа
        positions = output['direction'][:, 2] - output['direction'][:, 0]
        strategy_returns = positions * labels['future_returns']
        sharpe_loss = -strategy_returns.mean() / (strategy_returns.std() + 1e-8)

        # Потери прогноза риска
        vol_loss = F.mse_loss(output['volatility_pred'].squeeze(),
                              labels['realized_vol'])

        # Потери размера позиции
        size_loss = F.mse_loss(
            output['position_size'].squeeze(),
            labels['optimal_size']
        )

        # Комбинируем потери
        total = direction_loss + 0.3 * sharpe_loss + 0.2 * vol_loss + 0.1 * size_loss

        return total
```

### Ключевые метрики производительности

| Метрика | Цель | Описание |
|---------|------|----------|
| Точность направления | > 55% | Правильное предсказание направления движения цены |
| Коэффициент Шарпа | > 2.0 | Доходность с учётом риска (годовая) |
| Макс. просадка | < 15% | Наибольшее снижение от пика до дна |
| Доля прибыльных | > 50% | Процент прибыльных сделок |
| Профит-фактор | > 1.5 | Отношение валовой прибыли к валовому убытку |

### Варианты E-GNN для торговли

| Вариант | Обновление координат | Признаки рёбер | Применение |
|---------|---------------------|----------------|------------|
| E-GNN базовый | Да | Только расстояние | Общая структура рынка |
| SE(3)-GNN | Эквивариантно вращению | Полные геометрические | Сложные корреляции |
| Temporal E-GNN | С учётом времени | Признаки лага | Детекция режимов |
| Hierarchical E-GNN | Многомасштабный | Секторные рёбра | Микроструктура рынка |

### Зависимости

```toml
[dependencies]
# Ядро ML
torch = ">=2.0.0"
torch-geometric = ">=2.4.0"
numpy = ">=1.23.0"

# Специфично для E-GNN
e3nn = ">=0.5.0"  # Опционально: для SE(3) эквивариантности

# Обработка данных
pandas = ">=2.0.0"
polars = ">=0.19.0"

# Bybit API
pybit = ">=5.6.0"

# Визуализация
matplotlib = ">=3.6.0"
networkx = ">=3.0"
```

## Ожидаемые результаты

1. **Модель E-GNN для торговли** — Эквивариантная архитектура, уважающая рыночные симметрии
2. **Динамическое построение графа** — Графы активов на основе корреляций из рыночных данных
3. **Генерация сигналов для нескольких активов** — Торговые решения на уровне портфеля
4. **Надёжное бэктестирование** — Оценка производительности на исторических данных Bybit
5. **Управление рисками** — Размер позиции с прогнозом волатильности

## Ссылки

- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) — Оригинальная статья EGNN
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) — Полный обзор GDL
- [Graph Neural Networks for Financial Market Prediction](https://arxiv.org/abs/2106.06272)
- [Equivariant Architectures for Learning in Deep Weight Spaces](https://arxiv.org/abs/2301.12780)
- [Документация Bybit API](https://bybit-exchange.github.io/docs/)

## Уровень сложности

⭐⭐⭐⭐⭐ (Экспертный)

**Предварительные требования:** Графовые нейронные сети, Основы теории групп, Геометрическое глубокое обучение, Микроструктура финансовых рынков, Торговля криптовалютами
