# Глава 349: Graph Pooling для Трейдинга — Иерархические Представления Рынка

## Обзор

Реальные финансовые рынки обладают иерархической структурой: отдельные активы образуют секторы, секторы формируют отрасли, а отрасли составляют общий рынок. **Graph Pooling** — это техника из графовых нейронных сетей (GNN), которая обучает иерархические представления путём последовательного огрубления графовых структур, позволяя моделям улавливать многомасштабную динамику рынка.

Традиционные плоские представления обрабатывают все активы одинаково, игнорируя естественные иерархии. Graph pooling учится агрегировать узлы в осмысленные кластеры, обнаруживая скрытые рыночные структуры, которые могут предсказывать системные движения и выявлять возможности для диверсификации.

### Почему Graph Pooling для Трейдинга?

1. **Обнаружение иерархической структуры**: Автоматически обучает группировки секторов/отраслей на основе корреляций цен
2. **Многомасштабный анализ**: Улавливает как сигналы на уровне активов, так и общерыночные тренды
3. **Снижение размерности**: Сжимает большие вселенные активов в управляемые представления
4. **Детекция системного риска**: Выявляет, когда различные сегменты рынка становятся коррелированными (режимы risk-on/risk-off)
5. **Построение портфеля**: Создаёт иерархическую диверсификацию на основе обученных кластеров

## Торговая Стратегия

### Концепция Стратегии

Построение иерархического представления рынка с использованием graph pooling для:

1. **Обнаружение кластеров**: Обучение тому, какие активы естественно группируются вместе на основе корреляций доходностей и межактивных информационных потоков
2. **Детекция режимов**: Мониторинг динамики на уровне кластеров для обнаружения смены рыночных режимов
3. **Генерация альфы**: Торговля на основе расхождений между поведением актива и поведением его кластера

### Преимущество (Edge)

> Активы, отклоняющиеся от поведения своего кластера (кластерный относительный моментум), часто возвращаются к среднему, предоставляя возможности для mean-reversion стратегий. И наоборот, когда кластеры начинают двигаться вместе (разрушение корреляционной структуры), это сигнализирует о системных событиях, требующих снижения риска.

## Теоретические Основы

### Графовое Представление Рынков

Мы представляем рынок как граф G = (V, E, X):
- **V**: Узлы, представляющие активы (акции, криптовалютные пары)
- **E**: Рёбра, представляющие связи (корреляции, опережение-запаздывание, принадлежность к сектору)
- **X**: Признаки узлов (доходности, волатильность, объём, технические индикаторы)

### Методы Graph Pooling

#### 1. DiffPool (Дифференцируемый Пулинг)

Обучает мягкие назначения кластеров через дифференцируемый процесс:

```
S = softmax(GNN_pool(A, X))  # Матрица назначений кластеров [N x K]
X' = S^T * X                  # Агрегированные признаки узлов [K x F]
A' = S^T * A * S              # Агрегированная матрица смежности [K x K]
```

Где:
- N = количество узлов (активов)
- K = количество кластеров
- F = размерность признаков

#### 2. Top-K Пулинг

Выбирает top-K узлов на основе обученных оценок важности:

```
y = X * p / ||p||             # Проецируем признаки в скалярные оценки
idx = top-k(y)                # Выбираем индексы top-k узлов
X' = X[idx] * sigmoid(y[idx]) # Гейтируем выбранные признаки
A' = A[idx, idx]              # Индуцированный подграф
```

#### 3. SAGPool (Self-Attention Graph Pooling)

Использует граф-attention для определения важности узлов:

```
Z = GNN(A, X)                 # Получаем эмбеддинги узлов
y = Z * p                     # Вычисляем attention-оценки
idx = top-k(y)                # Выбираем важные узлы
X' = Z[idx] * tanh(y[idx])    # Взвешенные attention признаки
```

#### 4. MinCutPool

Оптимизирует для сбалансированных, хорошо разделённых кластеров с использованием спектральной теории графов:

```
S = softmax(MLP(X))           # Мягкие назначения кластеров
Loss_mincut = -Tr(S^T * A * S) / Tr(S^T * D * S)  # Целевая функция MinCut
Loss_ortho = ||S^T*S - I||_F                      # Регуляризация ортогональности
```

### Архитектура Иерархического Рынка

```
┌─────────────────────────────────────────────────────────────────┐
│                   Уровень 3: Обзор Рынка                        │
│                    ┌───────────────┐                             │
│                    │Состояние Рынка│                             │
│                    │  (1 узел)     │                             │
│                    └───────┬───────┘                             │
│                            │ DiffPool                            │
├────────────────────────────┼────────────────────────────────────┤
│                   Уровень 2: Обзор Секторов                      │
│     ┌──────────┐    ┌──────┴─────┐    ┌──────────┐             │
│     │ Кластер1 │    │  Кластер2  │    │ Кластер3 │             │
│     │  (Tech?) │    │  (DeFi?)   │    │  (Meme?) │             │
│     └────┬─────┘    └─────┬──────┘    └────┬─────┘             │
│          │ SAGPool        │                 │                    │
├──────────┼────────────────┼─────────────────┼────────────────────┤
│                   Уровень 1: Обзор Активов                       │
│  ┌───┐ ┌───┐ ┌───┐  ┌───┐ ┌───┐  ┌───┐ ┌───┐ ┌───┐           │
│  │BTC│ │ETH│ │SOL│  │UNI│ │AAVE│  │DOGE│ │SHIB│ │PEPE│          │
│  └───┘ └───┘ └───┘  └───┘ └───┘  └───┘ └───┘ └───┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Техническая Реализация

### Иерархическая Графовая Сеть

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn import dense_diff_pool


class HierarchicalMarketGNN(nn.Module):
    """
    Иерархическая графовая нейронная сеть для анализа рынка.

    Использует несколько слоёв graph pooling для создания
    многомасштабного представления рыночной структуры.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        n_clusters_l1: int = 10,  # Первый уровень пулинга
        n_clusters_l2: int = 3,   # Второй уровень пулинга
        hidden_dim: int = 64
    ):
        super().__init__()

        self.n_assets = n_assets

        # Уровень 1: GNN на уровне активов
        self.gnn1_embed = nn.Sequential(
            DenseGCNConv(n_features, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, hidden_dim)
        )
        self.gnn1_pool = nn.Sequential(
            DenseGCNConv(n_features, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, n_clusters_l1)
        )

        # Уровень 2: GNN на уровне кластеров
        self.gnn2_embed = nn.Sequential(
            DenseGCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, hidden_dim)
        )
        self.gnn2_pool = nn.Sequential(
            DenseGCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, n_clusters_l2)
        )

        # Уровень 3: GNN на уровне рынка
        self.gnn3 = nn.Sequential(
            DenseGCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, hidden_dim)
        )

        # Головы предсказаний
        self.asset_predictor = nn.Linear(hidden_dim, 1)   # Предсказание по активам
        self.market_predictor = nn.Linear(hidden_dim, 1)  # Рыночное предсказание

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Прямой проход через иерархическую сеть.

        Args:
            x: Признаки узлов [batch, n_assets, n_features]
            adj: Матрица смежности [batch, n_assets, n_assets]

        Returns:
            asset_pred: Предсказания по активам
            market_pred: Рыночное предсказание
            cluster_assignments: Обученные принадлежности к кластерам
        """
        batch_size = x.size(0)

        # === Уровень 1: Активы → Кластеры ===
        # Получаем эмбеддинги
        z1 = self.gnn1_embed[0](x, adj)
        z1 = F.relu(z1)
        z1 = self.gnn1_embed[1](z1, adj)

        # Получаем назначения кластеров
        s1 = self.gnn1_pool[0](x, adj)
        s1 = F.relu(s1)
        s1 = self.gnn1_pool[1](s1, adj)
        s1 = F.softmax(s1, dim=-1)  # Мягкие назначения [batch, n_assets, n_clusters_l1]

        # Пулинг до уровня кластеров
        x1, adj1, loss1, _ = dense_diff_pool(z1, adj, s1)

        # === Уровень 2: Кластеры → Суперкластеры ===
        z2 = self.gnn2_embed[0](x1, adj1)
        z2 = F.relu(z2)
        z2 = self.gnn2_embed[1](z2, adj1)

        s2 = self.gnn2_pool[0](x1, adj1)
        s2 = F.relu(s2)
        s2 = self.gnn2_pool[1](s2, adj1)
        s2 = F.softmax(s2, dim=-1)

        x2, adj2, loss2, _ = dense_diff_pool(z2, adj1, s2)

        # === Уровень 3: Рыночное представление ===
        z3 = self.gnn3[0](x2, adj2)
        z3 = F.relu(z3)
        z3 = self.gnn3[1](z3, adj2)

        # Глобальный readout
        market_embedding = z3.mean(dim=1)  # [batch, hidden_dim]

        # === Предсказания ===
        # На уровне активов (используем эмбеддинги уровня 1)
        asset_pred = self.asset_predictor(z1).squeeze(-1)  # [batch, n_assets]

        # На уровне рынка
        market_pred = self.market_predictor(market_embedding)  # [batch, 1]

        # Потери пулинга (для обучения)
        pool_loss = loss1 + loss2

        return {
            'asset_predictions': asset_pred,
            'market_prediction': market_pred,
            'cluster_assignments_l1': s1,
            'cluster_assignments_l2': s2,
            'pool_loss': pool_loss,
            'embeddings': {
                'asset': z1,
                'cluster': z2,
                'market': market_embedding
            }
        }


class SAGPoolLayer(nn.Module):
    """
    Слой Self-Attention Graph Pooling.

    Выбирает top-k узлов на основе обученных attention-оценок,
    полезен для идентификации наиболее важных активов.
    """

    def __init__(self, in_dim: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_dim, 1)
        self.gnn = DenseGCNConv(in_dim, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: Признаки узлов [batch, n_nodes, in_dim]
            adj: Матрица смежности [batch, n_nodes, n_nodes]

        Returns:
            x_pooled: Агрегированные признаки
            adj_pooled: Агрегированная матрица смежности
            importance_scores: Оценки важности узлов
            selected_idx: Индексы выбранных узлов
        """
        # Получаем эмбеддинги узлов
        z = self.gnn(x, adj)

        # Вычисляем attention-оценки
        scores = self.score_layer(z).squeeze(-1)  # [batch, n_nodes]
        scores = torch.tanh(scores)

        # Выбираем top-k узлов
        n_nodes = x.size(1)
        k = max(1, int(n_nodes * self.ratio))

        _, idx = torch.topk(scores, k, dim=1)
        idx = idx.sort(dim=1)[0]  # Сортируем для консистентности

        # Собираем выбранные узлы
        batch_size = x.size(0)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, k)

        x_pooled = z[batch_idx, idx] * scores[batch_idx, idx].unsqueeze(-1)

        # Строим агрегированную матрицу смежности
        adj_pooled = adj[batch_idx.unsqueeze(-1), idx.unsqueeze(-1), idx.unsqueeze(1)]

        return x_pooled, adj_pooled, scores, idx


class MinCutPoolLayer(nn.Module):
    """
    Слой MinCut Pooling.

    Оптимизирует для сбалансированных кластеров с минимальным
    количеством межкластерных рёбер, вдохновлён спектральным
    разбиением графов.
    """

    def __init__(self, in_dim: int, n_clusters: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, n_clusters)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: Признаки узлов [batch, n_nodes, in_dim]
            adj: Матрица смежности [batch, n_nodes, n_nodes]

        Returns:
            x_pooled: Агрегированные признаки
            adj_pooled: Агрегированная матрица смежности
            s: Мягкие назначения кластеров
            mincut_loss: Потери регуляризации MinCut
            ortho_loss: Потери регуляризации ортогональности
        """
        # Вычисляем мягкие назначения кластеров
        s = F.softmax(self.mlp(x), dim=-1)  # [batch, n_nodes, n_clusters]

        # Агрегируем признаки: X' = S^T @ X
        x_pooled = torch.bmm(s.transpose(1, 2), x)

        # Агрегируем матрицу смежности: A' = S^T @ A @ S
        adj_pooled = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)

        # === Вычисляем потери ===
        # Потери MinCut: минимизируем разрез между кластерами
        d = adj.sum(dim=-1, keepdim=True)  # Степень
        stas = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)
        stds = torch.bmm(torch.bmm(s.transpose(1, 2), d.expand_as(adj)), s)

        mincut_loss = -torch.diagonal(stas, dim1=1, dim2=2).sum(dim=1)
        mincut_loss = mincut_loss / (torch.diagonal(stds, dim1=1, dim2=2).sum(dim=1) + 1e-8)

        # Потери ортогональности: кластеры не должны перекрываться
        sts = torch.bmm(s.transpose(1, 2), s)
        sts_norm = sts / (torch.norm(sts, dim=(1, 2), keepdim=True) + 1e-8)
        identity = torch.eye(self.n_clusters, device=x.device).unsqueeze(0) / (self.n_clusters ** 0.5)
        ortho_loss = torch.norm(sts_norm - identity, dim=(1, 2))

        return x_pooled, adj_pooled, s, mincut_loss.mean(), ortho_loss.mean()
```

### Реализация Торговой Стратегии

```python
class GraphPoolingTradingStrategy:
    """
    Торговая стратегия на основе иерархического graph pooling.

    Генерирует сигналы из:
    1. Кластерного относительного моментума (актив vs его кластер)
    2. Детекции корреляционного режима кластеров
    3. Иерархической оценки риска
    """

    def __init__(
        self,
        model: HierarchicalMarketGNN,
        lookback: int = 20,
        n_assets: int = 50,
        rebalance_freq: int = 5
    ):
        self.model = model
        self.lookback = lookback
        self.n_assets = n_assets
        self.rebalance_freq = rebalance_freq

        # Отслеживание
        self.cluster_history = []
        self.correlation_regime = 'normal'

    def compute_correlation_graph(self, returns: np.ndarray) -> np.ndarray:
        """
        Построение корреляционного графа из доходностей.

        Args:
            returns: [lookback, n_assets] матрица доходностей

        Returns:
            adj: [n_assets, n_assets] матрица смежности
        """
        corr = np.corrcoef(returns.T)

        # Отсекаем слабые корреляции
        adj = np.where(np.abs(corr) > 0.3, np.abs(corr), 0)
        np.fill_diagonal(adj, 0)

        return adj

    def compute_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Вычисление признаков узлов для каждого актива.

        Args:
            prices: [lookback, n_assets] история цен
            volumes: [lookback, n_assets] история объёмов

        Returns:
            features: [n_assets, n_features]
        """
        returns = np.diff(np.log(prices), axis=0)

        features = []
        for i in range(self.n_assets):
            r = returns[:, i]
            v = volumes[1:, i]

            feat = [
                r[-1],                          # Последняя доходность
                r[-5:].mean(),                  # 5-дневный моментум
                r.std(),                        # Волатильность
                (r[-5:].mean() - r.mean()) / (r.std() + 1e-8),  # Z-score моментума
                v[-1] / (v.mean() + 1e-8),     # Относительный объём
                (r > 0).sum() / len(r),        # Доля выигрышей
                r.cumsum()[-1],                # Кумулятивная доходность
                np.corrcoef(r, v)[0, 1] if len(r) > 1 else 0  # Корреляция доходность-объём
            ]
            features.append(feat)

        return np.array(features)

    def detect_correlation_regime(self, cluster_assignments: np.ndarray) -> str:
        """
        Детекция изменения корреляционного режима.

        Когда кластеры становятся менее выраженными (более равномерные назначения),
        это указывает на разрушение корреляций / режим risk-off.
        """
        # Измеряем чистоту кластеров (насколько выражены кластеры)
        entropy = -np.sum(cluster_assignments * np.log(cluster_assignments + 1e-8), axis=-1)
        avg_entropy = entropy.mean()

        # Отслеживаем историю энтропии
        self.cluster_history.append(avg_entropy)
        if len(self.cluster_history) > 20:
            self.cluster_history.pop(0)

        # Определяем режим
        if len(self.cluster_history) >= 10:
            recent = np.mean(self.cluster_history[-5:])
            historical = np.mean(self.cluster_history[:-5])

            if recent > historical * 1.2:
                return 'correlation_breakdown'
            elif recent < historical * 0.8:
                return 'cluster_formation'

        return 'normal'

    def compute_cluster_relative_momentum(
        self,
        returns: np.ndarray,
        cluster_assignments: np.ndarray
    ) -> np.ndarray:
        """
        Вычисление отклонения каждого актива от поведения его кластера.

        Активы, опережающие кластер: положительный моментум
        Активы, отстающие от кластера: отрицательный моментум
        """
        # Получаем жёсткие назначения кластеров
        clusters = cluster_assignments.argmax(axis=-1)

        # Вычисляем доходности кластеров (взвешенное среднее)
        n_clusters = cluster_assignments.shape[-1]
        cluster_returns = np.zeros(n_clusters)

        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() > 0:
                # Взвешиваем по вероятности принадлежности к кластеру
                weights = cluster_assignments[mask, c]
                weights = weights / weights.sum()
                cluster_returns[c] = (returns[-1, mask] * weights).sum()

        # Относительный моментум = доходность актива - доходность кластера
        relative_momentum = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            c = clusters[i]
            relative_momentum[i] = returns[-1, i] - cluster_returns[c]

        return relative_momentum

    def generate_signals(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> dict:
        """
        Генерация торговых сигналов.

        Args:
            prices: [lookback + 1, n_assets] история цен
            volumes: [lookback + 1, n_assets] история объёмов

        Returns:
            signals: Словарь с позициями и метаданными
        """
        # Вычисляем входные данные
        returns = np.diff(np.log(prices), axis=0)
        features = self.compute_features(prices, volumes)
        adj = self.compute_correlation_graph(returns)

        # Конвертируем в тензоры
        x = torch.FloatTensor(features).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0)

        # Прямой проход
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, adj_tensor)

        # Извлекаем предсказания и кластеры
        asset_pred = output['asset_predictions'].squeeze().numpy()
        cluster_assignments = output['cluster_assignments_l1'].squeeze().numpy()

        # Определяем режим
        regime = self.detect_correlation_regime(cluster_assignments)

        # Вычисляем относительный моментум
        rel_momentum = self.compute_cluster_relative_momentum(returns, cluster_assignments)

        # === Генерируем позиции ===
        positions = np.zeros(self.n_assets)

        if regime == 'correlation_breakdown':
            # Risk-off: сокращаем позиции, оставляем только низкокоррелированные активы
            cluster_sizes = cluster_assignments.sum(axis=0)
            smallest_cluster = cluster_sizes.argmin()

            # Небольшие позиции только в этом кластере
            mask = cluster_assignments[:, smallest_cluster] > 0.5
            positions[mask] = 0.3 * asset_pred[mask]

        elif regime == 'cluster_formation':
            # Кластеры становятся более выраженными: моментум внутри кластеров
            # Идём с лидерами кластеров
            for c in range(cluster_assignments.shape[-1]):
                mask = cluster_assignments[:, c] > 0.5
                if mask.sum() > 0:
                    cluster_leader = np.argmax(rel_momentum[mask])
                    positions[mask][cluster_leader] = asset_pred[mask][cluster_leader]
        else:
            # Нормальный режим: mean-reversion на относительном моментуме
            positions = -rel_momentum * 0.5  # Контрарианство
            positions += asset_pred * 0.5    # Предсказание модели

        # Нормализуем позиции
        if np.abs(positions).sum() > 0:
            positions = positions / np.abs(positions).sum()

        return {
            'positions': positions,
            'regime': regime,
            'cluster_assignments': cluster_assignments,
            'predictions': asset_pred,
            'relative_momentum': rel_momentum
        }
```

### Фреймворк Бэктестинга

```python
class GraphPoolingBacktester:
    """
    Фреймворк бэктестинга для стратегии graph pooling.
    """

    def __init__(
        self,
        strategy: GraphPoolingTradingStrategy,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        self.strategy = strategy
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run_backtest(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> dict:
        """
        Запуск бэктеста за указанный период.

        Args:
            prices: DataFrame с ценами активов
            volumes: DataFrame с объёмами активов
            start_date: Строка начальной даты
            end_date: Строка конечной даты

        Returns:
            results: Словарь с метриками производительности и историей
        """
        lookback = self.strategy.lookback
        rebalance_freq = self.strategy.rebalance_freq

        # Выравниваем данные
        dates = prices.loc[start_date:end_date].index

        # Инициализация
        portfolio_value = 1.0
        positions = np.zeros(len(prices.columns))

        history = {
            'dates': [],
            'portfolio_value': [],
            'positions': [],
            'regimes': [],
            'returns': []
        }

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            # Получаем историческое окно
            window_start = dates[i - lookback]
            price_window = prices.loc[window_start:date].values
            volume_window = volumes.loc[window_start:date].values

            # Генерируем сигналы в дни ребалансировки
            if i % rebalance_freq == 0:
                signals = self.strategy.generate_signals(price_window, volume_window)
                new_positions = signals['positions']

                # Вычисляем оборот и издержки
                turnover = np.abs(new_positions - positions).sum()
                costs = turnover * (self.transaction_cost + self.slippage)

                positions = new_positions
                portfolio_value *= (1 - costs)

                history['regimes'].append(signals['regime'])
            else:
                history['regimes'].append(history['regimes'][-1] if history['regimes'] else 'normal')

            # Вычисляем доходности
            if i > lookback:
                daily_returns = prices.loc[date].values / prices.loc[dates[i-1]].values - 1
                portfolio_return = (positions * daily_returns).sum()
                portfolio_value *= (1 + portfolio_return)
                history['returns'].append(portfolio_return)

            history['dates'].append(date)
            history['portfolio_value'].append(portfolio_value)
            history['positions'].append(positions.copy())

        # Вычисляем метрики
        returns = np.array(history['returns'])

        metrics = {
            'total_return': portfolio_value - 1,
            'annualized_return': (portfolio_value ** (252 / len(returns))) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(history['portfolio_value']),
            'sortino_ratio': self._calculate_sortino(returns),
            'calmar_ratio': ((portfolio_value ** (252 / len(returns))) - 1) /
                           abs(self._calculate_max_drawdown(history['portfolio_value']))
        }

        return {
            'metrics': metrics,
            'history': history
        }

    def _calculate_max_drawdown(self, portfolio_values: list) -> float:
        """Вычисление максимальной просадки."""
        peak = portfolio_values[0]
        max_dd = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Вычисление коэффициента Сортино."""
        downside = returns[returns < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = downside.std() * np.sqrt(252)
        if downside_std == 0:
            return np.inf
        return (returns.mean() * 252) / downside_std
```

## Ключевые Метрики

### Метрики Модели
| Метрика | Описание |
|---------|----------|
| **Pooling Loss** | Потери предсказания связей и энтропии DiffPool |
| **Cluster Purity** | Насколько выражены обученные кластеры |
| **Hierarchy Stability** | Консистентность назначений кластеров во времени |
| **Prediction Accuracy** | Качество предсказания доходностей активов |

### Торговые Метрики
| Метрика | Описание |
|---------|----------|
| **Sharpe Ratio** | Доходность с поправкой на риск |
| **Sortino Ratio** | Доходность с поправкой на нисходящий риск |
| **Max Drawdown** | Максимальное падение от пика до дна |
| **Calmar Ratio** | Доходность / Max Drawdown |
| **Turnover** | Средний оборот портфеля |
| **Regime Detection Accuracy** | Точность определения режимов |

### Кластерные Метрики
| Метрика | Описание |
|---------|----------|
| **Silhouette Score** | Качество разделения кластеров |
| **Cluster Return Spread** | Дисперсия доходностей кластеров |
| **Intra-cluster Correlation** | Средняя корреляция внутри кластеров |
| **Cross-cluster Correlation** | Корреляция между кластерами |

## Зависимости

```python
# Графовые нейронные сети
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# Обработка данных
numpy>=1.23.0
pandas>=1.5.0

# Визуализация
matplotlib>=3.6.0
networkx>=3.0
plotly>=5.0.0

# ML утилиты
scikit-learn>=1.2.0

# API
requests>=2.28.0
websockets>=10.0
```

## Ожидаемые Результаты

1. **Иерархическая модель рынка**: GNN, обучающая многомасштабную рыночную структуру
2. **Обнаружение кластеров**: Автоматическая группировка коррелированных активов
3. **Детекция режимов**: Раннее предупреждение об изменениях корреляционного режима
4. **Торговая стратегия**: Кластерный относительный моментум с адаптацией к режиму
5. **Управление рисками**: Размер позиций на основе уровней иерархии

## Реализация на Rust

См. директорию `rust/` для высокопроизводительной реализации с интеграцией Bybit API:

```
rust/
├── src/
│   ├── lib.rs                 # Главная библиотека
│   ├── api/                   # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── types.rs
│   ├── graph/                 # Графовые структуры
│   │   ├── mod.rs
│   │   ├── market_graph.rs
│   │   └── correlation.rs
│   ├── pooling/               # Алгоритмы пулинга
│   │   ├── mod.rs
│   │   ├── diffpool.rs
│   │   ├── sagpool.rs
│   │   └── mincut.rs
│   ├── strategy/              # Торговая стратегия
│   │   ├── mod.rs
│   │   └── hierarchical.rs
│   └── bin/                   # Примеры исполняемых файлов
│       ├── demo.rs
│       └── backtest.rs
├── Cargo.toml
└── README.md
```

## Ссылки

### Научные Статьи
- [Hierarchical Graph Representation Learning with DiffPool](https://arxiv.org/abs/1806.08804) - Ying et al., 2018
- [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) - Lee et al., 2019
- [Spectral Clustering and the High-dimensional Stochastic Blockmodel](https://arxiv.org/abs/1007.1684) - Rohe et al., 2011
- [Graph Neural Networks for Financial Time Series](https://arxiv.org/abs/2104.12389)

### Документация
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)

## Уровень Сложности

Рейтинг: Эксперт

### Необходимые Знания
- **Графовые нейронные сети**: Message passing, эмбеддинги узлов, графовые свёртки
- **Спектральная теория графов**: Лапласиан, разрезы графов, спектральная кластеризация
- **Глубокое обучение**: Backpropagation через дискретные операции, механизмы внимания
- **Финансовые рынки**: Корреляционные структуры, смена режимов, теория портфеля
- **Программная инженерия**: Асинхронное программирование на Rust, интеграция с API
