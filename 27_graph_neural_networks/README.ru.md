# Графовые нейронные сети для торговли криптовалютами

Эта глава представляет **Graph Neural Networks (GNN)** — графовые нейронные сети — для анализа взаимосвязей между криптовалютами и построения торговых стратегий на основе распространения моментума в сети активов.

<p align="center">
<img src="https://i.imgur.com/GNN_crypto_diagram.png" width="70%">
</p>

## Содержание

1. [Введение в графовые нейронные сети](#введение-в-графовые-нейронные-сети)
    * [Зачем графы для торговли](#зачем-графы-для-торговли)
    * [Ключевые концепции](#ключевые-концепции)
    * [Типы графовых нейронных сетей](#типы-графовых-нейронных-сетей)
2. [Построение графа криптовалют](#построение-графа-криптовалют)
    * [Методы построения рёбер](#методы-построения-рёбер)
    * [Признаки узлов](#признаки-узлов)
    * [Динамические графы](#динамические-графы)
3. [Архитектуры GNN](#архитектуры-gnn)
    * [Graph Convolutional Networks (GCN)](#graph-convolutional-networks-gcn)
    * [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
    * [GraphSAGE](#graphsage)
    * [Temporal Graph Networks](#temporal-graph-networks)
4. [Торговая стратегия](#торговая-стратегия)
5. [Примеры кода](#примеры-кода)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические рекомендации](#практические-рекомендации)
8. [Ресурсы](#ресурсы)

## Введение в графовые нейронные сети

### Зачем графы для торговли

Финансовые рынки — это не набор изолированных активов, а сложная **сеть взаимосвязей**:

- **Корреляции**: Bitcoin движется вверх → альткоины часто следуют
- **Сектора**: DeFi токены реагируют на схожие новости
- **Ликвидность**: крупные монеты влияют на менее ликвидные
- **Lead-lag эффекты**: одни активы предсказывают движение других

Традиционные модели (LSTM, Transformer) рассматривают каждый актив отдельно, игнорируя эти связи. GNN позволяют **явно моделировать структуру взаимосвязей**.

```
Традиционный подход:        Графовый подход:

BTC → [LSTM] → прогноз      BTC ─────────┐
ETH → [LSTM] → прогноз           GNN    ├→ все прогнозы
SOL → [LSTM] → прогноз      ETH ─────────┤   одновременно
                            SOL ─────────┘
(независимо)                (совместно)
```

### Ключевые концепции

**Граф** состоит из:
- **Узлов (nodes)**: криптовалюты (BTC, ETH, SOL, ...)
- **Рёбер (edges)**: связи между активами (корреляция, сектор)
- **Признаков узлов**: технические индикаторы, объём, momentum
- **Весов рёбер**: сила связи (опционально)

**Message Passing**: ключевой механизм GNN
```
Для каждого узла v:
    1. Собрать сообщения от соседей N(v)
    2. Агрегировать сообщения (mean, sum, max)
    3. Обновить своё представление
```

### Типы графовых нейронных сетей

| Архитектура | Особенность | Когда использовать |
|-------------|-------------|-------------------|
| **GCN** | Спектральные свёртки | Простые задачи, стабильный граф |
| **GAT** | Attention для соседей | Разная важность связей |
| **GraphSAGE** | Сэмплирование соседей | Большие графы, новые узлы |
| **TGN** | Память для истории | Эволюция графа во времени |

## Построение графа криптовалют

### Методы построения рёбер

#### Метод 1: Корреляционный порог

```python
import networkx as nx
import numpy as np
from itertools import combinations

# Скользящая корреляция за 60 дней
corr_matrix = returns.rolling(60).corr()

# Построение графа
G = nx.Graph()
threshold = 0.5

for coin_i, coin_j in combinations(coins, 2):
    corr = corr_matrix.loc[coin_i, coin_j]
    if abs(corr) > threshold:
        G.add_edge(coin_i, coin_j, weight=corr)

# Результат: граф где связаны только сильно коррелированные монеты
```

#### Метод 2: k-ближайших соседей (kNN)

```python
from sklearn.neighbors import kneighbors_graph

# Матрица признаков: [n_coins, n_features]
features = compute_features(returns)  # momentum, volatility, ...

# k=5 ближайших соседей
knn_graph = kneighbors_graph(features, n_neighbors=5, mode='connectivity')

# Каждая монета связана с 5 наиболее похожими
```

#### Метод 3: Секторная принадлежность

```python
# Определяем сектора
sectors = {
    'DeFi': ['UNI', 'AAVE', 'COMP', 'MKR', 'SUSHI'],
    'Layer1': ['ETH', 'SOL', 'AVAX', 'NEAR', 'ATOM'],
    'Layer2': ['MATIC', 'ARB', 'OP'],
    'Meme': ['DOGE', 'SHIB', 'PEPE'],
    'Store of Value': ['BTC', 'LTC', 'BCH'],
}

# Связываем монеты внутри секторов
G = nx.Graph()
for sector, coins in sectors.items():
    for i, j in combinations(coins, 2):
        G.add_edge(i, j, sector=sector)
```

#### Метод 4: Причинность Грейнджера (Lead-lag)

```python
from statsmodels.tsa.stattools import grangercausalitytests

def test_granger_causality(x, y, max_lag=5, significance=0.05):
    """Тест причинности Грейнджера: x предсказывает y?"""
    try:
        results = grangercausalitytests(
            np.column_stack([y, x]),
            maxlag=max_lag,
            verbose=False
        )
        # Берём минимальное p-value
        min_pval = min(
            results[lag][0]['ssr_ftest'][1]
            for lag in range(1, max_lag + 1)
        )
        return min_pval < significance
    except:
        return False

# Построение направленного графа lead-lag
G = nx.DiGraph()
for coin_i in coins:
    for coin_j in coins:
        if coin_i != coin_j:
            if test_granger_causality(returns[coin_i], returns[coin_j]):
                G.add_edge(coin_i, coin_j)  # coin_i → coin_j
```

### Признаки узлов

Каждый узел (криптовалюта) описывается вектором признаков:

```python
def compute_node_features(df, window=20):
    """Вычислить признаки для одной криптовалюты."""
    features = {}

    # Технические индикаторы
    features['momentum_1d'] = df['close'].pct_change(1).iloc[-1]
    features['momentum_7d'] = df['close'].pct_change(7).iloc[-1]
    features['momentum_30d'] = df['close'].pct_change(30).iloc[-1]

    features['volatility'] = df['close'].pct_change().rolling(window).std().iloc[-1]

    features['rsi'] = compute_rsi(df['close'], window)
    features['macd'], features['macd_signal'] = compute_macd(df['close'])

    # Объёмные индикаторы
    features['volume_ratio'] = (
        df['volume'].iloc[-1] / df['volume'].rolling(window).mean().iloc[-1]
    )

    # Волатильность относительно рынка
    features['beta'] = compute_beta(df['close'], market_returns)

    # Позиция цены
    high_52w = df['close'].rolling(365).max().iloc[-1]
    low_52w = df['close'].rolling(365).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_52w) / (high_52w - low_52w)

    return features
```

### Динамические графы

Структура связей меняется со временем:

```python
class DynamicCryptoGraph:
    """Граф криптовалют с эволюцией во времени."""

    def __init__(self, coins, window=60, threshold=0.5):
        self.coins = coins
        self.window = window
        self.threshold = threshold
        self.snapshots = {}

    def build_snapshot(self, returns_df, date):
        """Построить граф на определённую дату."""
        # Корреляции за последние window дней
        end_idx = returns_df.index.get_loc(date)
        start_idx = max(0, end_idx - self.window)

        window_returns = returns_df.iloc[start_idx:end_idx + 1]
        corr_matrix = window_returns.corr()

        G = nx.Graph()
        G.add_nodes_from(self.coins)

        for i, j in combinations(self.coins, 2):
            if abs(corr_matrix.loc[i, j]) > self.threshold:
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])

        self.snapshots[date] = G
        return G

    def get_temporal_edges(self, dates):
        """Получить историю рёбер для Temporal GNN."""
        temporal_edges = []
        for date in dates:
            G = self.snapshots.get(date)
            if G:
                for u, v, data in G.edges(data=True):
                    temporal_edges.append({
                        'source': u,
                        'target': v,
                        'timestamp': date.timestamp(),
                        'weight': data.get('weight', 1.0)
                    })
        return temporal_edges
```

## Архитектуры GNN

### Graph Convolutional Networks (GCN)

**GCN** (Kipf & Welling, 2017) — базовая архитектура графовой свёртки:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

Где:
- $\tilde{A} = A + I$ — матрица смежности с самосвязями
- $\tilde{D}$ — степенная матрица
- $H^{(l)}$ — скрытые представления на слое $l$
- $W^{(l)}$ — обучаемые веса

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class CryptoGCN(nn.Module):
    """GCN для предсказания направления движения криптовалют."""

    def __init__(self, num_features, hidden_dim=64, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight=None):
        # Слой 1
        h = self.conv1(x, edge_index, edge_weight)
        h = torch.relu(h)
        h = self.dropout(h)

        # Слой 2
        h = self.conv2(h, edge_index, edge_weight)
        h = torch.relu(h)
        h = self.dropout(h)

        # Слой 3
        h = self.conv3(h, edge_index, edge_weight)
        h = torch.relu(h)

        # Классификация: Down / Neutral / Up
        out = self.classifier(h)
        return out
```

### Graph Attention Networks (GAT)

**GAT** (Veličković et al., 2018) использует attention для взвешивания соседей:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_i \| W\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_i \| W\mathbf{h}_k]))}$$

```python
from torch_geometric.nn import GATConv

class CryptoGAT(nn.Module):
    """GAT с multi-head attention для криптовалют."""

    def __init__(self, num_features, hidden_dim=64, heads=8, num_classes=3):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = torch.elu(h)
        h = self.conv2(h, edge_index)
        h = torch.elu(h)

        # Attention веса можно интерпретировать:
        # какие криптовалюты сильнее всего влияют друг на друга

        return self.classifier(h)
```

### GraphSAGE

**GraphSAGE** (Hamilton et al., 2017) — inductive learning с сэмплированием соседей:

```python
from torch_geometric.nn import SAGEConv

class CryptoGraphSAGE(nn.Module):
    """GraphSAGE для масштабируемого обучения."""

    def __init__(self, num_features, hidden_dim=64, num_classes=3):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        return self.classifier(h)
```

**Преимущества GraphSAGE**:
- Работает с новыми (unseen) узлами
- Масштабируется на большие графы
- Mini-batch обучение

### Temporal Graph Networks

**TGN** (Rossi et al., 2020) для динамических графов:

```python
from torch_geometric.nn import TGNMemory, TransformerConv

class CryptoTGN(nn.Module):
    """Temporal Graph Network для эволюции крипто-графа."""

    def __init__(self, num_features, hidden_dim=64, time_dim=16):
        super().__init__()

        # Память для отслеживания истории узлов
        self.memory = TGNMemory(
            num_nodes=100,  # max number of crypto assets
            raw_msg_dim=num_features,
            memory_dim=hidden_dim,
            time_dim=time_dim,
        )

        # Временное кодирование
        self.time_encoder = TimeEncoder(time_dim)

        # Графовые слои
        self.conv1 = TransformerConv(hidden_dim + time_dim, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, timestamps):
        # Получить память узлов
        z, last_update = self.memory(torch.arange(x.size(0)))

        # Закодировать время
        time_enc = self.time_encoder(timestamps - last_update)

        # Объединить с признаками
        h = torch.cat([z, time_enc], dim=-1)

        # Графовые свёртки
        h = torch.relu(self.conv1(h, edge_index))
        h = torch.relu(self.conv2(h, edge_index))

        return self.classifier(h)
```

## Торговая стратегия

### Суть стратегии: Momentum Propagation

Когда **лидер сектора** показывает сильный momentum, этот сигнал **распространяется** на связанные активы с задержкой. GNN предсказывает, какие криптовалюты последуют за лидером.

```
Временная шкала:
────────────────────────────────────────────────
t=0    BTC ↑↑↑ (сильный рост)
t+1h   ETH ↑↑  (следует за BTC)  ← предсказываем GNN
t+2h   SOL ↑   (следует за ETH)  ← предсказываем GNN
t+3h   AVAX ↑  (следует за SOL)  ← предсказываем GNN
────────────────────────────────────────────────
       Мы торгуем лаггеров ДО их движения
```

### Сигналы на вход

```python
class MomentumPropagationStrategy:
    """Торговая стратегия на основе GNN и momentum propagation."""

    def __init__(self, model, graph, threshold=0.6):
        self.model = model
        self.graph = graph
        self.threshold = threshold
        self.positions = {}

    def generate_signals(self, features, edge_index):
        """Генерация торговых сигналов."""
        self.model.eval()
        with torch.no_grad():
            # Предсказания: [num_coins, 3] - down/neutral/up
            predictions = torch.softmax(
                self.model(features, edge_index), dim=1
            )

        signals = {}
        for coin_idx, coin in enumerate(self.coins):
            prob_up = predictions[coin_idx, 2].item()
            prob_down = predictions[coin_idx, 0].item()

            if prob_up > self.threshold:
                signals[coin] = {
                    'action': 'LONG',
                    'confidence': prob_up,
                    'reason': self._explain_signal(coin_idx, features, edge_index)
                }
            elif prob_down > self.threshold:
                signals[coin] = {
                    'action': 'SHORT',
                    'confidence': prob_down,
                    'reason': self._explain_signal(coin_idx, features, edge_index)
                }

        return signals

    def _explain_signal(self, coin_idx, features, edge_index):
        """Объяснение сигнала через attention."""
        # Найти соседей с сильным momentum
        neighbors = self._get_neighbors(coin_idx, edge_index)
        neighbor_momentum = [
            (self.coins[n], features[n, 0].item())  # momentum feature
            for n in neighbors
        ]
        # Сортировка по силе momentum
        neighbor_momentum.sort(key=lambda x: abs(x[1]), reverse=True)

        return f"Following leaders: {neighbor_momentum[:3]}"
```

### Edge: Lead-lag Exploitation

```python
def detect_leaders_and_laggers(returns, graph, window=20):
    """Определить лидеров и лаггеров в графе."""

    leaders = {}
    laggers = {}

    for coin in graph.nodes():
        neighbors = list(graph.neighbors(coin))
        if not neighbors:
            continue

        # Cross-correlation с соседями
        lags = []
        for neighbor in neighbors:
            # Положительный лаг = coin запаздывает за neighbor
            lag = cross_correlation_lag(
                returns[neighbor],
                returns[coin],
                max_lag=window
            )
            lags.append(lag)

        avg_lag = np.mean(lags)

        if avg_lag < -2:  # coin опережает соседей
            leaders[coin] = {'avg_lag': avg_lag, 'neighbors': neighbors}
        elif avg_lag > 2:  # coin отстаёт от соседей
            laggers[coin] = {'avg_lag': avg_lag, 'neighbors': neighbors}

    return leaders, laggers
```

### Backtesting

```python
def backtest_momentum_propagation(
    model,
    historical_graphs,
    returns,
    initial_capital=100000,
    transaction_cost=0.001
):
    """Бэктест стратегии momentum propagation."""

    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [capital]

    for date in sorted(historical_graphs.keys()):
        graph = historical_graphs[date]
        features = compute_features_for_date(returns, date)
        edge_index = graph_to_edge_index(graph)

        # Генерация сигналов
        signals = generate_signals(model, features, edge_index)

        # Закрытие позиций
        for coin, position in list(positions.items()):
            coin_return = returns.loc[date, coin]
            pnl = position['size'] * coin_return * position['direction']
            capital += pnl

            # Если сигнал сменился
            if coin in signals and signals[coin]['action'] != position['action']:
                trades.append({
                    'date': date,
                    'coin': coin,
                    'action': 'CLOSE',
                    'pnl': pnl
                })
                del positions[coin]

        # Открытие новых позиций
        for coin, signal in signals.items():
            if coin not in positions and signal['confidence'] > 0.7:
                size = capital * 0.1  # 10% на позицию
                cost = size * transaction_cost
                capital -= cost

                positions[coin] = {
                    'size': size,
                    'direction': 1 if signal['action'] == 'LONG' else -1,
                    'entry_date': date,
                    'action': signal['action']
                }

                trades.append({
                    'date': date,
                    'coin': coin,
                    'action': signal['action'],
                    'size': size
                })

        equity_curve.append(capital)

    # Метрики
    equity = pd.Series(equity_curve)
    daily_returns = equity.pct_change().dropna()

    metrics = {
        'total_return': (capital - initial_capital) / initial_capital,
        'sharpe_ratio': daily_returns.mean() / daily_returns.std() * np.sqrt(365),
        'max_drawdown': (equity / equity.cummax() - 1).min(),
        'win_rate': sum(1 for t in trades if t.get('pnl', 0) > 0) / max(len(trades), 1),
        'num_trades': len(trades)
    }

    return metrics, equity_curve, trades
```

## Примеры кода

| Ноутбук | Описание |
|---------|----------|
| [01_data_and_graph_construction.ipynb](01_data_and_graph_construction.ipynb) | Построение графа криптовалют из корреляций |
| [02_graph_analysis.ipynb](02_graph_analysis.ipynb) | Анализ: centrality, communities, hubs |
| [03_lead_lag_detection.ipynb](03_lead_lag_detection.ipynb) | Обнаружение lead-lag пар |
| [04_gnn_architecture.ipynb](04_gnn_architecture.ipynb) | GCN, GAT, GraphSAGE на PyTorch Geometric |
| [05_node_feature_engineering.ipynb](05_node_feature_engineering.ipynb) | Признаки узлов: momentum, volatility, RSI |
| [06_model_training.ipynb](06_model_training.ipynb) | Обучение GNN для предсказания направления |
| [07_momentum_propagation.ipynb](07_momentum_propagation.ipynb) | Моделирование распространения momentum |
| [08_trading_strategy.ipynb](08_trading_strategy.ipynb) | Backtesting и реализация стратегии |
| [09_dynamic_graphs.ipynb](09_dynamic_graphs.ipynb) | Temporal GNN для эволюции графа |

## Реализация на Rust

Директория [rust_gnn_crypto](rust_gnn_crypto/) содержит модульную реализацию на Rust:

```
rust_gnn_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Публичный API
│   ├── main.rs             # CLI
│   ├── data/               # Загрузка данных
│   │   ├── mod.rs
│   │   ├── bybit_client.rs # Bybit API клиент
│   │   ├── ohlcv.rs        # OHLCV структуры
│   │   └── features.rs     # Feature engineering
│   ├── graph/              # Построение графов
│   │   ├── mod.rs
│   │   ├── correlation.rs  # Корреляционный граф
│   │   ├── knn.rs          # k-NN граф
│   │   └── temporal.rs     # Динамические графы
│   ├── model/              # GNN модели
│   │   ├── mod.rs
│   │   ├── gcn.rs          # Graph Convolutional Network
│   │   ├── gat.rs          # Graph Attention Network
│   │   └── layers.rs       # Графовые слои
│   ├── training/           # Обучение
│   │   ├── mod.rs
│   │   └── trainer.rs
│   ├── strategy/           # Торговые стратегии
│   │   ├── mod.rs
│   │   └── momentum.rs
│   └── utils/              # Утилиты
│       ├── mod.rs
│       └── config.rs
├── examples/
│   ├── fetch_data.rs       # Загрузка данных с Bybit
│   ├── build_graph.rs      # Построение графа
│   ├── train_gnn.rs        # Обучение модели
│   └── backtest.rs         # Бэктестинг стратегии
└── data/                   # Локальные данные
    └── .gitkeep
```

Смотрите [rust_gnn_crypto/README.md](rust_gnn_crypto/README.md) для подробностей.

## Практические рекомендации

### Когда использовать GNN для трейдинга

**Хорошие кейсы**:
- Торговля корзинами взаимосвязанных активов
- Секторный momentum и mean reversion
- Lead-lag стратегии
- Анализ влияния новостей на группы активов
- Портфельная оптимизация с учётом взаимосвязей

**Не идеально для**:
- Торговля одним активом
- Высокочастотная торговля (HFT)
- Активы без явных связей

### Выбор архитектуры

| Сценарий | Рекомендуемая архитектура |
|----------|---------------------------|
| Стабильные связи, небольшой граф | GCN |
| Разная важность связей | GAT |
| Добавляются новые монеты | GraphSAGE |
| Связи меняются во времени | TGN |
| Крипто + акции + форекс | Heterogeneous GNN |

### Частые ошибки

1. **Переобучение на структуру графа**: используйте dropout на рёбрах
2. **Статический граф**: обновляйте граф регулярно (ежедневно/еженедельно)
3. **Look-ahead bias**: при построении графа используйте только прошлые данные
4. **Игнорирование транзакционных издержек**: в крипто спреды и комиссии значительны

### Вычислительные требования

| Компонент | GPU память | Время |
|-----------|------------|-------|
| Граф 50 узлов, GCN | 2 GB | < 1 мин/эпоха |
| Граф 200 узлов, GAT | 8 GB | 5 мин/эпоха |
| Динамический граф, TGN | 12 GB | 15 мин/эпоха |

## Ресурсы

### Статьи

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), Kipf & Welling, 2017
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903), Veličković et al., 2018
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), Hamilton et al., 2017
- [Temporal Graph Networks](https://arxiv.org/abs/2006.10637), Rossi et al., 2020
- [Graph Neural Networks for Financial Time Series](https://arxiv.org/abs/2104.14621), 2021

### Реализации

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — библиотека графовых нейросетей
- [DGL (Deep Graph Library)](https://www.dgl.ai/) — альтернатива PyG
- [NetworkX](https://networkx.org/) — анализ графов на Python
- [petgraph](https://docs.rs/petgraph/) — графы на Rust

### Связанные главы

- [Глава 19: RNN для временных рядов](../19_recurrent_neural_nets) — LSTM/GRU модели
- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — attention для временных рядов
- [Глава 28: Hidden Markov Models](../28_hidden_markov_models) — режимы рынка

## Уровень сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание:
- Теория графов (смежность, степени, пути)
- Нейронные сети (PyTorch)
- Статистический анализ временных рядов
- Корреляции и причинность
