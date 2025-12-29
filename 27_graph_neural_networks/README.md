# Chapter 27: Graph Neural Networks — Sector Momentum via Correlation Networks

## Overview

Graph Neural Networks (GNN) позволяют моделировать взаимосвязи между активами как граф, где узлы — это акции, а рёбра — корреляции или экономические связи. В этой главе мы используем GNN для обнаружения lead-lag relationships между акциями и построения momentum стратегии на основе "заражения" momentum от лидеров к лаггерам.

## Trading Strategy

**Суть стратегии:** Когда лидер сектора показывает сильный momentum, с задержкой этот momentum "распространяется" на связанные акции. GNN предсказывает, какие акции последуют за лидером.

**Сигнал на вход:**
- Long: Лаггер с высоким predicted momentum после сигнала от лидера
- Short: Лаггер с низким predicted momentum после негативного сигнала лидера

**Edge:** Lead-lag relationship exploitation — торгуем лаггера, когда лидер уже показал направление

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_and_graph_construction.ipynb` | Построение графа из корреляционной матрицы |
| 2 | `02_graph_analysis.ipynb` | Анализ графа: centrality, communities, hubs |
| 3 | `03_lead_lag_detection.ipynb` | Статистическое обнаружение lead-lag пар |
| 4 | `04_gnn_architecture.ipynb` | Разбор GCN, GAT, GraphSAGE архитектур |
| 5 | `05_node_feature_engineering.ipynb` | Features для каждой акции (node attributes) |
| 6 | `06_model_training.ipynb` | Обучение GNN для node classification/regression |
| 7 | `07_momentum_propagation.ipynb` | Моделирование распространения momentum |
| 8 | `08_trading_strategy.ipynb` | Реализация и backtesting стратегии |
| 9 | `09_dynamic_graphs.ipynb` | Временные графы — эволюция структуры |

### Data Requirements

```
Stock Universe:
├── S&P 500 constituents (500 акций)
├── Daily OHLCV (5+ лет)
├── Sector/Industry classification (GICS)
├── Market cap, Book value (для фильтрации)
└── Earnings dates (опционально)

Graph Construction:
├── Rolling correlation matrix (60-day window)
├── Partial correlations (controlling for market)
├── Supply chain relationships (альтернатива)
└── Sector membership edges
```

### Graph Construction Methods

```python
# Method 1: Correlation threshold
G = nx.Graph()
corr_matrix = returns.rolling(60).corr()
for i, j in combinations(stocks, 2):
    if abs(corr_matrix.loc[i, j]) > threshold:
        G.add_edge(i, j, weight=corr_matrix.loc[i, j])

# Method 2: k-Nearest Neighbors
from sklearn.neighbors import kneighbors_graph
knn_graph = kneighbors_graph(return_features, k=10, mode='connectivity')

# Method 3: Minimum Spanning Tree
from scipy.sparse.csgraph import minimum_spanning_tree
mst = minimum_spanning_tree(distance_matrix)
```

### GNN Architecture Options

```
1. Graph Convolutional Network (GCN):
   - Простая свёртка по соседям
   - h_v = σ(W · MEAN({h_u : u ∈ N(v)}))

2. Graph Attention Network (GAT):
   - Attention weights для соседей
   - Разная важность разных связей

3. GraphSAGE:
   - Inductive learning (новые ноды)
   - Sampling соседей для масштабируемости

4. Temporal Graph Networks:
   - Эволюция графа во времени
   - Memory для отслеживания истории
```

### Node Features (per stock)

```
Technical:
├── Momentum (1m, 3m, 6m, 12m returns)
├── Volatility (realized, ATR)
├── RSI, MACD signals
└── Volume anomaly

Fundamental:
├── Sector one-hot encoding
├── Market cap decile
├── Book-to-market
└── Earnings surprise (last)

Graph-derived:
├── Degree centrality
├── PageRank
├── Clustering coefficient
└── Community membership
```

### Key Metrics

- **Graph Quality:** Modularity, Average clustering, Assortativity
- **Prediction:** Accuracy, AUC-ROC for direction, IC for returns
- **Strategy:** Sharpe Ratio, Hit Rate, Average holding period

### Dependencies

```python
torch-geometric>=2.3.0
networkx>=3.0
scipy>=1.10.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **Dynamic correlation graph** для S&P 500 с визуализацией communities
2. **Lead-lag pairs detection** — статистически значимые пары лидер-лаггер
3. **GNN модель** для предсказания momentum propagation
4. **Trading strategy** с exploitation lead-lag relationships
5. **Сравнение** с простым sector momentum

## References

- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
- [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)
- [Deep Learning for Asset Correlation Analysis](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3623739)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Требуется понимание: Graph theory, GNN архитектур, Lead-lag analysis, Network analysis
