# Chapter 349: Graph Pooling for Trading — Hierarchical Market Representations

## Overview

Real financial markets exhibit hierarchical structures: individual assets form sectors, sectors form industries, and industries form the broader market. **Graph Pooling** is a technique from Graph Neural Networks (GNNs) that learns hierarchical representations by progressively coarsening graph structures, enabling models to capture multi-scale market dynamics.

Traditional flat representations treat all assets equally, ignoring natural hierarchies. Graph pooling learns to aggregate nodes into meaningful clusters, discovering latent market structures that can predict systemic movements and identify diversification opportunities.

### Why Graph Pooling for Trading?

1. **Hierarchical Structure Discovery**: Automatically learns sector/industry groupings from price correlations
2. **Multi-Scale Analysis**: Captures both asset-level signals and market-wide trends
3. **Dimensionality Reduction**: Compresses large asset universes into manageable representations
4. **Systemic Risk Detection**: Identifies when different market segments become correlated (risk-on/risk-off regimes)
5. **Portfolio Construction**: Creates hierarchical diversification based on learned clusters

## Trading Strategy

### Strategy Concept

Build a hierarchical market representation using graph pooling to:

1. **Cluster Discovery**: Learn which assets naturally group together based on return correlations and cross-asset information flow
2. **Regime Detection**: Monitor cluster-level dynamics to detect market regime changes
3. **Alpha Generation**: Trade based on discrepancies between asset behavior and its cluster's behavior

### The Edge

> Assets that deviate from their cluster's behavior (cluster-relative momentum) often revert, providing mean-reversion opportunities. Conversely, when clusters begin moving together (correlation breakdown), it signals systemic events requiring risk reduction.

## Theoretical Foundation

### Graph Representation of Markets

We represent the market as a graph G = (V, E, X):
- **V**: Nodes representing assets (stocks, crypto pairs)
- **E**: Edges representing relationships (correlations, lead-lag, sector membership)
- **X**: Node features (returns, volatility, volume, technical indicators)

### Graph Pooling Methods

#### 1. DiffPool (Differentiable Pooling)

Learns soft cluster assignments through a differentiable process:

```
S = softmax(GNN_pool(A, X))  # Cluster assignment matrix [N x K]
X' = S^T * X                  # Pooled node features [K x F]
A' = S^T * A * S              # Pooled adjacency matrix [K x K]
```

Where:
- N = number of nodes (assets)
- K = number of clusters
- F = feature dimension

#### 2. Top-K Pooling

Selects top-K nodes based on learned importance scores:

```
y = X * p / ||p||             # Project features to scalar scores
idx = top-k(y)                # Select top-k node indices
X' = X[idx] * sigmoid(y[idx]) # Gate selected features
A' = A[idx, idx]              # Induced subgraph
```

#### 3. SAGPool (Self-Attention Graph Pooling)

Uses graph attention to determine node importance:

```
Z = GNN(A, X)                 # Get node embeddings
y = Z * p                     # Compute attention scores
idx = top-k(y)                # Select important nodes
X' = Z[idx] * tanh(y[idx])    # Attention-weighted features
```

#### 4. MinCutPool

Optimizes for balanced, well-separated clusters using spectral graph theory:

```
S = softmax(MLP(X))           # Soft cluster assignments
Loss_mincut = -Tr(S^T * A * S) / Tr(S^T * D * S)  # MinCut objective
Loss_ortho = ||S^T*S - I||_F                      # Orthogonality regularization
```

### Hierarchical Market Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Level 3: Market View                          │
│                    ┌───────────────┐                             │
│                    │  Market State │                             │
│                    │  (1 node)     │                             │
│                    └───────┬───────┘                             │
│                            │ DiffPool                            │
├────────────────────────────┼────────────────────────────────────┤
│                    Level 2: Sector View                          │
│     ┌──────────┐    ┌──────┴─────┐    ┌──────────┐             │
│     │ Cluster1 │    │  Cluster2  │    │ Cluster3 │             │
│     │ (Tech?)  │    │  (DeFi?)   │    │ (Meme?)  │             │
│     └────┬─────┘    └─────┬──────┘    └────┬─────┘             │
│          │ SAGPool        │                 │                    │
├──────────┼────────────────┼─────────────────┼────────────────────┤
│                    Level 1: Asset View                           │
│  ┌───┐ ┌───┐ ┌───┐  ┌───┐ ┌───┐  ┌───┐ ┌───┐ ┌───┐           │
│  │BTC│ │ETH│ │SOL│  │UNI│ │AAVE│  │DOGE│ │SHIB│ │PEPE│          │
│  └───┘ └───┘ └───┘  └───┘ └───┘  └───┘ └───┘ └───┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### HierarchicalGraphNetwork

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn import dense_diff_pool


class HierarchicalMarketGNN(nn.Module):
    """
    Hierarchical Graph Neural Network for market analysis.

    Uses multiple graph pooling layers to create a multi-scale
    representation of the market structure.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        n_clusters_l1: int = 10,  # First pooling level
        n_clusters_l2: int = 3,   # Second pooling level
        hidden_dim: int = 64
    ):
        super().__init__()

        self.n_assets = n_assets

        # Level 1: Asset-level GNN
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

        # Level 2: Cluster-level GNN
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

        # Level 3: Market-level GNN
        self.gnn3 = nn.Sequential(
            DenseGCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            DenseGCNConv(hidden_dim, hidden_dim)
        )

        # Prediction heads
        self.asset_predictor = nn.Linear(hidden_dim, 1)  # Per-asset prediction
        self.market_predictor = nn.Linear(hidden_dim, 1)  # Market prediction

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Forward pass through hierarchical network.

        Args:
            x: Node features [batch, n_assets, n_features]
            adj: Adjacency matrix [batch, n_assets, n_assets]

        Returns:
            asset_pred: Per-asset predictions
            market_pred: Market-level prediction
            cluster_assignments: Learned cluster memberships
        """
        batch_size = x.size(0)

        # === Level 1: Asset → Clusters ===
        # Get embeddings
        z1 = self.gnn1_embed[0](x, adj)
        z1 = F.relu(z1)
        z1 = self.gnn1_embed[1](z1, adj)

        # Get cluster assignments
        s1 = self.gnn1_pool[0](x, adj)
        s1 = F.relu(s1)
        s1 = self.gnn1_pool[1](s1, adj)
        s1 = F.softmax(s1, dim=-1)  # Soft assignments [batch, n_assets, n_clusters_l1]

        # Pool to cluster level
        x1, adj1, loss1, _ = dense_diff_pool(z1, adj, s1)

        # === Level 2: Clusters → Super-clusters ===
        z2 = self.gnn2_embed[0](x1, adj1)
        z2 = F.relu(z2)
        z2 = self.gnn2_embed[1](z2, adj1)

        s2 = self.gnn2_pool[0](x1, adj1)
        s2 = F.relu(s2)
        s2 = self.gnn2_pool[1](s2, adj1)
        s2 = F.softmax(s2, dim=-1)

        x2, adj2, loss2, _ = dense_diff_pool(z2, adj1, s2)

        # === Level 3: Market representation ===
        z3 = self.gnn3[0](x2, adj2)
        z3 = F.relu(z3)
        z3 = self.gnn3[1](z3, adj2)

        # Global readout
        market_embedding = z3.mean(dim=1)  # [batch, hidden_dim]

        # === Predictions ===
        # Asset-level (use level 1 embeddings)
        asset_pred = self.asset_predictor(z1).squeeze(-1)  # [batch, n_assets]

        # Market-level
        market_pred = self.market_predictor(market_embedding)  # [batch, 1]

        # Pooling loss (for training)
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
    Self-Attention Graph Pooling layer.

    Selects top-k nodes based on learned attention scores,
    useful for identifying the most important assets.
    """

    def __init__(self, in_dim: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_dim, 1)
        self.gnn = DenseGCNConv(in_dim, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: Node features [batch, n_nodes, in_dim]
            adj: Adjacency [batch, n_nodes, n_nodes]

        Returns:
            x_pooled: Pooled features
            adj_pooled: Pooled adjacency
            importance_scores: Node importance scores
            selected_idx: Indices of selected nodes
        """
        # Get node embeddings
        z = self.gnn(x, adj)

        # Compute attention scores
        scores = self.score_layer(z).squeeze(-1)  # [batch, n_nodes]
        scores = torch.tanh(scores)

        # Select top-k nodes
        n_nodes = x.size(1)
        k = max(1, int(n_nodes * self.ratio))

        _, idx = torch.topk(scores, k, dim=1)
        idx = idx.sort(dim=1)[0]  # Sort for consistency

        # Gather selected nodes
        batch_size = x.size(0)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, k)

        x_pooled = z[batch_idx, idx] * scores[batch_idx, idx].unsqueeze(-1)

        # Build pooled adjacency
        adj_pooled = adj[batch_idx.unsqueeze(-1), idx.unsqueeze(-1), idx.unsqueeze(1)]

        return x_pooled, adj_pooled, scores, idx


class MinCutPoolLayer(nn.Module):
    """
    MinCut Pooling layer.

    Optimizes for balanced clusters with minimal inter-cluster edges,
    inspired by spectral graph partitioning.
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
            x: Node features [batch, n_nodes, in_dim]
            adj: Adjacency [batch, n_nodes, n_nodes]

        Returns:
            x_pooled: Pooled features
            adj_pooled: Pooled adjacency
            s: Soft cluster assignments
            mincut_loss: MinCut regularization loss
            ortho_loss: Orthogonality regularization loss
        """
        # Compute soft cluster assignments
        s = F.softmax(self.mlp(x), dim=-1)  # [batch, n_nodes, n_clusters]

        # Pool features: X' = S^T @ X
        x_pooled = torch.bmm(s.transpose(1, 2), x)

        # Pool adjacency: A' = S^T @ A @ S
        adj_pooled = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)

        # === Compute losses ===
        # MinCut loss: minimize cut between clusters
        # -Tr(S^T A S) / Tr(S^T D S)
        d = adj.sum(dim=-1, keepdim=True)  # Degree
        stas = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)
        stds = torch.bmm(torch.bmm(s.transpose(1, 2), d.expand_as(adj)), s)

        mincut_loss = -torch.diagonal(stas, dim1=1, dim2=2).sum(dim=1)
        mincut_loss = mincut_loss / (torch.diagonal(stds, dim1=1, dim2=2).sum(dim=1) + 1e-8)

        # Orthogonality loss: clusters should be non-overlapping
        # ||S^T S / ||S^T S||_F - I/sqrt(n_clusters)||_F
        sts = torch.bmm(s.transpose(1, 2), s)
        sts_norm = sts / (torch.norm(sts, dim=(1, 2), keepdim=True) + 1e-8)
        identity = torch.eye(self.n_clusters, device=x.device).unsqueeze(0) / (self.n_clusters ** 0.5)
        ortho_loss = torch.norm(sts_norm - identity, dim=(1, 2))

        return x_pooled, adj_pooled, s, mincut_loss.mean(), ortho_loss.mean()
```

### Trading Strategy Implementation

```python
class GraphPoolingTradingStrategy:
    """
    Trading strategy based on hierarchical graph pooling.

    Generates signals from:
    1. Cluster-relative momentum (asset vs its cluster)
    2. Cluster correlation regime detection
    3. Hierarchical risk assessment
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

        # Tracking
        self.cluster_history = []
        self.correlation_regime = 'normal'

    def compute_correlation_graph(self, returns: np.ndarray) -> np.ndarray:
        """
        Build correlation graph from returns.

        Args:
            returns: [lookback, n_assets] return matrix

        Returns:
            adj: [n_assets, n_assets] adjacency matrix
        """
        corr = np.corrcoef(returns.T)

        # Threshold weak correlations
        adj = np.where(np.abs(corr) > 0.3, np.abs(corr), 0)
        np.fill_diagonal(adj, 0)

        return adj

    def compute_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Compute node features for each asset.

        Args:
            prices: [lookback, n_assets] price history
            volumes: [lookback, n_assets] volume history

        Returns:
            features: [n_assets, n_features]
        """
        returns = np.diff(np.log(prices), axis=0)

        features = []
        for i in range(self.n_assets):
            r = returns[:, i]
            v = volumes[1:, i]

            feat = [
                r[-1],                          # Latest return
                r[-5:].mean(),                  # 5-day momentum
                r.std(),                        # Volatility
                (r[-5:].mean() - r.mean()) / (r.std() + 1e-8),  # Momentum z-score
                v[-1] / (v.mean() + 1e-8),     # Volume ratio
                (r > 0).sum() / len(r),        # Win rate
                r.cumsum()[-1],                # Cumulative return
                np.corrcoef(r, v)[0, 1] if len(r) > 1 else 0  # Return-volume correlation
            ]
            features.append(feat)

        return np.array(features)

    def detect_correlation_regime(self, cluster_assignments: np.ndarray) -> str:
        """
        Detect if correlation regime is changing.

        When clusters become less distinct (more uniform assignments),
        it indicates correlation breakdown / risk-off regime.
        """
        # Measure cluster purity (how distinct are clusters)
        entropy = -np.sum(cluster_assignments * np.log(cluster_assignments + 1e-8), axis=-1)
        avg_entropy = entropy.mean()

        # Track entropy history
        self.cluster_history.append(avg_entropy)
        if len(self.cluster_history) > 20:
            self.cluster_history.pop(0)

        # Detect regime
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
        Compute how each asset deviates from its cluster's behavior.

        Assets outperforming their cluster: positive momentum
        Assets underperforming their cluster: negative momentum
        """
        # Get hard cluster assignments
        clusters = cluster_assignments.argmax(axis=-1)

        # Compute cluster returns (weighted average)
        n_clusters = cluster_assignments.shape[-1]
        cluster_returns = np.zeros(n_clusters)

        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() > 0:
                # Weight by cluster assignment probability
                weights = cluster_assignments[mask, c]
                weights = weights / weights.sum()
                cluster_returns[c] = (returns[-1, mask] * weights).sum()

        # Relative momentum = asset return - cluster return
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
        Generate trading signals.

        Args:
            prices: [lookback + 1, n_assets] price history
            volumes: [lookback + 1, n_assets] volume history

        Returns:
            signals: Dictionary with positions and metadata
        """
        # Compute inputs
        returns = np.diff(np.log(prices), axis=0)
        features = self.compute_features(prices, volumes)
        adj = self.compute_correlation_graph(returns)

        # Convert to tensors
        x = torch.FloatTensor(features).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, adj_tensor)

        # Extract predictions and clusters
        asset_pred = output['asset_predictions'].squeeze().numpy()
        cluster_assignments = output['cluster_assignments_l1'].squeeze().numpy()

        # Detect regime
        regime = self.detect_correlation_regime(cluster_assignments)

        # Compute relative momentum
        rel_momentum = self.compute_cluster_relative_momentum(returns, cluster_assignments)

        # === Generate positions ===
        positions = np.zeros(self.n_assets)

        if regime == 'correlation_breakdown':
            # Risk-off: reduce positions, only keep low-correlation assets
            # Find assets in least correlated cluster
            cluster_sizes = cluster_assignments.sum(axis=0)
            smallest_cluster = cluster_sizes.argmin()

            # Small positions in that cluster only
            mask = cluster_assignments[:, smallest_cluster] > 0.5
            positions[mask] = 0.3 * asset_pred[mask]

        elif regime == 'cluster_formation':
            # Clusters becoming more distinct: momentum within clusters
            # Go with cluster leaders
            for c in range(cluster_assignments.shape[-1]):
                mask = cluster_assignments[:, c] > 0.5
                if mask.sum() > 0:
                    cluster_leader = np.argmax(rel_momentum[mask])
                    positions[mask][cluster_leader] = asset_pred[mask][cluster_leader]
        else:
            # Normal regime: mean-reversion on relative momentum
            # Assets below cluster: buy (expect catch-up)
            # Assets above cluster: sell (expect mean reversion)
            positions = -rel_momentum * 0.5  # Contrarian
            positions += asset_pred * 0.5    # Model prediction

        # Normalize positions
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

### Backtesting Framework

```python
class GraphPoolingBacktester:
    """
    Backtesting framework for graph pooling strategy.
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
        Run backtest over specified period.

        Args:
            prices: DataFrame with asset prices
            volumes: DataFrame with asset volumes
            start_date: Start date string
            end_date: End date string

        Returns:
            results: Dictionary with performance metrics and history
        """
        lookback = self.strategy.lookback
        rebalance_freq = self.strategy.rebalance_freq

        # Align data
        dates = prices.loc[start_date:end_date].index

        # Initialize
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

            # Get historical window
            window_start = dates[i - lookback]
            price_window = prices.loc[window_start:date].values
            volume_window = volumes.loc[window_start:date].values

            # Generate signals on rebalance days
            if i % rebalance_freq == 0:
                signals = self.strategy.generate_signals(price_window, volume_window)
                new_positions = signals['positions']

                # Calculate turnover and costs
                turnover = np.abs(new_positions - positions).sum()
                costs = turnover * (self.transaction_cost + self.slippage)

                positions = new_positions
                portfolio_value *= (1 - costs)

                history['regimes'].append(signals['regime'])
            else:
                history['regimes'].append(history['regimes'][-1] if history['regimes'] else 'normal')

            # Calculate returns
            if i > lookback:
                daily_returns = prices.loc[date].values / prices.loc[dates[i-1]].values - 1
                portfolio_return = (positions * daily_returns).sum()
                portfolio_value *= (1 + portfolio_return)
                history['returns'].append(portfolio_return)

            history['dates'].append(date)
            history['portfolio_value'].append(portfolio_value)
            history['positions'].append(positions.copy())

        # Calculate metrics
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
        """Calculate maximum drawdown."""
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
        """Calculate Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = downside.std() * np.sqrt(252)
        if downside_std == 0:
            return np.inf
        return (returns.mean() * 252) / downside_std
```

## Key Metrics

### Model Metrics
| Metric | Description |
|--------|-------------|
| **Pooling Loss** | DiffPool link prediction and entropy loss |
| **Cluster Purity** | How distinct are learned clusters |
| **Hierarchy Stability** | Consistency of cluster assignments over time |
| **Prediction Accuracy** | Asset return prediction quality |

### Trading Metrics
| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted returns |
| **Sortino Ratio** | Downside-adjusted returns |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Return / Max Drawdown |
| **Turnover** | Average portfolio turnover |
| **Regime Detection Accuracy** | Correctness of regime calls |

### Cluster Metrics
| Metric | Description |
|--------|-------------|
| **Silhouette Score** | Cluster separation quality |
| **Cluster Return Spread** | Dispersion of cluster returns |
| **Intra-cluster Correlation** | Average correlation within clusters |
| **Cross-cluster Correlation** | Correlation between clusters |

## Dependencies

```python
# Graph Neural Networks
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# Data Processing
numpy>=1.23.0
pandas>=1.5.0

# Visualization
matplotlib>=3.6.0
networkx>=3.0
plotly>=5.0.0

# ML Utilities
scikit-learn>=1.2.0

# API
requests>=2.28.0
websockets>=10.0
```

## Expected Outcomes

1. **Hierarchical Market Model**: GNN that learns multi-scale market structure
2. **Cluster Discovery**: Automatic grouping of correlated assets
3. **Regime Detection**: Early warning for correlation regime changes
4. **Trading Strategy**: Cluster-relative momentum with regime adaptation
5. **Risk Management**: Position sizing based on hierarchy levels

## Rust Implementation

See the `rust/` directory for a high-performance implementation with Bybit API integration:

```
rust/
├── src/
│   ├── lib.rs                 # Main library
│   ├── api/                   # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── types.rs
│   ├── graph/                 # Graph structures
│   │   ├── mod.rs
│   │   ├── market_graph.rs
│   │   └── correlation.rs
│   ├── pooling/               # Pooling algorithms
│   │   ├── mod.rs
│   │   ├── diffpool.rs
│   │   ├── sagpool.rs
│   │   └── mincut.rs
│   ├── strategy/              # Trading strategy
│   │   ├── mod.rs
│   │   └── hierarchical.rs
│   └── bin/                   # Example binaries
│       ├── demo.rs
│       └── backtest.rs
├── Cargo.toml
└── README.md
```

## References

### Academic Papers
- [Hierarchical Graph Representation Learning with DiffPool](https://arxiv.org/abs/1806.08804) - Ying et al., 2018
- [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) - Lee et al., 2019
- [Spectral Clustering and the High-dimensional Stochastic Blockmodel](https://arxiv.org/abs/1007.1684) - Rohe et al., 2011
- [Graph Neural Networks for Financial Time Series](https://arxiv.org/abs/2104.12389)

### Documentation
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)

## Difficulty Level

Rating: Expert

### Prerequisites
- **Graph Neural Networks**: Message passing, node embeddings, graph convolutions
- **Spectral Graph Theory**: Laplacian, graph cuts, spectral clustering
- **Deep Learning**: Backpropagation through discrete operations, attention mechanisms
- **Financial Markets**: Correlation structures, regime changes, portfolio theory
- **Software Engineering**: Rust async programming, API integration
