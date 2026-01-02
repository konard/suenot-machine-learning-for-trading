# Chapter 342: Equivariant Graph Neural Networks for Cryptocurrency Trading

## Overview

Equivariant Graph Neural Networks (E-GNNs) represent a paradigm shift in geometric deep learning, designed to respect the symmetries inherent in data. In cryptocurrency trading, assets form complex relational structures where price movements, correlations, and market dynamics exhibit geometric properties. E-GNNs preserve these symmetries during learning, enabling more robust pattern recognition and improved generalization across market regimes.

The key insight is that financial markets have inherent symmetries: scaling invariance (doubling all prices doesn't change relative returns), permutation equivariance (asset ordering shouldn't affect predictions), and rotational invariance in feature space (correlated assets should behave similarly regardless of their absolute feature values).

## Trading Strategy

**Core Concept:** Model cryptocurrency market as a dynamic graph where nodes represent assets, edges encode correlations and trading relationships, and node/edge features capture market microstructure. The E-GNN learns trading signals while respecting the geometric structure of this financial graph.

**Key Advantages for Trading:**
1. **Symmetry Preservation** — Model learns patterns invariant to irrelevant transformations (scale, permutation)
2. **Correlation Capture** — Graph structure naturally encodes asset correlations and market regimes
3. **Geometric Features** — Embeddings preserve distance relationships in feature space
4. **Robustness** — Equivariance provides built-in regularization against spurious patterns

**Edge:** E-GNNs filter out noise from coordinate system choices, focusing on genuine market patterns that persist across different representations.

## Technical Specification

### Equivariance in Finance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    E(n) Equivariance in Trading                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  What is Equivariance?                                                  │
│  ────────────────────                                                   │
│                                                                         │
│  A function f is G-equivariant if:                                      │
│                                                                         │
│     f(g · x) = g · f(x)  for all g ∈ G                                  │
│                                                                         │
│  Transforming input then applying f = Applying f then transforming     │
│                                                                         │
│                                                                         │
│  Financial Symmetries:                                                  │
│  ────────────────────                                                   │
│                                                                         │
│  1. Permutation Equivariance:                                           │
│     - Reordering assets shouldn't change predictions                    │
│     - BTC, ETH, SOL → SOL, BTC, ETH gives same relative signals        │
│                                                                         │
│  2. Scale Invariance:                                                   │
│     - Returns matter, not absolute prices                               │
│     - $100 → $110 same pattern as $1000 → $1100                        │
│                                                                         │
│  3. Translation Invariance:                                             │
│     - Relative positions matter, not absolute                           │
│     - Feature differences drive predictions, not levels                 │
│                                                                         │
│  4. Rotation Invariance (in feature space):                             │
│     - Correlated asset clusters preserve structure                      │
│     - PCA rotation shouldn't change predictions                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│            Equivariant GNN for Cryptocurrency Trading                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Asset Graph G = (V, E)                                          │
│  ────────────────────────────                                           │
│  • Nodes V: Cryptocurrency assets (BTC, ETH, SOL, ...)                 │
│  • Edges E: Correlation/causality links between assets                  │
│  • Node features h_i: Price, volume, volatility, momentum               │
│  • Coordinates x_i: Position in feature embedding space                 │
│                                                                         │
│                    ┌─────────────────────┐                              │
│                    │   Asset Graph       │                              │
│                    │   Construction      │                              │
│                    └──────────┬──────────┘                              │
│                               │                                         │
│                               ▼                                         │
│              ┌────────────────────────────────┐                         │
│              │   Input Embedding Layer        │                         │
│              │   h_i → (h_i, x_i)             │                         │
│              └───────────────┬────────────────┘                         │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                      │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │  E-GNN      │     │  E-GNN      │     │  E-GNN      │              │
│   │  Layer 1    │────▶│  Layer 2    │────▶│  Layer L    │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                  │                      │
│                                                  ▼                      │
│                              ┌────────────────────────────┐             │
│                              │   Invariant Aggregation    │             │
│                              │   (Graph-level pooling)    │             │
│                              └───────────────┬────────────┘             │
│                                              │                          │
│                      ┌───────────────────────┼───────────────────────┐  │
│                      │                       │                       │  │
│                      ▼                       ▼                       ▼  │
│               ┌────────────┐         ┌────────────┐         ┌──────────┐│
│               │ Direction  │         │ Position   │         │ Risk     ││
│               │ Prediction │         │ Sizing     │         │ Estimate ││
│               └────────────┘         └────────────┘         └──────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### EGNN Layer Mathematics

The Equivariant Graph Neural Network layer updates both node features and coordinates:

```python
class EGNNLayer:
    """
    E(n) Equivariant Graph Neural Network Layer

    Updates:
    1. Messages: m_ij = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)
    2. Coordinates: x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)
    3. Features: h_i' = φ_h(h_i, Σ_j m_ij)

    Key: Coordinate updates use relative positions (x_i - x_j),
         ensuring translation equivariance
    """

    def __init__(self, hidden_dim, edge_dim=0, act_fn=SiLU,
                 coords_agg='mean', update_coords=True):
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Edge MLP: computes messages
        self.edge_mlp = Sequential(
            Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            act_fn(),
            Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        # Node MLP: updates node features
        self.node_mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            act_fn(),
            Linear(hidden_dim, hidden_dim)
        )

        # Coordinate MLP: updates coordinates
        if update_coords:
            self.coord_mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                act_fn(),
                Linear(hidden_dim, 1, bias=False)
            )

        self.coords_agg = coords_agg

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: Node features [N, hidden_dim]
        x: Node coordinates [N, coord_dim]
        edge_index: Edge indices [2, E]
        edge_attr: Edge features [E, edge_dim]
        """
        row, col = edge_index

        # Compute squared distances (invariant scalar)
        coord_diff = x[row] - x[col]  # [E, coord_dim]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Edge features
        edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Compute messages
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Update coordinates (equivariant)
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # [E, 1]
            coord_update = coord_diff * coord_weights  # [E, coord_dim]

            # Aggregate coordinate updates
            x_new = x + scatter_mean(coord_update, row, dim=0, dim_size=x.size(0))
        else:
            x_new = x

        # Aggregate messages
        m_i = scatter_sum(m_ij, row, dim=0, dim_size=h.size(0))  # [N, hidden_dim]

        # Update node features
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_new = h + h_new  # Residual connection

        return h_new, x_new
```

### Financial Graph Construction

```python
class CryptoMarketGraph:
    """
    Constructs a dynamic graph representing cryptocurrency market structure.

    Nodes: Individual assets (BTC, ETH, SOL, etc.)
    Edges: Based on correlation, causality, or sector relationships
    """

    def __init__(self, correlation_threshold=0.3, window_size=168):
        self.corr_threshold = correlation_threshold
        self.window_size = window_size  # 1 week of hourly data

    def build_graph(self, returns_df, orderbook_features=None):
        """
        Build market graph from return series.

        Args:
            returns_df: DataFrame with asset returns [time, assets]
            orderbook_features: Optional order book data

        Returns:
            Graph with node/edge features and coordinates
        """
        n_assets = len(returns_df.columns)

        # Node features: Technical indicators per asset
        node_features = self._compute_node_features(returns_df)

        # Initial coordinates: PCA embedding of returns
        coords = self._compute_initial_coords(returns_df)

        # Edges: Based on rolling correlation
        edge_index, edge_attr = self._build_edges(returns_df)

        return {
            'x': coords,
            'h': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'asset_names': list(returns_df.columns)
        }

    def _compute_node_features(self, returns_df):
        """Compute per-asset features"""
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
        """Embed assets in feature space using PCA"""
        from sklearn.decomposition import PCA

        # Compute correlation matrix
        corr_matrix = returns_df.corr().values

        # PCA embedding
        pca = PCA(n_components=dim)
        coords = pca.fit_transform(corr_matrix)

        return torch.tensor(coords, dtype=torch.float32)

    def _build_edges(self, returns_df):
        """Build edges based on correlation"""
        corr_matrix = returns_df.corr().values
        n = len(returns_df.columns)

        edges = []
        edge_features = []

        for i in range(n):
            for j in range(n):
                if i != j and abs(corr_matrix[i, j]) > self.corr_threshold:
                    edges.append([i, j])
                    edge_features.append([
                        corr_matrix[i, j],  # Correlation
                        abs(corr_matrix[i, j]),  # Absolute correlation
                        1 if corr_matrix[i, j] > 0 else -1  # Sign
                    ])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return edge_index, edge_attr

    def _compute_rsi(self, returns, period=14):
        delta = returns
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return (100 - 100 / (1 + rs)).iloc[-1]

    def _compute_macd_signal(self, returns):
        prices = (1 + returns).cumprod()
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]
```

### Complete E-GNN Trading Model

```python
class EquivariantGNNTrader:
    """
    Complete E-GNN model for cryptocurrency trading.

    Architecture:
    1. Graph construction from market data
    2. Multi-layer E-GNN for feature extraction
    3. Invariant pooling for graph-level representation
    4. Multi-head output (direction, sizing, risk)
    """

    def __init__(
        self,
        input_dim: int = 10,       # Node feature dimension
        hidden_dim: int = 64,      # Hidden layer size
        coord_dim: int = 3,        # Coordinate dimension
        n_layers: int = 4,         # Number of E-GNN layers
        output_classes: int = 3,   # Long, Hold, Short
        dropout: float = 0.1
    ):
        # Input embedding
        self.node_embed = Linear(input_dim, hidden_dim)

        # E-GNN layers
        self.egnn_layers = ModuleList([
            EGNNLayer(hidden_dim, edge_dim=3, update_coords=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        # Normalization
        self.layer_norms = ModuleList([
            LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.dropout = Dropout(dropout)

        # Invariant pooling
        self.graph_pool = AttentionalPooling(hidden_dim)

        # Output heads
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
            Linear(hidden_dim // 2, 2)  # Volatility, VaR
        )

    def forward(self, graph):
        """
        Forward pass through E-GNN trader.

        Args:
            graph: Dict with 'h', 'x', 'edge_index', 'edge_attr', 'batch'

        Returns:
            Dict with 'direction', 'position_size', 'risk_metrics'
        """
        h = graph['h']
        x = graph['x']
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_attr', None)
        batch = graph.get('batch', torch.zeros(h.size(0), dtype=torch.long))

        # Embed input features
        h = self.node_embed(h)

        # E-GNN layers
        for i, (egnn, norm) in enumerate(zip(self.egnn_layers, self.layer_norms)):
            h_new, x = egnn(h, x, edge_index, edge_attr)
            h = norm(h_new)
            h = self.dropout(h)

        # Graph-level pooling (invariant)
        graph_repr = self.graph_pool(h, batch)

        # Predictions
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

    def predict_signals(self, graph, threshold=0.4):
        """Generate trading signals from model output"""
        output = self.forward(graph)
        probs = output['direction']

        signals = []
        for i in range(probs.size(0)):
            if probs[i, 2] > threshold:  # Long
                signals.append(1)
            elif probs[i, 0] > threshold:  # Short
                signals.append(-1)
            else:
                signals.append(0)  # Hold

        return signals, output['position_size']


class AttentionalPooling(Module):
    """
    Attention-based invariant pooling over graph nodes.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = Sequential(
            Linear(hidden_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, 1)
        )

    def forward(self, h, batch):
        """
        h: Node features [N, hidden_dim]
        batch: Batch assignment [N]
        """
        # Compute attention weights
        attn_weights = self.attention(h)  # [N, 1]
        attn_weights = scatter_softmax(attn_weights, batch, dim=0)

        # Weighted sum
        weighted = h * attn_weights
        pooled = scatter_sum(weighted, batch, dim=0)

        return pooled
```

### Training Pipeline

```python
class EGNNTrainingPipeline:
    """
    Training pipeline for Equivariant GNN Trader
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
        """Single training step"""
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
        Multi-task loss function:
        1. Direction classification (focal loss for imbalance)
        2. Sharpe ratio proxy
        3. Risk prediction
        """
        # Direction loss (focal)
        direction_loss = self._focal_loss(
            output['direction_logits'],
            labels['direction'],
            gamma=2.0
        )

        # Sharpe-inspired loss
        positions = output['direction'][:, 2] - output['direction'][:, 0]
        strategy_returns = positions * labels['future_returns']
        sharpe_loss = -strategy_returns.mean() / (strategy_returns.std() + 1e-8)

        # Risk prediction loss
        vol_loss = F.mse_loss(output['volatility_pred'].squeeze(),
                              labels['realized_vol'])

        # Position sizing loss
        size_loss = F.mse_loss(
            output['position_size'].squeeze(),
            labels['optimal_size']
        )

        # Combine losses
        total = direction_loss + 0.3 * sharpe_loss + 0.2 * vol_loss + 0.1 * size_loss

        return total

    def _focal_loss(self, logits, targets, gamma=2.0, alpha=None):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()

    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for graph, labels in val_loader:
                output = self.model(graph)
                loss = self._compute_loss(output, labels)
                total_loss += loss.item()

                # Direction accuracy
                preds = output['direction'].argmax(dim=-1)
                correct += (preds == labels['direction']).sum().item()
                total += preds.size(0)

        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total
        }
```

### Backtesting Framework

```python
class EGNNBacktester:
    """
    Backtesting framework for E-GNN trading signals.
    """

    def __init__(
        self,
        model: EquivariantGNNTrader,
        graph_builder: CryptoMarketGraph,
        fee_rate: float = 0.0004,  # Bybit taker
        slippage: float = 0.0001
    ):
        self.model = model
        self.graph_builder = graph_builder
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        initial_capital: float = 10000,
        max_position: float = 0.5  # Max 50% of capital
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            price_data: OHLCV data for multiple assets
            initial_capital: Starting capital in USD
            max_position: Maximum position size as fraction

        Returns:
            Dict with performance metrics and equity curve
        """
        self.model.eval()

        results = {
            'timestamps': [],
            'equity': [],
            'positions': [],
            'trades': [],
            'returns': []
        }

        capital = initial_capital
        positions = {}  # asset -> (direction, size, entry_price)

        # Rolling window backtest
        for t in range(168, len(price_data) - 1):  # Start after 1 week
            window_data = price_data.iloc[t-168:t]
            current_prices = price_data.iloc[t]
            next_prices = price_data.iloc[t+1]

            # Build graph from recent data
            returns_df = window_data.pct_change().dropna()
            graph = self.graph_builder.build_graph(returns_df)

            # Get model predictions
            with torch.no_grad():
                signals, sizes = self.model.predict_signals(graph)

            # Execute trades
            daily_pnl = 0
            for i, asset in enumerate(graph['asset_names']):
                signal = signals[i]
                size = sizes[i].item() * max_position
                current_price = current_prices[asset]

                # Close existing position if signal changed
                if asset in positions:
                    old_dir, old_size, entry_price = positions[asset]
                    if signal != old_dir:
                        # Close position
                        pnl = old_dir * old_size * (current_price - entry_price) / entry_price
                        pnl -= self.fee_rate + self.slippage
                        daily_pnl += pnl * capital

                        results['trades'].append({
                            'asset': asset,
                            'exit_time': t,
                            'pnl': pnl
                        })
                        del positions[asset]

                # Open new position
                if signal != 0 and asset not in positions:
                    positions[asset] = (signal, size, current_price)
                    daily_pnl -= (self.fee_rate + self.slippage) * size * capital

                    results['trades'].append({
                        'asset': asset,
                        'entry_time': t,
                        'direction': signal,
                        'size': size
                    })

            # Mark-to-market
            unrealized_pnl = 0
            for asset, (direction, size, entry_price) in positions.items():
                current_price = price_data.iloc[t][asset]
                unrealized_pnl += direction * size * (current_price - entry_price) / entry_price

            capital += daily_pnl
            equity = capital * (1 + unrealized_pnl)

            results['timestamps'].append(t)
            results['equity'].append(equity)
            results['positions'].append(dict(positions))
            results['returns'].append(daily_pnl / capital if capital > 0 else 0)

        return self._compute_metrics(results, initial_capital)

    def _compute_metrics(self, results, initial_capital):
        """Compute trading performance metrics"""
        equity = np.array(results['equity'])
        returns = np.array(results['returns'])

        # Total return
        total_return = (equity[-1] / initial_capital) - 1

        # Sharpe ratio (hourly to annual)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 365)

        # Sortino ratio
        downside = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(24 * 365)

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        closed_trades = [t for t in results['trades'] if 'pnl' in t]
        if closed_trades:
            wins = sum(1 for t in closed_trades if t['pnl'] > 0)
            win_rate = wins / len(closed_trades)
        else:
            win_rate = 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(closed_trades),
            'equity_curve': equity,
            'timestamps': results['timestamps']
        }
```

### Key Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Direction Accuracy | > 55% | Correct prediction of price movement direction |
| Sharpe Ratio | > 2.0 | Risk-adjusted returns (annualized) |
| Max Drawdown | < 15% | Largest peak-to-trough decline |
| Win Rate | > 50% | Percentage of profitable trades |
| Profit Factor | > 1.5 | Gross profit / gross loss ratio |

### E-GNN Variants for Trading

| Variant | Coord Update | Edge Features | Use Case |
|---------|--------------|---------------|----------|
| E-GNN Basic | Yes | Distance only | General market structure |
| SE(3)-GNN | Rotation equivariant | Full geometric | Complex correlations |
| Temporal E-GNN | Time-aware | Lag features | Regime detection |
| Hierarchical E-GNN | Multi-scale | Sector edges | Market microstructure |

### Dependencies

```toml
[dependencies]
# Core ML
torch = ">=2.0.0"
torch-geometric = ">=2.4.0"
numpy = ">=1.23.0"

# E-GNN specific
e3nn = ">=0.5.0"  # Optional: for SE(3) equivariance

# Data handling
pandas = ">=2.0.0"
polars = ">=0.19.0"

# Bybit API
pybit = ">=5.6.0"

# Visualization
matplotlib = ">=3.6.0"
networkx = ">=3.0"
```

## Expected Outcomes

1. **E-GNN Trading Model** — Equivariant architecture respecting market symmetries
2. **Dynamic Graph Construction** — Correlation-based asset graphs from market data
3. **Multi-Asset Signal Generation** — Portfolio-level trading decisions
4. **Robust Backtesting** — Performance evaluation on Bybit historical data
5. **Risk Management** — Position sizing with volatility prediction

## References

- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) — Original EGNN paper
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) — Comprehensive GDL survey
- [Graph Neural Networks for Financial Market Prediction](https://arxiv.org/abs/2106.06272)
- [Equivariant Architectures for Learning in Deep Weight Spaces](https://arxiv.org/abs/2301.12780)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

**Prerequisites:** Graph Neural Networks, Group Theory basics, Geometric Deep Learning, Financial market microstructure, Cryptocurrency trading
