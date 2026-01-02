# Chapter 340: Associative Memory Trading — Retrieval-Based Predictions with Dense Associative Networks

## Overview

Associative Memory networks represent a powerful paradigm for pattern recognition and retrieval in trading. Unlike traditional feedforward neural networks that learn input-output mappings, associative memories store patterns and retrieve the closest match when presented with partial or noisy input. This makes them particularly suited for recognizing market patterns and making predictions based on historical similarity.

## Core Concepts

### What is Associative Memory?

Associative memory is a content-addressable memory system that:
1. **Stores** a set of patterns during training
2. **Retrieves** the most similar stored pattern when given a query
3. **Completes** partial patterns by filling in missing information
4. **Corrects** noisy inputs by converging to the nearest stored pattern

### From Hopfield Networks to Modern Associative Memory

```
Classical Hopfield (1982):
├── Binary patterns: {-1, +1}
├── Energy-based dynamics
├── Limited capacity: ~0.14N patterns (N = neurons)
└── Converges to stored attractors

Modern Dense Associative Memory (2016):
├── Continuous patterns: ℝ^d
├── Exponential storage capacity
├── Attention-like retrieval mechanism
└── Differentiable for deep learning integration
```

### Why Associative Memory for Trading?

1. **Pattern Matching**: Markets exhibit recurring patterns; AM finds the most similar historical regime
2. **Noise Robustness**: Real market data is noisy; AM naturally filters noise
3. **Interpretability**: Retrieved patterns provide context for predictions
4. **Memory Efficiency**: Store representative patterns instead of all data
5. **One-Shot Learning**: Learn from rare but important market events

## Trading Strategy

**Strategy Overview:** Use Dense Associative Memory to identify similar historical market patterns and predict future price movements based on what happened after those patterns.

### Signal Generation

```
1. Feature Extraction:
   - Compute market features: returns, volatility, volume patterns
   - Normalize to create pattern vector

2. Pattern Retrieval:
   - Query the associative memory with current pattern
   - Retrieve K most similar historical patterns
   - Weight by similarity score

3. Prediction:
   - Aggregate outcomes from retrieved patterns
   - Generate directional signal with confidence

4. Position Sizing:
   - Scale position by retrieval confidence
   - Higher similarity = larger position
```

### Entry Signals

- **Long Signal**: Retrieved patterns predominantly followed by positive returns
- **Short Signal**: Retrieved patterns predominantly followed by negative returns
- **Confidence Threshold**: Only trade when similarity score exceeds threshold

### Risk Management

- **Novelty Detection**: Low retrieval similarity indicates novel market conditions → reduce exposure
- **Consensus Check**: Multiple retrieved patterns should agree on direction
- **Volatility Scaling**: Adjust position size based on expected volatility from retrieved patterns

## Technical Specification

### Mathematical Foundation

#### Classical Hopfield Energy

For binary patterns x ∈ {-1, +1}^N with weight matrix W:

```
E(x) = -½ x^T W x

Update rule (asynchronous):
x_i ← sign(Σ_j W_ij x_j)

Weight learning (Hebbian):
W_ij = (1/P) Σ_μ ξ_i^μ ξ_j^μ
```

#### Modern Dense Associative Memory

For patterns {ξ^μ} ∈ ℝ^d, the energy function becomes:

```
E(x) = -log Σ_μ exp(β x · ξ^μ)

Retrieval dynamics:
x_new = Σ_μ softmax(β x · ξ^μ) ξ^μ

This is equivalent to attention!
```

#### Connection to Attention Mechanism

```
Query:    q = W_q x
Keys:     K = [ξ^1, ξ^2, ..., ξ^M]
Values:   V = [v^1, v^2, ..., v^M]

Attention output:
output = Σ_μ softmax(q · ξ^μ / √d) v^μ
```

### Architecture Diagram

```
                    Market Data Stream
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Feature Engineering      │
            │  ├── Returns & Volatility   │
            │  ├── Technical Indicators   │
            │  ├── Volume Patterns        │
            │  └── Cross-Asset Features   │
            └──────────────┬──────────────┘
                           │
                           ▼ Query Pattern x
            ┌─────────────────────────────┐
            │   Dense Associative Memory  │
            │                             │
            │  ┌───────────────────────┐  │
            │  │   Pattern Memory      │  │
            │  │   ξ^1, ξ^2, ..., ξ^M  │  │
            │  │   (Historical States) │  │
            │  └───────────────────────┘  │
            │            ↓ ↓ ↓            │
            │  ┌───────────────────────┐  │
            │  │  Similarity Scoring   │  │
            │  │  softmax(β x · ξ^μ)   │  │
            │  └───────────────────────┘  │
            │            ↓ ↓ ↓            │
            │  ┌───────────────────────┐  │
            │  │  Weighted Retrieval   │  │
            │  │  Σ α_μ v^μ            │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Predicted  │ │ Retrieval   │ │   Similar   │
     │  Returns    │ │ Confidence  │ │  Patterns   │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Trading Decision        │
            │  ├── Signal Direction       │
            │  ├── Position Size          │
            │  └── Risk Parameters        │
            └─────────────────────────────┘
```

### Feature Engineering for Pattern Storage

```python
def compute_market_pattern(df, lookback=20):
    """
    Create market pattern vector for associative memory
    """
    pattern = {}

    # Return-based features
    returns = df['close'].pct_change()
    pattern['return_mean'] = returns.rolling(lookback).mean().iloc[-1]
    pattern['return_std'] = returns.rolling(lookback).std().iloc[-1]
    pattern['return_skew'] = returns.rolling(lookback).skew().iloc[-1]
    pattern['return_kurt'] = returns.rolling(lookback).kurt().iloc[-1]

    # Trend features
    pattern['trend_5'] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
    pattern['trend_20'] = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)

    # Volatility features
    pattern['volatility'] = returns.rolling(lookback).std().iloc[-1] * np.sqrt(252)
    pattern['volatility_change'] = (
        returns.rolling(5).std().iloc[-1] /
        returns.rolling(20).std().iloc[-1]
    )

    # Volume features
    volume_ma = df['volume'].rolling(lookback).mean()
    pattern['volume_ratio'] = df['volume'].iloc[-1] / volume_ma.iloc[-1]

    # Range features
    atr = (df['high'] - df['low']).rolling(lookback).mean()
    pattern['atr_ratio'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / atr.iloc[-1]

    # Close position in range
    pattern['close_position'] = (
        (df['close'].iloc[-1] - df['low'].iloc[-lookback:].min()) /
        (df['high'].iloc[-lookback:].max() - df['low'].iloc[-lookback:].min())
    )

    return np.array(list(pattern.values()))
```

### Dense Associative Memory Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseAssociativeMemory(nn.Module):
    """
    Dense Associative Memory for pattern retrieval

    Stores patterns and retrieves based on similarity,
    using softmax attention for smooth differentiable retrieval.
    """

    def __init__(self, pattern_dim: int, memory_size: int,
                 beta: float = 1.0, n_heads: int = 4):
        super().__init__()

        self.pattern_dim = pattern_dim
        self.memory_size = memory_size
        self.beta = beta
        self.n_heads = n_heads

        # Learnable pattern memory
        self.patterns = nn.Parameter(torch.randn(memory_size, pattern_dim))
        self.values = nn.Parameter(torch.randn(memory_size, pattern_dim))

        # Multi-head attention components
        self.head_dim = pattern_dim // n_heads
        self.W_q = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_k = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_v = nn.Linear(pattern_dim, pattern_dim, bias=False)
        self.W_o = nn.Linear(pattern_dim, pattern_dim, bias=False)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(pattern_dim, pattern_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(pattern_dim // 2, 1),
            nn.Tanh()
        )

    def store_patterns(self, patterns: torch.Tensor, values: torch.Tensor):
        """
        Store patterns and their associated values (labels/outcomes)
        """
        with torch.no_grad():
            # Normalize patterns
            patterns_norm = F.normalize(patterns, dim=-1)
            self.patterns.data = patterns_norm
            self.values.data = values

    def retrieve(self, query: torch.Tensor, return_attention: bool = False):
        """
        Retrieve from memory using attention mechanism

        Args:
            query: (batch, pattern_dim) query pattern
            return_attention: whether to return attention weights

        Returns:
            retrieved: (batch, pattern_dim) retrieved pattern
            attention: (batch, memory_size) attention weights (optional)
        """
        batch_size = query.shape[0]

        # Project query
        q = self.W_q(query)  # (batch, pattern_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.n_heads, self.head_dim)

        # Project memory
        k = self.W_k(self.patterns)  # (memory_size, pattern_dim)
        v = self.W_v(self.values)    # (memory_size, pattern_dim)

        k = k.view(1, self.memory_size, self.n_heads, self.head_dim)
        v = v.view(1, self.memory_size, self.n_heads, self.head_dim)

        # Compute attention scores
        # q: (batch, n_heads, head_dim)
        # k: (1, memory_size, n_heads, head_dim)
        scores = torch.einsum('bhd,bmhd->bhm', q, k)
        scores = scores * self.beta / (self.head_dim ** 0.5)

        # Softmax attention
        attention = F.softmax(scores, dim=-1)  # (batch, n_heads, memory_size)

        # Retrieve values
        # v: (1, memory_size, n_heads, head_dim)
        retrieved = torch.einsum('bhm,bmhd->bhd', attention, v)
        retrieved = retrieved.view(batch_size, self.pattern_dim)

        # Output projection
        retrieved = self.W_o(retrieved)

        if return_attention:
            # Average attention across heads
            attention_avg = attention.mean(dim=1)  # (batch, memory_size)
            return retrieved, attention_avg

        return retrieved

    def forward(self, query: torch.Tensor):
        """
        Retrieve and predict

        Args:
            query: (batch, pattern_dim) current market pattern

        Returns:
            prediction: (batch, 1) predicted direction
            confidence: (batch, 1) retrieval confidence
        """
        retrieved, attention = self.retrieve(query, return_attention=True)

        # Prediction from retrieved pattern
        prediction = self.predictor(retrieved)

        # Confidence based on attention concentration
        # High entropy = low confidence (no clear match)
        entropy = -(attention * (attention + 1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.memory_size, dtype=torch.float32))
        confidence = 1 - (entropy / max_entropy)

        return prediction, confidence

    def get_similar_patterns(self, query: torch.Tensor, k: int = 5):
        """
        Get k most similar stored patterns

        Args:
            query: (pattern_dim,) query pattern
            k: number of patterns to retrieve

        Returns:
            indices: (k,) indices of similar patterns
            similarities: (k,) similarity scores
        """
        query = F.normalize(query.unsqueeze(0), dim=-1)
        patterns = F.normalize(self.patterns, dim=-1)

        similarities = (query @ patterns.T).squeeze(0)
        topk = torch.topk(similarities, k=min(k, self.memory_size))

        return topk.indices, topk.values


class AssociativeMemoryTrader:
    """
    Trading system based on Dense Associative Memory
    """

    def __init__(self, memory: DenseAssociativeMemory,
                 confidence_threshold: float = 0.3,
                 position_scale: float = 1.0):
        self.memory = memory
        self.confidence_threshold = confidence_threshold
        self.position_scale = position_scale

    def generate_signal(self, current_pattern: torch.Tensor):
        """
        Generate trading signal from current market pattern

        Returns:
            signal: float in [-1, 1], direction and strength
            confidence: float in [0, 1], retrieval confidence
            similar_patterns: list of (index, similarity) tuples
        """
        self.memory.eval()

        with torch.no_grad():
            prediction, confidence = self.memory(current_pattern.unsqueeze(0))
            indices, similarities = self.memory.get_similar_patterns(
                current_pattern, k=5
            )

        prediction = prediction.item()
        confidence = confidence.item()

        # Only trade if confidence exceeds threshold
        if confidence < self.confidence_threshold:
            return 0.0, confidence, list(zip(indices.tolist(), similarities.tolist()))

        # Scale signal by confidence
        signal = prediction * confidence * self.position_scale

        return signal, confidence, list(zip(indices.tolist(), similarities.tolist()))
```

### Continuous Hopfield Network (Modern Version)

```python
class ContinuousHopfieldNetwork(nn.Module):
    """
    Continuous Hopfield Network with exponential storage capacity

    Based on "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, beta: float = None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.beta = beta if beta else (1.0 / (hidden_dim ** 0.5))

        # Pattern projection
        self.W_pattern = nn.Linear(input_dim, hidden_dim)
        self.W_query = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.W_out = nn.Linear(hidden_dim, input_dim)

        # Pattern memory (set during training)
        self.register_buffer('stored_patterns', None)

    def store(self, patterns: torch.Tensor):
        """
        Store patterns in memory

        Args:
            patterns: (N, input_dim) patterns to store
        """
        with torch.no_grad():
            projected = self.W_pattern(patterns)  # (N, hidden_dim)
            self.stored_patterns = F.normalize(projected, dim=-1)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of current state

        E(x) = -log Σ exp(β ξ^T x)

        Lower energy = closer to stored patterns
        """
        if self.stored_patterns is None:
            raise ValueError("No patterns stored")

        state_proj = self.W_query(state)  # (batch, hidden_dim)
        state_proj = F.normalize(state_proj, dim=-1)

        # Compute similarities
        similarities = state_proj @ self.stored_patterns.T  # (batch, N)

        # Log-sum-exp for energy
        energy = -torch.logsumexp(self.beta * similarities, dim=-1)

        return energy

    def update(self, state: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """
        Update state towards stored patterns

        Uses softmax-weighted combination of patterns
        """
        if self.stored_patterns is None:
            raise ValueError("No patterns stored")

        for _ in range(n_steps):
            state_proj = self.W_query(state)
            state_proj = F.normalize(state_proj, dim=-1)

            # Attention over stored patterns
            similarities = state_proj @ self.stored_patterns.T
            attention = F.softmax(self.beta * similarities, dim=-1)

            # Weighted combination
            new_state_proj = attention @ self.stored_patterns

            # Project back
            state = self.W_out(new_state_proj)

        return state

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve closest pattern to query
        """
        return self.update(query, n_steps=3)


class HopfieldPooling(nn.Module):
    """
    Hopfield-based pooling for sequence encoding

    Useful for encoding price sequences into fixed-size patterns
    """

    def __init__(self, input_dim: int, n_patterns: int, beta: float = None):
        super().__init__()

        self.input_dim = input_dim
        self.n_patterns = n_patterns
        self.beta = beta if beta else (1.0 / (input_dim ** 0.5))

        # Learnable prototype patterns (queries)
        self.prototypes = nn.Parameter(torch.randn(n_patterns, input_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence into fixed-size representation

        Args:
            sequence: (batch, seq_len, input_dim)

        Returns:
            pooled: (batch, n_patterns * input_dim)
        """
        batch_size, seq_len, _ = sequence.shape

        # Normalize
        seq_norm = F.normalize(sequence, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        # Compute attention: each prototype attends to sequence
        # proto_norm: (n_patterns, input_dim)
        # seq_norm: (batch, seq_len, input_dim)
        attention = torch.einsum('pd,bsd->bps', proto_norm, seq_norm)
        attention = F.softmax(self.beta * attention, dim=-1)  # (batch, n_patterns, seq_len)

        # Weighted sum of sequence
        pooled = torch.einsum('bps,bsd->bpd', attention, sequence)  # (batch, n_patterns, input_dim)

        return pooled.view(batch_size, -1)
```

### Pattern Memory Management

```python
class PatternMemoryManager:
    """
    Manages pattern storage with capacity limits and relevance scoring
    """

    def __init__(self, max_patterns: int, pattern_dim: int,
                 similarity_threshold: float = 0.95):
        self.max_patterns = max_patterns
        self.pattern_dim = pattern_dim
        self.similarity_threshold = similarity_threshold

        self.patterns = []
        self.outcomes = []
        self.timestamps = []
        self.retrieval_counts = []

    def add_pattern(self, pattern: np.ndarray, outcome: float, timestamp):
        """
        Add pattern to memory, potentially replacing old/similar ones
        """
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)

        # Check for similar existing patterns
        if len(self.patterns) > 0:
            patterns_arr = np.array(self.patterns)
            similarities = patterns_arr @ pattern

            # If very similar pattern exists, update it instead
            if np.max(similarities) > self.similarity_threshold:
                idx = np.argmax(similarities)
                # Exponential moving average update
                alpha = 0.3
                self.patterns[idx] = alpha * pattern + (1 - alpha) * self.patterns[idx]
                self.patterns[idx] /= np.linalg.norm(self.patterns[idx])
                self.outcomes[idx] = alpha * outcome + (1 - alpha) * self.outcomes[idx]
                return

        # Add new pattern
        if len(self.patterns) >= self.max_patterns:
            self._evict_pattern()

        self.patterns.append(pattern)
        self.outcomes.append(outcome)
        self.timestamps.append(timestamp)
        self.retrieval_counts.append(0)

    def _evict_pattern(self):
        """
        Remove least useful pattern based on recency and usage
        """
        if len(self.patterns) == 0:
            return

        # Score based on recency and usage
        n = len(self.patterns)
        recency_scores = np.arange(n) / n  # Older = lower score
        usage_scores = np.array(self.retrieval_counts)
        usage_scores = usage_scores / (usage_scores.max() + 1)

        scores = 0.5 * recency_scores + 0.5 * usage_scores

        # Remove pattern with lowest score
        remove_idx = np.argmin(scores)
        self.patterns.pop(remove_idx)
        self.outcomes.pop(remove_idx)
        self.timestamps.pop(remove_idx)
        self.retrieval_counts.pop(remove_idx)

    def query(self, query: np.ndarray, k: int = 5):
        """
        Query memory for similar patterns
        """
        if len(self.patterns) == 0:
            return [], [], []

        query = query / (np.linalg.norm(query) + 1e-8)
        patterns_arr = np.array(self.patterns)

        similarities = patterns_arr @ query

        k = min(k, len(self.patterns))
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        # Update retrieval counts
        for idx in top_k_idx:
            self.retrieval_counts[idx] += 1

        return (
            [self.patterns[i] for i in top_k_idx],
            [self.outcomes[i] for i in top_k_idx],
            [similarities[i] for i in top_k_idx]
        )

    def predict(self, query: np.ndarray, k: int = 5) -> tuple:
        """
        Predict outcome based on similar patterns
        """
        patterns, outcomes, similarities = self.query(query, k)

        if len(patterns) == 0:
            return 0.0, 0.0

        # Weighted average of outcomes
        similarities = np.array(similarities)
        weights = similarities / (similarities.sum() + 1e-8)
        prediction = np.sum(weights * np.array(outcomes))

        # Confidence based on average similarity
        confidence = np.mean(similarities)

        return prediction, confidence

    def to_tensors(self):
        """
        Convert to PyTorch tensors for neural network
        """
        if len(self.patterns) == 0:
            return None, None

        patterns = torch.tensor(np.array(self.patterns), dtype=torch.float32)
        outcomes = torch.tensor(np.array(self.outcomes), dtype=torch.float32)

        return patterns, outcomes
```

### Training Loop

```python
def train_associative_memory(
    model: DenseAssociativeMemory,
    train_patterns: torch.Tensor,
    train_labels: torch.Tensor,
    val_patterns: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64
):
    """
    Train the associative memory model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    # Store patterns in memory
    model.store_patterns(train_patterns, train_patterns)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle training data
        perm = torch.randperm(len(train_patterns))

        for i in range(0, len(train_patterns), batch_size):
            batch_idx = perm[i:i+batch_size]
            patterns = train_patterns[batch_idx]
            labels = train_labels[batch_idx].unsqueeze(-1)

            optimizer.zero_grad()

            predictions, confidence = model(patterns)

            # Loss: weighted MSE by confidence
            loss = (confidence * (predictions - labels).pow(2)).mean()

            # Add regularization for memory diversity
            patterns_norm = F.normalize(model.patterns, dim=-1)
            similarity_matrix = patterns_norm @ patterns_norm.T
            diversity_loss = (similarity_matrix - torch.eye(model.memory_size)).pow(2).mean()

            total_loss = loss + 0.1 * diversity_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_conf = model(val_patterns)
            val_loss = criterion(val_pred.squeeze(), val_labels)

            # Direction accuracy
            direction_acc = ((val_pred.squeeze() > 0) == (val_labels > 0)).float().mean()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={total_loss/n_batches:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Direction Acc={direction_acc:.2%}")

    model.load_state_dict(best_state)
    return model
```

### Backtesting Framework

```python
class AssociativeMemoryBacktest:
    """
    Backtest trading strategy based on associative memory
    """

    def __init__(self,
                 model: DenseAssociativeMemory,
                 pattern_lookback: int = 20,
                 min_confidence: float = 0.3,
                 position_size: float = 0.1):
        self.model = model
        self.pattern_lookback = pattern_lookback
        self.min_confidence = min_confidence
        self.position_size = position_size

    def run(self, prices: pd.DataFrame, warmup: int = 100):
        """
        Run backtest on price data

        Args:
            prices: DataFrame with OHLCV columns
            warmup: Number of periods for warmup
        """
        results = {
            'timestamp': [],
            'price': [],
            'signal': [],
            'confidence': [],
            'position': [],
            'pnl': [],
            'cumulative_pnl': []
        }

        position = 0.0
        cumulative_pnl = 0.0

        self.model.eval()

        for i in range(warmup, len(prices)):
            # Compute current pattern
            window = prices.iloc[i-self.pattern_lookback:i]
            pattern = compute_market_pattern(window)
            pattern_tensor = torch.tensor(pattern, dtype=torch.float32)

            # Get prediction
            with torch.no_grad():
                pred, conf = self.model(pattern_tensor.unsqueeze(0))

            pred = pred.item()
            conf = conf.item()

            # Determine position
            if conf >= self.min_confidence:
                target_position = np.sign(pred) * self.position_size * conf
            else:
                target_position = 0.0

            # Calculate PnL (assumes daily returns)
            if i > warmup:
                daily_return = (prices['close'].iloc[i] / prices['close'].iloc[i-1]) - 1
                pnl = position * daily_return
                cumulative_pnl += pnl
            else:
                pnl = 0.0

            # Update position
            position = target_position

            # Store results
            results['timestamp'].append(prices.index[i])
            results['price'].append(prices['close'].iloc[i])
            results['signal'].append(pred)
            results['confidence'].append(conf)
            results['position'].append(position)
            results['pnl'].append(pnl)
            results['cumulative_pnl'].append(cumulative_pnl)

        return pd.DataFrame(results)

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate performance metrics
        """
        returns = results['pnl']

        # Basic metrics
        total_return = results['cumulative_pnl'].iloc[-1]
        n_days = len(results)

        # Sharpe Ratio (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum Drawdown
        cumulative = results['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()

        # Win Rate
        winning = returns[returns > 0]
        win_rate = len(winning) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0

        # Average confidence when trading
        trading_conf = results[results['position'] != 0]['confidence']
        avg_confidence = trading_conf.mean() if len(trading_conf) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': (results['position'].diff() != 0).sum(),
            'avg_confidence': avg_confidence
        }
```

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of hourly data
├── Recommended: 3+ years for diverse patterns
├── Frequency: 1-hour to daily recommended
└── Source: Bybit, Binance, or other exchanges

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: turnover, trades count

Pattern Construction:
├── Lookback: 20-60 periods for pattern
├── Features: 10-50 dimensions
├── Normalization: Z-score or min-max
└── Update: Rolling window
```

## Key Metrics

- **Retrieval Accuracy**: Percentage of times correct pattern is in top-K
- **Pattern Coverage**: Percentage of market conditions covered by stored patterns
- **Confidence Calibration**: Correlation between confidence and accuracy
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Dependencies

```python
# Core
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Deep Learning
torch>=2.0.0
pytorch-lightning>=2.0.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0

# Market Data
ccxt>=4.0.0
websocket-client>=1.4.0

# Utilities
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Expected Outcomes

1. **Pattern Recognition System** with exponential storage capacity
2. **Retrieval-Based Predictions** using attention mechanisms
3. **Confidence-Calibrated Trading** with novelty detection
4. **Interpretable Decisions** through similar pattern analysis
5. **Backtest Results**: Expected Sharpe Ratio 1.0-2.0 with proper tuning

## References

1. **Dense Associative Memory for Pattern Recognition** (Krotov & Hopfield, 2016)
   - URL: https://arxiv.org/abs/1606.01164

2. **Hopfield Networks is All You Need** (Ramsauer et al., 2020)
   - URL: https://arxiv.org/abs/2008.02217

3. **Modern Hopfield Networks and Attention for Immune Repertoire Classification** (Widrich et al., 2020)
   - URL: https://arxiv.org/abs/2007.13505

4. **Associative Memory in Machine Learning** - Survey and Applications

5. **Neural Networks and Deep Learning** (Michael Nielsen) - Chapter on Hopfield Networks

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance associative memory trading on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Real-time data fetching from Bybit
- Dense Associative Memory implementation
- Pattern storage and retrieval
- Confidence-based trading signals
- Backtesting framework
- Modular and extensible design

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Neural Networks, Attention Mechanisms, Energy-Based Models, Pattern Recognition, Trading Systems
