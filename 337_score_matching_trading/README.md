# Chapter 337: Score Matching Trading — Learning Data Distributions for Market Prediction

## Overview

Score Matching is a powerful technique for training Energy-Based Models (EBMs) and Diffusion Models without requiring expensive normalization constant computation. In trading, score matching enables learning the underlying distribution of market states, detecting anomalies, and generating predictions based on the gradient of the data distribution (the "score function").

This chapter explores how to apply score matching to financial markets, using the learned score function to identify market regime changes, predict price movements, and generate trading signals.

## Core Concepts

### What is Score Matching?

Score matching is a method for estimating the parameters of a probability distribution by matching the **score function** — the gradient of the log probability density:

```
Score function: s(x) = ∇_x log p(x)

Key insight: The score function captures how probability changes
as we move through the data space, without needing to know
the normalization constant Z = ∫ p(x)dx
```

### Why Score Matching for Trading?

1. **Distribution Learning**: Learn the true distribution of market states, not just point predictions
2. **Anomaly Detection**: Low probability regions indicate unusual market conditions
3. **Regime Identification**: Score gradients reveal structure in market data
4. **Generative Modeling**: Sample from learned distributions for scenario analysis
5. **Noise Robustness**: Denoising score matching naturally handles market noise

### From Energy-Based Models to Score Matching

```
Energy-Based Model:
├── Energy function: E(x; θ)
├── Probability: p(x) = exp(-E(x)) / Z
├── Problem: Z is intractable to compute
└── Solution: Match scores instead!

Score Matching:
├── Model score: s_θ(x) = -∇_x E(x; θ)
├── Data score: ∇_x log p_data(x)
├── Objective: minimize ||s_θ(x) - ∇_x log p_data(x)||²
└── No Z needed!
```

## Trading Strategy

**Strategy Overview:** Use score matching to learn the distribution of market patterns. Trading signals are generated based on:
1. The score function indicating likely market movements
2. Probability density indicating regime confidence
3. Denoising predictions for price direction

### Signal Generation

```
1. Feature Extraction:
   - Compute market features: returns, volatility, volume
   - Normalize to match training distribution

2. Score Computation:
   - Query the learned score function s_θ(x)
   - Score indicates direction of higher probability

3. Signal Interpretation:
   - Positive return score → expect positive returns → Long
   - Negative return score → expect negative returns → Short
   - Score magnitude indicates signal strength

4. Probability Filtering:
   - Estimate local probability density
   - Only trade in high-density (familiar) regions
```

### Entry Signals

- **Long Signal**: Score function points toward positive returns with high magnitude
- **Short Signal**: Score function points toward negative returns with high magnitude
- **Confidence Filter**: Only trade when estimated density exceeds threshold

### Risk Management

- **Novelty Detection**: Low density indicates unfamiliar market conditions → reduce exposure
- **Score Consistency**: Check if score direction is stable over short window
- **Volatility Scaling**: Scale position by inverse of score variance

## Technical Specification

### Mathematical Foundation

#### Score Function Definition

For a probability density p(x), the score function is:

```
s(x) = ∇_x log p(x) = ∇_x p(x) / p(x)

Properties:
├── Points toward regions of higher probability
├── Zero at local maxima of p(x)
├── Independent of normalization constant Z
└── Uniquely determines p(x) up to a constant
```

#### Original Score Matching Objective (Hyvärinen, 2005)

```
J(θ) = E_p_data [½||s_θ(x)||² + tr(∇_x s_θ(x))]

Where:
├── s_θ(x) = model score function
├── tr(∇_x s_θ(x)) = trace of Jacobian
└── No need to know ∇_x log p_data(x)!
```

#### Denoising Score Matching (Vincent, 2011)

Instead of computing the Jacobian trace, perturb data with noise:

```
1. Perturb data: x̃ = x + σε, where ε ~ N(0, I)
2. Train to denoise: minimize E[||s_θ(x̃) - ∇_x̃ log p(x̃|x)||²]
3. For Gaussian noise: ∇_x̃ log p(x̃|x) = -(x̃ - x)/σ² = -ε/σ

Simplified objective:
J_DSM(θ) = E_x,ε [||s_θ(x + σε) + ε/σ||²]
```

#### Sliced Score Matching (Song et al., 2020)

Efficient score matching using random projections:

```
J_SSM(θ) = E_p_data E_v [v^T ∇_x s_θ(x) v + ½(v^T s_θ(x))²]

Where v is a random direction vector
└── Reduces O(d²) Jacobian to O(d) computation
```

### Architecture Diagram

```
                    Market Data Stream
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Feature Engineering      │
            │  ├── Returns (multi-scale)  │
            │  ├── Volatility measures    │
            │  ├── Volume patterns        │
            │  └── Technical indicators   │
            └──────────────┬──────────────┘
                           │
                           ▼ Market State x
            ┌─────────────────────────────┐
            │    Score Network s_θ(x)     │
            │                             │
            │  ┌───────────────────────┐  │
            │  │   Input Layer (d)     │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │   Hidden Layers       │  │
            │  │   (MLP / ResNet)      │  │
            │  │   + Skip Connections  │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │   Output Layer (d)    │  │
            │  │   Score Vector s_θ(x) │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Return     │ │  Density    │ │   Score     │
     │  Score      │ │  Estimate   │ │  Magnitude  │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Trading Decision        │
            │  ├── Signal Direction       │
            │  ├── Position Size          │
            │  ├── Confidence Level       │
            │  └── Risk Parameters        │
            └─────────────────────────────┘
```

### Feature Engineering for Score Matching

```python
import numpy as np
import pandas as pd

def compute_market_state(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Create market state vector for score matching
    """
    features = {}

    # Multi-scale returns
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}'] = returns.rolling(period).sum().iloc[-1]

    # Volatility features
    features['volatility'] = returns.rolling(lookback).std().iloc[-1]
    features['volatility_ratio'] = (
        returns.rolling(5).std().iloc[-1] /
        returns.rolling(20).std().iloc[-1]
    )

    # Momentum
    features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1

    # Volume features
    volume_ma = df['volume'].rolling(lookback).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / volume_ma.iloc[-1]

    # Price position
    high_20 = df['high'].rolling(lookback).max().iloc[-1]
    low_20 = df['low'].rolling(lookback).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    # RSI-like feature
    gains = returns.clip(lower=0).rolling(14).mean().iloc[-1]
    losses = (-returns.clip(upper=0)).rolling(14).mean().iloc[-1]
    features['rsi'] = gains / (gains + losses + 1e-8)

    return np.array(list(features.values()))
```

### Score Network Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNetwork(nn.Module):
    """
    Neural network to estimate the score function s(x) = ∇_x log p(x)

    Architecture: ResNet-style MLP with skip connections
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection (same dimension as input for score)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score function s_θ(x)

        Args:
            x: (batch, input_dim) input market states

        Returns:
            score: (batch, input_dim) score vectors
        """
        h = self.input_proj(x)

        for block in self.res_blocks:
            h = block(h)

        score = self.output_proj(h)

        return score


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class NoiseLevelEmbedding(nn.Module):
    """Embedding for noise level σ (for multi-scale denoising)"""

    def __init__(self, hidden_dim: int, max_noise_level: int = 10):
        super().__init__()

        self.embedding = nn.Embedding(max_noise_level, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(noise_level)
        return self.proj(emb)
```

### Multi-Scale Denoising Score Matching

```python
class MultiScaleScoreNetwork(nn.Module):
    """
    Score network with multiple noise scales

    Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, num_noise_levels: int = 10,
                 sigma_min: float = 0.01, sigma_max: float = 1.0):
        super().__init__()

        self.input_dim = input_dim
        self.num_noise_levels = num_noise_levels

        # Geometric noise schedule
        self.register_buffer(
            'sigmas',
            torch.exp(torch.linspace(
                np.log(sigma_max), np.log(sigma_min), num_noise_levels
            ))
        )

        # Score network
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.noise_embed = NoiseLevelEmbedding(hidden_dim, num_noise_levels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """
        Compute score at given noise level

        Args:
            x: (batch, input_dim) noisy input
            noise_level: (batch,) noise level indices

        Returns:
            score: (batch, input_dim) estimated score
        """
        h = self.input_proj(x)
        h = h + self.noise_embed(noise_level)

        for block in self.res_blocks:
            h = block(h)

        score = self.output_proj(h)

        return score

    def get_sigma(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Get noise standard deviation for given level"""
        return self.sigmas[noise_level]


class DenoisingScoreMatchingLoss(nn.Module):
    """
    Denoising Score Matching loss for training

    J_DSM(θ) = E_{x,ε,σ} [||s_θ(x + σε) + ε/σ||²]
    """

    def __init__(self, score_network: MultiScaleScoreNetwork):
        super().__init__()
        self.score_network = score_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DSM loss

        Args:
            x: (batch, dim) clean data samples

        Returns:
            loss: scalar loss value
        """
        batch_size = x.shape[0]

        # Sample random noise levels
        noise_levels = torch.randint(
            0, self.score_network.num_noise_levels, (batch_size,),
            device=x.device
        )
        sigmas = self.score_network.get_sigma(noise_levels).unsqueeze(-1)

        # Add noise
        noise = torch.randn_like(x)
        x_noisy = x + sigmas * noise

        # Compute score
        score = self.score_network(x_noisy, noise_levels)

        # Target score: -noise/sigma
        target = -noise / sigmas

        # Weighted loss (weight by sigma^2 for balanced training)
        loss = ((score - target) ** 2).sum(dim=-1) * (sigmas.squeeze() ** 2)

        return loss.mean()
```

### Langevin Dynamics for Sampling and Prediction

```python
class LangevinDynamics:
    """
    Annealed Langevin Dynamics for sampling from learned distribution

    Can be used to:
    1. Generate synthetic market scenarios
    2. Denoise current market state to find "true" pattern
    3. Predict future states by following score
    """

    def __init__(self, score_network: MultiScaleScoreNetwork,
                 n_steps_per_level: int = 100,
                 step_size: float = 0.01):
        self.score_network = score_network
        self.n_steps = n_steps_per_level
        self.step_size = step_size

    @torch.no_grad()
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples using annealed Langevin dynamics
        """
        # Start from random noise
        x = torch.randn(batch_size, self.score_network.input_dim, device=device)

        # Anneal through noise levels
        for i in range(self.score_network.num_noise_levels):
            noise_level = torch.full((batch_size,), i, device=device, dtype=torch.long)
            sigma = self.score_network.get_sigma(noise_level)[0]

            # Langevin step size for this noise level
            alpha = self.step_size * (sigma ** 2)

            for _ in range(self.n_steps):
                score = self.score_network(x, noise_level)
                noise = torch.randn_like(x)
                x = x + alpha * score + np.sqrt(2 * alpha) * noise

        return x

    @torch.no_grad()
    def denoise(self, x: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Denoise input by following score at lowest noise level
        """
        device = x.device
        batch_size = x.shape[0]

        # Use lowest noise level
        noise_level = torch.full(
            (batch_size,),
            self.score_network.num_noise_levels - 1,
            device=device,
            dtype=torch.long
        )
        sigma = self.score_network.get_sigma(noise_level)[0]
        alpha = self.step_size * (sigma ** 2)

        for _ in range(n_steps):
            score = self.score_network(x, noise_level)
            x = x + alpha * score

        return x

    @torch.no_grad()
    def get_trading_signal(self, x: torch.Tensor,
                           return_idx: int = 0) -> tuple:
        """
        Get trading signal from score function

        Args:
            x: (batch, dim) current market state
            return_idx: index of return feature in state vector

        Returns:
            signal: predicted direction of returns
            confidence: magnitude of signal (higher = more confident)
        """
        device = x.device
        batch_size = x.shape[0]

        # Use lowest noise level for cleanest score
        noise_level = torch.full(
            (batch_size,),
            self.score_network.num_noise_levels - 1,
            device=device,
            dtype=torch.long
        )

        score = self.score_network(x, noise_level)

        # Return score indicates direction of higher probability for returns
        return_score = score[:, return_idx]

        signal = torch.sign(return_score)
        confidence = torch.abs(return_score)

        # Normalize confidence
        confidence = confidence / (confidence.max() + 1e-8)

        return signal, confidence
```

### Probability Density Estimation

```python
class ScoreBasedDensityEstimator:
    """
    Estimate probability density from learned score function

    Uses the identity: ∇·s(x) = Δ log p(x)
    """

    def __init__(self, score_network: MultiScaleScoreNetwork):
        self.score_network = score_network

    def log_density_up_to_const(self, x: torch.Tensor,
                                  n_integration_steps: int = 100) -> torch.Tensor:
        """
        Estimate log density by integrating score from reference point

        Uses path integral: log p(x) - log p(x_0) = ∫ s(x(t)) · dx(t)
        """
        batch_size = x.shape[0]
        device = x.device

        # Reference point (e.g., data mean)
        x_0 = torch.zeros(1, self.score_network.input_dim, device=device)

        # Straight line path from x_0 to x
        log_density = torch.zeros(batch_size, device=device)

        noise_level = torch.full(
            (batch_size,),
            self.score_network.num_noise_levels - 1,
            device=device,
            dtype=torch.long
        )

        for i in range(n_integration_steps):
            t = i / n_integration_steps
            x_t = x_0 + t * (x - x_0)

            score = self.score_network(x_t, noise_level)

            # Dot product with path direction
            dx = (x - x_0) / n_integration_steps
            log_density = log_density + (score * dx).sum(dim=-1)

        return log_density

    def is_in_distribution(self, x: torch.Tensor,
                           threshold: float = -5.0) -> torch.Tensor:
        """
        Check if samples are within learned distribution

        Returns boolean mask indicating in-distribution samples
        """
        log_density = self.log_density_up_to_const(x)
        return log_density > threshold
```

### Score Matching Trading System

```python
class ScoreMatchingTrader:
    """
    Trading system based on score matching
    """

    def __init__(self,
                 score_network: MultiScaleScoreNetwork,
                 feature_dim: int,
                 return_feature_idx: int = 0,
                 confidence_threshold: float = 0.3,
                 density_threshold: float = -5.0):
        self.score_network = score_network
        self.feature_dim = feature_dim
        self.return_idx = return_feature_idx
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold

        self.langevin = LangevinDynamics(score_network)
        self.density_estimator = ScoreBasedDensityEstimator(score_network)

    def generate_signal(self, market_state: torch.Tensor) -> dict:
        """
        Generate trading signal from current market state

        Args:
            market_state: (dim,) current market features

        Returns:
            dict with signal, confidence, in_distribution flag
        """
        self.score_network.eval()

        x = market_state.unsqueeze(0)

        with torch.no_grad():
            # Get signal from score
            signal, confidence = self.langevin.get_trading_signal(x, self.return_idx)

            # Check if in distribution
            in_dist = self.density_estimator.is_in_distribution(
                x, self.density_threshold
            )

            # Denoise to get "true" market state
            denoised = self.langevin.denoise(x, n_steps=20)

        signal = signal.item()
        confidence = confidence.item()
        in_dist = in_dist.item()

        # Only trade if in distribution and confident
        if not in_dist:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'in_distribution': False,
                'raw_signal': signal,
                'denoised_state': denoised.squeeze().numpy()
            }

        if confidence < self.confidence_threshold:
            return {
                'signal': 0.0,
                'confidence': confidence,
                'in_distribution': True,
                'raw_signal': signal,
                'denoised_state': denoised.squeeze().numpy()
            }

        return {
            'signal': signal * confidence,
            'confidence': confidence,
            'in_distribution': True,
            'raw_signal': signal,
            'denoised_state': denoised.squeeze().numpy()
        }
```

### Training Loop

```python
def train_score_network(
    model: MultiScaleScoreNetwork,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
):
    """
    Train score network using denoising score matching
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = DenoisingScoreMatchingLoss(model)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle data
        perm = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]

            optimizer.zero_grad()
            loss = loss_fn(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={total_loss/n_batches:.4f}, "
                  f"Val Loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
```

### Backtesting Framework

```python
class ScoreMatchingBacktest:
    """
    Backtest score matching trading strategy
    """

    def __init__(self,
                 trader: ScoreMatchingTrader,
                 lookback: int = 20):
        self.trader = trader
        self.lookback = lookback

    def run(self, prices: pd.DataFrame, warmup: int = 100) -> pd.DataFrame:
        """
        Run backtest on price data
        """
        results = {
            'timestamp': [],
            'price': [],
            'signal': [],
            'confidence': [],
            'in_distribution': [],
            'position': [],
            'pnl': [],
            'cumulative_pnl': []
        }

        position = 0.0
        cumulative_pnl = 0.0

        for i in range(warmup, len(prices)):
            window = prices.iloc[i-self.lookback:i]
            state = compute_market_state(window)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Get signal
            signal_info = self.trader.generate_signal(state_tensor)

            # Calculate PnL
            if i > warmup:
                daily_return = prices['close'].iloc[i] / prices['close'].iloc[i-1] - 1
                pnl = position * daily_return
                cumulative_pnl += pnl
            else:
                pnl = 0.0

            # Update position
            position = signal_info['signal']

            results['timestamp'].append(prices.index[i])
            results['price'].append(prices['close'].iloc[i])
            results['signal'].append(signal_info['signal'])
            results['confidence'].append(signal_info['confidence'])
            results['in_distribution'].append(signal_info['in_distribution'])
            results['position'].append(position)
            results['pnl'].append(pnl)
            results['cumulative_pnl'].append(cumulative_pnl)

        return pd.DataFrame(results)

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate performance metrics
        """
        returns = results['pnl']

        total_return = results['cumulative_pnl'].iloc[-1]

        # Sharpe Ratio
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
        trading_returns = returns[returns != 0]
        if len(trading_returns) > 0:
            win_rate = (trading_returns > 0).mean()
        else:
            win_rate = 0.0

        # In-distribution ratio
        in_dist_ratio = results['in_distribution'].mean()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'in_distribution_ratio': in_dist_ratio,
            'avg_confidence': results['confidence'].mean()
        }
```

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 1 year of hourly data
├── Recommended: 2+ years for distribution learning
├── Frequency: 1-hour to daily
└── Source: Bybit, Binance, or other exchanges

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: funding rate, open interest

Preprocessing:
├── Normalization: Z-score per feature
├── Outlier handling: Clip to ±5 std
├── Missing data: Forward fill then drop
└── Train/Val/Test split: 70/15/15
```

## Key Metrics

- **DSM Loss**: Denoising score matching training loss
- **Score Accuracy**: Correlation between predicted and true score
- **Density Calibration**: Quality of probability estimates
- **In-Distribution Ratio**: Fraction of trading days in learned distribution
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

## Dependencies

```python
# Core
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Deep Learning
torch>=2.0.0

# Market Data
ccxt>=4.0.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Utilities
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Expected Outcomes

1. **Distribution Learning**: Model captures the structure of market states
2. **Anomaly Detection**: Low-density regions flag unusual conditions
3. **Directional Predictions**: Score function indicates likely price movement
4. **Risk-Aware Trading**: Position sizing based on distribution confidence
5. **Backtest Results**: Expected Sharpe Ratio 0.8-1.5 with proper tuning

## References

1. **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019)
   - URL: https://arxiv.org/abs/1907.05600

2. **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011)
   - URL: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf

3. **Estimation of Non-Normalized Statistical Models by Score Matching** (Hyvärinen, 2005)
   - URL: http://www.jmlr.org/papers/v6/hyvarinen05a.html

4. **Sliced Score Matching: A Scalable Approach to Density and Score Estimation** (Song et al., 2020)
   - URL: https://arxiv.org/abs/1905.07088

5. **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021)
   - URL: https://arxiv.org/abs/2011.13456

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance score matching trading on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Real-time data fetching from Bybit API
- Score network implementation with neural network layers
- Denoising score matching training
- Langevin dynamics for sampling and prediction
- Backtesting framework with comprehensive metrics
- Modular and extensible design

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Probability Theory, Gradient-Based Methods, Neural Networks, Energy-Based Models, Generative Modeling, Trading Systems
