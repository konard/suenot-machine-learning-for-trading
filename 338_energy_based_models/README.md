# Chapter 338: Energy-Based Models for Trading

## Overview

Energy-Based Models (EBMs) are a class of probabilistic models that associate a scalar energy value with each configuration of input variables. Unlike traditional discriminative models that directly compute probabilities, EBMs learn an energy function where low energy corresponds to high probability regions. This flexibility makes EBMs particularly powerful for financial applications where complex, multimodal distributions are common.

In trading, EBMs excel at:
- **Density estimation**: Modeling complex return distributions
- **Anomaly detection**: High energy = unusual market states
- **Generative modeling**: Sampling realistic market scenarios
- **Flexible classification**: Joint modeling of features and labels

## Key Concepts

### Energy Function

The core of EBM is the energy function E(x):
- Low energy → High probability (likely/normal)
- High energy → Low probability (unlikely/anomalous)

The probability is defined via Boltzmann distribution:
```
p(x) = exp(-E(x)) / Z
```
where Z is the partition function (normalizing constant).

### Why EBMs for Trading?

1. **No distributional assumptions**: Unlike Gaussian models, EBMs can capture fat tails, skewness, and multimodality
2. **Flexible scoring**: Energy provides a natural anomaly score
3. **Joint modeling**: Can model p(x,y) for both features and targets
4. **Calibration**: Better uncertainty quantification than standard classifiers

## Trading Strategy

**Core Strategy: Energy-Based Regime Detection and Risk Management**

### Signal Generation

1. **Anomaly Detection via Energy**:
   - Compute energy E(x) for current market state
   - High energy = unusual market conditions → reduce exposure
   - Low energy = typical conditions → normal trading

2. **Regime Classification**:
   - Use EBM as joint classifier p(regime|features)
   - Regimes: Bull, Bear, High-Volatility, Mean-Reverting

3. **Contrarian Signals**:
   - After energy spike resolves, enter contrarian positions
   - Energy normalization = market returning to typical state

### Risk Management

```
Position Size = Base Size × (1 - normalized_energy)
```
- Higher energy → smaller positions
- Provides automatic risk scaling

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_ebm_fundamentals.ipynb` | Energy functions and Boltzmann distribution |
| 2 | `02_contrastive_divergence.ipynb` | CD training algorithm |
| 3 | `03_score_matching.ipynb` | Alternative training via score matching |
| 4 | `04_noise_contrastive_estimation.ipynb` | NCE for EBM training |
| 5 | `05_ebm_classifier.ipynb` | EBM as classifier (JEM approach) |
| 6 | `06_ebm_density_estimation.ipynb` | Density estimation for returns |
| 7 | `07_rbm_trading.ipynb` | Restricted Boltzmann Machines |
| 8 | `08_deep_ebm.ipynb` | Deep Energy-Based Models |
| 9 | `09_ebm_anomaly_detection.ipynb` | Using energy for anomaly scoring |
| 10 | `10_ebm_regime_detection.ipynb` | Market regime classification |
| 11 | `11_trading_signals.ipynb` | Converting energy to signals |
| 12 | `12_backtesting.ipynb` | Full strategy backtest |

### EBM Architecture Types

```
Energy-Based Model Types:
├── Restricted Boltzmann Machines (RBM)
│   ├── Visible-Hidden architecture
│   ├── Contrastive Divergence training
│   └── Can stack into Deep Belief Networks
├── Deep Energy-Based Models
│   ├── Neural network energy function
│   ├── Score matching / NCE training
│   └── More expressive than RBMs
├── Joint Energy Models (JEM)
│   ├── Classifier as EBM
│   ├── p(x,y) joint modeling
│   └── Improved calibration
└── MCMC-Free EBMs
    ├── Denoising score matching
    ├── Sliced score matching
    └── More efficient training
```

### Energy Function Implementation

```python
import torch
import torch.nn as nn

class EnergyNet(nn.Module):
    """
    Neural network that computes energy E(x) for input x
    Low energy = high probability (typical)
    High energy = low probability (anomalous)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # Smooth activation
            ])
            prev_dim = hidden_dim

        # Final layer outputs scalar energy
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Compute energy for input x"""
        return self.network(x).squeeze(-1)

    def energy(self, x):
        """Alias for forward"""
        return self.forward(x)

    def log_prob(self, x, log_z=0.0):
        """
        Approximate log probability (up to normalizing constant)
        log p(x) ≈ -E(x) - log(Z)
        """
        return -self.energy(x) - log_z
```

### Contrastive Divergence Training

```python
class ContrastiveDivergenceLoss:
    """
    Train EBM using Contrastive Divergence

    The idea: Push down energy of real data,
    push up energy of model samples (negative samples)
    """
    def __init__(self, model, n_steps=10, step_size=0.01, noise_scale=0.01):
        self.model = model
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = noise_scale

    def sample_negative(self, x_init):
        """
        Generate negative samples using Langevin dynamics

        x_{t+1} = x_t - λ∇E(x_t) + ε, where ε ~ N(0, σ²)
        """
        x = x_init.clone().requires_grad_(True)

        for _ in range(self.n_steps):
            energy = self.model.energy(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]

            # Langevin update
            noise = torch.randn_like(x) * self.noise_scale
            x = x - self.step_size * grad + noise
            x = x.detach().requires_grad_(True)

        return x.detach()

    def __call__(self, x_real):
        """
        Compute CD loss

        L = E[E(x_real)] - E[E(x_negative)]

        We want to minimize energy of real data
        and maximize energy of negative samples
        """
        # Energy of real data
        energy_real = self.model.energy(x_real)

        # Generate negative samples
        x_init = torch.randn_like(x_real)
        x_negative = self.sample_negative(x_init)

        # Energy of negative samples
        energy_negative = self.model.energy(x_negative)

        # CD loss: push down real, push up negative
        loss = energy_real.mean() - energy_negative.mean()

        # Add regularization to prevent energy collapse
        reg = 0.01 * (energy_real**2 + energy_negative**2).mean()

        return loss + reg
```

### Joint Energy Model (JEM) for Classification

```python
class JointEnergyModel(nn.Module):
    """
    Joint Energy Model: Your Classifier is Secretly an Energy-Based Model

    Key insight: The logits of a classifier define an energy function
    E(x) = -LogSumExp(logits(x))

    This allows joint modeling of p(x,y) and improved calibration
    """
    def __init__(self, input_dim, n_classes, hidden_dims=[128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def logits(self, x):
        """Get class logits"""
        return self.network(x)

    def energy(self, x):
        """
        Energy = -LogSumExp(logits)
        Low energy = classifier is confident about some class
        High energy = classifier is uncertain (OOD)
        """
        logits = self.logits(x)
        return -torch.logsumexp(logits, dim=-1)

    def classify(self, x):
        """Standard classification"""
        logits = self.logits(x)
        return logits.argmax(dim=-1)

    def class_probabilities(self, x):
        """Get class probabilities p(y|x)"""
        logits = self.logits(x)
        return torch.softmax(logits, dim=-1)

    def anomaly_score(self, x):
        """
        Use energy as anomaly score
        High energy = anomalous (OOD)
        """
        return self.energy(x)
```

### Noise Contrastive Estimation (NCE) Training

```python
class NCELoss:
    """
    Noise Contrastive Estimation for EBM training

    Instead of computing intractable partition function,
    train to distinguish real data from noise
    """
    def __init__(self, model, noise_scale=1.0):
        self.model = model
        self.noise_scale = noise_scale

    def __call__(self, x_real):
        batch_size = x_real.shape[0]

        # Generate noise samples (same size as real)
        x_noise = torch.randn_like(x_real) * self.noise_scale

        # Energy of real and noise
        energy_real = self.model.energy(x_real)
        energy_noise = self.model.energy(x_noise)

        # Log probability of noise distribution
        log_p_noise_real = -0.5 * (x_real**2 / self.noise_scale**2).sum(dim=-1)
        log_p_noise_noise = -0.5 * (x_noise**2 / self.noise_scale**2).sum(dim=-1)

        # NCE objective: classify real vs noise
        # h(x) = log p_model(x) - log p_noise(x)
        h_real = -energy_real - log_p_noise_real
        h_noise = -energy_noise - log_p_noise_noise

        # Binary cross-entropy
        loss_real = torch.log(1 + torch.exp(-h_real)).mean()
        loss_noise = torch.log(1 + torch.exp(h_noise)).mean()

        return loss_real + loss_noise
```

### Score Matching Training

```python
class ScoreMatchingLoss:
    """
    Score Matching: train without negative sampling

    Match the score (gradient of log-density) of the model
    to the score of the data distribution

    ∇_x log p_model(x) ≈ ∇_x log p_data(x)
    """
    def __init__(self, model, noise_scale=0.1):
        self.model = model
        self.noise_scale = noise_scale

    def __call__(self, x):
        """
        Denoising Score Matching:
        Add noise to data and train model to predict the score
        """
        # Add noise
        noise = torch.randn_like(x) * self.noise_scale
        x_noisy = x + noise
        x_noisy.requires_grad_(True)

        # Compute energy and its gradient (score)
        energy = self.model.energy(x_noisy)
        score = torch.autograd.grad(
            energy.sum(), x_noisy, create_graph=True
        )[0]

        # Target score: gradient of log p_noise(x_noisy | x) = -(x_noisy - x) / σ²
        target_score = -noise / (self.noise_scale ** 2)

        # Score matching loss
        loss = ((score - target_score) ** 2).sum(dim=-1).mean()

        return loss
```

### EBM for Market Feature Extraction

```python
def compute_ebm_features(df):
    """
    Feature engineering for EBM-based trading
    """
    features = {}

    # Returns and log-returns
    features['return'] = df['close'].pct_change()
    features['log_return'] = np.log(df['close']).diff()

    # Multi-scale volatility
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = features['return'].rolling(window).std()

    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(20).mean()) /
        df['volume'].rolling(20).std()
    )

    # Price position
    features['price_position'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    )

    # Range features
    features['range'] = (df['high'] - df['low']) / df['close']
    features['range_ratio'] = features['range'] / features['range'].rolling(20).mean()

    # Momentum
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = df['close'].pct_change(window)

    # Higher moments
    features['skewness'] = features['return'].rolling(20).skew()
    features['kurtosis'] = features['return'].rolling(20).kurt()

    return pd.DataFrame(features)
```

### Trading Signals from Energy

```python
class EBMTradingStrategy:
    """
    Trading strategy based on Energy-Based Model
    """
    def __init__(self, model, energy_threshold=2.0, lookback=20):
        self.model = model
        self.energy_threshold = energy_threshold
        self.lookback = lookback
        self.energy_history = []

    def compute_signal(self, features):
        """
        Generate trading signals from energy
        """
        with torch.no_grad():
            energy = self.model.energy(features).item()

        self.energy_history.append(energy)
        if len(self.energy_history) > self.lookback:
            self.energy_history.pop(0)

        # Normalize energy relative to recent history
        if len(self.energy_history) >= 5:
            mean_energy = np.mean(self.energy_history)
            std_energy = np.std(self.energy_history) + 1e-8
            normalized_energy = (energy - mean_energy) / std_energy
        else:
            normalized_energy = 0.0

        signal = {
            'energy': energy,
            'normalized_energy': normalized_energy,
            'is_anomaly': normalized_energy > self.energy_threshold,
            'position_scale': max(0.0, 1.0 - normalized_energy / self.energy_threshold),
        }

        # Regime detection
        if normalized_energy < -1.0:
            signal['regime'] = 'calm'
        elif normalized_energy < 1.0:
            signal['regime'] = 'normal'
        elif normalized_energy < 2.0:
            signal['regime'] = 'elevated'
        else:
            signal['regime'] = 'crisis'

        return signal

    def should_reduce_position(self, signal):
        """Check if position should be reduced"""
        return signal['normalized_energy'] > 1.5

    def should_exit(self, signal):
        """Check if should exit all positions"""
        return signal['is_anomaly']
```

### Restricted Boltzmann Machine (RBM)

```python
class RBM(nn.Module):
    """
    Restricted Boltzmann Machine

    Two-layer network with visible and hidden units
    Energy: E(v,h) = -v·W·h - a·v - b·h
    """
    def __init__(self, n_visible, n_hidden):
        super().__init__()

        # Weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))   # hidden bias

    def energy(self, v, h):
        """Compute energy E(v, h)"""
        return -torch.mm(v, self.W).sum() - torch.mv(v, self.a).sum() - torch.mv(h, self.b).sum()

    def free_energy(self, v):
        """
        Free energy: F(v) = -log Σ_h exp(-E(v,h))
        Used for training since we marginalize over hidden units
        """
        wx_b = torch.mm(v, self.W) + self.b
        hidden_term = torch.log(1 + torch.exp(wx_b)).sum(dim=1)
        visible_term = torch.mv(v, self.a)
        return -visible_term - hidden_term

    def sample_hidden(self, v):
        """Sample hidden units given visible"""
        prob_h = torch.sigmoid(torch.mm(v, self.W) + self.b)
        return prob_h, torch.bernoulli(prob_h)

    def sample_visible(self, h):
        """Sample visible units given hidden"""
        prob_v = torch.sigmoid(torch.mm(h, self.W.t()) + self.a)
        return prob_v, torch.bernoulli(prob_v)

    def contrastive_divergence(self, v, k=1):
        """
        k-step Contrastive Divergence for training
        """
        v_pos = v

        # Positive phase
        prob_h_pos, h_pos = self.sample_hidden(v_pos)

        # Negative phase (k steps of Gibbs sampling)
        h_neg = h_pos
        for _ in range(k):
            prob_v_neg, v_neg = self.sample_visible(h_neg)
            prob_h_neg, h_neg = self.sample_hidden(v_neg)

        # CD loss: difference in free energies
        loss = self.free_energy(v_pos).mean() - self.free_energy(v_neg).mean()

        return loss
```

## Architecture Diagram

```
                     Market Data Input
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌─────────────┐ ┌─────────────┐
            │ OHLCV Data  │ │ Order Book  │
            │ (Bybit API) │ │ (optional)  │
            └──────┬──────┘ └──────┬──────┘
                   │               │
                   └───────┬───────┘
                           ▼
                 ┌─────────────────┐
                 │ Feature Engine  │
                 ├─────────────────┤
                 │ - Returns       │
                 │ - Volatility    │
                 │ - Volume        │
                 │ - Technical     │
                 └────────┬────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  Energy-Based Model     │
            ├─────────────────────────┤
            │                         │
            │  ┌─────────────────┐   │
            │  │  Energy Net     │   │
            │  │  E(x) → ℝ       │   │
            │  └────────┬────────┘   │
            │           │            │
            │  p(x) = exp(-E(x))/Z   │
            │                         │
            └────────────┬────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Energy      │ │ Regime      │ │ Anomaly     │
    │ Score       │ │ Detection   │ │ Detection   │
    │             │ │             │ │             │
    │ Low = safe  │ │ - Calm      │ │ High E →    │
    │ High = risk │ │ - Normal    │ │ anomaly     │
    │             │ │ - Volatile  │ │             │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                 ┌─────────────────┐
                 │ Signal Generator│
                 ├─────────────────┤
                 │ - Position size │
                 │ - Entry/Exit    │
                 │ - Risk scaling  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Order Execution │
                 │ (Bybit API)     │
                 └─────────────────┘
```

## Training Approaches Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Contrastive Divergence | Simple, widely used | Biased gradients | RBMs, quick prototyping |
| Score Matching | No MCMC needed | Memory intensive | High-dimensional data |
| NCE | Stable training | Requires noise design | Medium-scale models |
| Langevin MCMC | Asymptotically correct | Slow, mode collapse | Research, accuracy |

## Data Requirements

```
Historical OHLCV Data:
├── Minimum: 6 months of hourly data
├── Recommended: 2+ years for robust training
├── Frequency: 1-minute to daily
└── Source: Bybit API (cryptocurrency)

Required Fields:
├── timestamp
├── open, high, low, close
├── volume
└── Optional: trades count, funding rate

Training Data Split:
├── Train: 70% (oldest data)
├── Validation: 15%
└── Test: 15% (most recent)
```

## Key Metrics

### Model Metrics
- **Log-likelihood**: Model's probability of test data
- **Energy distribution**: Should be lower for real data
- **Sample quality**: For generative evaluation

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Anomaly Detection Metrics
- **AUROC**: Area under ROC curve
- **Precision@K**: Precision for top-K energy scores
- **Detection latency**: Time to detect regime change

## Dependencies

```python
# Core
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0

# Deep Learning
torch>=2.0.0
pytorch-lightning>=2.0.0

# Machine Learning
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0

# Market Data
ccxt>=4.0.0
websocket-client>=1.4.0

# Progress and Logging
tqdm>=4.64.0
tensorboard>=2.12.0
```

## Expected Outcomes

1. **Energy-based density estimation** for return distributions
2. **Regime detection** using energy thresholds
3. **Anomaly scoring** for unusual market states
4. **Risk-aware position sizing** based on energy levels
5. **Backtest results**: 20-40% improvement in risk-adjusted returns

## Scientific References

1. **Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One** (Grathwohl et al., 2019)
   - URL: https://arxiv.org/abs/1912.03263
   - Key insight: Classifiers can be interpreted as EBMs

2. **A Tutorial on Energy-Based Learning** (LeCun et al., 2006)
   - URL: http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
   - Comprehensive introduction to EBMs

3. **How to Train Your Energy-Based Models** (Song & Kingma, 2021)
   - URL: https://arxiv.org/abs/2101.03288
   - Modern training techniques for EBMs

4. **Noise Contrastive Estimation** (Gutmann & Hyvärinen, 2010)
   - Foundational paper on NCE training

5. **Score Matching with Langevin Dynamics** (Song & Ermon, 2019)
   - Connection between score matching and MCMC

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance EBM-based trading with cryptocurrency data from Bybit. See `rust_ebm_crypto/` directory.

### Features:
- Real-time data fetching from Bybit
- Energy-based anomaly detection
- Multiple training approaches (Score Matching, NCE-inspired)
- Modular and extensible design
- High-performance inference

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Requires understanding of: Probability Theory, Statistical Mechanics, Deep Learning, MCMC Methods, Financial Markets
