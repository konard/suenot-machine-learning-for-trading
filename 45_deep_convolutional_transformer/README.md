# Chapter 45: Deep Convolutional Transformer (DCT) for Stock Movement Prediction

This chapter explores the **Deep Convolutional Transformer (DCT)**, a hybrid architecture that combines Convolutional Neural Networks (CNN) with Transformer-based multi-head attention to extract both local patterns and global dependencies from financial time series data.

<p align="center">
<img src="https://i.imgur.com/zKqMvBN.png" width="70%">
</p>

## Contents

1. [Introduction to DCT](#introduction-to-dct)
    * [Why Combine CNN and Transformer?](#why-combine-cnn-and-transformer)
    * [Key Innovations](#key-innovations)
    * [Comparison with Other Models](#comparison-with-other-models)
2. [DCT Architecture](#dct-architecture)
    * [Inception Convolutional Embedding](#inception-convolutional-embedding)
    * [Multi-Head Self-Attention](#multi-head-self-attention)
    * [Separable Fully Connected Layers](#separable-fully-connected-layers)
    * [Classification Head](#classification-head)
3. [Data Preprocessing](#data-preprocessing)
    * [Feature Engineering](#feature-engineering)
    * [Normalization Techniques](#normalization-techniques)
    * [Look-Back Window](#look-back-window)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: DCT Model Architecture](#02-dct-model-architecture)
    * [03: Training Pipeline](#03-training-pipeline)
    * [04: Stock Movement Prediction](#04-stock-movement-prediction)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to DCT

The Deep Convolutional Transformer (DCT) addresses a fundamental challenge in stock price prediction: capturing both **local temporal patterns** (short-term price movements) and **global dependencies** (long-range market trends) simultaneously.

Traditional approaches often focus on one aspect:
- **CNNs** excel at extracting local features but struggle with long-range dependencies
- **Transformers** capture global context but may miss fine-grained local patterns

DCT combines the strengths of both architectures.

### Why Combine CNN and Transformer?

```
Financial Time Series Challenge:
├── Local Patterns (CNN strength)
│   ├── Candlestick patterns (2-5 bars)
│   ├── Short-term momentum
│   └── Volume spikes
│
└── Global Dependencies (Transformer strength)
    ├── Trend direction
    ├── Market regime changes
    └── Seasonal patterns
```

**DCT Solution**: Use inception-style convolutional layers to extract multi-scale local features, then apply Transformer attention to capture global dependencies.

### Key Innovations

1. **Inception Convolutional Token Embedding**
   - Multiple parallel convolutional kernels with different sizes
   - Captures patterns at various temporal scales (1-day, 3-day, 5-day, etc.)
   - Similar to GoogLeNet's inception module but adapted for time series

2. **Separable Fully Connected Layers**
   - Reduces parameter count and computational complexity
   - Depthwise separable convolutions applied to fully connected operations
   - Improves generalization on limited financial data

3. **Multi-Head Self-Attention**
   - Standard Transformer attention for capturing global dependencies
   - Each head can focus on different aspects (trend, volatility, momentum)
   - Interpretable attention weights show which time steps matter

4. **Movement Classification**
   - Predicts price direction: Up, Down, or Stable
   - Binary classification can also be used (Up/Down only)
   - Threshold-based labeling for movement definition

### Comparison with Other Models

| Feature | LSTM | CNN | Transformer | TFT | DCT |
|---------|------|-----|-------------|-----|-----|
| Local patterns | ✓ | ✓✓ | ✗ | ✓ | ✓✓ |
| Global dependencies | ✓ | ✗ | ✓✓ | ✓✓ | ✓✓ |
| Multi-scale features | ✗ | ✗ | ✗ | ✓ | ✓✓ |
| Parameter efficiency | ✗ | ✓ | ✗ | ✗ | ✓ |
| Interpretability | ✗ | ✗ | ✓ | ✓✓ | ✓ |
| Training stability | ✗ | ✓ | ✓ | ✓ | ✓ |

## DCT Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DEEP CONVOLUTIONAL TRANSFORMER                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    INPUT: [batch, seq_len, features]                 │ │
│  │         (Open, High, Low, Close, Volume, Technical Indicators)       │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              INCEPTION CONVOLUTIONAL EMBEDDING                       │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │ │
│  │  │ Conv1D  │  │ Conv1D  │  │ Conv1D  │  │MaxPool1D│               │ │
│  │  │  k=1    │  │  k=3    │  │  k=5    │  │  k=3    │               │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │ │
│  │       │            │            │            │                       │ │
│  │       └────────────┴──────┬─────┴────────────┘                       │ │
│  │                           │ Concatenate                              │ │
│  │                           ▼                                          │ │
│  │                    ┌────────────┐                                    │ │
│  │                    │  Conv1D    │                                    │ │
│  │                    │  k=1       │  Reduce channels                   │ │
│  │                    └────────────┘                                    │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                           │
│                               ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                   POSITIONAL ENCODING                                │ │
│  │         Add sinusoidal position information to embeddings            │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                           │
│                               ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              TRANSFORMER ENCODER (N layers)                          │ │
│  │  ┌───────────────────────────────────────────────────────────────┐  │ │
│  │  │  Multi-Head Self-Attention                                    │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │  Q  K  V  ──→ Attention Scores ──→ Weighted Sum          │  │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │ │
│  │  │                           │                                   │  │ │
│  │  │                    Add & LayerNorm                            │  │ │
│  │  │                           │                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │  Separable Feed-Forward Network                         │  │  │ │
│  │  │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐          │  │  │ │
│  │  │  │  │ Depthwise│ ──→│   ReLU   │ ──→│ Pointwise│          │  │  │ │
│  │  │  │  │  Conv    │    │          │    │   Conv   │          │  │  │ │
│  │  │  │  └──────────┘    └──────────┘    └──────────┘          │  │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │ │
│  │  │                           │                                   │  │ │
│  │  │                    Add & LayerNorm                            │  │ │
│  │  └───────────────────────────┬───────────────────────────────────┘  │ │
│  │                              │ × N layers                           │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        GLOBAL POOLING                                │ │
│  │              Average pool over sequence dimension                    │ │
│  └────────────────────────────┬────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    CLASSIFICATION HEAD                               │ │
│  │           Linear → Dropout → Linear → Softmax                        │ │
│  │              Output: [Up, Down, Stable] probabilities                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Inception Convolutional Embedding

The inception module uses parallel convolutions with different kernel sizes to capture patterns at multiple temporal scales:

```python
class InceptionConvEmbedding(nn.Module):
    """
    Inception-style convolutional embedding for time series.

    Captures local patterns at multiple temporal scales simultaneously:
    - kernel_size=1: Point-wise features (immediate price changes)
    - kernel_size=3: Short-term patterns (2-3 day movements)
    - kernel_size=5: Medium-term patterns (weekly trends)
    - max_pool: Most prominent features in local window
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Parallel branches with different receptive fields
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        )

        # Reduce concatenated channels
        self.reduce = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)

        # Concatenate along channel dimension
        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.reduce(out)
        return self.activation(out)
```

**Why Inception?**
- Different kernel sizes capture patterns at different time scales
- MaxPooling branch extracts the most prominent features
- 1x1 convolutions reduce dimensionality efficiently
- Parallel branches are computed simultaneously (efficient on GPU)

### Multi-Head Self-Attention

Standard Transformer attention adapted for time series:

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal dependencies.

    Each attention head can learn to focus on:
    - Recent time steps (short-term momentum)
    - Periodic patterns (daily/weekly seasonality)
    - Key events (earnings, announcements)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: [batch, seq_len, d_model]
        B, L, D = x.shape

        # Compute Q, K, V in single projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        if return_attention:
            return out, attn
        return out
```

### Separable Fully Connected Layers

Inspired by depthwise separable convolutions, this reduces parameter count:

```python
class SeparableFFN(nn.Module):
    """
    Separable feed-forward network.

    Decomposes the standard FFN into:
    1. Depthwise operation: processes each channel independently
    2. Pointwise operation: mixes information across channels

    This reduces parameters from O(d_model * d_ff) to O(d_model + d_ff)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # Depthwise: expand dimension while keeping channels separate
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=1, groups=d_model
        )

        # Pointwise: mix across expanded channels
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Apply depthwise conv
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.depthwise(x)
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]

        # Apply pointwise transformation
        return self.pointwise(x)
```

### Classification Head

The final layer predicts stock movement direction:

```python
class MovementClassifier(nn.Module):
    """
    Classification head for stock movement prediction.

    Output classes:
    - Up: Price increase > threshold
    - Down: Price decrease > threshold
    - Stable: Price change within threshold
    """

    def __init__(self, d_model, num_classes=3, dropout=0.2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [batch, d_model] (after global pooling)
        return self.classifier(x)
```

## Data Preprocessing

### Feature Engineering

Standard OHLCV features plus technical indicators:

```python
def prepare_features(df):
    """
    Prepare features for DCT model.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with normalized features
    """
    features = pd.DataFrame()

    # Price features (normalized)
    features['open'] = df['open'] / df['close']
    features['high'] = df['high'] / df['close']
    features['low'] = df['low'] / df['close']
    features['close'] = df['close'].pct_change()

    # Volume (log-normalized)
    features['volume'] = np.log1p(df['volume'] / df['volume'].rolling(20).mean())

    # Technical indicators
    features['ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
    features['volatility'] = df['close'].pct_change().rolling(20).std()
    features['rsi'] = compute_rsi(df['close'], 14)
    features['macd'] = compute_macd(df['close'])

    return features.dropna()
```

### Normalization Techniques

DCT uses several normalization approaches:

1. **Z-score normalization**: For most features
2. **Min-max scaling**: For bounded indicators (RSI)
3. **Log transformation**: For volume data
4. **Ratio normalization**: Price relative to moving average

```python
class FeatureNormalizer:
    def __init__(self):
        self.scalers = {}

    def fit_transform(self, data, columns):
        normalized = pd.DataFrame()
        for col in columns:
            self.scalers[col] = StandardScaler()
            normalized[col] = self.scalers[col].fit_transform(
                data[col].values.reshape(-1, 1)
            ).flatten()
        return normalized
```

### Look-Back Window

The paper uses a 30-day look-back window:

```python
def create_sequences(data, lookback=30, horizon=1):
    """
    Create sequences for training.

    Args:
        data: Feature matrix [n_samples, n_features]
        lookback: Number of past days to use
        horizon: Number of days ahead to predict

    Returns:
        X: [n_sequences, lookback, n_features]
        y: [n_sequences] movement labels
    """
    X, y = [], []

    for i in range(lookback, len(data) - horizon):
        X.append(data[i-lookback:i])

        # Calculate movement label
        future_return = (data[i + horizon, 3] - data[i, 3]) / data[i, 3]
        if future_return > 0.005:  # Up threshold
            y.append(0)  # Up
        elif future_return < -0.005:  # Down threshold
            y.append(1)  # Down
        else:
            y.append(2)  # Stable

    return np.array(X), np.array(y)
```

## Practical Examples

### 01: Data Preparation

```python
# Example data preparation (see python/data.py for full implementation)

import pandas as pd
import numpy as np
from typing import Tuple, List
import yfinance as yf

def download_stock_data(
    symbol: str,
    start_date: str = "2013-01-01",
    end_date: str = "2024-08-31"
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for DCT model.
    """
    result = df.copy()

    # Price ratios
    result['hl_ratio'] = (df['high'] - df['low']) / df['close']
    result['oc_ratio'] = (df['close'] - df['open']) / df['open']

    # Moving averages
    for window in [5, 10, 20]:
        result[f'ma_{window}'] = df['close'].rolling(window).mean()
        result[f'ma_ratio_{window}'] = df['close'] / result[f'ma_{window}']

    # Volatility
    result['volatility'] = df['close'].pct_change().rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    result['macd'] = ema12 - ema26
    result['macd_signal'] = result['macd'].ewm(span=9).mean()

    # Volume features
    result['volume_ma'] = df['volume'].rolling(20).mean()
    result['volume_ratio'] = df['volume'] / result['volume_ma']

    return result.dropna()


def create_movement_labels(
    prices: pd.Series,
    threshold: float = 0.005,
    horizon: int = 1
) -> pd.Series:
    """
    Create movement labels based on future returns.

    Args:
        prices: Close prices
        threshold: Movement threshold (0.5% default)
        horizon: Prediction horizon in days

    Returns:
        Series with labels: 0=Up, 1=Down, 2=Stable
    """
    future_return = prices.pct_change(horizon).shift(-horizon)

    labels = pd.Series(index=prices.index, dtype=int)
    labels[future_return > threshold] = 0   # Up
    labels[future_return < -threshold] = 1  # Down
    labels[(future_return >= -threshold) & (future_return <= threshold)] = 2  # Stable

    return labels


if __name__ == "__main__":
    # Example usage
    df = download_stock_data("AAPL")
    df = compute_technical_indicators(df)
    df['label'] = create_movement_labels(df['close'])

    print(f"Data shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
```

### 02: DCT Model Architecture

See [python/model.py](python/model.py) for complete implementation.

### 03: Training Pipeline

```python
# Example training pipeline (see python/example_usage.py for full implementation)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import DCTModel, DCTConfig
import numpy as np

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10
):
    """
    Train DCT model with early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs['logits'], y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs['logits'], y_batch)
                val_loss += loss.item()

                _, predicted = outputs['logits'].max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

### 04: Stock Movement Prediction

```python
# Example prediction (see python/model.py for predict_movement method)

import torch
import numpy as np
from model import DCTModel, DCTConfig

def predict_movement(model, X, return_attention=False):
    """
    Predict stock movement direction.

    Args:
        model: Trained DCT model
        X: Input features [batch, seq_len, features]
        return_attention: Whether to return attention weights

    Returns:
        predictions: Movement class (0=Up, 1=Down, 2=Stable)
        probabilities: Class probabilities
        attention: Optional attention weights
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        outputs = model(X, return_attention=return_attention)

        probs = torch.softmax(outputs['logits'], dim=-1)
        predictions = probs.argmax(dim=-1)

        result = {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'class_names': ['Up', 'Down', 'Stable']
        }

        if return_attention:
            result['attention'] = outputs['attention']

    return result


def visualize_attention(attention_weights, timestamps):
    """
    Visualize attention weights to understand model focus.
    """
    import matplotlib.pyplot as plt

    # Average over heads
    avg_attention = attention_weights.mean(axis=1)[0]  # [seq_len, seq_len]

    plt.figure(figsize=(12, 8))
    plt.imshow(avg_attention, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position (Past Days)')
    plt.ylabel('Query Position (Current Day)')
    plt.title('DCT Temporal Attention Weights')
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.close()
```

### 05: Backtesting Strategy

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict

def backtest_dct_strategy(
    model,
    test_data: np.ndarray,
    test_prices: pd.Series,
    initial_capital: float = 100000,
    position_size: float = 0.1,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Backtest DCT movement prediction strategy.

    Strategy:
    - Predict Up → Buy (long position)
    - Predict Down → Sell/Short
    - Predict Stable → Hold current position
    """
    capital = initial_capital
    position = 0  # Current position (-1, 0, 1)

    results = []
    trades = []

    for i in range(len(test_data)):
        current_price = test_prices.iloc[i]

        # Get prediction
        X = test_data[i:i+1]
        pred = predict_movement(model, X)
        movement = pred['predictions'][0]
        confidence = pred['probabilities'][0].max()

        # Trading logic
        new_position = position
        if movement == 0 and confidence > 0.6:  # Confident Up
            new_position = 1
        elif movement == 1 and confidence > 0.6:  # Confident Down
            new_position = -1
        elif movement == 2:  # Stable
            pass  # Keep position

        # Execute trade if position changes
        if new_position != position:
            trade_cost = abs(new_position - position) * position_size * current_price * transaction_cost
            capital -= trade_cost

            trades.append({
                'date': test_prices.index[i],
                'price': current_price,
                'from_position': position,
                'to_position': new_position,
                'cost': trade_cost
            })

            position = new_position

        # Calculate P&L
        if i > 0:
            price_change = (current_price - test_prices.iloc[i-1]) / test_prices.iloc[i-1]
            pnl = position * position_size * capital * price_change
            capital += pnl

        results.append({
            'date': test_prices.index[i],
            'capital': capital,
            'position': position,
            'prediction': ['Up', 'Down', 'Stable'][movement],
            'confidence': confidence
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics
    returns = results_df['capital'].pct_change().dropna()

    metrics = {
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(results_df['capital']),
        'num_trades': len(trades),
        'win_rate': calculate_win_rate(trades, test_prices)
    }

    return {
        'results': results_df,
        'trades': pd.DataFrame(trades),
        'metrics': metrics
    }


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown percentage."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100


def calculate_win_rate(trades, prices):
    """Calculate percentage of winning trades."""
    if len(trades) < 2:
        return 0

    wins = 0
    for i in range(1, len(trades)):
        entry = trades[i-1]
        exit_trade = trades[i]

        price_change = (exit_trade['price'] - entry['price']) / entry['price']

        if entry['to_position'] == 1 and price_change > 0:
            wins += 1
        elif entry['to_position'] == -1 and price_change < 0:
            wins += 1

    return wins / (len(trades) - 1) * 100
```

## Rust Implementation

See [rust_dct](rust_dct/) for complete Rust implementation using Bybit data.

```
rust_dct/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # DCT architecture
│   │   ├── mod.rs
│   │   ├── inception.rs    # Inception conv embedding
│   │   ├── attention.rs    # Multi-head attention
│   │   ├── encoder.rs      # Transformer encoder
│   │   └── dct.rs          # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── demo.rs             # Complete usage demonstration
    └── fetch_bybit.rs      # Download Bybit data
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_dct

# Fetch data from Bybit
cargo run --example fetch_bybit

# Run complete demo (model creation, data processing, backtest)
cargo run --example demo
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── model.py                # Main DCT model implementation
├── data.py                 # Data loading (yfinance, Bybit) and feature engineering
├── strategy.py             # Trading strategy and backtesting utilities
├── example_usage.py        # Complete usage demonstration
├── requirements.txt        # Dependencies
└── __init__.py             # Package exports
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r python/requirements.txt

# Run complete example (crypto trading)
python python/example_usage.py --mode crypto

# Run stock trading example
python python/example_usage.py --mode stock

# Run both examples
python python/example_usage.py --mode both
```

## Best Practices

### When to Use DCT

**Good use cases:**
- Stock movement direction prediction
- Binary (Up/Down) or ternary (Up/Down/Stable) classification
- Medium-term predictions (daily to weekly)
- Markets with clear trending behavior

**Not ideal for:**
- High-frequency trading (latency concerns)
- Exact price prediction (use regression models)
- Very noisy/illiquid markets
- Short sequences (<10 time steps)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `d_model` | 64-128 | Higher for more complex patterns |
| `num_heads` | 4-8 | Should divide d_model |
| `num_layers` | 2-4 | More can overfit |
| `lookback` | 30 | As per the paper |
| `dropout` | 0.1-0.3 | Higher for small datasets |
| `threshold` | 0.5% | Movement classification threshold |

### Common Pitfalls

1. **Class imbalance**: Market often trends, causing imbalanced labels
   - Solution: Use weighted loss or resampling

2. **Look-ahead bias**: Using future information in features
   - Solution: Careful feature engineering and date-aware splits

3. **Overfitting**: Financial data has low signal-to-noise ratio
   - Solution: Strong regularization, early stopping, cross-validation

4. **Non-stationarity**: Market regimes change over time
   - Solution: Rolling training windows, regime detection

## Resources

### Papers

- [Deep Convolutional Transformer Network for Stock Movement Prediction](https://www.mdpi.com/2079-9292/13/21/4225) - Original DCT paper (2024)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) - Inception architecture

### Implementations

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Burn](https://burn.dev/) - Rust deep learning framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data

### Related Chapters

- [Chapter 18: Convolutional Neural Networks](../18_convolutional_neural_nets) - CNN basics
- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) - TFT for forecasting
- [Chapter 41: Higher Order Transformers](../41_higher_order_transformers) - Advanced attention
- [Chapter 43: Stockformer](../43_stockformer_multivariate) - Multivariate prediction

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Understanding of CNN architectures
- Transformer and attention mechanisms
- Time series preprocessing
- PyTorch or similar deep learning framework
