# Saliency Maps for Trading: Visualizing Model Interpretability in Financial Predictions

Saliency maps are powerful visualization techniques that reveal which input features most strongly influence a neural network's predictions. Originally developed for computer vision to highlight important regions in images, saliency maps have been adapted for time series analysis in trading, helping traders and researchers understand why models make specific predictions about market movements.

In algorithmic trading, model interpretability is crucial for several reasons:
- **Risk Management**: Understanding which factors drive predictions helps identify potential model failures
- **Regulatory Compliance**: Financial regulators increasingly require explainability in automated trading systems
- **Strategy Development**: Insights from saliency maps can inform new trading hypotheses
- **Model Debugging**: Visualizing feature importance helps detect overfitting or spurious correlations

This chapter covers the theory and implementation of saliency maps for financial time series, with practical applications using both stock market and cryptocurrency data.

## Content

1. [Introduction to Saliency Maps](#introduction-to-saliency-maps)
2. [Types of Saliency Methods](#types-of-saliency-methods)
   * [Vanilla Gradients](#vanilla-gradients)
   * [Gradient × Input](#gradient--input)
   * [Integrated Gradients](#integrated-gradients)
   * [SmoothGrad](#smoothgrad)
3. [Saliency Maps for Time Series](#saliency-maps-for-time-series)
4. [Implementation with PyTorch](#implementation-with-pytorch)
   * [Code Example: Building a Trading Model](#code-example-building-a-trading-model)
   * [Code Example: Computing Saliency Maps](#code-example-computing-saliency-maps)
5. [Trading Strategy Based on Saliency](#trading-strategy-based-on-saliency)
   * [Feature Importance Analysis](#feature-importance-analysis)
   * [Adaptive Feature Selection](#adaptive-feature-selection)
6. [Backtesting the Strategy](#backtesting-the-strategy)
7. [Rust Implementation](#rust-implementation)
8. [References](#references)

## Introduction to Saliency Maps

A saliency map is a visualization that highlights which parts of the input have the greatest impact on the model's output. For a neural network with output y and input x, the saliency is computed as the gradient of the output with respect to the input:

```
S(x) = ∂y/∂x
```

In the context of trading, if our model predicts price direction based on historical OHLCV (Open, High, Low, Close, Volume) data, the saliency map tells us which time steps and which features (price, volume, technical indicators) the model considers most important for its prediction.

### Why Saliency Maps Matter for Trading

Traditional feature importance methods like permutation importance or SHAP values provide global insights but may miss temporal dynamics. Saliency maps offer:

- **Instance-level explanations**: Understand specific trade signals
- **Temporal resolution**: See which time steps matter most
- **Feature interactions**: Identify when combinations of features trigger signals
- **Real-time computation**: Gradients are fast to compute during live trading

## Types of Saliency Methods

### Vanilla Gradients

The simplest saliency method computes the gradient of the output class score with respect to the input:

```python
saliency = torch.autograd.grad(output, input)[0]
```

The absolute value of the gradient indicates the sensitivity of the prediction to each input feature. Larger gradients suggest more important features.

**Limitations**:
- Can be noisy due to sharp gradients in ReLU networks
- May highlight irrelevant features that happen to have large local gradients

### Gradient × Input

This method multiplies the gradient by the input value itself:

```
S(x) = x × ∂y/∂x
```

This modification ensures that features with zero input values have zero saliency, providing better attribution. It captures both the sensitivity (gradient) and the actual contribution (input magnitude).

### Integrated Gradients

Integrated Gradients (IG) addresses the gradient saturation problem by integrating gradients along a path from a baseline (typically zero) to the actual input:

```
IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x - x'))/∂x dα
```

Where x' is the baseline input. This method satisfies important axioms:
- **Sensitivity**: If changing a feature changes the prediction, it gets non-zero attribution
- **Implementation Invariance**: Attributions are the same for functionally equivalent networks
- **Completeness**: Attributions sum to the difference between output at x and baseline

### SmoothGrad

SmoothGrad reduces noise in gradient-based saliency by averaging gradients over multiple noisy versions of the input:

```
SmoothGrad(x) = (1/n) × Σ ∂y/∂(x + N(0, σ²))
```

This produces visually cleaner saliency maps while preserving the most important attributions.

## Saliency Maps for Time Series

Adapting saliency maps to financial time series requires considering the temporal structure:

### Input Representation

For a trading model, the input typically has shape `(batch, sequence_length, features)`:
- **sequence_length**: Number of historical time steps (e.g., 60 days)
- **features**: OHLCV data, technical indicators, fundamental data

### Temporal Saliency Visualization

The saliency map has the same shape as the input. We can visualize it as:
- **Heatmap**: Time steps × Features showing importance
- **Aggregated time importance**: Sum absolute saliency over features for each time step
- **Aggregated feature importance**: Sum absolute saliency over time for each feature

### Important Considerations

1. **Normalization**: Scale saliency values for visualization (e.g., min-max or percentile)
2. **Sign**: Positive gradients push toward bullish predictions; negative toward bearish
3. **Baseline selection**: For integrated gradients, choose appropriate baselines (zero, historical mean, or market-neutral state)

## Implementation with PyTorch

### Code Example: Building a Trading Model

The notebook [01_saliency_trading_model.ipynb](01_saliency_trading_model.ipynb) demonstrates building a neural network for price direction prediction:

```python
import torch
import torch.nn as nn

class TradingLSTM(nn.Module):
    """LSTM model for predicting price direction."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.sigmoid(self.fc(last_hidden))
        return output
```

### Code Example: Computing Saliency Maps

The notebook [02_saliency_computation.ipynb](02_saliency_computation.ipynb) shows how to compute various saliency maps:

```python
class SaliencyComputer:
    """Compute saliency maps for trading models."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def vanilla_gradient(self, x):
        """Compute vanilla gradient saliency."""
        x = x.clone().requires_grad_(True)
        output = self.model(x)
        output.backward(torch.ones_like(output))
        return x.grad.abs()

    def gradient_x_input(self, x):
        """Compute gradient × input saliency."""
        x = x.clone().requires_grad_(True)
        output = self.model(x)
        output.backward(torch.ones_like(output))
        return (x.grad * x).abs()

    def integrated_gradients(self, x, baseline=None, steps=50):
        """Compute integrated gradients."""
        if baseline is None:
            baseline = torch.zeros_like(x)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
        interpolated = baseline + alphas * (x - baseline)
        interpolated = interpolated.view(-1, x.shape[1], x.shape[2])
        interpolated.requires_grad_(True)

        # Compute gradients
        outputs = self.model(interpolated)
        outputs.sum().backward()

        # Average gradients and multiply by (x - baseline)
        avg_grads = interpolated.grad.view(steps, -1, x.shape[1], x.shape[2]).mean(0)
        ig = (x - baseline) * avg_grads
        return ig.abs()
```

## Trading Strategy Based on Saliency

### Feature Importance Analysis

Saliency maps enable dynamic feature importance analysis:

1. **Identify regime changes**: When the model suddenly starts focusing on different features, it may signal a market regime change
2. **Validate trading signals**: High-confidence predictions should have clear, interpretable saliency patterns
3. **Filter noise**: Ignore signals where saliency is diffuse or focuses on irrelevant features

### Adaptive Feature Selection

The notebook [03_adaptive_strategy.ipynb](03_adaptive_strategy.ipynb) implements a strategy that:

1. Computes saliency maps for each prediction
2. Identifies the top-k most important features for that prediction
3. Only trades when the important features align with domain knowledge
4. Adjusts position size based on saliency concentration

```python
def saliency_weighted_signal(model, x, feature_names, threshold=0.7):
    """Generate trading signal weighted by saliency interpretability."""

    # Get prediction and saliency
    saliency = compute_saliency(model, x)
    prediction = model(x).item()

    # Aggregate saliency by feature
    feature_importance = saliency.mean(dim=1).squeeze()  # Average over time

    # Check if top features are interpretable
    top_features = feature_importance.topk(3).indices
    interpretable = check_feature_interpretability(top_features, feature_names)

    # Compute concentration (entropy-based)
    concentration = 1 - entropy(feature_importance.softmax(dim=0))

    if interpretable and concentration > threshold:
        return prediction, concentration
    else:
        return 0.5, 0  # No signal
```

## Backtesting the Strategy

The notebook [04_backtest.ipynb](04_backtest.ipynb) demonstrates backtesting with performance metrics:

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return: (Return - Rf) / Std |
| **Sortino Ratio** | Downside-adjusted return |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |

### Comparison with Baseline

We compare:
1. **Baseline Model**: Trade on all signals
2. **Saliency-Filtered**: Only trade when saliency is interpretable
3. **Saliency-Weighted**: Position size based on saliency concentration

## Rust Implementation

The `rust_examples/` directory contains a production-ready Rust implementation with:

- **High-performance inference**: Optimized for low-latency trading
- **Bybit API integration**: Real-time cryptocurrency data
- **Saliency computation**: Gradient-based attribution in Rust

See [rust_examples/README.md](rust_examples/README.md) for details.

### Running Rust Examples

```bash
cd rust_examples

# Fetch data from Bybit
cargo run --example fetch_data

# Train model and compute saliency
cargo run --example saliency_trading

# Run backtest
cargo run --example backtest
```

## References

1. **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps**
   - Simonyan, K., Vedaldi, A., & Zisserman, A. (2013)
   - URL: https://arxiv.org/abs/1312.6034
   - *Introduced vanilla gradient saliency maps*

2. **Axiomatic Attribution for Deep Networks (Integrated Gradients)**
   - Sundararajan, M., Taly, A., & Yan, Q. (2017)
   - URL: https://arxiv.org/abs/1703.01365
   - *Proposed integrated gradients method*

3. **SmoothGrad: Removing Noise by Adding Noise**
   - Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017)
   - URL: https://arxiv.org/abs/1706.03825
   - *Introduced SmoothGrad for cleaner attributions*

4. **Interpretable Machine Learning for Finance**
   - Molnar, C. (2022)
   - URL: https://christophm.github.io/interpretable-ml-book/
   - *Comprehensive guide to ML interpretability*

5. **Explainable AI for Algorithmic Trading**
   - Chen, J., et al. (2021)
   - *Application of interpretability methods to trading*
