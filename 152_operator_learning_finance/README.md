# Chapter 152: Operator Learning for Finance

## Overview

Operator learning represents a paradigm shift in applying neural networks to financial problems. Rather than learning mappings between finite-dimensional vectors (as standard neural networks do), operator learning frameworks learn mappings between **function spaces** --- that is, they learn operators of the form:

```
G: u(x) -> s(x)
```

where both the input `u` and output `s` are functions. This is profoundly relevant for finance, where many core problems are naturally expressed as operator equations: mapping yield curves to bond prices, volatility surfaces to option values, initial market conditions to future price distributions, or boundary conditions to PDE solutions.

## Why Operator Learning for Finance?

### The Function-Space Perspective

Financial markets produce data that is inherently functional:

- **Yield curves** are functions of maturity: `r(T)` for `T in [0, 30]` years
- **Volatility surfaces** are functions of strike and expiry: `sigma(K, T)`
- **Order book profiles** are functions of price: `q(p)` for bid/ask sides
- **Price paths** are functions of time: `S(t)` for `t in [0, T]`

Traditional ML approaches discretize these functions into fixed-length vectors, losing the continuous nature and creating resolution dependencies. Operator learning preserves the functional structure.

### Key Advantages Over Standard Neural Networks

| Feature | Standard NN | Neural Operator |
|---------|------------|-----------------|
| Input/Output | Fixed-dimension vectors | Functions (any resolution) |
| Discretization | Resolution-dependent | Resolution-invariant |
| Generalization | Interpolation in R^n | Generalization across function spaces |
| Transfer | Limited to same grid | Zero-shot to new grids/resolutions |
| Physics | Black-box | Can encode PDE structure |

### Financial Applications

1. **PDE Solution Operators**: Learn the Black-Scholes, Heston, or SABR solution operator once, then evaluate for any parameter set instantly
2. **Yield Curve Evolution**: Map current yield curve shape to future yield curve under different scenarios
3. **Volatility Surface Dynamics**: Learn how implied volatility surfaces evolve through time
4. **Risk Management**: Map portfolio positions (functions of asset) to risk measure functions
5. **Crypto Market Microstructure**: Learn order book dynamics operators for real-time trading

## Mathematical Foundation

### Operator Learning Problem

Given observations `{(u_i, s_i)}_{i=1}^{N}` where `u_i` and `s_i` are functions, learn an operator:

```
G_theta: U -> S
```

parameterized by `theta`, such that `G_theta(u) ≈ G*(u)` for the true operator `G*`.

The key challenge is that `U` and `S` are infinite-dimensional function spaces (e.g., Banach spaces), and we need architectures that can handle this infinite-dimensional nature.

### Universal Approximation for Operators

**Theorem (Chen & Chen, 1995)**: For any continuous nonlinear operator `G: U -> S` and any `epsilon > 0`, there exists a neural network architecture that can approximate `G` to within `epsilon` accuracy.

This foundational result guarantees that neural operator architectures have the capacity to learn any continuous operator mapping between function spaces.

## Key Architectures

### 1. DeepONet (Deep Operator Network)

DeepONet consists of two sub-networks:

- **Branch network**: Encodes the input function `u` at sensor points `{x_1, ..., x_m}`
- **Trunk network**: Encodes the evaluation location `y`

The output is computed as:

```
G_theta(u)(y) = sum_{k=1}^{p} b_k(u(x_1), ..., u(x_m)) * t_k(y) + b_0
```

where `b_k` are outputs of the branch network and `t_k` are outputs of the trunk network.

```
Architecture Diagram:

Input function u(x)          Evaluation point y
at sensor points
[u(x_1), ..., u(x_m)]               |
        |                            |
   Branch Net                   Trunk Net
   (FNN/CNN/RNN)               (FNN)
        |                            |
   [b_1, ..., b_p]            [t_1, ..., t_p]
        \                          /
         \                        /
          Dot Product + Bias
                 |
          G(u)(y) = sum b_k * t_k + b_0
```

**Financial Application**: Given a yield curve `u(T)` sampled at maturities `{T_1, ..., T_m}`, predict the bond price function `P(T)` at any maturity `T`.

### 2. Fourier Neural Operator (FNO)

FNO operates in Fourier space, learning global operators through spectral convolutions:

```
v_{l+1}(x) = sigma(W_l * v_l(x) + K_l(v_l)(x))
```

where `K_l` is a kernel integral operator parameterized in Fourier space:

```
K_l(v)(x) = F^{-1}(R_l * F(v))(x)
```

Here `F` is the Fourier transform and `R_l` is a learnable complex-valued weight tensor.

```
FNO Layer:

v_l(x) ──┬── FFT ── R_l (pointwise multiply) ── IFFT ──┬── sigma ── v_{l+1}(x)
          │                                              │
          └──────────── W_l (linear transform) ─────────┘
```

**Financial Application**: Learn the operator mapping initial price distributions to evolved distributions under various stochastic processes, capturing the full dynamics in Fourier space.

### 3. Kernel-Based Operator Learning

Graph Neural Operator (GNO) and kernel-based methods learn operators through:

```
(K(a; phi) v)(x) = integral k_phi(x, y, a(x), a(y)) v(y) dy
```

where `k_phi` is a learned kernel function parameterized by `phi`.

This is particularly suited for irregular financial data (e.g., non-uniformly spaced yield curve observations, irregularly sampled tick data).

### 4. Transformer-Based Operators

Attention mechanisms naturally implement operator-like computations:

```
Operator Attention(Q, K, V)(x) = integral softmax(q(x)^T k(y) / sqrt(d)) v(y) dy
```

This continuous attention formulation extends transformers to operator learning, enabling them to process functions at arbitrary resolutions.

## Application: PDE Solution Operators for Option Pricing

### Black-Scholes Operator

The Black-Scholes PDE:

```
dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0
```

with terminal condition `V(S, T) = max(S - K, 0)` for a call option.

**Operator formulation**: Learn `G: (sigma(.), r(.), payoff(.)) -> V(S, t)` mapping parameter functions and boundary conditions to the full price surface.

Advantages over traditional methods:
- **Finite Difference**: O(N^2) per solve; operator: O(1) after training
- **Monte Carlo**: Variance issues; operator: deterministic evaluation
- **Analytical**: Limited to simple models; operator: works for any PDE

### Heston Model Operator

```
dS = mu * S * dt + sqrt(v) * S * dW_1
dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_2
<dW_1, dW_2> = rho * dt
```

Learn `G: (kappa, theta, xi, rho, v_0, payoff) -> V(S, v, t)` to instantly price options under any Heston parameter configuration.

## Application: Yield Curve Evolution

### Problem Setup

Given today's yield curve `r_0(T)` for `T in [0, 30]`, predict the yield curve `r_t(T)` at future time `t` under various market scenarios.

**Operator**: `G: r_0(T) -> r_t(T)` parameterized by scenario indicators.

### Multi-Factor HJM Framework

The Heath-Jarrow-Morton framework models yield curve dynamics:

```
df(t, T) = alpha(t, T) dt + sum_{i=1}^{d} sigma_i(t, T) dW_i(t)
```

Learning the HJM solution operator captures the full yield curve dynamics including:
- Parallel shifts
- Steepening/flattening
- Butterfly movements
- Higher-order deformations

## Application: Volatility Surface Dynamics

### Learning Vol Surface Evolution

The implied volatility surface `sigma_imp(K, T)` evolves through time. An operator:

```
G: sigma_imp(K, T; t=0) -> sigma_imp(K, T; t=delta_t)
```

can predict how the entire surface moves, enabling:
- Dynamic hedging with surface-level Greeks
- Arbitrage detection across strikes/maturities
- Risk scenario generation

### Zero-Shot Generalization

A key advantage: after training on standard maturities (1M, 3M, 6M, 1Y, 2Y), the operator can predict volatility at **any** maturity --- including those never seen during training. This "zero-shot" generalization to new maturities and strikes is impossible with standard neural networks.

## Application: Crypto Trading with Bybit Data

### Order Book Operator

Learn the operator mapping current order book shape to future mid-price distribution:

```
G: (bid_profile(p), ask_profile(p), recent_trades) -> P(delta_mid | tau)
```

This enables:
- High-frequency signal generation
- Optimal execution strategies
- Market impact prediction

### Funding Rate Dynamics

For perpetual futures on Bybit, learn the operator:

```
G: (funding_history(t), price_spread(t), OI(t)) -> funding_rate(t + delta)
```

to predict funding rate evolution and capture funding rate arbitrage opportunities.

## Transfer Learning Across Market Conditions

### Regime-Adaptive Operators

Train operators on data spanning multiple market regimes:

```
G_regime: (market_state_function, input_function) -> output_function
```

where `market_state_function` encodes the current regime (volatility level, correlation structure, liquidity conditions).

### Cross-Asset Transfer

An operator trained on one asset class can transfer to another:
- Equity vol surface operator -> FX vol surface operator
- Treasury yield curve operator -> Corporate bond spread operator
- BTC order book operator -> ETH order book operator

This works because operators learn the **structural mapping** rather than asset-specific patterns.

## Comparison with Traditional Methods

### Option Pricing Benchmark

| Method | Training Time | Inference Time | Error (RMSE) | Generalizes to New Params? |
|--------|--------------|----------------|--------------|---------------------------|
| Finite Difference | N/A | 100ms | Exact | Yes (re-solve) |
| Monte Carlo (10K) | N/A | 500ms | ~0.01 | Yes (re-simulate) |
| Standard NN | 2 hours | 0.1ms | ~0.005 | No (retrain needed) |
| DeepONet | 4 hours | 0.2ms | ~0.003 | Yes (zero-shot) |
| FNO | 3 hours | 0.15ms | ~0.002 | Yes (zero-shot) |

### Key Takeaway

Neural operators trade **one-time training cost** for **instant inference with generalization**. Once trained, they can evaluate for any parameter configuration without retraining, making them ideal for real-time trading systems.

## Implementation Details

### Data Generation for Training

For PDE-based applications, training data is generated by solving PDEs with various parameters:

```python
# Generate training pairs (parameter_function, solution_function)
training_data = []
for params in parameter_grid:
    # Solve PDE with these parameters using traditional method
    solution = solve_pde(params, grid)
    training_data.append((params, solution))
```

For market data applications:
```python
# Collect function-valued observations from market
yield_curves_today = fetch_yield_curves(date_range)
yield_curves_future = fetch_yield_curves(date_range + horizon)
# Each (today, future) pair is a training example
```

### Loss Functions

**Relative L2 Loss** (standard for operator learning):
```
L(theta) = (1/N) * sum_{i=1}^{N} ||G_theta(u_i) - s_i||_2 / ||s_i||_2
```

**Functional Sobolev Loss** (includes derivative matching):
```
L_H1(theta) = L_L2 + lambda * (1/N) * sum_{i=1}^{N} ||dG_theta(u_i)/dx - ds_i/dx||_2 / ||ds_i/dx||_2
```

### Training Strategy

1. **Multi-resolution training**: Train on coarse grids first, then fine-tune on finer grids
2. **Curriculum learning**: Start with simple parameter ranges, gradually increase complexity
3. **Physics-informed loss**: Add PDE residual as regularization term
4. **Data augmentation**: Generate synthetic function pairs using known transformations

## Architecture Design for Financial Operators

### DeepONet for Yield Curves

```
Branch Network:
  Input: [r(T_1), r(T_2), ..., r(T_m)]  (yield curve at m maturities)
  Architecture: 1D CNN or Transformer encoder
  Output: [b_1, ..., b_p]  (p latent coefficients)

Trunk Network:
  Input: [T, t, scenario_features]  (maturity, time horizon, market state)
  Architecture: MLP with residual connections
  Output: [t_1, ..., t_p]  (p basis function evaluations)

Output: r_predicted(T, t) = sum(b_k * t_k) + bias
```

### FNO for Volatility Surfaces

```
Input: sigma_imp(K_i, T_j) on grid [K_1...K_n] x [T_1...T_m]

FNO Architecture:
  Lifting: R^1 -> R^d (channel expansion)
  4 Fourier Layers:
    - 2D FFT
    - Spectral truncation (keep K_max modes)
    - Learnable spectral weights R_l
    - 2D IFFT
    - Linear bypass W_l
    - GELU activation
  Projection: R^d -> R^1

Output: sigma_imp_next(K_i, T_j) on same or different grid
```

## Advanced Topics

### Physics-Informed Neural Operators (PINO)

Combine operator learning with PDE constraints:

```
L_total = L_data + alpha * L_PDE + beta * L_BC
```

where:
- `L_data`: Standard data-fitting loss
- `L_PDE`: PDE residual evaluated at collocation points
- `L_BC`: Boundary/terminal condition satisfaction

This dramatically reduces data requirements and improves extrapolation.

### Multi-Fidelity Operator Learning

Use cheap low-fidelity solvers (coarse grids, few Monte Carlo paths) to generate abundant training data, and expensive high-fidelity solvers (fine grids, many paths) for a small number of correction examples:

```
G_HF(u) ≈ G_LF(u) + G_correction(u, G_LF(u))
```

### Compositional Operators

Chain operators to build complex financial models:

```
G_total = G_pricing ∘ G_vol_dynamics ∘ G_rate_evolution
```

Each sub-operator can be trained independently and composed at inference time.

## Practical Considerations

### Computational Requirements

- **Training**: GPU recommended (NVIDIA A100/V100), 2-8 hours depending on problem complexity
- **Inference**: CPU sufficient for real-time applications, ~0.1-1ms per evaluation
- **Memory**: O(N * p) where N is resolution and p is number of latent dimensions

### Hyperparameter Selection

| Hyperparameter | DeepONet | FNO |
|---------------|----------|-----|
| Latent dim (p) | 64-256 | N/A |
| Fourier modes | N/A | 12-32 |
| Hidden layers | 4-8 | 4-6 Fourier layers |
| Hidden width | 128-512 | 32-128 channels |
| Learning rate | 1e-3 to 1e-4 | 1e-3 to 1e-4 |
| Batch size | 32-128 | 16-64 |

### Common Pitfalls

1. **Insufficient resolution during training**: Train on highest resolution available
2. **Ignoring normalization**: Function inputs and outputs should be normalized
3. **Overfitting to specific parameter ranges**: Use diverse training distributions
4. **Neglecting boundary conditions**: Explicitly enforce financial constraints (no-arbitrage, put-call parity)

## Code Structure

### Python Implementation

```
python/
  __init__.py          - Package initialization
  model.py             - DeepONet and FNO architectures
  train.py             - Training loop with multi-resolution strategy
  data_loader.py       - Market data fetching (stocks + Bybit crypto)
  visualize.py         - Operator learning visualization
  backtest.py          - Trading strategy backtesting
  requirements.txt     - Dependencies
```

### Rust Implementation

```
rust_operator_learning/
  Cargo.toml           - Rust dependencies
  src/
    lib.rs             - Core operator learning library
    bin/
      train.rs         - Training binary
      predict.rs       - Prediction/inference binary
      fetch_data.rs    - Data fetching from Bybit
  examples/
    yield_curve.rs     - Yield curve evolution example
    vol_surface.rs     - Volatility surface operator example
```

## References

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G.E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

2. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.

3. Kissas, G., Yang, Y., Hwuang, E., Witschey, W.R., Doshi, J.A., & Perdikaris, P. (2022). Learning operators with coupled attention. *Journal of Machine Learning Research*, 23(215), 1-63.

4. Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Transactions on Neural Networks*, 6(4), 911-917.

5. Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhatt, K., Stuart, A., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research*, 24(89), 1-97.

6. Hesthaven, J.S., & Ubbiali, S. (2018). Non-intrusive reduced order modeling of nonlinear problems using neural networks. *Journal of Computational Physics*, 363, 55-78.

7. De Spiegeleer, J., Madan, D.B., Reyners, S., & Schoutens, W. (2018). Machine learning for quantitative finance: fast derivative pricing, hedging and fitting. *Quantitative Finance*, 18(10), 1635-1643.

8. Becker, S., Cheridito, P., & Jentzen, A. (2020). Pricing and hedging American-style options with deep learning. *Journal of Financial Economics*, 139(1), 68-83.

## Summary

Operator learning brings a fundamentally new capability to financial machine learning: the ability to learn mappings between function spaces rather than just between vectors. This enables:

- **Resolution-invariant models** that work at any data granularity
- **Zero-shot generalization** to new strikes, maturities, and market conditions
- **Real-time PDE solving** for pricing, hedging, and risk management
- **Transfer learning** across asset classes and market regimes
- **Physically consistent** predictions through PDE-informed training

For trading applications, operator learning is particularly powerful when combined with real-time market data (including crypto exchanges like Bybit), enabling sophisticated strategies that leverage the functional structure of financial data.

The key insight is that financial problems are naturally **operator problems** --- and operator learning provides the right mathematical framework to tackle them with neural networks.
