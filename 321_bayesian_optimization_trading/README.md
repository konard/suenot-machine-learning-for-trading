# Chapter 321: Bayesian Optimization for Trading Strategy Hyperparameters

## 1. Introduction

Every trading strategy comes with hyperparameters: the lookback window for a moving average, the threshold for a mean-reversion signal, the risk budget in a portfolio optimizer, the aggression parameter of an execution algorithm. Choosing these parameters well is the difference between a profitable strategy and one that bleeds money through transaction costs and mistimed entries.

Traditional approaches to hyperparameter tuning — grid search and random search — are wasteful. Grid search evaluates every point in a predefined lattice, scaling exponentially with the number of parameters. Random search is more efficient but still ignores what has already been learned. In the trading domain, each evaluation corresponds to a full backtest (or worse, a live paper-trading run), making every wasted evaluation expensive in both compute time and opportunity cost.

**Bayesian Optimization (BO)** offers a principled alternative. It builds a probabilistic surrogate model of the objective function (e.g., Sharpe ratio as a function of strategy parameters), uses this model to decide which point to evaluate next, updates the model with the new observation, and repeats. The key insight is that BO balances **exploration** (trying parameters in uncertain regions) with **exploitation** (refining parameters near known good regions), converging to the optimum in far fewer evaluations than grid or random search.

This chapter covers the mathematical foundations of Bayesian Optimization, its application to trading strategy tuning, portfolio optimization, and execution optimization, and provides a complete Rust implementation with Bybit market data integration.

## 2. Mathematical Foundations

### 2.1 The Optimization Problem

We seek to maximize an expensive black-box function:

```
x* = argmax f(x),  x ∈ X ⊂ R^d
```

where `f` is the strategy's performance metric (Sharpe ratio, total return, etc.), `x` is the vector of hyperparameters, and `X` is the bounded search space. We assume `f` has no known closed-form expression and each evaluation requires a full backtest.

### 2.2 Gaussian Process Surrogate Model

A **Gaussian Process (GP)** defines a distribution over functions. Given observations `D = {(x_1, y_1), ..., (x_n, y_n)}`, the GP posterior at a new point `x*` is:

```
μ(x*) = k(x*, X) [K + σ²I]⁻¹ y
σ²(x*) = k(x*, x*) - k(x*, X) [K + σ²I]⁻¹ k(X, x*)
```

where:
- `K` is the kernel (covariance) matrix with `K_ij = k(x_i, x_j)`
- `k` is the kernel function (we use the Radial Basis Function / Squared Exponential kernel)
- `σ²` is the observation noise variance
- `y` is the vector of observed values

The **RBF kernel** is defined as:

```
k(x, x') = σ_f² exp(-||x - x'||² / (2 l²))
```

where `σ_f²` is the signal variance and `l` is the length scale. The length scale controls how quickly the function varies: a small `l` means the function changes rapidly, while a large `l` implies smooth variation.

### 2.3 Acquisition Functions

The acquisition function `α(x)` determines which point to evaluate next by trading off exploration and exploitation.

#### Expected Improvement (EI)

EI measures the expected amount by which a new point will improve over the current best observation `f_best`:

```
EI(x) = E[max(f(x) - f_best, 0)]
       = (μ(x) - f_best) Φ(Z) + σ(x) φ(Z)
```

where:
```
Z = (μ(x) - f_best) / σ(x)
```

`Φ` is the standard normal CDF and `φ` is the standard normal PDF. When `σ(x) = 0`, `EI(x) = 0`.

EI naturally balances exploration (high `σ(x)`) and exploitation (high `μ(x)`). It is zero at already-observed points and positive everywhere else.

#### Upper Confidence Bound (UCB)

UCB constructs an optimistic estimate of the function value:

```
UCB(x) = μ(x) + κ σ(x)
```

where `κ > 0` is the exploration-exploitation trade-off parameter. Higher `κ` encourages more exploration. A common choice is `κ = 2.0`. The theoretical guarantee for GP-UCB sets `κ_t = √(2 log(t² d π² / 6δ))` where `t` is the iteration, `d` the dimension, and `δ` the failure probability.

#### Probability of Improvement (PI)

PI measures the probability that a new point improves over the current best:

```
PI(x) = P(f(x) > f_best) = Φ(Z)
```

PI is simpler than EI but tends to be too greedy (exploitative), often getting stuck in local optima. It is sometimes useful as a secondary criterion.

### 2.4 The Bayesian Optimization Loop

```
1. Initialize: Evaluate f at n_init random points
2. Repeat for t = 1, 2, ..., T:
   a. Fit GP to all observations D_t
   b. Find x_{t+1} = argmax α(x) over X
   c. Evaluate y_{t+1} = f(x_{t+1}) + ε
   d. Update D_{t+1} = D_t ∪ {(x_{t+1}, y_{t+1})}
3. Return x with highest observed f(x)
```

Step 2b (optimizing the acquisition function) is itself an optimization problem, but the acquisition function is cheap to evaluate, so we can use dense random sampling or gradient-based methods.

## 3. Applications in Trading

### 3.1 Strategy Parameter Tuning

The most direct application is tuning the hyperparameters of a trading strategy. Consider a simple moving average (MA) crossover strategy with parameters:
- **Fast MA window** (`w_fast`): 5 to 50 periods
- **Slow MA window** (`w_slow`): 20 to 200 periods
- **Stop-loss percentage** (`sl`): 0.5% to 5%

The objective is to maximize the Sharpe ratio over a historical backtest. BO can find near-optimal parameters in 20-30 evaluations, compared to hundreds for grid search.

More complex strategies may have 10-20 parameters (entry thresholds, exit thresholds, position sizing rules, regime filters). BO handles these higher-dimensional spaces effectively because the GP can model correlations between parameters and focus on promising regions.

### 3.2 Portfolio Optimization

BO can optimize portfolio construction parameters:
- **Risk aversion coefficient** in mean-variance optimization
- **Rebalancing frequency** and **turnover constraints**
- **Factor exposure limits** in constrained optimization
- **Shrinkage intensity** for covariance estimation

The objective might be a composite metric like the Calmar ratio (annualized return / maximum drawdown) that has no analytical gradient.

### 3.3 Execution Optimization

Execution algorithms (TWAP, VWAP, Implementation Shortfall) have parameters that affect fill quality:
- **Participation rate** and urgency
- **Slice size** and timing
- **Passive/aggressive ratio**

Each evaluation requires running the algorithm against historical order book data, making BO's sample efficiency particularly valuable.

### 3.4 Avoiding Overfitting

A critical concern when optimizing strategy parameters is **overfitting to the backtest**. Bayesian Optimization helps in several ways:

1. **Fewer evaluations** mean less opportunity for the optimizer to find spurious patterns
2. **The GP prior** acts as a regularizer, assuming the objective function is smooth
3. **Cross-validation** can be incorporated: use walk-forward validation as the objective
4. **Noise modeling** in the GP explicitly accounts for the stochastic nature of returns

Best practice is to optimize on a training set, validate on a holdout period, and only accept parameters that perform well on both.

## 4. Rust Implementation

Our implementation provides:

- **`GaussianProcess`**: A GP with RBF kernel, supporting fit and predict operations. The kernel matrix is inverted using Cholesky decomposition for numerical stability.
- **`BayesianOptimizer`**: The main optimization loop with configurable acquisition function (EI, UCB, or PI), search space bounds, and iteration budget.
- **`TradingBacktester`**: A simple MA crossover backtester that computes Sharpe ratio, total return, and maximum drawdown.
- **`BybitClient`**: Fetches historical OHLCV data from the Bybit public API.

Key design decisions:
- We use `ndarray` for linear algebra operations
- The GP uses a noise term `σ² = 1e-6` for numerical stability
- Acquisition function optimization uses random sampling (1000 candidates per iteration) for simplicity and robustness
- The backtester handles the constraint `w_fast < w_slow` by returning a penalty value

### Example usage:

```rust
use bayesian_optimization_trading::*;

// Define search space: [fast_window, slow_window]
let bounds = vec![(5.0, 50.0), (20.0, 200.0)];
let mut optimizer = BayesianOptimizer::new(bounds, AcquisitionFunction::ExpectedImprovement);

// Run optimization
let result = optimizer.optimize(|params| {
    let fast = params[0] as usize;
    let slow = params[1] as usize;
    backtester.run(fast, slow, &prices)
}, 30); // 30 iterations

println!("Best params: {:?}, Sharpe: {:.4}", result.best_params, result.best_value);
```

## 5. Bybit Data Integration

The implementation fetches historical kline (candlestick) data from Bybit's public REST API:

```
GET https://api.bybit.com/v5/market/kline
    ?category=linear
    &symbol=BTCUSDT
    &interval=60
    &limit=1000
```

The data is parsed into OHLCV candles and the close prices are extracted for backtesting. The API requires no authentication for public market data, making it straightforward to integrate.

We use `reqwest` with blocking mode for simplicity in the example, and `serde` for JSON deserialization. The Bybit API returns data in reverse chronological order, so we reverse it before use.

## 6. Key Takeaways

1. **Bayesian Optimization is sample-efficient**: It typically finds near-optimal parameters in 10-50 evaluations, compared to hundreds or thousands for grid/random search.

2. **The GP surrogate model provides uncertainty quantification**: Unlike point estimates, the GP gives both a predicted value and confidence interval at every point, enabling intelligent exploration.

3. **Acquisition functions formalize the exploration-exploitation trade-off**: EI is the most popular choice for its balance and analytical tractability. UCB offers a tunable exploration parameter. PI is simpler but greedier.

4. **Trading-specific considerations matter**: Walk-forward validation, transaction cost modeling, and regime awareness should be incorporated into the objective function to avoid overfitting.

5. **BO complements other optimization methods**: For low-dimensional problems (< 20 parameters), BO excels. For higher dimensions, consider combining BO with random embeddings (REMBO) or using tree-structured methods (TPE).

6. **The surrogate model itself is informative**: The fitted GP reveals which parameters matter most (via length scales) and how they interact, providing interpretability beyond just the optimal point.

7. **Rust provides the performance needed for real-time use**: The combination of fast backtesting and efficient GP inference makes it feasible to run BO in production pipelines where latency matters.

## References

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." NeurIPS.
- Shahriari, B., et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." Proceedings of the IEEE.
- Brochu, E., Cora, V. M., & de Freitas, N. (2010). "A Tutorial on Bayesian Optimization." arXiv:1012.2599.
- Rasmussen, C. E., & Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning." MIT Press.
- Frazier, P. I. (2018). "A Tutorial on Bayesian Optimization." arXiv:1807.02811.
