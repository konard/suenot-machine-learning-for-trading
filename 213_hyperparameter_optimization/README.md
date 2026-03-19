# Chapter 213: Hyperparameter Optimization for Trading Models

## 1. Introduction

Machine learning models used in algorithmic trading depend heavily on hyperparameters -- configuration values that are set before the learning process begins and cannot be learned from data directly. These include the learning rate, the number of hidden layers in a neural network, the lookback window for feature engineering, regularization strength, and many others. Choosing hyperparameters wisely can mean the difference between a strategy that generates consistent alpha and one that overfits to noise and bleeds capital in live trading.

Hyperparameter optimization (HPO) is the systematic process of searching for the best hyperparameter configuration. In a trading context, HPO carries unique challenges. The objective function is expensive to evaluate (it typically involves running a full backtest), the search space is often high-dimensional and mixed-type (continuous learning rates alongside discrete layer counts and categorical activation functions), and the risk of overfitting to historical data is severe. A configuration that maximizes Sharpe ratio on a backtest may simply be memorizing market regimes that will never repeat.

This chapter covers the core HPO algorithms -- grid search, random search, Bayesian optimization, Tree-structured Parzen Estimators (TPE), Hyperband, and BOHB -- with particular emphasis on their application to quantitative trading. We provide a full Rust implementation that integrates with Bybit exchange data, demonstrating how to tune a trading model's hyperparameters while guarding against overfitting.

## 2. Mathematical Foundation

### 2.1 Problem Formulation

Let $\lambda \in \Lambda$ denote a hyperparameter configuration drawn from a search space $\Lambda$. Let $f(\lambda)$ be an objective function that measures model performance (e.g., Sharpe ratio on a validation set). The goal of HPO is:

$$\lambda^* = \arg\max_{\lambda \in \Lambda} f(\lambda)$$

Because $f$ is typically a black-box function with no closed-form expression and expensive to evaluate, we need efficient search strategies.

### 2.2 Grid Search

Grid search is the simplest approach. For each hyperparameter, we define a finite set of candidate values. The algorithm evaluates every point on the Cartesian product of these sets.

Given $k$ hyperparameters with $n_1, n_2, \ldots, n_k$ candidate values respectively, grid search evaluates $\prod_{i=1}^{k} n_i$ configurations. This grows exponentially with the number of hyperparameters, making grid search impractical for high-dimensional spaces. However, it is exhaustive within its grid and easy to parallelize.

**Complexity:** $O(\prod_{i=1}^{k} n_i)$ evaluations.

### 2.3 Random Search

Random search samples configurations uniformly at random from the search space. Bergstra and Bengio (2012) demonstrated that random search is more efficient than grid search when only a few hyperparameters significantly affect performance. The key insight is that grid search wastes evaluations by exploring all combinations of unimportant hyperparameters, while random search projects more diverse values onto each axis.

For a budget of $T$ trials, random search samples $\lambda_1, \lambda_2, \ldots, \lambda_T \sim \text{Uniform}(\Lambda)$ and returns $\lambda^* = \arg\max_t f(\lambda_t)$.

**Theoretical guarantee:** If the top $\alpha$ fraction of the search space yields good performance, then with $T = \lceil \log(1 - p) / \log(1 - \alpha) \rceil$ trials, random search finds a configuration in that top fraction with probability at least $p$. For $\alpha = 5\%$ and $p = 95\%$, only $T = 59$ trials suffice.

### 2.4 Bayesian Optimization

Bayesian optimization is a sequential model-based approach that builds a probabilistic surrogate model of the objective function and uses it to decide where to evaluate next. The two key components are:

1. **Surrogate model:** A probabilistic model $\hat{f}(\lambda)$ that approximates $f(\lambda)$ and provides uncertainty estimates.
2. **Acquisition function:** A function $\alpha(\lambda)$ that uses the surrogate's predictions to balance exploration (trying uncertain regions) and exploitation (refining near known good configurations).

#### Gaussian Process Surrogate

A Gaussian Process (GP) defines a distribution over functions. Given observed data $D = \{(\lambda_i, y_i)\}_{i=1}^{n}$, the GP posterior at a new point $\lambda$ is:

$$\mu(\lambda) = k(\lambda, X) [K(X, X) + \sigma_n^2 I]^{-1} y$$

$$\sigma^2(\lambda) = k(\lambda, \lambda) - k(\lambda, X) [K(X, X) + \sigma_n^2 I]^{-1} k(X, \lambda)$$

where $k(\cdot, \cdot)$ is the kernel function. We use the Radial Basis Function (RBF) kernel:

$$k(\lambda_i, \lambda_j) = \sigma_f^2 \exp\left(-\frac{\|\lambda_i - \lambda_j\|^2}{2 \ell^2}\right)$$

where $\ell$ is the length scale and $\sigma_f^2$ is the signal variance.

#### Acquisition Functions

**Expected Improvement (EI):** Measures the expected amount by which a new point will improve over the current best $f^* = \max_i y_i$:

$$\text{EI}(\lambda) = \mathbb{E}[\max(0, f(\lambda) - f^*)] = (mu - f^*)\Phi(Z) + \sigma \phi(Z)$$

where $Z = (\mu - f^*) / \sigma$, and $\Phi$, $\phi$ are the standard normal CDF and PDF.

**Upper Confidence Bound (UCB):** Balances mean and uncertainty with a parameter $\kappa$:

$$\text{UCB}(\lambda) = \mu(\lambda) + \kappa \cdot \sigma(\lambda)$$

Higher $\kappa$ encourages more exploration. A common choice is $\kappa = 2.0$.

**Probability of Improvement (PI):** The probability that a new point exceeds the current best:

$$\text{PI}(\lambda) = \Phi\left(\frac{\mu(\lambda) - f^*}{\sigma(\lambda)}\right)$$

EI is the most commonly used acquisition function because it naturally balances exploration and exploitation without requiring a tunable trade-off parameter.

### 2.5 Tree-structured Parzen Estimators (TPE)

TPE, introduced by Bergstra et al. (2011), takes a different approach from GP-based Bayesian optimization. Instead of modeling $p(y | \lambda)$, TPE models:

$$p(\lambda | y) = \begin{cases} \ell(\lambda) & \text{if } y < y^* \\ g(\lambda) & \text{if } y \geq y^* \end{cases}$$

where $y^*$ is a quantile threshold (typically the top 15% of observations). The ratio $\ell(\lambda) / g(\lambda)$ is proportional to the Expected Improvement, so TPE selects configurations that are more likely under $\ell$ (the "good" distribution) than under $g$ (the "bad" distribution).

TPE handles conditional and categorical hyperparameters naturally through its tree-structured decomposition, making it well-suited for complex search spaces common in trading model pipelines.

### 2.6 Hyperband

Hyperband, proposed by Li et al. (2018), is an early-stopping-based approach that frames HPO as a resource allocation problem. The core idea is Successive Halving (SHA):

1. Start with $n$ random configurations, each allocated a small budget $b$.
2. Evaluate all configurations with budget $b$.
3. Keep the top $1/\eta$ fraction (typically $\eta = 3$).
4. Increase the budget by a factor of $\eta$ and repeat.

Hyperband runs SHA with different starting values of $n$ and $b$ to balance the exploration-exploitation trade-off between many configurations with small budgets versus fewer configurations with larger budgets.

The total budget for one bracket of SHA with $n$ initial configurations and maximum budget $R$ is approximately $n \cdot R / \eta^{s}$ per round, where $s$ is the number of halving rounds.

### 2.7 BOHB (Bayesian Optimization and Hyperband)

BOHB combines Bayesian optimization with Hyperband. Instead of sampling configurations uniformly at random as Hyperband does, BOHB uses a TPE model to sample promising configurations. This combines the sample efficiency of Bayesian optimization with the early-stopping efficiency of Hyperband, making it one of the most effective general-purpose HPO algorithms available.

## 3. Search Spaces

Defining the search space correctly is critical for HPO efficiency. Hyperparameters come in several types:

### 3.1 Continuous Hyperparameters

These take real values within a range. Examples include learning rate ($[10^{-5}, 10^{-1}]$, typically searched on a log scale), dropout rate ($[0.0, 0.5]$), and L2 regularization weight ($[10^{-6}, 10^{-2}]$).

Log-uniform sampling is essential for parameters that span several orders of magnitude. Searching learning rate linearly in $[0.00001, 0.1]$ would waste most samples near the upper end.

### 3.2 Discrete Hyperparameters

Integer-valued parameters such as the number of hidden units ($\{32, 64, 128, 256, 512\}$), batch size ($\{16, 32, 64, 128\}$), and lookback window ($\{5, 10, 20, 50, 100\}$ days).

### 3.3 Categorical Hyperparameters

Unordered choices such as activation function ($\{\text{ReLU}, \text{GELU}, \text{Tanh}\}$), optimizer type ($\{\text{Adam}, \text{SGD}, \text{AdamW}\}$), or feature set ($\{\text{OHLCV}, \text{OHLCV+indicators}, \text{orderbook}\}$).

### 3.4 Conditional Hyperparameters

Some hyperparameters only matter when another takes a specific value. For example, momentum is only relevant when the optimizer is SGD, or the number of attention heads only matters when the model architecture is a Transformer. Properly encoding these conditional relationships prevents wasting evaluations on irrelevant configurations.

## 4. Trading-Specific HPO

### 4.1 Walk-Forward Validation as Objective

In trading, standard k-fold cross-validation violates temporal ordering and causes data leakage. Instead, walk-forward validation respects the time series structure:

1. Train on $[t_0, t_1)$, validate on $[t_1, t_2)$.
2. Train on $[t_0, t_2)$, validate on $[t_2, t_3)$.
3. Continue sliding forward.

The HPO objective is the average performance across all forward-validation windows. This provides a more realistic estimate of out-of-sample performance than a single train-test split.

### 4.2 Avoiding Overfitting to Backtest

HPO introduces a meta-level overfitting risk: even if each trial uses proper walk-forward validation, searching over thousands of configurations can find one that happens to perform well on the particular historical period by chance. Mitigation strategies include:

- **Limiting the number of trials:** Use efficient methods (Bayesian optimization, Hyperband) to find good configurations in fewer evaluations.
- **Multiple time periods:** Validate the best configuration on a completely held-out period not used during HPO.
- **Regularization of the search space:** Constrain the space to reasonable ranges based on domain knowledge.
- **Deflating the Sharpe ratio:** Apply the Bailey-Lopez de Prado deflation formula to account for multiple testing.
- **Combinatorial purged cross-validation:** Use CPCV to generate more robust out-of-sample estimates.

### 4.3 Multi-Objective Optimization

Trading objectives are inherently multi-dimensional. A strategy with Sharpe ratio 2.0 but maximum drawdown of 40% may be less desirable than one with Sharpe 1.5 and drawdown of 15%. Multi-objective HPO seeks the Pareto frontier of non-dominated configurations.

A common approach is to define a scalarized objective:

$$f(\lambda) = w_1 \cdot \text{Sharpe}(\lambda) - w_2 \cdot \text{MaxDrawdown}(\lambda) + w_3 \cdot \text{ProfitFactor}(\lambda)$$

Alternatively, Pareto-based methods maintain a set of non-dominated solutions, allowing the trader to inspect trade-offs and choose based on their risk preferences.

## 5. Early Stopping and Efficient Resource Allocation

### 5.1 Successive Halving

Successive Halving is the foundation of efficient HPO. The algorithm works as follows:

```
Input: n configurations, maximum budget R, halving factor eta
s_max = floor(log_eta(n))
For each round i = 0, 1, ..., s_max:
    n_i = floor(n / eta^i)        # number of surviving configs
    r_i = R * eta^(i - s_max)     # budget per config
    Evaluate all n_i configs with budget r_i
    Keep top 1/eta fraction
Return best configuration
```

In a trading context, the "budget" could be the number of training epochs, the length of the backtest period, or the number of walk-forward folds. Low-budget evaluations (e.g., training for fewer epochs or on shorter history) provide noisy but informative estimates that allow us to discard bad configurations early.

### 5.2 Hyperband Brackets

Hyperband hedges against the uncertainty in the budget-performance relationship by running multiple SHA brackets:

- Bracket $s = s_{\max}$: Many configurations, each with minimal initial budget.
- Bracket $s = 0$: Few configurations, each with the full budget from the start.

This ensures robustness regardless of whether the learning curve is predictive at low budgets.

## 6. Implementation Walkthrough (Rust)

Our Rust implementation provides a complete HPO framework in `rust/src/lib.rs`:

**Hyperparameter Space Definition:** The `SearchSpace` struct holds a collection of `HyperparameterDef` entries, each specifying a name, type (continuous, discrete, or categorical), and valid range. Continuous parameters support log-scale sampling.

**Grid Search:** Generates the full Cartesian product of discretized parameter values and evaluates each. Simple but exhaustive.

**Random Search:** Samples configurations uniformly from the search space. Each continuous parameter is sampled uniformly (or log-uniformly) within its range; discrete parameters are sampled uniformly from their set; categorical parameters are chosen with equal probability.

**Bayesian Optimization:** Maintains a GP surrogate with an RBF kernel. After an initial set of random evaluations (typically 5-10), it fits the GP to observed data and maximizes the Expected Improvement acquisition function to select the next configuration. The GP posterior mean and variance are computed via the standard Cholesky-based approach for numerical stability.

**Successive Halving / Hyperband:** Implements the early-stopping resource allocation scheme. Configurations are evaluated at increasing budget levels, with the bottom fraction eliminated at each round.

**Trial History:** All evaluated configurations and their scores are stored in a `TrialHistory` struct, allowing inspection of the optimization trajectory and selection of the best configuration.

## 7. Bybit Data Integration

The implementation includes a Bybit API client that fetches historical OHLCV (Open, High, Low, Close, Volume) data for any trading pair. The `BybitClient` struct in our library provides:

- `fetch_klines(symbol, interval, limit)`: Retrieves candlestick data from the Bybit v5 API.
- Automatic parsing of the JSON response into typed `Candle` structs.
- Support for multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d).

This data feeds into the walk-forward validation objective. For each hyperparameter configuration, the system:

1. Splits the historical candles into walk-forward folds.
2. Computes features (returns, moving averages, volatility) for each fold.
3. Simulates a simple momentum/mean-reversion strategy parameterized by the hyperparameters.
4. Calculates the Sharpe ratio and maximum drawdown on the validation portion.
5. Returns a scalarized multi-objective score.

The example in `rust/examples/trading_example.rs` demonstrates the full pipeline: fetching BTCUSDT daily data from Bybit, defining a search space for lookback window, momentum threshold, position size, and stop-loss, then running grid search, random search, and Bayesian optimization to compare their efficiency.

## 8. Key Takeaways

1. **Grid search is simple but scales exponentially.** Use it only for low-dimensional spaces (2-3 hyperparameters) or as a final refinement around a known good region.

2. **Random search is surprisingly effective.** For most problems, random search with 50-100 trials outperforms grid search with the same budget, especially when some hyperparameters matter more than others.

3. **Bayesian optimization is the gold standard for expensive objectives.** When each evaluation (backtest) takes minutes to hours, the overhead of fitting a GP surrogate is negligible compared to the savings from fewer evaluations.

4. **Hyperband and BOHB offer the best of both worlds.** Early stopping eliminates bad configurations quickly, while Bayesian sampling focuses on promising regions.

5. **Walk-forward validation is non-negotiable in trading.** Standard cross-validation causes temporal data leakage and produces overly optimistic performance estimates.

6. **Guard against meta-overfitting.** The more configurations you try, the higher the probability of finding one that looks good by chance. Always validate on a truly held-out period.

7. **Multi-objective HPO captures real trading requirements.** Optimizing Sharpe ratio alone ignores drawdown risk, tail behavior, and other practically important characteristics.

8. **Define the search space carefully.** Use log-scale for parameters spanning orders of magnitude, encode conditional dependencies, and constrain ranges based on domain knowledge to make the search more efficient.

9. **Start simple, then scale.** Begin with random search to understand the landscape, then apply Bayesian optimization or BOHB to refine. This two-phase approach is both practical and robust.

10. **Rust provides the performance needed for large-scale HPO.** The combination of low-level control, zero-cost abstractions, and memory safety makes Rust an excellent choice for implementing HPO loops that may evaluate thousands of configurations.
