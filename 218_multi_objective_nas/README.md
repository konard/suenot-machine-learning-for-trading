# Chapter 218: Multi-Objective Neural Architecture Search (NAS)

## 1. Introduction

Neural Architecture Search (NAS) has revolutionized the way we design deep learning models by automating the traditionally manual process of architecture engineering. In most early NAS formulations, the search was driven by a single objective — typically prediction accuracy on a validation set. However, real-world deployment of machine learning models in trading systems demands simultaneous optimization across multiple, often competing, objectives.

Consider the challenges a quantitative trading firm faces when deploying a model. The model must be accurate enough to generate profitable signals, fast enough to execute within tight latency constraints, small enough to fit within memory budgets on edge devices or co-located servers, robust enough to withstand regime changes, and interpretable enough for risk managers to understand. Optimizing for any single one of these objectives while ignoring the others leads to impractical solutions: a highly accurate model that takes seconds to produce a prediction is useless for high-frequency trading, while a tiny fast model that produces random signals is equally worthless.

Multi-objective NAS (MO-NAS) addresses this fundamental tension by searching for architectures that represent the best possible trade-offs among all objectives simultaneously. Rather than returning a single "best" architecture, MO-NAS produces a set of Pareto-optimal architectures — a frontier of solutions where no architecture can improve on one objective without sacrificing another. This gives practitioners the power to choose the architecture that best matches their specific deployment constraints.

In this chapter, we develop a complete multi-objective NAS system in Rust, designed specifically for trading applications. We integrate live market data from the Bybit exchange API and demonstrate how to find architectures that balance prediction accuracy, inference speed, and model compactness.

## 2. Mathematical Foundation

### 2.1 Pareto Optimality and Dominance

The foundation of multi-objective optimization rests on the concept of Pareto dominance. Given two solution vectors **x** and **y** in the objective space, we say that **x** dominates **y** (written **x** ≺ **y**) if and only if:

1. **x** is no worse than **y** in all objectives: f_i(**x**) ≤ f_i(**y**) for all i ∈ {1, ..., M}
2. **x** is strictly better than **y** in at least one objective: ∃ j such that f_j(**x**) < f_j(**y**)

Here we assume minimization; for objectives like accuracy or Sharpe ratio that should be maximized, we negate them.

A solution **x*** is called **Pareto optimal** if no other feasible solution dominates it. The set of all Pareto-optimal solutions is the **Pareto set**, and its image in objective space is the **Pareto front**. The Pareto front represents the best achievable trade-offs — moving along the front always involves sacrificing performance in one objective to gain in another.

### 2.2 NSGA-II Algorithm

The Non-dominated Sorting Genetic Algorithm II (NSGA-II), proposed by Deb et al. (2002), remains one of the most widely used multi-objective evolutionary algorithms. Its key innovations are:

**Non-dominated Sorting**: The population is partitioned into successive fronts. The first front F₁ contains all non-dominated solutions. The second front F₂ contains solutions dominated only by members of F₁, and so on. This creates a ranking where lower front indices indicate better solutions.

The algorithm proceeds as follows:
1. Initialize population P₀ of size N
2. For each generation t:
   a. Create offspring population Q_t using selection, crossover, and mutation
   b. Combine R_t = P_t ∪ Q_t
   c. Perform non-dominated sorting on R_t
   d. Fill next population P_{t+1} by including complete fronts in order
   e. If a front would exceed N, use crowding distance to select the best members

**Crowding Distance**: When a front must be partially included, crowding distance determines which solutions to keep. For each objective, solutions are sorted and boundary solutions receive infinite distance. Interior solutions receive distance proportional to the gap between their neighbors:

```
d_i = Σ_m (f_m(i+1) - f_m(i-1)) / (f_m_max - f_m_min)
```

This preserves diversity by preferring solutions in less crowded regions of the front.

### 2.3 NSGA-III and Reference-Point Based Selection

NSGA-III extends NSGA-II for many-objective problems (typically more than 3 objectives) by replacing crowding distance with reference-point based selection. A set of well-distributed reference points is placed on a normalized hyperplane, and solutions are associated with their nearest reference point. Selection then prefers solutions associated with reference points that have the fewest associated members, ensuring uniform coverage of the Pareto front.

### 2.4 Hypervolume Indicator

The hypervolume indicator (HV) measures the volume of objective space dominated by the Pareto front approximation, bounded by a reference point. It is the only unary quality indicator that is strictly monotone with respect to Pareto dominance — if front A dominates front B, then HV(A) > HV(B).

For a set of points S and reference point **r**:

```
HV(S, r) = volume( ∪_{s ∈ S} [s₁, r₁] × [s₂, r₂] × ... × [s_M, r_M] )
```

In two dimensions, this reduces to a simple area calculation. In higher dimensions, exact computation is exponential in the number of objectives, but efficient algorithms exist for low dimensions.

## 3. Objectives for Trading

Trading applications present a rich set of objectives that frequently conflict with one another:

### 3.1 Prediction Accuracy
The fundamental goal — how well the model predicts future price movements, returns, or signals. Measured via MSE, MAE, directional accuracy, or classification metrics.

### 3.2 Inference Latency
In latency-sensitive trading, every microsecond counts. Larger, more complex architectures tend to be more accurate but slower. For HFT applications, models must produce predictions within strict time budgets (often sub-millisecond).

### 3.3 Model Size
Memory footprint matters for deployment on co-located servers, FPGAs, or embedded systems. Smaller models also tend to generalize better due to implicit regularization.

### 3.4 Robustness
Markets undergo regime changes — from trending to mean-reverting, from low to high volatility. A robust model maintains acceptable performance across different market conditions. This can be measured as worst-case performance across validation sets drawn from different regimes.

### 3.5 Interpretability
Risk managers and regulators may require explanations for model decisions. Simpler, more interpretable architectures (e.g., shallow networks with attention mechanisms) are preferred over opaque deep models when transparency is needed.

### 3.6 Sharpe Ratio
Rather than raw prediction accuracy, traders often care about risk-adjusted returns. The Sharpe ratio SR = (E[R] - R_f) / σ(R) directly measures the quality of a trading strategy derived from model predictions.

### 3.7 Maximum Drawdown
The largest peak-to-trough decline in portfolio value. A model that produces high Sharpe ratio but occasional catastrophic drawdowns may be unacceptable for risk-constrained portfolios.

## 4. Trading Applications

### 4.1 Balancing Accuracy and Execution Speed

The most common trade-off in trading NAS is between prediction quality and latency. Consider a market-making strategy where the model must predict short-term price movements to set bid-ask quotes. A transformer-based model with 12 attention layers might achieve 62% directional accuracy but require 5ms per prediction. A simple two-layer MLP might achieve only 55% accuracy but respond in 50 microseconds.

Multi-objective NAS explores the space between these extremes, finding architectures like a 3-layer network with a single attention head that achieves 59% accuracy in 200 microseconds — a trade-off that might be optimal for a specific market-making strategy with 1ms latency requirements.

### 4.2 Robustness Across Market Regimes

A single-objective NAS that optimizes for accuracy on recent data will overfit to the current market regime. By including robustness as an explicit objective — measured as the minimum accuracy across validation sets from different regimes (bull, bear, sideways, high-volatility, low-volatility) — MO-NAS finds architectures that are inherently more stable.

The Pareto front in this case reveals the cost of robustness: how much accuracy on the current regime must be sacrificed to maintain performance during regime changes. This information is invaluable for portfolio construction and risk management.

### 4.3 Multi-Strategy Deployment

A trading desk running multiple strategies across different time horizons and asset classes can use MO-NAS to find a portfolio of architectures. Short-horizon strategies demand low latency; longer-horizon strategies can afford larger models. The Pareto front from MO-NAS provides a menu of architectures matched to each strategy's constraints.

## 5. Scalarization vs. True Multi-Objective Approaches

### 5.1 Weighted Sum (Scalarization)

The simplest approach converts multiple objectives into a single scalar:

```
F(x) = Σ_i w_i * f_i(x)
```

where w_i are user-specified weights. This is easy to implement but has fundamental limitations:

1. **Non-convex Pareto fronts**: Scalarization can only find solutions on the convex hull of the Pareto front, potentially missing important trade-off regions.
2. **Weight sensitivity**: Small changes in weights can lead to dramatically different solutions, and the mapping from weights to solutions is non-intuitive.
3. **Single solution**: Each weight vector produces only one solution, requiring multiple runs with different weights to approximate the front.

### 5.2 Pareto-Based Approaches

True multi-objective methods like NSGA-II maintain a population of diverse solutions and evolve the entire Pareto front simultaneously. Advantages include:

1. **Complete front**: A single run produces many trade-off solutions.
2. **No weight specification**: No need to specify relative importance a priori.
3. **Non-convex fronts**: Can discover solutions in non-convex regions.
4. **Decision flexibility**: Practitioners inspect the full front and choose based on deployment constraints.

The primary disadvantage is computational cost — maintaining and evolving a population is more expensive than optimizing a single scalar. However, for NAS where architecture evaluation is the bottleneck, the overhead of multi-objective selection is negligible.

### 5.3 Hybrid Approaches

In practice, hybrid approaches work well. An initial MO-NAS run identifies the Pareto front, followed by a focused single-objective search in the most promising region. Alternatively, scalarization with adaptive weight adjustment can approximate the Pareto front more efficiently than pure multi-objective search in high-dimensional objective spaces.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete NSGA-II based multi-objective NAS system. The key components are:

### 6.1 Architecture Representation

Each candidate architecture is represented as a vector of categorical and continuous parameters: number of layers, hidden dimensions, activation functions, attention heads, dropout rates, and so on. The `NasArchitecture` struct encodes this search space.

### 6.2 Multi-Objective Evaluation

Each architecture is evaluated on all objectives simultaneously. The `ObjectiveValues` type is a vector of floats, one per objective. Lower values are better (objectives to maximize are negated).

### 6.3 Pareto Dominance and Sorting

The `dominates` function checks Pareto dominance between two objective vectors. The `non_dominated_sort` function partitions the population into fronts using the fast non-dominated sorting algorithm from NSGA-II (O(MN²) complexity where M is the number of objectives and N is the population size).

### 6.4 Crowding Distance

The `crowding_distance` function computes crowding distances for solutions within a front. Boundary solutions receive f64::INFINITY to ensure they are always preserved.

### 6.5 Evolutionary Operators

Tournament selection uses dominance rank and crowding distance. Crossover creates offspring by combining genes from two parents. Mutation randomly perturbs architecture parameters. These operators are standard but specialized for the architecture encoding.

### 6.6 NSGA-II Main Loop

The `multi_objective_search` function ties everything together: initialize, evaluate, sort, select, breed, repeat. After the specified number of generations, the first Pareto front is returned.

See `rust/src/lib.rs` for the complete implementation and `rust/examples/trading_example.rs` for a working example that fetches Bybit data and runs the search.

## 7. Bybit Data Integration

Our implementation fetches real-time candlestick (kline) data from the Bybit V5 API. The endpoint `https://api.bybit.com/v5/market/kline` provides OHLCV data for any supported trading pair and timeframe.

The data pipeline:
1. **Fetch**: HTTP GET request with parameters for symbol, interval, and limit
2. **Parse**: Deserialize the JSON response into structured candlestick data
3. **Feature Engineering**: Compute returns, volatility, and technical indicators from raw OHLCV data
4. **Objective Evaluation**: Use the processed features to simulate architecture performance on each objective

This integration ensures our NAS experiments use realistic market data, capturing the statistical properties (fat tails, volatility clustering, non-stationarity) that make trading ML uniquely challenging.

The Bybit API is free and requires no authentication for public market data endpoints, making it ideal for research and experimentation.

## 8. Key Takeaways

1. **Multi-objective NAS produces Pareto fronts, not single solutions.** This gives practitioners a menu of architectures representing the best achievable trade-offs among competing objectives.

2. **Trading has naturally competing objectives.** Accuracy vs. latency, Sharpe ratio vs. drawdown, model complexity vs. robustness — these tensions are inherent to the domain and demand multi-objective treatment.

3. **NSGA-II is the workhorse algorithm.** Its combination of non-dominated sorting and crowding distance provides a robust foundation for multi-objective architecture search. For problems with more than 3 objectives, NSGA-III with reference points is preferred.

4. **Pareto-based methods outperform scalarization.** True multi-objective approaches find complete Pareto fronts including non-convex regions, without requiring weight specification. The computational overhead is minimal compared to architecture evaluation cost.

5. **Hypervolume is the gold standard metric.** The hypervolume indicator is the only unary quality indicator that is strictly Pareto-compliant, making it ideal for comparing different NAS runs or algorithm configurations.

6. **Robustness should be an explicit objective.** Including worst-case or average performance across market regimes as an objective ensures the NAS does not produce architectures that overfit to the current market conditions.

7. **Rust enables production-grade performance.** The combination of zero-cost abstractions, memory safety, and fearless concurrency makes Rust an excellent choice for implementing compute-intensive NAS algorithms that must eventually run in production trading systems.

8. **The Pareto front guides deployment decisions.** Rather than making accuracy-latency trade-offs implicitly during model development, MO-NAS makes these trade-offs explicit and quantified, enabling more informed deployment decisions.
