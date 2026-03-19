# Chapter 319: Population Based Training for Trading

## 1. Introduction

Population Based Training (PBT) is a powerful hyperparameter optimization technique introduced by DeepMind that combines the advantages of random search and hand-tuning with the efficiency of evolutionary methods. Unlike traditional hyperparameter optimization approaches such as grid search or Bayesian optimization that treat training runs as independent black-box evaluations, PBT jointly optimizes a population of models and their hyperparameters in a single training process.

In the context of algorithmic trading, PBT is particularly valuable because financial markets are non-stationary environments where the optimal set of hyperparameters can shift over time. A model that performs well during a trending market may require fundamentally different hyperparameters during a mean-reverting regime. PBT addresses this by maintaining a diverse population of trading agents, each with different hyperparameters, and allowing them to adapt throughout training by sharing information and exploring new configurations.

Traditional hyperparameter tuning for trading systems involves expensive sequential or parallel searches over a predefined grid. A typical trading model might have dozens of hyperparameters: learning rate, batch size, lookback window, risk parameters, feature selection thresholds, and more. The combinatorial explosion makes exhaustive search impractical. PBT sidesteps this by leveraging the population itself as an implicit search mechanism, where poorly performing agents inherit the weights and hyperparameters of better-performing agents while also exploring perturbations that might lead to even better configurations.

The key insight of PBT is that hyperparameters are not static quantities to be determined before training. Instead, they are dynamic schedules that should evolve alongside the model's parameters. This is especially relevant for trading, where market conditions change and the optimal learning rate or regularization strength at the beginning of training may differ substantially from what is optimal later.

## 2. Mathematical Foundations

### 2.1 Population Formulation

Consider a population of N agents {(theta_i, h_i)} for i=1..N, where theta_i represents the model parameters (weights) and h_i represents the hyperparameters of agent i. Each agent has an associated performance metric p_i = f(theta_i, h_i), which in trading could be the Sharpe ratio, total return, or any risk-adjusted performance measure.

### 2.2 The Exploit Step

The exploit step determines whether an agent should copy the parameters of a better-performing agent. For agent i, we define the exploit operation as:

```
exploit(theta_i, h_i, p_i) =
    (theta_j, h_j)   if p_j > p_i * (1 + epsilon)
    (theta_i, h_i)    otherwise
```

where j = argmax_k(p_k) is the best-performing agent in the population, and epsilon is a tolerance threshold that prevents unnecessary copying when performance differences are marginal.

Common exploit strategies include:

- **Truncation selection**: Copy weights from a randomly sampled agent in the top 20% of the population.
- **Binary tournament**: Randomly sample another agent; if it performs better, copy its weights.
- **Proportional selection**: Copy weights from agent j with probability proportional to p_j.

### 2.3 The Explore Step

After exploitation, the explore step perturbs the hyperparameters to introduce diversity:

```
h_i' = h_i * (1 + delta),    delta ~ Uniform(-alpha, alpha)
```

where alpha controls the perturbation magnitude. For discrete hyperparameters (e.g., batch size), we resample from a predefined set:

```
h_i_discrete ~ Categorical({v_1, v_2, ..., v_m})
```

with probability p_resample and keep the current value otherwise.

### 2.4 Asynchronous Evolution

Unlike synchronous evolutionary methods, PBT operates asynchronously. Each agent trains independently, and the exploit/explore steps are triggered based on a schedule (e.g., every T training steps). This can be formalized as:

```
At step t: if t mod T = 0 then (theta_i, h_i) <- explore(exploit(theta_i, h_i, p_i))
```

The asynchronous nature means agents do not need to wait for each other, which is crucial for practical scalability. In a trading context, this means different agents can process different amounts of market data before being evaluated, allowing faster agents to evolve more quickly.

### 2.5 Convergence Properties

PBT converges to a region of hyperparameter space that produces good performance because:

1. **Exploitation pressure**: Low-performing configurations are replaced by high-performing ones, concentrating the population around good solutions.
2. **Exploration diversity**: Perturbation ensures the population does not collapse to a single point, maintaining the ability to discover better configurations.
3. **Schedule discovery**: Because hyperparameters can change over time, PBT implicitly discovers hyperparameter schedules rather than fixed values.

The balance between exploitation and exploration is governed by the population size N, the perturbation magnitude alpha, and the evaluation frequency T.

## 3. Applications in Trading

### 3.1 Trading Strategy Ensembles

PBT naturally produces an ensemble of trading strategies at convergence. Rather than selecting a single best model, the entire population can be used as an ensemble:

```
y_hat_ensemble = (1/N) * sum(w_i * y_hat_i)
```

where w_i can be uniform or proportional to each agent's performance. This ensemble approach provides several benefits for trading:

- **Diversification**: Different agents may specialize in different market regimes.
- **Robustness**: The ensemble is less sensitive to overfitting by any single agent.
- **Adaptive allocation**: Weights can be dynamically adjusted based on recent performance.

### 3.2 Reinforcement Learning Agent Populations

In RL-based trading, PBT is used to simultaneously optimize the RL algorithm's hyperparameters (learning rate, discount factor gamma, entropy coefficient, etc.) alongside the policy parameters. A population of RL agents trades in parallel, with each agent's Sharpe ratio or PnL serving as the fitness metric.

Key hyperparameters for RL trading agents include:

| Hyperparameter | Typical Range | Perturbation Strategy |
|---|---|---|
| Learning rate | 1e-5 to 1e-2 | Log-scale multiply |
| Discount factor gamma | 0.9 to 0.999 | Linear perturbation |
| Batch size | 32, 64, 128, 256 | Categorical resample |
| Entropy coefficient | 0.001 to 0.1 | Log-scale multiply |
| Lookback window | 10, 20, 50, 100 | Categorical resample |
| Risk penalty lambda | 0.01 to 1.0 | Linear perturbation |

### 3.3 Non-Stationary Market Adaptation

Markets exhibit regime changes. PBT can be run continuously, allowing the population to adapt to changing market conditions. When a regime shift occurs, previously good hyperparameters may become suboptimal. The exploration mechanism ensures that new configurations are always being tested, while exploitation ensures that agents quickly converge to configurations that work well under the new regime.

## 4. DeepMind PBT Algorithm Details

The original PBT algorithm by Jaderberg et al. (2017) works as follows:

### Algorithm: Population Based Training

```
Initialize population P = {(theta_i, h_i)} for i=1..N randomly
For each agent i in parallel:
    While not converged:
        1. Train: theta_i <- train(theta_i, h_i, data)    // standard gradient update
        2. Evaluate: p_i <- evaluate(theta_i, validation_data)
        3. If ready(i, t):                                  // time-based trigger
            a. EXPLOIT:
               j <- select_parent(P, p)                    // pick a better agent
               if p_j > p_i:
                   theta_i <- copy(theta_j)                // copy weights
                   h_i <- copy(h_j)                        // copy hyperparams
            b. EXPLORE:
               h_i <- perturb(h_i)                         // mutate hyperparams
    Return best agent from P
```

### Key Design Decisions

1. **Population size**: Typically 10-50 agents. Larger populations provide more diversity but require more compute. For trading, 10-20 agents is usually sufficient.

2. **Evaluation frequency**: How often to perform exploit/explore. Too frequent leads to instability; too infrequent wastes compute on bad configurations. A common choice is every epoch or every fixed number of gradient updates.

3. **Exploit strategy**: Truncation selection (copy from top 20%) is the most common choice due to its simplicity and effectiveness.

4. **Perturbation factors**: Continuous hyperparameters are typically multiplied by 0.8 or 1.2 (20% perturbation). This provides sufficient exploration without destroying good configurations.

5. **Warm-up period**: Agents should train for a minimum number of steps before any exploit/explore to ensure meaningful performance differences.

### Advantages over Alternatives

| Method | Parallelism | Schedule Discovery | Compute Efficiency |
|---|---|---|---|
| Grid Search | Yes | No | Low |
| Random Search | Yes | No | Medium |
| Bayesian Optimization | Limited | No | High (per trial) |
| PBT | Yes | Yes | High (overall) |

## 5. Rust Implementation

Our Rust implementation provides a high-performance PBT framework for trading applications. The core components include:

- **`Agent`**: Represents a single member of the population with model weights and hyperparameters.
- **`Population`**: Manages the collection of agents and orchestrates the PBT process.
- **`TradingEnvironment`**: Simulates a trading environment using price data.
- **`BybitClient`**: Fetches real market data from the Bybit exchange API.

The implementation leverages Rust's performance characteristics for efficient parallel evaluation of agents and uses `ndarray` for numerical operations. Key design choices include:

- Agents use simple linear models for price prediction, keeping the focus on the PBT mechanism itself.
- The exploit step uses truncation selection (top 20%).
- The explore step perturbs continuous hyperparameters by a configurable factor and resamples discrete ones.
- Performance is measured by a simplified Sharpe ratio computed from agent returns.

See `rust/src/lib.rs` for the full implementation and `rust/examples/trading_example.rs` for a complete example using Bybit market data.

## 6. Bybit Data Integration

The implementation integrates with the Bybit public API to fetch real cryptocurrency market data. We use the `/v5/market/kline` endpoint to retrieve OHLCV (Open, High, Low, Close, Volume) candlestick data for any trading pair.

### API Integration

The `BybitClient` struct provides methods to fetch kline data with configurable:

- **Symbol**: Trading pair (e.g., BTCUSDT, ETHUSDT)
- **Interval**: Candlestick interval (e.g., 1m, 5m, 1h, 1d)
- **Limit**: Number of candles to retrieve

The fetched data is converted into a format suitable for the trading environment, where close prices serve as the primary signal for the simple price prediction models used by PBT agents.

### Data Pipeline

```
Bybit API -> JSON Response -> Kline structs -> Price vectors -> TradingEnvironment -> Agent evaluation
```

This pipeline allows PBT to optimize trading agents on real market data, providing realistic performance evaluation for the exploit/explore decisions.

## 7. Key Takeaways

1. **PBT jointly optimizes model parameters and hyperparameters** by maintaining a population of agents that share information through exploitation and explore new configurations through perturbation. This is fundamentally different from treating hyperparameter optimization as an outer loop around training.

2. **Asynchronous evolution is practical and scalable**. Agents do not need synchronized evaluation, making PBT suitable for distributed training setups common in quantitative finance.

3. **PBT discovers hyperparameter schedules, not just fixed values**. This is crucial for trading where market regimes change and optimal hyperparameters shift accordingly.

4. **The exploit/explore balance mirrors the classic exploration-exploitation tradeoff** in reinforcement learning. Population size and perturbation magnitude are the key knobs for controlling this balance.

5. **Trading strategy ensembles emerge naturally from PBT**. The final population provides a diverse set of strategies that can be combined for more robust trading decisions.

6. **PBT is particularly well-suited for RL-based trading** because RL agents have many sensitive hyperparameters (learning rate, discount factor, entropy coefficient) that interact in complex ways.

7. **Rust provides the performance needed** for running large populations of agents efficiently. The combination of zero-cost abstractions and memory safety makes it ideal for production trading systems where both speed and reliability matter.

8. **Real market data integration via Bybit** ensures that PBT optimization reflects actual market conditions rather than synthetic benchmarks, leading to more practical trading strategies.

## References

- Jaderberg, M., Dalibard, V., Osindero, S., et al. (2017). "Population Based Training of Neural Networks." arXiv:1711.09846.
- Li, A., et al. (2019). "Generalized Population Based Training." arXiv:1902.01894.
- Parker-Holder, J., et al. (2020). "Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits." NeurIPS 2020.
