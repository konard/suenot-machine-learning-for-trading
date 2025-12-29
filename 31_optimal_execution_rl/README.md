# Chapter 31: Optimal Execution with Reinforcement Learning — Beating TWAP/VWAP

## Overview

При исполнении крупных ордеров возникает trade-off между market impact (быстрое исполнение) и timing risk (медленное исполнение). Классические алгоритмы TWAP/VWAP используют детерминистические расписания. В этой главе мы обучаем RL агента, который адаптивно выбирает оптимальную скорость исполнения.

## Trading Strategy

**Суть стратегии:** RL агент решает, сколько акций исполнить в каждый момент времени, минимизируя implementation shortfall (разницу между decision price и execution price).

**State:** Order book state, remaining quantity, time left, recent price moves
**Action:** Quantity to execute in current interval (0% to 100% of remaining)
**Reward:** Negative implementation shortfall - penalty for risk

**Edge:** Адаптация к текущим market conditions vs статичное расписание

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_execution_theory.ipynb` | Almgren-Chriss model, implementation shortfall |
| 2 | `02_market_impact_models.ipynb` | Linear, square-root, transient impact models |
| 3 | `03_twap_vwap_baselines.ipynb` | Реализация классических алгоритмов |
| 4 | `04_gym_environment.ipynb` | Custom Gym environment для execution |
| 5 | `05_reward_function.ipynb` | Дизайн reward: shortfall + risk penalty |
| 6 | `06_dqn_agent.ipynb` | Deep Q-Network для дискретных actions |
| 7 | `07_ppo_agent.ipynb` | PPO для continuous action space |
| 8 | `08_training_curriculum.ipynb` | Curriculum learning: easy → hard scenarios |
| 9 | `09_evaluation.ipynb` | Comparison vs TWAP/VWAP/Almgren-Chriss |
| 10 | `10_robustness.ipynb` | Out-of-sample, different market conditions |

### Data Requirements

```
Order Book Data:
├── L2 snapshots (1-second frequency)
├── Trade prints with sizes
├── At least 6 months for training
└── Liquid instruments (ES futures, SPY, AAPL)

Execution Scenarios:
├── Order sizes: 1%, 5%, 10% of ADV
├── Time horizons: 5min, 30min, 1hour, 1day
├── Market conditions: normal, volatile, trending
└── Simulated orders with various urgency levels
```

### Gym Environment Design

```python
class ExecutionEnv(gym.Env):
    """
    Environment for optimal execution
    """
    def __init__(self, order_size, time_horizon, market_data):
        self.total_shares = order_size
        self.remaining_shares = order_size
        self.time_horizon = time_horizon
        self.current_step = 0

        # Action: fraction of remaining to execute (0 to 1)
        self.action_space = gym.spaces.Box(0, 1, shape=(1,))

        # State: [remaining_frac, time_frac, spread, volatility, imbalance, momentum]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,))

    def step(self, action):
        execute_qty = action * self.remaining_shares
        execution_price = self._simulate_execution(execute_qty)

        # Implementation shortfall
        shortfall = (execution_price - self.arrival_price) * execute_qty

        # Risk penalty (variance of remaining)
        risk_penalty = self.lambda_risk * self.remaining_shares * self.volatility

        reward = -shortfall - risk_penalty

        self.remaining_shares -= execute_qty
        self.current_step += 1

        done = (self.remaining_shares <= 0) or (self.current_step >= self.max_steps)

        return self._get_state(), reward, done, {}
```

### Market Impact Model

```python
def temporary_impact(quantity, params):
    """
    Temporary impact: affects only current trade
    Linear-in-volume model
    """
    return params['eta'] * quantity / params['adv']

def permanent_impact(quantity, params):
    """
    Permanent impact: shifts price for future trades
    Square-root model (more realistic)
    """
    return params['gamma'] * np.sign(quantity) * np.sqrt(abs(quantity) / params['adv'])

def total_cost(execution_schedule, params):
    """
    Total execution cost = sum of (temporary + permanent) impacts
    """
    cost = 0
    price = params['initial_price']
    for qty in execution_schedule:
        temp = temporary_impact(qty, params)
        perm = permanent_impact(qty, params)
        execution_price = price + temp + perm/2
        cost += qty * (execution_price - params['arrival_price'])
        price += perm
    return cost
```

### RL Algorithms

```
DQN (Discrete Actions):
├── Actions: [0%, 10%, 20%, ..., 100%] of remaining
├── Network: State → Q-values for each action
├── Experience replay, target network
└── Good for: interpretability, simpler training

PPO (Continuous Actions):
├── Actions: continuous [0, 1] fraction
├── Actor-Critic architecture
├── Clipped objective for stability
└── Good for: fine-grained control

SAC (Soft Actor-Critic):
├── Entropy regularization
├── Better exploration
└── Good for: complex market dynamics
```

### Almgren-Chriss Baseline

```python
def almgren_chriss_optimal(total_shares, time_horizon, risk_aversion, volatility, impact_params):
    """
    Closed-form optimal execution trajectory
    Trades off impact cost vs timing risk
    """
    kappa = np.sqrt(risk_aversion * volatility**2 / impact_params['eta'])

    trajectory = []
    for t in range(time_horizon):
        remaining_time = time_horizon - t
        optimal_trade = total_shares * np.sinh(kappa * remaining_time) / np.sinh(kappa * time_horizon)
        trajectory.append(optimal_trade)

    return trajectory
```

### Key Metrics

- **Execution Quality:** Implementation Shortfall, Arrival Price Slippage
- **Comparison:** vs TWAP, VWAP, Almgren-Chriss, Optimal IS
- **Risk-Adjusted:** Sharpe of execution cost distribution
- **Robustness:** Performance across different volatility regimes

### Dependencies

```python
gymnasium>=0.29.0
stable-baselines3>=2.1.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
```

## Expected Outcomes

1. **Custom Gym environment** для optimal execution
2. **Market impact models** (linear, square-root, transient)
3. **RL agents** (DQN, PPO) обученные на execution
4. **Comparison framework** vs classical algorithms
5. **Results:** 5-15% improvement vs TWAP in implementation shortfall

## References

- [Optimal Execution of Portfolio Transactions](https://www.math.nyu.edu/~almgren/papers/optliq.pdf) (Almgren-Chriss, 2000)
- [Optimal Execution with Reinforcement Learning](https://arxiv.org/abs/1906.02312)
- [Deep Reinforcement Learning for Optimal Execution](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3304766)

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: Reinforcement Learning, Market Microstructure, Execution Algorithms
