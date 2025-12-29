# Chapter 38: Neural ODE — Continuous-Time Portfolio Dynamics

## Overview

Neural Ordinary Differential Equations (Neural ODEs) моделируют непрерывную динамику систем. В отличие от дискретных RNN/LSTM, Neural ODEs естественно обрабатывают нерегулярные временные ряды и позволяют моделировать плавную эволюцию портфеля.

## Trading Strategy

**Суть стратегии:** Моделирование непрерывной динамики оптимального портфеля. Neural ODE предсказывает траекторию оптимальных весов, минимизируя transaction costs при движении к target allocation.

**Сигнал на ребалансировку:**
- Rebalance: Когда текущие веса отклонились от траектории на > threshold
- Target: Предсказанные ODE веса на следующем временном шаге

**Edge:** Плавное ребалансирование vs дискретное monthly rebalancing

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_ode_theory.ipynb` | Теория ODE: Euler, RK4, adjoint method |
| 2 | `02_neural_ode_basics.ipynb` | Neural ODE архитектура с torchdiffeq |
| 3 | `03_financial_dynamics.ipynb` | Моделирование price dynamics как ODE |
| 4 | `04_portfolio_ode.ipynb` | Portfolio weights как решение ODE |
| 5 | `05_continuous_control.ipynb` | Continuous-time optimal control |
| 6 | `06_transaction_costs.ipynb` | Включение transaction costs в loss |
| 7 | `07_training.ipynb` | Обучение Neural ODE на historical data |
| 8 | `08_trajectory_prediction.ipynb` | Предсказание оптимальной траектории |
| 9 | `09_rebalancing_strategy.ipynb` | Стратегия ребалансировки |
| 10 | `10_backtesting.ipynb` | Comparison vs discrete rebalancing |

### Neural ODE Fundamentals

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    Defines the dynamics dz/dt = f(z, t)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, t, z):
        return self.net(z)


class NeuralODE(nn.Module):
    """
    Neural ODE model
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x0, t):
        """
        x0: initial state [batch, features]
        t: time points to evaluate [num_times]
        """
        # Encode initial state
        z0 = self.encoder(x0)

        # Solve ODE
        z_trajectory = odeint(self.ode_func, z0, t, method='dopri5')

        # Decode all time points
        x_trajectory = self.decoder(z_trajectory)

        return x_trajectory
```

### Portfolio Dynamics ODE

```python
class PortfolioDynamics(nn.Module):
    """
    Models portfolio weight evolution as ODE
    dw/dt = f(w, returns, costs)
    """
    def __init__(self, n_assets, hidden_dim=64):
        super().__init__()
        self.n_assets = n_assets

        # Network predicts optimal drift direction
        self.net = nn.Sequential(
            nn.Linear(n_assets * 3, hidden_dim),  # weights, returns, target
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )

        # Transaction cost penalty
        self.cost_weight = 0.001

    def forward(self, t, state):
        """
        state: [weights, returns_forecast, target_weights]
        """
        weights = state[:self.n_assets]
        returns = state[self.n_assets:2*self.n_assets]
        target = state[2*self.n_assets:]

        # Concatenate inputs
        x = torch.cat([weights, returns, target])

        # Predict drift (direction to move weights)
        drift = self.net(x)

        # Apply constraints:
        # 1. Weights should sum to 1 (project to simplex)
        drift = drift - drift.mean()

        # 2. Penalize rapid changes (transaction costs)
        drift = drift * (1 - self.cost_weight * torch.abs(drift))

        return drift


class ContinuousPortfolioOptimizer(nn.Module):
    """
    Full portfolio optimization with Neural ODE
    """
    def __init__(self, n_assets):
        super().__init__()
        self.dynamics = PortfolioDynamics(n_assets)
        self.returns_predictor = nn.LSTM(n_assets, n_assets, batch_first=True)

    def forward(self, initial_weights, historical_returns, time_horizon):
        # Predict future returns
        returns_forecast, _ = self.returns_predictor(historical_returns)
        returns_forecast = returns_forecast[:, -1, :]  # Last prediction

        # Compute target weights (e.g., from mean-variance)
        target_weights = self.compute_target(returns_forecast)

        # Initial state
        state0 = torch.cat([initial_weights, returns_forecast, target_weights])

        # Time points
        t = torch.linspace(0, time_horizon, steps=100)

        # Solve ODE
        trajectory = odeint(self.dynamics, state0, t)

        # Extract weight trajectory
        weight_trajectory = trajectory[:, :self.n_assets]

        return weight_trajectory
```

### Continuous-Time Optimal Control

```python
class OptimalControlODE(nn.Module):
    """
    Hamilton-Jacobi-Bellman inspired continuous control
    """
    def __init__(self, n_assets, risk_aversion=1.0, cost_param=0.001):
        super().__init__()
        self.n_assets = n_assets
        self.gamma = risk_aversion
        self.kappa = cost_param

        # Value function approximator
        self.value_net = nn.Sequential(
            nn.Linear(n_assets + 1, 64),  # weights + time
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Policy (optimal control)
        self.policy_net = nn.Sequential(
            nn.Linear(n_assets + 1, 64),
            nn.Tanh(),
            nn.Linear(64, n_assets),
            nn.Softmax(dim=-1)
        )

    def optimal_drift(self, t, weights, expected_returns, covariance):
        """
        Compute optimal portfolio drift using HJB equation
        """
        # Current state
        state = torch.cat([weights, t.unsqueeze(0)])

        # Optimal target from policy network
        target = self.policy_net(state)

        # Move towards target with speed depending on deviation
        deviation = target - weights
        drift = deviation * self.adjustment_speed(t)

        return drift

    def loss_function(self, trajectory, returns, costs):
        """
        Loss = negative utility + transaction costs
        """
        # Portfolio returns along trajectory
        portfolio_returns = (trajectory * returns).sum(dim=-1)

        # Utility (mean-variance)
        utility = portfolio_returns.mean() - self.gamma * portfolio_returns.var()

        # Transaction costs
        weight_changes = torch.diff(trajectory, dim=0)
        transaction_costs = self.kappa * torch.abs(weight_changes).sum()

        return -utility + transaction_costs
```

### Adjoint Method for Training

```python
def train_neural_ode(model, data, epochs=100):
    """
    Train using adjoint sensitivity method (memory efficient)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for batch in data:
            initial_weights, historical_returns, future_returns = batch

            optimizer.zero_grad()

            # Forward pass (ODE solve)
            # torchdiffeq uses adjoint method by default for backprop
            weight_trajectory = model(initial_weights, historical_returns, time_horizon=1.0)

            # Compute loss
            realized_returns = (weight_trajectory[-1] * future_returns).sum()
            transaction_costs = compute_costs(weight_trajectory)
            loss = -realized_returns + transaction_costs

            # Backward pass (adjoint method)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(data)}")
```

### Rebalancing Strategy

```python
class ContinuousRebalancer:
    """
    Rebalancing strategy based on Neural ODE trajectory
    """
    def __init__(self, model, threshold=0.02):
        self.model = model
        self.threshold = threshold

    def should_rebalance(self, current_weights, time_since_last):
        """
        Check if rebalancing is needed
        """
        # Get predicted optimal weights at current time
        predicted_trajectory = self.model(current_weights, self.market_state, time_horizon=0.1)
        target_weights = predicted_trajectory[-1]

        # Compute deviation
        deviation = torch.abs(current_weights - target_weights).max()

        return deviation > self.threshold

    def get_target_weights(self, current_weights):
        """
        Get target weights from ODE solution
        """
        predicted_trajectory = self.model(current_weights, self.market_state, time_horizon=0.1)
        return predicted_trajectory[-1]

    def execute_rebalance(self, current_weights, target_weights, portfolio_value):
        """
        Execute trades to move towards target
        """
        trades = {}
        for i, asset in enumerate(self.assets):
            weight_diff = target_weights[i] - current_weights[i]
            dollar_amount = weight_diff * portfolio_value
            trades[asset] = dollar_amount

        return trades
```

### Key Metrics

- **Trajectory Quality:** MSE vs realized optimal, Smoothness
- **Rebalancing:** Frequency, Transaction costs, Tracking error
- **Strategy:** Sharpe, Return, Max DD
- **Comparison:** vs monthly rebalance, vs daily rebalance

### Dependencies

```python
torchdiffeq>=0.2.3
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scipy>=1.10.0
```

## Expected Outcomes

1. **Neural ODE implementation** для portfolio dynamics
2. **Continuous-time optimization** с transaction costs
3. **Smooth weight trajectories** vs discrete jumps
4. **Rebalancing strategy** основанная на trajectory deviation
5. **Results:** Lower transaction costs with similar returns

## References

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Chen et al., 2018)
- [torchdiffeq Documentation](https://github.com/rtqichen/torchdiffeq)
- [Continuous-Time Portfolio Optimization](https://www.jstor.org/stable/2328831)
- [Deep Learning for Continuous-Time Finance](https://arxiv.org/abs/2007.04154)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Требуется понимание: Differential equations, Neural ODEs, Continuous optimization, Portfolio theory
