# Chapter 35: Multi-Agent Reinforcement Learning — Market Simulation and Competitive Strategies

## Overview

Реальные рынки состоят из множества взаимодействующих агентов. Single-agent RL игнорирует эту динамику. Multi-agent RL (MARL) позволяет моделировать конкурентную среду и обучать агентов, робастных к действиям других участников.

## Trading Strategy

**Суть стратегии:** Обучение торгового агента через self-play и конкуренцию с другими агентами. Агент учится:
1. Адаптироваться к различным типам противников
2. Находить Nash equilibrium стратегии
3. Быть робастным к market manipulation

**Edge:** Стратегии, выживающие в конкурентной симуляции, более робастны в реальности

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_market_simulation.ipynb` | Построение симулятора рынка с order book |
| 2 | `02_agent_types.ipynb` | Разные типы агентов: trend, mean-rev, noise |
| 3 | `03_marl_theory.ipynb` | Теория MARL: Nash, Pareto, learning dynamics |
| 4 | `04_environment.ipynb` | Multi-agent Gym environment |
| 5 | `05_independent_learners.ipynb` | Independent DQN агенты |
| 6 | `06_centralized_critic.ipynb` | MADDPG: centralized training |
| 7 | `07_self_play.ipynb` | Self-play для robustness |
| 8 | `08_population_training.ipynb` | Population-based training |
| 9 | `09_equilibrium_analysis.ipynb` | Анализ emergent strategies |
| 10 | `10_adversarial_robustness.ipynb` | Тестирование против adversarial agents |
| 11 | `11_transfer_to_real.ipynb` | Перенос стратегий на реальные данные |

### Market Simulation Environment

```python
class MultiAgentMarketEnv:
    """
    Simulated market with multiple trading agents
    """
    def __init__(self, n_agents, initial_price=100):
        self.n_agents = n_agents
        self.price = initial_price
        self.order_book = OrderBook()

        # Each agent has cash and inventory
        self.cash = {i: 100000 for i in range(n_agents)}
        self.inventory = {i: 0 for i in range(n_agents)}

    def step(self, actions):
        """
        actions: dict of agent_id -> (order_type, price, quantity)
        """
        # Collect all orders
        orders = []
        for agent_id, action in actions.items():
            order = self._parse_action(agent_id, action)
            orders.append(order)

        # Match orders (price-time priority)
        trades = self.order_book.match(orders)

        # Update agent positions
        for trade in trades:
            self._settle_trade(trade)

        # Update price based on order flow
        self.price = self.order_book.mid_price()

        # Calculate rewards (PnL)
        rewards = self._calculate_rewards()

        return self._get_observations(), rewards, self._is_done(), {}

    def _calculate_rewards(self):
        rewards = {}
        for agent_id in range(self.n_agents):
            # Mark-to-market PnL
            portfolio_value = self.cash[agent_id] + self.inventory[agent_id] * self.price
            rewards[agent_id] = portfolio_value - self.prev_portfolio[agent_id]
        return rewards
```

### Agent Types for Population

```python
class TrendFollowingAgent:
    """Buys on uptrend, sells on downtrend"""
    def act(self, observation):
        momentum = observation['price_change_10']
        if momentum > self.threshold:
            return ('buy', self.size)
        elif momentum < -self.threshold:
            return ('sell', self.size)
        return ('hold', 0)

class MeanReversionAgent:
    """Bets on price reverting to mean"""
    def act(self, observation):
        deviation = observation['price'] - observation['sma_50']
        if deviation > self.threshold:
            return ('sell', self.size)
        elif deviation < -self.threshold:
            return ('buy', self.size)
        return ('hold', 0)

class NoiseTrader:
    """Random trading (provides liquidity)"""
    def act(self, observation):
        action = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4])
        size = np.random.randint(1, 10) if action != 'hold' else 0
        return (action, size)

class MarketMaker:
    """Provides liquidity, profits from spread"""
    def act(self, observation):
        spread = observation['spread']
        return ('quote', observation['mid'] - spread/2, observation['mid'] + spread/2, self.size)
```

### MARL Algorithms

```python
# 1. Independent Learners (IL)
class IndependentDQN:
    """Each agent learns independently, treating others as environment"""
    def __init__(self, n_agents):
        self.agents = [DQNAgent() for _ in range(n_agents)]

    def act(self, observations):
        return {i: agent.act(obs) for i, (agent, obs) in
                enumerate(zip(self.agents, observations))}

    def learn(self, experiences):
        for i, agent in enumerate(self.agents):
            agent.learn(experiences[i])


# 2. MADDPG (Multi-Agent DDPG)
class MADDPG:
    """
    Centralized training, decentralized execution
    Critic sees all agents' observations and actions
    """
    def __init__(self, n_agents, obs_dim, action_dim):
        self.actors = [Actor(obs_dim, action_dim) for _ in range(n_agents)]
        self.critics = [Critic(n_agents * obs_dim, n_agents * action_dim)
                       for _ in range(n_agents)]

    def act(self, observations, explore=True):
        actions = {}
        for i, (actor, obs) in enumerate(zip(self.actors, observations)):
            action = actor(obs)
            if explore:
                action += noise()
            actions[i] = action
        return actions

    def learn(self, batch):
        # Critic uses all observations and actions
        all_obs = concat([b['obs'] for b in batch])
        all_actions = concat([b['action'] for b in batch])

        for i in range(self.n_agents):
            # Update critic
            Q_target = batch[i]['reward'] + gamma * self.critics[i](next_all_obs, next_all_actions)
            critic_loss = mse(self.critics[i](all_obs, all_actions), Q_target)

            # Update actor using policy gradient
            actor_loss = -self.critics[i](all_obs, self.actors[i](batch[i]['obs']))
```

### Self-Play Training

```python
class SelfPlayTrainer:
    """
    Train agent against copies of itself
    """
    def __init__(self, agent_class, n_opponents=4):
        self.main_agent = agent_class()
        self.opponent_pool = [agent_class() for _ in range(n_opponents)]
        self.win_rates = [0.5] * n_opponents

    def train_episode(self):
        # Select opponent (prioritize challenging ones)
        opponent_idx = self._select_opponent()
        opponent = self.opponent_pool[opponent_idx]

        # Run episode
        env = TwoPlayerMarketEnv()
        obs = env.reset()

        while not done:
            action_main = self.main_agent.act(obs[0])
            action_opp = opponent.act(obs[1])
            obs, rewards, done, _ = env.step([action_main, action_opp])

        # Update main agent
        self.main_agent.learn(episode_buffer)

        # Periodically add main agent to opponent pool
        if episode % 100 == 0:
            self._update_opponent_pool()

    def _update_opponent_pool(self):
        # Replace worst opponent with current main agent
        worst_idx = np.argmin(self.win_rates)
        self.opponent_pool[worst_idx] = copy.deepcopy(self.main_agent)
```

### Population-Based Training

```python
class PopulationTrainer:
    """
    Evolve population of agents with different hyperparameters
    """
    def __init__(self, population_size=20):
        self.population = [
            {'agent': DQNAgent(lr=lr, gamma=gamma, ...),
             'hyperparams': {'lr': lr, 'gamma': gamma},
             'fitness': 0}
            for lr, gamma in random_hyperparams(population_size)
        ]

    def evolve(self, n_generations=100):
        for gen in range(n_generations):
            # Evaluate all agents in multi-agent environment
            fitness_scores = self._evaluate_population()

            # Update fitness
            for i, score in enumerate(fitness_scores):
                self.population[i]['fitness'] = score

            # Select top performers
            sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
            top_half = sorted_pop[:len(sorted_pop)//2]

            # Exploit: copy top agents
            # Explore: mutate hyperparameters
            new_population = []
            for agent_info in top_half:
                new_population.append(agent_info)  # Keep original
                mutated = self._mutate(agent_info)  # Create mutant
                new_population.append(mutated)

            self.population = new_population
```

### Emergent Behavior Analysis

```python
def analyze_equilibrium(trained_agents, env, n_episodes=1000):
    """
    Analyze emergent strategies and equilibrium
    """
    metrics = {
        'price_volatility': [],
        'market_efficiency': [],
        'agent_profits': {i: [] for i in range(len(trained_agents))},
        'order_flow_balance': []
    }

    for _ in range(n_episodes):
        obs = env.reset()
        episode_data = []

        while not done:
            actions = {i: agent.act(obs[i]) for i, agent in enumerate(trained_agents)}
            obs, rewards, done, info = env.step(actions)
            episode_data.append(info)

        # Compute metrics
        metrics['price_volatility'].append(compute_volatility(episode_data))
        # ... other metrics

    return metrics
```

### Key Metrics

- **Individual:** Agent profit, Sharpe ratio, Win rate vs opponents
- **Population:** Diversity of strategies, Convergence to equilibrium
- **Market Quality:** Price efficiency, Volatility, Liquidity
- **Robustness:** Performance vs unseen agent types

### Dependencies

```python
pettingzoo>=1.24.0     # Multi-agent environments
stable-baselines3>=2.1.0
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.23.0
matplotlib>=3.6.0
```

## Expected Outcomes

1. **Market simulator** с order book и multiple agent types
2. **MARL training pipeline** (IL, MADDPG, self-play)
3. **Population-based training** с hyperparameter evolution
4. **Emergent strategy analysis** — какие стратегии выживают
5. **Robust agent** с хорошей performance против разных противников

## References

- [Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635)
- [MADDPG: Multi-Agent Actor-Critic](https://arxiv.org/abs/1706.02275)
- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Требуется понимание: Reinforcement Learning, Game Theory, Market Microstructure, Multi-agent systems
