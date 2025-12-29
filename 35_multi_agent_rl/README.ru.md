# Глава 35: Мультиагентное Обучение с Подкреплением — Симуляция Рынка и Конкурентные Стратегии

## Обзор

Реальные финансовые рынки представляют собой сложные экосистемы, где множество участников одновременно принимают решения, влияющие друг на друга. Традиционное обучение с подкреплением (single-agent RL) моделирует одного агента, который взаимодействует со статичной средой — это упрощение игнорирует критически важную динамику конкуренции и кооперации между участниками рынка.

**Мультиагентное обучение с подкреплением (Multi-Agent Reinforcement Learning, MARL)** позволяет моделировать эту сложную динамику, обучая несколько агентов одновременно. Агенты учатся адаптироваться не только к рыночным условиям, но и к поведению других участников.

### Почему это важно для трейдинга?

1. **Робастность стратегий**: Стратегии, выжившие в конкурентной симуляции против разнообразных противников, более устойчивы на реальном рынке
2. **Market impact**: Моделирование влияния ваших ордеров на рынок и реакции других участников
3. **Равновесие Нэша**: Поиск стратегий, оптимальных при любом поведении конкурентов
4. **Эмерджентное поведение**: Обнаружение неочевидных рыночных паттернов через симуляцию

## Торговая Стратегия

### Суть Стратегии

Обучение торгового агента через **self-play** (игра против копий себя) и **конкуренцию с разнообразными противниками**. Агент учится:

1. **Адаптироваться** к различным типам противников (тренд-следящие, mean-reversion, маркет-мейкеры)
2. **Находить равновесие Нэша** — стратегии, оптимальные независимо от действий других
3. **Быть робастным** к манипуляциям и неожиданному поведению

### Преимущество (Edge)

> Стратегии, которые выживают в конкурентной симуляции с множеством агентов, демонстрируют значительно большую робастность при переносе на реальные рыночные данные.

## Теоретические Основы MARL

### Формализация: Стохастические Игры (Markov Games)

В отличие от MDP для одного агента, мультиагентная среда описывается **стохастической игрой**:

- **S** — множество состояний среды
- **A₁, A₂, ..., Aₙ** — множества действий для каждого агента
- **T: S × A₁ × ... × Aₙ → Δ(S)** — функция перехода (зависит от действий всех агентов)
- **R₁, R₂, ..., Rₙ** — функции награды для каждого агента

### Типы Игр

| Тип | Описание | Пример в Трейдинге |
|-----|----------|-------------------|
| **Полностью кооперативные** | Все агенты имеют общую награду | Портфельная оптимизация командой |
| **Полностью конкурентные (zero-sum)** | Выигрыш одного = проигрыш другого | Высокочастотная торговля |
| **Смешанные (general-sum)** | Награды независимы | Реальный рынок |

### Равновесие Нэша

Набор стратегий (π₁*, π₂*, ..., πₙ*) является **равновесием Нэша**, если ни один агент не может улучшить свою награду, изменив только свою стратегию:

```
∀i: V_i(π₁*, ..., πᵢ*, ..., πₙ*) ≥ V_i(π₁*, ..., πᵢ, ..., πₙ*)
```

В трейдинге это означает стратегию, оптимальную при любых действиях конкурентов.

### Проблема Нестационарности

Ключевая сложность MARL: среда **нестационарна** с точки зрения каждого агента, так как другие агенты тоже обучаются. Это нарушает теоретические гарантии сходимости стандартных RL-алгоритмов.

## Техническая Спецификация

### Notebooks для Создания

| # | Notebook | Описание |
|---|----------|----------|
| 1 | `01_market_simulation.ipynb` | Построение симулятора рынка с order book |
| 2 | `02_agent_types.ipynb` | Реализация разных типов агентов |
| 3 | `03_marl_theory.ipynb` | Теория MARL: Nash, Pareto, динамика обучения |
| 4 | `04_environment.ipynb` | Multi-agent Gym environment |
| 5 | `05_independent_learners.ipynb` | Независимые DQN агенты |
| 6 | `06_centralized_critic.ipynb` | MADDPG: централизованное обучение |
| 7 | `07_self_play.ipynb` | Self-play для робастности |
| 8 | `08_population_training.ipynb` | Population-based training |
| 9 | `09_equilibrium_analysis.ipynb` | Анализ эмерджентных стратегий |
| 10 | `10_adversarial_robustness.ipynb` | Тестирование против adversarial агентов |
| 11 | `11_transfer_to_real.ipynb` | Перенос стратегий на реальные данные |

### Архитектура Симулятора Рынка

```
┌─────────────────────────────────────────────────────────────┐
│                    Market Simulator                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agent 1   │  │   Agent 2   │  │   Agent N   │  ...    │
│  │ (RL/Rule)   │  │ (RL/Rule)   │  │ (RL/Rule)   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Order Book                        │   │
│  │  ┌─────────────────┬─────────────────┐              │   │
│  │  │   Bids (Buy)    │   Asks (Sell)   │              │   │
│  │  │  99.50 x 100    │  100.50 x 150   │              │   │
│  │  │  99.00 x 200    │  101.00 x 100   │              │   │
│  │  └─────────────────┴─────────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Matching Engine                         │   │
│  │  • Price-Time Priority                               │   │
│  │  • Trade Execution                                   │   │
│  │  • Price Discovery                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Окружение для Мультиагентного Рынка

```python
class MultiAgentMarketEnv:
    """
    Симулятор рынка с несколькими торговыми агентами.

    Каждый агент имеет:
    - Начальный капитал (cash)
    - Инвентарь актива (inventory)
    - Возможность размещать ордера в order book
    """

    def __init__(self, n_agents: int, initial_price: float = 100.0):
        self.n_agents = n_agents
        self.price = initial_price
        self.order_book = OrderBook()

        # Начальное состояние каждого агента
        self.cash = {i: 100_000.0 for i in range(n_agents)}
        self.inventory = {i: 0.0 for i in range(n_agents)}
        self.prev_portfolio = {i: 100_000.0 for i in range(n_agents)}

    def step(self, actions: dict) -> tuple:
        """
        Выполнить шаг симуляции.

        Args:
            actions: словарь {agent_id: (order_type, price, quantity)}

        Returns:
            observations: наблюдения для каждого агента
            rewards: награды (изменение портфеля)
            done: флаг завершения
            info: дополнительная информация
        """
        # 1. Собираем ордера от всех агентов
        orders = []
        for agent_id, action in actions.items():
            order = self._parse_action(agent_id, action)
            orders.append(order)

        # 2. Сопоставляем ордера (price-time priority)
        trades = self.order_book.match(orders)

        # 3. Исполняем сделки
        for trade in trades:
            self._settle_trade(trade)

        # 4. Обновляем цену на основе потока ордеров
        self.price = self.order_book.mid_price()

        # 5. Рассчитываем награды (PnL)
        rewards = self._calculate_rewards()

        return self._get_observations(), rewards, self._is_done(), {}

    def _calculate_rewards(self) -> dict:
        """
        Награда = изменение стоимости портфеля (mark-to-market PnL).
        """
        rewards = {}
        for agent_id in range(self.n_agents):
            # Текущая стоимость: наличные + инвентарь по рыночной цене
            portfolio_value = (
                self.cash[agent_id] +
                self.inventory[agent_id] * self.price
            )
            # Награда = изменение стоимости
            rewards[agent_id] = portfolio_value - self.prev_portfolio[agent_id]
            self.prev_portfolio[agent_id] = portfolio_value
        return rewards

    def _get_observations(self) -> dict:
        """
        Формирует наблюдения для каждого агента.

        Каждый агент видит:
        - Текущую цену и спред
        - Глубину order book
        - Свой инвентарь и P&L
        - (Опционально) агрегированную активность других
        """
        obs = {}
        for agent_id in range(self.n_agents):
            obs[agent_id] = {
                'price': self.price,
                'spread': self.order_book.spread(),
                'bid_depth': self.order_book.bid_depth(),
                'ask_depth': self.order_book.ask_depth(),
                'inventory': self.inventory[agent_id],
                'cash': self.cash[agent_id],
                'pnl': self.prev_portfolio[agent_id] - 100_000
            }
        return obs
```

### Типы Агентов для Популяции

```python
class TrendFollowingAgent:
    """
    Тренд-следящий агент.

    Покупает при восходящем тренде, продаёт при нисходящем.
    Типичная стратегия momentum-трейдеров.
    """

    def __init__(self, threshold: float = 0.01, size: int = 10):
        self.threshold = threshold
        self.size = size

    def act(self, observation: dict) -> tuple:
        momentum = observation['price_change_10']  # 10-периодный momentum

        if momentum > self.threshold:
            return ('buy', self.size)
        elif momentum < -self.threshold:
            return ('sell', self.size)
        return ('hold', 0)


class MeanReversionAgent:
    """
    Mean-reversion агент.

    Делает ставку на возврат цены к среднему значению.
    Продаёт, когда цена выше среднего, покупает — когда ниже.
    """

    def __init__(self, threshold: float = 2.0, size: int = 10):
        self.threshold = threshold  # Стандартные отклонения
        self.size = size

    def act(self, observation: dict) -> tuple:
        deviation = observation['price'] - observation['sma_50']
        std = observation['std_50']
        z_score = deviation / std if std > 0 else 0

        if z_score > self.threshold:
            return ('sell', self.size)  # Цена слишком высока
        elif z_score < -self.threshold:
            return ('buy', self.size)   # Цена слишком низка
        return ('hold', 0)


class NoiseTrader:
    """
    Шумовой трейдер.

    Торгует случайно, обеспечивая ликвидность рынку.
    Моделирует неинформированных розничных трейдеров.
    """

    def act(self, observation: dict) -> tuple:
        action = np.random.choice(
            ['buy', 'sell', 'hold'],
            p=[0.3, 0.3, 0.4]
        )
        size = np.random.randint(1, 10) if action != 'hold' else 0
        return (action, size)


class MarketMaker:
    """
    Маркет-мейкер.

    Предоставляет ликвидность, выставляя двусторонние котировки.
    Зарабатывает на спреде bid-ask.
    """

    def __init__(self, base_spread: float = 0.1, size: int = 50):
        self.base_spread = base_spread
        self.size = size

    def act(self, observation: dict) -> tuple:
        mid = observation['price']
        inventory = observation.get('inventory', 0)

        # Корректируем спред на основе инвентаря (inventory risk)
        inventory_skew = inventory * 0.001

        bid_price = mid - self.base_spread/2 - inventory_skew
        ask_price = mid + self.base_spread/2 - inventory_skew

        return ('quote', bid_price, ask_price, self.size)


class InformedTrader:
    """
    Информированный трейдер.

    Имеет доступ к "будущей" информации (в симуляции).
    Используется для тестирования робастности других стратегий.
    """

    def __init__(self, alpha: float = 0.8, size: int = 20):
        self.alpha = alpha  # Точность сигнала
        self.size = size

    def act(self, observation: dict, future_return: float) -> tuple:
        # С вероятностью alpha получаем правильный сигнал
        if np.random.random() < self.alpha:
            signal = future_return
        else:
            signal = np.random.normal(0, 0.02)

        if signal > 0.005:
            return ('buy', self.size)
        elif signal < -0.005:
            return ('sell', self.size)
        return ('hold', 0)
```

## Алгоритмы MARL

### 1. Независимые Learners (Independent Learners)

Простейший подход: каждый агент обучается независимо, воспринимая других как часть среды.

```python
class IndependentDQN:
    """
    Независимые DQN агенты.

    Каждый агент использует стандартный DQN,
    игнорируя, что другие агенты тоже обучаются.

    Плюсы: простота, масштабируемость
    Минусы: нестационарность среды, нет гарантий сходимости
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int):
        self.agents = [
            DQNAgent(obs_dim, action_dim)
            for _ in range(n_agents)
        ]

    def act(self, observations: dict, explore: bool = True) -> dict:
        """Каждый агент выбирает действие на основе своего наблюдения."""
        actions = {}
        for i, agent in enumerate(self.agents):
            actions[i] = agent.act(observations[i], explore)
        return actions

    def learn(self, experiences: dict):
        """Каждый агент обучается на своём опыте."""
        for i, agent in enumerate(self.agents):
            agent.learn(experiences[i])
```

### 2. MADDPG (Multi-Agent DDPG)

Централизованное обучение, децентрализованное исполнение.

```python
class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient.

    Ключевая идея:
    - При ОБУЧЕНИИ критик видит действия и наблюдения ВСЕХ агентов
    - При ИСПОЛНЕНИИ каждый агент использует только своё наблюдение

    Это решает проблему нестационарности: критик моделирует
    совместную динамику всех агентов.
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int):
        self.n_agents = n_agents

        # Актор для каждого агента (локальное наблюдение → действие)
        self.actors = [
            Actor(obs_dim, action_dim)
            for _ in range(n_agents)
        ]

        # Критик для каждого агента (ВСЕ наблюдения + ВСЕ действия → Q-value)
        self.critics = [
            Critic(n_agents * obs_dim, n_agents * action_dim)
            for _ in range(n_agents)
        ]

        # Target networks для стабильности
        self.target_actors = [copy.deepcopy(a) for a in self.actors]
        self.target_critics = [copy.deepcopy(c) for c in self.critics]

    def act(self, observations: dict, explore: bool = True) -> dict:
        """
        Децентрализованное исполнение.
        Каждый агент видит только своё наблюдение.
        """
        actions = {}
        for i, actor in enumerate(self.actors):
            action = actor(observations[i])
            if explore:
                action += self._noise()
            actions[i] = action.clip(-1, 1)
        return actions

    def learn(self, batch: list):
        """
        Централизованное обучение.
        Критик видит состояния и действия всех агентов.
        """
        # Конкатенируем наблюдения и действия всех агентов
        all_obs = torch.cat([b['obs'] for b in batch], dim=-1)
        all_actions = torch.cat([b['action'] for b in batch], dim=-1)
        all_next_obs = torch.cat([b['next_obs'] for b in batch], dim=-1)

        # Получаем действия target акторов
        next_actions = []
        for i, target_actor in enumerate(self.target_actors):
            next_actions.append(target_actor(batch[i]['next_obs']))
        all_next_actions = torch.cat(next_actions, dim=-1)

        for i in range(self.n_agents):
            # === Обновление критика ===
            with torch.no_grad():
                Q_target_next = self.target_critics[i](all_next_obs, all_next_actions)
                Q_target = batch[i]['reward'] + self.gamma * Q_target_next

            Q_expected = self.critics[i](all_obs, all_actions)
            critic_loss = F.mse_loss(Q_expected, Q_target)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # === Обновление актора ===
            # Заменяем действие i-го агента на выход его актора
            current_actions = list(batch[j]['action'] for j in range(self.n_agents))
            current_actions[i] = self.actors[i](batch[i]['obs'])
            all_current_actions = torch.cat(current_actions, dim=-1)

            actor_loss = -self.critics[i](all_obs, all_current_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Мягкое обновление target сетей
        self._soft_update_targets()
```

### 3. Self-Play Training

Обучение агента против копий самого себя.

```python
class SelfPlayTrainer:
    """
    Обучение через self-play.

    Агент играет против предыдущих версий себя.
    Это обеспечивает:
    - Автоматическую генерацию curriculum
    - Робастность к широкому спектру стратегий
    - Постепенное усложнение противников
    """

    def __init__(self, agent_class, n_opponents: int = 4):
        self.main_agent = agent_class()
        self.opponent_pool = [agent_class() for _ in range(n_opponents)]
        self.win_rates = [0.5] * n_opponents  # Отслеживаем сложность

    def train_episode(self):
        # 1. Выбираем противника (приоритет — сложные)
        opponent_idx = self._select_opponent()
        opponent = self.opponent_pool[opponent_idx]

        # 2. Проводим эпизод
        env = TwoPlayerMarketEnv()
        obs = env.reset()
        episode_buffer = []
        done = False

        while not done:
            action_main = self.main_agent.act(obs[0])
            action_opp = opponent.act(obs[1])

            next_obs, rewards, done, info = env.step([action_main, action_opp])

            episode_buffer.append({
                'obs': obs[0],
                'action': action_main,
                'reward': rewards[0],
                'next_obs': next_obs[0],
                'done': done
            })

            obs = next_obs

        # 3. Обучаем главного агента
        self.main_agent.learn(episode_buffer)

        # 4. Обновляем статистику win-rate
        main_won = rewards[0] > rewards[1]
        self.win_rates[opponent_idx] = (
            0.95 * self.win_rates[opponent_idx] +
            0.05 * (1.0 if main_won else 0.0)
        )

        # 5. Периодически добавляем агента в пул противников
        if self.episode % 100 == 0:
            self._update_opponent_pool()

    def _select_opponent(self) -> int:
        """
        Выбираем противника с приоритетом на сложных.
        Используем softmax с temperature для баланса.
        """
        difficulties = [1 - wr for wr in self.win_rates]
        probs = softmax(np.array(difficulties) / self.temperature)
        return np.random.choice(len(self.opponent_pool), p=probs)

    def _update_opponent_pool(self):
        """
        Заменяем самого слабого противника текущей версией агента.
        """
        worst_idx = np.argmin(self.win_rates)
        self.opponent_pool[worst_idx] = copy.deepcopy(self.main_agent)
        self.win_rates[worst_idx] = 0.5  # Сбрасываем статистику
```

### 4. Population-Based Training (PBT)

Эволюция популяции агентов с разными гиперпараметрами.

```python
class PopulationTrainer:
    """
    Population-Based Training.

    Одновременно обучаем популяцию агентов с разными гиперпараметрами.
    Плохие агенты копируют веса и гиперпараметры хороших,
    с небольшими мутациями.
    """

    def __init__(self, population_size: int = 20):
        self.population = []

        for _ in range(population_size):
            # Случайные гиперпараметры
            hyperparams = {
                'lr': np.random.loguniform(1e-5, 1e-2),
                'gamma': np.random.uniform(0.9, 0.999),
                'epsilon_decay': np.random.uniform(0.99, 0.9999),
                'hidden_size': np.random.choice([64, 128, 256])
            }

            self.population.append({
                'agent': DQNAgent(**hyperparams),
                'hyperparams': hyperparams,
                'fitness': 0.0
            })

    def evolve(self, n_generations: int = 100):
        for gen in range(n_generations):
            # 1. Оцениваем всех агентов в мультиагентной среде
            fitness_scores = self._evaluate_population()

            for i, score in enumerate(fitness_scores):
                # Экспоненциальное скользящее среднее
                self.population[i]['fitness'] = (
                    0.8 * self.population[i]['fitness'] +
                    0.2 * score
                )

            # 2. Сортируем по fitness
            sorted_pop = sorted(
                self.population,
                key=lambda x: x['fitness'],
                reverse=True
            )

            # 3. Нижняя половина копирует верхнюю + мутации
            n_half = len(sorted_pop) // 2
            for i in range(n_half, len(sorted_pop)):
                # Выбираем случайного из верхней половины
                source_idx = np.random.randint(0, n_half)
                source = sorted_pop[source_idx]

                # Копируем веса
                sorted_pop[i]['agent'].load_weights(
                    source['agent'].get_weights()
                )

                # Мутируем гиперпараметры
                sorted_pop[i]['hyperparams'] = self._mutate_hyperparams(
                    source['hyperparams']
                )
                sorted_pop[i]['agent'].update_hyperparams(
                    sorted_pop[i]['hyperparams']
                )

            self.population = sorted_pop

            print(f"Gen {gen}: Best fitness = {sorted_pop[0]['fitness']:.4f}")

    def _mutate_hyperparams(self, hyperparams: dict) -> dict:
        """Применяем небольшие мутации к гиперпараметрам."""
        new_hp = hyperparams.copy()

        if np.random.random() < 0.3:  # 30% шанс мутации
            for key in new_hp:
                if np.random.random() < 0.5:
                    if key == 'lr':
                        new_hp[key] *= np.random.choice([0.8, 1.2])
                    elif key == 'gamma':
                        new_hp[key] = np.clip(
                            new_hp[key] + np.random.normal(0, 0.01),
                            0.9, 0.999
                        )

        return new_hp

    def _evaluate_population(self) -> list:
        """
        Оцениваем агентов в турнире.
        Каждый агент играет несколько раундов против случайных противников.
        """
        scores = [0.0] * len(self.population)
        n_games = 10

        for _ in range(n_games):
            # Случайно выбираем агентов для игры
            agents = random.sample(range(len(self.population)),
                                   min(4, len(self.population)))

            env = MultiAgentMarketEnv(n_agents=len(agents))
            obs = env.reset()

            total_rewards = {i: 0.0 for i in range(len(agents))}

            for _ in range(1000):  # 1000 шагов
                actions = {}
                for j, agent_idx in enumerate(agents):
                    actions[j] = self.population[agent_idx]['agent'].act(obs[j])

                obs, rewards, done, _ = env.step(actions)

                for j in range(len(agents)):
                    total_rewards[j] += rewards[j]

                if done:
                    break

            # Обновляем scores
            for j, agent_idx in enumerate(agents):
                scores[agent_idx] += total_rewards[j]

        return [s / n_games for s in scores]
```

## Анализ Эмерджентного Поведения

```python
def analyze_equilibrium(trained_agents: list, env, n_episodes: int = 1000):
    """
    Анализ эмерджентных стратегий и рыночного равновесия.

    Исследуем:
    - Волатильность цен: стабилизируется ли рынок?
    - Эффективность рынка: отражают ли цены информацию?
    - Распределение прибыли: кто выигрывает?
    - Баланс ордеров: есть ли стабильная ликвидность?
    """
    metrics = {
        'price_volatility': [],
        'market_efficiency': [],
        'agent_profits': {i: [] for i in range(len(trained_agents))},
        'order_flow_balance': [],
        'strategy_diversity': []
    }

    for episode in range(n_episodes):
        obs = env.reset()
        episode_data = []
        done = False

        while not done:
            # Собираем действия
            actions = {
                i: agent.act(obs[i])
                for i, agent in enumerate(trained_agents)
            }

            obs, rewards, done, info = env.step(actions)
            episode_data.append({
                'price': info['price'],
                'actions': actions,
                'rewards': rewards,
                'spread': info['spread']
            })

        # Вычисляем метрики эпизода
        prices = [d['price'] for d in episode_data]

        metrics['price_volatility'].append(np.std(prices))
        metrics['market_efficiency'].append(
            compute_efficiency(prices, env.fundamental_value)
        )

        for i in range(len(trained_agents)):
            total_profit = sum(d['rewards'][i] for d in episode_data)
            metrics['agent_profits'][i].append(total_profit)

        # Анализ разнообразия стратегий
        action_counts = count_action_types(episode_data)
        metrics['strategy_diversity'].append(
            compute_entropy(action_counts)
        )

    return metrics


def visualize_market_dynamics(metrics: dict):
    """Визуализация динамики рынка."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Волатильность во времени
    ax1 = axes[0, 0]
    ax1.plot(metrics['price_volatility'])
    ax1.set_title('Волатильность цен по эпизодам')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Стандартное отклонение')

    # 2. Распределение прибыли агентов
    ax2 = axes[0, 1]
    for agent_id, profits in metrics['agent_profits'].items():
        ax2.hist(profits, alpha=0.5, label=f'Agent {agent_id}')
    ax2.set_title('Распределение прибыли по агентам')
    ax2.legend()

    # 3. Эффективность рынка
    ax3 = axes[1, 0]
    ax3.plot(metrics['market_efficiency'])
    ax3.set_title('Эффективность рынка')
    ax3.set_xlabel('Эпизод')
    ax3.set_ylabel('Корреляция с fundamental value')

    # 4. Разнообразие стратегий
    ax4 = axes[1, 1]
    ax4.plot(metrics['strategy_diversity'])
    ax4.set_title('Разнообразие стратегий (энтропия)')
    ax4.set_xlabel('Эпизод')

    plt.tight_layout()
    return fig
```

## Ключевые Метрики

### Метрики Отдельного Агента
| Метрика | Описание |
|---------|----------|
| **Cumulative PnL** | Суммарная прибыль/убыток |
| **Sharpe Ratio** | Доходность с поправкой на риск |
| **Win Rate vs Opponents** | Процент побед против разных типов противников |
| **Exploitability** | Насколько агент уязвим к эксплуатации |

### Метрики Популяции
| Метрика | Описание |
|---------|----------|
| **Strategy Diversity** | Энтропия распределения стратегий |
| **Nash Convergence** | Близость к равновесию Нэша |
| **Regret** | Сожаление относительно лучшей стратегии в ретроспективе |

### Метрики Качества Рынка
| Метрика | Описание |
|---------|----------|
| **Price Efficiency** | Корреляция цены с фундаментальной стоимостью |
| **Volatility Clustering** | Паттерны волатильности |
| **Bid-Ask Spread** | Средний спред (ликвидность) |
| **Price Impact** | Влияние крупных ордеров на цену |

## Зависимости

```python
# Мультиагентные окружения
pettingzoo>=1.24.0

# Reinforcement Learning
stable-baselines3>=2.1.0
gymnasium>=0.29.0

# Deep Learning
torch>=2.0.0

# Численные вычисления
numpy>=1.23.0
pandas>=1.5.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0

# Дополнительно
tqdm>=4.64.0        # Прогресс-бары
tensorboard>=2.12.0 # Логирование
```

## Ожидаемые Результаты

1. **Симулятор рынка** с полноценным order book и несколькими типами агентов
2. **Pipeline обучения MARL** (Independent Learners, MADDPG, Self-Play)
3. **Population-Based Training** с эволюцией гиперпараметров
4. **Анализ эмерджентных стратегий** — какие подходы выживают в конкуренции
5. **Робастный агент** с хорошей производительностью против разнообразных противников

## Rust Реализация

В директории `rust_marl_trading/` находится модульная реализация на Rust с использованием данных криптобиржи Bybit:

```
rust_marl_trading/
├── src/
│   ├── lib.rs                 # Главный модуль
│   ├── api/                   # Bybit API клиент
│   ├── orderbook/             # Order book симуляция
│   ├── agents/                # Типы агентов
│   ├── environment/           # Multi-agent окружение
│   ├── algorithms/            # MARL алгоритмы
│   └── bin/                   # Примеры
└── Cargo.toml
```

## Ссылки

### Научные Статьи
- [Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — Обзор MARL
- [MADDPG: Multi-Agent Actor-Critic](https://arxiv.org/abs/1706.02275) — Алгоритм MADDPG
- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748) — OpenAI, эмерджентное поведение
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — DeepMind, PBT

### Документация и Ресурсы
- [PettingZoo Documentation](https://pettingzoo.farama.org/) — Библиотека мультиагентных сред
- [RLlib Multi-Agent](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) — Ray RLlib

## Уровень Сложности

⭐⭐⭐⭐⭐ (Экспертный)

### Требуемые Знания
- **Reinforcement Learning**: MDP, Q-learning, Policy Gradient, Actor-Critic
- **Теория игр**: Равновесие Нэша, Парето-оптимальность
- **Рыночная микроструктура**: Order book, Market making, Price impact
- **Deep Learning**: Нейронные сети, оптимизация
- **Мультиагентные системы**: Кооперация, конкуренция, emergence
