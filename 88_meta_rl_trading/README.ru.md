# Глава 88: Мета-обучение с подкреплением (Meta-RL) для трейдинга

## Обзор

Мета-обучение с подкреплением (Meta-RL) объединяет мета-обучение с обучением с подкреплением для создания торговых агентов, способных быстро адаптироваться к новым рыночным условиям. В отличие от стандартного RL, который требует обширного переобучения при изменении рыночных условий, Meta-RL обучает сам алгоритм обучения - агента, который может быстро находить эффективные торговые стратегии в новых рыночных режимах с минимальным взаимодействием.

Эта глава реализует Meta-RL для адаптивных торговых агентов с использованием фреймворка RL^2 (Learning to Reinforcement Learn), где RNN-агент обучается на распределении торговых окружений и учится адаптировать поведение через своё скрытое состояние.

## Содержание

1. [Введение в Meta-RL](#введение-в-meta-rl)
2. [Математические основы](#математические-основы)
3. [Meta-RL vs стандартное RL и мета-обучение](#meta-rl-vs-стандартное-rl-и-мета-обучение)
4. [Meta-RL для торговых приложений](#meta-rl-для-торговых-приложений)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Направления развития](#направления-развития)

---

## Введение в Meta-RL

### Что такое мета-обучение с подкреплением?

Мета-обучение с подкреплением находится на пересечении двух мощных парадигм:

- **Мета-обучение**: Обучение тому, как эффективно учиться на распределении задач
- **Обучение с подкреплением**: Обучение принятию последовательных решений через взаимодействие с окружением

Meta-RL создает агента, который при помещении в новое окружение может быстро находить эффективную стратегию без обширного переобучения. "Алгоритм обучения" агента закодирован в его рекуррентном скрытом состоянии.

### Основные подходы к Meta-RL

Существуют три основных семейства алгоритмов Meta-RL:

1. **Рекуррентный Meta-RL (RL^2)**: Использует RNN, скрытое состояние которой кодирует алгоритм обучения. Агент учится адаптироваться, обрабатывая последовательности (состояние, действие, награда, завершение) между эпизодами.

2. **Градиентный Meta-RL (MAML-RL)**: Применяет MAML к градиентам политики RL, обучая инициализацию, которая быстро адаптируется за несколько шагов градиента.

3. **Контекстный Meta-RL**: Обучает скрытую контекстную переменную, которая захватывает информацию о задаче и обусловливает политику.

Эта глава фокусируется на подходе RL^2 с элементами контекстных методов, поскольку они наиболее практичны для торговых приложений.

### Почему Meta-RL для трейдинга?

Трейдинг представляет вызовы, которые делают Meta-RL особенно привлекательным:

- **Нестационарные окружения**: Рыночная динамика постоянно меняется, делая фиксированные стратегии неэффективными
- **Смена режимов**: Бычьи/медвежьи рынки, режимы волатильности требуют быстрой адаптации политики
- **Мультирыночное развертывание**: Единый агент должен работать с различными активами
- **Эффективность выборки**: Реальные торговые данные ограничены; агент должен быстро учиться
- **Исследование vs использование**: Агент должен балансировать сбор рыночной информации с исполнением прибыльных сделок

---

## Математические основы

### Целевая функция Meta-RL

В Meta-RL мы оптимизируем на распределении MDP (Марковских процессов принятия решений):

**Стандартная цель RL:**
```
max_π E_τ~π [Σ_t γ^t r_t]
```

**Цель Meta-RL:**
```
max_θ E_{M~p(M)} E_{τ~π_θ(M)} [Σ_t γ^t r_t]
```

Где:
- θ: Мета-обученные параметры
- M: Конкретный MDP (рыночное окружение), выбранный из распределения p(M)
- π_θ: Политика, параметризованная θ
- γ: Коэффициент дисконтирования
- r_t: Награда в момент t

### RL^2: Обучение обучению с подкреплением

В фреймворке RL^2 (Duan и др., 2016; Wang и др., 2016) агент представляет собой RNN, которая получает предыдущее действие, награду и сигнал завершения как дополнительные входы:

**Вход на каждом шаге:**
```
x_t = [s_t, a_{t-1}, r_{t-1}, d_{t-1}]
```

**Обновление скрытого состояния (GRU):**
```
h_t = GRU(x_t, h_{t-1})
```

**Выходы политики и ценности:**
```
π(a_t | s_t, h_t) = softmax(W_π h_t + b_π)
V(s_t, h_t) = W_v h_t + b_v
```

Ключевой инсайт: скрытое состояние h_t неявно кодирует алгоритм обучения. За несколько эпизодов в одном окружении скрытое состояние накапливает информацию о задаче и адаптирует политику.

### Обучение с PPO

Мета-политика обучается с помощью Proximal Policy Optimization (PPO) на выбранных окружениях:

**Обрезанная цель PPO:**
```
L^CLIP(θ) = E_t [min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

Где:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) - отношение вероятностей
- A_t - оценка преимущества (вычисляется через GAE)
- ε - параметр обрезки

### Обобщенная оценка преимущества (GAE)

```
A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

---

## Meta-RL vs стандартное RL и мета-обучение

### Сравнительная таблица

| Аспект | Стандартное RL | Мета-обучение (MAML) | Meta-RL |
|--------|---------------|---------------------|---------|
| Адаптация | Нет (фиксированная политика) | Несколько градиентных шагов | Внутри эпизода (скрытое состояние) |
| Определение задачи | Единый MDP | Задачи с учителем | Распределение MDP |
| Оптимизация | Градиент политики / Q-обучение | Двухуровневая оптимизация | Градиент политики по распределению MDP |
| Скорость адаптации | Требует переобучения | 3-5 градиентных шагов | Мгновенная (без градиентных шагов) |
| Исследование | Фиксированная стратегия | Н/Д | Обученное исследование |
| Последовательные решения | Да | Нет | Да |

### Когда использовать Meta-RL

**Используйте Meta-RL когда:**
- Окружение часто меняется и политики должны адаптироваться в реальном времени
- Нужен агент, который разумно исследует новые окружения
- Важно последовательное принятие решений (размер позиции, тайминг входа/выхода)
- Нужен единый агент для множества рыночных режимов

**Рассмотрите альтернативы когда:**
- Рыночные условия относительно стабильны (используйте стандартное RL)
- Нужны только точечные предсказания (используйте MAML)
- Вычислительные ресурсы сильно ограничены

---

## Meta-RL для торговых приложений

### 1. Режимно-адаптивный торговый агент

Агент встречает различные рыночные режимы как отдельные MDP:

```
MDPs = {Bull_Market_MDP, Bear_Market_MDP, High_Volatility_MDP, Sideways_MDP}
Состояние: [ценовые_признаки, технические_индикаторы, позиция, стоимость_портфеля]
Действия: {Покупка, Продажа, Удержание}
Награда: риск-скорректированная доходность (на основе Sharpe)
```

### 2. Мультиактивный Meta-RL агент

Каждый актив определяет отдельный MDP:

```
MDPs = {BTCUSDT_MDP, ETHUSDT_MDP, SOLUSDT_MDP, ...}
Цель: Единый агент, адаптирующийся к динамике любого актива за несколько эпизодов
```

### 3. Кросс-таймфреймная адаптация

Разные таймфреймы как разные MDP:

```
MDPs = {1min_MDP, 5min_MDP, 1hour_MDP, Daily_MDP}
Цель: Агент изучает временные паттерны, переносимые между таймфреймами
```

### 4. Адаптивное управление позицией

Meta-RL для динамического управления позициями:

```
Состояние: [рыночные_признаки, текущая_позиция, нереализованный_PnL, волатильность]
Действия: {Увеличить, Уменьшить, Поддержать, Закрыть}
Цель: Изучить правила управления позицией, адаптирующиеся к условиям риска
```

---

## Реализация на Python

### Основной Meta-RL агент (RL^2 с GRU)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class MetaRLAgent(nn.Module):
    """
    RL^2 агент мета-обучения с подкреплением для трейдинга.

    Использует GRU для кодирования алгоритма обучения в скрытом состоянии.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Вход: состояние + предыдущее действие (one-hot) + предыдущая награда + done
        input_size = state_dim + action_dim + 2

        # Кодировщик признаков
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Рекуррентное ядро (GRU для алгоритма обучения)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Голова политики
        self.policy_head = nn.Linear(hidden_size, action_dim)

        # Голова ценности
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        done: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Прямой проход Meta-RL агента."""
        x = torch.cat([state, prev_action, prev_reward, done], dim=-1)
        x = self.encoder(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        action_logits = self.policy_head(gru_out)
        value = self.value_head(gru_out)

        return action_logits, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния нулями."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

    def get_action(
        self,
        state: np.ndarray,
        prev_action: int,
        prev_reward: float,
        done: bool,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Выбор действия с использованием текущей политики."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            prev_action_t = torch.zeros(1, self.action_dim)
            prev_action_t[0, prev_action] = 1.0
            prev_reward_t = torch.FloatTensor([[prev_reward]])
            done_t = torch.FloatTensor([[float(done)]])

            logits, value, new_hidden = self.forward(
                state_t, prev_action_t, prev_reward_t, done_t, hidden
            )

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), new_hidden
```

### Торговое окружение для Meta-RL

```python
class TradingEnvironment:
    """
    Торговое окружение, действующее как MDP для обучения Meta-RL.

    Каждый экземпляр окружения представляет конкретный рыночный режим
    или актив, обеспечивая различный MDP для мета-обучения.
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        max_steps: int = 200
    ):
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.state_dim = features.shape[1] + 3
        self.action_dim = 3  # Покупка, Продажа, Удержание
        self.reset()

    def reset(self) -> np.ndarray:
        """Сброс окружения в начальное состояние."""
        self.step_count = 0
        self.position = 0
        self.capital = self.initial_capital
        self.entry_price = 0.0
        self.total_pnl = 0.0
        max_start = len(self.prices) - self.max_steps - 1
        self.start_idx = np.random.randint(0, max(1, max_start))
        self.current_idx = self.start_idx
        return self._get_state()

    def step(self, action: int):
        """Выполнение действия в окружении."""
        current_price = self.prices[self.current_idx]
        prev_position = self.position

        if action == 0 and self.position <= 0:  # Покупка
            if self.position == -1:
                pnl = (self.entry_price - current_price) / self.entry_price
                self.total_pnl += pnl
                self.capital *= (1 + pnl - self.transaction_cost)
            self.position = 1
            self.entry_price = current_price
            self.capital *= (1 - self.transaction_cost)

        elif action == 1 and self.position >= 0:  # Продажа
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
                self.total_pnl += pnl
                self.capital *= (1 + pnl - self.transaction_cost)
            self.position = -1
            self.entry_price = current_price
            self.capital *= (1 - self.transaction_cost)

        self.current_idx += 1
        self.step_count += 1

        done = (self.step_count >= self.max_steps or
                self.current_idx >= len(self.prices) - 1)

        next_price = self.prices[self.current_idx]
        step_return = 0.0
        if self.position == 1:
            step_return = (next_price - current_price) / current_price
        elif self.position == -1:
            step_return = (current_price - next_price) / current_price

        transaction_penalty = self.transaction_cost if prev_position != self.position else 0.0
        reward = step_return - transaction_penalty

        return self._get_state(), reward, done, {
            'capital': self.capital,
            'position': self.position,
            'total_pnl': self.total_pnl
        }

    def _get_state(self) -> np.ndarray:
        """Формирование вектора состояния."""
        if self.current_idx >= len(self.features):
            market_features = np.zeros(self.features.shape[1])
        else:
            market_features = self.features[self.current_idx]

        state = np.concatenate([
            market_features,
            [float(self.position), self.total_pnl, self.step_count / self.max_steps]
        ])
        return state.astype(np.float32)
```

---

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительный Meta-RL агент для продакшен торговых систем.

### Структура проекта

```
88_meta_rl_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── agent/
│   │   ├── mod.rs
│   │   └── meta_rl.rs
│   ├── env/
│   │   ├── mod.rs
│   │   └── trading_env.rs
│   ├── trainer/
│   │   ├── mod.rs
│   │   └── ppo.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs
├── examples/
│   ├── basic_meta_rl.rs
│   ├── multi_asset_training.rs
│   └── trading_strategy.rs
└── python/
    ├── __init__.py
    ├── meta_rl_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Основная реализация на Rust

Смотрите директорию `src/` для полной реализации на Rust с:

- GRU-агентом для рекуррентного мета-RL
- Торговым окружением с реалистичной симуляцией рынка
- Циклом обучения PPO для мета-обучения на распределении окружений
- Асинхронной интеграцией с API Bybit для криптовалютных данных
- Продакшен-готовой обработкой ошибок и логированием
- Бэктестинговым движком с комплексными метриками производительности

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Мульти-режимное мета-обучение

```python
import yfinance as yf

# Загрузка данных для нескольких активов (разные MDP)
assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Создание окружений для каждого актива
environments = []
for name, df in assets.items():
    prices = df['Close'].values
    features = create_trading_features(prices)
    prices_aligned = prices[len(prices) - len(features):]
    env = TradingEnvironment(prices=prices_aligned, features=features)
    environments.append(env)

# Инициализация Meta-RL агента
agent = MetaRLAgent(state_dim=14, action_dim=3, hidden_size=128)
trainer = PPOMetaTrainer(agent=agent, lr=3e-4, num_episodes_per_trial=3)

# Мета-обучение
for epoch in range(500):
    metrics = trainer.meta_train_step(environments)
    if epoch % 50 == 0:
        print(f"Эпоха {epoch}: потеря={metrics['total_loss']:.4f}")
```

### Пример 2: Адаптация к новому активу

```python
# Новый актив, не виденный во время обучения
new_asset = yf.download('TSLA', period='1y')
new_env = TradingEnvironment(
    prices=new_prices, features=new_features
)

# Агент адаптируется через скрытое состояние за 3 эпизода
hidden = None
for episode in range(3):
    state = new_env.reset()
    done = False
    total_reward = 0.0
    prev_action, prev_reward, prev_done = 0, 0.0, False

    while not done:
        action, _, _, hidden = agent.get_action(
            state, prev_action, prev_reward, prev_done, hidden
        )
        state, reward, done, info = new_env.step(action)
        total_reward += reward
        prev_action, prev_reward, prev_done = action, reward, done

    print(f"Эпизод {episode + 1}: награда={total_reward:.4f}, "
          f"капитал={info['capital']:.2f}")
```

### Пример 3: Торговля криптовалютами на Bybit

```python
# Получение данных для нескольких криптовалютных пар
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_envs = []

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol, interval='60', limit=1000)
    prices = df['close'].values
    features = create_trading_features(prices)
    prices_aligned = prices[len(prices) - len(features):]
    env = TradingEnvironment(prices=prices_aligned, features=features)
    crypto_envs.append(env)

# Мета-обучение на криптовалютных окружениях
for epoch in range(300):
    metrics = trainer.meta_train_step(crypto_envs)
    if epoch % 30 == 0:
        print(f"Эпоха {epoch}: потеря={metrics['total_loss']:.4f}")
```

---

## Оценка производительности

### Целевые показатели производительности

| Метрика | Целевой диапазон |
|---------|-----------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Максимальная просадка | < 20% |
| Win Rate | > 50% |
| Скорость адаптации | < 3 эпизода |

### Meta-RL vs Baseline

В типичных экспериментах Meta-RL показывает:
- **Мгновенную адаптацию** к новым окружениям (без градиентных шагов)
- **Обученное исследование**, эффективно собирающее рыночную информацию
- **Улучшение Sharpe ratio на 20-40%** по сравнению с фиксированными RL-политиками
- **Лучшую обработку смены режимов** по сравнению со стандартными RL-агентами

### Преимущества перед MAML-подходами

| Особенность | MAML | Meta-RL (RL^2) |
|-------------|------|----------------|
| Требует градиентных шагов для адаптации | Да | Нет |
| Обрабатывает последовательные решения нативно | Нет | Да |
| Обучает стратегию исследования | Нет | Да |
| Вычислительные затраты при адаптации | Средние | Низкие (один прямой проход) |
| Обрабатывает частичную наблюдаемость | Нет | Да (через скрытое состояние) |

---

## Направления развития

### 1. Meta-RL на основе Transformer

Замена GRU на Transformer для лучшего захвата дальних зависимостей.

### 2. Сети вывода задач

Явный вывод эмбеддинга задачи для обусловливания политики.

### 3. Иерархический Meta-RL

Многоуровневое принятие решений:
- Уровень 1: Мета-политика выбирает стратегии
- Уровень 2: Под-политики выполняют конкретные стратегии
- Уровень 3: Управление размером позиции и рисками

### 4. Оффлайн Meta-RL

Обучение на записанных рыночных данных без онлайн-взаимодействия.

### 5. Мульти-агентный Meta-RL

Несколько агентов, адаптирующихся к стратегиям друг друга на конкурентных рынках.

---

## Ссылки

1. Duan, Y., Schulman, J., et al. (2016). RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning. [arXiv:1611.02779](https://arxiv.org/abs/1611.02779)

2. Wang, J. X., et al. (2016). Learning to Reinforcement Learn. [arXiv:1611.05763](https://arxiv.org/abs/1611.05763)

3. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

4. Rakelly, K., et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML. [arXiv:1903.08254](https://arxiv.org/abs/1903.08254)

5. Beck, J., et al. (2023). A Survey of Meta-Reinforcement Learning. [arXiv:2301.08028](https://arxiv.org/abs/2301.08028)

---

## Запуск примеров

### Python

```bash
cd 88_meta_rl_trading
pip install -r python/requirements.txt
python python/meta_rl_trader.py
```

### Rust

```bash
cd 88_meta_rl_trading
cargo build --release
cargo test
cargo run --example basic_meta_rl
cargo run --example multi_asset_training
cargo run --example trading_strategy
```

---

## Резюме

Мета-обучение с подкреплением предоставляет мощный фреймворк для адаптивного трейдинга:

- **Интернализация алгоритма обучения**: Скрытое состояние агента кодирует полный алгоритм обучения
- **Адаптация без градиентов**: Не нужны градиентные шаги при развертывании; адаптация происходит через прямые проходы
- **Обученное исследование**: Агент учится эффективно собирать информацию на новых рынках
- **Последовательное принятие решений**: Естественно обрабатывает временные аспекты торговых решений

Обучаясь на разнообразных рыночных окружениях, агенты Meta-RL развивают способность быстро определять рыночную динамику и адаптировать торговое поведение - критически важная способность для робастных алгоритмических торговых систем.

---

*Предыдущая глава: [Глава 87: Task-Agnostic Meta-Learning](../87_task_agnostic_meta_learning)*

*Следующая глава: [Глава 89: Meta-SGD для трейдинга](../89_meta_sgd_trading)*
