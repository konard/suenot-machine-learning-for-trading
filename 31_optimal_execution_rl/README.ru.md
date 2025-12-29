# Глава 31: Оптимальное исполнение ордеров с обучением с подкреплением — Превосходя TWAP/VWAP

## Обзор

При исполнении крупных ордеров возникает компромисс между **воздействием на рынок** (market impact, при быстром исполнении) и **временным риском** (timing risk, при медленном исполнении). Классические алгоритмы TWAP (Time-Weighted Average Price) и VWAP (Volume-Weighted Average Price) используют детерминистические расписания. В этой главе мы обучаем RL-агента, который **адаптивно выбирает оптимальную скорость исполнения** в зависимости от текущих рыночных условий.

### Зачем это нужно?

Представьте, что вам нужно продать 10 000 BTC. Если продать всё сразу:
- Цена упадёт из-за огромного давления продавцов (market impact)
- Вы получите намного меньше ожидаемого

Если продавать очень медленно:
- Цена может измениться за время исполнения (timing risk)
- Вы рискуете продать по невыгодной цене

**Задача:** найти оптимальный баланс между этими рисками.

---

## Торговая стратегия

### Суть стратегии

RL-агент решает, **сколько актива исполнить в каждый момент времени**, минимизируя **implementation shortfall** — разницу между ценой принятия решения (decision price) и фактической ценой исполнения (execution price).

### Компоненты MDP (Марковский процесс принятия решений)

**Состояние (State):**
- Состояние книги ордеров (спред, глубина)
- Оставшееся количество для исполнения
- Оставшееся время
- Недавние движения цены
- Волатильность
- Дисбаланс ордеров

**Действие (Action):**
- Доля оставшегося объёма для исполнения в текущем интервале (от 0% до 100%)

**Вознаграждение (Reward):**
- Отрицательный implementation shortfall минус штраф за риск

### Преимущество над классическими методами

**Edge:** Адаптация к текущим рыночным условиям вместо статичного расписания. Агент учится:
- Ускоряться при благоприятных условиях (высокая ликвидность, узкий спред)
- Замедляться при неблагоприятных (низкая ликвидность, высокая волатильность)

---

## Техническая спецификация

### Ноутбуки

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_execution_theory.ipynb` | Модель Almgren-Chriss, implementation shortfall |
| 2 | `02_market_impact_models.ipynb` | Линейная, квадратного корня, временная модели impact |
| 3 | `03_twap_vwap_baselines.ipynb` | Реализация классических алгоритмов TWAP/VWAP |
| 4 | `04_gym_environment.ipynb` | Кастомная Gym-среда для execution |
| 5 | `05_reward_function.ipynb` | Дизайн reward: shortfall + штраф за риск |
| 6 | `06_dqn_agent.ipynb` | Deep Q-Network для дискретных действий |
| 7 | `07_ppo_agent.ipynb` | PPO для непрерывного пространства действий |
| 8 | `08_training_curriculum.ipynb` | Curriculum learning: от простых к сложным сценариям |
| 9 | `09_evaluation.ipynb` | Сравнение с TWAP/VWAP/Almgren-Chriss |
| 10 | `10_robustness.ipynb` | Out-of-sample тесты, разные рыночные условия |

### Требования к данным

```
Данные книги ордеров:
├── L2 снимки (частота 1 секунда)
├── Отпечатки сделок с размерами
├── Минимум 6 месяцев для обучения
└── Ликвидные инструменты (BTC, ETH на Bybit)

Сценарии исполнения:
├── Размеры ордеров: 1%, 5%, 10% от ADV (Average Daily Volume)
├── Горизонты времени: 5 мин, 30 мин, 1 час, 1 день
├── Рыночные условия: нормальные, волатильные, трендовые
└── Симулированные ордера с различными уровнями срочности
```

---

## Модели воздействия на рынок (Market Impact)

### Временное воздействие (Temporary Impact)

Влияет только на текущую сделку. Возникает из-за пересечения спреда и потребления ликвидности.

```python
def temporary_impact(quantity, params):
    """
    Линейная модель временного воздействия
    """
    return params['eta'] * quantity / params['adv']
```

### Постоянное воздействие (Permanent Impact)

Сдвигает цену для всех будущих сделок. Отражает информационное содержание сделки.

```python
def permanent_impact(quantity, params):
    """
    Модель квадратного корня (более реалистичная)
    """
    return params['gamma'] * np.sign(quantity) * np.sqrt(abs(quantity) / params['adv'])
```

### Полная стоимость исполнения

```python
def total_cost(execution_schedule, params):
    """
    Суммарная стоимость = сумма (временного + постоянного) воздействия
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

---

## Дизайн среды (Gym Environment)

```python
class ExecutionEnv(gym.Env):
    """
    Среда для оптимального исполнения ордеров
    """
    def __init__(self, order_size, time_horizon, market_data):
        self.total_shares = order_size
        self.remaining_shares = order_size
        self.time_horizon = time_horizon
        self.current_step = 0

        # Действие: доля оставшегося для исполнения (от 0 до 1)
        self.action_space = gym.spaces.Box(0, 1, shape=(1,))

        # Состояние: [remaining_frac, time_frac, spread, volatility, imbalance, momentum]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,))

    def step(self, action):
        execute_qty = action * self.remaining_shares
        execution_price = self._simulate_execution(execute_qty)

        # Implementation shortfall
        shortfall = (execution_price - self.arrival_price) * execute_qty

        # Штраф за риск (дисперсия оставшегося)
        risk_penalty = self.lambda_risk * self.remaining_shares * self.volatility

        reward = -shortfall - risk_penalty

        self.remaining_shares -= execute_qty
        self.current_step += 1

        done = (self.remaining_shares <= 0) or (self.current_step >= self.max_steps)

        return self._get_state(), reward, done, {}
```

---

## Алгоритмы RL

### DQN (Дискретные действия)

```
DQN:
├── Действия: [0%, 10%, 20%, ..., 100%] от оставшегося
├── Сеть: Состояние → Q-значения для каждого действия
├── Experience replay, target network
└── Плюсы: интерпретируемость, простота обучения
```

### PPO (Непрерывные действия)

```
PPO:
├── Действия: непрерывное [0, 1] — доля исполнения
├── Архитектура Actor-Critic
├── Clipped objective для стабильности
└── Плюсы: точный контроль
```

### SAC (Soft Actor-Critic)

```
SAC:
├── Энтропийная регуляризация
├── Лучшее исследование
└── Плюсы: для сложной динамики рынка
```

---

## Базовый алгоритм: Almgren-Chriss

Аналитическое решение для оптимальной траектории исполнения:

```python
def almgren_chriss_optimal(total_shares, time_horizon, risk_aversion, volatility, impact_params):
    """
    Оптимальная траектория исполнения в закрытой форме.
    Балансирует стоимость воздействия и временной риск.
    """
    kappa = np.sqrt(risk_aversion * volatility**2 / impact_params['eta'])

    trajectory = []
    for t in range(time_horizon):
        remaining_time = time_horizon - t
        optimal_trade = total_shares * np.sinh(kappa * remaining_time) / np.sinh(kappa * time_horizon)
        trajectory.append(optimal_trade)

    return trajectory
```

**Интуиция:**
- Высокая неприятие риска → быстрое исполнение (избегаем timing risk)
- Низкая неприятие риска → медленное исполнение (минимизируем market impact)

---

## Ключевые метрики

### Качество исполнения
- **Implementation Shortfall:** разница между ценой решения и фактической ценой
- **Arrival Price Slippage:** скольжение относительно цены при входе

### Сравнение с бенчмарками
- vs TWAP (равномерное по времени)
- vs VWAP (взвешенное по объёму)
- vs Almgren-Chriss (оптимальное аналитическое)

### Риск-скорректированные метрики
- Sharpe ratio распределения стоимости исполнения
- Consistency (стабильность результатов)

### Робастность
- Производительность в разных режимах волатильности
- Out-of-sample тестирование

---

## Зависимости

```python
gymnasium>=0.29.0         # Среда для RL
stable-baselines3>=2.1.0  # Реализации DQN, PPO, SAC
torch>=2.0.0              # Нейронные сети
pandas>=1.5.0             # Работа с данными
numpy>=1.23.0             # Численные вычисления
matplotlib>=3.6.0         # Визуализация
```

---

## Ожидаемые результаты

1. **Кастомная Gym-среда** для optimal execution
2. **Модели market impact** (линейная, квадратного корня, временная)
3. **RL-агенты** (DQN, PPO) обученные на исполнении
4. **Фреймворк сравнения** с классическими алгоритмами
5. **Результаты:** 5-15% улучшение vs TWAP в implementation shortfall

---

## Rust реализация

Для высокопроизводительной работы с данными Bybit в этой главе также представлена реализация на Rust:

```
rust_optimal_execution/
├── src/
│   ├── api/           # Клиент Bybit API
│   ├── impact/        # Модели market impact
│   ├── environment/   # Execution среда
│   ├── agent/         # RL агенты
│   ├── baselines/     # TWAP/VWAP/Almgren-Chriss
│   └── utils/         # Утилиты и метрики
├── examples/          # Примеры использования
└── data/              # Данные (автосоздаётся)
```

---

## Литература

- [Optimal Execution of Portfolio Transactions](https://www.math.nyu.edu/~almgren/papers/optliq.pdf) (Almgren-Chriss, 2000) — Основополагающая работа по optimal execution
- [Optimal Execution with Reinforcement Learning](https://arxiv.org/abs/1906.02312) — RL подход к исполнению
- [Deep Reinforcement Learning for Optimal Execution](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3304766) — Глубокий RL для исполнения

---

## Уровень сложности

⭐⭐⭐⭐☆ (Продвинутый)

**Требуется понимание:**
- Обучение с подкреплением (Q-learning, Policy Gradient)
- Микроструктура рынка (книга ордеров, ликвидность)
- Алгоритмы исполнения (TWAP, VWAP, Almgren-Chriss)

---

## Практическое применение

### Когда использовать

- Исполнение крупных ордеров (>1% дневного объёма)
- Алгоритмическая торговля на криптовалютных биржах
- Маркет-мейкинг с управлением инвентарём
- Ребалансировка портфеля

### Риски и ограничения

- Модель обучена на исторических данных (may not generalize)
- Market impact модели — упрощение реальности
- Требуется регулярная переобучение
- Не учитывает редкие события (flash crash)

---

*"Оптимальное исполнение — это искусство быть невидимым на рынке."*
