# Глава 38: Neural ODE — Непрерывная динамика для плавного ребалансирования портфеля

## Обзор

**Neural ODE (Neural Ordinary Differential Equations)** — это революционный подход к моделированию непрерывной динамики систем. В отличие от дискретных моделей (RNN, LSTM, Transformer), которые обрабатывают данные в дискретные моменты времени, Neural ODE моделирует эволюцию системы как **непрерывный поток**.

<p align="center">
<img src="https://i.imgur.com/JQxKb8H.png" width="70%">
</p>

### Ключевая идея

Вместо того чтобы описывать состояние системы на каждом временном шаге $t_0, t_1, t_2, ...$, Neural ODE описывает **скорость изменения** состояния:

$$\frac{dz}{dt} = f_\theta(z(t), t)$$

где:
- $z(t)$ — состояние системы в момент времени $t$ (например, веса портфеля)
- $f_\theta$ — нейронная сеть с параметрами $\theta$, описывающая динамику
- $\frac{dz}{dt}$ — скорость изменения состояния

### Применение в трейдинге

**Стратегия ребалансирования:** Neural ODE моделирует оптимальную траекторию весов портфеля, позволяя:
- Плавно переходить от текущего распределения к целевому
- Минимизировать транзакционные издержки
- Адаптироваться к изменениям рыночных условий в реальном времени

## Содержание

1. [Теоретические основы Neural ODE](#теоретические-основы-neural-ode)
   - [Обыкновенные дифференциальные уравнения](#обыкновенные-дифференциальные-уравнения)
   - [Нейронные сети как ODE](#нейронные-сети-как-ode)
   - [Adjoint метод для обучения](#adjoint-метод-для-обучения)
2. [Архитектура Neural ODE для портфеля](#архитектура-neural-ode-для-портфеля)
   - [Кодирование начального состояния](#кодирование-начального-состояния)
   - [Динамика весов портфеля](#динамика-весов-портфеля)
   - [Ограничения и регуляризация](#ограничения-и-регуляризация)
3. [Непрерывное оптимальное управление](#непрерывное-оптимальное-управление)
   - [Hamilton-Jacobi-Bellman уравнение](#hamilton-jacobi-bellman-уравнение)
   - [Учёт транзакционных издержек](#учёт-транзакционных-издержек)
4. [Примеры кода](#примеры-кода)
5. [Реализация на Rust](#реализация-на-rust)
6. [Практические соображения](#практические-соображения)
7. [Ресурсы](#ресурсы)

---

## Теоретические основы Neural ODE

### Обыкновенные дифференциальные уравнения

**ODE (Ordinary Differential Equation)** — уравнение, связывающее функцию с её производными. Классический пример — экспоненциальный рост:

$$\frac{dy}{dt} = ky \implies y(t) = y_0 e^{kt}$$

#### Численные методы решения

| Метод | Формула | Точность | Скорость |
|-------|---------|----------|----------|
| **Euler** | $z_{t+1} = z_t + h \cdot f(z_t, t)$ | O(h) | Очень быстро |
| **RK4** | 4-х этапная схема | O(h⁴) | Средне |
| **Dopri5** | Адаптивный шаг | O(h⁵) | Адаптивно |

**Dopri5** (Dormand-Prince) — наиболее популярный метод для Neural ODE, так как автоматически подбирает шаг интегрирования.

### Нейронные сети как ODE

Ключевое наблюдение: **ResNet можно рассматривать как дискретизацию ODE**.

ResNet block:
$$z_{t+1} = z_t + f_\theta(z_t)$$

Это метод Эйлера с шагом $h=1$ для ODE:
$$\frac{dz}{dt} = f_\theta(z)$$

**Преимущество:** Вместо фиксированного числа слоёв мы имеем непрерывную глубину!

```python
# Дискретный ResNet
for layer in layers:
    x = x + layer(x)

# Непрерывный Neural ODE
x = odeint(neural_net, x0, t, method='dopri5')
```

### Adjoint метод для обучения

Классическое обратное распространение требует хранить все промежуточные состояния. Для ODE с 1000 шагами это **огромный расход памяти**.

**Adjoint метод** решает эту проблему, вычисляя градиенты через обратное интегрирование:

$$\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f}{\partial \theta} dt$$

где $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$ — сопряжённое состояние.

**Преимущества:**
- Память O(1) вместо O(T)
- Точные градиенты (не приближение)
- Работает с адаптивными солверами

---

## Архитектура Neural ODE для портфеля

### Общая схема

```
┌─────────────────────────────────────────────────────────────────┐
│                     Neural ODE Portfolio                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Encoder    │────▶│   ODE Func   │────▶│   Decoder    │   │
│  │              │     │   dz/dt=f(z) │     │              │   │
│  │ Market Data  │     │              │     │  Portfolio   │   │
│  │ + Features   │     │   odeint()   │     │   Weights    │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│    [N×features]         [trajectory]        [N×assets]        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Кодирование начального состояния

```python
class PortfolioEncoder(nn.Module):
    """
    Кодирует рыночные данные и текущее состояние портфеля
    в латентное пространство для ODE
    """
    def __init__(self, n_assets, n_features, hidden_dim=64):
        super().__init__()
        self.n_assets = n_assets

        # Обработка рыночных признаков
        self.feature_net = nn.Sequential(
            nn.Linear(n_features * n_assets, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Обработка текущих весов
        self.weight_net = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.SiLU()
        )

        # Объединение
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, weights, features):
        """
        weights: [batch, n_assets] - текущие веса портфеля
        features: [batch, n_assets, n_features] - рыночные признаки
        """
        # Flatten features
        feat_flat = features.flatten(start_dim=1)

        # Encode
        feat_enc = self.feature_net(feat_flat)
        weight_enc = self.weight_net(weights)

        # Combine
        combined = torch.cat([feat_enc, weight_enc], dim=-1)
        z0 = self.combine(combined)

        return z0
```

### Динамика весов портфеля

```python
class PortfolioODEFunc(nn.Module):
    """
    Определяет динамику dw/dt = f(w, t, context)

    Ключевые особенности:
    - Веса всегда суммируются к 1 (симплекс)
    - Штраф за слишком быстрые изменения (транзакционные издержки)
    - Учёт целевого распределения
    """
    def __init__(self, hidden_dim=64, n_assets=5):
        super().__init__()
        self.n_assets = n_assets

        # Основная сеть динамики
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 для времени
            nn.Tanh(),  # Tanh для ограниченных значений
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Регулятор скорости изменения
        self.speed_controller = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # от 0 до 1
        )

    def forward(self, t, z):
        """
        t: scalar - текущее время
        z: [batch, hidden_dim] - текущее латентное состояние
        """
        # Добавляем время как признак
        t_embed = t.expand(z.shape[0], 1)
        z_t = torch.cat([z, t_embed], dim=-1)

        # Вычисляем направление изменения
        drift = self.dynamics(z_t)

        # Контролируем скорость (медленнее к концу интервала)
        speed = self.speed_controller(z)

        # Финальная динамика
        dz_dt = drift * speed

        return dz_dt
```

### Полная модель Neural ODE для портфеля

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

class NeuralODEPortfolio(nn.Module):
    """
    Полная модель для оптимизации портфеля через Neural ODE
    """
    def __init__(self, n_assets, n_features, hidden_dim=64,
                 use_adjoint=True):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.use_adjoint = use_adjoint

        # Компоненты
        self.encoder = PortfolioEncoder(n_assets, n_features, hidden_dim)
        self.ode_func = PortfolioODEFunc(hidden_dim, n_assets)
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        """Декодер: латентное пространство -> веса портфеля"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_assets),
            nn.Softmax(dim=-1)  # Гарантируем sum(weights) = 1
        )

    def forward(self, initial_weights, features, t_span,
                n_steps=50, method='dopri5'):
        """
        initial_weights: [batch, n_assets] - начальные веса
        features: [batch, n_assets, n_features] - рыночные данные
        t_span: (t0, t1) - временной интервал

        Returns:
            trajectory: [n_steps, batch, n_assets] - траектория весов
        """
        # Кодируем начальное состояние
        z0 = self.encoder(initial_weights, features)

        # Точки времени для оценки
        t = torch.linspace(t_span[0], t_span[1], n_steps)
        t = t.to(z0.device)

        # Выбираем ODE солвер
        solver = odeint_adjoint if self.use_adjoint else odeint

        # Решаем ODE
        z_trajectory = solver(
            self.ode_func,
            z0,
            t,
            method=method,
            rtol=1e-4,
            atol=1e-5
        )

        # Декодируем траекторию в веса
        # z_trajectory: [n_steps, batch, hidden_dim]
        weight_trajectory = self.decoder(z_trajectory)

        return weight_trajectory

    def get_target_weights(self, current_weights, features, horizon=1.0):
        """Получить целевые веса через horizon времени"""
        trajectory = self.forward(
            current_weights, features,
            t_span=(0, horizon),
            n_steps=10
        )
        return trajectory[-1]  # Финальные веса
```

---

## Непрерывное оптимальное управление

### Hamilton-Jacobi-Bellman уравнение

Для оптимального управления портфелем используем подход HJB:

$$\frac{\partial V}{\partial t} + \max_u \left[ f(x, u)^T \nabla_x V + L(x, u) \right] = 0$$

где:
- $V(x, t)$ — функция ценности (value function)
- $u$ — управление (скорость изменения весов)
- $L(x, u)$ — мгновенная функция полезности минус издержки

### Учёт транзакционных издержек

```python
class TransactionCostODE(nn.Module):
    """
    ODE динамика с явным учётом транзакционных издержек

    Идея: штрафуем быстрые изменения весов
    """
    def __init__(self, hidden_dim, n_assets,
                 cost_linear=0.001,    # 0.1% за сделку
                 cost_quadratic=0.0005):  # Штраф за большие объёмы
        super().__init__()
        self.n_assets = n_assets
        self.cost_linear = cost_linear
        self.cost_quadratic = cost_quadratic

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + n_assets + 1, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, t, state):
        """
        state содержит: [z, current_weights]
        """
        # Распаковываем состояние
        z = state[..., :-self.n_assets]
        weights = state[..., -self.n_assets:]

        # Время как признак
        t_feat = t.expand(z.shape[0], 1)

        # Вычисляем желаемое изменение
        input_feat = torch.cat([z, weights, t_feat], dim=-1)
        raw_drift = self.net(input_feat)

        # Применяем cost-aware scaling
        # Чем больше изменение, тем больше "сопротивление"
        drift_magnitude = torch.norm(raw_drift, dim=-1, keepdim=True)
        cost_penalty = 1.0 / (1.0 + self.cost_linear * drift_magnitude +
                              self.cost_quadratic * drift_magnitude ** 2)

        adjusted_drift = raw_drift * cost_penalty

        # Drift для весов (из латентного пространства)
        weight_drift = torch.zeros_like(weights)

        return torch.cat([adjusted_drift, weight_drift], dim=-1)


class OptimalControlLoss(nn.Module):
    """
    Функция потерь для обучения Neural ODE портфеля
    """
    def __init__(self, risk_aversion=1.0, cost_weight=0.01):
        super().__init__()
        self.gamma = risk_aversion
        self.cost_weight = cost_weight

    def forward(self, weight_trajectory, returns,
                initial_weights):
        """
        weight_trajectory: [T, batch, n_assets]
        returns: [batch, n_assets] - реализованные доходности
        """
        # Финальные веса
        final_weights = weight_trajectory[-1]

        # 1. Доходность портфеля
        portfolio_return = (final_weights * returns).sum(dim=-1)

        # 2. Риск (variance penalty)
        # Используем отклонение от среднего
        mean_return = portfolio_return.mean()
        risk_penalty = ((portfolio_return - mean_return) ** 2).mean()

        # 3. Транзакционные издержки
        # Сумма всех изменений вдоль траектории
        weight_changes = torch.diff(weight_trajectory, dim=0)
        transaction_costs = torch.abs(weight_changes).sum()

        # 4. Smoothness penalty (плавность траектории)
        second_derivative = torch.diff(weight_changes, dim=0)
        smoothness_penalty = (second_derivative ** 2).sum()

        # Итоговая функция потерь
        loss = (
            -portfolio_return.mean()  # Максимизируем доходность
            + self.gamma * risk_penalty  # Минимизируем риск
            + self.cost_weight * transaction_costs  # Минимизируем издержки
            + 0.01 * smoothness_penalty  # Плавная траектория
        )

        return loss
```

---

## Стратегия ребалансирования

### Непрерывный vs Дискретный ребалансинг

| Подход | Частота | Преимущества | Недостатки |
|--------|---------|--------------|------------|
| **Monthly** | 1 раз в месяц | Низкие издержки | Большие отклонения |
| **Daily** | Каждый день | Точное следование | Высокие издержки |
| **Threshold** | По отклонению | Адаптивно | Может пропустить момент |
| **Neural ODE** | Непрерывно | Оптимальный путь | Сложность модели |

### Алгоритм ребалансирования с Neural ODE

```python
class ContinuousRebalancer:
    """
    Стратегия ребалансирования на основе Neural ODE
    """
    def __init__(self, model, threshold=0.02,
                 min_trade_size=100):
        self.model = model
        self.threshold = threshold
        self.min_trade_size = min_trade_size

    def get_optimal_trajectory(self, current_weights,
                                features, horizon=1.0):
        """
        Получить оптимальную траекторию весов на горизонте
        """
        with torch.no_grad():
            trajectory = self.model(
                current_weights.unsqueeze(0),
                features.unsqueeze(0),
                t_span=(0, horizon),
                n_steps=100
            )
        return trajectory.squeeze(0)

    def should_rebalance(self, current_weights, features):
        """
        Определить, нужно ли ребалансировать
        """
        # Получаем целевые веса через небольшой горизонт
        trajectory = self.get_optimal_trajectory(
            current_weights, features, horizon=0.1
        )
        target_weights = trajectory[-1]

        # Максимальное отклонение
        max_deviation = torch.abs(
            current_weights - target_weights
        ).max().item()

        return max_deviation > self.threshold

    def compute_trades(self, current_weights, features,
                       portfolio_value):
        """
        Вычислить сделки для перехода к целевому распределению

        Returns:
            trades: dict {asset: dollar_amount}
            target_weights: tensor
        """
        trajectory = self.get_optimal_trajectory(
            current_weights, features, horizon=0.5
        )

        # Берём не финальную точку, а оптимальную промежуточную
        # Это позволяет плавно двигаться к цели
        target_weights = trajectory[10]  # 10% пути

        # Вычисляем изменения в долларах
        weight_diff = target_weights - current_weights
        dollar_changes = weight_diff * portfolio_value

        # Фильтруем слишком маленькие сделки
        trades = {}
        for i, change in enumerate(dollar_changes):
            if abs(change.item()) >= self.min_trade_size:
                trades[f'asset_{i}'] = change.item()

        return trades, target_weights

    def backtest(self, historical_weights, historical_features,
                 historical_returns, portfolio_value=100000):
        """
        Бэктест стратегии на исторических данных
        """
        results = {
            'portfolio_values': [portfolio_value],
            'rebalance_dates': [],
            'transaction_costs': [],
            'weights_history': [historical_weights[0]]
        }

        current_weights = historical_weights[0]

        for t in range(1, len(historical_weights)):
            features = historical_features[t]
            returns = historical_returns[t]

            # Обновляем стоимость с учётом доходности
            portfolio_return = (current_weights * returns).sum()
            portfolio_value *= (1 + portfolio_return)

            # Проверяем необходимость ребалансировки
            if self.should_rebalance(current_weights, features):
                trades, new_weights = self.compute_trades(
                    current_weights, features, portfolio_value
                )

                # Транзакционные издержки (0.1% от объёма)
                cost = sum(abs(v) for v in trades.values()) * 0.001
                portfolio_value -= cost

                results['rebalance_dates'].append(t)
                results['transaction_costs'].append(cost)

                current_weights = new_weights

            # Пассивное изменение весов (без ребалансировки)
            else:
                # Веса меняются пропорционально доходности
                current_weights = current_weights * (1 + returns)
                current_weights = current_weights / current_weights.sum()

            results['portfolio_values'].append(portfolio_value)
            results['weights_history'].append(current_weights.clone())

        return results
```

---

## Примеры кода

| Ноутбук | Описание |
|---------|----------|
| [01_ode_theory.ipynb](01_ode_theory.ipynb) | Теория ODE: методы Euler, RK4, adjoint |
| [02_neural_ode_basics.ipynb](02_neural_ode_basics.ipynb) | Основы Neural ODE с torchdiffeq |
| [03_financial_dynamics.ipynb](03_financial_dynamics.ipynb) | Моделирование ценовой динамики как ODE |
| [04_portfolio_ode.ipynb](04_portfolio_ode.ipynb) | Веса портфеля как решение ODE |
| [05_continuous_control.ipynb](05_continuous_control.ipynb) | Непрерывное оптимальное управление |
| [06_transaction_costs.ipynb](06_transaction_costs.ipynb) | Включение транзакционных издержек |
| [07_training.ipynb](07_training.ipynb) | Обучение Neural ODE на исторических данных |
| [08_trajectory_prediction.ipynb](08_trajectory_prediction.ipynb) | Предсказание оптимальной траектории |
| [09_rebalancing_strategy.ipynb](09_rebalancing_strategy.ipynb) | Стратегия ребалансирования |
| [10_backtesting.ipynb](10_backtesting.ipynb) | Сравнение с дискретным ребалансированием |

---

## Реализация на Rust

Директория [rust_neural_ode_crypto](rust_neural_ode_crypto/) содержит модульную Rust-реализацию:

```
rust_neural_ode_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный модуль
│   ├── main.rs             # CLI приложение
│   ├── data/               # Работа с данными
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API клиент
│   │   ├── candles.rs      # Свечные данные
│   │   └── features.rs     # Технические индикаторы
│   ├── ode/                # ODE солверы
│   │   ├── mod.rs
│   │   ├── euler.rs        # Метод Эйлера
│   │   ├── rk4.rs          # Runge-Kutta 4
│   │   └── dopri5.rs       # Dormand-Prince 5(4)
│   ├── model/              # Neural ODE модели
│   │   ├── mod.rs
│   │   ├── network.rs      # Нейронные сети
│   │   ├── portfolio.rs    # Портфельная динамика
│   │   └── training.rs     # Обучение
│   └── strategy/           # Торговые стратегии
│       ├── mod.rs
│       ├── rebalancer.rs   # Ребалансировщик
│       └── backtest.rs     # Бэктестинг
└── examples/
    ├── fetch_bybit_data.rs
    ├── train_portfolio_ode.rs
    └── live_rebalancing.rs
```

Смотрите [rust_neural_ode_crypto/README.md](rust_neural_ode_crypto/README.md) для подробной документации.

---

## Практические соображения

### Когда использовать Neural ODE

**Хорошие сценарии:**
- Оптимизация ребалансирования с минимизацией издержек
- Моделирование непрерывной эволюции портфеля
- Работа с нерегулярными временными рядами
- Когда важна интерпретируемость траектории

**Не идеально для:**
- Ultra-high-frequency trading (латентность ODE солвера)
- Очень короткие горизонты (<1 день)
- Когда достаточно простых правил ребалансирования

### Вычислительные требования

| Компонент | GPU память | Время inference | Время обучения |
|-----------|------------|-----------------|----------------|
| Encoder | 0.5 GB | 1 мс | - |
| ODE Solve (50 steps) | 1 GB | 20 мс | 200 мс/batch |
| Decoder | 0.1 GB | 0.5 мс | - |
| **Итого** | **2 GB** | **~25 мс** | **1-2 часа** |

### Советы по обучению

1. **Начните с простого:** Используйте метод Эйлера для отладки, затем переключитесь на Dopri5
2. **Регуляризация:** Добавьте штраф за сложность траектории
3. **Адаптивный шаг:** Dopri5 автоматически уменьшает шаг при резких изменениях
4. **Adjoint метод:** Используйте для экономии памяти при длинных траекториях

---

## Сравнение с другими подходами

| Метод | Траектория | Издержки | Адаптивность | Сложность |
|-------|------------|----------|--------------|-----------|
| Monthly rebalance | Дискретная | Низкие | Нет | Низкая |
| Threshold-based | Дискретная | Средние | Да | Низкая |
| MVO daily | Дискретная | Высокие | Да | Средняя |
| **Neural ODE** | Непрерывная | **Оптимальные** | **Да** | Высокая |

---

## Ключевые метрики

### Качество траектории
- **Trajectory MSE:** Среднеквадратичная ошибка vs реализованные оптимальные веса
- **Smoothness:** $\sum \|\ddot{w}(t)\|^2$ — гладкость траектории
- **Endpoint accuracy:** Точность финальных весов

### Эффективность ребалансирования
- **Turnover:** Общий объём торговли за период
- **Transaction costs:** Суммарные издержки
- **Tracking error:** Отклонение от целевого распределения

### Финансовые результаты
- **Sharpe Ratio:** Доходность с учётом риска
- **Max Drawdown:** Максимальная просадка
- **Return after costs:** Чистая доходность

---

## Ресурсы

### Статьи

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Chen et al., NeurIPS 2018) — основополагающая работа
- [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) (Dupont et al., 2019)
- [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907) (Rubanova et al., 2019)
- [Continuous-Time Portfolio Optimization](https://www.jstor.org/stable/2328831) (Merton, 1969)
- [Deep Learning for Continuous-Time Finance](https://arxiv.org/abs/2007.04154) (2020)

### Реализации

- [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) — официальная PyTorch реализация
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) — Julia реализация
- [neural-ode-features](https://github.com/rtqichen/ffjord) — FFJORD для генеративных моделей

### Связанные главы

- [Глава 17: Deep Learning для трейдинга](../17_deep_learning) — основы нейронных сетей
- [Глава 19: RNN для временных рядов](../19_recurrent_neural_nets) — дискретные последовательные модели
- [Глава 22: Deep Reinforcement Learning](../22_deep_reinforcement_learning) — обучение с подкреплением для трейдинга

---

## Уровень сложности

⭐⭐⭐⭐⭐ (Экспертный)

**Требуется понимание:**
- Обыкновенные дифференциальные уравнения
- Численные методы интегрирования
- Neural ODE и adjoint метод
- Оптимальное управление портфелем
- Теория Марковица / Mean-Variance оптимизация
