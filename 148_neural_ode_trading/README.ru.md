# Глава 148: Нейронные ОДУ для трейдинга

## Обзор

**Нейронные обыкновенные дифференциальные уравнения (Neural ODEs)** представляют собой смену парадигмы в глубоком обучении: вместо дискретных слоёв мы определяем сеть как непрерывную динамическую систему и решаем её с помощью численных методов для ОДУ. В этой главе мы исследуем, как Neural ODEs могут трансформировать трейдинг, моделируя рынки как системы непрерывного времени -- естественно обрабатывая нерегулярные временные метки, обеспечивая обучение с постоянной памятью и позволяя гладкую интерполяцию рыночной динамики.

Ключевая идея элегантно проста: заменить дискретное остаточное соединение `h_{l+1} = h_l + f(h_l)` на непрерывную динамику `dh/dt = f_theta(h(t), t)`, и решить вперёд по времени для получения предсказаний.

## Почему Neural ODEs для трейдинга?

### Проблема дискретных моделей

Традиционные модели последовательностей (LSTM, GRU, Transformer) работают на фиксированной временной сетке. Финансовые данные, однако, по своей природе нерегулярны:

- **Тиковые данные** приходят через непредсказуемые интервалы (миллисекунды до секунд)
- **Торговые остановки** создают пропуски в наблюдениях
- **Разные биржи** отчитываются с разной частотой
- **Пропущенные данные** из-за сетевых проблем или неликвидных рынков
- **Мультитаймфреймовый анализ** требует согласования разных частот дискретизации

Дискретные модели справляются с этим плохо -- они требуют либо интерполяции (внося артефакты), либо дополнения (расходуя вычисления впустую).

### Решение Neural ODE

Neural ODEs моделируют **непрерывную эволюцию** состояния рынка:

```
Состояние рынка в момент t:  h(t)
Динамика:                    dh/dt = f_theta(h(t), t)
Предсказание в момент T:     h(T) = h(0) + integral_0^T f_theta(h(t), t) dt
```

**Ключевые преимущества для трейдинга:**

| Свойство | Дискретные модели | Neural ODE |
|----------|-------------------|------------|
| Нерегулярные метки | Требует интерполяции | Нативная поддержка |
| Память (обратное распр.) | O(L) слоёв | O(1) через adjoint |
| Временное разрешение | Фиксированная сетка | Непрерывное (любое t) |
| Пропущенные данные | Нужна импутация | Естественная обработка |
| Многошаговый прогноз | Авторегрессивный | Одно решение ОДУ |
| Контроль глубины | Целочисленные слои | Непрерывная "глубина" |

## Математические основы

### От ResNet к Neural ODEs

Остаточная сеть вычисляет:

```
h_{l+1} = h_l + f_theta(h_l, theta_l)      для l = 0, 1, ..., L-1
```

При L -> infinity и шаге -> 0 это становится:

```
dh(t)/dt = f_theta(h(t), t)
```

где:
- `h(t)` -- скрытое состояние в непрерывном времени t
- `f_theta` -- нейронная сеть, параметризующая динамику
- `theta` -- обучаемые параметры (общие для всех "глубин")

Результат получается решением задачи Коши (IVP):

```
h(T) = h(0) + integral_0^T f_theta(h(t), t) dt
```

### Решатели ОДУ (ODE Solvers)

Прямой проход требует численного решения ОДУ. Основные методы:

#### Метод Эйлера (1-й порядок)
```
y_{n+1} = y_n + h * f(t_n, y_n)
```
Простой, но неточный. Ошибка O(h). Используется только для сравнения.

#### Рунге-Кутта 4-го порядка (RK4)
```python
k1 = f(t_n, y_n)
k2 = f(t_n + h/2, y_n + h*k1/2)
k3 = f(t_n + h/2, y_n + h*k2/2)
k4 = f(t_n + h, y_n + h*k3)
y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
```
Хороший баланс точности и эффективности. Локальная ошибка O(h^4).

#### Дорманд-Принс (адаптивный, 4/5-й порядок)
```
- Вложенный метод RK с решениями 4-го и 5-го порядков
- Оценка ошибки = |y5 - y4|
- Адаптация шага: h_new = h * (tol / error)^(1/5)
```
Решатель по умолчанию в `torchdiffeq` (`dopri5`). Адаптирует размер шага для поддержания точности -- делает малые шаги в областях быстрых изменений и большие в гладких областях.

### Метод сопряжённых (Adjoint Sensitivity Method)

Ключевая инновация, делающая обучение Neural ODE практичным. Вместо обратного распространения через все шаги решателя (память O(L)), мы решаем **расширенное ОДУ назад во времени**:

```
Прямой проход:  h(T) = ODESolve(f_theta, h(0), [0, T])
Функция потерь: L = L(h(T))

Сопряжённая:    a(t) = dL/dh(t)   (чувствительность потерь к состоянию в момент t)

Обратное ОДУ:
  da/dt = -a(t)^T * (df/dh)       (сопряжённая динамика)

Градиент параметров:
  dL/d_theta = integral_T^0 a(t)^T * (df/d_theta) dt
```

**Память: O(1)** -- нужно хранить только текущее состояние, не всю прямую траекторию!

### Латентные ОДУ для нерегулярных временных рядов

**Latent ODE** (Rubanova et al., 2019) сочетает Neural ODEs с фреймворком VAE для нерегулярно дискретизированных последовательностей:

```
Архитектура:
1. Кодировщик RNN:  обрабатывает наблюдения в обратном порядке
                    x_T, x_{T-1}, ..., x_1 -> q(z_0 | x_{1:T})

2. Латентное ОДУ:   dz/dt = f_theta(z(t), t)
                    z_0 ~ q(z_0 | x_{1:T})
                    z(t_1), z(t_2), ... = ODESolve(f_theta, z_0, [t_1, t_2, ...])

3. Декодер:         x_hat(t_i) = g_phi(z(t_i))

Потери = Реконструкция + KL(q(z_0|x) || p(z_0))
```

**Почему это важно для трейдинга:**
- Кодировщик обрабатывает исторические данные любой длины и нерегулярности
- Латентное ОДУ захватывает гладкую рыночную динамику
- Декодер реконструирует/предсказывает в любой момент времени
- Фреймворк VAE обеспечивает оценку неопределённости

### Гибрид ODE-RNN

**ODE-RNN** сочетает непрерывную динамику ОДУ с дискретными обновлениями RNN:

```
Для каждого наблюдения (x_i, t_i):
  1. Между наблюдениями: h(t_{i-1}) -> h(t_i^-) через ОДУ
     dh/dt = f_theta(h, t)    для t из [t_{i-1}, t_i]

  2. При наблюдении:     h(t_i^-) -> h(t_i) через RNN
     h(t_i) = GRU(x_i, h(t_i^-))
```

**Интуиция для трейдинга:**
- Между сделками: состояние рынка эволюционирует плавно (ОДУ захватывает это)
- При каждой сделке: поступает новая информация, и мы обновляем наше представление (RNN захватывает это)

### Непрерывные нормализующие потоки

Neural ODEs могут определять **Continuous Normalizing Flows (CNF)** для оценки плотности:

```
Преобразование: z(0) ~ p_0(z)  (простое базовое распределение, напр., гауссово)
                dz/dt = f_theta(z(t), t)
                z(1) = x  (сложное распределение данных)

Логарифм вероятности:
  log p(x) = log p_0(z(0)) - integral_0^1 tr(df/dz) dt
```

**Для трейдинга:** Моделирование полного распределения доходностей, а не только среднего. Позволяет:
- Оценку VaR (Value at Risk)
- Анализ хвостовых рисков
- Ценообразование опционов

## Реализация

### Реализация на Python

Наша реализация на Python использует `torchdiffeq` (Chen et al., 2018) для решения ОДУ с автоматическим дифференцированием:

#### Модель Neural ODE (`python/neural_ode.py`)

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ODEFunc(nn.Module):
    """Динамика: dh/dt = f_theta(h(t), t)"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 для времени
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.nfe = 0  # Отслеживание вычислений функции

    def forward(self, t, h):
        self.nfe += 1
        t_expand = t.expand(h.shape[0], 1)
        h_aug = torch.cat([h, t_expand], dim=-1)
        return self.net(h_aug)

class NeuralODE(nn.Module):
    """Полная Neural ODE для трейдинга."""

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t_pred=None):
        # Кодирование последовательности в начальное состояние
        _, h0 = self.encoder(x)
        h0 = h0.squeeze(0)

        # Решение ОДУ вперёд
        if t_pred is None:
            t_pred = torch.tensor([0.0, 1.0])

        h_trajectory = odeint_adjoint(
            self.ode_func, h0, t_pred,
            method='dopri5', rtol=1e-4, atol=1e-5
        )

        # Декодирование финального состояния
        return self.decoder(h_trajectory[-1])
```

### Реализация на Rust

Наша реализация на Rust предоставляет решатели ОДУ и движок для инференса Neural ODE:

#### Решатель RK4 (`rust_neural_ode/src/lib.rs`)

```rust
pub struct RK4Solver;

impl ODESolver for RK4Solver {
    fn solve<F>(&self, f: &F, y0: &Array1<f64>,
                t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
    where F: Fn(f64, &Array1<f64>) -> Array1<f64> {
        let mut t = t_start;
        let mut y = y0.clone();

        while t < t_end - 1e-12 {
            let h = (t_end - t).min(dt);
            let k1 = f(t, &y);
            let k2 = f(t + h * 0.5, &(&y + &(&k1 * (h * 0.5))));
            let k3 = f(t + h * 0.5, &(&y + &(&k2 * (h * 0.5))));
            let k4 = f(t + h, &(&y + &(&k3 * h)));
            y = &y + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (h / 6.0));
            t += h;
        }
        y
    }
}
```

## Применения в трейдинге

### 1. Моделирование нерегулярно дискретизированных тиковых данных

```python
# Получение тиковых данных с Bybit (нерегулярные временные метки)
loader = BybitDataLoader()
ticks = loader.fetch_recent_trades("BTCUSDT", limit=1000)

# Тиковые данные имеют нерегулярные временные промежутки:
# timestamp_ms    price      size    side   time_delta_ms
# 1706000000100   65432.50   0.001   buy    0.0
# 1706000000342   65433.00   0.005   buy    242.0   <- 242мс
# 1706000000343   65432.80   0.010   sell   1.0     <- 1мс
# 1706000001567   65435.00   0.100   buy    1224.0  <- 1.2с

# ODE-RNN обрабатывает это естественно
model = ODERNN(input_dim=3, hidden_dim=64, output_dim=1)
prediction = model(features, timestamps, mask)
```

### 2. Непрерывная динамика портфеля

```python
# Состояние портфеля: [вес_BTC, вес_ETH, вес_SOL, кэш]
# dw/dt = f_theta(w(t), market_state(t), t)
model = NeuralODE(input_dim=12, hidden_dim=64, output_dim=4)
weight_trajectory = model.predict_trajectory(current_state, t_rebalance)
```

### 3. Эволюция латентных факторов

```python
# Латентное ОДУ обнаруживает скрытые факторы из наблюдаемых цен
latent_ode = LatentODE(input_dim=5, latent_dim=8, hidden_dim=64)
result = latent_ode(observations, timestamps, mask)
z_trajectory = result['z_trajectory']  # (time, batch, 8)
# 8-мерное латентное пространство может захватывать:
# z_0: Общий тренд рынка
# z_1: Режим волатильности
# z_2: Фактор моментума
# z_3: Фактор возврата к среднему
```

### 4. Работа с пропущенными данными

```python
mask = torch.ones(batch_size, seq_len, n_features)
mask[:, 30:35, :] = 0  # Имитация 5 минут пропущенных данных

# Latent ODE обрабатывает это естественно
result = latent_ode(observations, timestamps, mask=mask)
# ОДУ интегрирует гладко через пропуски
```

### 5. Криптотрейдинг через Bybit

```python
# Полный конвейер для криптотрейдинга на Bybit
from python.data_loader import BybitDataLoader, create_irregular_dataset
from python.neural_ode import NeuralODE
from python.backtest import NeuralODEBacktester

# 1. Получение данных
loader = BybitDataLoader()
df = loader.fetch_klines("BTCUSDT", interval="1", limit=1000)

# 2. Создание датасета с нерегулярными метками
train_ds, test_ds = create_irregular_dataset(symbol="BTCUSDT", source="bybit")

# 3. Обучение модели
model = NeuralODE(input_dim=5, hidden_dim=64, output_dim=1)
history = train_neural_ode(model, train_loader, test_loader, epochs=100)

# 4. Бэктест
backtester = NeuralODEBacktester(model, initial_capital=100_000)
signals = backtester.generate_signals(test_ds)
metrics = backtester.run_backtest(signals, strategy="momentum")
```

## Сравнение: Neural ODE vs RNN vs ResNet

### Концептуальное сравнение

```
ResNet (дискретный, фиксированная глубина):
  h_0 -> [Слой 1] -> h_1 -> [Слой 2] -> h_2 -> ... -> h_L
  Память: O(L)

RNN (дискретный, переменная длина):
  x_1 -> [RNN] -> h_1 -> x_2 -> [RNN] -> h_2 -> ... -> h_T
  Память: O(T), требует фиксированных шагов

Neural ODE (непрерывный, адаптивный):
  h(0) -> |--решатель ОДУ---| -> h(T)
  Память: O(1) с adjoint методом
  Вычисляет в любой момент времени
```

### Количественное сравнение

| Метрика | LSTM | GRU | Transformer | Neural ODE | ODE-RNN |
|---------|------|-----|-------------|------------|---------|
| Нерегулярные данные | Плохо | Плохо | Средне | Отлично | Отлично |
| Масштабирование памяти | O(T) | O(T) | O(T^2) | O(1) | O(T) |
| Длинные последовательности | Средне | Средне | Хорошо | Хорошо | Хорошо |
| Скорость обучения | Быстро | Быстро | Быстро | Медленнее | Медленнее |
| Неопределённость | Нет | Нет | Нет | Через Latent ODE | Через Bayes |
| Непрерывное время | Нет | Нет | Нет | Да | Да |

## Руководство по выбору решателя ОДУ

| Решатель | Порядок | Адаптивный | Шагов для tol=1e-4 | Лучше для |
|----------|---------|------------|-------------------|-----------|
| Euler | 1 | Нет | ~10000 | Только отладка |
| Midpoint | 2 | Нет | ~1000 | Простые задачи |
| RK4 | 4 | Нет | ~100 | Продакшн, фикс. шаг |
| Дорманд-Принс | 4/5 | Да | ~20-50 | Общее назначение |
| Adams | Перем. | Да | ~10-30 | Жёсткие задачи |

## Структура проекта

```
148_neural_ode_trading/
|
|-- README.md                          # Английская версия
|-- README.ru.md                       # Этот файл (русская версия)
|-- readme.simple.md                   # Простое объяснение (English)
|-- readme.simple.ru.md                # Простое объяснение (русский)
|
|-- python/
|   |-- __init__.py                    # Инициализация пакета
|   |-- requirements.txt               # Зависимости
|   |-- neural_ode.py                  # Neural ODE, ODEFunc, CNF
|   |-- latent_ode.py                  # Latent ODE для нерегулярных рядов
|   |-- ode_rnn.py                     # Гибрид ODE-RNN, GRU-ODE-Bayes
|   |-- train.py                       # Конвейер обучения
|   |-- data_loader.py                 # Загрузчики данных Bybit + акции
|   |-- visualize.py                   # Утилиты визуализации
|   |-- backtest.py                    # Бэктестирование стратегий
|
|-- rust_neural_ode/
|   |-- Cargo.toml                     # Зависимости Rust
|   |-- src/
|   |   |-- lib.rs                     # Ядро (решатели ОДУ, модель, данные)
|   |   |-- bin/
|   |       |-- train.rs               # Бинарник обучения
|   |       |-- predict.rs             # Бинарник предсказаний + бэктест
|   |       |-- fetch_data.rs          # Бинарник загрузки данных
|   |-- examples/
|       |-- basic_ode.rs               # Базовые примеры решателей ОДУ
|       |-- trading_demo.rs            # Полный демо торгового конвейера
```

## Быстрый старт

### Python

```bash
cd 148_neural_ode_trading

# Установка зависимостей
pip install -r python/requirements.txt

# Обучение модели Neural ODE
python -m python.train --model neural_ode --symbol BTCUSDT --epochs 100

# Обучение Latent ODE (для нерегулярных данных)
python -m python.train --model latent_ode --symbol ETHUSDT --epochs 200

# Обучение гибрида ODE-RNN
python -m python.train --model ode_rnn --source stock --epochs 150

# Запуск бэктеста
python -m python.backtest --model neural_ode --symbol BTCUSDT --strategy momentum
```

### Rust

```bash
cd 148_neural_ode_trading/rust_neural_ode

# Загрузка данных с Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --data-type klines --limit 1000

# Загрузка тиковых данных (нерегулярные метки)
cargo run --bin fetch_data -- --symbol BTCUSDT --data-type ticks --limit 500

# Обучение модели
cargo run --bin train -- --symbol BTCUSDT --epochs 50 --hidden-dim 32

# Предсказания и бэктест
cargo run --bin predict -- --symbol BTCUSDT --compare-solvers

# Запуск примеров
cargo run --example basic_ode
cargo run --example trading_demo
```

## Ключевые гиперпараметры

| Параметр | Типичный диапазон | Примечания |
|----------|-------------------|------------|
| `hidden_dim` | 32-128 | Больше = выразительнее, но медленнее ОДУ |
| `latent_dim` | 8-32 | Для Latent ODE; захватывает латентные факторы |
| `n_ode_layers` | 2-4 | Слои в сети f_theta |
| `solver` | `dopri5` | Адаптивный; `rk4` для фиксированного шага |
| `rtol` | 1e-3 до 1e-5 | Относительная точность для адаптивных методов |
| `atol` | 1e-4 до 1e-6 | Абсолютная точность |
| `use_adjoint` | True | O(1) память; False для маленьких моделей |
| `kl_weight` | 0.001-0.1 | Latent ODE: вес KL-дивергенции |
| `activation` | `tanh` | Гладкая активация для стабильной динамики ОДУ |

## Ссылки

1. **Chen et al.** "Neural Ordinary Differential Equations" (NeurIPS 2018) -- Основополагающая работа, вводящая Neural ODEs и метод сопряжённых.

2. **Rubanova et al.** "Latent ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019) -- Расширение Neural ODEs для нерегулярных временных меток через VAE.

3. **De Brouwer et al.** "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series" (NeurIPS 2019) -- Динамика ОДУ в стиле GRU с байесовской неопределённостью.

4. **Kidger et al.** "Neural Controlled Differential Equations for Irregular Time Series" (NeurIPS 2020) -- Дальнейшее расширение с использованием управляемых дифференциальных уравнений.

5. **Grathwohl et al.** "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (ICLR 2019) -- Непрерывные нормализующие потоки для оценки плотности.

6. **Dupont et al.** "Augmented Neural ODEs" (NeurIPS 2019) -- Решение ограничений стандартных Neural ODEs расширением пространства состояний.

7. **Jia & Benson** "Neural Jump Stochastic Differential Equations" (NeurIPS 2019) -- Расширение до процессов jump-diffusion, релевантных для финансового моделирования.

---

*Эта глава является частью серии "Машинное обучение для трейдинга". Весь код содержится в этой директории и может быть запущен независимо.*
