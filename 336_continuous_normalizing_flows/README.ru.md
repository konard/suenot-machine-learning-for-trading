# Глава 336: Непрерывные нормализующие потоки — Моделирование рыночной динамики с помощью Neural ODE

## Обзор

Непрерывные нормализующие потоки (Continuous Normalizing Flows, CNF) представляют собой смену парадигмы в генеративном моделировании, заменяя дискретные шаги преобразования непрерывной динамикой, управляемой нейронными обыкновенными дифференциальными уравнениями (Neural ODE). В трейдинге CNF позволяют моделировать непрерывную эволюцию рыночных состояний, изучать сложные распределения доходности и генерировать реалистичные рыночные сценарии для управления рисками и разработки стратегий.

В этой главе рассматривается применение CNF для торговли криптовалютами, используя мощь непрерывных преобразований для улавливания рыночной динамики, которую могут упустить дискретные модели.

## Основные концепции

### Что такое непрерывные нормализующие потоки?

В отличие от дискретных нормализующих потоков, применяющих последовательность фиксированных преобразований, CNF определяют непрерывное во времени преобразование от простого базового распределения к сложному целевому:

```
Дискретный поток:        z₀ → f₁ → z₁ → f₂ → z₂ → ... → zₙ
Непрерывный поток:       z(0) → динамика ОДУ → z(T)

Ключевая идея: Преобразование определяется через ОДУ:
dz/dt = f(z(t), t; θ)

Где f — нейронная сеть, параметризующая поле скоростей.
```

### Почему непрерывные нормализующие потоки для трейдинга?

1. **Гибкие распределения**: Моделирование произвольных распределений доходности без архитектурных ограничений
2. **Непрерывная динамика**: Улавливание плавных рыночных переходов вместо дискретных скачков
3. **Эффективная выборка**: Генерация рыночных сценариев решением ОДУ вперёд по времени
4. **Точное правдоподобие**: Вычисление точных лог-вероятностей через мгновенную замену переменных
5. **Эффективность по памяти**: Постоянная стоимость памяти независимо от глубины преобразования (adjoint-метод)

### От дискретных к непрерывным потокам

```
Дискретный нормализующий поток:
├── Фиксированное число слоёв
├── Замена переменных: log p(x) = log p(z) - Σ log|det(∂fᵢ/∂zᵢ₋₁)|
├── Определитель якобиана на каждом слое
└── Память масштабируется с глубиной

Непрерывный нормализующий поток (FFJORD):
├── Непрерывное преобразование через ОДУ
├── Замена переменных: log p(x) = log p(z(0)) - ∫₀ᵀ tr(∂f/∂z(t)) dt
├── След якобиана (не полный определитель!)
└── O(1) памяти через adjoint-метод
```

## Торговая стратегия

**Обзор стратегии:** Использование CNF для изучения совместного распределения рыночных признаков и будущей доходности. Торговые сигналы генерируются путём:
1. Вычисления правдоподобия текущих рыночных состояний
2. Выборки условных распределений доходности
3. Выявления смен режима через динамику распределения

### Генерация сигналов

```
1. Извлечение признаков:
   - Вычисление рыночных признаков: доходность, волатильность, дисбаланс стакана
   - Нормализация признаков для соответствия обучающему распределению

2. Вычисление правдоподобия:
   - Преобразование текущего состояния через обученный поток
   - Вычисление лог-правдоподобия через интеграл следа
   - Высокое правдоподобие → знакомый паттерн

3. Условная выборка:
   - Для заданных признаков выборка распределения будущей доходности
   - Вычисление ожидаемой доходности и доверительных интервалов
   - Среднее > 0 с высокой уверенностью → сигнал на покупку

4. Детекция режима:
   - Отслеживание траектории правдоподобия во времени
   - Резкие падения указывают на смену режима
   - Снижение экспозиции во время переходов
```

### Сигналы входа

- **Сигнал на покупку (Long)**: Условное распределение доходности центрировано выше нуля с узкой дисперсией
- **Сигнал на продажу (Short)**: Условное распределение доходности центрировано ниже нуля с узкой дисперсией
- **Без сделки**: Широкая дисперсия (неопределённость) или правдоподобие ниже порога (новое состояние)

### Управление рисками

- **Фильтрация по правдоподобию**: Торговля только когда текущее состояние имеет высокое правдоподобие
- **Размер позиции на основе дисперсии**: Размер позиции обратно пропорционален условной дисперсии
- **Детекция режима**: Снижение экспозиции при значительном падении правдоподобия
- **Расходимость ОДУ**: Мониторинг численной устойчивости преобразований потока

## Техническая спецификация

### Математическое обоснование

#### Определение Neural ODE

Основа CNF — это Neural ODE, определяющее преобразование:

```
Динамика состояния:
dz/dt = f_θ(z(t), t)

Где:
├── z(t) ∈ ℝᵈ — состояние в момент t
├── f_θ: ℝᵈ × ℝ → ℝᵈ — нейронная сеть
├── t ∈ [0, T] — время интегрирования
└── θ — обучаемые параметры

Решение через численное интегрирование:
z(T) = z(0) + ∫₀ᵀ f_θ(z(t), t) dt
```

#### Мгновенная замена переменных

Лог-вероятность эволюционирует согласно:

```
d log p(z(t))/dt = -tr(∂f_θ/∂z(t))

Это даёт нам:
log p(z(T)) = log p(z(0)) - ∫₀ᵀ tr(∂f_θ/∂z(t)) dt

Ключевые свойства:
├── Нужен только след якобиана, не полный определитель!
├── След — O(d), определитель — O(d³)
├── Позволяет моделирование высокой размерности
└── Оценка следа Хатчинсона: O(d) → O(1)
```

#### Оценка следа Хатчинсона

Для эффективного вычисления следа:

```
tr(A) = E_v[v^T A v]

Где v — случайный вектор с E[vv^T] = I

Для якобиана:
tr(∂f/∂z) ≈ E_ε[ε^T (∂f/∂z) ε]
          = E_ε[ε^T ∂(f^T ε)/∂z]  (через VJP)

Требуется только одно векторно-якобианное произведение!
```

#### Целевая функция обучения FFJORD

```
Loss = -E_{x~p_data}[log p_θ(x)]

Где:
log p_θ(x) = log p(z(0)) - ∫₀ᵀ tr(∂f_θ/∂z(t)) dt
z(0) = ODESolve(z(T)=x, f_θ, T→0)  # Обратное ОДУ

Процедура обучения:
1. Выборка x из данных
2. Решение ОДУ назад для получения z(0)
3. Вычисление log p(z(0)) под базовым распределением
4. Оценка интеграла следа при обратном проходе
5. Минимизация отрицательного лог-правдоподобия
```

### Диаграмма архитектуры

```
                    Поток рыночных данных
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Инженерия признаков      │
            │  ├── Многомасштабные доходы │
            │  ├── Меры волатильности     │
            │  ├── Паттерны объёма        │
            │  └── Технические индикаторы │
            └──────────────┬──────────────┘
                           │
                           ▼ x = Состояние рынка
            ┌─────────────────────────────┐
            │    Непрерывный нормализ.    │
            │         поток (CNF)         │
            │                             │
            │  ┌───────────────────────┐  │
            │  │  Обратное преобраз.   │  │
            │  │   dz/dt = f_θ(z, t)   │  │
            │  │   ОДУ: x → z(0)       │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────▼───────────┐  │
            │  │  Базовое распределение│  │
            │  │   p(z) = N(0, I)      │  │
            │  │   log p(z(0))         │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────▼───────────┐  │
            │  │   Интеграл следа      │  │
            │  │   ∫ tr(∂f/∂z) dt      │  │
            │  │   Оценка Хатчинсона   │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Лог-       │ │  Условные   │ │  Детекция   │
     │  правдоп.   │ │  выборки    │ │  режима     │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Торговое решение        │
            │  ├── Направление сигнала    │
            │  ├── Размер позиции         │
            │  ├── Доверительный интервал │
            │  └── Параметры риска        │
            └─────────────────────────────┘
```

### Сеть поля скоростей

```python
import torch
import torch.nn as nn
import numpy as np

class VelocityField(nn.Module):
    """
    Нейронная сеть, определяющая динамику ОДУ.

    dz/dt = f(z, t; θ)

    Сеть принимает (z, t) на вход и выводит dz/dt.
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, time_embed_dim: int = 16):
        super().__init__()

        self.dim = dim
        self.time_embed_dim = time_embed_dim

        # Временное эмбеддинг (синусоидальное)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU()
        )

        # Входная проекция
        self.input_proj = nn.Linear(dim, hidden_dim)

        # Основная сеть (остаточный MLP)
        layers = []
        for _ in range(num_layers):
            layers.append(ConcatResBlock(hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Выходная проекция
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        # Нулевая инициализация для стабильного обучения
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Вычисление поля скоростей в состоянии z и времени t.

        Аргументы:
            z: (batch, dim) текущее состояние
            t: (batch,) или скаляр текущего времени

        Возвращает:
            dz_dt: (batch, dim) скорость
        """
        # Обработка скалярного времени
        if t.dim() == 0:
            t = t.expand(z.shape[0])

        # Эмбеддинг времени
        t_emb = self.time_embed(t)

        # Проекция входа
        h = self.input_proj(z)

        # Применение остаточных блоков с кондиционированием по времени
        for layer in self.layers:
            h = layer(h, t_emb)

        # Выход
        dz_dt = self.output_proj(h)

        return dz_dt


class SinusoidalEmbedding(nn.Module):
    """Синусоидальный временной эмбеддинг (из Transformer/Diffusion моделей)"""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) *
            torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConcatResBlock(nn.Module):
    """Остаточный блок с кондиционированием по времени через конкатенацию"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim * 2, dim * 4)
        self.norm2 = nn.LayerNorm(dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = torch.cat([h, t_emb], dim=-1)
        h = self.linear1(h)
        h = nn.functional.gelu(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h
```

### Реализация решателя ОДУ

```python
class ODESolver:
    """
    Численный решатель ОДУ для непрерывных нормализующих потоков.

    Поддерживает несколько методов интегрирования:
    - Эйлер (быстрый, менее точный)
    - RK4 (сбалансированный)
    - Dopri5 (адаптивный, наиболее точный)
    """

    def __init__(self, method: str = 'rk4', atol: float = 1e-5,
                 rtol: float = 1e-5):
        self.method = method
        self.atol = atol
        self.rtol = rtol

    def solve(self, func, z0: torch.Tensor, t_span: tuple,
              num_steps: int = 100) -> torch.Tensor:
        """
        Решение ОДУ от t_span[0] до t_span[1].

        Аргументы:
            func: функция поля скоростей f(z, t)
            z0: (batch, dim) начальное состояние
            t_span: (t0, t1) интервал интегрирования
            num_steps: число шагов интегрирования

        Возвращает:
            z1: (batch, dim) конечное состояние
        """
        if self.method == 'euler':
            return self._euler(func, z0, t_span, num_steps)
        elif self.method == 'rk4':
            return self._rk4(func, z0, t_span, num_steps)
        elif self.method == 'dopri5':
            return self._dopri5(func, z0, t_span)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

    def solve_with_trace(self, func, z0: torch.Tensor, t_span: tuple,
                         num_steps: int = 100) -> tuple:
        """
        Решение ОДУ и вычисление интеграла следа для log-det якобиана.

        Возвращает:
            z1: конечное состояние
            trace_integral: ∫ tr(∂f/∂z) dt
        """
        if self.method == 'euler':
            return self._euler_with_trace(func, z0, t_span, num_steps)
        elif self.method == 'rk4':
            return self._rk4_with_trace(func, z0, t_span, num_steps)
        else:
            raise ValueError(f"След не реализован для {self.method}")

    def _euler(self, func, z, t_span, num_steps):
        """Метод Эйлера"""
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        t = t0

        for _ in range(num_steps):
            dz = func(z, torch.tensor(t, device=z.device))
            z = z + dt * dz
            t = t + dt

        return z

    def _rk4(self, func, z, t_span, num_steps):
        """Метод Рунге-Кутты 4-го порядка"""
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        t = t0
        device = z.device

        for _ in range(num_steps):
            t_tensor = torch.tensor(t, device=device)
            k1 = func(z, t_tensor)
            k2 = func(z + 0.5 * dt * k1, t_tensor + 0.5 * dt)
            k3 = func(z + 0.5 * dt * k2, t_tensor + 0.5 * dt)
            k4 = func(z + dt * k3, t_tensor + dt)

            z = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + dt

        return z

    def _euler_with_trace(self, func, z, t_span, num_steps):
        """Эйлер с оценкой следа Хатчинсона"""
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        t = t0
        trace_integral = 0.0
        device = z.device

        for _ in range(num_steps):
            t_tensor = torch.tensor(t, device=device)

            # Вычисление следа через оценку Хатчинсона
            epsilon = torch.randn_like(z)
            with torch.enable_grad():
                z_in = z.detach().requires_grad_(True)
                dz = func(z_in, t_tensor)

                # Векторно-якобианное произведение
                vjp = torch.autograd.grad(
                    dz, z_in, epsilon,
                    create_graph=True
                )[0]
                trace_est = (epsilon * vjp).sum(dim=-1)

            z = z + dt * dz.detach()
            trace_integral = trace_integral + dt * trace_est
            t = t + dt

        return z, trace_integral

    def _rk4_with_trace(self, func, z, t_span, num_steps):
        """RK4 с оценкой следа на каждом шаге"""
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        t = t0
        trace_integral = torch.zeros(z.shape[0], device=z.device)
        device = z.device

        for _ in range(num_steps):
            t_tensor = torch.tensor(t, device=device)

            # RK4 со следом в средней точке
            epsilon = torch.randn_like(z)

            with torch.enable_grad():
                z_mid = z.detach().requires_grad_(True)
                dz = func(z_mid, t_tensor + 0.5 * dt)

                vjp = torch.autograd.grad(
                    dz, z_mid, epsilon,
                    create_graph=True
                )[0]
                trace_est = (epsilon * vjp).sum(dim=-1)

            # Стандартный шаг RK4
            k1 = func(z, t_tensor)
            k2 = func(z + 0.5 * dt * k1, t_tensor + 0.5 * dt)
            k3 = func(z + 0.5 * dt * k2, t_tensor + 0.5 * dt)
            k4 = func(z + dt * k3, t_tensor + dt)

            z = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trace_integral = trace_integral + dt * trace_est
            t = t + dt

        return z, trace_integral
```

### Модель непрерывного нормализующего потока

```python
class ContinuousNormalizingFlow(nn.Module):
    """
    Непрерывный нормализующий поток в стиле FFJORD для рыночных данных.

    Особенности:
    - Гибкая динамика нейронного ОДУ
    - Оценка следа Хатчинсона для O(1) вычисления log-det
    - Регуляризация для стабильного обучения
    - Двунаправленная выборка и оценка плотности
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, t_span: tuple = (0.0, 1.0)):
        super().__init__()

        self.dim = dim
        self.t_span = t_span

        # Сеть поля скоростей
        self.velocity_field = VelocityField(
            dim=dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Решатель ОДУ
        self.solver = ODESolver(method='rk4')

        # Базовое распределение
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, x: torch.Tensor,
                reverse: bool = False) -> tuple:
        """
        Преобразование данных через поток.

        Аргументы:
            x: (batch, dim) входные данные
            reverse: если True, выборка (z→x); если False, кодирование (x→z)

        Возвращает:
            z_or_x: преобразованные данные
            log_det_jacobian: log |det(dx/dz)|
        """
        if reverse:
            # Выборка: z → x (прямое ОДУ)
            return self._sample(x)
        else:
            # Кодирование: x → z (обратное ОДУ)
            return self._encode(x)

    def _encode(self, x: torch.Tensor) -> tuple:
        """Кодирование данных в латентное пространство (x → z)"""
        # Решение ОДУ назад по времени
        t_span = (self.t_span[1], self.t_span[0])

        z, neg_trace = self.solver.solve_with_trace(
            self.velocity_field, x, t_span, num_steps=50
        )

        # log_det = -∫ tr(∂f/∂z) dt (минус, т.к. идём назад)
        log_det = -neg_trace

        return z, log_det

    def _sample(self, z: torch.Tensor) -> tuple:
        """Выборка из латентного пространства (z → x)"""
        x, trace = self.solver.solve_with_trace(
            self.velocity_field, z, self.t_span, num_steps=50
        )

        log_det = -trace

        return x, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисление лог-вероятности данных под потоком.

        log p(x) = log p(z) + log |det(dz/dx)|
                 = log p(z) - ∫ tr(∂f/∂z) dt
        """
        z, log_det = self._encode(x)

        # Лог-вероятность под базовым распределением
        log_p_z = self._log_prob_base(z)

        return log_p_z + log_det

    def _log_prob_base(self, z: torch.Tensor) -> torch.Tensor:
        """Лог-вероятность под стандартным нормальным"""
        return -0.5 * (
            z.shape[-1] * np.log(2 * np.pi) +
            (z ** 2).sum(dim=-1)
        )

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Генерация выборок из обученного распределения"""
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self._sample(z)
        return x
```

### Торговая система на основе CNF

```python
class CNFTrader:
    """
    Торговая система на основе непрерывных нормализующих потоков.

    Использует обученное распределение для:
    - Обнаружения аномалий на основе правдоподобия
    - Условного предсказания доходности
    - Размера позиции, взвешенного по уверенности
    """

    def __init__(self, cnf: ContinuousNormalizingFlow,
                 return_idx: int = 0,
                 likelihood_threshold: float = -10.0,
                 confidence_threshold: float = 0.6):
        self.cnf = cnf
        self.return_idx = return_idx
        self.likelihood_threshold = likelihood_threshold
        self.confidence_threshold = confidence_threshold

        # Для отслеживания режима
        self.likelihood_history = []
        self.likelihood_ma = None

    def generate_signal(self, features: np.ndarray) -> dict:
        """
        Генерация торгового сигнала из рыночных признаков.

        Возвращает dict с:
        - signal: направление торговли (-1, 0, 1)
        - confidence: сила сигнала [0, 1]
        - log_likelihood: мера новизны
        - expected_return: предсказанная доходность
        """
        self.cnf.eval()

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Вычисление правдоподобия
            log_prob = self.cnf.log_prob(x).item()

            # Обновление отслеживания режима
            self._update_likelihood_tracking(log_prob)

            # Проверка, в распределении ли
            if log_prob < self.likelihood_threshold:
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'log_likelihood': log_prob,
                    'expected_return': 0.0,
                    'regime_change': self._detect_regime_change()
                }

            # Выборка условной доходности
            expected_return, return_std = self._estimate_conditional_return(x)

            # Уверенность на основе распределения доходности
            z_score = abs(expected_return) / (return_std + 1e-8)
            confidence = min(z_score / 3.0, 1.0)

            # Направление сигнала
            if confidence < self.confidence_threshold:
                signal = 0
            else:
                signal = 1 if expected_return > 0 else -1

            return {
                'signal': signal,
                'confidence': confidence,
                'log_likelihood': log_prob,
                'expected_return': expected_return,
                'return_std': return_std,
                'regime_change': self._detect_regime_change()
            }

    def _estimate_conditional_return(self, x: torch.Tensor,
                                     num_samples: int = 100) -> tuple:
        """Оценка ожидаемой доходности и std через выборку"""
        z, _ = self.cnf._encode(x)

        # Возмущение в латентном пространстве и декодирование
        noise = torch.randn(num_samples, z.shape[-1]) * 0.1
        z_perturbed = z + noise

        samples, _ = self.cnf._sample(z_perturbed)

        returns = samples[:, self.return_idx].numpy()

        return returns.mean(), returns.std()

    def _update_likelihood_tracking(self, log_prob: float):
        """Отслеживание правдоподобия для детекции режима"""
        self.likelihood_history.append(log_prob)

        if len(self.likelihood_history) > 50:
            self.likelihood_history = self.likelihood_history[-50:]

        if len(self.likelihood_history) >= 10:
            self.likelihood_ma = np.mean(self.likelihood_history[-10:])

    def _detect_regime_change(self) -> bool:
        """Детекция смены режима через падение правдоподобия"""
        if self.likelihood_ma is None or len(self.likelihood_history) < 20:
            return False

        recent = self.likelihood_history[-1]
        baseline = np.mean(self.likelihood_history[-20:-10])

        return recent < baseline - 2.0
```

## Требования к данным

```
Исторические данные OHLCV:
├── Минимум: 6 месяцев часовых данных
├── Рекомендуется: 1+ год для робастного обучения распределения
├── Частота: от 1 часа до дневных
└── Источник: Bybit, Binance или другие биржи

Обязательные поля:
├── timestamp
├── open, high, low, close
├── volume
└── Опционально: funding rate, open interest

Предобработка:
├── Лог-доходности для стационарности
├── Z-score нормализация по признакам
├── Обрезка выбросов до ±4 std
├── Разделение Train/Val/Test: 70/15/15
└── Сохранение временного порядка
```

## Ключевые метрики

- **Отрицательное лог-правдоподобие (NLL)**: Целевая функция обучения, меньше лучше
- **Битов на измерение (BPD)**: NLL / (dim * log(2)), сравнимо между размерностями
- **Качество выборок**: Визуальный осмотр сгенерированных рыночных сценариев
- **ODE NFE**: Число вычислений функции (вычислительная стоимость)
- **Коэффициент Шарпа**: Доходность с поправкой на риск
- **Максимальная просадка**: Худшее падение от пика до дна

## Зависимости

```python
# Основные
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Глубокое обучение
torch>=2.0.0

# Решатели ОДУ (опционально, для продвинутых методов)
torchdiffeq>=0.2.3

# Рыночные данные
ccxt>=4.0.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0

# Утилиты
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Ожидаемые результаты

1. **Обучение распределения**: CNF захватывает сложные, мультимодальные распределения доходности
2. **Обнаружение аномалий**: Детекция смены режима на основе правдоподобия
3. **Генерация сценариев**: Реалистичные рыночные сценарии для управления рисками
4. **Торговые сигналы**: Ожидаемый коэффициент Шарпа 0.7-1.3 при правильной настройке
5. **Вычислительная эффективность**: O(1) памяти через adjoint-метод

## Литература

1. **FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models** (Grathwohl et al., 2018)
   - URL: https://arxiv.org/abs/1810.01367

2. **Neural Ordinary Differential Equations** (Chen et al., 2018)
   - URL: https://arxiv.org/abs/1806.07366

3. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2021)
   - URL: https://arxiv.org/abs/1912.02762

4. **Flow Matching for Generative Modeling** (Lipman et al., 2023)
   - URL: https://arxiv.org/abs/2210.02747

5. **Scalable Reversible Generative Models with Free-form Continuous Dynamics** (Grathwohl et al., 2019)
   - URL: https://openreview.net/forum?id=rJxgknCcK7

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительной торговли CNF на криптовалютных данных с Bybit. См. директорию `rust/`.

### Особенности:
- Получение данных в реальном времени с Bybit API
- Собственная реализация решателя нейронного ОДУ
- Сеть поля скоростей с кондиционированием по времени
- Эффективная оценка следа для вычисления log-det
- Фреймворк бэктестинга с комплексными метриками
- Модульный и расширяемый дизайн

## Уровень сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание: Обыкновенные дифференциальные уравнения, Численные методы, Теория вероятностей, Нейронные сети, Формула замены переменных, Генеративное моделирование
