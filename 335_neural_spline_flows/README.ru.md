# Глава 335: Neural Spline Flows — Гибкая оценка плотности для трейдинга

## Обзор

Neural Spline Flows (NSF) — это современный подход к нормализующим потокам, использующий монотонные рационально-квадратичные сплайны в качестве преобразований связывающих слоёв. В отличие от более простых аффинных преобразований, потоки на основе сплайнов могут моделировать произвольно сложные распределения с высокой точностью, что делает их идеальными для захвата тяжелохвостых, асимметричных и мультимодальных распределений финансовых доходностей.

В трейдинге точная оценка плотности критически важна для:
- **Управления рисками**: Понимание хвостовых рисков и Value-at-Risk (VaR)
- **Обнаружения режимов**: Выявление сдвигов в рыночном распределении
- **Обнаружения аномалий**: Маркировка необычных рыночных условий
- **Ценообразования опционов**: Более точное моделирование подразумеваемой волатильности
- **Оптимизации портфеля**: Лучшая оценка ковариации

В этой главе рассматривается применение Neural Spline Flows для торговли криптовалютами с использованием данных биржи Bybit.

## Основные концепции

### Что такое нормализующие потоки?

Нормализующие потоки преобразуют простое базовое распределение (например, гауссово) в сложное целевое распределение через серию обратимых преобразований:

```
Нормализующий поток:
├── Базовое распределение: z ~ N(0, I)
├── Преобразование: x = f(z)
├── Обратное: z = f⁻¹(x)
└── Плотность: p(x) = p(z) |det(∂f⁻¹/∂x)|

Ключевые свойства:
├── Биективные (обратимые) преобразования
├── Вычислимый определитель Якобиана
├── Точное вычисление правдоподобия
└── Эффективная выборка
```

### Почему Neural Spline Flows?

Традиционные связывающие потоки используют аффинные преобразования:
```
Аффинное связывание: y = x ⊙ exp(s) + t
├── Простое и быстрое
├── Ограниченная выразительность
└── Требует много слоёв для сложных распределений
```

Neural Spline Flows используют монотонные рационально-квадратичные сплайны:
```
Сплайн-связывание: y = RQS(x; w, h, d)
├── Высокая выразительность одного слоя
├── Захватывает мультимодальные распределения
├── Лучшее поведение хвостов
└── Требует меньше параметров
```

### Рационально-квадратичные сплайны

Ключевая инновация NSF — рационально-квадратичный сплайн (RQS):

```
Определение RQS:
├── Область: [x₀, xₖ] разделена на K интервалов
├── Позиции узлов: (xₖ, yₖ) для k = 0, ..., K
├── Производные в узлах: dₖ > 0 (обеспечивает монотонность)
└── В каждом интервале: рационально-квадратичная интерполяция

Для входа ξ ∈ [0, 1] в интервале k:
y = RQS(x) = [yₖ(1-ξ)² + yₖ₊₁ξ² + 2yₘξ(1-ξ)] / [(1-ξ)² + ξ² + 2ξ(1-ξ)sₖ]

Где:
├── yₘ = (yₖ + yₖ₊₁)/2 + (dₖ₊₁ - dₖ)wₖ/8
├── sₖ = (yₖ₊₁ - yₖ)/(xₖ₊₁ - xₖ)
├── wₖ = ширина интервала
└── ξ = (x - xₖ)/wₖ
```

### Почему NSF для трейдинга?

1. **Тяжёлые хвосты**: Финансовые доходности имеют толстые хвосты; сплайны моделируют их точно
2. **Асимметрия**: Рынки часто асимметричны; сплайны естественно захватывают асимметрию
3. **Мультимодальность**: Разные режимы создают мультимодальные распределения
4. **Точное правдоподобие**: Позволяет точные вычисления вероятности
5. **Быстрая выборка**: Эффективная генерация сценариев для стресс-тестирования

## Торговая стратегия

**Обзор стратегии:** Используйте Neural Spline Flows для изучения истинного распределения рыночных характеристик. Торговые сигналы генерируются на основе плотности вероятности, мер хвостового риска и обнаружения режимов.

### Конвейер генерации сигналов

```
1. Извлечение признаков:
   - Доходности на разных временных масштабах
   - Меры волатильности
   - Паттерны объёма
   - Технические индикаторы

2. Преобразование потока:
   - Преобразование признаков через обученный NSF
   - Вычисление логарифма правдоподобия текущего состояния
   - Оценка плотности в латентном пространстве

3. Генерация сигнала:
   - Высокая плотность + положительная ожидаемая доходность → Лонг
   - Высокая плотность + отрицательная ожидаемая доходность → Шорт
   - Низкая плотность → Снижение экспозиции (необычные условия)

4. Управление рисками:
   - VaR/CVaR из изученного распределения
   - Размер позиции на основе хвостового риска
   - Стоп-лоссы с учётом режима
```

### Входные сигналы

- **Сигнал на покупку**: Текущее состояние имеет высокую плотность вероятности И преобразование потока указывает на положительный импульс доходности
- **Сигнал на продажу**: Текущее состояние имеет высокую плотность вероятности И преобразование потока указывает на отрицательный импульс доходности
- **Без торговли**: Низкая плотность указывает на нетипичные рыночные условия

### Управление рисками

- **Хвостовой риск**: Используйте обратную функцию распределения для вычисления VaR на любом уровне доверия
- **Порог плотности**: Торгуйте только когда логарифм правдоподобия превышает порог
- **Обнаружение режимов**: Отслеживайте эволюцию плотности для сигналов смены режима
- **Динамический размер**: Масштабируйте позиции обратно пропорционально хвостовому риску

## Техническая спецификация

### Математические основы

#### Архитектура связывающего слоя

```
Для входа x = [x₁, x₂], разделённого на две части:
├── x₁: без изменений (тождественное преобразование)
└── x₂: преобразуется на основе x₁

Прямой проход:
├── θ = NN(x₁)  // Нейросеть выдаёт параметры сплайна
├── y₁ = x₁
└── y₂ = RQS(x₂; θ)

Обратный проход:
├── x₁ = y₁
├── θ = NN(y₁)
└── x₂ = RQS⁻¹(y₂; θ)

Логарифм определителя:
└── log|det(J)| = Σ log|d RQS/dx₂|
```

#### Параметры сплайна

Для K интервалов нейросеть выдаёт:
```
Параметры на измерение:
├── K ширин интервалов (сумма равна ширине области)
├── K высот интервалов (сумма равна высоте области)
└── K+1 значений производных в узлах

Всего параметров: 3K + 1 на преобразуемое измерение

Ограничения:
├── Ширины: softmax-нормализация
├── Высоты: softmax-нормализация
└── Производные: softplus + 1 (обеспечивает положительность)
```

#### Многомасштабная архитектура

```
                    Поток рыночных данных
                           │
                           ▼
            ┌─────────────────────────────┐
            │   Нормализация входа        │
            │   (Скользящее среднее/std)  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
    ┌───────────────┐              ┌───────────────┐
    │ Разделение:x₁ │              │ Разделение:x₂ │
    └───────┬───────┘              └───────┬───────┘
            │                              │
            │    ┌──────────────────┐      │
            └───►│  Кондиционер NN  │      │
                 │  (MLP / ResNet)  │      │
                 └────────┬─────────┘      │
                          │                │
                          ▼                │
            ┌─────────────────────────────┐│
            │  Параметры сплайна          ││
            │  ├── Ширины (K)             ││
            │  ├── Высоты (K)             ││
            │  └── Производные (K+1)      ││
            └──────────────┬──────────────┘│
                           │               │
                           ▼               │
            ┌─────────────────────────────┐│
            │  Рационально-квадратичный   │◄┘
            │  сплайн y₂ = RQS(x₂; params)│
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
       y₁ = x₁                        y₂ = RQS(x₂)
            │                              │
            └──────────────┬───────────────┘
                           ▼
                    ┌─────────────┐
                    │  Перестанов.│
                    └──────┬──────┘
                           │
                           ▼
                  (Следующий связывающий слой)
                           │
                        × L слоёв
                           │
                           ▼
                  Латентное пространство z
```

### Реализация связывающего потока

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class RationalQuadraticSpline(nn.Module):
    """
    Рационально-квадратичный сплайн

    На основе "Neural Spline Flows" (Durkan et al., 2019)
    """

    def __init__(self,
                 num_bins: int = 8,
                 bound: float = 3.0,
                 min_derivative: float = 1e-3):
        super().__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_derivative = min_derivative

    def forward(self, x: torch.Tensor,
                params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Применить сплайн-преобразование

        Args:
            x: (batch, dim) вход
            params: (batch, dim, 3*num_bins + 1) параметры сплайна

        Returns:
            y: преобразованный выход
            log_det: логарифм определителя Якобиана
        """
        # Разделение параметров
        W = params[..., :self.num_bins]
        H = params[..., self.num_bins:2*self.num_bins]
        D = params[..., 2*self.num_bins:]

        # Нормализация ширин и высот
        W = F.softmax(W, dim=-1) * 2 * self.bound
        H = F.softmax(H, dim=-1) * 2 * self.bound
        D = F.softplus(D) + self.min_derivative

        # Вычисление кумулятивных ширин и высот
        cumwidths = torch.cumsum(W, dim=-1)
        cumheights = torch.cumsum(H, dim=-1)

        # Добавить нули в начало
        cumwidths = F.pad(cumwidths, (1, 0), value=-self.bound)
        cumheights = F.pad(cumheights, (1, 0), value=-self.bound)

        # Найти интервал для каждого входа
        x_clamped = x.clamp(-self.bound, self.bound)
        bin_idx = torch.searchsorted(cumwidths[..., 1:], x_clamped.unsqueeze(-1))
        bin_idx = bin_idx.squeeze(-1).clamp(0, self.num_bins - 1)

        # Собрать параметры интервала
        input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_widths = W.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_heights = H.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_delta = input_bin_heights / input_bin_widths
        input_derivatives = D.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_derivatives_plus = D.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        # Вычислить сплайн
        xi = (x_clamped - input_cumwidths) / input_bin_widths
        xi_squared = xi ** 2
        one_minus_xi = 1 - xi
        one_minus_xi_squared = one_minus_xi ** 2

        numerator = input_bin_heights * (
            input_delta * xi_squared +
            input_derivatives * xi * one_minus_xi
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus - 2 * input_delta) *
            xi * one_minus_xi
        )

        y = input_cumheights + numerator / denominator

        # Вычислить логарифм определителя
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus * xi_squared +
            2 * input_delta * xi * one_minus_xi +
            input_derivatives * one_minus_xi_squared
        )
        log_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return y, log_det.sum(dim=-1)


class CouplingLayer(nn.Module):
    """
    Связывающий слой с преобразованием на основе нейросплайна
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: int = 128,
                 num_bins: int = 8,
                 num_hidden_layers: int = 2):
        super().__init__()

        self.dim = dim
        self.split_dim = dim // 2
        self.num_bins = num_bins

        # Размерность выхода: ширины + высоты + производные для каждого измерения
        output_dim = (dim - self.split_dim) * (3 * num_bins + 1)

        # Сеть-кондиционер
        layers = [nn.Linear(self.split_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.conditioner = nn.Sequential(*layers)
        self.spline = RationalQuadraticSpline(num_bins=num_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        # Получить параметры сплайна от кондиционера
        params = self.conditioner(x1)
        params = params.reshape(*x2.shape, 3 * self.num_bins + 1)

        # Применить сплайн
        y2, log_det = self.spline(x2, params)

        y = torch.cat([x1, y2], dim=-1)

        return y, log_det


class NeuralSplineFlow(nn.Module):
    """
    Полная модель Neural Spline Flow

    Преобразует сложное распределение данных в простое базовое распределение
    """

    def __init__(self,
                 dim: int,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 num_bins: int = 8):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers

        # Создать связывающие слои с чередующимися масками
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CouplingLayer(dim, hidden_dim, num_bins)
            )

        # Матрицы перестановок
        self.register_buffer(
            'permutations',
            torch.stack([
                torch.randperm(dim) for _ in range(num_layers)
            ])
        )
        self.register_buffer(
            'inverse_permutations',
            torch.stack([
                torch.argsort(self.permutations[i])
                for i in range(num_layers)
            ])
        )

        # Статистики для нормализации входа
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Преобразование из пространства данных в латентное пространство
        """
        # Нормализация входа
        z = (x - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        log_det_normalization = -0.5 * torch.log(self.running_var + 1e-6).sum()

        total_log_det = log_det_normalization.expand(x.shape[0])

        for i, layer in enumerate(self.layers):
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
            z = z[..., self.permutations[i]]

        return z, total_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычислить логарифм вероятности данных под моделью
        """
        z, log_det = self.forward(x)

        # Логарифм вероятности стандартного нормального
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        return log_pz + log_det

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Генерация выборок из изученного распределения
        """
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

### Инженерия признаков для NSF

```python
def compute_market_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Создание вектора признаков для Neural Spline Flow

    Признаки разработаны для захвата:
    - Динамики доходности на разных масштабах
    - Режима волатильности
    - Паттернов объёма
    - Ценового импульса
    """
    features = {}

    # Доходности на разных масштабах
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}d'] = returns.rolling(period).sum().iloc[-1]

    # Признаки волатильности
    features['volatility_20d'] = returns.rolling(lookback).std().iloc[-1]
    features['volatility_5d'] = returns.rolling(5).std().iloc[-1]
    features['vol_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)

    # Высшие моменты
    features['skewness'] = returns.rolling(lookback).skew().iloc[-1]
    features['kurtosis'] = returns.rolling(lookback).kurt().iloc[-1]

    # Признаки объёма
    volume_ma = df['volume'].rolling(lookback).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / (volume_ma.iloc[-1] + 1e-8)

    # Позиция цены в диапазоне
    high_20 = df['high'].rolling(lookback).max().iloc[-1]
    low_20 = df['low'].rolling(lookback).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    return np.array(list(features.values()))
```

### Торговая система NSF

```python
class NSFTrader:
    """
    Торговая система на основе Neural Spline Flows
    """

    def __init__(self,
                 model: NeuralSplineFlow,
                 feature_dim: int,
                 return_feature_idx: int = 0,
                 density_threshold: float = -10.0,
                 var_confidence: float = 0.95):
        self.model = model
        self.feature_dim = feature_dim
        self.return_idx = return_feature_idx
        self.density_threshold = density_threshold
        self.var_confidence = var_confidence

    def compute_var(self, num_samples: int = 10000) -> Tuple[float, float]:
        """
        Вычисление Value-at-Risk из изученного распределения
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            returns = samples[:, self.return_idx].numpy()

        var = np.percentile(returns, (1 - self.var_confidence) * 100)
        cvar = returns[returns <= var].mean()

        return var, cvar

    def generate_signal(self, market_state: torch.Tensor) -> dict:
        """
        Генерация торгового сигнала из текущего состояния рынка
        """
        self.model.eval()
        x = market_state.unsqueeze(0)

        with torch.no_grad():
            # Вычислить логарифм вероятности
            log_prob = self.model.log_prob(x).item()

            # Преобразовать в латентное пространство
            z, _ = self.model.forward(x)
            z = z.squeeze()

            # Компонента доходности в латентном пространстве
            return_z = z[self.return_idx].item()

            # Генерация условных выборок для ожидаемой доходности
            samples = self.model.sample(1000)
            expected_return = samples[:, self.return_idx].mean().item()

        # Определить, находится ли в распределении
        in_distribution = log_prob > self.density_threshold

        if not in_distribution:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'log_prob': log_prob,
                'in_distribution': False,
                'expected_return': expected_return,
                'reason': 'Вне распределения'
            }

        # Сила сигнала на основе z-оценки
        confidence = min(abs(return_z) / 2.0, 1.0)

        if expected_return > 0 and return_z > 0.5:
            signal = confidence
        elif expected_return < 0 and return_z < -0.5:
            signal = -confidence
        else:
            signal = 0.0

        return {
            'signal': signal,
            'confidence': confidence,
            'log_prob': log_prob,
            'in_distribution': True,
            'expected_return': expected_return,
            'latent_return': return_z
        }
```

### Конвейер обучения

```python
def train_nsf_model(
    model: NeuralSplineFlow,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3
) -> NeuralSplineFlow:
    """
    Обучение Neural Spline Flow методом максимального правдоподобия
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        perm = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]

            optimizer.zero_grad()

            # Отрицательное логарифмическое правдоподобие
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_log_prob = model.log_prob(val_data)
            val_loss = -val_log_prob.mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch+1}/{epochs}: "
                  f"Train NLL={total_loss/n_batches:.4f}, "
                  f"Val NLL={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
```

### Фреймворк бэктестинга

```python
class NSFBacktest:
    """
    Фреймворк бэктестинга для торговли с Neural Spline Flow
    """

    def __init__(self,
                 trader: NSFTrader,
                 lookback: int = 20):
        self.trader = trader
        self.lookback = lookback

    def run(self, prices: pd.DataFrame, warmup: int = 50) -> pd.DataFrame:
        """
        Запуск бэктеста на исторических данных
        """
        results = {
            'timestamp': [],
            'price': [],
            'signal': [],
            'confidence': [],
            'log_prob': [],
            'in_distribution': [],
            'position': [],
            'pnl': [],
            'cumulative_pnl': []
        }

        position = 0.0
        cumulative_pnl = 0.0

        for i in range(warmup, len(prices)):
            window = prices.iloc[i-self.lookback:i]
            state = compute_market_features(window)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            signal_info = self.trader.generate_signal(state_tensor)

            if i > warmup:
                daily_return = prices['close'].iloc[i] / prices['close'].iloc[i-1] - 1
                pnl = position * daily_return
                cumulative_pnl += pnl
            else:
                pnl = 0.0

            position = signal_info['signal']

            results['timestamp'].append(prices.index[i])
            results['price'].append(prices['close'].iloc[i])
            results['signal'].append(signal_info['signal'])
            results['confidence'].append(signal_info['confidence'])
            results['log_prob'].append(signal_info['log_prob'])
            results['in_distribution'].append(signal_info['in_distribution'])
            results['position'].append(position)
            results['pnl'].append(pnl)
            results['cumulative_pnl'].append(cumulative_pnl)

        return pd.DataFrame(results)

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Расчёт комплексных метрик производительности
        """
        returns = results['pnl']

        total_return = results['cumulative_pnl'].iloc[-1]

        # Коэффициент Шарпа
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Коэффициент Сортино
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Максимальная просадка
        cumulative = results['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()

        # Доля выигрышных сделок
        trading_returns = returns[returns != 0]
        if len(trading_returns) > 0:
            win_rate = (trading_returns > 0).mean()
        else:
            win_rate = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'in_distribution_ratio': results['in_distribution'].mean()
        }
```

## Требования к данным

```
Исторические данные OHLCV:
├── Минимум: 1 год данных
├── Рекомендуется: 2+ года для надёжной оценки плотности
├── Частота: от 1 часа до дневных
└── Источник: биржа Bybit

Обязательные поля:
├── timestamp
├── open, high, low, close
├── volume
└── Опционально: ставка финансирования, открытый интерес

Предобработка:
├── Обработка пропущенных значений (forward fill)
├── Удаление выбросов (> 5 std)
├── Нормализация признаков (z-score)
└── Разделение Train/Val/Test: 70/15/15
```

## Ключевые метрики

### Метрики качества модели
- **Логарифм правдоподобия**: Среднее log probability на тестовых данных
- **Биты на измерение**: Нормализованная мера правдоподобия
- **KL-дивергенция**: Расстояние от истинного распределения

### Метрики торговой производительности
- **Коэффициент Шарпа**: Доходность с поправкой на риск
- **Коэффициент Сортино**: Доходность с поправкой на нисходящий риск
- **Максимальная просадка**: Наибольшее падение от пика до дна
- **Доля выигрышей**: Процент прибыльных сделок

### Метрики распределения
- **Доля в распределении**: Доля дней с высоким log-probability
- **Покрытие хвостов**: Насколько хорошо модель захватывает экстремальные события
- **Калибровка**: Оценки вероятности vs. наблюдаемые частоты

## Зависимости

```python
# Ядро
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Глубокое обучение
torch>=2.0.0

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

1. **Точная оценка плотности**: Модель захватывает тяжёлые хвосты, асимметрию и мультимодальность
2. **Обнаружение режимов**: Изменения log-probability указывают на смену рыночного режима
3. **Квантификация риска**: Точные оценки VaR/CVaR из изученного распределения
4. **Торговая производительность**: Ожидаемый коэффициент Шарпа 0.8-1.5
5. **Обнаружение аномалий**: Маркировка событий с низкой вероятностью

## Сравнение с другими методами

| Метод | Гибкость | Точное правдоподобие | Скорость выборки | Стабильность обучения |
|-------|----------|---------------------|------------------|----------------------|
| **Neural Spline Flows** | ⭐⭐⭐⭐⭐ | ✅ Да | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Аффинные потоки | ⭐⭐⭐ | ✅ Да | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| VAE | ⭐⭐⭐⭐ | ❌ ELBO | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| GAN | ⭐⭐⭐⭐⭐ | ❌ Нет | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Диффузия | ⭐⭐⭐⭐⭐ | ❌ Приблизительно | ⭐⭐ | ⭐⭐⭐⭐ |

## Ссылки

1. **Neural Spline Flows** (Durkan et al., 2019)
   - URL: https://arxiv.org/abs/1906.04032
   - Ключевой вклад: Рационально-квадратичные сплайны для связывающих слоёв

2. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - URL: https://arxiv.org/abs/1912.02762
   - Всеобъемлющий обзор нормализующих потоков

3. **Density Estimation Using Real-NVP** (Dinh et al., 2017)
   - URL: https://arxiv.org/abs/1605.08803
   - Основа потоков на связывающих слоях

4. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2015)
   - URL: https://arxiv.org/abs/1410.8516
   - Оригинальная идея связывающего слоя

5. **Glow: Generative Flow with Invertible 1x1 Convolutions** (Kingma & Dhariwal, 2018)
   - URL: https://arxiv.org/abs/1807.03039
   - Введение обратимых свёрток 1x1

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительной торговли Neural Spline Flow на криптовалютных данных Bybit. См. директорию `rust/`.

### Возможности:
- Получение данных в реальном времени через API Bybit
- Реализация Neural Spline Flow с рационально-квадратичными сплайнами
- Обучение методом максимального правдоподобия
- Оценка плотности и генерация выборок
- Метрики риска VaR/CVaR
- Фреймворк бэктестинга с комплексными метриками
- Модульный и расширяемый дизайн

### Структура модулей:
```
rust/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs
│   ├── flow/
│   │   ├── mod.rs
│   │   ├── spline.rs
│   │   ├── coupling.rs
│   │   └── nsf.rs
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── signals.rs
│   │   └── risk.rs
│   ├── backtest/
│   │   └── mod.rs
│   └── utils/
│       └── mod.rs
└── examples/
    ├── basic_nsf.rs
    ├── bybit_trading.rs
    └── backtest.rs
```

## Уровень сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание: Теория вероятностей, Нормализующие потоки, Замена переменных, Нейронные сети, Теория сплайнов, Управление рисками, Торговые системы
