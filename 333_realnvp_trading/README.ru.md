# Глава 333: RealNVP Трейдинг — Обратимые Преобразования для Обучения Распределений Рынка

## Обзор

RealNVP (Real-valued Non-Volume Preserving) — это мощная модель нормализующих потоков, которая обучает обратимые преобразования между сложными распределениями данных и простыми базовыми распределениями. В трейдинге RealNVP позволяет точно оценивать плотность рыночных состояний, обнаруживать аномалии и генерировать реалистичные рыночные сценарии для анализа рисков и разработки стратегий.

В этой главе мы рассмотрим применение RealNVP к торговле криптовалютами, используя биективные преобразования для моделирования рыночной динамики, обнаружения смены режимов и генерации вероятностных торговых сигналов.

## Основные Концепции

### Что такое RealNVP?

RealNVP — это модель нормализующих потоков, которая преобразует данные через последовательность обратимых аффинных слоёв связывания:

```
Нормализующий Поток: z = f(x)

Где:
├── x = сложные данные (рыночные состояния)
├── z = простое распределение (обычно Гауссово)
├── f = последовательность обратимых преобразований
└── Ключевое свойство: точная плотность через замену переменных

p(x) = p(z) |det(∂f/∂x)|

Якобиан детерминанта вычислим благодаря конструкции слоёв связывания!
```

### Почему RealNVP для Трейдинга?

1. **Точная Оценка Плотности**: Вычисление точной вероятности любого рыночного состояния
2. **Биективное Отображение**: Идеальное восстановление — без потери информации
3. **Эффективная Выборка**: Генерация реалистичных рыночных сценариев
4. **Простое Обучение**: Не нужны MCMC или вариационные границы
5. **Обнаружение Аномалий**: Низкая вероятность указывает на необычные условия
6. **Интерпретируемое Латентное Пространство**: Изученное z-пространство раскрывает структуру рынка

### Аффинные Слои Связывания

```
Преобразование Слоя Связывания:
├── Разделить вход x на (x₁, x₂)
├── Вычислить масштаб s = s_θ(x₁) и сдвиг t = t_θ(x₁)
├── Преобразовать: y₂ = x₂ ⊙ exp(s) + t
├── Пропустить: y₁ = x₁
└── Выход: y = (y₁, y₂)

Обратное (тривиально вычислить):
├── x₂ = (y₂ - t) ⊙ exp(-s)
├── x₁ = y₁
└── Якобиан: det = exp(sum(s))

Ключевая идея: Половина переменных определяет преобразование другой половины
```

## Торговая Стратегия

**Обзор Стратегии:** Используем RealNVP для изучения совместного распределения рыночных признаков. Торговые сигналы генерируются на основе:
1. Плотности вероятности текущего рыночного состояния
2. Направления в латентном пространстве, указывающего на режим рынка
3. Сгенерированных сценариев для оценки рисков

### Генерация Сигналов

```
1. Извлечение Признаков:
   - Вычислить рыночные признаки: доходности, волатильность, соотношения объёма
   - Нормализовать признаки к стандартной шкале

2. Преобразование Потока:
   - Преобразовать рыночное состояние x → z через обученный поток
   - Вычислить лог-вероятность: log p(x) = log p(z) + sum(log|det J|)

3. Интерпретация Сигнала:
   - Высокая плотность → нормальные условия → следование тренду
   - Низкая плотность → аномалия → снизить экспозицию или возврат к среднему
   - Направление в латентном пространстве → индикатор режима → корректировка стратегии

4. Генерация Сценариев:
   - Сэмплировать z ~ N(0, I)
   - Обратное преобразование: x = f⁻¹(z)
   - Анализ распределения будущих сценариев
```

### Сигналы Входа

- **Сигнал на Покупку**: Высокая плотность + положительный моментум в латентном пространстве
- **Сигнал на Продажу**: Высокая плотность + отрицательный моментум в латентном пространстве
- **Сигнал Выхода**: Низкая плотность (необычное состояние рынка) → уменьшить позицию

### Управление Рисками

- **Порог Плотности**: Торговать только когда log p(x) > порог
- **Анализ Сценариев**: Использовать сгенерированные сценарии для оценки VaR
- **Определение Режима**: Мониторить кластеры в латентном пространстве для обнаружения смены режимов

## Техническая Спецификация

### Математические Основы

#### Формула Замены Переменных

Для биективной функции f: X → Z с обратной g = f⁻¹:

```
p_X(x) = p_Z(f(x)) |det(∂f/∂x)|

Для композиции потоков f = f_K ∘ f_{K-1} ∘ ... ∘ f_1:

log p(x) = log p(z) + Σ_{k=1}^K log|det(∂f_k/∂h_{k-1})|

Где h_k = f_k(h_{k-1}) и h_0 = x, h_K = z
```

#### Аффинный Слой Связывания

```
Для входа x ∈ R^d, разделить на x = (x_{1:d/2}, x_{d/2+1:d})

Прямое преобразование:
├── y_{1:d/2} = x_{1:d/2}  (тождество)
├── y_{d/2+1:d} = x_{d/2+1:d} ⊙ exp(s(x_{1:d/2})) + t(x_{1:d/2})
│
├── Где s, t: R^{d/2} → R^{d/2} — нейронные сети
└── ⊙ обозначает поэлементное умножение

Якобиан:
├── ∂y/∂x — нижнетреугольная матрица со структурой:
│   [I    0   ]
│   [*  diag(exp(s))]
├── det = Π exp(s_i) = exp(Σ s_i)
└── log|det| = Σ s_i  (простая сумма!)
```

#### Многомасштабная Архитектура

```
Многомасштабная Архитектура RealNVP:
├── Уровень 1: Полное разрешение
│   ├── Слои связывания (x4)
│   ├── Операция сжатия
│   └── Разделение: выделить половину измерений
│
├── Уровень 2: Половинное разрешение
│   ├── Слои связывания (x4)
│   ├── Операция сжатия
│   └── Разделение: выделить половину измерений
│
└── Уровень 3: Четвертное разрешение
    └── Финальные слои связывания

Выделенные измерения идут напрямую в z, остальные продолжают на следующий уровень
```

### Диаграмма Архитектуры

```
                    Поток Рыночных Данных
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Извлечение Признаков     │
            │  ├── Доходности (много-масш)│
            │  ├── Меры волатильности     │
            │  ├── Паттерны объёма        │
            │  └── Технические индикаторы │
            └──────────────┬──────────────┘
                           │
                           ▼ Рыночное Состояние x ∈ R^d
            ┌─────────────────────────────┐
            │        Поток RealNVP        │
            │                             │
            │  ┌───────────────────────┐  │
            │  │  Слой Связывания 1    │  │
            │  │  ├── Разделить x→(x₁,x₂)│
            │  │  ├── s,t = НС(x₁)     │  │
            │  │  └── y₂ = x₂⊙eˢ + t   │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │  Слой Связывания 2    │  │
            │  │  (чередующееся разд.) │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │          ... x K            │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │  Латентное Простр. z  │  │
            │  │  z ~ N(0, I)          │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │ Лог Вероят. │ │  Латентное  │ │ Сгенерир.   │
     │  log p(x)   │ │ Направление │ │  Сценарии   │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Торговое Решение        │
            │  ├── Фильтр по плотности    │
            │  ├── Индикатор режима       │
            │  ├── Оценка рисков          │
            │  └── Размер позиции         │
            └─────────────────────────────┘
```

### Извлечение Признаков для RealNVP

```python
import numpy as np
import pandas as pd

def compute_market_state(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Создание вектора рыночного состояния для RealNVP
    """
    features = {}

    # Многомасштабные доходности
    returns = df['close'].pct_change()
    for period in [1, 5, 10, 20]:
        features[f'return_{period}'] = returns.rolling(period).sum().iloc[-1]

    # Признаки волатильности
    features['volatility'] = returns.rolling(lookback).std().iloc[-1]
    features['volatility_ratio'] = (
        returns.rolling(5).std().iloc[-1] /
        returns.rolling(20).std().iloc[-1]
    )

    # Моментум
    features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1

    # Признаки объёма
    volume_ma = df['volume'].rolling(lookback).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / volume_ma.iloc[-1]

    # Позиция цены
    high_20 = df['high'].rolling(lookback).max().iloc[-1]
    low_20 = df['low'].rolling(lookback).min().iloc[-1]
    features['price_position'] = (df['close'].iloc[-1] - low_20) / (high_20 - low_20 + 1e-8)

    # RSI-подобный признак
    gains = returns.clip(lower=0).rolling(14).mean().iloc[-1]
    losses = (-returns.clip(upper=0)).rolling(14).mean().iloc[-1]
    features['rsi'] = gains / (gains + losses + 1e-8)

    return np.array(list(features.values()))
```

### Реализация Слоя Связывания

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """
    Аффинный слой связывания для RealNVP

    Преобразует вход x:
    - Разделение на (x1, x2)
    - Вычисление масштаба s и сдвига t из x1
    - Преобразование: y2 = x2 * exp(s) + t, y1 = x1
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, mask_type: str = 'even'):
        super().__init__()

        self.dim = dim
        self.mask_type = mask_type

        # Создание маски
        if mask_type == 'even':
            self.register_buffer('mask', torch.arange(dim) % 2 == 0)
        else:  # 'odd'
            self.register_buffer('mask', torch.arange(dim) % 2 == 1)

        self.n_masked = self.mask.sum().item()
        self.n_unmasked = dim - self.n_masked

        # Сети масштаба и сдвига
        layers = [nn.Linear(self.n_masked, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.shared_net = nn.Sequential(*layers)
        self.scale_net = nn.Linear(hidden_dim, self.n_unmasked)
        self.translation_net = nn.Linear(hidden_dim, self.n_unmasked)

        # Инициализация выхода масштаба нулями для стабильного обучения
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход: x → y

        Возвращает:
            y: преобразованный выход
            log_det: логарифм детерминанта Якобиана
        """
        x_masked = x[:, self.mask]
        x_unmasked = x[:, ~self.mask]

        h = self.shared_net(x_masked)
        s = self.scale_net(h)
        t = self.translation_net(h)

        # Ограничение масштаба для численной стабильности
        s = torch.clamp(s, -5, 5)

        # Преобразование немаскированной части
        y_unmasked = x_unmasked * torch.exp(s) + t

        # Объединение
        y = torch.zeros_like(x)
        y[:, self.mask] = x_masked
        y[:, ~self.mask] = y_unmasked

        # Лог детерминант
        log_det = s.sum(dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Обратный проход: y → x
        """
        y_masked = y[:, self.mask]
        y_unmasked = y[:, ~self.mask]

        h = self.shared_net(y_masked)
        s = self.scale_net(h)
        t = self.translation_net(h)

        s = torch.clamp(s, -5, 5)

        # Обратное преобразование
        x_unmasked = (y_unmasked - t) * torch.exp(-s)

        # Объединение
        x = torch.zeros_like(y)
        x[:, self.mask] = y_masked
        x[:, ~self.mask] = x_unmasked

        return x


class RealNVP(nn.Module):
    """
    Модель нормализующего потока RealNVP

    Состоит из чередующихся слоёв связывания с разными масками
    """

    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_coupling_layers: int = 8, num_hidden_layers: int = 2):
        super().__init__()

        self.dim = dim

        # Чередующиеся слои связывания
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(
                dim, hidden_dim, num_hidden_layers,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_coupling_layers)
        ])

        # Базовое распределение
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход: x → z

        Возвращает:
            z: латентное представление
            log_det: общий лог детерминант
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Обратный проход: z → x
        """
        x = z

        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисление лог-вероятности x
        """
        z, log_det = self.forward(x)

        # Лог-вероятность под базовым распределением (стандартное Гауссово)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        # Замена переменных
        log_px = log_pz + log_det

        return log_px

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Генерация сэмплов через сэмплирование z и обращение
        """
        z = torch.randn(n_samples, self.dim, device=device)
        x = self.inverse(z)
        return x
```

### Торговая Система RealNVP

```python
class RealNVPTrader:
    """
    Торговая система на основе оценки плотности RealNVP
    """

    def __init__(self,
                 model: RealNVP,
                 feature_dim: int,
                 log_prob_threshold: float = -10.0,
                 position_scale: float = 1.0):
        self.model = model
        self.feature_dim = feature_dim
        self.log_prob_threshold = log_prob_threshold
        self.position_scale = position_scale

        # Статистика признаков для нормализации
        self.feature_mean = None
        self.feature_std = None

    def fit_normalizer(self, features: np.ndarray):
        """Обучение нормализатора признаков"""
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8

    def normalize(self, features: np.ndarray) -> torch.Tensor:
        """Нормализация признаков"""
        normalized = (features - self.feature_mean) / self.feature_std
        return torch.tensor(normalized, dtype=torch.float32)

    def generate_signal(self, market_state: np.ndarray) -> dict:
        """
        Генерация торгового сигнала из текущего рыночного состояния

        Args:
            market_state: Текущие рыночные признаки

        Returns:
            dict с сигналом, уверенностью и анализом
        """
        self.model.eval()

        x = self.normalize(market_state).unsqueeze(0)

        with torch.no_grad():
            # Получение лог-вероятности и латентного представления
            z, log_det = self.model.forward(x)
            log_prob = self.model.log_prob(x).item()

            # Анализ латентного пространства
            z_np = z.squeeze().numpy()

            # Генерация сценариев
            scenarios = self.model.sample(100)
            scenario_returns = scenarios[:, 0].numpy()

        # Определение сигнала на основе плотности и латентного направления
        in_distribution = log_prob > self.log_prob_threshold

        if not in_distribution:
            # Необычные рыночные условия - снизить экспозицию
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'log_prob': log_prob,
                'in_distribution': False,
                'latent': z_np,
                'scenario_mean': scenario_returns.mean(),
                'scenario_std': scenario_returns.std()
            }

        # Использование латентного направления для сигнала
        latent_signal = np.sign(z_np[0]) if abs(z_np[0]) > 0.5 else 0.0

        # Уверенность на основе плотности
        confidence = np.clip((log_prob - self.log_prob_threshold) / 10.0, 0, 1)

        return {
            'signal': latent_signal * confidence * self.position_scale,
            'confidence': confidence,
            'log_prob': log_prob,
            'in_distribution': True,
            'latent': z_np,
            'scenario_mean': scenario_returns.mean(),
            'scenario_std': scenario_returns.std()
        }
```

### Цикл Обучения

```python
def train_realnvp(
    model: RealNVP,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5
):
    """
    Обучение модели RealNVP максимизацией лог-правдоподобия
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Перемешивание данных
        perm = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch = train_data[perm[i:i+batch_size]]

            optimizer.zero_grad()

            # Отрицательное лог-правдоподобие
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()

            loss.backward()

            # Обрезка градиентов
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

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}/{epochs}: "
                  f"Train NLL={total_loss/n_batches:.4f}, "
                  f"Val NLL={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
```

## Требования к Данным

```
Исторические OHLCV Данные:
├── Минимум: 1 год часовых данных
├── Рекомендуется: 2+ года для обучения плотности
├── Частота: от 1 часа до дневных
└── Источник: Bybit, Binance или другие биржи

Необходимые Поля:
├── timestamp
├── open, high, low, close
├── volume
└── Опционально: funding rate, open interest

Предобработка:
├── Нормализация: Z-score для каждого признака
├── Обработка выбросов: Обрезка до ±5 std
├── Пропущенные данные: Forward fill затем удаление
└── Разделение Train/Val/Test: 70/15/15
```

## Ключевые Метрики

- **Отрицательное Лог-Правдоподобие (NLL)**: Цель обучения, чем меньше тем лучше
- **Биты на Измерение**: NLL / (dim * log(2)), нормализованная метрика
- **Лог-Вероятность**: Оценка плотности для каждого сэмпла
- **Коэффициент В-Распределении**: Доля торговых дней с высокой плотностью
- **Коэффициент Шарпа**: Доходность с учётом риска
- **Максимальная Просадка**: Наибольшее падение от пика до дна

## Зависимости

```python
# Ядро
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# Глубокое Обучение
torch>=2.0.0

# Рыночные Данные
ccxt>=4.0.0

# Визуализация
matplotlib>=3.6.0
seaborn>=0.12.0

# Утилиты
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## Ожидаемые Результаты

1. **Обучение Плотности**: Модель точно захватывает распределение рыночных состояний
2. **Обнаружение Аномалий**: Области низкой вероятности помечают необычные условия
3. **Генерация Сценариев**: Реалистичные рыночные сценарии для анализа рисков
4. **Вероятностные Сигналы**: Торговые сигналы с калиброванной уверенностью
5. **Результаты Бэктеста**: Ожидаемый коэффициент Шарпа 0.8-1.5 при правильной настройке

## Ссылки

1. **Density Estimation Using Real-NVP** (Dinh et al., 2016)
   - URL: https://arxiv.org/abs/1605.08803

2. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2014)
   - URL: https://arxiv.org/abs/1410.8516

3. **Glow: Generative Flow with Invertible 1×1 Convolutions** (Kingma & Dhariwal, 2018)
   - URL: https://arxiv.org/abs/1807.03039

4. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - URL: https://arxiv.org/abs/1912.02762

5. **Flow++: Improving Flow-Based Generative Models** (Ho et al., 2019)
   - URL: https://arxiv.org/abs/1902.00275

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительной торговли RealNVP на данных криптовалют с Bybit. Смотрите директорию `rust/`.

### Особенности:
- Получение данных в реальном времени из API Bybit
- Реализация аффинных слоёв связывания
- Поток RealNVP с обратимыми преобразованиями
- Точное вычисление лог-вероятности
- Генерация сэмплов для анализа сценариев
- Фреймворк бэктестинга с комплексными метриками
- Модульный и расширяемый дизайн

## Уровень Сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание: Теории Вероятностей, Замены Переменных, Нейронных Сетей, Нормализующих Потоков, Обратимых Преобразований, Торговых Систем
