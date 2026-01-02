# Глава 337: Score Matching в Трейдинге — Обучение Распределений Данных для Прогнозирования Рынка

## Обзор

Score Matching (сопоставление градиентов) — мощный метод обучения энергетических моделей (EBM) и диффузионных моделей без необходимости вычисления дорогостоящей нормализационной константы. В трейдинге score matching позволяет изучать базовое распределение рыночных состояний, обнаруживать аномалии и генерировать прогнозы на основе градиента распределения данных ("score-функции").

В этой главе рассматривается применение score matching к финансовым рынкам: использование изученной score-функции для определения смены рыночных режимов, прогнозирования движения цен и генерации торговых сигналов.

## Основные Концепции

### Что такое Score Matching?

Score matching — это метод оценки параметров распределения вероятностей путём сопоставления **score-функции** — градиента логарифма плотности вероятности:

```
Score-функция: s(x) = ∇_x log p(x)

Ключевая идея: Score-функция показывает, как меняется
вероятность при перемещении в пространстве данных,
без необходимости знать нормализационную константу
Z = ∫ p(x)dx
```

### Почему Score Matching для Трейдинга?

1. **Обучение Распределений**: Изучение истинного распределения рыночных состояний, а не только точечных прогнозов
2. **Обнаружение Аномалий**: Области с низкой вероятностью указывают на необычные рыночные условия
3. **Идентификация Режимов**: Градиенты score-функции выявляют структуру в рыночных данных
4. **Генеративное Моделирование**: Сэмплирование из изученных распределений для анализа сценариев
5. **Устойчивость к Шуму**: Denoising score matching естественно справляется с рыночным шумом

### От Энергетических Моделей к Score Matching

```
Энергетическая Модель (EBM):
├── Функция энергии: E(x; θ)
├── Вероятность: p(x) = exp(-E(x)) / Z
├── Проблема: Z невозможно вычислить напрямую
└── Решение: Сопоставлять градиенты!

Score Matching:
├── Score модели: s_θ(x) = -∇_x E(x; θ)
├── Score данных: ∇_x log p_data(x)
├── Цель: минимизировать ||s_θ(x) - ∇_x log p_data(x)||²
└── Z не нужен!
```

## Торговая Стратегия

**Обзор стратегии:** Использование score matching для изучения распределения рыночных паттернов. Торговые сигналы генерируются на основе:
1. Score-функции, указывающей вероятные движения рынка
2. Плотности вероятности, определяющей уверенность в режиме
3. Denoising-прогнозов для направления цены

### Генерация Сигналов

```
1. Извлечение Признаков:
   - Вычисление рыночных признаков: доходность, волатильность, объём
   - Нормализация для соответствия обучающему распределению

2. Вычисление Score:
   - Запрос изученной score-функции s_θ(x)
   - Score указывает направление большей вероятности

3. Интерпретация Сигнала:
   - Положительный score доходности → ожидаем рост → Long
   - Отрицательный score доходности → ожидаем падение → Short
   - Величина score определяет силу сигнала

4. Фильтрация по Вероятности:
   - Оценка локальной плотности вероятности
   - Торговля только в областях высокой плотности (знакомых)
```

### Сигналы Входа

- **Сигнал Long**: Score-функция указывает на положительную доходность с высокой величиной
- **Сигнал Short**: Score-функция указывает на отрицательную доходность с высокой величиной
- **Фильтр Уверенности**: Торговля только когда оценённая плотность превышает порог

### Управление Рисками

- **Обнаружение Новизны**: Низкая плотность указывает на незнакомые рыночные условия → снижение экспозиции
- **Стабильность Score**: Проверка устойчивости направления score на коротком окне
- **Масштабирование по Волатильности**: Размер позиции обратно пропорционален дисперсии score

## Техническая Спецификация

### Математические Основы

#### Определение Score-Функции

Для плотности вероятности p(x) score-функция определяется как:

```
s(x) = ∇_x log p(x) = ∇_x p(x) / p(x)

Свойства:
├── Указывает в направлении большей вероятности
├── Равна нулю в локальных максимумах p(x)
├── Не зависит от нормализационной константы Z
└── Однозначно определяет p(x) с точностью до константы
```

#### Оригинальная Целевая Функция Score Matching (Hyvärinen, 2005)

```
J(θ) = E_p_data [½||s_θ(x)||² + tr(∇_x s_θ(x))]

Где:
├── s_θ(x) = score-функция модели
├── tr(∇_x s_θ(x)) = след якобиана
└── Не нужно знать ∇_x log p_data(x)!
```

#### Denoising Score Matching (Vincent, 2011)

Вместо вычисления следа якобиана возмущаем данные шумом:

```
1. Зашумление: x̃ = x + σε, где ε ~ N(0, I)
2. Обучение денойзингу: минимизация E[||s_θ(x̃) - ∇_x̃ log p(x̃|x)||²]
3. Для гауссова шума: ∇_x̃ log p(x̃|x) = -(x̃ - x)/σ² = -ε/σ

Упрощённая целевая функция:
J_DSM(θ) = E_x,ε [||s_θ(x + σε) + ε/σ||²]
```

#### Sliced Score Matching (Song et al., 2020)

Эффективный score matching с использованием случайных проекций:

```
J_SSM(θ) = E_p_data E_v [v^T ∇_x s_θ(x) v + ½(v^T s_θ(x))²]

Где v — случайный вектор направления
└── Снижает сложность с O(d²) для якобиана до O(d)
```

### Архитектурная Диаграмма

```
                    Поток Рыночных Данных
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Feature Engineering      │
            │  ├── Доходности (multi-scale)│
            │  ├── Меры волатильности     │
            │  ├── Паттерны объёма        │
            │  └── Технические индикаторы │
            └──────────────┬──────────────┘
                           │
                           ▼ Рыночное состояние x
            ┌─────────────────────────────┐
            │    Score-Сеть s_θ(x)        │
            │                             │
            │  ┌───────────────────────┐  │
            │  │   Входной слой (d)    │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │   Скрытые слои        │  │
            │  │   (MLP / ResNet)      │  │
            │  │   + Skip-соединения   │  │
            │  └───────────┬───────────┘  │
            │              ▼              │
            │  ┌───────────────────────┐  │
            │  │   Выходной слой (d)   │  │
            │  │   Score-вектор s_θ(x) │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │   Score     │ │   Оценка    │ │  Величина   │
     │ доходности  │ │ плотности   │ │   Score     │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │    Торговое Решение         │
            │  ├── Направление сигнала    │
            │  ├── Размер позиции         │
            │  ├── Уровень уверенности    │
            │  └── Параметры риска        │
            └─────────────────────────────┘
```

### Инженерия Признаков для Score Matching

```python
import numpy as np
import pandas as pd

def compute_market_state(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Создание вектора рыночного состояния для score matching
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

### Реализация Score-Сети

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNetwork(nn.Module):
    """
    Нейронная сеть для оценки score-функции s(x) = ∇_x log p(x)

    Архитектура: ResNet-подобный MLP с skip-соединениями
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Входная проекция
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Остаточные блоки
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Выходная проекция (та же размерность, что и вход, для score)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисление score-функции s_θ(x)

        Args:
            x: (batch, input_dim) входные рыночные состояния

        Returns:
            score: (batch, input_dim) score-векторы
        """
        h = self.input_proj(x)

        for block in self.res_blocks:
            h = block(h)

        score = self.output_proj(h)

        return score


class ResidualBlock(nn.Module):
    """Остаточный блок с пред-нормализацией"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))
```

### Многомасштабный Denoising Score Matching

```python
class MultiScaleScoreNetwork(nn.Module):
    """
    Score-сеть с несколькими уровнями шума

    На основе "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, num_noise_levels: int = 10,
                 sigma_min: float = 0.01, sigma_max: float = 1.0):
        super().__init__()

        self.input_dim = input_dim
        self.num_noise_levels = num_noise_levels

        # Геометрическое расписание шума
        self.register_buffer(
            'sigmas',
            torch.exp(torch.linspace(
                np.log(sigma_max), np.log(sigma_min), num_noise_levels
            ))
        )

        # Score-сеть
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.noise_embed = NoiseLevelEmbedding(hidden_dim, num_noise_levels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """
        Вычисление score на заданном уровне шума
        """
        h = self.input_proj(x)
        h = h + self.noise_embed(noise_level)

        for block in self.res_blocks:
            h = block(h)

        score = self.output_proj(h)
        return score


class DenoisingScoreMatchingLoss(nn.Module):
    """
    Функция потерь Denoising Score Matching

    J_DSM(θ) = E_{x,ε,σ} [||s_θ(x + σε) + ε/σ||²]
    """

    def __init__(self, score_network: MultiScaleScoreNetwork):
        super().__init__()
        self.score_network = score_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Выбор случайных уровней шума
        noise_levels = torch.randint(
            0, self.score_network.num_noise_levels, (batch_size,),
            device=x.device
        )
        sigmas = self.score_network.sigmas[noise_levels].unsqueeze(-1)

        # Добавление шума
        noise = torch.randn_like(x)
        x_noisy = x + sigmas * noise

        # Вычисление score
        score = self.score_network(x_noisy, noise_levels)

        # Целевой score: -noise/sigma
        target = -noise / sigmas

        # Взвешенные потери
        loss = ((score - target) ** 2).sum(dim=-1) * (sigmas.squeeze() ** 2)

        return loss.mean()
```

### Динамика Ланжевена для Сэмплирования и Прогнозирования

```python
class LangevinDynamics:
    """
    Отжигаемая динамика Ланжевена для сэмплирования из изученного распределения

    Применения:
    1. Генерация синтетических рыночных сценариев
    2. Денойзинг текущего рыночного состояния для поиска "истинного" паттерна
    3. Прогнозирование будущих состояний путём следования за score
    """

    def __init__(self, score_network: MultiScaleScoreNetwork,
                 n_steps_per_level: int = 100,
                 step_size: float = 0.01):
        self.score_network = score_network
        self.n_steps = n_steps_per_level
        self.step_size = step_size

    @torch.no_grad()
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """
        Генерация сэмплов с использованием отжигаемой динамики Ланжевена
        """
        x = torch.randn(batch_size, self.score_network.input_dim, device=device)

        for i in range(self.score_network.num_noise_levels):
            noise_level = torch.full((batch_size,), i, device=device, dtype=torch.long)
            sigma = self.score_network.sigmas[noise_level][0]
            alpha = self.step_size * (sigma ** 2)

            for _ in range(self.n_steps):
                score = self.score_network(x, noise_level)
                noise = torch.randn_like(x)
                x = x + alpha * score + np.sqrt(2 * alpha) * noise

        return x

    @torch.no_grad()
    def get_trading_signal(self, x: torch.Tensor,
                           return_idx: int = 0) -> tuple:
        """
        Получение торгового сигнала из score-функции
        """
        device = x.device
        batch_size = x.shape[0]

        noise_level = torch.full(
            (batch_size,),
            self.score_network.num_noise_levels - 1,
            device=device,
            dtype=torch.long
        )

        score = self.score_network(x, noise_level)
        return_score = score[:, return_idx]

        signal = torch.sign(return_score)
        confidence = torch.abs(return_score)
        confidence = confidence / (confidence.max() + 1e-8)

        return signal, confidence
```

### Торговая Система на Score Matching

```python
class ScoreMatchingTrader:
    """
    Торговая система на основе score matching
    """

    def __init__(self,
                 score_network: MultiScaleScoreNetwork,
                 feature_dim: int,
                 return_feature_idx: int = 0,
                 confidence_threshold: float = 0.3,
                 density_threshold: float = -5.0):
        self.score_network = score_network
        self.feature_dim = feature_dim
        self.return_idx = return_feature_idx
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold

        self.langevin = LangevinDynamics(score_network)
        self.density_estimator = ScoreBasedDensityEstimator(score_network)

    def generate_signal(self, market_state: torch.Tensor) -> dict:
        """
        Генерация торгового сигнала из текущего рыночного состояния
        """
        self.score_network.eval()
        x = market_state.unsqueeze(0)

        with torch.no_grad():
            signal, confidence = self.langevin.get_trading_signal(x, self.return_idx)
            in_dist = self.density_estimator.is_in_distribution(
                x, self.density_threshold
            )
            denoised = self.langevin.denoise(x, n_steps=20)

        signal = signal.item()
        confidence = confidence.item()
        in_dist = in_dist.item()

        if not in_dist:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'in_distribution': False,
                'raw_signal': signal,
                'denoised_state': denoised.squeeze().numpy()
            }

        if confidence < self.confidence_threshold:
            return {
                'signal': 0.0,
                'confidence': confidence,
                'in_distribution': True,
                'raw_signal': signal,
                'denoised_state': denoised.squeeze().numpy()
            }

        return {
            'signal': signal * confidence,
            'confidence': confidence,
            'in_distribution': True,
            'raw_signal': signal,
            'denoised_state': denoised.squeeze().numpy()
        }
```

## Требования к Данным

```
Исторические OHLCV Данные:
├── Минимум: 1 год часовых данных
├── Рекомендуется: 2+ года для обучения распределения
├── Частота: от 1 часа до дневной
└── Источник: Bybit, Binance или другие биржи

Обязательные Поля:
├── timestamp
├── open, high, low, close
├── volume
└── Опционально: funding rate, open interest

Предобработка:
├── Нормализация: Z-score по каждому признаку
├── Обработка выбросов: Обрезка до ±5 std
├── Пропущенные данные: Forward fill, затем удаление
└── Разбиение Train/Val/Test: 70/15/15
```

## Ключевые Метрики

- **DSM Loss**: Потери denoising score matching при обучении
- **Score Accuracy**: Корреляция между предсказанным и истинным score
- **Density Calibration**: Качество оценок вероятности
- **In-Distribution Ratio**: Доля торговых дней в изученном распределении
- **Sharpe Ratio**: Доходность с поправкой на риск
- **Maximum Drawdown**: Максимальная просадка

## Зависимости

```python
# Основные
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

## Ожидаемые Результаты

1. **Обучение Распределений**: Модель захватывает структуру рыночных состояний
2. **Обнаружение Аномалий**: Области низкой плотности сигнализируют о необычных условиях
3. **Направленные Прогнозы**: Score-функция указывает вероятное движение цены
4. **Торговля с Учётом Рисков**: Размер позиции на основе уверенности в распределении
5. **Результаты Бэктеста**: Ожидаемый Sharpe Ratio 0.8-1.5 при правильной настройке

## Источники

1. **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019)
   - URL: https://arxiv.org/abs/1907.05600

2. **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011)
   - URL: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf

3. **Estimation of Non-Normalized Statistical Models by Score Matching** (Hyvärinen, 2005)
   - URL: http://www.jmlr.org/papers/v6/hyvarinen05a.html

4. **Sliced Score Matching: A Scalable Approach to Density and Score Estimation** (Song et al., 2020)
   - URL: https://arxiv.org/abs/1905.07088

5. **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021)
   - URL: https://arxiv.org/abs/2011.13456

## Реализация на Rust

Эта глава включает полную реализацию на Rust для высокопроизводительной торговли на основе score matching с криптовалютными данными от Bybit. См. директорию `rust/`.

### Возможности:
- Получение данных в реальном времени через Bybit API
- Реализация score-сети со слоями нейронной сети
- Обучение с denoising score matching
- Динамика Ланжевена для сэмплирования и прогнозирования
- Фреймворк бэктестинга с полными метриками
- Модульный и расширяемый дизайн

## Уровень Сложности

⭐⭐⭐⭐⭐ (Эксперт)

Требуется понимание: Теории вероятностей, Градиентных методов, Нейронных сетей, Энергетических моделей, Генеративного моделирования, Торговых систем
