# Глава 325: MC Dropout для трейдинга

## Обзор

Monte Carlo Dropout (MC Dropout) - это мощная техника, которая превращает стандартные нейронные сети с dropout в байесовские аппроксиматоры, позволяя оценивать неопределенность в моделях глубокого обучения. Для торговых приложений такая количественная оценка неопределенности критически важна для принятия обоснованных решений о размере позиции, управлении рисками и выборе сделок.

## Почему MC Dropout для трейдинга?

### Проблема точечных прогнозов

Традиционные нейронные сети выдают единственный прогноз (например, "цена вырастет на 2%"). Но в трейдинге нам нужно знать:
- **Насколько уверена** модель в этом прогнозе?
- **Каков диапазон** возможных исходов?
- **Можно ли доверять** этому конкретному прогнозу?

### Решение MC Dropout

MC Dropout предоставляет:
- **Неопределенность прогноза** - знание, когда модель уверена, а когда нет
- **Решения с учетом риска** - размер позиций на основе уверенности
- **Эпистемическая неопределенность** - обнаружение нетипичных рыночных условий

```
Традиционная НС:
  Вход → Модель → Единственный прогноз: "Цена +2%"

MC Dropout:
  Вход → Модель (dropout ВКЛ) → Прямой проход 1: +2.5%
                              → Прямой проход 2: +1.8%
                              → Прямой проход 3: +3.2%
                              → ...
                              → Прямой проход N: +1.5%

  Результат: Среднее = +2.2%, Std = 0.6% (доверительный интервал!)
```

## Теоретическое обоснование

### Gal & Ghahramani (2016): Dropout как байесовская аппроксимация

Основополагающая статья "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" доказала, что:

1. **Dropout при обучении** можно рассматривать как приближенный байесовский вывод
2. **Dropout при инференсе** (MC Dropout) аппроксимирует апостериорное предиктивное распределение
3. **Дисперсия прогнозов** по множеству прямых проходов представляет неопределенность модели

### Математическая формулировка

Для нейронной сети с весами `W` и применяемым dropout, каждый прямой проход выбирает различную подсеть. Для входа `x*`:

```
Предиктивное среднее:
  E[y*] ≈ (1/T) Σ f(x*, W_t)   где t = 1, ..., T

Предиктивная дисперсия:
  Var[y*] ≈ τ^(-1) + (1/T) Σ f(x*, W_t)^2 - E[y*]^2

где:
  T = количество прямых проходов
  W_t = веса с маской dropout t
  τ = точность модели (связана с weight decay)
```

### Типы неопределенности

MC Dropout улавливает два типа неопределенности:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ТИПЫ НЕОПРЕДЕЛЕННОСТИ                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ЭПИСТЕМИЧЕСКАЯ (Неопределенность модели)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ "Что модель не знает"                                    │   │
│  │ - Можно уменьшить с большим количеством данных           │   │
│  │ - Высокая в областях вне распределения                   │   │
│  │ - Дисперсия MC Dropout улавливает это                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  АЛЕАТОРИЧЕСКАЯ (Неопределенность данных)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ "Присущий шум в данных"                                  │   │
│  │ - Нельзя уменьшить (фундаментальная случайность)         │   │
│  │ - Шум микроструктуры рынка, новостные события            │   │
│  │ - Требует гетероскедастического моделирования            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Техническая архитектура

### Архитектура модели с Dropout

```
┌─────────────────────────────────────────────────────────────────┐
│                    МОДЕЛЬ MC DROPOUT                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ВХОДНОЙ СЛОЙ                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Рыночные признаки:                                        │   │
│  │   - Ценовые доходности (мультитаймфрейм)                  │   │
│  │   - Индикаторы объема                                     │   │
│  │   - Технические индикаторы (RSI, MACD, Bollinger)         │   │
│  │   - Признаки стакана                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  СКРЫТЫЕ СЛОИ (с Dropout)                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Слой 1: Linear(input, 256) → ReLU → Dropout(p=0.1)        │   │
│  │ Слой 2: Linear(256, 128) → ReLU → Dropout(p=0.1)          │   │
│  │ Слой 3: Linear(128, 64) → ReLU → Dropout(p=0.1)           │   │
│  │ Слой 4: Linear(64, 32) → ReLU → Dropout(p=0.2)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ВЫХОДНОЙ СЛОЙ                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Linear(32, output_dim)                                    │   │
│  │ Выход: Прогноз доходности или вероятности направления     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  MC DROPOUT ИНФЕРЕНС (Dropout остается ВКЛ)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Прямой проход 1 ─┐                                        │   │
│  │ Прямой проход 2 ─┼──→ Агрегация → Среднее, Дисперсия      │   │
│  │ Прямой проход 3 ─┤                                        │   │
│  │ ...             ─┤                                        │   │
│  │ Прямой проход T ─┘                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Компромисс количества прямых проходов

Количество прямых проходов `T` включает компромисс:

| Прямые проходы (T) | Оценка дисперсии | Латентность | Применение |
|-------------------|------------------|-------------|------------|
| 10-20 | Грубая оценка | ~10-20мс | Быстрый скрининг |
| 30-50 | Хорошая оценка | ~30-50мс | Стандартная торговля |
| 100+ | Точная оценка | ~100мс+ | Критические решения |

```python
# Ошибка оценки дисперсии уменьшается как 1/sqrt(T)
# Стандартная ошибка среднего: σ / sqrt(T)

T=10:  Стандартная ошибка ~ 0.316σ
T=50:  Стандартная ошибка ~ 0.141σ
T=100: Стандартная ошибка ~ 0.100σ
```

## Реализация

### Базовый слой MC Dropout

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)  # Dropout после каждого слоя
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        MC Dropout инференс: несколько прямых проходов с dropout ВКЛ
        """
        self.train()  # Держим dropout активным!

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_samples, batch, output]

        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        std = predictions.std(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'samples': predictions
        }
```

### Торговая стратегия с неопределенностью

```python
class MCDropoutTradingStrategy:
    def __init__(self, model, confidence_threshold=1.5, uncertainty_threshold=0.02):
        self.model = model
        self.confidence_threshold = confidence_threshold  # порог z-score
        self.uncertainty_threshold = uncertainty_threshold

    def generate_signal(self, features, n_samples=50):
        """Генерация торгового сигнала с фильтрацией по неопределенности"""

        result = self.model.predict_with_uncertainty(features, n_samples)

        mean_pred = result['mean'].item()
        std_pred = result['std'].item()

        # Рассчитываем z-score (сколько std от нуля)
        z_score = abs(mean_pred) / std_pred if std_pred > 0 else 0

        # Фильтр по неопределенности
        if std_pred > self.uncertainty_threshold:
            return Signal(
                direction='HOLD',
                confidence=0.0,
                reason='Неопределенность слишком высокая'
            )

        # Фильтр по уверенности
        if z_score < self.confidence_threshold:
            return Signal(
                direction='HOLD',
                confidence=z_score,
                reason='Прогноз недостаточно уверенный'
            )

        # Генерируем сигнал
        direction = 'LONG' if mean_pred > 0 else 'SHORT'

        return Signal(
            direction=direction,
            confidence=z_score,
            predicted_return=mean_pred,
            uncertainty=std_pred
        )
```

## Concrete Dropout

### Проблема фиксированной ставки Dropout

Стандартный dropout требует ручной настройки вероятности `p`. Слишком высокое значение ведет к недообучению, слишком низкое не обеспечивает достаточную регуляризацию.

### Решение Concrete Dropout (Gal et al., 2017)

Concrete Dropout обучает оптимальную ставку dropout во время тренировки:

```python
class ConcreteDropout(nn.Module):
    """
    Слой Concrete Dropout, который обучает вероятность dropout
    """
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super().__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        # Обучаемый параметр для вероятности dropout
        self.p_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def p(self):
        """Получить вероятность dropout из logit"""
        return torch.sigmoid(self.p_logit)

    def forward(self, x, layer):
        """
        Применить concrete dropout с релаксированной выборкой Бернулли
        """
        p = self.p

        if self.training:
            # Concrete/Gumbel-Softmax релаксация
            eps = 1e-7
            temp = 0.1

            unif_noise = torch.rand_like(x)
            drop_prob = (
                torch.log(p + eps)
                - torch.log(1 - p + eps)
                + torch.log(unif_noise + eps)
                - torch.log(1 - unif_noise + eps)
            )
            drop_prob = torch.sigmoid(drop_prob / temp)

            mask = 1 - drop_prob
            x = x * mask / (1 - p)

        # Применяем слой
        out = layer(x)

        # Рассчитываем термы регуляризации
        weight_reg = self.weight_regularizer * torch.sum(layer.weight ** 2)
        dropout_reg = self.dropout_regularizer * (
            p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        )

        return out, weight_reg + dropout_reg
```

### Преимущества Concrete Dropout

| Аспект | Фиксированный Dropout | Concrete Dropout |
|--------|----------------------|------------------|
| Ставка dropout | Настраивается вручную | Обучается автоматически |
| Калибровка неопределенности | Часто пере/недоуверенная | Лучше калиброванная |
| Разные слои | Одинаковая ставка везде | Специфичные ставки для слоев |
| Адаптация | Статическая | Адаптируется к сложности данных |

## Сравнение с другими методами оценки неопределенности

### Обзор методов

| Метод | Тип неопределенности | Вычислительная стоимость | Калибровка |
|-------|---------------------|-------------------------|------------|
| **MC Dropout** | Эпистемическая | Низкая (T проходов) | Хорошая |
| **Deep Ensembles** | Оба типа | Высокая (обучить N моделей) | Отличная |
| **Вариационный вывод** | Эпистемическая | Средняя | Хорошая |
| **Байесовская НС** | Оба типа | Очень высокая | Отличная |
| **Evidential NN** | Оба типа | Низкая (один проход) | Переменная |
| **Квантильная регрессия** | Алеаторическая | Низкая | Хорошая для целей |

### MC Dropout vs Deep Ensembles

```
MC Dropout:
┌─────────────────────┐
│   Одна модель       │ → Обучить один раз
│   с Dropout         │ → T прямых проходов
└─────────────────────┘
Стоимость: 1 × обучение + T × инференс

Deep Ensembles:
┌─────┐ ┌─────┐ ┌─────┐
│ M_1 │ │ M_2 │ │ M_N │ → Обучить N раз
└─────┘ └─────┘ └─────┘ → N прямых проходов
Стоимость: N × обучение + N × инференс

Компромисс:
- Ансамбли точнее, но дороги в обучении
- MC Dropout быстрее обучается, сравнимая стоимость инференса
- Ансамбли лучше улавливают функциональное разнообразие
```

## Определение размера позиции с учетом неопределенности

### Критерий Келли с неопределенностью

Критерий Келли для оптимального размера позиции можно улучшить с учетом неопределенности:

```python
def uncertainty_adjusted_kelly(predicted_return, uncertainty, risk_free_rate=0.0):
    """
    Рассчитать размер позиции с использованием критерия Келли с учетом неопределенности

    Args:
        predicted_return: Ожидаемая доходность из среднего MC Dropout
        uncertainty: Стандартное отклонение из MC Dropout
        risk_free_rate: Безрисковая ставка

    Returns:
        Оптимальная доля позиции (от 0 до 1)
    """
    # Стандартная доля Келли
    excess_return = predicted_return - risk_free_rate

    if uncertainty <= 0:
        return 0.0

    # Доля Келли: f = (μ - r) / σ²
    kelly_fraction = excess_return / (uncertainty ** 2)

    # Применяем штраф за неопределенность
    # Выше неопределенность → меньше позиция
    confidence = 1.0 / (1.0 + uncertainty)
    adjusted_fraction = kelly_fraction * confidence

    # Ограничиваем полным Келли и обеспечиваем неотрицательность
    return max(0.0, min(1.0, adjusted_fraction))


def generate_position_sizes(predictions, uncertainties, max_leverage=1.0):
    """
    Генерация размеров позиций для портфеля

    Возвращает размеры позиций, которые в сумме дают max_leverage
    """
    positions = {}

    for symbol, (pred, unc) in zip(predictions.keys(),
                                    zip(predictions.values(), uncertainties.values())):
        # Пропускаем прогнозы с высокой неопределенностью
        if unc > 0.05:  # порог неопределенности 5%
            positions[symbol] = 0.0
            continue

        kelly = uncertainty_adjusted_kelly(pred, unc)
        positions[symbol] = kelly

    # Нормализуем до max leverage
    total = sum(abs(p) for p in positions.values())
    if total > max_leverage:
        scale = max_leverage / total
        positions = {k: v * scale for k, v in positions.items()}

    return positions
```

### Правила управления рисками

```python
class UncertaintyBasedRiskManager:
    def __init__(self,
                 max_position_size=0.1,      # Макс 10% на позицию
                 uncertainty_threshold=0.03,  # Пропускаем если std > 3%
                 confidence_threshold=2.0,    # Нужна 2-сигма уверенность
                 max_portfolio_uncertainty=0.05):
        self.max_position_size = max_position_size
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.max_portfolio_uncertainty = max_portfolio_uncertainty

    def filter_signals(self, signals):
        """Фильтрация сигналов на основе неопределенности"""
        filtered = []

        for signal in signals:
            # Проверяем индивидуальную неопределенность
            if signal.uncertainty > self.uncertainty_threshold:
                continue

            # Проверяем уверенность (z-score)
            z_score = abs(signal.predicted_return) / signal.uncertainty
            if z_score < self.confidence_threshold:
                continue

            # Корректируем размер позиции на основе неопределенности
            signal.position_size = min(
                self.max_position_size,
                signal.position_size * (1 - signal.uncertainty / self.uncertainty_threshold)
            )

            filtered.append(signal)

        return filtered

    def check_portfolio_uncertainty(self, positions, uncertainties):
        """
        Рассчитать неопределенность на уровне портфеля с использованием распространения ошибок
        """
        weights = np.array(list(positions.values()))
        stds = np.array(list(uncertainties.values()))

        # Упрощенно: предполагаем некоррелированность
        portfolio_variance = np.sum((weights * stds) ** 2)
        portfolio_std = np.sqrt(portfolio_variance)

        return portfolio_std < self.max_portfolio_uncertainty
```

## Ключевые метрики

### Метрики качества неопределенности

```python
def evaluate_uncertainty_quality(predictions, uncertainties, actual_returns):
    """
    Оценить насколько хорошо калиброваны оценки неопределенности
    """
    metrics = {}

    # 1. Покрытие: Как часто истинное значение попадает в доверительный интервал?
    for confidence_level in [0.5, 0.9, 0.95]:
        z = stats.norm.ppf((1 + confidence_level) / 2)
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties

        coverage = np.mean((actual_returns >= lower) & (actual_returns <= upper))
        metrics[f'coverage_{int(confidence_level*100)}'] = coverage

    # 2. Negative Log-Likelihood (NLL)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * uncertainties**2) +
        (actual_returns - predictions)**2 / (uncertainties**2)
    )
    metrics['nll'] = nll

    # 3. Ошибка калибровки
    # Ожидаемая ошибка калибровки для оценок неопределенности
    metrics['calibration_error'] = abs(metrics['coverage_95'] - 0.95)

    # 4. Точность (средняя неопределенность - ниже лучше если калибровано)
    metrics['sharpness'] = np.mean(uncertainties)

    return metrics
```

### Метрики торговой эффективности

| Метрика | Описание | Цель |
|---------|----------|------|
| Sharpe Ratio | Доходность с учетом риска | > 2.0 |
| Sortino Ratio | Доходность с учетом нисходящего риска | > 2.5 |
| Maximum Drawdown | Наибольшее падение от пика до дна | < 15% |
| Win Rate | % прибыльных сделок | > 55% |
| Profit Factor | Валовая прибыль / Валовый убыток | > 1.5 |
| Покрытие неопределенности | Фактическое покрытие при 95% ДИ | ~95% |
| Корреляция неопределенности | Корреляция с ошибкой | > 0.3 |

## Производственные аспекты

### Пайплайн инференса

```
Пайплайн инференса MC Dropout:
├── Сбор данных (Bybit WebSocket)
│   └── Обновления OHLCV + стакана в реальном времени
├── Вычисление признаков
│   └── Технические индикаторы, доходности, объем
├── Инференс MC Dropout
│   ├── Прямой проход 1 с dropout
│   ├── Прямой проход 2 с dropout
│   ├── ...
│   └── Прямой проход T с dropout
├── Агрегация неопределенности
│   ├── Расчет среднего прогноза
│   └── Расчет стандартного отклонения
├── Генерация сигнала
│   ├── Применение порога уверенности
│   └── Применение порога неопределенности
└── Исполнение ордера
    └── Размер позиции на основе неопределенности

Бюджет латентности:
├── Сбор данных: ~10мс (WebSocket)
├── Вычисление признаков: ~5мс
├── MC Dropout (50 проходов): ~25мс (GPU) / ~100мс (CPU)
├── Генерация сигнала: ~1мс
└── Итого: ~40мс (GPU) / ~120мс (CPU)
```

### Стратегии оптимизации

```python
# Пакетируем несколько прямых проходов для эффективности GPU
def efficient_mc_dropout(model, x, n_samples=50, batch_size=10):
    """
    Эффективный запуск MC Dropout с пакетированием прямых проходов
    """
    model.train()

    all_predictions = []

    # Реплицируем вход для параллельных прямых проходов
    x_batched = x.repeat(batch_size, 1)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            x_batch = x_batched[:current_batch]

            predictions = model(x_batch)
            all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions, dim=0)

    return {
        'mean': all_predictions.mean(dim=0),
        'std': all_predictions.std(dim=0)
    }
```

## Структура директории

```
325_mc_dropout_trading/
├── README.md                    # Английская версия
├── README.ru.md                 # Этот файл
├── readme.simple.md             # Простое объяснение для начинающих
├── readme.simple.ru.md          # Русская версия для начинающих
├── python/                      # Python реализация
│   ├── requirements.txt
│   ├── mc_dropout_model.py      # Модель MC Dropout
│   ├── concrete_dropout.py      # Реализация Concrete Dropout
│   ├── trading_strategy.py      # Торговая стратегия с неопределенностью
│   ├── bybit_client.py          # Bybit API через CCXT
│   ├── backtest.py              # Движок бэктестинга
│   └── main.py                  # Главная точка входа
└── rust_mc_dropout/             # Rust реализация
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Точка входа библиотеки
    │   ├── api/                 # Клиент Bybit API
    │   ├── model/               # Реализация MC Dropout
    │   ├── features/            # Инженерия признаков
    │   ├── strategy/            # Торговая стратегия
    │   └── backtest/            # Движок бэктестинга
    └── examples/
        ├── fetch_market_data.rs
        ├── mc_dropout_inference.rs
        ├── trading_signals.rs
        └── backtest.rs
```

## Ссылки

1. **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning** (Gal & Ghahramani, 2016)
   - https://arxiv.org/abs/1506.02142
   - Основополагающая статья, доказывающая что dropout аппроксимирует байесовский вывод

2. **Concrete Dropout** (Gal et al., 2017)
   - https://arxiv.org/abs/1705.07832
   - Обучение вероятности dropout во время тренировки

3. **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?** (Kendall & Gal, 2017)
   - https://arxiv.org/abs/1703.04977
   - Разделение эпистемической и алеаторической неопределенности

4. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** (Lakshminarayanan et al., 2017)
   - https://arxiv.org/abs/1612.01474
   - Метод сравнения: глубокие ансамбли

5. **Uncertainty Quantification for Deep Learning in Finance** (Различные)
   - Применение оценки неопределенности к трейдингу

## Уровень сложности

**Средний до Продвинутого** - Требует понимания:
- Нейронных сетей и dropout
- Основ байесовского вывода
- Количественной оценки неопределенности
- Проектирования торговых стратегий
- Принципов управления рисками

## Отказ от ответственности

Эта глава предназначена **только для образовательных целей**. Торговля криптовалютами связана со значительным риском. Стратегии, описанные здесь, не были проверены в реальной торговле и должны быть тщательно протестированы перед любым применением в реальном мире. Прошлые результаты не гарантируют будущих результатов.
