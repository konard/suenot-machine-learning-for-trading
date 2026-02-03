# Глава 116: Контрфактические объяснения для трейдинга

## Обзор

Контрфактические объяснения отвечают на вопрос: "Что нужно изменить для получения другого предсказания?" В торговых приложениях эта техника предоставляет действенные инсайты о решениях модели, определяя минимальные изменения рыночных условий, которые переключили бы предсказание с "продавать" на "покупать" или наоборот. Этот подход к объяснимости критически важен для понимания моделей-чёрных ящиков и построения доверия к алгоритмическим торговым системам.

<p align="center">
<img src="https://i.imgur.com/8vZH3mL.png" width="70%">
</p>

## Содержание

1. [Введение](#введение)
   * [Что такое контрфактические объяснения?](#что-такое-контрфактические-объяснения)
   * [Зачем использовать их в трейдинге?](#зачем-использовать-их-в-трейдинге)
2. [Математические основы](#математические-основы)
   * [Формальное определение](#формальное-определение)
   * [Целевая функция оптимизации](#целевая-функция-оптимизации)
   * [Ограничения и регуляризация](#ограничения-и-регуляризация)
3. [Архитектура](#архитектура)
   * [Генератор контрфактов](#генератор-контрфактов)
   * [Валидность и близость](#валидность-и-близость)
   * [Полная архитектура системы](#полная-архитектура-системы)
4. [Реализация](#реализация)
   * [Реализация на Rust](#реализация-на-rust)
   * [Python референс](#python-референс)
5. [Применение в трейдинге](#применение-в-трейдинге)
   * [Обработка рыночных данных](#обработка-рыночных-данных)
   * [Инженерия признаков](#инженерия-признаков)
   * [Действенные инсайты](#действенные-инсайты)
6. [Бэктестинг](#бэктестинг)
7. [Ресурсы](#ресурсы)

## Введение

### Что такое контрфактические объяснения?

Контрфактическое объяснение описывает наименьшее изменение входных признаков, которое приведёт к другому предсказанию модели. В отличие от других методов объяснимости, которые говорят *почему* модель приняла решение, контрфакты говорят *что должно измениться* для другого результата.

**Пример в трейдинге:**

```
Исходные данные:
- RSI: 75 (перекупленность)
- MACD: -0.5 (медвежий)
- Объём: 1.2x от среднего
- Предсказание модели: ПРОДАВАТЬ (80% уверенность)

Контрфактическое объяснение:
"Если бы RSI был 45 вместо 75, модель предсказала бы ПОКУПАТЬ"

ИЛИ

"Если бы MACD был +0.3 вместо -0.5, модель предсказала бы ДЕРЖАТЬ"
```

Это даёт действенный инсайт — трейдеры могут понять, какие рыночные условия должны измениться для другого сигнала.

### Зачем использовать их в трейдинге?

1. **Управление рисками:** Понимание, насколько близок рынок к переключению сигнала
2. **Прозрачность:** Объяснение решений модели заинтересованным сторонам и регуляторам
3. **Улучшение стратегии:** Определение признаков, наиболее влияющих на предсказания
4. **Оценка уверенности:** Измерение стабильности предсказаний через расстояние до контрфакта
5. **Отладка:** Поиск граничных случаев, где модели ведут себя неожиданно

```
Традиционная объяснимость:        Контрфактическая объяснимость:
"RSI внёс 40% в ПРОДАВАТЬ"       "Если RSI упадёт на 30 пунктов → ПОКУПАТЬ"
"Объём внёс 20%"                 "ИЛИ если объём упадёт на 50% → ДЕРЖАТЬ"

↓                                 ↓
Понимание ПОЧЕМУ                  Понимание ЧТО-ЕСЛИ
```

## Математические основы

### Формальное определение

Для классификатора `f: X → Y` и входного экземпляра `x` с предсказанием `f(x) = y`, контрфакт `x'` удовлетворяет:

```
f(x') = y'  где y' ≠ y
```

Цель — найти `x'`, который:
1. **Валидность:** Даёт желаемое другое предсказание
2. **Близость:** Минимально отличается от исходного `x`
3. **Правдоподобность:** Представляет реалистичную точку данных

### Целевая функция оптимизации

Задача генерации контрфактов обычно формулируется как:

```
x' = argmin L(x, x')
     при условии f(x') = y_target

где L(x, x') = λ₁ · d(x, x') + λ₂ · loss(f(x'), y_target) + λ₃ · plausibility(x')
```

**Компоненты:**

- `d(x, x')`: Метрика расстояния (L1, L2 или предметно-специфичная)
- `loss(f(x'), y_target)`: Функция потерь классификации для обеспечения валидного контрфакта
- `plausibility(x')`: Обеспечивает реалистичность контрфакта

```python
# Концептуальная иллюстрация
def counterfactual_loss(x, x_cf, model, target_class, lambda1=1.0, lambda2=1.0, lambda3=0.1):
    """
    Вычисление функции потерь для оптимизации контрфакта

    Args:
        x: Исходный вход
        x_cf: Кандидат контрфакта
        model: Модель классификатора
        target_class: Желаемый класс предсказания
        lambda1: Вес для члена близости
        lambda2: Вес для члена валидности
        lambda3: Вес для члена правдоподобности

    Returns:
        Общее значение функции потерь
    """
    # Близость: Насколько отличается контрфакт?
    proximity_loss = torch.norm(x - x_cf, p=1)  # L1 расстояние

    # Валидность: Достигает ли целевой класс?
    logits = model(x_cf)
    validity_loss = F.cross_entropy(logits, target_class)

    # Правдоподобность: Реалистичен ли? (например, в пределах распределения данных)
    plausibility_loss = mahalanobis_distance(x_cf, data_mean, data_cov)

    return lambda1 * proximity_loss + lambda2 * validity_loss + lambda3 * plausibility_loss
```

### Ограничения и регуляризация

**Ограничения действенности:**

В трейдинге некоторые признаки нельзя изменить:
- Исторические цены (неизменны)
- Прошлый объём (уже произошёл)
- Внешние события (новости, регуляции)

Мы применяем маски, чтобы контрфакты изменяли только действенные признаки:

```python
# Разрешаем изменения только для перспективных признаков
actionable_mask = torch.tensor([
    0,  # past_price (неизменен)
    0,  # past_volume (неизменен)
    1,  # rsi (может измениться с движением цены)
    1,  # macd (может измениться)
    1,  # bollinger_position (может измениться)
    0,  # days_since_event (неизменен)
])

x_cf = x + actionable_mask * delta  # Изменяем только действенные признаки
```

**Регуляризация разреженности:**

Для генерации интерпретируемых объяснений мы предпочитаем контрфакты, изменяющие мало признаков:

```
L_sparse = ||x - x'||_0  ≈  Σᵢ (1 - exp(-|xᵢ - x'ᵢ|/τ))
```

Это поощряет объяснения типа "только RSI нужно изменить" вместо "RSI, MACD, объём и ширина Боллинджера — всё нужно немного изменить".

## Архитектура

### Генератор контрфактов

```
┌─────────────────────────────────────────────────────────────┐
│              Сеть генерации контрфактов                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Вход x ∈ ℝ^d                                             │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Энкодер │ → Латентное представление z                  │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ Кондиционирование   │                                  │
│    │ целевого класса     │ → z' = z ⊕ target_embedding      │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Декодер │ → Кандидат x_cf                              │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────────────────┐                                  │
│    │ Проекция            │                                  │
│    │ (действенность +    │ → Валидный контрфакт x'          │
│    │  правдоподобность)  │                                  │
│    └────┬────────────────┘                                  │
│         │                                                   │
│    Выход x' ∈ ℝ^d                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Валидность и близость

**Проверка валидности:**

Контрфакт валиден, если модель предсказывает целевой класс:

```python
def is_valid(model, x_cf, target_class, threshold=0.5):
    """Проверка, достигает ли контрфакт целевого предсказания"""
    with torch.no_grad():
        probs = F.softmax(model(x_cf), dim=-1)
        return probs[target_class] > threshold
```

**Метрики близости:**

| Метрика | Формула | Применение |
|---------|---------|------------|
| L1 (Манхэттен) | Σ\|xᵢ - x'ᵢ\| | Разреженные изменения |
| L2 (Евклид) | √Σ(xᵢ - x'ᵢ)² | Плавные изменения |
| L0 (Счёт) | Σ𝟙[xᵢ ≠ x'ᵢ] | Минимум признаков |
| Махаланобис | √((x-x')ᵀΣ⁻¹(x-x')) | Учёт распределения |

### Полная архитектура системы

```
┌────────────────────────────────────────────────────────────────┐
│      Система контрфактических объяснений для трейдинга          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐                                              │
│  │ Рыночные     │──┐                                           │
│  │ данные(OHLCV)│  │                                           │
│  └──────────────┘  │    ┌─────────────────┐                    │
│                    ├───→│ Инженерия       │                    │
│  ┌──────────────┐  │    │ признаков       │                    │
│  │ Технические  │──┤    └────────┬────────┘                    │
│  │ индикаторы   │  │             │                             │
│  └──────────────┘  │    ┌────────┴────────┐                    │
│                    │    │ Торговая модель │                    │
│  ┌──────────────┐  │    │ (Чёрный ящик)   │                    │
│  │ Сентимент    │──┘    └────────┬────────┘                    │
│  └──────────────┘               │                              │
│                         ┌───────┴───────┐                      │
│                         │  Предсказание │                      │
│                         │(КУПИТЬ/ПРОДАТЬ)│                     │
│                         └───────┬───────┘                      │
│                                 │                              │
│                         ┌───────┴───────────────┐              │
│                         │ Генератор             │              │
│                         │ контрфактов           │              │
│                         └───────┬───────────────┘              │
│                                 │                              │
│                         ┌───────┴───────────────┐              │
│                         │ Объяснения:           │              │
│                         │ "Если RSI → 45, то    │              │
│                         │  предсказание = КУПИТЬ"│             │
│                         └───────────────────────┘              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Реализация

### Реализация на Rust

Директория [rust_counterfactual](rust_counterfactual/) содержит модульную реализацию на Rust:

```
rust_counterfactual/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Главный модуль библиотеки
│   ├── api/
│   │   ├── mod.rs              # Модуль API
│   │   └── bybit.rs            # Клиент Bybit API
│   ├── data/
│   │   ├── mod.rs              # Модуль данных
│   │   ├── loader.rs           # Загрузка данных
│   │   └── features.rs         # Инженерия признаков
│   ├── model/
│   │   ├── mod.rs              # Модуль модели
│   │   ├── classifier.rs       # Торговый классификатор
│   │   └── config.rs           # Конфигурация модели
│   ├── counterfactual/
│   │   ├── mod.rs              # Модуль контрфактов
│   │   ├── generator.rs        # Генератор КФ
│   │   ├── optimizer.rs        # Алгоритмы оптимизации
│   │   ├── constraints.rs      # Ограничения действенности
│   │   └── metrics.rs          # Метрики близости
│   └── strategy/
│       ├── mod.rs              # Модуль стратегии
│       ├── signals.rs          # Торговые сигналы
│       └── backtest.rs         # Фреймворк бэктестинга
└── examples/
    ├── fetch_data.rs           # Загрузка данных Bybit
    ├── train_classifier.rs     # Обучение торговой модели
    ├── generate_cf.rs          # Генерация контрфактов
    └── backtest.rs             # Бэктест стратегии
```

### Быстрый старт с Rust

```bash
# Перейдите в проект Rust
cd 116_counterfactual_explanations/rust_counterfactual

# Загрузка данных криптовалют с Bybit
cargo run --example fetch_data

# Обучение торгового классификатора
cargo run --example train_classifier

# Генерация контрфактических объяснений
cargo run --example generate_cf

# Запуск полного бэктеста
cargo run --example backtest
```

### Python референс

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CounterfactualGenerator(nn.Module):
    """
    Нейросетевой генератор контрфактов
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Эмбеддинг класса
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, target_class):
        # Кодируем вход
        z = self.encoder(x)

        # Получаем эмбеддинг целевого класса
        class_emb = self.class_embedding(target_class)

        # Конкатенируем и декодируем
        z_combined = torch.cat([z, class_emb], dim=-1)
        delta = self.decoder(z_combined)

        # Генерируем контрфакт как возмущение
        x_cf = x + delta

        return x_cf


class CounterfactualOptimizer:
    """
    Градиентный оптимизатор контрфактов
    """
    def __init__(self, model, lambda_proximity=1.0, lambda_validity=1.0,
                 lambda_sparsity=0.1, actionable_mask=None):
        self.model = model
        self.lambda_proximity = lambda_proximity
        self.lambda_validity = lambda_validity
        self.lambda_sparsity = lambda_sparsity
        self.actionable_mask = actionable_mask

    def generate(self, x, target_class, num_steps=100, lr=0.01):
        """
        Генерация контрфакта через градиентный спуск

        Args:
            x: Исходный входной тензор (batch, features)
            target_class: Желаемый класс предсказания
            num_steps: Шаги оптимизации
            lr: Скорость обучения

        Returns:
            x_cf: Контрфактическое объяснение
            info: Словарь с информацией об оптимизации
        """
        x = x.clone().detach()
        x_cf = x.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)

        best_cf = None
        best_loss = float('inf')

        for step in range(num_steps):
            optimizer.zero_grad()

            # Вычисляем потери
            loss, loss_dict = self._compute_loss(x, x_cf, target_class)

            # Обратное распространение
            loss.backward()

            # Применяем маску действенности к градиентам
            if self.actionable_mask is not None:
                x_cf.grad.data *= self.actionable_mask

            optimizer.step()

            # Отслеживаем лучший валидный контрфакт
            if self._is_valid(x_cf, target_class) and loss.item() < best_loss:
                best_loss = loss.item()
                best_cf = x_cf.clone().detach()

        return best_cf if best_cf is not None else x_cf.detach(), {
            'final_loss': loss.item(),
            'is_valid': self._is_valid(x_cf, target_class),
            'num_features_changed': self._count_changes(x, x_cf)
        }

    def _compute_loss(self, x, x_cf, target_class):
        """Вычисление комбинированной функции потерь для оптимизации контрфакта"""
        # Потери близости (L1)
        proximity = torch.norm(x - x_cf, p=1)

        # Потери валидности (кросс-энтропия к цели)
        logits = self.model(x_cf)
        validity = F.cross_entropy(logits, target_class)

        # Потери разреженности (аппроксимация L0)
        sparsity = torch.sum(1 - torch.exp(-torch.abs(x - x_cf) / 0.1))

        total_loss = (
            self.lambda_proximity * proximity +
            self.lambda_validity * validity +
            self.lambda_sparsity * sparsity
        )

        return total_loss, {
            'proximity': proximity.item(),
            'validity': validity.item(),
            'sparsity': sparsity.item()
        }

    def _is_valid(self, x_cf, target_class, threshold=0.5):
        """Проверка, достигает ли контрфакт целевого класса"""
        with torch.no_grad():
            probs = F.softmax(self.model(x_cf), dim=-1)
            return probs[0, target_class].item() > threshold

    def _count_changes(self, x, x_cf, threshold=0.01):
        """Подсчёт количества изменённых признаков"""
        return (torch.abs(x - x_cf) > threshold).sum().item()


class TradingClassifier(nn.Module):
    """
    Простой торговый классификатор (модель для объяснения)
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)
```

## Применение в трейдинге

### Обработка рыночных данных

Для торговли криптовалютами с Bybit:

```python
CRYPTO_UNIVERSE = {
    'major': ['BTCUSDT', 'ETHUSDT'],
    'large_cap': ['SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'],
    'mid_cap': ['AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT'],
}

FEATURES = {
    'price': ['close', 'returns', 'log_returns'],
    'technical': ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower'],
    'volume': ['volume', 'volume_sma', 'volume_ratio'],
    'derived': ['volatility', 'momentum', 'trend_strength']
}
```

### Инженерия признаков

```python
def prepare_features(df, lookback=20):
    """
    Подготовка признаков для торговой модели

    Args:
        df: DataFrame с OHLCV
        lookback: Период для технических индикаторов

    Returns:
        X: Матрица признаков
        feature_names: Список имён признаков
    """
    features = {}

    # Ценовые признаки
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close']).diff()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()

    # Полосы Боллинджера
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    features['bb_position'] = (df['close'] - sma) / (2 * std)

    # Объём
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # Волатильность
    features['volatility'] = features['returns'].rolling(window=20).std()

    # Объединяем
    X = pd.DataFrame(features).dropna()

    return X.values, list(features.keys())
```

### Действенные инсайты

```python
def explain_prediction(model, x, feature_names, target_class=None):
    """
    Генерация контрфактического объяснения для торгового предсказания

    Args:
        model: Обученный классификатор
        x: Входные признаки (1, num_features)
        feature_names: Список имён признаков
        target_class: Желаемый альтернативный класс (None = переключить предсказание)

    Returns:
        explanation: Человекочитаемое объяснение
        counterfactual: Экземпляр контрфакта
    """
    # Получаем исходное предсказание
    with torch.no_grad():
        orig_probs = F.softmax(model(x), dim=-1)
        orig_class = orig_probs.argmax().item()

    # Определяем целевой класс
    class_names = ['ПРОДАВАТЬ', 'ДЕРЖАТЬ', 'ПОКУПАТЬ']
    if target_class is None:
        target_class = (orig_class + 1) % 3  # Переключаем на другой класс

    # Генерируем контрфакт
    optimizer = CounterfactualOptimizer(model, actionable_mask=None)
    x_cf, info = optimizer.generate(x, torch.tensor([target_class]))

    # Находим изменённые признаки
    changes = []
    for i, (orig, cf, name) in enumerate(zip(x[0], x_cf[0], feature_names)):
        diff = cf - orig
        if abs(diff) > 0.01:
            direction = "увеличить" if diff > 0 else "уменьшить"
            changes.append(f"  - {name}: {orig:.3f} → {cf:.3f} ({direction} на {abs(diff):.3f})")

    explanation = f"""
Контрфактическое объяснение
===========================
Исходное предсказание: {class_names[orig_class]} ({orig_probs[0, orig_class]:.1%} уверенность)
Целевое предсказание: {class_names[target_class]}

Чтобы изменить предсказание с {class_names[orig_class]} на {class_names[target_class]}:
{chr(10).join(changes) if changes else "  Валидный контрфакт не найден"}

Количество изменённых признаков: {info['num_features_changed']}
Контрфакт валиден: {info['is_valid']}
"""

    return explanation, x_cf
```

**Пример вывода:**

```
Контрфактическое объяснение
===========================
Исходное предсказание: ПРОДАВАТЬ (78.5% уверенность)
Целевое предсказание: ПОКУПАТЬ

Чтобы изменить предсказание с ПРОДАВАТЬ на ПОКУПАТЬ:
  - rsi: 72.500 → 45.200 (уменьшить на 27.300)
  - macd: -0.450 → 0.120 (увеличить на 0.570)

Количество изменённых признаков: 2
Контрфакт валиден: True
```

## Бэктестинг

### Ключевые метрики

| Метрика | Описание | Хорошее значение |
|---------|----------|------------------|
| **Валидность контрфакта** | % КФ, достигающих целевого класса | > 90% |
| **Разреженность** | Среднее изменённых признаков | < 3 |
| **Близость** | Среднее L1 расстояние | Зависит от масштаба |
| **Правдоподобность** | % КФ в пределах распределения | > 80% |
| **Стабильность** | Согласованность для похожих входов | Высокая |

### Ожидаемые результаты

| Метод | Валидность | Разреженность | Близость | Правдоподобность |
|-------|------------|---------------|----------|------------------|
| Случайный поиск | 45% | 8.2 | 5.4 | 30% |
| Градиентный спуск | 85% | 4.1 | 2.1 | 65% |
| **Нейросетевой КФ генератор** | **92%** | **2.3** | **1.5** | **82%** |

### Торговая стратегия с контрфактическими инсайтами

```
Правила входа:
├── Уверенность предсказания модели > 60%
├── Расстояние до контрфакта > порога (стабильное предсказание)
├── Ключевые признаки не близки к границе переключения
└── Нет конфликтующих сигналов в коррелированных активах

Правила выхода:
├── Предсказание модели переключилось
├── Расстояние до контрфакта упало ниже порога
├── Стоп-лосс: -2%
├── Тейк-профит: +4%
└── По времени: выход через 12 часов при отсутствии чёткого направления

Управление рисками:
├── Расстояние до контрфакта указывает на стабильность предсказания
├── Меньшее расстояние → меньший размер позиции
├── Отслеживание признаков, ближайших к границам
└── Оповещение при приближении рыночных условий к точкам переключения
```

## Ресурсы

### Научные статьи

1. **Counterfactual Explanations without Opening the Black Box**
   - arXiv: [1711.00399](https://arxiv.org/abs/1711.00399)
   - Ключевые идеи: Контрфакты с минимальным возмущением

2. **Diverse Counterfactual Explanations for Anomaly Detection**
   - Множественные контрфакты для всестороннего понимания

3. **Actionable Recourse in Machine Learning**
   - Фокус на реалистичных, действенных изменениях

### Книги

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (Christoph Molnar)
- [Explainable AI: Interpreting and Explaining ML Models](https://www.springer.com/gp/book/9783030289539)

### Связанные главы

- [Глава 115: SHAP Values](../115_shap_values) — Объяснения важности признаков
- [Глава 117: LIME Explanations](../117_lime_explanations) — Локальные суррогатные модели
- [Глава 118: Integrated Gradients](../118_integrated_gradients) — Методы атрибуции

## Зависимости

### Rust

```toml
ndarray = "0.16"
ndarray-linalg = "0.16"
reqwest = "0.12"
tokio = "1.0"
serde = "1.0"
serde_json = "1.0"
chrono = "0.4"
rand = "0.8"
anyhow = "1.0"
```

### Python

```python
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
shap>=0.42.0  # Для сравнения
```

## Уровень сложности

**Средний**

**Необходимые знания:**
- Базовые концепции машинного обучения
- Градиентная оптимизация
- Анализ временных рядов
- Принципы управления рисками

---

*Этот материал предназначен для образовательных целей. Торговля криптовалютами связана со значительным риском. Прошлые результаты не гарантируют будущих.*
