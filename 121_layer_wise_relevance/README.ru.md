# Глава 121: Послойное распространение релевантности (LRP)

В этой главе рассматривается **послойное распространение релевантности (Layer-wise Relevance Propagation, LRP)** — мощная техника объяснения предсказаний нейронных сетей путём декомпозиции выхода до входных признаков. В отличие от моделей «чёрного ящика», LRP обеспечивает интерпретируемость, критически важную для понимания торговых решений и построения доверия к алгоритмическим стратегиям.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ПОТОК РАСПРОСТРАНЕНИЯ LRP                    │
│                                                                 │
│  Входные признаки      Скрытые слои           Выход            │
│  ┌─────────┐           ┌─────────┐          ┌─────────┐        │
│  │ Цена    │──────────▶│ Слой 1  │─────────▶│         │        │
│  │ Объём   │◀──────────│ Слой 2  │◀─────────│Предска- │        │
│  │ RSI     │Релевант-  │ Слой 3  │Релевант- │ зание   │        │
│  └─────────┘ность      └─────────┘ность     │  f(x)   │        │
│              назад                  назад   └─────────┘        │
│                                                                 │
│  R_input = декомпозированная релевантность, показывающая        │
│            важность признаков                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Содержание

1. [Введение в LRP](#введение-в-lrp)
    * [Почему интерпретируемость важна в трейдинге](#почему-интерпретируемость-важна-в-трейдинге)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими методами объяснения](#сравнение-с-другими-методами-объяснения)
2. [Математические основы](#математические-основы)
    * [Принцип сохранения](#принцип-сохранения)
    * [Правила LRP](#правила-lrp)
    * [Распространение через слои](#распространение-через-слои)
3. [Варианты LRP](#варианты-lrp)
    * [LRP-0 (базовое правило)](#lrp-0-базовое-правило)
    * [LRP-ε (эпсилон-правило)](#lrp-ε-эпсилон-правило)
    * [LRP-γ (гамма-правило)](#lrp-γ-гамма-правило)
    * [Композитные правила](#композитные-правила)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Реализация LRP](#02-реализация-lrp)
    * [03: Обучение модели с объяснениями](#03-обучение-модели-с-объяснениями)
    * [04: Анализ торговых сигналов](#04-анализ-торговых-сигналов)
    * [05: Бэктестинг с инсайтами LRP](#05-бэктестинг-с-инсайтами-lrp)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в LRP

Послойное распространение релевантности (LRP) — это техника объяснения, представленная Bach et al. (2015), которая декомпозирует предсказания нейронной сети на вклады отдельных входных признаков. Метод удовлетворяет свойству сохранения: сумма всех входных релевантностей равна выходу сети.

### Почему интерпретируемость важна в трейдинге

В алгоритмической торговле понимание **почему** модель делает предсказания часто так же важно, как и сами предсказания:

1. **Регуляторное соответствие**: Финансовые регуляторы всё чаще требуют объяснимости моделей
2. **Управление рисками**: Понимание, какие признаки управляют предсказаниями, помогает выявить риски модели
3. **Валидация стратегии**: Проверка того, что модели изучают значимые паттерны, а не ложные корреляции
4. **Отладка и улучшение**: Выявление когда модели фокусируются на неправильных признаках
5. **Построение доверия**: Трейдерам нужна уверенность в автоматизированных решениях

```
Поток торговых решений с LRP:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Рыночные данные ──▶ Нейронная сеть ──▶ Предсказание          │
│   (OHLCV и др.)       (Чёрный ящик)      (Покупка/Продажа)     │
│                            │                    │               │
│                            ▼                    ▼               │
│                          LRP              Оценки релевантности  │
│                            │                    │               │
│                            ▼                    ▼               │
│                    "Ценовой импульс (45%)                       │
│                     Всплеск объёма (30%)                        │
│                     Перепроданность RSI (25%)"                  │
│                                                                 │
│   ════════════════════════════════════════════════════════════  │
│   Результат: Интерпретируемые торговые решения с полной         │
│              атрибуцией                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые преимущества

1. **Свойство сохранения**
   - Общая релевантность равна выходному значению
   - Релевантность не создаётся и не уничтожается
   - Математически обоснованная декомпозиция

2. **Послойная интерпретация**
   - Понимание вклада на каждом слое
   - Визуализация потока информации через сеть
   - Отладка специфичных для слоя проблем

3. **Атрибуция с учётом знака**
   - Положительная релевантность: поддерживает предсказание
   - Отрицательная релевантность: противоречит предсказанию
   - Богатая интерпретация помимо простой важности

4. **Независимость от архитектуры**
   - Работает с MLP, CNN, RNN, Трансформерами
   - Применимо к любой дифференцируемой архитектуре
   - Согласованная интерпретация между моделями

### Сравнение с другими методами объяснения

| Метод | Подход | Плюсы | Минусы |
|-------|--------|-------|--------|
| **LRP** | Распространение релевантности | Сохранение, послойность | Сложность выбора правил |
| SHAP | Значения Шепли | Теоретико-игровое обоснование | Вычислительно затратен |
| LIME | Локальные суррогаты | Не зависит от модели | Нестабильность, аппроксимация |
| Integrated Gradients | Интегрирование пути | Аксиоматичность | Выбор базовой линии |
| Attention Weights | Напрямую из модели | Легко извлечь | Только для моделей внимания |
| Gradient × Input | Простой градиент | Быстрое вычисление | Шум, нет сохранения |

## Математические основы

### Принцип сохранения

Фундаментальный принцип LRP — это **сохранение релевантности**:

```
Закон сохранения:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Для каждого слоя l:  Σ_i R_i^(l) = Σ_j R_j^(l+1)              │
│                                                                 │
│  На выходе:           R_output = f(x)  (предсказание сети)      │
│                                                                 │
│  На входе:            Σ_i R_i^input = f(x)                     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│  Общая релевантность сохраняется через все слои                 │
└─────────────────────────────────────────────────────────────────┘
```

Для сети со слоями L₁, L₂, ..., Lₙ:

```
f(x) = R^(n) = Σ R^(n-1) = Σ R^(n-2) = ... = Σ R^(0) = Σ R_input
```

### Правила LRP

LRP определяет, как перераспределить релевантность со слоя l+1 на слой l:

```python
# Общее правило LRP для нейрона j, получающего релевантность R_j:
# Распределить на вносящие вклад нейроны i на основе взвешенных активаций

R_i←j = (a_i * w_ij / Σ_k a_k * w_kj) * R_j

# Где:
# a_i = активация нейрона i в предыдущем слое
# w_ij = вес, соединяющий нейрон i с нейроном j
# R_j = релевантность в нейроне j (для распределения)
# R_i←j = релевантность, переданная от j к i
```

Общая релевантность в нейроне i — сумма всех вкладов:

```
R_i = Σ_j R_i←j
```

### Распространение через слои

```
Послойное распространение:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Слой L (выход)       Слой L-1            Слой L-2     ...     │
│  ┌─────┐              ┌─────┐             ┌─────┐              │
│  │ R_1 │ ──────────▶  │ R_1 │ ──────────▶ │ R_1 │              │
│  │ R_2 │ Правило LRP  │ R_2 │ Правило LRP │ R_2 │              │
│  │ R_3 │ ──────────▶  │ R_3 │ ──────────▶ │ R_3 │              │
│  │ ... │              │ ... │             │ ... │              │
│  └─────┘              └─────┘             └─────┘              │
│                                                                 │
│  Начало: R = [f(x), 0, 0, ...]  (только предсказанный класс)   │
│  Конец: R_input = релевантности признаков, суммирующиеся в f(x)│
└─────────────────────────────────────────────────────────────────┘
```

## Варианты LRP

### LRP-0 (базовое правило)

Простейшее правило распределяет релевантность пропорционально взвешенным активациям:

```python
def lrp_0(a, w, R_next):
    """
    LRP-0: Базовое правило распространения релевантности.

    R_i←j = (a_i * w_ij) / (Σ_k a_k * w_kj) * R_j

    Аргументы:
        a: Активации предыдущего слоя [batch, in_features]
        w: Матрица весов [in_features, out_features]
        R_next: Релевантность следующего слоя [batch, out_features]

    Возвращает:
        Релевантность для текущего слоя [batch, in_features]
    """
    z = a.unsqueeze(-1) * w.unsqueeze(0)  # [batch, in, out]
    z_sum = z.sum(dim=1, keepdim=True)     # [batch, 1, out]

    # Избежание деления на ноль
    z_sum = z_sum + 1e-9 * (z_sum == 0).float()

    # Доля вклада
    s = z / z_sum  # [batch, in, out]

    # Распределение релевантности
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)  # [batch, in]

    return R
```

**Свойства:**
- Простой и интуитивный
- Может быть нестабильным при малых знаменателях
- Нет обработки отрицательных вкладов

### LRP-ε (эпсилон-правило)

Добавляет небольшой стабилизатор ε к знаменателю для численной устойчивости:

```python
def lrp_epsilon(a, w, R_next, epsilon=0.01):
    """
    LRP-ε: Стабилизированное эпсилон-правило.

    R_i←j = (a_i * w_ij) / (Σ_k a_k * w_kj + ε * sign(Σ_k a_k * w_kj)) * R_j

    Аргументы:
        a: Активации [batch, in_features]
        w: Веса [in_features, out_features]
        R_next: Релевантность [batch, out_features]
        epsilon: Член стабилизации

    Возвращает:
        Релевантность [batch, in_features]
    """
    z = a.unsqueeze(-1) * w.unsqueeze(0)  # [batch, in, out]
    z_sum = z.sum(dim=1, keepdim=True)     # [batch, 1, out]

    # Добавление epsilon с сохранением знака
    z_sum_stabilized = z_sum + epsilon * torch.sign(z_sum)
    z_sum_stabilized = torch.where(
        z_sum == 0,
        torch.ones_like(z_sum) * epsilon,
        z_sum_stabilized
    )

    s = z / z_sum_stabilized
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)

    return R
```

**Свойства:**
- Стабильнее, чем LRP-0
- ε поглощает часть релевантности (слабое сохранение)
- Хороший выбор по умолчанию для большинства слоёв

### LRP-γ (гамма-правило)

Подчёркивает положительные вклады над отрицательными:

```python
def lrp_gamma(a, w, R_next, gamma=0.25):
    """
    LRP-γ: Гамма-правило, подчёркивающее положительные вклады.

    w+ = max(w, 0),  w- = min(w, 0)
    R_i←j = (a_i * (w_ij + γ*w_ij+)) / (Σ_k a_k * (w_kj + γ*w_kj+)) * R_j

    Аргументы:
        a: Активации [batch, in_features]
        w: Веса [in_features, out_features]
        R_next: Релевантность [batch, out_features]
        gamma: Коэффициент усиления для положительных весов

    Возвращает:
        Релевантность [batch, in_features]
    """
    w_positive = torch.clamp(w, min=0)
    w_modified = w + gamma * w_positive

    z = a.unsqueeze(-1) * w_modified.unsqueeze(0)
    z_sum = z.sum(dim=1, keepdim=True) + 1e-9

    s = z / z_sum
    R = (s * R_next.unsqueeze(1)).sum(dim=-1)

    return R
```

**Свойства:**
- Фокус на возбуждающих (положительных) свидетельствах
- γ > 0 увеличивает вес положительных вкладов
- Полезно для задач классификации

### Композитные правила

Лучшая практика — использовать разные правила для разных типов слоёв:

```python
class CompositeLRP:
    """
    Композитная стратегия LRP с использованием разных правил для разных слоёв.

    Рекомендуемая конфигурация:
    - Нижние слои (близко к входу): LRP-γ (γ=0.25)
    - Средние слои: LRP-ε (ε=0.25)
    - Верхние слои (близко к выходу): LRP-0

    Эта комбинация обеспечивает:
    - Стабильные объяснения (ε в середине)
    - Фокус на положительных свидетельствах (γ на входе)
    - Точную атрибуцию (0 на выходе)
    """

    def __init__(self, model, rules=None):
        self.model = model
        self.rules = rules or self._default_rules()

    def _default_rules(self):
        """Назначение композитных правил по умолчанию."""
        num_layers = len(list(self.model.modules()))
        rules = {}

        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                position = i / num_layers
                if position < 0.33:
                    rules[name] = ('gamma', 0.25)
                elif position < 0.66:
                    rules[name] = ('epsilon', 0.25)
                else:
                    rules[name] = ('zero', None)

        return rules
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

def prepare_lrp_data(
    symbols: List[str],
    lookback: int = 60,
    horizon: int = 1,
    features: List[str] = None
) -> Dict:
    """
    Подготовка финансовых данных для модели с анализом LRP.

    Аргументы:
        symbols: Торговые пары (напр., ['BTCUSDT', 'ETHUSDT'])
        lookback: Размер исторического окна
        horizon: Горизонт прогнозирования
        features: Названия используемых признаков

    Возвращает:
        Словарь с X (признаки), y (цели), feature_names
    """
    if features is None:
        features = [
            'log_return', 'volume_ratio', 'volatility_20',
            'rsi_14', 'macd', 'bb_position', 'atr_14'
        ]

    all_data = []

    for symbol in symbols:
        df = load_market_data(symbol)  # Из Bybit или другого источника

        # Расчёт признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['rsi_14'] = compute_rsi(df['close'], 14)
        df['macd'] = compute_macd(df['close'])
        df['bb_position'] = compute_bollinger_position(df['close'])
        df['atr_14'] = compute_atr(df, 14)

        # Цель: направление доходности следующего периода
        df['target'] = (df['log_return'].shift(-horizon) > 0).astype(int)

        all_data.append(df[features + ['target']].dropna())

    combined = pd.concat(all_data)

    # Создание последовательностей
    X, y = [], []
    for i in range(lookback, len(combined) - horizon):
        X.append(combined[features].iloc[i-lookback:i].values)
        y.append(combined['target'].iloc[i])

    return {
        'X': np.array(X),
        'y': np.array(y),
        'feature_names': features,
        'lookback': lookback,
        'horizon': horizon
    }
```

### 02: Реализация LRP

Смотрите [python/model.py](python/model.py) для полной реализации.

```python
# Основной модуль LRP

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class LRPRule(Enum):
    """Доступные правила LRP."""
    ZERO = "zero"
    EPSILON = "epsilon"
    GAMMA = "gamma"
    ALPHA_BETA = "alpha_beta"


@dataclass
class LRPConfig:
    """Конфигурация для анализа LRP."""
    epsilon: float = 0.01
    gamma: float = 0.25
    alpha: float = 2.0
    beta: float = 1.0
    default_rule: LRPRule = LRPRule.EPSILON


class LRPLinear(nn.Module):
    """Линейный слой с поддержкой LRP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: LRPConfig = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.config = config or LRPConfig()
        self.activations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = x.detach().clone()
        return self.linear(x)

    def lrp(
        self,
        R: torch.Tensor,
        rule: LRPRule = None
    ) -> torch.Tensor:
        """
        Распространение релевантности через этот слой.

        Аргументы:
            R: Релевантность от следующего слоя
            rule: Правило LRP для применения

        Возвращает:
            Релевантность для предыдущего слоя
        """
        rule = rule or self.config.default_rule
        a = self.activations
        w = self.linear.weight.T  # [in, out]

        if rule == LRPRule.ZERO:
            return self._lrp_zero(a, w, R)
        elif rule == LRPRule.EPSILON:
            return self._lrp_epsilon(a, w, R)
        elif rule == LRPRule.GAMMA:
            return self._lrp_gamma(a, w, R)
        elif rule == LRPRule.ALPHA_BETA:
            return self._lrp_alpha_beta(a, w, R)
```

### 03: Обучение модели с объяснениями

```python
# python/03_train_with_lrp.py

def train_explainable_model(
    train_loader,
    val_loader,
    input_dim: int,
    num_epochs: int = 100,
    learning_rate: float = 1e-3
) -> Tuple[LRPNetwork, Dict]:
    """
    Обучение нейронной сети с периодическим анализом объяснений LRP.

    Аргументы:
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        input_dim: Размерность входных признаков
        num_epochs: Эпохи обучения
        learning_rate: Скорость обучения

    Возвращает:
        Обученную модель и историю обучения с объяснениями
    """
    config = LRPConfig(epsilon=0.01, gamma=0.25)
    model = LRPNetwork(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=2,
        dropout=0.2,
        config=config
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'feature_importance': []
    }

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        # Вычисление объяснений LRP периодически
        if epoch % 10 == 0:
            importance = compute_feature_importance(model, val_loader)
            history['feature_importance'].append(importance)

            print(f"Эпоха {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            print(f"  Топ признаки: {importance[:3]}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history
```

### 04: Анализ торговых сигналов

```python
# python/04_signal_analysis.py

def analyze_trading_signal(
    model: LRPNetwork,
    sample: torch.Tensor,
    feature_names: List[str],
    threshold: float = 0.1
) -> Dict:
    """
    Анализ одного торгового сигнала с объяснением LRP.

    Аргументы:
        model: Обученная модель LRP
        sample: Один входной образец [1, seq_len, features]
        feature_names: Названия признаков
        threshold: Минимальная релевантность для отчёта

    Возвращает:
        Словарь с предсказанием, уверенностью и объяснениями
    """
    model.eval()

    with torch.no_grad():
        # Получение предсказания
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        # Получение объяснения LRP
        relevance = model.explain(sample, target_class=pred_class)

        # Агрегирование релевантности по временному измерению
        if len(relevance.shape) == 3:
            relevance = relevance.mean(dim=1)  # [1, features]

        relevance = relevance[0].numpy()

        # Нормализация в проценты
        total = np.abs(relevance).sum()
        relevance_pct = relevance / total * 100

        # Создание объяснения
        explanations = []
        for name, rel in zip(feature_names, relevance_pct):
            if abs(rel) >= threshold * 100:
                direction = "поддерживает" if rel > 0 else "противоречит"
                explanations.append({
                    'feature': name,
                    'relevance_pct': rel,
                    'direction': direction
                })

        # Сортировка по абсолютной релевантности
        explanations.sort(key=lambda x: abs(x['relevance_pct']), reverse=True)

    signal = "ПОКУПКА" if pred_class == 1 else "ПРОДАЖА/ДЕРЖАТЬ"

    return {
        'signal': signal,
        'confidence': confidence,
        'explanations': explanations,
        'raw_relevance': relevance,
        'feature_names': feature_names
    }
```

### 05: Бэктестинг с инсайтами LRP

```python
# python/05_backtest.py

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    confidence_threshold: float = 0.6
    max_position: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.04


def backtest_with_lrp(
    model: LRPNetwork,
    test_data,
    feature_names: List[str],
    config: BacktestConfig = None
) -> Dict:
    """
    Бэктестинг торговой стратегии с размером позиции на основе LRP.

    Стратегия использует уверенность LRP для корректировки размеров позиций:
    - Высокая уверенность в ключевых признаках = большие позиции
    - Противоречащие сигналы от признаков = меньшие позиции

    Аргументы:
        model: Обученная модель LRP
        test_data: Тестовый DataLoader
        feature_names: Названия признаков
        config: Конфигурация бэктестинга

    Возвращает:
        Результаты бэктестинга с инсайтами LRP
    """
    config = config or BacktestConfig()
    model.eval()

    capital = config.initial_capital
    position = 0.0

    history = {
        'capital': [capital],
        'positions': [],
        'returns': [],
        'signals': [],
        'explanations': []
    }

    # ... реализация бэктестинга ...

    # Расчёт метрик
    results = {
        'total_return': (capital - config.initial_capital) / config.initial_capital,
        'sharpe_ratio': calculate_sharpe(returns),
        'sortino_ratio': calculate_sortino(returns),
        'max_drawdown': calculate_max_drawdown(history['capital']),
        'win_rate': calculate_win_rate(history['signals']),
        'avg_confidence': np.mean([s['confidence'] for s in history['signals']]),
        'avg_coherence': np.mean([s['coherence'] for s in history['signals']]),
        'history': history,
        'feature_importance': calculate_backtest_importance(
            history['explanations'], feature_names
        )
    }

    return results
```

## Реализация на Rust

Смотрите [rust_lrp](rust_lrp/) для полной реализации на Rust.

```
rust_lrp/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент API биржи
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Модель LRP
│   │   ├── mod.rs
│   │   ├── config.rs       # Конфигурация модели
│   │   ├── linear.rs       # Линейный слой с LRP
│   │   ├── network.rs      # Полная сеть
│   │   └── lrp.rs          # Правила распространения LRP
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка рыночных данных
    ├── train.rs            # Обучение модели
    ├── explain.rs          # Генерация объяснений
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust_lrp

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 1h

# Обучить модель
cargo run --example train -- --epochs 100 --batch-size 32

# Сгенерировать объяснения
cargo run --example explain -- --model model.bin --input latest_data.json

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py             # Инициализация пакета
├── model.py                # Реализация сети LRP
├── data.py                 # Загрузка и предобработка данных
├── strategy.py             # Торговая стратегия и бэктестинг
├── example_usage.py        # Полный пример
└── requirements.txt        # Зависимости
```

### Быстрый старт (Python)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить полный пример
python example_usage.py

# Или использовать как библиотеку
python -c "from model import LRPNetwork; print('LRP готов!')"
```

## Лучшие практики

### Когда использовать LRP

**Идеальные случаи использования:**
- Регуляторное соответствие, требующее объяснимости модели
- Управление рисками и отладка модели
- Построение доверия к торговым решениям
- Анализ важности признаков
- Валидация модели и проверка здравого смысла

**Рассмотрите альтернативы для:**
- Предсказаний в реальном времени (LRP добавляет накладные расходы)
- Простых линейных моделей (коэффициенты уже интерпретируемы)
- Когда нужна только агрегированная важность (используйте важность перестановки)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендация | Примечания |
|----------|--------------|------------|
| `epsilon` | 0.01 - 0.1 | Меньше = точнее, больше = стабильнее |
| `gamma` | 0.1 - 0.5 | Выше = больше фокус на положительных свидетельствах |
| `alpha/beta` | 2/1 | Стандартные значения α-β правила |
| Композитные правила | Да | Разные правила для разных слоёв |

### Распространённые ошибки

1. **Использование одного правила везде**: Композитные правила работают лучше
2. **Игнорирование численной стабильности**: Всегда используйте эпсилон-стабилизацию
3. **Не нормализовать релевантности**: Сравнивайте относительные, а не абсолютные значения
4. **Забыть сохранить активации**: Требуется для обратного прохода LRP
5. **Неправильная интерпретация отрицательной релевантности**: Означает противоречие, а не неважность

## Ресурсы

### Научные статьи

- [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) — Оригинальная статья LRP (Bach et al., 2015)
- [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10) — Комплексный обзор (2019)
- [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://arxiv.org/abs/1512.02479) — Теоретические основы
- [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1711.07104) — Обзор, включающий LRP

### Реализации

- [LRP Tutorial (Heatmapping.org)](https://heatmapping.org/) — Официальный туториал и ПО
- [Captum (PyTorch)](https://captum.ai/) — Библиотека интерпретируемости Facebook с LRP
- [iNNvestigate](https://github.com/albermax/innvestigate) — Реализация LRP для Keras/TensorFlow
- [Zennit](https://github.com/chr5tphr/zennit) — Библиотека LRP для PyTorch

### Связанные главы

- [Глава 120: SHAP Trading Analysis](../120_shap_trading) — Объяснения на основе значений Шепли
- [Глава 122: Integrated Gradients](../122_integrated_gradients) — Атрибуция на основе пути
- [Глава 123: Attention Visualization](../123_attention_visualization) — Интерпретируемость трансформеров
- [Глава 124: Feature Importance](../124_feature_importance) — Методы на основе перестановок

---

## Уровень сложности

**Средний**

Предварительные требования:
- Основы нейронных сетей
- Понимание обратного распространения
- Программирование ML на PyTorch/Rust
- Базовая линейная алгебра

Путь обучения:
1. Начните с LRP-ε (наиболее стабильный)
2. Поймите свойство сохранения
3. Экспериментируйте с разными правилами
4. Применяйте композитные стратегии
5. Интегрируйте с торговыми стратегиями
