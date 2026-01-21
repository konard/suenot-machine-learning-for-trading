# Глава 89: Непрерывное мета-обучение для алгоритмической торговли

## Обзор

Непрерывное мета-обучение (Continual Meta-Learning, CML) объединяет возможности быстрой адаптации мета-обучения со способностью непрерывно учиться на новом опыте без забывания ранее приобретённых знаний. В торговле это критически важно, поскольку рынки эволюционируют со временем, и модель должна адаптироваться к новым рыночным режимам, сохраняя при этом знания об исторических паттернах, которые могут повториться.

В отличие от традиционных подходов мета-обучения, которые предполагают фиксированное распределение задач, CML работает в нестационарной среде, где само распределение задач меняется со временем. Это делает его особенно подходящим для финансовых рынков, где рыночные режимы (бычий, медвежий, высокая волатильность, низкая волатильность) динамически сменяют друг друга.

## Содержание

1. [Введение в непрерывное мета-обучение](#введение-в-непрерывное-мета-обучение)
2. [Проблема катастрофического забывания](#проблема-катастрофического-забывания)
3. [Математические основы](#математические-основы)
4. [Алгоритмы CML для торговли](#алгоритмы-cml-для-торговли)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Перспективы развития](#перспективы-развития)

---

## Введение в непрерывное мета-обучение

### Что такое непрерывное обучение?

Непрерывное обучение (Continual Learning, CL), также известное как обучение на протяжении жизни (Lifelong Learning), — это способность модели машинного обучения:

1. **Обучаться новым задачам** последовательно с течением времени
2. **Сохранять знания** о ранее изученных задачах
3. **Переносить знания** между связанными задачами

### Что такое мета-обучение?

Мета-обучение, или «обучение учиться», фокусируется на:

1. **Быстрой адаптации** к новым задачам с ограниченными данными
2. **Изучении априорных знаний**, которые обобщаются на разные задачи
3. **Независимых от задачи представлениях**, которые хорошо переносятся

### Объединение обоих: непрерывное мета-обучение

CML объединяет эти парадигмы для создания моделей, которые:

- Быстро адаптируются к новым рыночным условиям (мета-обучение)
- Помнят прошлые рыночные режимы (непрерывное обучение)
- Переносят знания между различными активами и временными периодами

### Почему CML для торговли?

Финансовые рынки представляют уникальные вызовы:

1. **Нестационарность**: динамика рынка меняется со временем
2. **Смена режимов**: бычьи/медвежьи рынки, кластеры волатильности
3. **Ограниченные данные для каждого режима**: каждое рыночное условие имеет ограниченное количество исторических примеров
4. **Повторяющиеся паттерны**: похожие рыночные условия могут повторяться спустя годы

Традиционные подходы терпят неудачу, потому что:
- Дообучение на новых данных вызывает забывание старых паттернов
- Статическое мета-обучение предполагает, что распределение задач не меняется
- Чистое непрерывное обучение не обеспечивает быструю адаптацию

---

## Проблема катастрофического забывания

### Что такое катастрофическое забывание?

Когда нейронные сети обучаются новым задачам, они часто «забывают» ранее изученные задачи. Это называется катастрофическим забыванием.

```
До: Модель знает задачи A, B, C
После изучения задачи D: Модель хорошо знает только задачу D, забывает A, B, C!
```

### Почему это важно в торговле

Представьте модель, которая изучила:
- Задача A: Паттерны бычьего рынка (2020-2021)
- Задача B: Паттерны медвежьего рынка (2022)
- Задача C: Паттерны бокового рынка (2023)

Когда она изучает задачу D (новые паттерны 2024), она может забыть, как торговать на медвежьих рынках — что может быть катастрофическим, когда медвежьи рынки вернутся!

### Решения: три подхода к CML

#### 1. Методы на основе регуляризации

Добавление ограничений для предотвращения изменений параметров, которые ухудшили бы производительность на прошлых задачах.

**Эластичная консолидация весов (EWC):**
```
L_total = L_new_task + λ Σᵢ Fᵢ (θᵢ - θ*ᵢ)²
```
Где Fᵢ — информация Фишера, измеряющая важность параметра.

#### 2. Методы на основе памяти

Хранение примеров из прошлых задач и их воспроизведение во время нового обучения.

**Воспроизведение опыта (Experience Replay):**
- Поддержание буфера памяти с прошлым опытом
- Чередование старых и новых данных во время обучения
- Может быть объединено с целями мета-обучения

#### 3. Методы на основе архитектуры

Выделение различных частей сети для различных задач.

**Прогрессивные нейронные сети:**
- Добавление новых колонок для новых задач
- Сохранение старых колонок замороженными
- Разрешение боковых соединений для переноса

---

## Математические основы

### Цель CML

Мы хотим найти параметры θ, которые минимизируют:

```
L(θ) = E_{τ~p(τ,t)} [L_τ(f_θ)]
```

Где p(τ,t) — распределение задач, которое меняется со временем t.

### Онлайн мета-обучение с памятью

Дано:
- θ: Мета-изученная инициализация
- M: Буфер памяти с данными прошлых задач
- τ_new: Новая задача

Обновление CML:

```
1. Внутренний цикл (быстрая адаптация):
   θ'_new = θ - α ∇_θ L_τ_new(f_θ)

2. Воспроизведение из памяти:
   Для τ_mem ~ M:
     θ'_mem = θ - α ∇_θ L_τ_mem(f_θ)

3. Мета-обновление (с консолидацией):
   θ ← θ + ε [ (θ'_new - θ) + β Σ_mem (θ'_mem - θ) + λ R(θ, θ_old) ]
```

Где:
- α: Внутренняя скорость обучения
- ε: Мета-скорость обучения
- β: Вес памяти
- λ: Сила регуляризации
- R(θ, θ_old): Член регуляризации, предотвращающий забывание

### Информация Фишера для EWC

Матрица информации Фишера аппроксимирует важность параметров:

```
F = E[(∇_θ log p(x|θ))²]
```

На практике вычисляется как:

```python
F_i = (1/N) Σ_n (∂L/∂θ_i)²
```

### Стратегии выбора памяти

**Резервуарная выборка:** Поддержание равномерного распределения по всем просмотренным примерам
**На основе градиента:** Сохранение примеров с наибольшей величиной градиента
**На основе разнообразия:** Максимизация охвата пространства признаков
**На основе потерь:** Сохранение примеров, с которыми модель справляется хуже всего

---

## Алгоритмы CML для торговли

### 1. Онлайн мета-обучение (OML)

Простой подход, использующий недавние задачи как распределение задач:

```python
def oml_update(model, new_data, memory_buffer):
    # Выборка недавних задач из памяти
    memory_tasks = sample_tasks(memory_buffer, k=4)

    # Объединение с новой задачей
    all_tasks = memory_tasks + [new_data]

    # Стандартное мета-обновление (например, Reptile)
    for task in all_tasks:
        adapted_params = inner_loop(model, task)
        update_meta_params(model, adapted_params)

    # Добавление новой задачи в память
    memory_buffer.add(new_data)
```

### 2. Мета-непрерывное обучение (Meta-CL)

Явное моделирование переходов между задачами:

```python
def meta_cl_update(model, task_t, task_t_minus_1):
    # Изучение динамики перехода
    transition = learn_transition(task_t_minus_1, task_t)

    # Адаптация с учётом перехода
    adapted_params = conditioned_inner_loop(model, task_t, transition)

    # Обновление с регуляризацией, учитывающей переход
    update_with_transition(model, adapted_params, transition)
```

### 3. Градиентная эпизодическая память (GEM) для мета-обучения

Проецирование градиентов для избежания забывания:

```python
def gem_meta_update(model, new_task, memory):
    # Вычисление градиента на новой задаче
    g_new = compute_meta_gradient(model, new_task)

    # Вычисление градиентов на задачах из памяти
    g_mem = [compute_meta_gradient(model, task) for task in memory]

    # Проецирование g_new, чтобы не увеличить потери на задачах из памяти
    g_projected = project_gradient(g_new, g_mem)

    # Применение спроецированного градиента
    model.params -= lr * g_projected
```

### 4. Эластичное мета-обучение (EML)

Объединение EWC с мета-обучением:

```python
def eml_update(model, tasks, fisher_info, importance_weight):
    # Стандартное мета-обновление
    meta_loss = compute_meta_loss(model, tasks)

    # Добавление регуляризации EWC
    ewc_loss = importance_weight * sum(
        fisher_info[p] * (model.params[p] - old_params[p])**2
        for p in model.params
    )

    # Комбинированное обновление
    total_loss = meta_loss + ewc_loss
    total_loss.backward()
```

---

## Реализация на Python

### Основной класс непрерывного мета-обучения

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import copy
import numpy as np
from collections import deque


class ContinualMetaLearner:
    """
    Алгоритм непрерывного мета-обучения для адаптации торговых стратегий.

    Объединяет быструю адаптацию мета-обучения со способностью
    непрерывного обучения сохранять знания при смене рыночных режимов.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        memory_size: int = 100,
        ewc_lambda: float = 0.4,
        replay_batch_size: int = 4
    ):
        """
        Инициализация непрерывного мета-обучения.

        Args:
            model: Модель нейронной сети для торговых предсказаний
            inner_lr: Скорость обучения для адаптации к конкретной задаче
            outer_lr: Скорость мета-обучения
            inner_steps: Количество шагов SGD на задачу
            memory_size: Максимальное количество задач для хранения в памяти
            ewc_lambda: Сила эластичной консолидации весов
            replay_batch_size: Количество прошлых задач для воспроизведения за обновление
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.memory_size = memory_size
        self.ewc_lambda = ewc_lambda
        self.replay_batch_size = replay_batch_size

        # Буфер памяти для прошлых задач
        self.memory_buffer: deque = deque(maxlen=memory_size)

        # Информация Фишера для EWC
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Отслеживание рыночных режимов
        self.regime_history: List[str] = []

        self.device = next(model.parameters()).device

    def compute_fisher_information(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Вычисление матрицы информации Фишера для EWC.

        Информация Фишера аппроксимирует важность параметров,
        измеряя кривизну поверхности потерь.
        """
        self.model.eval()
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.model.named_parameters()}

        for features, labels in tasks:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            predictions = self.model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # Усреднение по задачам
        for name in fisher:
            fisher[name] /= len(tasks)
            # Онлайн обновление: смешивание с предыдущей информацией Фишера
            if name in self.fisher_info:
                fisher[name] = 0.5 * fisher[name] + 0.5 * self.fisher_info[name]

        self.fisher_info = fisher
        self.optimal_params = {name: param.clone()
                               for name, param in self.model.named_parameters()}

    def ewc_penalty(self) -> torch.Tensor:
        """
        Вычисление штрафа EWC для предотвращения забывания.

        Returns:
            Член штрафа на основе взвешенного по Фишеру отклонения параметров
        """
        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                penalty += (self.fisher_info[name] *
                           (param - self.optimal_params[name]) ** 2).sum()

        return penalty

    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[nn.Module, float]:
        """
        Выполнение адаптации к конкретной задаче (внутренний цикл).

        Args:
            support_data: (признаки, метки) для адаптации
            query_data: (признаки, метки) для оценки

        Returns:
            Адаптированная модель и потери на запросе
        """
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        adapted_model.train()
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            inner_optimizer.step()

        # Оценка на запросном наборе
        adapted_model.eval()
        with torch.no_grad():
            query_features, query_labels = query_data
            query_features = query_features.to(self.device)
            query_labels = query_labels.to(self.device)
            query_predictions = adapted_model(query_features)
            query_loss = nn.MSELoss()(query_predictions, query_labels).item()

        return adapted_model, query_loss

    def meta_train_step(
        self,
        new_task: Tuple[Tuple[torch.Tensor, torch.Tensor],
                        Tuple[torch.Tensor, torch.Tensor]],
        regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Выполнение одного шага непрерывного мета-обучения.

        Объединяет:
        1. Мета-обучение на новой задаче
        2. Воспроизведение опыта из памяти
        3. Регуляризацию EWC для предотвращения забывания

        Args:
            new_task: (support_data, query_data) для новой задачи
            regime: Необязательная метка рыночного режима

        Returns:
            Словарь с метриками потерь
        """
        # Сохранение исходных параметров
        original_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Накопление обновлений параметров
        param_updates = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        # Обработка новой задачи
        support_data, query_data = new_task
        adapted_model, new_task_loss = self.inner_loop(support_data, query_data)

        with torch.no_grad():
            for (name, param), (_, adapted_param) in zip(
                self.model.named_parameters(),
                adapted_model.named_parameters()
            ):
                param_updates[name] += adapted_param - original_params[name]

        # Воспроизведение опыта из памяти
        replay_losses = []
        if len(self.memory_buffer) > 0:
            # Выборка из памяти
            replay_size = min(self.replay_batch_size, len(self.memory_buffer))
            replay_indices = np.random.choice(
                len(self.memory_buffer), replay_size, replace=False
            )

            for idx in replay_indices:
                # Сброс к исходным параметрам
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        param.copy_(original_params[name])

                mem_support, mem_query = self.memory_buffer[idx]
                adapted_model, replay_loss = self.inner_loop(mem_support, mem_query)
                replay_losses.append(replay_loss)

                with torch.no_grad():
                    for (name, param), (_, adapted_param) in zip(
                        self.model.named_parameters(),
                        adapted_model.named_parameters()
                    ):
                        param_updates[name] += adapted_param - original_params[name]

        # Применение мета-обновления с регуляризацией EWC
        total_tasks = 1 + len(replay_losses)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Мета-обновление
                new_param = (original_params[name] +
                            self.outer_lr * param_updates[name] / total_tasks)

                # Притяжение регуляризации EWC к оптимальным параметрам
                if name in self.optimal_params:
                    new_param = (new_param -
                                self.ewc_lambda * self.outer_lr *
                                (new_param - self.optimal_params[name]))

                param.copy_(new_param)

        # Добавление новой задачи в память
        self.memory_buffer.append(new_task)

        # Отслеживание режима при наличии
        if regime:
            self.regime_history.append(regime)

        # Периодическое обновление информации Фишера
        if len(self.memory_buffer) % 10 == 0:
            recent_tasks = list(self.memory_buffer)[-20:]
            task_data = [(s[0], s[1]) for s, q in recent_tasks]
            self.compute_fisher_information(task_data)

        return {
            'new_task_loss': new_task_loss,
            'replay_loss': np.mean(replay_losses) if replay_losses else 0.0,
            'memory_size': len(self.memory_buffer)
        }

    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Адаптация мета-обученной модели к новой задаче.

        Args:
            support_data: Небольшое количество данных от новой задачи
            adaptation_steps: Количество градиентных шагов

        Returns:
            Адаптированная модель
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        adapted_model.train()
        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        return adapted_model
```

### Подготовка данных для рыночных режимов

```python
import pandas as pd
from typing import Generator

def detect_market_regime(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Определение рыночного режима на основе динамики цен.

    Returns:
        Series с метками режимов: 'bull', 'bear', 'high_vol', 'low_vol'
    """
    returns = prices.pct_change()
    rolling_return = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()

    # Определение порогов
    vol_median = rolling_vol.median()

    regimes = pd.Series(index=prices.index, dtype=str)

    for i in range(window, len(prices)):
        ret = rolling_return.iloc[i]
        vol = rolling_vol.iloc[i]

        if vol > vol_median * 1.5:
            regimes.iloc[i] = 'high_vol'
        elif vol < vol_median * 0.5:
            regimes.iloc[i] = 'low_vol'
        elif ret > 0:
            regimes.iloc[i] = 'bull'
        else:
            regimes.iloc[i] = 'bear'

    return regimes


def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Создание технических признаков для торговли.
    """
    features = pd.DataFrame(index=prices.index)

    # Доходности на разных горизонтах
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Скользящие средние
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()

    # Волатильность
    features['volatility'] = prices.pct_change().rolling(window).std()

    # Моментум
    features['momentum'] = prices / prices.shift(window) - 1

    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Нормализация RSI до [-1, 1]
    features['rsi'] = (features['rsi'] - 50) / 50

    return features.dropna()
```

---

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительную генерацию торговых сигналов, подходящую для продакшн-окружений.

### Структура проекта

```
89_continual_meta_learning/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── continual/
│   │   ├── mod.rs
│   │   ├── memory.rs
│   │   ├── ewc.rs
│   │   └── learner.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── strategy.rs
│   │   └── signals.rs
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs
├── examples/
│   ├── basic_cml.rs
│   ├── regime_adaptation.rs
│   └── trading_strategy.rs
└── python/
    ├── continual_meta_learner.py
    ├── data_loader.py
    └── backtest.py
```

Смотрите директорию `src/` для полной реализации на Rust с:

- Высокопроизводительными матричными операциями
- Эффективным по памяти воспроизведением опыта
- Потокобезопасным вычислением информации Фишера
- Асинхронной загрузкой данных с Bybit
- Готовой к продакшн обработкой ошибок

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Непрерывное обучение на различных рыночных режимах

```python
import yfinance as yf

# Загрузка данных
btc = yf.download('BTC-USD', period='3y')
prices = btc['Close']
features = create_trading_features(prices)
regimes = detect_market_regime(prices)

# Инициализация CML
model = TradingModel(input_size=8)
cml = ContinualMetaLearner(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    memory_size=100,
    ewc_lambda=0.4,
    replay_batch_size=4
)

# Создание генератора задач
task_gen = create_regime_tasks(prices, features, regimes)

# Непрерывное мета-обучение
for epoch in range(500):
    (support, query), regime = next(task_gen)

    metrics = cml.meta_train_step(
        new_task=(support, query),
        regime=regime
    )

    if epoch % 50 == 0:
        print(f"Эпоха {epoch}, Режим: {regime}, "
              f"Потери новой: {metrics['new_task_loss']:.6f}, "
              f"Потери replay: {metrics['replay_loss']:.6f}, "
              f"Память: {metrics['memory_size']}")
```

### Пример 2: Оценка забывания

```python
# Создание тестовых задач из разных режимов
test_tasks_by_regime = {}
for regime in ['bull', 'bear', 'high_vol', 'low_vol']:
    regime_data = aligned[aligned['regime'] == regime]
    if len(regime_data) >= 30:
        idx = np.random.randint(0, len(regime_data) - 30)
        window = regime_data.iloc[idx:idx + 30]

        support = (
            torch.FloatTensor(window.iloc[:20][feature_cols].values),
            torch.FloatTensor(window.iloc[:20]['target'].values).unsqueeze(1)
        )
        query = (
            torch.FloatTensor(window.iloc[20:][feature_cols].values),
            torch.FloatTensor(window.iloc[20:]['target'].values).unsqueeze(1)
        )
        test_tasks_by_regime[regime] = (support, query)

# Оценка забывания
for regime, task in test_tasks_by_regime.items():
    metrics = cml.evaluate_forgetting([task])
    print(f"Режим {regime}: Потери = {metrics['mean_loss']:.6f}")
```

### Пример 3: Торговля криптовалютой на Bybit с CML

```python
import requests

def fetch_bybit_klines(symbol: str, interval: str = '1h', limit: int = 1000):
    """Получение исторических свечей с Bybit."""
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df

# Получение данных нескольких криптоактивов
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']
crypto_data = {}

for symbol in symbols:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_trading_features(prices)
    regimes = detect_market_regime(prices)
    crypto_data[symbol] = (prices, features, regimes)

# Обучение CML на нескольких криптоактивах
for symbol, (prices, features, regimes) in crypto_data.items():
    task_gen = create_regime_tasks(prices, features, regimes)

    for _ in range(100):
        (support, query), regime = next(task_gen)
        cml.meta_train_step(
            new_task=(support, query),
            regime=f"{symbol}_{regime}"
        )

    print(f"Завершено обучение на {symbol}")
```

---

## Фреймворк для бэктестинга

### Бэктестер CML

```python
class CMLBacktester:
    """
    Фреймворк бэктестинга для стратегий непрерывного мета-обучения.
    """

    def __init__(
        self,
        cml: ContinualMetaLearner,
        adaptation_window: int = 20,
        adaptation_steps: int = 5,
        prediction_threshold: float = 0.001,
        retraining_frequency: int = 20
    ):
        self.cml = cml
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps
        self.threshold = prediction_threshold
        self.retraining_frequency = retraining_frequency

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        regimes: pd.Series,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Запуск бэктеста с непрерывным мета-обучением.

        Модель быстро адаптируется к новым данным, сохраняя
        знания о прошлых рыночных режимах.
        """
        results = []
        capital = initial_capital
        position = 0

        feature_cols = list(features.columns)

        for i in range(self.adaptation_window, len(features) - 1):
            # Получение данных для адаптации
            adapt_features = torch.FloatTensor(
                features.iloc[i-self.adaptation_window:i][feature_cols].values
            )
            adapt_returns = torch.FloatTensor(
                prices.pct_change().iloc[i-self.adaptation_window+1:i+1].values
            ).unsqueeze(1)

            # Адаптация модели
            adapted = self.cml.adapt(
                (adapt_features[:-1], adapt_returns[:-1]),
                adaptation_steps=self.adaptation_steps
            )

            # Предсказание
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = adapted(current_features).item()

            # Логика торговли
            if prediction > self.threshold:
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            # Расчёт доходности
            actual_return = prices.iloc[i+1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            # Периодическое переобучение с непрерывным обучением
            if i % self.retraining_frequency == 0:
                support = (adapt_features[:-1], adapt_returns[:-1])
                query = (adapt_features[-5:], adapt_returns[-5:])
                regime = regimes.iloc[i] if not pd.isna(regimes.iloc[i]) else 'unknown'
                self.cml.meta_train_step((support, query), regime=regime)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital,
                'regime': regimes.iloc[i] if not pd.isna(regimes.iloc[i]) else 'unknown'
            })

            position = new_position

        return pd.DataFrame(results)
```

---

## Оценка производительности

### Ключевые метрики

```python
def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Расчёт метрик торговой производительности.
    """
    returns = results['position_return']

    # Базовые метрики
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Метрики с учётом риска
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    sortino_ratio = np.sqrt(252) * returns.mean() / (returns[returns < 0].std() + 1e-8)

    # Просадка
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Доля выигрышных сделок
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(results[results['position'] != 0])
    }
```

### Ожидаемая производительность

| Метрика | Стандартное мета-обучение | Непрерывное мета-обучение |
|---------|--------------------------|--------------------------|
| Коэффициент Шарпа | > 1.0 | > 1.0 |
| Забывание режимов | Высокое | Низкое |
| Скорость адаптации | Быстрая | Быстрая |
| Эффективность памяти | O(1) | O(memory_size) |
| Максимальная просадка | < 20% | < 18% |

Ключевое преимущество CML — поддержание стабильной производительности на различных рыночных режимах, даже при последовательном обучении.

---

## Перспективы развития

### 1. Непрерывное мета-обучение без явных задач

Автоматическое определение границ задач без явных меток режимов:

```python
def detect_task_shift(features: torch.Tensor, threshold: float = 0.5):
    """Определение момента, когда распределение изменилось достаточно для новой задачи."""
    # Использование статистических тестов или определение на основе представлений
    pass
```

### 2. Иерархическая память

Организация памяти по типу режима для более эффективного воспроизведения:

```python
class HierarchicalMemory:
    def __init__(self):
        self.regime_memories = {
            'bull': deque(maxlen=50),
            'bear': deque(maxlen=50),
            'high_vol': deque(maxlen=50),
            'low_vol': deque(maxlen=50)
        }
```

### 3. Мета-обучение скорости забывания

Обучение тому, когда забывать, а когда помнить:

```python
class AdaptiveEWC:
    def __init__(self):
        self.lambda_network = nn.Linear(feature_size, 1)

    def compute_lambda(self, task_features):
        """Обучение оптимальной скорости забывания на основе похожести задач."""
        return torch.sigmoid(self.lambda_network(task_features))
```

### 4. Многомасштабное обучение

Различные скорости адаптации для разных типов изменений:

- Быстрая: Дневной рыночный шум
- Средняя: Еженедельные смены режимов
- Медленная: Долгосрочные изменения структуры рынка

---

## Ссылки

1. Javed, K., & White, M. (2019). Meta-Learning Representations for Continual Learning. NeurIPS.
2. Riemer, M., et al. (2019). Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference. ICLR.
3. Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. PNAS.
4. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
5. Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
6. Parisi, G. I., et al. (2019). Continual Lifelong Learning with Neural Networks: A Review. Neural Networks.

---

## Запуск примеров

### Python

```bash
# Перейдите в директорию главы
cd 89_continual_meta_learning

# Установите зависимости
pip install torch numpy pandas yfinance scikit-learn requests

# Запустите примеры Python
python python/continual_meta_learner.py
```

### Rust

```bash
# Перейдите в директорию главы
cd 89_continual_meta_learning

# Соберите проект
cargo build --release

# Запустите примеры
cargo run --example basic_cml
cargo run --example regime_adaptation
cargo run --example trading_strategy
```

---

## Итог

Непрерывное мета-обучение решает критическую задачу в алгоритмической торговле: как быстро адаптироваться к новым рыночным условиям, сохраняя при этом ценные знания из прошлого опыта.

Ключевые преимущества:

- **Быстрая адаптация**: Изучение новых рыночных режимов с минимальными данными
- **Без забывания**: Сохранение знаний о прошлых режимах, которые могут повториться
- **Перенос обучения**: Знания переносятся между активами и временными периодами
- **Осведомлённость о режимах**: Явное моделирование переходов между рыночными режимами

Объединяя быструю адаптацию мета-обучения с сохранением памяти непрерывного обучения, CML предоставляет надёжную основу для построения торговых систем, которые улучшаются со временем, не теряя ценных исторических знаний.

---

*Предыдущая глава: [Глава 88: Мета-RL для торговли](../88_meta_rl_trading)*

*Следующая глава: [Глава 90: Оптимизация мета-градиентов](../90_meta_gradient_optimization)*
