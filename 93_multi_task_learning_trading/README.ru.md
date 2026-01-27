# Глава 93: Мультизадачное обучение для трейдинга

## Обзор

Мультизадачное обучение (Multi-Task Learning, MTL) — это парадигма машинного обучения, при которой одна модель обучается решать несколько связанных задач одновременно, разделяя представления между задачами для улучшения обобщения. Предложенная Caruana (1997), MTL использует индуктивный перенос между вспомогательными задачами для изучения более богатых признаковых представлений, чем однозадачные модели.

В алгоритмическом трейдинге MTL — естественный выбор: финансовые данные содержат взаимосвязанные сигналы для прогнозирования доходности, оценки волатильности, классификации трендов и прогнозирования объёмов. При совместном обучении этим задачам модели MTL обнаруживают общую структуру в рыночных данных, которую однозадачные подходы упускают.

## Содержание

1. [Введение в мультизадачное обучение](#введение-в-мультизадачное-обучение)
2. [Математические основы](#математические-основы)
3. [Архитектуры MTL](#архитектуры-mtl)
4. [MTL для торговых приложений](#mtl-для-торговых-приложений)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Направления развития](#направления-развития)

---

## Введение в мультизадачное обучение

### Что такое мультизадачное обучение?

Мультизадачное обучение обучает модель на нескольких связанных задачах одновременно, разделяя изученные представления. Вместо построения отдельных моделей для каждой задачи (например, одной для прогнозирования доходности, другой для прогнозирования волатильности), MTL строит единую модель, которая решает все задачи через общий базовый блок и задаче-специфичные выходные головки.

### Ключевая идея

Фундаментальная предпосылка заключается в том, что задачи, имеющие общую базовую структуру, выигрывают от совместного обучения. При обучении на нескольких задачах:

- Модель вынуждена изучать представления, обобщающиеся между задачами
- Вспомогательные задачи действуют как форма регуляризации, снижая переобучение
- Общие признаки фиксируют более устойчивые закономерности в данных

### Почему MTL для трейдинга?

Финансовые рынки дают убедительные причины для мультизадачного обучения:

- **Коррелированные сигналы**: Доходность, волатильность и объём разделяют общую рыночную динамику
- **Регуляризация**: Множественные цели предотвращают переобучение модели к шуму в отдельном таргете
- **Эффективность**: Одна модель обрабатывает множество предсказаний, снижая стоимость инференса
- **Межактивный перенос**: Паттерны, изученные на одном классе активов, информируют прогнозы для другого
- **Более богатые представления**: Общие признаки фиксируют микроструктуру рынка, которую однозадачные модели упускают

---

## Математические основы

### Целевая функция MTL

Для K задач с индивидуальными функциями потерь целевая функция MTL:

```
L_total = Σ_{k=1}^{K} w_k * L_k(θ_shared, θ_k)
```

Где:
- θ_shared: Параметры общего представления
- θ_k: Задаче-специфичные параметры для задачи k
- w_k: Вес для задачи k
- L_k: Функция потерь для задачи k

### Жёсткое разделение параметров

При жёстком разделении параметров все задачи используют общий набор скрытых слоёв (общий базовый блок), с задаче-специфичными выходными головками, ответвляющимися наверху:

```
Вход → [Общие слои] → Головка задачи 1 → Выход 1
                     → Головка задачи 2 → Выход 2
                     → Головка задачи 3 → Выход 3
```

Это наиболее распространённая архитектура MTL, обеспечивающая сильную регуляризацию, поскольку общие слои должны изучать признаки, полезные для всех задач.

### Мягкое разделение параметров

При мягком разделении параметров каждая задача имеет свою модель, но регуляризационный член поощряет сходство параметров между моделями:

```
L_total = Σ_k L_k(θ_k) + λ * Σ_{i≠j} ||θ_i - θ_j||²
```

Это более гибкий подход, но вводит дополнительные гиперпараметры.

### Стратегии взвешивания задач

Балансировка потерь задач критически важна. Основные подходы:

**1. Взвешивание неопределённостью (Kendall и др., 2018)**
```
L_total = Σ_k (1/(2σ_k²)) * L_k + log(σ_k)
```
Где σ_k — обучаемый параметр неопределённости для каждой задачи.

**2. GradNorm (Chen и др., 2018)**
Динамически корректирует веса задач для балансировки величин градиентов:
```
w_k(t+1) = w_k(t) * (||∇L_k|| / E[||∇L_k||])^α
```

**3. Динамическое среднее весов (Liu и др., 2019)**
Использует скорость изменения потерь задач для корректировки весов:
```
w_k(t) = K * exp(r_k(t-1) / T) / Σ_j exp(r_j(t-1) / T)
```
Где r_k — отношение последовательных потерь.

---

## Архитектуры MTL

### 1. Архитектура с общим основанием (Shared-Bottom)

Простейшая и наиболее распространённая архитектура MTL:

```
Входные признаки
      │
  [Общий энкодер]
      │
  ┌───┼───┐
  │   │   │
[H1] [H2] [H3]    ← Задаче-специфичные головки
  │   │   │
 O1  O2  O3       ← Выходы задач
```

### 2. Сети с перекрёстной строчкой (Cross-Stitch Networks)

Позволяют модели обучать линейные комбинации задаче-специфичных признаков на каждом слое:

```
Задача A слой i    Задача B слой i
    │                  │
    └──── Перекрёстная ┘
          строчка
    ┌──── блоки ───────┐
    │                  │
Задача A слой i+1  Задача B слой i+1
```

### 3. Multi-Gate Mixture-of-Experts (MMoE)

Использует несколько экспертных подсетей с задаче-специфичной маршрутизацией:

```
Вход → Эксперт 1 ─┐
Вход → Эксперт 2 ─┼→ Шлюз(Задача k) → Взвешенная сумма → Головка задачи k
Вход → Эксперт 3 ─┘
```

### 4. Progressive Layered Extraction (PLE)

Расширяет MMoE общими и задаче-специфичными экспертами:

```
Общие эксперты ─────┐
Эксперты задачи A ┐  ├→ Шлюз(A) → Задача A
Эксперты задачи B ┤  │
                   └──┤
                      └→ Шлюз(B) → Задача B
```

---

## MTL для торговых приложений

### Торговые задачи для MTL

Типичная модель MTL для трейдинга совместно предсказывает:

| Задача | Тип | Функция потерь | Описание |
|--------|-----|---------------|----------|
| Прогноз доходности | Регрессия | MSE | Предсказание доходности следующего периода |
| Классификация направления | Классификация | BCE | Предсказание направления цены (вверх/вниз) |
| Оценка волатильности | Регрессия | MSE | Оценка будущей волатильности |
| Прогноз объёма | Регрессия | MSE | Предсказание торгового объёма |

### Межактивный MTL

Одновременное обучение на нескольких активах:

```
Задачи = {
    (AAPL, доходность), (AAPL, волатильность), (AAPL, направление),
    (MSFT, доходность), (MSFT, волатильность), (MSFT, направление),
    (BTCUSDT, доходность), (BTCUSDT, волатильность), (BTCUSDT, направление),
}
```

### Мульти-горизонтный MTL

Одновременное предсказание на нескольких временных горизонтах:

```
Задачи = {
    1-часовая доходность, 4-часовая доходность, дневная доходность,
    1-часовая волатильность, 4-часовая волатильность, дневная волатильность
}
```

---

## Реализация на Python

### Основная мультизадачная модель

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class SharedEncoder(nn.Module):
    """Общий энкодер признаков для мультизадачного обучения."""

    def __init__(self, input_size: int, hidden_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        self.encoder = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TaskHead(nn.Module):
    """Задаче-специфичная выходная головка."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.task_type == "classification":
            out = torch.sigmoid(out)
        return out


class MultiTaskTradingModel(nn.Module):
    """
    Мультизадачная модель для трейдинга.

    Совместно предсказывает доходность, волатильность, направление и объём
    через общий энкодер и задаче-специфичные головки.
    """

    def __init__(self, input_size: int, shared_hidden: List[int],
                 head_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = SharedEncoder(input_size, shared_hidden, dropout)
        enc_out = self.encoder.output_size

        self.task_heads = nn.ModuleDict({
            "return": TaskHead(enc_out, head_hidden, 1, "regression"),
            "volatility": TaskHead(enc_out, head_hidden, 1, "regression"),
            "direction": TaskHead(enc_out, head_hidden, 1, "classification"),
            "volume": TaskHead(enc_out, head_hidden, 1, "regression"),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.encoder(x)
        return {name: head(shared) for name, head in self.task_heads.items()}


class UncertaintyWeighting(nn.Module):
    """Обучаемое взвешивание задач через гомоскедастическую неопределённость."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total += precision * loss + self.log_sigma[i]
        return total
```

### Подготовка данных

```python
import pandas as pd
import requests

def create_mtl_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """Создание технических признаков для мультизадачного обучения."""
    features = pd.DataFrame(index=prices.index)

    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()
    features['volatility'] = prices.pct_change().rolling(window).std()
    features['momentum'] = prices / prices.shift(window) - 1

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / prices

    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    features['bb_position'] = (prices - sma) / (2 * std + 1e-10)

    return features.dropna()


def fetch_bybit_klines(symbol: str, interval: str = '60', limit: int = 1000) -> pd.DataFrame:
    """Получение исторических свечей с Bybit."""
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df
```

---

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительное мультизадачное обучение для продакшн торговых систем.

### Структура проекта

```
93_multi_task_learning_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── mtl/
│   │   ├── mod.rs
│   │   └── trainer.rs
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
│   ├── basic_mtl.rs
│   ├── multi_asset.rs
│   └── trading_strategy.rs
└── python/
    ├── __init__.py
    ├── mtl_model.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Основная реализация на Rust

Полная реализация на Rust в директории `src/` включает:

- Мультизадачную нейронную сеть с общим энкодером и задаче-специфичными головками
- Взвешивание задач на основе неопределённости
- Асинхронную интеграцию с API Bybit для данных криптовалют
- Пайплайн инженерии признаков
- Движок бэктестинга с моделированием транзакционных издержек
- Продакшн-готовую обработку ошибок и логирование

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Мультизадачное обучение на данных акций

```python
import yfinance as yf

assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Подготовка признаков и целей
model = MultiTaskTradingModel(input_size=10, shared_hidden=[64, 32])
trainer = MTLTrainer(model, lr=1e-3, use_uncertainty_weighting=True)

for epoch in range(200):
    losses = trainer.train_step(X, y)
    if epoch % 50 == 0:
        print(f"Эпоха {epoch}: {losses}")
```

### Пример 2: Крипто-трейдинг с данными Bybit

```python
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_mtl_features(prices)
```

---

## Оценка производительности

### Целевые метрики

| Метрика | Целевой диапазон |
|---------|-----------------|
| Коэффициент Шарпа | > 1.0 |
| Коэффициент Сортино | > 1.5 |
| Максимальная просадка | < 20% |
| Процент выигрышных сделок | > 50% |
| Фактор прибыли | > 1.5 |

### Сравнение MTL и однозадачных моделей

В типичных экспериментах мультизадачные модели показывают:

- **Улучшение на 10-25%** коэффициента Шарпа по сравнению с однозадачными базовыми моделями
- **Лучшую калибровку** оценок волатильности при совместном обучении с прогнозом доходности
- **Снижение переобучения** благодаря регуляризационному эффекту вспомогательных задач
- **Более стабильную производительность** в различных рыночных режимах

---

## Направления развития

### 1. MTL со смесью экспертов

Использование шлюзовых сетей для динамической маршрутизации входов к специализированным экспертным подсетям.

### 2. Иерархический MTL

Организация задач в иерархию, отражающую их взаимосвязи:

- Уровень 1: Низкоуровневые признаки (доходность, объём)
- Уровень 2: Задачи среднего уровня (режимы волатильности, сила тренда)
- Уровень 3: Высокоуровневые решения (размер позиции, сигналы входа/выхода)

### 3. Градиентная хирургия

Разрешение конфликтующих градиентов между задачами путём проецирования на нормальную плоскость градиентов других задач (Yu и др., 2020).

### 4. Курикулярное обучение задачам

Постепенное увеличение сложности задач во время обучения.

### 5. Автоматическое обнаружение вспомогательных задач

Автоматическое обнаружение полезных вспомогательных задач из данных через обучаемое конструирование задач.

---

## Литература

1. Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41-75.

2. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.

3. Chen, Z., et al. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML.

4. Liu, S., Johns, E., & Davison, A. J. (2019). End-to-End Multi-Task Learning with Attention. CVPR.

5. Ma, J., et al. (2018). Modeling Task Relationships in Multi-Task Learning with Multi-Gate Mixture-of-Experts. KDD.

6. Yu, T., et al. (2020). Gradient Surgery for Multi-Task Learning. NeurIPS.

---

## Запуск примеров

### Python

```bash
cd 93_multi_task_learning_trading
pip install -r python/requirements.txt
python python/mtl_model.py
```

### Rust

```bash
cd 93_multi_task_learning_trading
cargo build --release
cargo test
cargo run --example basic_mtl
cargo run --example multi_asset
cargo run --example trading_strategy
```

---

## Итого

Мультизадачное обучение предоставляет мощную парадигму для разработки торговых моделей:

- **Общие представления**: Единый энкодер изучает признаки, полезные для множества торговых целей
- **Регуляризация**: Вспомогательные задачи предотвращают переобучение к шуму в отдельном таргете
- **Эффективность**: Одна модель производит множество предсказаний за один прямой проход
- **Устойчивость**: Совместное обучение обеспечивает более стабильную производительность в различных рыночных условиях

Одновременно обучаясь предсказывать доходность, волатильность, направление и объём, модели MTL более эффективно улавливают взаимосвязанную природу финансовых рынков, чем изолированные однозадачные подходы.

---

*Предыдущая глава: [Глава 92: Адаптация домена для финансов](../92_domain_adaptation_finance)*

*Следующая глава: [Глава 94: QuantNet Transfer Trading](../94_quantnet_transfer_trading)*
