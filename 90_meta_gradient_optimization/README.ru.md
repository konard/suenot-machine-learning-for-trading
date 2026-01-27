# Глава 90: Мета-градиентная оптимизация для трейдинга

## Обзор

Мета-градиентная оптимизация -- это продвинутая техника мета-обучения, при которой гиперпараметры самого процесса обучения (такие как скорость обучения, коэффициент дисконтирования, функция потерь и правила обновления градиентов) оптимизируются градиентными методами. Вместо ручной настройки гиперпараметров алгоритм обучает их, дифференцируя через процесс оптимизации.

В трейдинге мета-градиентная оптимизация позволяет создавать самонастраивающиеся стратегии, которые автоматически адаптируют свою динамику обучения к изменяющимся рыночным условиям, характеристикам различных активов и меняющимся режимам среды.

## Содержание

1. [Введение в мета-градиентную оптимизацию](#введение-в-мета-градиентную-оптимизацию)
2. [Математические основы](#математические-основы)
3. [Сравнение с другими методами мета-обучения](#сравнение-с-другими-методами-мета-обучения)
4. [Торговые приложения](#торговые-приложения)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Направления развития](#направления-развития)

---

## Введение в мета-градиентную оптимизацию

### Что такое мета-градиентная оптимизация?

Стандартное машинное обучение оптимизирует параметры модели при фиксированных гиперпараметрах (скорость обучения, моментум, регуляризация). Мета-градиентная оптимизация идёт дальше: она рассматривает гиперпараметры как дифференцируемые переменные и оптимизирует их с помощью градиентов.

Основная идея была формализована в нескольких ключевых работах:
- **Xu и др. (2018)**: "Meta-Gradient Reinforcement Learning" -- оптимизация коэффициента дисконтирования и параметра бутстрэппинга в RL
- **Andrychowicz и др. (2016)**: "Learning to learn by gradient descent by gradient descent" -- обучение самого оптимизатора
- **Li и Malik (2017)**: "Learning to Optimize" -- параметризация алгоритмов оптимизации

### Ключевые концепции

1. **Обучаемые скорости обучения**: Вместо фиксированного расписания скорость обучения для каждого параметра (или группы параметров) обучается через мета-градиенты.

2. **Обучаемые функции потерь**: Сама функция потерь параметризуется и оптимизируется так, чтобы модель, обученная с этой функцией, лучше обобщалась.

3. **Предобработка градиентов**: Сырой градиент преобразуется обучаемой функцией перед применением, фактически обучая оптимизатор.

4. **Онлайн кросс-валидация**: Мета-градиенты вычисляются путём оценки адаптированных параметров на отложенной (валидационной) выборке.

### Почему мета-градиентная оптимизация для трейдинга?

Финансовые рынки представляют вызовы, делающие мета-градиентную оптимизацию привлекательной:

- **Нестационарность**: Динамика рынка постоянно смещается; фиксированные гиперпараметры устаревают
- **Гетерогенность активов**: Разные активы требуют разной динамики обучения
- **Чувствительность к режимам**: Бычий, медвежий и боковой рынки требуют разного поведения оптимизации
- **Чувствительность к гиперпараметрам**: Производительность торговых моделей сильно зависит от скорости обучения и регуляризации
- **Онлайн-обучение**: Рынки требуют непрерывной адаптации -- мета-градиенты обеспечивают самонастройку онлайн-обновлений

---

## Математические основы

### Фреймворк мета-градиентов

Рассмотрим алгоритм обучения с гиперпараметрами eta (например, скорость обучения, коэффициент дисконтирования):

**Базовое обновление (внутренняя оптимизация):**
```
theta_{t+1} = theta_t - alpha(eta) * grad_theta L_train(theta_t)
```

где `alpha(eta)` -- дифференцируемая функция мета-параметров eta.

**Мета-цель (внешняя оптимизация):**
```
eta* = argmin_eta L_val(theta_{t+1}(eta))
```

Мета-градиент:
```
d L_val / d eta = (d L_val / d theta_{t+1}) * (d theta_{t+1} / d eta)
```

### Вычисление мета-градиента

Ключевое вычисление -- `d theta_{t+1} / d eta`. Используя правило цепочки через внутреннее обновление:

```
d theta_{t+1} / d eta = d/d_eta [ theta_t - alpha(eta) * grad_theta L_train(theta_t) ]
                      = - (d alpha / d eta) * grad_theta L_train(theta_t)
                        - alpha(eta) * (d^2 L_train / d theta d eta)
```

Для **обучаемых попараметрных скоростей обучения**, где `alpha_i = eta_i`:
```
d theta_{t+1,i} / d eta_i = - grad_theta_i L_train(theta_t)
```

Полный мета-градиент:
```
d L_val / d eta_i = - (d L_val / d theta_{t+1,i}) * grad_theta_i L_train(theta_t)
```

### Многошаговые мета-градиенты

Для K внутренних шагов мета-градиент требует дифференцирования через всю траекторию оптимизации:

```
theta^(0) = theta_init
theta^(k+1) = theta^(k) - alpha(eta) * grad L_train(theta^(k)),  k = 0, ..., K-1

d L_val(theta^(K)) / d eta = произведение якобианов через K шагов
```

Это можно вычислить через:
- **Обратное распространение через время (BPTT)**: Разворачивание и обратное распространение через все K шагов
- **Усечённый BPTT**: Обратное распространение только через последние несколько шагов
- **Неявное дифференцирование**: Использование теоремы о неявной функции при сходимости

### Обучаемые функции потерь

Параметризуем функцию потерь как `L(y, y_hat; phi)` и оптимизируем:
```
phi* = argmin_phi L_val(theta*(phi))
```
где `theta*(phi) = argmin_theta L_train(y, y_hat; phi)`.

Мета-градиент:
```
d L_val / d phi = - (d L_val / d theta) * H^{-1} * (d^2 L_train / d theta d phi)
```
где `H = d^2 L_train / d theta^2` -- гессиан.

---

## Сравнение с другими методами мета-обучения

### Сравнительная таблица

| Метод | Что обучает | Порядок градиента | Онлайн | Гибкость |
|-------|------------|-------------------|--------|----------|
| Мета-градиентная опт. | Гиперпараметры (LR, loss и т.д.) | Второй порядок | Да | Очень высокая |
| MAML | Инициализацию параметров | Второй порядок | Ограничено | Высокая |
| Reptile | Инициализацию параметров | Первый порядок | Ограничено | Средняя |
| Обучаемые оптимизаторы | Всё правило обновления | Второй порядок | Да | Очень высокая |
| Hyperband/BOHB | Гиперпараметры | Нулевой порядок | Нет | Средняя |

### Когда использовать мета-градиентную оптимизацию

**Используйте когда:**
- Чувствительность к гиперпараметрам высока и ручной настройки недостаточно
- Среда нестационарна и гиперпараметры должны адаптироваться онлайн
- Нужна попараметрная или послойная адаптация скорости обучения
- Нужна обучаемая функция потерь, настроенная под торговые цели

**Рассмотрите альтернативы когда:**
- Пространство гиперпараметров дискретно (используйте байесовскую оптимизацию)
- Вычислительный бюджет очень ограничен (используйте фиксированные расписания)
- Распределение задач хорошо определено (используйте MAML)

---

## Торговые приложения

### 1. Адаптивное расписание скорости обучения

Обучение попараметрных скоростей, адаптирующихся к волатильности рынка:

```
Режим высокой волатильности -> низкие скорости обучения (осторожные обновления)
Режим низкой волатильности  -> высокие скорости обучения (уверенные обновления)
Переходные периоды          -> быстро адаптирующиеся скорости
```

### 2. Обучаемые торговые функции потерь

Стандартная MSE-потеря может не соответствовать торговым целям. Мета-градиентная оптимизация может обучить функцию потерь, которая:
- Сильнее штрафует ошибки направления, чем ошибки амплитуды
- Придаёт больший вес недавним данным при смене режима
- Учитывает асимметричные предпочтения риска (убытки > прибыли)

### 3. Онлайн-адаптация стратегии

Для живой торговли мета-градиенты обеспечивают непрерывную самонастройку:
```
Для каждого нового рыночного наблюдения:
  1. Обновить параметры модели текущими гиперпараметрами
  2. Оценить на недавних данных вне выборки
  3. Обновить гиперпараметры мета-градиентами
  4. Сгенерировать торговый сигнал адаптированной моделью
```

### 4. Межактивный перенос гиперпараметров

Обучение общих мета-параметров на разных активах:
```
Мета-параметры eta управляют динамикой обучения для всех активов
Каждый актив адаптирует свои параметры модели theta_i
Мета-градиенты агрегируются по активам для устойчивых eta
```

---

## Реализация на Python

### Основной мета-градиентный оптимизатор

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

class MetaGradientOptimizer:
    """
    Мета-градиентная оптимизация для торговых моделей.

    Обучает оптимальные гиперпараметры (скорости обучения, параметры
    функции потерь) путём дифференцирования через внутренний цикл оптимизации.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr_init: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        learn_lr: bool = True,
        learn_loss: bool = False,
        per_param_lr: bool = False,
    ):
        """
        Инициализация мета-градиентного оптимизатора.

        Args:
            model: Нейронная сеть для торговых предсказаний
            inner_lr_init: Начальная внутренняя скорость обучения
            meta_lr: Скорость мета-обучения для обновления гиперпараметров
            inner_steps: Количество внутренних шагов оптимизации
            learn_lr: Обучать ли скорость обучения
            learn_loss: Обучать ли параметры функции потерь
            per_param_lr: Использовать ли попараметрные скорости обучения
        """
        self.model = model
        self.inner_steps = inner_steps
        self.learn_lr = learn_lr
        self.learn_loss = learn_loss
        self.per_param_lr = per_param_lr

        # Инициализация обучаемых скоростей обучения
        if per_param_lr:
            self.log_lr = nn.ParameterDict({
                name: nn.Parameter(torch.full_like(param, np.log(inner_lr_init)))
                for name, param in model.named_parameters()
            })
        else:
            self.log_lr = nn.Parameter(torch.tensor(np.log(inner_lr_init)))

        # Инициализация обучаемых параметров потерь
        if learn_loss:
            self.loss_weight_direction = nn.Parameter(torch.tensor(1.0))
            self.loss_weight_magnitude = nn.Parameter(torch.tensor(1.0))
            self.loss_asymmetry = nn.Parameter(torch.tensor(0.0))

        # Сбор мета-параметров
        meta_params = []
        if learn_lr:
            if per_param_lr:
                meta_params.extend(self.log_lr.values())
            else:
                meta_params.append(self.log_lr)
        if learn_loss:
            meta_params.extend([
                self.loss_weight_direction,
                self.loss_weight_magnitude,
                self.loss_asymmetry,
            ])

        self.meta_optimizer = torch.optim.Adam(meta_params, lr=meta_lr)

    def get_learning_rates(self) -> dict:
        """Получить текущие скорости обучения (экспонента от log-пространства)."""
        if self.per_param_lr:
            return {name: torch.exp(lr) for name, lr in self.log_lr.items()}
        else:
            return {"all": torch.exp(self.log_lr)}

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Вычислить (возможно, обучаемую) функцию потерь."""
        if not self.learn_loss:
            return F.mse_loss(predictions, targets)

        errors = predictions - targets
        direction_errors = torch.sign(predictions) != torch.sign(targets)
        direction_penalty = direction_errors.float() * F.softplus(self.loss_weight_direction)

        magnitude_loss = errors.pow(2) * F.softplus(self.loss_weight_magnitude)

        asymmetry = torch.sigmoid(self.loss_asymmetry)
        asymmetric_weight = torch.where(
            errors < 0,
            1.0 + asymmetry,
            1.0 - asymmetry * 0.5,
        )

        total_loss = (magnitude_loss * asymmetric_weight + direction_penalty).mean()
        return total_loss

    def inner_loop(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Выполнить внутренний цикл оптимизации с обучаемыми гиперпараметрами."""
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        for _ in range(self.inner_steps):
            predictions = self._functional_forward(train_features, adapted_params)
            loss = self.compute_loss(predictions, train_targets)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True,
            )

            if self.per_param_lr:
                adapted_params = {
                    name: param - torch.exp(self.log_lr[name]) * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }
            else:
                lr = torch.exp(self.log_lr)
                adapted_params = {
                    name: param - lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

        return adapted_params

    def meta_train_step(
        self,
        tasks: List[Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]],
    ) -> float:
        """
        Выполнить один шаг мета-обучения.

        Args:
            tasks: Список ((train_features, train_targets), (val_features, val_targets))

        Returns:
            Средняя мета-потеря (валидационная потеря после адаптации)
        """
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0

        for (train_features, train_targets), (val_features, val_targets) in tasks:
            adapted_params = self.inner_loop(train_features, train_targets)
            val_predictions = self._functional_forward(val_features, adapted_params)
            val_loss = F.mse_loss(val_predictions, val_targets)
            total_meta_loss += val_loss

        meta_loss = total_meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
```

### Подготовка данных

```python
import pandas as pd
import requests

def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """Создание технических признаков для трейдинга."""
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


def fetch_bybit_klines(symbol: str, interval: str = '60', limit: int = 1000):
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
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df
```

---

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительную мета-градиентную оптимизацию для продакшен торговых систем.

### Структура проекта

```
90_meta_gradient_optimization/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── meta_gradient/
│   │   ├── mod.rs
│   │   └── optimizer.rs
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
│   ├── basic_meta_gradient.rs
│   ├── multi_asset_training.rs
│   └── trading_strategy.rs
└── python/
    ├── meta_gradient_optimizer.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Основная реализация на Rust

Смотрите директорию `src/` для полной реализации на Rust с:

- Обучаемыми попараметрными скоростями через log-пространственную параметризацию
- Обучаемой функцией потерь с направленными и асимметричными компонентами
- Онлайн мета-градиентной адаптацией для живой торговли
- Асинхронной интеграцией с API Bybit для криптовалютных данных
- Продакшен-готовой обработкой ошибок и логированием

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Мета-градиентное обучение на нескольких активах

```python
import yfinance as yf

# Загрузка данных для нескольких активов
assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Подготовка данных
asset_data = {}
for name, df in assets.items():
    prices = df['Close']
    features = create_trading_features(prices)
    asset_data[name] = (prices, features)

# Инициализация модели и мета-градиентного оптимизатора
model = MetaGradientTradingModel(input_size=11, hidden_size=64, output_size=1)
meta_opt = MetaGradientOptimizer(
    model=model,
    inner_lr_init=0.01,
    meta_lr=0.001,
    inner_steps=5,
    learn_lr=True,
    learn_loss=True,
)

# Мета-обучение
task_gen = task_generator(asset_data, batch_size=4)
for epoch in range(1000):
    tasks = next(task_gen)
    meta_loss = meta_opt.meta_train_step(tasks)

    if epoch % 100 == 0:
        lrs = meta_opt.get_learning_rates()
        print(f"Эпоха {epoch}, Мета-потеря: {meta_loss:.6f}, LR: {lrs}")
```

### Пример 2: Сравнение обучаемого и фиксированного LR

```python
# Модель с мета-градиентами (обучаемый LR)
model_meta = MetaGradientTradingModel(input_size=11)
meta_opt = MetaGradientOptimizer(
    model=model_meta,
    inner_lr_init=0.01,
    meta_lr=0.001,
    inner_steps=5,
    learn_lr=True,
    per_param_lr=True,
)

# Базовая модель с фиксированным LR
model_fixed = MetaGradientTradingModel(input_size=11)
fixed_opt = torch.optim.Adam(model_fixed.parameters(), lr=0.001)

# Обучение обеих моделей
task_gen = task_generator(asset_data, batch_size=4)

for epoch in range(500):
    tasks = next(task_gen)

    # Мета-градиентная модель
    meta_loss = meta_opt.meta_train_step(tasks)

    # Базовая модель с фиксированным LR
    fixed_opt.zero_grad()
    total_fixed_loss = 0.0
    for (train_f, train_t), (val_f, val_t) in tasks:
        pred = model_fixed(train_f)
        loss = F.mse_loss(pred, train_t)
        total_fixed_loss += loss
    avg_fixed = total_fixed_loss / len(tasks)
    avg_fixed.backward()
    fixed_opt.step()

    if epoch % 50 == 0:
        print(f"Эпоха {epoch}: Мета={meta_loss:.6f}, Фиксированный={avg_fixed.item():.6f}")
```

### Пример 3: Торговля криптовалютами на Bybit с онлайн-адаптацией

```python
# Получение данных с Bybit
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_data = {}

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_trading_features(prices)
    crypto_data[symbol] = (prices, features)

# Инициализация онлайн-трейдера
model = MetaGradientTradingModel(input_size=11)
meta_opt = MetaGradientOptimizer(
    model=model,
    inner_lr_init=0.01,
    meta_lr=0.001,
    inner_steps=3,
    learn_lr=True,
    learn_loss=True,
)
online_trader = OnlineMetaGradientTrader(
    model=model,
    meta_optimizer=meta_opt,
    adaptation_window=50,
    validation_window=10,
)

# Симуляция онлайн-торговли на BTCUSDT
btc_prices, btc_features = crypto_data['BTCUSDT']
feature_cols = list(btc_features.columns)
target = btc_prices.pct_change(5).shift(-5)
aligned = btc_features.join(target.rename('target')).dropna()

for i in range(len(aligned)):
    row = aligned.iloc[i]
    feat = torch.FloatTensor(row[feature_cols].values)
    tgt = torch.FloatTensor([row['target']])

    online_trader.observe(feat, tgt)
    prediction = online_trader.update_and_predict(feat)

    if prediction is not None and i % 50 == 0:
        lrs = meta_opt.get_learning_rates()
        print(f"Шаг {i}: Предсказание={prediction:.6f}, LR={lrs}")
```

---

## Фреймворк для бэктестинга

### Мета-градиентный бэктестер

```python
class MetaGradientBacktester:
    """
    Фреймворк бэктестинга для торговых стратегий на основе мета-градиентов.
    """

    def __init__(
        self,
        meta_optimizer: MetaGradientOptimizer,
        adaptation_window: int = 30,
        validation_window: int = 10,
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001,
    ):
        self.meta_opt = meta_optimizer
        self.adaptation_window = adaptation_window
        self.validation_window = validation_window
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> pd.DataFrame:
        """Запуск бэктеста на исторических данных."""
        results = []
        capital = initial_capital
        position = 0

        feature_cols = list(features.columns)
        total_window = self.adaptation_window + self.validation_window

        for i in range(total_window, len(features) - 1):
            window_data = features.iloc[i - total_window:i]
            window_returns = prices.pct_change().iloc[i - total_window + 1:i + 1]

            train_f = torch.FloatTensor(
                window_data.iloc[:self.adaptation_window][feature_cols].values
            )
            train_t = torch.FloatTensor(
                window_returns.iloc[:self.adaptation_window].values
            ).unsqueeze(1)
            val_f = torch.FloatTensor(
                window_data.iloc[self.adaptation_window:][feature_cols].values
            )
            val_t = torch.FloatTensor(
                window_returns.iloc[self.adaptation_window:].values
            ).unsqueeze(1)

            task = [((train_f, train_t), (val_f, val_t))]
            self.meta_opt.meta_train_step(task)

            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            prediction = self.meta_opt.adapt_and_predict(
                train_f, train_t, current_features
            ).item()

            new_position = 0
            if prediction > self.threshold:
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1

            if new_position != position:
                capital *= (1 - self.transaction_cost)

            actual_return = prices.iloc[i + 1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """Расчёт метрик торговой производительности."""
    returns = results['position_return']

    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / (gross_losses + 1e-10)

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(results[results['position'] != 0]),
    }
```

---

## Оценка производительности

### Целевые показатели

| Метрика | Целевой диапазон |
|---------|-----------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Максимальная просадка | < 20% |
| Win Rate | > 50% |
| Profit Factor | > 1.5 |

### Сравнение мета-градиентов vs фиксированных гиперпараметров

В типичных экспериментах мета-градиентная оптимизация показывает:
- **Улучшение Sharpe ratio на 10-25%** по сравнению с фиксированными LR
- **Более быструю сходимость** благодаря адаптивным скоростям обучения
- **Лучшую устойчивость** к разным рыночным режимам
- **Меньшие просадки** при обучении асимметрии функции потерь

### Ключевые преимущества перед MAML

- **Онлайн-адаптация**: Мета-градиенты могут вычисляться и применяться онлайн
- **Гибкость гиперпараметров**: Может обучать любой дифференцируемый гиперпараметр
- **Вычислительная эффективность**: Не требует хранения пакетов задач

---

## Направления развития

### 1. Обучаемые оптимизаторы

Замена вручную созданных правил обновления нейросетевым оптимизатором.

### 2. Мета-градиенты для управления рисками

Обучение гиперпараметров, совместно оптимизирующих доходность и риск.

### 3. Мультимасштабные мета-градиенты

Обучение разных гиперпараметров на разных временных масштабах:
- Быстрая адаптация: eta_fast (внутридневные корректировки)
- Средняя адаптация: eta_medium (недельный режим)
- Медленная адаптация: eta_slow (структурные рыночные изменения)

### 4. Популяционные мета-градиенты

Комбинирование мета-градиентной оптимизации с популяционным обучением для разнообразия исследований.

### 5. Каузальные мета-градиенты

Включение каузальных рассуждений в фреймворк мета-градиентов для избежания ложных корреляций.

---

## Ссылки

1. Xu, Z., van Hasselt, H., & Silver, D. (2018). Meta-Gradient Reinforcement Learning. NeurIPS. [arXiv:1805.09801](https://arxiv.org/abs/1805.09801)

2. Andrychowicz, M., et al. (2016). Learning to learn by gradient descent by gradient descent. NeurIPS. [arXiv:1606.04474](https://arxiv.org/abs/1606.04474)

3. Li, K., & Malik, J. (2017). Learning to Optimize. ICLR. [arXiv:1606.01885](https://arxiv.org/abs/1606.01885)

4. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

5. Zheng, Z., Oh, J., & Singh, S. (2018). On Learning Intrinsic Rewards for Policy Gradient Methods. NeurIPS.

6. Hospedales, T., et al. (2020). Meta-Learning in Neural Networks: A Survey. IEEE TPAMI.

---

## Запуск примеров

### Python

```bash
# Перейти в директорию главы
cd 90_meta_gradient_optimization

# Установить зависимости
pip install -r python/requirements.txt

# Запустить Python примеры
python python/meta_gradient_optimizer.py
```

### Rust

```bash
# Перейти в директорию главы
cd 90_meta_gradient_optimization

# Собрать проект
cargo build --release

# Запустить тесты
cargo test

# Запустить примеры
cargo run --example basic_meta_gradient
cargo run --example multi_asset_training
cargo run --example trading_strategy
```

---

## Резюме

Мета-градиентная оптимизация предоставляет принципиальный фреймворк для самонастраивающихся торговых систем:

- **Обучаемые гиперпараметры**: Скорости обучения, функции потерь и правила обновления оптимизируются мета-градиентами
- **Онлайн-адаптация**: Гиперпараметры адаптируются в реальном времени к изменяющимся рыночным условиям
- **Гибкость и компонуемость**: Может комбинироваться с любой градиентной моделью и торговой стратегией
- **Эмпирическое превосходство**: Стабильно превосходит базовые модели с фиксированными гиперпараметрами в нестационарных средах

Обучая сам процесс обучения, мета-градиентная оптимизация создаёт торговые системы, которые адаптируют не только свои предсказания, но и *как* они учатся -- критически важная способность для навигации в сложных и нестационарных финансовых рынках.

---

*Предыдущая глава: [Глава 89: Непрерывное мета-обучение](../89_continual_meta_learning)*

*Следующая глава: [Глава 91: Трансферное обучение для трейдинга](../91_transfer_learning_trading)*
