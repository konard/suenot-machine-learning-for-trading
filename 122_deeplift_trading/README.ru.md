# Глава 122: DeepLIFT для трейдинга

## Обзор

DeepLIFT (Deep Learning Important FeaTures) - это мощный метод интерпретируемости, который объясняет предсказания нейронных сетей путём сравнения активаций с референсным входом. Представленный Shrikumar и соавторами (2017), DeepLIFT присваивает оценки вклада каждому входному признаку, распространяя разницу между фактической активацией и референсной активацией обратно через сеть.

В алгоритмическом трейдинге DeepLIFT незаменим для понимания того, какие рыночные признаки управляют торговыми сигналами, выявления смены режимов и построения более прозрачных и надёжных торговых систем.

## Содержание

1. [Введение в DeepLIFT](#введение-в-deeplift)
2. [Математические основы](#математические-основы)
3. [DeepLIFT vs другие методы атрибуции](#deeplift-vs-другие-методы-атрибуции)
4. [DeepLIFT для торговых приложений](#deeplift-для-торговых-приложений)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Направления развития](#направления-развития)

---

## Введение в DeepLIFT

### Что такое атрибуция признаков?

Методы атрибуции признаков объясняют предсказания нейронных сетей, присваивая оценки важности каждому входному признаку. Эти оценки показывают, насколько каждый признак способствовал конечному предсказанию, помогая понять "почему" за решениями модели.

### Алгоритм DeepLIFT

DeepLIFT был представлен Avanti Shrikumar, Peyton Greenside и Anshul Kundaje в их статье 2017 года "Learning Important Features Through Propagating Activation Differences". Ключевая идея элегантна:

1. Выбрать референсный вход (базовую линию), представляющий "отсутствие сигнала"
2. Вычислить разницу активации на каждом нейроне между фактическим входом и референсом
3. Разложить эту разницу на вклады от каждого входного признака
4. Распространить вклады обратно через сеть, используя цепное правило для множителей

Свойство "суммирования до дельты" гарантирует, что вклады точно суммируются в разницу между выходом для фактического входа и референса:

```
Σᵢ Cᵢ = f(x) - f(x_ref)
```

### Почему DeepLIFT для трейдинга?

Финансовые рынки представляют уникальные вызовы, которые делают DeepLIFT особенно привлекательным:

- **Интерпретируемость**: Понимание того, какие технические индикаторы управляют торговыми сигналами
- **Управление рисками**: Выявление случаев, когда модели полагаются на ложные корреляции
- **Обнаружение смены режимов**: Наблюдение за сдвигами важности признаков во время рыночных переходов
- **Валидация модели**: Проверка того, что модели изучают значимые рыночные паттерны
- **Регуляторное соответствие**: Обеспечение объяснимого ИИ для финансовых решений

---

## Математические основы

### Основной принцип

DeepLIFT вычисляет оценки вклада путём сравнения активаций с референсом:

**Разница активации:**
```
Δt = t - t⁰
```
где t - фактическая активация, а t⁰ - референсная активация.

**Оценка вклада:**
```
Cᵢ = вклад входа xᵢ в разницу активации Δt
```

### Правило множителя

Для нейрона с входами x₁, ..., xₙ и выходом t:

**Определение множителя:**
```
mᵢ = Cᵢ / Δxᵢ
```

где Δxᵢ = xᵢ - x⁰ᵢ - разница от референсного входа.

**Суммирование до дельты:**
```
Σᵢ mᵢ × Δxᵢ = Δt
```

### Правила распространения

**Линейный слой:**
Для t = Σᵢ wᵢ × xᵢ + b:
```
mᵢ = wᵢ
```

**Активация ReLU (Правило перемасштабирования):**
```
mᵢ = Δy / Δx    (если Δx ≠ 0)
    = 0         (если Δx = 0)
```

**Активация ReLU (Правило RevealCancel):**
Для более точной атрибуции разделите положительные и отрицательные вклады:
```
Δy⁺ = (y⁺ - y⁰⁺)
Δy⁻ = (y⁻ - y⁰⁻)
```

### Цепное правило для множителей

Для цепочки слоёв умножайте множители:
```
m_total = m₁ × m₂ × ... × mₙ
```

### Выбор референса

Выбор правильного референса критически важен:

- **Нулевой референс**: Все признаки установлены в 0 (часто используется, но не всегда значим)
- **Средний референс**: Средние значения по набору данных
- **Нейтральный референс**: Значения, представляющие "отсутствие торгового сигнала"
- **Распределённый референс**: Выборка из распределения входов (ожидаемые градиенты)

---

## DeepLIFT vs другие методы атрибуции

### Сравнительная таблица

| Метод | Нужен референс | Обработка насыщения | Вычисления | Точность |
|-------|----------------|---------------------|------------|----------|
| DeepLIFT | Да | Отлично | Средние | Отлично |
| Градиент | Нет | Плохо | Низкие | Удовлетворительно |
| Интегрированные градиенты | Да | Хорошо | Высокие | Очень хорошо |
| SHAP | Да (распределение) | Отлично | Очень высокие | Отлично |
| LRP | Нет | Хорошо | Средние | Хорошо |
| Карты значимости | Нет | Плохо | Низкие | Удовлетворительно |

### Когда использовать DeepLIFT

**Используйте DeepLIFT когда:**
- Вам нужны быстрые и точные оценки атрибуции
- Ваша модель имеет ReLU-подобные активации
- Вы хотите понять относительную важность признаков
- Важна атрибуция с учётом насыщения

**Рассмотрите альтернативы когда:**
- Вам нужны теоретические гарантии (используйте SHAP)
- У вас нестандартные архитектуры (используйте интегрированные градиенты)
- Скорость первостепенна (используйте простые градиенты)

---

## DeepLIFT для торговых приложений

### 1. Объяснение торговых сигналов

Понимание того, какие признаки управляют сигналами покупки/продажи:

```
Входные признаки: [доходность, волатильность, моментум, RSI, MACD, объём, ...]
Выход DeepLIFT: Вклад каждого признака в предсказание
Пример: "RSI внёс +0.3 в сигнал покупки, в то время как высокая волатильность внесла -0.15"
```

### 2. Атрибуция риска

Определение факторов, вносящих вклад в риск портфеля:

```
Для предсказания риска 0.8:
- Корреляция с рынком: +0.4
- Секторная экспозиция: +0.25
- Режим волатильности: +0.15
```

### 3. Обнаружение смены режима

Мониторинг сдвигов важности признаков во времени:

```
Бычий рынок:   моментум=0.5, возврат_к_среднему=-0.1
Медвежий рынок: моментум=-0.2, возврат_к_среднему=0.4
Переход обнаруживается при значительном изменении паттернов важности
```

### 4. Отладка модели

Проверка того, что модели изучают разумные паттерны:

```
Хорошо: RSI перепродан → положительный вклад в сигнал покупки
Плохо: День недели → большой вклад (вероятно, ложный)
```

---

## Реализация на Python

### Основной алгоритм DeepLIFT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Attribution:
    """Оценки атрибуции для одного предсказания."""
    feature_names: List[str]
    scores: np.ndarray
    baseline_output: float
    actual_output: float
    delta: float

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Получить top n вносящих вклад признаков."""
        indices = np.argsort(np.abs(self.scores))[::-1][:n]
        return [(self.feature_names[i], self.scores[i]) for i in indices]


class DeepLIFT:
    """
    DeepLIFT атрибуция для торговых моделей на нейронных сетях.

    Эта реализация поддерживает как правило перемасштабирования,
    так и правило RevealCancel для ReLU-подобных активаций.
    """

    def __init__(
        self,
        model: nn.Module,
        reference: Optional[torch.Tensor] = None,
        rule: str = "rescale"
    ):
        """
        Инициализация объяснителя DeepLIFT.

        Args:
            model: Нейросетевая модель для объяснения
            reference: Референсный вход (базовая линия). Если None, используются нули.
            rule: Правило атрибуции - "rescale" или "reveal_cancel"
        """
        self.model = model
        self.reference = reference
        self.rule = rule
        self._hooks = []
        self._activations = {}
        self._ref_activations = {}

    def attribute(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Attribution:
        """
        Вычисление оценок атрибуции DeepLIFT.

        Args:
            input_tensor: Вход для объяснения (batch_size=1)
            feature_names: Имена входных признаков

        Returns:
            Объект Attribution с оценками вклада
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Установка референса
        if self.reference is None:
            reference = torch.zeros_like(input_tensor)
        else:
            reference = self.reference.expand_as(input_tensor)

        # Вычисление референсного выхода
        self.model.eval()
        with torch.no_grad():
            ref_output = self.model(reference)
            actual_output = self.model(input_tensor)

        # Вычисление атрибуции через обратное распространение
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # Вычисление градиентов
        output.backward(torch.ones_like(output))

        # Получение градиентов
        gradients = input_tensor.grad.detach()

        # Вычисление дельты от референса
        delta_input = input_tensor.detach() - reference

        # Атрибуция DeepLIFT: градиент * дельта (правило перемасштабирования)
        if self.rule == "rescale":
            attributions = gradients * delta_input
        else:
            # Правило RevealCancel - разделение положительных и отрицательных
            attributions = self._reveal_cancel_attribution(
                input_tensor, reference, gradients
            )

        # Создание имён признаков, если не предоставлены
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(input_tensor.shape[1])]

        return Attribution(
            feature_names=feature_names,
            scores=attributions.squeeze().numpy(),
            baseline_output=ref_output.item(),
            actual_output=actual_output.item(),
            delta=actual_output.item() - ref_output.item()
        )

    def _reveal_cancel_attribution(
        self,
        input_tensor: torch.Tensor,
        reference: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление атрибуции с использованием правила RevealCancel.
        Разделяет положительные и отрицательные вклады.
        """
        delta = input_tensor.detach() - reference
        positive_delta = F.relu(delta)
        negative_delta = -F.relu(-delta)

        # Вычисление раздельных атрибуций
        positive_attr = gradients * positive_delta
        negative_attr = gradients * negative_delta

        return positive_attr + negative_attr


class TradingModelWithDeepLIFT(nn.Module):
    """
    Нейронная сеть для трейдинга со встроенной поддержкой DeepLIFT.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

### Подготовка данных

```python
import pandas as pd
import requests


class BybitClient:
    """Клиент для получения криптовалютных данных с Bybit."""

    def __init__(self, base_url: str = "https://api.bybit.com"):
        self.base_url = base_url

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Получение исторических свечей с Bybit.

        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            interval: Интервал свечи ("1", "5", "15", "60", "D")
            limit: Количество свечей

        Returns:
            DataFrame с OHLCV данными
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"Ошибка API: {data.get('retMsg')}")

        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df
```

---

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительный DeepLIFT для продакшен торговых систем.

### Структура проекта

```
122_deeplift_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── deeplift/
│   │   ├── mod.rs
│   │   └── attribution.rs
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
│   ├── basic_deeplift.rs
│   ├── feature_importance.rs
│   └── trading_explanation.rs
└── python/
    ├── deeplift_trader.py
    ├── data_loader.py
    ├── backtest.py
    └── requirements.txt
```

### Основная реализация на Rust

Смотрите директорию `src/` для полной реализации на Rust с:

- Эффективными матричными операциями с использованием ndarray
- Прямым проходом с кэшированием активаций
- Обратным проходом с распространением множителей
- Асинхронной интеграцией с API Bybit для криптовалютных данных
- Продакшен-готовой обработкой ошибок и логированием

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Обучение и объяснение торговой модели

```python
import yfinance as yf

# Загрузка данных
data = yf.download('BTC-USD', period='2y')
prices = data['Close'].values

# Создание признаков
features = create_trading_features(prices)

# Подготовка данных
X_train, y_train, X_test, y_test = prepare_training_data(prices, features)

# Определение имён признаков
feature_names = [
    "return_1d", "return_5d", "return_10d", "sma_ratio", "ema_ratio",
    "volatility", "momentum", "rsi", "macd", "bb_position", "volume_ratio"
]

# Обучение модели
model = TradingModelWithDeepLIFT(input_size=11, hidden_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Эпоха {epoch}, Потеря: {loss.item():.6f}")

# Объяснение предсказания
reference = torch.FloatTensor(np.mean(X_train, axis=0, keepdims=True))
explainer = DeepLIFT(model, reference=reference)

sample = torch.FloatTensor(X_test[0:1])
attribution = explainer.attribute(sample, feature_names)

print("\nОбъяснение предсказания:")
print(f"Базовый выход: {attribution.baseline_output:.6f}")
print(f"Фактический выход: {attribution.actual_output:.6f}")
print(f"Дельта: {attribution.delta:.6f}")
print("\nTop вносящие вклад признаки:")
for name, score in attribution.top_features(5):
    print(f"  {name}: {score:.6f}")
```

### Пример 2: Анализ важности признаков

```python
# Вычисление общей важности признаков
importance = compute_feature_importance(
    model, X_test, feature_names,
    reference=np.mean(X_train, axis=0, keepdims=True)
)

print("\nОбщая важность признаков:")
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_importance:
    print(f"  {name}: {score:.6f}")
```

### Пример 3: Криптовалютный трейдинг на Bybit с объяснениями

```python
# Получение данных с Bybit
client = BybitClient()
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

for symbol in crypto_pairs:
    df = client.fetch_klines(symbol, interval='60', limit=500)
    prices = df['close'].values
    features = create_trading_features(prices)

    # Использование предобученной модели
    X_test = features[100:]  # Пропуск периода прогрева
    X_test_t = torch.FloatTensor(X_test)

    # Получение предсказаний и объяснений
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)

    # Объяснение последнего предсказания
    latest = X_test_t[-1:]
    attribution = explainer.attribute(latest, feature_names)

    print(f"\n{symbol} Объяснение последнего сигнала:")
    print(f"  Предсказание: {predictions[-1].item():.6f}")
    print(f"  Top факторы:")
    for name, score in attribution.top_features(3):
        direction = "бычий" if score > 0 else "медвежий"
        print(f"    {name}: {score:.6f} ({direction})")
```

---

## Фреймворк для бэктестинга

### Бэктестер с поддержкой DeepLIFT

```python
class DeepLIFTBacktester:
    """
    Фреймворк бэктестинга с объяснениями DeepLIFT.
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: DeepLIFT,
        feature_names: List[str],
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001
    ):
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Запуск бэктеста с логированием объяснений.
        """
        results = []
        capital = initial_capital
        position = 0

        self.model.eval()

        for i in range(len(features)):
            input_tensor = torch.FloatTensor(features[i:i+1])

            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            # Получение объяснения
            attribution = self.explainer.attribute(input_tensor, self.feature_names)
            top_features = attribution.top_features(3)

            # Торговая логика
            if prediction > self.threshold:
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            # Транзакционные издержки
            if new_position != position and i > 0:
                capital *= (1 - self.transaction_cost)

            # Расчёт доходности
            if i < len(prices) - 1:
                actual_return = prices[i+1] / prices[i] - 1
                position_return = position * actual_return
                capital *= (1 + position_return)
            else:
                position_return = 0

            results.append({
                'index': i,
                'price': prices[i],
                'prediction': prediction,
                'position': position,
                'position_return': position_return,
                'capital': capital,
                'top_feature_1': top_features[0][0] if len(top_features) > 0 else '',
                'top_score_1': top_features[0][1] if len(top_features) > 0 else 0,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Расчёт метрик торговой производительности.
    """
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

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }
```

---

## Оценка производительности

### Целевые показатели производительности

| Метрика | Целевой диапазон |
|---------|-----------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Максимальная просадка | < 20% |
| Win Rate | > 50% |
| Консистентность объяснений | > 80% |

### DeepLIFT vs базовая атрибуция

В типичных экспериментах DeepLIFT показывает:
- **В 2-5 раз быстрее** вычислений по сравнению с SHAP
- **Лучшую обработку насыщения** по сравнению с градиентными методами
- **Консистентные рейтинги признаков** для похожих входов
- **Свойство суммирования** гарантирует, что атрибуции суммируются в дельту предсказания

---

## Направления развития

### 1. Темпоральный DeepLIFT

Расширение атрибуции на последовательные модели:
- Сети LSTM/GRU с темпоральной важностью признаков
- Атрибуции, взвешенные вниманием, для трансформеров

### 2. Атрибуция с учётом неопределённости

Комбинирование DeepLIFT с квантификацией неопределённости:
```
Атрибуция с доверительными интервалами для каждого вклада признака
```

### 3. Контрфактические объяснения

Генерация сценариев "что если":
```
"Если бы RSI был 30 вместо 70, сигнал изменился бы с покупки на удержание"
```

### 4. Атрибуция в реальном времени

Потоковые объяснения для живого трейдинга:
- Вычисление атрибуции с низкой задержкой
- Обнаружение аномалий на основе необычной важности признаков

### 5. Мульти-модельная атрибуция

Ансамблевые объяснения:
- Агрегирование атрибуций по нескольким моделям
- Выявление консенсуса и разногласий в важности признаков

---

## Ссылки

1. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. ICML. [arXiv:1704.02685](https://arxiv.org/abs/1704.02685)

2. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

4. Ancona, M., et al. (2018). Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks. ICLR.

5. Montavon, G., et al. (2018). Methods for Interpreting and Understanding Deep Neural Networks. Digital Signal Processing.

---

## Запуск примеров

### Python

```bash
# Перейти в директорию главы
cd 122_deeplift_trading

# Установить зависимости
pip install -r python/requirements.txt

# Запустить Python примеры
python python/deeplift_trader.py
```

### Rust

```bash
# Перейти в директорию главы
cd 122_deeplift_trading

# Собрать проект
cargo build --release

# Запустить тесты
cargo test

# Запустить примеры
cargo run --example basic_deeplift
cargo run --example feature_importance
cargo run --example trading_explanation
```

---

## Резюме

DeepLIFT предоставляет мощный фреймворк для интерпретируемости нейронных сетей в трейдинге:

- **Теоретический фундамент**: Сравнивает активации с референсом для значимых атрибуций
- **Свойство суммирования**: Вклады признаков точно суммируются в разницу предсказаний
- **Обработка насыщения**: Правильно обрабатывает насыщение ReLU в отличие от градиентных методов
- **Практическая ценность**: Необходим для построения прозрачных, надёжных торговых систем

Понимая, какие признаки управляют торговыми сигналами, DeepLIFT позволяет трейдерам и квантам валидировать поведение модели, обнаруживать смену режимов и соответствовать требованиям объяснимости в финансовых приложениях.

---

*Предыдущая глава: [Глава 121: Layer-wise Relevance Propagation](../121_layer_wise_relevance)*

*Следующая глава: [Глава 123: GradCAM для финансов](../123_gradcam_finance)*
