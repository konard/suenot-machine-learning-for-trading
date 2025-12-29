# Глава 34: Онлайн-обучение — Адаптивный моментум с непрерывным переобучением

## Обзор

Традиционные модели машинного обучения обучаются на пакетных (batch) данных и деградируют со временем из-за **concept drift** (дрейфа концепций). Онлайн-обучение позволяет модели непрерывно адаптироваться к новым данным без полного переобучения.

В этой главе мы строим **адаптивную momentum стратегию**, которая эволюционирует вместе с рынком, автоматически корректируя веса факторов в реальном времени.

### Почему это важно?

Финансовые рынки постоянно меняются:
- Меняются режимы волатильности
- Появляются новые участники рынка
- Макроэкономические условия эволюционируют
- Стратегии конкурентов адаптируются

Статическая модель, обученная на данных 2020 года, может быть неэффективна в 2024 году. Онлайн-обучение решает эту проблему.

## Торговая стратегия

### Суть стратегии

**Online Gradient Descent (OGD)** для непрерывной адаптации весов momentum сигналов:
- Модель автоматически увеличивает вес факторов, которые работают
- Уменьшает вес тех факторов, что перестали работать
- Адаптация происходит после каждого нового наблюдения

### Сигналы входа

| Условие | Действие |
|---------|----------|
| Взвешенный momentum score > threshold | **Long** (покупка) |
| Взвешенный momentum score < -threshold | **Short** (продажа) |
| Между порогами | **Flat** (без позиции) |

### Преимущество (Edge)

Быстрая адаптация к смене режимов рынка без необходимости полного переобучения модели.

## Теоретические основы

### Concept Drift (Дрейф концепций)

**Concept drift** — изменение статистических свойств целевой переменной, которую модель пытается предсказать.

#### Типы дрейфа

| Тип | Описание | Пример |
|-----|----------|--------|
| **Sudden (Внезапный)** | Резкое изменение распределения | Финансовый кризис 2008, COVID-19 |
| **Gradual (Постепенный)** | Плавный переход между распределениями | Смена рыночного режима |
| **Incremental (Инкрементальный)** | Медленные, накапливающиеся изменения | Эволюция рыночной микроструктуры |
| **Recurring (Повторяющийся)** | Циклические изменения | Сезонность в торговле |

### Online Learning vs Batch Learning

| Характеристика | Batch Learning | Online Learning |
|----------------|---------------|-----------------|
| Данные | Фиксированный датасет | Поток данных |
| Обновление | Переобучение на всех данных | Инкрементальное обновление |
| Память | O(N) — хранит все данные | O(1) — константная память |
| Адаптация | Медленная | Мгновенная |
| Забывание | Нет | Можно настроить |

### Regret Bounds (Границы сожаления)

В онлайн-обучении качество алгоритма измеряется **regret** — разницей между накопленными потерями алгоритма и лучшим решением в ретроспективе:

```
Regret_T = Σ_t l(ŷ_t, y_t) - min_w Σ_t l(w · x_t, y_t)
```

Хороший онлайн-алгоритм имеет **sublinear regret**: `Regret_T = O(√T)`, что означает, что средний regret стремится к нулю.

## Техническая спецификация

### Ноутбуки

| # | Notebook | Описание |
|---|----------|----------|
| 1 | `01_concept_drift.ipynb` | Теория concept drift, типы изменений |
| 2 | `02_online_learning_basics.ipynb` | SGD, regret bounds, сходимость |
| 3 | `03_momentum_features.ipynb` | Набор momentum features для адаптации |
| 4 | `04_river_library.ipynb` | Использование River для online ML |
| 5 | `05_online_linear.ipynb` | Online linear regression для весов |
| 6 | `06_online_tree.ipynb` | Hoeffding Trees для нелинейных зависимостей |
| 7 | `07_drift_detection.ipynb` | ADWIN, DDM для обнаружения drift |
| 8 | `08_adaptive_windows.ipynb` | Динамический размер обучающего окна |
| 9 | `09_ensemble_online.ipynb` | Ансамбль online learners |
| 10 | `10_backtesting.ipynb` | Симуляция со streaming data |
| 11 | `11_comparison.ipynb` | Сравнение с batch retraining, static model |

### Требования к данным

```
Симуляция потоковых данных:
├── Дневные доходности акций (10+ лет)
├── Множественные momentum факторы
├── Симуляция как поток: по одному дню за раз
└── Запрещено заглядывание в будущее (lookahead)

Momentum факторы:
├── Price momentum (1m, 3m, 6m, 12m)
├── Volume momentum
├── Earnings momentum
├── Analyst revision momentum
└── Industry momentum
```

## Реализация онлайн-обучения

### Базовая структура с River

```python
from river import linear_model, preprocessing, optim, drift

# Онлайн линейная модель с адаптивной скоростью обучения
model = preprocessing.StandardScaler() | linear_model.LinearRegression(
    optimizer=optim.Adam(lr=0.01),
    l2=0.001
)

# Цикл потокового предсказания
for day in trading_days:
    # Получаем признаки на сегодня (известные на открытии рынка)
    x = get_features(day)

    # Предсказываем доходность
    y_pred = model.predict_one(x)

    # Генерируем торговый сигнал
    signal = 'long' if y_pred > threshold else 'short' if y_pred < -threshold else 'flat'

    # В конце дня наблюдаем фактическую доходность
    y_true = get_actual_return(day)

    # Обновляем модель новым наблюдением
    model.learn_one(x, y_true)
```

### Обнаружение Concept Drift

#### ADWIN (ADaptive WINdowing)

ADWIN автоматически определяет оптимальный размер окна, обнаруживая изменения в потоке данных:

```python
from river import drift

# ADWIN: Адаптивное окно
adwin = drift.ADWIN(delta=0.002)

# DDM: Drift Detection Method
ddm = drift.DDM(min_num_instances=30)

# Обнаружение дрейфа
for t, (x, y_true) in enumerate(stream):
    y_pred = model.predict_one(x)
    error = abs(y_pred - y_true)

    adwin.update(error)
    if adwin.drift_detected:
        print(f"Дрейф обнаружен в момент времени {t}")
        # Вариант 1: Сбросить модель
        model = create_fresh_model()
        # Вариант 2: Уменьшить влияние истории
        # Вариант 3: Переключиться на другую модель
```

#### Алгоритмы обнаружения дрейфа

| Алгоритм | Описание | Применение |
|----------|----------|------------|
| **ADWIN** | Адаптивное окно на основе статистического теста | Универсальный, хорошо для постепенного дрейфа |
| **DDM** | Drift Detection Method на основе биномиального теста | Хорош для внезапного дрейфа |
| **EDDM** | Early DDM, улучшенная версия | Раннее обнаружение |
| **Page-Hinkley** | CUSUM-подобный тест | Обнаружение изменения среднего |

### Адаптивное окно обучения

```python
class AdaptiveWindowModel:
    """
    Динамически настраивает размер обучающего окна
    на основе недавней производительности
    """
    def __init__(self, min_window=20, max_window=252):
        self.min_window = min_window
        self.max_window = max_window
        self.current_window = 60
        self.buffer = []
        self.error_threshold = 0.05

    def update(self, x, y):
        self.buffer.append((x, y))

        # Ограничиваем буфер максимальным размером
        if len(self.buffer) > self.max_window:
            self.buffer.pop(0)

        # Переобучаем на адаптивном окне
        window_data = self.buffer[-self.current_window:]
        self.model.fit(window_data)

        # Оцениваем недавнюю производительность
        recent_error = self.evaluate_recent(window=20)

        # Корректируем размер окна
        if recent_error > self.error_threshold:
            # Производительность падает: сужаем окно (адаптируемся быстрее)
            self.current_window = max(self.min_window, self.current_window - 10)
        else:
            # Производительность хорошая: расширяем окно (более стабильно)
            self.current_window = min(self.max_window, self.current_window + 5)
```

### Ансамбль онлайн-моделей

```python
from river import ensemble

# Адаптивный случайный лес
arf = ensemble.AdaptiveRandomForestRegressor(
    n_models=10,
    max_depth=6,
    drift_detector=drift.ADWIN()
)

# Бэггинг с онлайн базовыми моделями
bagging = ensemble.BaggingRegressor(
    model=linear_model.LinearRegression(),
    n_models=10
)

# Стэкинг онлайн-моделей
class OnlineStacking:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict_one(self, x):
        base_preds = [m.predict_one(x) for m in self.base_models]
        return self.meta_model.predict_one(base_preds)

    def learn_one(self, x, y):
        base_preds = [m.predict_one(x) for m in self.base_models]
        for m in self.base_models:
            m.learn_one(x, y)
        self.meta_model.learn_one(base_preds, y)
```

## Адаптивные веса моментума

### Реализация

```python
class AdaptiveMomentumWeights:
    """
    Онлайн-обучение оптимальных весов momentum факторов
    """
    def __init__(self, n_factors, learning_rate=0.01):
        self.weights = np.ones(n_factors) / n_factors
        self.lr = learning_rate
        self.factor_names = []

    def predict(self, factor_signals):
        """Взвешенное предсказание"""
        return np.dot(self.weights, factor_signals)

    def update(self, factor_signals, actual_return):
        """Обновление весов градиентным спуском"""
        prediction = self.predict(factor_signals)
        error = actual_return - prediction

        # Градиентное обновление (MSE loss)
        gradient = -2 * error * factor_signals
        self.weights -= self.lr * gradient

        # Нормализация весов (опционально)
        self.weights = self.weights / np.sum(np.abs(self.weights))

        return error

    def get_factor_importance(self):
        """Получить текущую важность факторов"""
        return dict(zip(self.factor_names, self.weights))
```

### Momentum факторы для адаптации

| Фактор | Описание | Лаг |
|--------|----------|-----|
| `mom_1m` | Доходность за 1 месяц | 21 день |
| `mom_3m` | Доходность за 3 месяца | 63 дня |
| `mom_6m` | Доходность за 6 месяцев | 126 дней |
| `mom_12m` | Доходность за 12 месяцев | 252 дня |
| `vol_mom` | Изменение объёма | 21 день |
| `rev_mom` | Momentum аналитических ревизий | 30 дней |
| `ind_mom` | Отраслевой momentum | 63 дня |

## Бэктестинг с потоковыми данными

### Фреймворк симуляции

```python
def online_backtest(data, model, initial_train=252):
    """
    Бэктест с симуляцией потоковых данных

    Args:
        data: DataFrame с признаками и целевой переменной
        model: Онлайн-модель с методами predict_one и learn_one
        initial_train: Начальный период обучения (без торговли)

    Returns:
        DataFrame с результатами
    """
    results = []

    # Начальный период обучения (без торговли)
    for t in range(initial_train):
        x, y = data.iloc[t]
        model.learn_one(x, y)

    # Торговый период
    for t in range(initial_train, len(data)):
        x, y = data.iloc[t]

        # Предсказание ДО наблюдения результата
        y_pred = model.predict_one(x)
        signal = generate_signal(y_pred)

        # Торговля
        pnl = signal * y

        # Обучение ПОСЛЕ наблюдения результата
        model.learn_one(x, y)

        results.append({
            'date': data.index[t],
            'prediction': y_pred,
            'actual': y,
            'signal': signal,
            'pnl': pnl
        })

    return pd.DataFrame(results)
```

### Важные принципы бэктестинга

1. **Строгий порядок**: Предсказание → Торговля → Обучение
2. **Никакого заглядывания в будущее**: Модель видит только прошлые данные
3. **Реалистичная задержка**: Учитывать время исполнения ордеров
4. **Транзакционные издержки**: Включать комиссии и slippage

## Метрики оценки

### Метрики предсказания

| Метрика | Описание |
|---------|----------|
| **Rolling IC** | Скользящий информационный коэффициент (корреляция Спирмена) |
| **MSE** | Среднеквадратичная ошибка |
| **Directional Accuracy** | Точность направления (знак предсказания) |

### Метрики адаптации

| Метрика | Описание |
|---------|----------|
| **Drift Frequency** | Как часто обнаруживается дрейф |
| **Weight Evolution** | Эволюция весов факторов во времени |
| **Adaptation Speed** | Скорость адаптации к новым режимам |

### Метрики стратегии

| Метрика | Описание |
|---------|----------|
| **Sharpe Ratio** | Доходность с поправкой на риск |
| **Max Drawdown** | Максимальная просадка |
| **Win Rate** | Доля прибыльных сделок |
| **Profit Factor** | Отношение прибылей к убыткам |

## Сравнение с альтернативами

### Эксперимент

Сравниваем три подхода:
1. **Static Model**: Обучена один раз, никогда не обновляется
2. **Monthly Retrain**: Переобучение раз в месяц на скользящем окне
3. **Online Learning**: Непрерывное обновление после каждого наблюдения

### Ожидаемые результаты

| Режим рынка | Static | Monthly | Online |
|-------------|--------|---------|--------|
| Стабильный | ✓ | ✓ | ✓ |
| Переходный | ✗ | ∼ | ✓ |
| Высокая волатильность | ✗ | ✗ | ✓ |

Онлайн-обучение должно показывать особенно высокую производительность в нестационарные периоды.

## Rust-реализация

В директории `rust_online_learning/` находится полная реализация на Rust:

```
rust_online_learning/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/                    # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── error.rs
│   ├── models/                 # Онлайн-модели
│   │   ├── mod.rs
│   │   ├── online_linear.rs    # Онлайн линейная регрессия
│   │   ├── adaptive_weights.rs # Адаптивные веса
│   │   └── hoeffding_tree.rs   # Hoeffding tree
│   ├── drift/                  # Обнаружение дрейфа
│   │   ├── mod.rs
│   │   ├── adwin.rs            # ADWIN алгоритм
│   │   └── ddm.rs              # DDM алгоритм
│   ├── features/               # Инженерия признаков
│   │   ├── mod.rs
│   │   └── momentum.rs         # Momentum индикаторы
│   ├── streaming/              # Потоковая симуляция
│   │   ├── mod.rs
│   │   └── simulator.rs
│   └── backtest/               # Бэктестинг
│       ├── mod.rs
│       └── engine.rs
└── examples/
    ├── fetch_data.rs           # Загрузка данных с Bybit
    ├── online_regression.rs    # Пример онлайн-регрессии
    ├── drift_detection.rs      # Пример обнаружения дрейфа
    ├── adaptive_momentum.rs    # Полная стратегия
    └── backtest_comparison.rs  # Сравнение подходов
```

### Запуск примеров

```bash
# Загрузка данных с Bybit
cargo run --example fetch_data

# Онлайн-регрессия
cargo run --example online_regression

# Обнаружение дрейфа
cargo run --example drift_detection

# Полная стратегия
cargo run --example adaptive_momentum

# Сравнение с batch-моделью
cargo run --example backtest_comparison
```

## Зависимости

### Python

```python
river>=0.18.0          # Основная библиотека онлайн-обучения
scikit-multiflow>=0.5  # Альтернативная библиотека
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
yfinance>=0.2.0        # Для загрузки данных (акции)
```

### Rust

```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
ndarray = "0.15"
```

## Ожидаемые результаты

1. **Фреймворк потоковой симуляции** для дневных данных
2. **Онлайн-модели** (линейные, деревья, ансамбли)
3. **Пайплайн обнаружения дрейфа** с ADWIN/DDM
4. **Адаптивные веса momentum**, эволюционирующие со временем
5. **Результаты:** Более высокий Sharpe в нестационарные периоды vs статическая модель

## Литература

### Книги

- [Online Machine Learning](https://www.amazon.com/Online-Machine-Learning-Foundations-Applications/dp/0262046113) — Основы онлайн-обучения
- [Prediction, Learning, and Games](https://www.cambridge.org/core/books/prediction-learning-and-games/2CFDACECE01D9A4E9F7ACF8B0D30D933) — Теоретические основы

### Статьи

- [A Survey on Concept Drift Adaptation](https://dl.acm.org/doi/10.1145/2523813) — Обзор методов адаптации к дрейфу
- [Adaptive Learning Rate Methods](https://arxiv.org/abs/1412.6980) — Статья про Adam optimizer

### Документация

- [River Documentation](https://riverml.xyz/) — Документация библиотеки River
- [scikit-multiflow](https://scikit-multiflow.readthedocs.io/) — Альтернативная библиотека

## Уровень сложности

⭐⭐⭐☆☆ (Средний)

**Необходимые знания:**
- Online optimization (онлайн-оптимизация)
- Concept drift (дрейф концепций)
- Streaming algorithms (потоковые алгоритмы)
- Momentum factors (факторы моментума)

## Практические советы

### Выбор learning rate

- **Слишком высокий**: Модель нестабильна, реагирует на шум
- **Слишком низкий**: Медленная адаптация к реальным изменениям
- **Рекомендация**: Начинать с 0.01, использовать адаптивные методы (Adam)

### Когда использовать онлайн-обучение

✅ **Используйте когда:**
- Данные поступают потоком
- Распределение данных меняется со временем
- Нужна быстрая адаптация
- Ограничены вычислительные ресурсы

❌ **Не используйте когда:**
- Данные статичны
- Нужна максимальная точность на фиксированном датасете
- Мало данных для обучения
