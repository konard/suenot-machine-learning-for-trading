# Глава 353: Расширенные Свёртки (Dilated Convolutions) для Трейдинга

## Обзор

**Расширенные свёртки** (также известные как atrous-свёртки или свёртки с дырами) — это мощная техника обработки последовательных данных, которая позволяет нейронным сетям иметь экспоненциально большое рецептивное поле без увеличения количества параметров и потери разрешения. В трейдинге это позволяет моделям одновременно улавливать краткосрочные паттерны и долгосрочные зависимости.

### Почему Расширенные Свёртки для Трейдинга?

1. **Многомасштабное распознавание паттернов**: Захват микроструктуры (тик-уровень) и макротрендов (недельные) в одной модели
2. **Вычислительная эффективность**: Экспоненциальный рост рецептивного поля при линейном росте параметров
3. **Без потери информации**: В отличие от pooling, сохраняется полное временное разрешение
4. **Параллелизуемость**: В отличие от RNN, можно обрабатывать всю последовательность параллельно
5. **Каузальное моделирование**: Можно настроить для онлайн-прогнозов (без утечки будущей информации)

### Ключевые Преимущества перед Традиционными Подходами

| Подход | Рецептивное поле | Параметры | Параллелизация | Потеря информации |
|--------|------------------|-----------|----------------|-------------------|
| Dense слои | Ограниченное | O(n²) | Да | Да |
| Стандартные CNN | Линейное | O(k) | Да | Опционально |
| RNN/LSTM | Неограниченное | O(1) | Нет | Затухание градиента |
| **Dilated CNN** | Экспоненциальное | O(log n) | Да | Нет |

## Теоретические Основы

### Стандартная vs Расширенная Свёртка

**Стандартная 1D свёртка** с ядром размера k:
```
y[t] = Σᵢ w[i] · x[t - i]  для i ∈ [0, k-1]
```

**Расширенная свёртка** с ядром размера k и коэффициентом расширения d:
```
y[t] = Σᵢ w[i] · x[t - i·d]  для i ∈ [0, k-1]
```

Коэффициент расширения `d` вводит промежутки между элементами ядра, позволяя свёртке "перепрыгивать" через входные значения.

### Рост Рецептивного Поля

Для стека из L слоёв с ядром размера k и коэффициентами расширения d₁, d₂, ..., dₗ:

**Рецептивное поле = 1 + Σᵢ (k - 1) × dᵢ**

При экспоненциально увеличивающемся расширении (d = 1, 2, 4, 8, ...):
- Слой 1: d=1 → рецептивное поле = k
- Слой 2: d=2 → рецептивное поле = k + 2(k-1) = 3k - 2
- Слой 3: d=4 → рецептивное поле = 3k - 2 + 4(k-1) = 7k - 6
- Слой L: рецептивное поле ≈ 2^L × k

### Архитектура WaveNet

Знаковая архитектура WaveNet использует:
1. **Каузальные свёртки**: Использует только прошлую информацию
2. **Расширенные свёртки**: Экспоненциально увеличивающееся расширение
3. **Остаточные соединения**: Для потока градиентов
4. **Gated-активации**: tanh(Wf * x) ⊙ σ(Wg * x)

```
                    ┌─────────────────────────────────┐
                    │         Выходной слой           │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │   Остаточный блок (d=8)         │
                    │ ┌─────────┐   ┌─────────┐       │
                    │ │Расшир.  │   │ 1×1     │       │
                 ┌──┼─│ свёртка ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Остаточный блок (d=4)         │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 ├──┼─│Расшир.  │   │ 1×1     │       │
                 │  │ │ свёртка ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Остаточный блок (d=2)         │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 ├──┼─│Расшир.  │   │ 1×1     │       │
                 │  │ │ свёртка ├───│ Conv    ├───────┼──►
                 │  │ └─────────┘   └─────────┘       │
                 │  └─────────────────────────────────┘
                 │                  │
                 │  ┌─────────────▼───────────────────┐
                 │  │   Остаточный блок (d=1)         │
                 │  │ ┌─────────┐   ┌─────────┐       │
                 └──┼─│Расшир.  │   │ 1×1     │       │
                    │ │ свёртка ├───│ Conv    ├───────┼──►
                    │ └─────────┘   └─────────┘       │
                    └─────────────────────────────────┘
                                  ▲
                    ┌─────────────┴───────────────────┐
                    │         Входной слой            │
                    │  (Цена, Объём, Признаки)        │
                    └─────────────────────────────────┘
```

## Торговая Стратегия

### Описание Стратегии

Используем расширенные каузальные свёртки для прогнозирования:
1. **Направление**: Движение цены в следующий период (вверх/вниз/нейтрально)
2. **Величина**: Ожидаемая доходность
3. **Волатильность**: Уровень риска для определения размера позиции

### Многомасштабное Извлечение Признаков

Ключевая идея: разные коэффициенты расширения захватывают разные временные масштабы:

| Коэффициент расширения | Ядро=3 Рецептивное поле | Интерпретация для трейдинга |
|------------------------|-------------------------|----------------------------|
| d=1 | 3 бара | Тик-уровневые паттерны |
| d=2 | 7 баров | Краткосрочный моментум |
| d=4 | 15 баров | Внутридневные тренды |
| d=8 | 31 бар | Дневные паттерны |
| d=16 | 63 бара | Недельные циклы |
| d=32 | 127 баров | Месячные тренды |

### Входные Признаки

```
Для каждого момента времени t:
- price_returns[t] = (close[t] - close[t-1]) / close[t-1]
- log_volume[t] = log(volume[t] + 1)
- high_low_range[t] = (high[t] - low[t]) / close[t]
- close_position[t] = (close[t] - low[t]) / (high[t] - low[t])
- volume_ma_ratio[t] = volume[t] / SMA(volume, 20)[t]
```

### Архитектура для Трейдинга

```python
class DilatedTradingModel:
    def __init__(self,
                 input_channels=5,
                 residual_channels=32,
                 skip_channels=64,
                 n_layers=8,
                 kernel_size=3):

        self.dilation_rates = [2**i for i in range(n_layers)]  # 1,2,4,8,16,32,64,128

        # Проекция входа
        self.input_conv = CausalConv1d(input_channels, residual_channels, 1)

        # Расширенные остаточные блоки
        self.residual_blocks = [
            DilatedResidualBlock(
                residual_channels,
                skip_channels,
                kernel_size,
                dilation=d
            ) for d in self.dilation_rates
        ]

        # Выходные слои
        self.output_conv1 = Conv1d(skip_channels, 64, 1)
        self.output_conv2 = Conv1d(64, 3, 1)  # [направление, величина, волатильность]

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)

        x = self.input_conv(x)

        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Суммируем все skip-соединения
        out = sum(skip_connections)
        out = F.relu(out)
        out = self.output_conv1(out)
        out = F.relu(out)
        out = self.output_conv2(out)

        return out
```

## Детали Реализации

### Каузальная Расширенная Свёртка

Для онлайн-прогнозирования нужны **каузальные** свёртки, которые смотрят только в прошлое:

```python
class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Левый padding для обеспечения каузальности
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        # Удаляем правый padding для каузальности
        return out[:, :, :-self.padding] if self.padding > 0 else out
```

### Gated-Активация

Gated-активация из WaveNet:

```python
class GatedActivation(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalDilatedConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalDilatedConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        return filter_out * gate_out
```

### Остаточный Блок

```python
class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()

        self.gated_activation = GatedActivation(residual_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x):
        out = self.gated_activation(x)

        skip = self.skip_conv(out)
        residual = self.residual_conv(out) + x

        return residual, skip
```

## Pipeline Обучения

### Подготовка Данных

```python
def prepare_training_data(klines, sequence_length=512, forecast_horizon=1):
    """
    Подготовка последовательностей для обучения.

    Args:
        klines: Список OHLCV свечей
        sequence_length: Длина входной последовательности
        forecast_horizon: На сколько шагов вперёд прогнозируем

    Returns:
        X: Входные последовательности (batch, channels, sequence_length)
        y: Целевые значения (batch, 3)  # [направление, величина, волатильность]
    """
    # Вычисляем признаки
    features = calculate_features(klines)

    # Создаём последовательности
    X, y = [], []
    for i in range(len(features) - sequence_length - forecast_horizon):
        X.append(features[i:i+sequence_length])

        # Цель: доходность следующего периода
        future_return = (klines[i+sequence_length+forecast_horizon-1].close -
                        klines[i+sequence_length-1].close) / klines[i+sequence_length-1].close

        direction = 1 if future_return > 0.001 else (-1 if future_return < -0.001 else 0)
        magnitude = abs(future_return)
        volatility = calculate_volatility(klines[i:i+sequence_length])

        y.append([direction, magnitude, volatility])

    return np.array(X), np.array(y)
```

### Функция Потерь

```python
class TradingLoss(nn.Module):
    def __init__(self, direction_weight=1.0, magnitude_weight=0.5, volatility_weight=0.3):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.volatility_weight = volatility_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # pred shape: (batch, 3, 1) - предсказание для последнего таймстепа
        # target shape: (batch, 3)

        direction_loss = self.ce_loss(pred[:, :3, -1], target[:, 0].long() + 1)
        magnitude_loss = self.mse_loss(pred[:, 3, -1], target[:, 1])
        volatility_loss = self.mse_loss(pred[:, 4, -1], target[:, 2])

        return (self.direction_weight * direction_loss +
                self.magnitude_weight * magnitude_loss +
                self.volatility_weight * volatility_loss)
```

## Ключевые Метрики

### Производительность Модели
| Метрика | Описание |
|---------|----------|
| **Direction Accuracy** | Процент правильных предсказаний направления |
| **Magnitude MAE** | Средняя абсолютная ошибка предсказания доходности |
| **Sharpe Ratio** | Доходность с поправкой на риск |
| **Max Drawdown** | Максимальная просадка |
| **Win Rate** | Процент прибыльных сделок |

### Анализ Рецептивного Поля
| Метрика | Описание |
|---------|----------|
| **Effective RF** | Фактический размер рецептивного поля в таймстепах |
| **RF Utilization** | Насколько активно используется рецептивное поле |
| **Multi-scale Contribution** | Важность каждого уровня расширения |

## Зависимости

```toml
[dependencies]
# HTTP клиент для Bybit API
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# Асинхронный runtime
tokio = { version = "1.0", features = ["full"] }

# Сериализация
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Математика и статистика
ndarray = "0.15"
ndarray-stats = "0.5"

# Работа со временем
chrono = { version = "0.4", features = ["serde"] }

# Обработка ошибок
thiserror = "1.0"
anyhow = "1.0"

# Логирование
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Ожидаемые Результаты

1. **Модуль расширенных свёрток**: Чистая Rust-реализация с настраиваемыми коэффициентами расширения
2. **Архитектура в стиле WaveNet**: Полная реализация остаточных блоков
3. **Интеграция с Bybit**: Получение данных в реальном времени и расчёт признаков
4. **Торговая стратегия**: Генерация сигналов с определением размера позиции
5. **Фреймворк бэктестинга**: Оценка производительности на исторических данных

## Структура Проекта

```
353_dilated_convolutions_trading/
├── README.md                    # Английская версия
├── README.ru.md                 # Этот файл
├── readme.simple.md             # Простое объяснение
├── readme.simple.ru.md          # Простое объяснение (русский)
└── rust/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Корень библиотеки
    │   ├── api/                 # Клиент Bybit API
    │   │   ├── mod.rs
    │   │   ├── client.rs
    │   │   ├── types.rs
    │   │   └── error.rs
    │   ├── conv/                # Расширенные свёртки
    │   │   ├── mod.rs
    │   │   ├── dilated.rs
    │   │   ├── causal.rs
    │   │   └── wavenet.rs
    │   ├── features/            # Feature engineering
    │   │   ├── mod.rs
    │   │   ├── technical.rs
    │   │   └── normalization.rs
    │   ├── strategy/            # Торговая стратегия
    │   │   ├── mod.rs
    │   │   ├── signals.rs
    │   │   └── position.rs
    │   └── utils/               # Утилиты
    │       ├── mod.rs
    │       └── metrics.rs
    └── examples/
        ├── fetch_data.rs        # Получение данных Bybit
        ├── dilated_conv_demo.rs # Демо расширенных свёрток
        ├── wavenet_features.rs  # Извлечение признаков WaveNet
        └── trading_backtest.rs  # Пример бэктестинга
```

## Ссылки

### Научные Статьи
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — Оригинальная статья WaveNet
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) — Расширенные свёртки для семантической сегментации
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) — Статья о TCN

### Применение в Трейдинге
- [Deep Learning for Financial Time Series Prediction](https://arxiv.org/abs/2101.09187)
- [WaveNet-based Trading Signal Generation](https://arxiv.org/abs/2003.06503)

### Документация
- [Документация Bybit API](https://bybit-exchange.github.io/docs/v5/intro)

## Уровень Сложности

⭐⭐⭐⭐ (Продвинутый)

### Требуемые Знания
- **Глубокое обучение**: Архитектуры CNN, остаточные соединения
- **Обработка сигналов**: Операции свёртки, рецептивные поля
- **Анализ временных рядов**: Feature engineering, стационарность
- **Финансовые рынки**: Данные OHLCV, торговые сигналы
- **Программирование на Rust**: Асинхронное программирование, обработка ошибок
