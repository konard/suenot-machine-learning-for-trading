# Глава 352: Temporal Convolutional Networks (TCN) для Трейдинга

## Обзор

Temporal Convolutional Networks (TCN) — это революционный подход к моделированию последовательностей, который доказал свою высокую эффективность для прогнозирования финансовых временных рядов. В отличие от традиционных RNN и LSTM, TCN используют каузальные свёртки с дилатациями для захвата долгосрочных зависимостей, сохраняя при этом параллелизм и стабильность градиентов.

## Торговая Стратегия

**Суть стратегии:** Использование TCN для прогнозирования движения цен криптовалют и генерации торговых сигналов на основе многомасштабных временных паттернов.

**Ключевые преимущества:**
1. **Параллелизм** — В отличие от RNN, все временные шаги могут вычисляться параллельно при обучении
2. **Стабильные градиенты** — Нет проблем с затухающими/взрывающимися градиентами, характерных для RNN
3. **Гибкое рецептивное поле** — Дилатация позволяет экспоненциально увеличивать рецептивное поле
4. **Каузальная архитектура** — Нет утечки информации из будущего в прошлое

**Edge:** TCN могут одновременно захватывать как краткосрочные паттерны микроструктуры, так и долгосрочные тренды благодаря дилатированным свёрткам.

## Техническая Спецификация

### Компоненты Архитектуры

| Компонент | Описание |
|-----------|----------|
| Каузальная свёртка | Гарантирует, что выход в момент t зависит только от входов в моменты ≤ t |
| Дилатированная свёртка | Экспоненциально увеличивает рецептивное поле без увеличения параметров |
| Остаточные связи | Позволяют обучать глубокие сети |
| Нормализация весов | Стабилизирует обучение |

### Структура TCN Блока

```
Вход → Conv1D(dilation=1) → ReLU → Dropout →
       Conv1D(dilation=1) → ReLU → Dropout →
       + Остаточная связь → Выход
```

Несколько блоков складываются с возрастающими коэффициентами дилатации: 1, 2, 4, 8, 16, ...

### Математические Основы

#### Каузальная Свёртка

Для входной последовательности **x** и фильтра **f** размера k:
```
(x *_c f)(t) = Σ_{i=0}^{k-1} f(i) · x(t - i)
```

Ключевое свойство: Выход в момент t зависит только от x(t), x(t-1), ..., x(t-k+1)

#### Дилатированная Свёртка

Для фактора дилатации d:
```
(x *_d f)(t) = Σ_{i=0}^{k-1} f(i) · x(t - d·i)
```

С дилатацией рецептивное поле растёт экспоненциально: **R = 1 + (k-1) · Σ d_i**

#### Расчёт Рецептивного Поля

Для L слоёв с удвоением дилатации:
```
Рецептивное поле = 1 + 2·(k-1)·(2^L - 1)
```

Пример: k=3, L=8 → RF = 1 + 2·2·255 = 1021 временной шаг

### Ключевые Метрики

| Метрика | Описание |
|---------|----------|
| Направленная точность | Правильное предсказание направления движения цены |
| Коэффициент Шарпа | Доходность, скорректированная на риск |
| Коэффициент Сортино | Доходность, скорректированная на нисходящий риск |
| Максимальная просадка | Наибольшее падение от пика до дна |
| Процент выигрышных сделок | Доля прибыльных сделок |
| Профит-фактор | Валовая прибыль / Валовой убыток |

## Содержание

1. [Архитектура TCN](#архитектура-tcn)
2. [Почему TCN для трейдинга](#почему-tcn-для-трейдинга)
3. [Детали реализации](#детали-реализации)
4. [Инженерия признаков](#инженерия-признаков)
5. [Генерация торговых сигналов](#генерация-торговых-сигналов)
6. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
7. [Реализация на Rust](#реализация-на-rust)
8. [Литература](#литература)

## Архитектура TCN

### Основные Компоненты

#### 1. Слой Каузальной Свёртки

Каузальные свёртки гарантируют, что прогнозы в момент t используют только информацию, доступную в момент t или ранее:

```
Время:    t-4   t-3   t-2   t-1    t
           │     │     │     │     │
           └──┬──┴──┬──┴──┬──┴──┬──┘
              │     │     │     │
           [Conv1D, kernel_size=2, dilation=1]
              │     │     │     │
              └──┬──┴──┬──┴──┬──┘
                 │     │     │
              [Conv1D, kernel_size=2, dilation=2]
                 │     │     │
                 └──┬──┴──┬──┘
                    │     │
              [Conv1D, kernel_size=2, dilation=4]
                    │
                    ▼
              Прогноз для момента t
```

#### 2. Остаточный Блок

```rust
struct TCNResidualBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    relu: ReLU,
    dropout: Dropout,
    residual_conv: Option<Conv1d>,  // если каналы входа/выхода различаются
}
```

#### 3. Полная Сеть TCN

```rust
struct TCN {
    input_projection: Linear,
    residual_blocks: Vec<TCNResidualBlock>,
    output_projection: Linear,
}
```

### Гиперпараметры

| Параметр | Типичное значение | Описание |
|----------|------------------|----------|
| num_channels | [64, 64, 64, 64] | Каналы на слой |
| kernel_size | 3 | Размер ядра свёртки |
| dropout | 0.2 | Вероятность dropout |
| dilation_base | 2 | Множитель фактора дилатации |

## Почему TCN для Трейдинга

### Преимущества над RNN/LSTM

| Аспект | RNN/LSTM | TCN |
|--------|----------|-----|
| Параллелизм | Последовательная обработка | Полностью параллельная |
| Поток градиентов | Затухающий/взрывающийся | Стабильный (остаточный) |
| Память | Ограничена скрытым состоянием | Явная через рецептивное поле |
| Скорость обучения | Медленная (последовательная) | Быстрая (параллельная) |
| Долгосрочные зависимости | Сложно | Естественно (дилатации) |

### Преимущества для Трейдинга

1. **Многомасштабное распознавание паттернов**
   - Малые дилатации захватывают паттерны на уровне тиков (микроструктура)
   - Большие дилатации захватывают трендовые паттерны (макродвижения)

2. **Инференс в реальном времени**
   - Один прямой проход для прогноза
   - Постоянное время инференса независимо от длины последовательности

3. **Интерпретируемость**
   - Рецептивное поле явное и контролируемое
   - Возможна визуализация внимания через анализ градиентов

## Детали Реализации

### Входные Признаки

```rust
struct TradingFeatures {
    // На основе цены
    returns: Vec<f64>,           // Логарифмические доходности
    volatility: Vec<f64>,        // Скользящая волатильность

    // Технические индикаторы
    rsi: Vec<f64>,               // Индекс относительной силы
    macd: Vec<f64>,              // Линия MACD
    macd_signal: Vec<f64>,       // Сигнальная линия MACD
    bollinger_upper: Vec<f64>,   // Верхняя полоса Боллинджера
    bollinger_lower: Vec<f64>,   // Нижняя полоса Боллинджера

    // Признаки объёма
    volume_ratio: Vec<f64>,      // Объём / Средний объём
    obv: Vec<f64>,               // Балансовый объём (OBV)

    // Поток ордеров (если доступен)
    bid_ask_imbalance: Vec<f64>, // Дисбаланс объёма bid-ask
    trade_flow: Vec<f64>,        // Чистый поток покупок/продаж
}
```

### Варианты Целевой Переменной

1. **Классификация (Направление)**
   - Классы: Вверх, Вниз, Нейтрально
   - На основе порога: |доходность| > порог → направленный сигнал

2. **Регрессия (Величина доходности)**
   - Прогноз фактического значения доходности
   - Полезно для определения размера позиции

3. **Мультигоризонт**
   - Прогноз доходности на нескольких будущих горизонтах
   - [1 бар, 5 баров, 15 баров, 60 баров]

### Функции Потерь

```rust
enum TradingLoss {
    // Классификация
    CrossEntropy,
    FocalLoss { gamma: f64 },  // Для несбалансированных классов

    // Регрессия
    MSE,
    Huber { delta: f64 },      // Устойчива к выбросам
    Quantile { tau: f64 },     // Для интервалов прогноза

    // Специфичные для трейдинга
    SharpeRatioLoss,           // Максимизация Шарпа напрямую
    DirectionalAccuracy,       // Sign(прогноз) == Sign(факт)
}
```

## Инженерия Признаков

### Технические Индикаторы

```rust
impl TechnicalIndicators {
    pub fn calculate_all(&self, candles: &[Candle]) -> FeatureMatrix {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        FeatureMatrix {
            // Моментум
            rsi_14: self.rsi(&closes, 14),
            rsi_7: self.rsi(&closes, 7),
            macd: self.macd(&closes, 12, 26, 9),
            momentum_10: self.momentum(&closes, 10),
            roc_10: self.rate_of_change(&closes, 10),

            // Волатильность
            atr_14: self.atr(&highs, &lows, &closes, 14),
            bollinger: self.bollinger_bands(&closes, 20, 2.0),
            keltner: self.keltner_channels(&highs, &lows, &closes, 20),

            // Тренд
            sma_20: self.sma(&closes, 20),
            sma_50: self.sma(&closes, 50),
            ema_12: self.ema(&closes, 12),
            ema_26: self.ema(&closes, 26),
            adx_14: self.adx(&highs, &lows, &closes, 14),

            // Объём
            obv: self.on_balance_volume(&closes, &volumes),
            vwap: self.vwap(&highs, &lows, &closes, &volumes),
            volume_sma: self.sma(&volumes, 20),

            // Ценовое действие
            candle_patterns: self.detect_patterns(&candles),
            support_resistance: self.find_sr_levels(&closes, 20),
        }
    }
}
```

### Стратегия Нормализации

```rust
enum NormalizationMethod {
    // Z-score нормализация (скользящее окно)
    ZScore { window: usize },

    // Min-max в [0, 1] (скользящее окно)
    MinMax { window: usize },

    // Робастное масштабирование (медиана, IQR)
    Robust { window: usize },

    // Логарифмическое преобразование для цен
    LogReturns,

    // Процентильный ранг (0-100)
    PercentileRank { window: usize },
}
```

## Генерация Торговых Сигналов

### Конвейер Генерации Сигналов

```rust
struct SignalGenerator {
    tcn_model: TCN,
    threshold_long: f64,   // например, 0.6 для 60% уверенности
    threshold_short: f64,  // например, 0.6 для 60% уверенности
    position_sizer: PositionSizer,
}

impl SignalGenerator {
    pub fn generate_signal(&self, features: &FeatureMatrix) -> TradingSignal {
        // Получаем прогноз модели
        let prediction = self.tcn_model.predict(features);

        // Преобразуем в вероятности (softmax для классификации)
        let probs = softmax(&prediction);

        // Генерируем сигнал на основе порогов
        let signal = if probs.up > self.threshold_long {
            SignalType::Long
        } else if probs.down > self.threshold_short {
            SignalType::Short
        } else {
            SignalType::Neutral
        };

        // Вычисляем размер позиции на основе уверенности
        let confidence = probs.max();
        let position_size = self.position_sizer.calculate(confidence);

        TradingSignal {
            signal_type: signal,
            confidence,
            position_size,
            predicted_return: prediction.expected_return,
            timestamp: Utc::now(),
        }
    }
}
```

### Интеграция с Риск-Менеджментом

```rust
struct RiskManager {
    max_position_size: f64,     // Максимальный размер позиции как доля капитала
    max_daily_loss: f64,        // Прекратить торговлю при превышении дневного убытка
    stop_loss_pct: f64,         // Стоп-лосс на сделку
    take_profit_pct: f64,       // Тейк-профит на сделку
    max_drawdown_pct: f64,      // Максимальная просадка до снижения экспозиции
}

impl RiskManager {
    pub fn validate_signal(&self, signal: &TradingSignal, state: &PortfolioState) -> ValidatedSignal {
        let mut adjusted = signal.clone();

        // Проверяем лимит дневного убытка
        if state.daily_pnl < -self.max_daily_loss * state.capital {
            return ValidatedSignal::Blocked("Достигнут лимит дневного убытка");
        }

        // Снижаем экспозицию при просадке
        if state.current_drawdown > self.max_drawdown_pct * 0.5 {
            adjusted.position_size *= 0.5;
        }

        // Применяем максимальный лимит позиции
        adjusted.position_size = adjusted.position_size.min(self.max_position_size);

        // Устанавливаем стоп-лосс и тейк-профит
        adjusted.stop_loss = Some(self.stop_loss_pct);
        adjusted.take_profit = Some(self.take_profit_pct);

        ValidatedSignal::Approved(adjusted)
    }
}
```

## Фреймворк Бэктестинга

### Движок Бэктеста

```rust
struct BacktestEngine {
    initial_capital: f64,
    commission: f64,          // Комиссия за сделку
    slippage: f64,            // Оценочное проскальзывание
    margin_requirement: f64,  // Для маржинальной торговли
}

impl BacktestEngine {
    pub fn run(&self, signals: &[TradingSignal], prices: &[Candle]) -> BacktestResult {
        let mut capital = self.initial_capital;
        let mut position = 0.0;
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();

        for (i, (signal, candle)) in signals.iter().zip(prices).enumerate() {
            // Исполняем сигнал
            if signal.signal_type != SignalType::Neutral {
                let trade = self.execute_trade(signal, candle, &mut capital, &mut position);
                trades.push(trade);
            }

            // Переоценка по рынку
            let equity = capital + position * candle.close;
            equity_curve.push(EquityPoint {
                timestamp: candle.timestamp,
                equity,
                position,
            });
        }

        // Вычисляем метрики
        BacktestResult {
            total_return: (capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio: self.calculate_sharpe(&equity_curve),
            sortino_ratio: self.calculate_sortino(&equity_curve),
            max_drawdown: self.calculate_max_drawdown(&equity_curve),
            win_rate: self.calculate_win_rate(&trades),
            profit_factor: self.calculate_profit_factor(&trades),
            total_trades: trades.len(),
            avg_trade_duration: self.calculate_avg_duration(&trades),
            equity_curve,
            trades,
        }
    }
}
```

### Walk-Forward Валидация

```rust
struct WalkForwardValidator {
    train_period: usize,    // например, 1000 баров
    test_period: usize,     // например, 200 баров
    retrain_frequency: usize, // Переобучение каждые N баров
}

impl WalkForwardValidator {
    pub fn validate(&self, data: &MarketData, model_factory: &TCNFactory) -> ValidationResult {
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        let mut start = 0;
        while start + self.train_period + self.test_period <= data.len() {
            // Разделяем данные
            let train_data = &data[start..start + self.train_period];
            let test_data = &data[start + self.train_period..start + self.train_period + self.test_period];

            // Обучаем модель
            let model = model_factory.create_and_train(train_data);

            // Генерируем прогнозы на тестовом наборе
            let predictions = model.predict_batch(test_data);
            let actuals = self.extract_targets(test_data);

            all_predictions.extend(predictions);
            all_actuals.extend(actuals);

            // Сдвигаем окно
            start += self.retrain_frequency;
        }

        ValidationResult {
            predictions: all_predictions,
            actuals: all_actuals,
            metrics: self.calculate_metrics(&all_predictions, &all_actuals),
        }
    }
}
```

## Реализация на Rust

Полная реализация на Rust доступна в директории `rust_tcn_trading/`:

```
rust_tcn_trading/
├── Cargo.toml
└── src/
    ├── lib.rs              # Точка входа библиотеки
    ├── tcn/                 # Реализация TCN
    │   ├── mod.rs
    │   ├── layer.rs         # Слои каузальных свёрток
    │   ├── block.rs         # Остаточные блоки
    │   └── model.rs         # Полная модель TCN
    ├── features/            # Инженерия признаков
    │   ├── mod.rs
    │   ├── technical.rs     # Технические индикаторы
    │   └── normalize.rs     # Нормализация
    ├── trading/             # Торговая логика
    │   ├── mod.rs
    │   ├── signal.rs        # Генерация сигналов
    │   ├── risk.rs          # Риск-менеджмент
    │   └── backtest.rs      # Бэктестинг
    ├── api/                 # Клиент API Bybit
    │   ├── mod.rs
    │   ├── client.rs
    │   └── types.rs
    ├── utils/               # Утилиты
    │   ├── mod.rs
    │   └── metrics.rs       # Метрики производительности
    └── bin/                 # Примеры бинарников
        ├── fetch_data.rs    # Получение крипто-данных с Bybit
        ├── train_tcn.rs     # Обучение модели TCN
        ├── backtest.rs      # Запуск бэктеста
        └── live_signals.rs  # Генерация сигналов в реальном времени
```

### Быстрый Старт

```bash
# Клонируйте и перейдите в директорию главы
cd machine-learning-for-trading/352_tcn_trading/rust_tcn_trading

# Соберите проект
cargo build --release

# Получите рыночные данные
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000

# Обучите модель TCN
cargo run --bin train_tcn -- --data data/btcusdt_1h.csv --epochs 100

# Запустите бэктест
cargo run --bin backtest -- --model models/tcn.bin --data data/btcusdt_1h.csv
```

## Литература

### Оригинальная статья о TCN

1. **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling**
   - Авторы: Shaojie Bai, J. Zico Kolter, Vladlen Koltun
   - URL: https://arxiv.org/abs/1803.01271
   - Год: 2018
   - Ключевой вывод: TCN превосходят канонические рекуррентные сети в различных задачах моделирования последовательностей

### Финансовые Приложения

2. **Temporal Convolutional Networks for Financial Time Series**
   - Демонстрирует эффективность TCN для прогнозирования цен акций
   - Сравнение с моделями LSTM и Transformer

3. **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books**
   - Использует архитектуру CNN для данных книги ордеров
   - Аналогичные принципы применяются к высокочастотной торговле

4. **Stock Price Prediction Using Temporal Convolution Network**
   - Применение к различным фондовым рынкам
   - Лучшие практики инженерии признаков

### Связанные Архитектуры

5. **WaveNet: A Generative Model for Raw Audio**
   - DeepMind, 2016
   - Оригинальная архитектура дилатированных каузальных свёрток
   - https://arxiv.org/abs/1609.03499

6. **Attention Is All You Need (Transformer)**
   - Vaswani et al., 2017
   - Альтернативный подход к моделированию последовательностей
   - https://arxiv.org/abs/1706.03762

### Трейдинг и Микроструктура Рынка

7. **Advances in Financial Machine Learning**
   - Marcos López de Prado, 2018
   - Всеобъемлющее руководство по ML в финансах

8. **Machine Learning for Algorithmic Trading**
   - Stefan Jansen, 2020
   - Практическое руководство по реализации

## Дополнительное Чтение

- [TCN GitHub Repository](https://github.com/locuslab/TCN) - Оригинальная реализация
- [Keras TCN](https://github.com/philipperemy/keras-tcn) - Популярная реализация на Keras
- [PyTorch TCN](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py) - Справочная реализация на PyTorch

## Лицензия

MIT License - См. файл LICENSE для деталей.
