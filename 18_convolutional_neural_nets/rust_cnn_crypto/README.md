# CNN Crypto Trading (Rust)

Модульная реализация CNN для торговли криптовалютой на языке Rust. Использует данные биржи Bybit.

## Возможности

- Загрузка исторических данных с Bybit API
- Расчёт технических индикаторов (RSI, MACD, Bollinger Bands, ATR, OBV)
- 1D CNN модель для классификации направления цены
- Торговая стратегия с risk management (stop-loss, take-profit)
- Бэктестинг с расчётом метрик (Sharpe, Sortino, Max Drawdown)

## Структура проекта

```
rust_cnn_crypto/
├── Cargo.toml              # Конфигурация проекта
├── src/
│   ├── lib.rs              # Корень библиотеки
│   ├── bybit/              # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   └── types.rs        # Типы данных (Kline, etc.)
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── dataset.rs      # Dataset для батчевой загрузки
│   │   ├── processor.rs    # Препроцессинг данных
│   │   └── sample.rs       # Структура образца
│   ├── indicators/         # Технические индикаторы
│   │   └── mod.rs          # RSI, MACD, BB, ATR, OBV...
│   ├── model/              # CNN модель
│   │   ├── mod.rs
│   │   ├── cnn.rs          # Архитектура сети
│   │   ├── config.rs       # Конфигурация
│   │   └── training.rs     # Обучение и оценка
│   ├── trading/            # Торговля
│   │   ├── mod.rs
│   │   ├── signal.rs       # Торговые сигналы
│   │   ├── position.rs     # Управление позициями
│   │   ├── strategy.rs     # Торговая стратегия
│   │   └── backtest.rs     # Бэктестинг
│   └── bin/                # Исполняемые файлы
│       ├── fetch_data.rs   # Загрузка данных
│       ├── train.rs        # Обучение модели
│       ├── predict.rs      # Генерация предсказаний
│       └── backtest.rs     # Запуск бэктеста
└── data/                   # Данные (создаётся автоматически)
```

## Установка

```bash
# Клонирование репозитория
cd 18_convolutional_neural_nets/rust_cnn_crypto

# Сборка проекта
cargo build --release
```

## Использование

### 1. Загрузка данных

```bash
# Загрузить 30 дней данных BTCUSDT с интервалом 15 минут
cargo run --release --bin fetch_data -- --symbol BTCUSDT --interval 15 --days 30

# Другие символы
cargo run --release --bin fetch_data -- --symbol ETHUSDT --interval 15 --days 60
```

### 2. Обучение модели

```bash
# Обучить модель на загруженных данных
cargo run --release --bin train -- --data data/BTCUSDT_15_30d.csv
```

### 3. Генерация предсказаний

```bash
# Получить предсказание для текущего момента
cargo run --release --bin predict -- --symbol BTCUSDT
```

### 4. Бэктестинг

```bash
# Запустить бэктест на исторических данных
cargo run --release --bin backtest -- --data data/BTCUSDT_15_30d.csv --capital 10000
```

## Архитектура CNN

```
Input: [batch, 10 channels, 60 timesteps]
    │
    ├─ Conv1D(10 → 32, kernel=3) + ReLU + MaxPool(2)
    │
    ├─ Conv1D(32 → 64, kernel=3) + ReLU + MaxPool(2)
    │
    ├─ Conv1D(64 → 128, kernel=3) + ReLU
    │
    ├─ Flatten
    │
    ├─ Linear(... → 64) + ReLU + Dropout(0.3)
    │
    └─ Linear(64 → 3)  # [Down, Neutral, Up]
```

## Входные признаки (каналы)

1. **Returns** - простая доходность
2. **Log Returns** - логарифмическая доходность
3. **RSI** - Relative Strength Index (нормализованный)
4. **MACD Histogram** - гистограмма MACD
5. **Price vs SMA** - отклонение цены от скользящей средней
6. **ATR%** - Average True Range в процентах
7. **Volume Z-score** - нормализованный объём
8. **EMA Crossover** - разница быстрой и медленной EMA
9. **BB Position** - позиция в канале Bollinger Bands
10. **MACD** - значение MACD

## Конфигурация

### Модель (`CnnConfig`)

```rust
CnnConfig {
    in_channels: 10,        // Количество входных признаков
    input_size: 60,         // Размер окна (свечей)
    num_classes: 3,         // Классы: Down, Neutral, Up
    conv1_filters: 32,      // Фильтры 1-го слоя
    conv2_filters: 64,      // Фильтры 2-го слоя
    conv3_filters: 128,     // Фильтры 3-го слоя
    kernel_size: 3,         // Размер ядра свёртки
    pool_size: 2,           // Размер пулинга
    fc_size: 64,            // Размер полносвязного слоя
    dropout: 0.3,           // Dropout rate
}
```

### Обучение (`TrainingConfig`)

```rust
TrainingConfig {
    num_epochs: 50,
    batch_size: 32,
    learning_rate: 0.001,
    patience: 10,           // Early stopping
    use_class_weights: true,
}
```

### Стратегия (`StrategyConfig`)

```rust
StrategyConfig {
    min_signal_strength: 0.6,  // Мин. сила сигнала
    position_size: 0.2,        // 20% от капитала
    stop_loss_pct: 2.0,        // Stop-loss 2%
    take_profit_pct: 4.0,      // Take-profit 4%
    commission_rate: 0.001,    // Комиссия 0.1%
}
```

## Метрики бэктеста

- **Win Rate** - процент прибыльных сделок
- **Profit Factor** - отношение прибыли к убыткам
- **Max Drawdown** - максимальная просадка
- **Sharpe Ratio** - коэффициент Шарпа
- **Sortino Ratio** - коэффициент Сортино
- **Calmar Ratio** - отношение доходности к max drawdown

## Зависимости

- **burn** - Deep Learning framework для Rust
- **reqwest** - HTTP клиент для API
- **tokio** - Async runtime
- **ndarray** - N-dimensional arrays
- **chrono** - Работа с датами
- **serde** - Сериализация

## Важные замечания

1. **Это учебный проект** - не используйте для реальной торговли без тщательного тестирования
2. **Модель не обучена** - примеры показывают архитектуру, для реальных результатов нужно обучение
3. **API лимиты** - Bybit имеет rate limits, учитывайте при загрузке больших объёмов данных
4. **Риски** - торговля криптовалютой сопряжена с высокими рисками

## Расширение

### Добавление новых индикаторов

```rust
// В src/indicators/mod.rs
pub fn my_indicator(&self, data: &[f64]) -> Vec<f64> {
    // Ваша реализация
}
```

### Добавление в препроцессинг

```rust
// В src/data/processor.rs, метод klines_to_features
let my_ind = indicators.my_indicator(&close);
features[[N, i]] = my_ind[i]; // Добавляем новый канал
```

## Лицензия

MIT
