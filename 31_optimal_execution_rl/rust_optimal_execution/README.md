# Rust Optimal Execution

Модульная библиотека для оптимального исполнения ордеров с использованием обучения с подкреплением на криптовалютной бирже Bybit.

## Структура проекта

```
rust_optimal_execution/
├── Cargo.toml              # Зависимости проекта
├── README.md               # Документация
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── api/                # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   ├── types.rs        # Типы данных (Candle, OrderBook)
│   │   └── error.rs        # Обработка ошибок
│   ├── impact/             # Модели market impact
│   │   ├── mod.rs
│   │   ├── models.rs       # LinearImpact, SquareRootImpact, TransientImpact
│   │   └── params.rs       # Параметры моделей
│   ├── environment/        # Gym-подобная среда
│   │   ├── mod.rs
│   │   ├── env.rs          # ExecutionEnv
│   │   ├── state.rs        # ExecutionState, ExecutionAction
│   │   └── simulator.rs    # MarketSimulator
│   ├── agent/              # RL агенты
│   │   ├── mod.rs
│   │   ├── traits.rs       # Трейт Agent
│   │   ├── q_learning.rs   # Табличный Q-Learning
│   │   ├── dqn.rs          # Deep Q-Network
│   │   ├── neural_network.rs   # Нейронная сеть
│   │   └── replay_buffer.rs    # Experience Replay
│   ├── baselines/          # Классические алгоритмы
│   │   ├── mod.rs
│   │   ├── twap.rs         # Time-Weighted Average Price
│   │   ├── vwap.rs         # Volume-Weighted Average Price
│   │   ├── almgren_chriss.rs   # Optimal Execution (Almgren-Chriss)
│   │   └── schedule.rs     # ExecutionSchedule
│   ├── utils/              # Утилиты
│   │   ├── mod.rs
│   │   ├── metrics.rs      # ExecutionMetrics, PerformanceStats
│   │   └── storage.rs      # CSV сохранение/загрузка
│   └── bin/                # Исполняемые файлы
│       ├── fetch_data.rs   # Загрузка данных с Bybit
│       ├── train_agent.rs  # Обучение RL агента
│       ├── evaluate.rs     # Оценка стратегий
│       └── backtest.rs     # Бэктестинг
├── examples/               # Примеры
│   ├── simple_execution.rs
│   └── compare_strategies.rs
└── data/                   # Данные (создаётся автоматически)
```

## Быстрый старт

### 1. Сборка проекта

```bash
cd rust_optimal_execution
cargo build --release
```

### 2. Загрузка данных с Bybit

```bash
# Загрузить данные BTCUSDT за 180 дней (часовые свечи)
cargo run --release --bin fetch_data -- BTCUSDT 180 --interval 60

# Загрузить данные ETHUSDT за 365 дней (15-минутные свечи)
cargo run --release --bin fetch_data -- ETHUSDT 365 --interval 15
```

### 3. Обучение DQN агента

```bash
# Обучить на синтетических данных
cargo run --release --bin train_agent -- --episodes 5000

# Обучить на загруженных данных
cargo run --release --bin train_agent -- --data data/BTCUSDT_60_180d.csv --episodes 10000
```

### 4. Оценка стратегий

```bash
# Сравнить стратегии на синтетических данных
cargo run --release --bin evaluate

# С обученной моделью
cargo run --release --bin evaluate -- --model models/dqn_agent.json --data data/BTCUSDT_60_180d.csv
```

### 5. Бэктестинг

```bash
cargo run --release --bin backtest -- --model models/dqn_agent.json --data data/BTCUSDT_60_180d.csv
```

## Примеры использования

### Запуск примеров

```bash
# Простой пример исполнения
cargo run --example simple_execution

# Сравнение стратегий
cargo run --example compare_strategies
```

### Программное использование

```rust
use rust_optimal_execution::{
    api::{BybitClient, Interval},
    environment::{ExecutionEnv, EnvConfig, ExecutionAction},
    agent::{DQNAgent, DQNConfig, Agent},
    baselines::TWAPExecutor,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Загрузка данных
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None).await?;

    // Создание среды
    let config = EnvConfig::default();
    let mut env = ExecutionEnv::from_candles(candles, config);

    // Создание агента
    let dqn_config = DQNConfig {
        state_dim: env.state_dim(),
        num_actions: env.action_dim(),
        ..Default::default()
    };
    let mut agent = DQNAgent::new(dqn_config);

    // Обучение
    for episode in 0..1000 {
        let mut state = env.reset();
        let mut done = false;

        while !done {
            let action = agent.select_action(&state, agent.get_epsilon());
            let result = env.step(action);

            agent.remember(state.clone(), action, result.reward, result.state.clone(), result.done);

            if agent.can_train() {
                agent.train_step();
            }

            state = result.state;
            done = result.done;
        }

        agent.decay_epsilon();
    }

    Ok(())
}
```

## Модули

### API (api/)

- **BybitClient** — HTTP клиент для Bybit REST API
- **Candle** — OHLCV свечи
- **OrderBook** — Книга ордеров
- **Interval** — Интервалы свечей

### Market Impact (impact/)

- **LinearImpact** — Линейная модель impact
- **SquareRootImpact** — Модель квадратного корня (эмпирически обоснованная)
- **TransientImpact** — Затухающее воздействие
- **ImpactParams** — Параметры моделей

### Environment (environment/)

- **ExecutionEnv** — Gym-подобная среда для RL
- **ExecutionState** — Состояние (remaining, time, volatility, etc.)
- **ExecutionAction** — Действие (дискретное или непрерывное)
- **MarketSimulator** — Симулятор рыночной динамики

### Agent (agent/)

- **QLearningAgent** — Табличный Q-Learning
- **DQNAgent** — Deep Q-Network с Double DQN
- **ReplayBuffer** — Буфер воспроизведения опыта
- **NeuralNetwork** — Простая полносвязная сеть

### Baselines (baselines/)

- **TWAPExecutor** — Time-Weighted Average Price
- **VWAPExecutor** — Volume-Weighted Average Price
- **AlmgrenChrissExecutor** — Оптимальное аналитическое решение

### Utils (utils/)

- **ExecutionMetrics** — Метрики качества исполнения
- **PerformanceStats** — Статистика производительности
- **save_candles_csv / load_candles_csv** — Работа с CSV

## Метрики качества

- **Implementation Shortfall** — Разница между arrival price и execution price
- **Arrival Slippage** — Скольжение относительно цены входа
- **VWAP Slippage** — Скольжение относительно рыночного VWAP
- **Win Rate** — Процент эпизодов лучше baseline
- **Sharpe Ratio** — Риск-скорректированная производительность

## Конфигурация

### Параметры обучения

```rust
DQNConfig {
    state_dim: 10,           // Размерность состояния
    num_actions: 11,         // Количество действий (0%, 10%, ..., 100%)
    hidden_layers: vec![128, 64],
    learning_rate: 0.001,
    gamma: 0.99,             // Discount factor
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay: 0.995,
    buffer_size: 100000,
    batch_size: 64,
    target_update_freq: 100,
    tau: 0.005,              // Soft update coefficient
    double_dqn: true,
}
```

### Параметры среды

```rust
EnvConfig {
    total_quantity: 1000.0,   // Объём для исполнения
    max_steps: 60,            // Максимум шагов
    num_actions: 11,          // Количество действий
    risk_aversion: 1e-6,      // Коэффициент неприятия риска
    trading_cost: 0.0001,     // Комиссия (1 bps)
    non_execution_penalty: 0.01,
    discrete_actions: true,
}
```

## Лицензия

MIT

## Отказ от ответственности

Это программное обеспечение предназначено только для образовательных целей. Торговля криптовалютами связана с высоким риском потери капитала. Авторы не несут ответственности за финансовые потери, понесённые при использовании данного программного обеспечения.
