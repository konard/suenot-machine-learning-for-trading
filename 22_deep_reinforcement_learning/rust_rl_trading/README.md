# Rust RL Trading

Модульная библиотека глубокого обучения с подкреплением для торговли криптовалютами на бирже Bybit.

## Структура проекта

```
rust_rl_trading/
├── Cargo.toml              # Зависимости проекта
├── README.md               # Документация
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── agent/              # RL агенты
│   │   ├── mod.rs
│   │   ├── traits.rs       # Трейт Agent
│   │   ├── q_learning.rs   # Табличный Q-Learning
│   │   ├── dqn_agent.rs    # Deep Q-Network агент
│   │   ├── neural_network.rs   # Нейронная сеть
│   │   └── experience_replay.rs # Буфер воспроизведения
│   ├── environment/        # Торговая среда
│   │   ├── mod.rs
│   │   ├── trading_env.rs  # Основная среда
│   │   └── trading_state.rs # Состояния и действия
│   ├── data/               # Работа с данными
│   │   ├── mod.rs
│   │   ├── bybit_client.rs # Клиент Bybit API
│   │   ├── candle.rs       # OHLCV свечи
│   │   └── market_data.rs  # Технические индикаторы
│   ├── utils/              # Утилиты
│   │   ├── mod.rs
│   │   ├── config.rs       # Конфигурация
│   │   └── metrics.rs      # Финансовые метрики
│   └── bin/                # Исполняемые файлы
│       ├── fetch_data.rs   # Загрузка данных
│       ├── train_agent.rs  # Обучение агента
│       └── backtest.rs     # Бэктестинг
├── data/                   # Данные (создаётся автоматически)
└── models/                 # Сохранённые модели
```

## Быстрый старт

### 1. Сборка проекта

```bash
cd rust_rl_trading
cargo build --release
```

### 2. Загрузка данных с Bybit

```bash
# Загрузить данные BTCUSDT за последний год (часовые свечи)
cargo run --release --bin fetch_data -- BTCUSDT 365 60

# Загрузить данные ETHUSDT за 180 дней (15-минутные свечи)
cargo run --release --bin fetch_data -- ETHUSDT 180 15
```

### 3. Обучение агента

```bash
# Обучить на загруженных данных
cargo run --release --bin train_agent -- data/BTCUSDT_60_365d.csv

# Или загрузить данные автоматически и обучить
cargo run --release --bin train_agent
```

### 4. Бэктестинг

```bash
cargo run --release --bin backtest -- models/best_agent.json data/BTCUSDT_60_365d.csv
```

## Конфигурация

### Переменные окружения

```bash
export BYBIT_API_KEY="your_api_key"        # Опционально для приватных эндпоинтов
export BYBIT_API_SECRET="your_api_secret"
export TRADING_SYMBOL="BTCUSDT"
export USE_TESTNET="false"
```

### Файл конфигурации (config.json)

```json
{
  "bybit": {
    "symbol": "BTCUSDT",
    "interval": "60",
    "use_testnet": false
  },
  "training": {
    "num_episodes": 1000,
    "batch_size": 64,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 100000,
    "hidden_layers": [128, 64],
    "double_dqn": true
  },
  "environment": {
    "episode_length": 252,
    "trading_cost_bps": 0.001,
    "initial_capital": 10000.0,
    "max_drawdown": 0.3
  }
}
```

## Модули

### Agent (Агенты)

- **QLearningAgent**: Табличный Q-Learning для простых сред
- **DQNAgent**: Deep Q-Network с Double DQN поддержкой
- **ReplayBuffer**: Буфер воспроизведения опыта
- **NeuralNetwork**: Простая полносвязная нейронная сеть

### Environment (Среда)

- **TradingEnvironment**: OpenAI Gym-подобная среда для торговли
- **TradingAction**: Действия (SHORT, HOLD, LONG)
- **TradingState**: Состояние (рыночные данные + позиция)

### Data (Данные)

- **BybitClient**: Клиент для Bybit REST API
- **Candle**: OHLCV свечи
- **MarketData**: Технические индикаторы (SMA, RSI, MACD, Bollinger Bands, ATR)

### Utils (Утилиты)

- **PerformanceMetrics**: Sharpe, Sortino, Max Drawdown, Calmar и др.
- **AppConfig**: Конфигурация приложения

## Технические индикаторы

Автоматически рассчитываются следующие индикаторы:
- Returns (доходность)
- SMA (простое скользящее среднее) - короткий и длинный периоды
- RSI (индекс относительной силы)
- MACD (схождение-расхождение скользящих средних)
- Bollinger Bands (полосы Боллинджера)
- ATR (средний истинный диапазон)

## Примеры использования

### Создание агента программно

```rust
use rust_rl_trading::{
    agent::{DQNAgent, dqn_agent::DQNConfig},
    environment::{TradingEnvironment, EnvConfig},
    data::{BybitClient, MarketData, Interval},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Загрузка данных
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None).await?;
    let market_data = MarketData::from_candles(candles);

    // Создание среды
    let env = TradingEnvironment::with_default_config(market_data);

    // Создание агента
    let config = DQNConfig::default();
    let agent = DQNAgent::new(env.state_size(), env.action_size(), config);

    Ok(())
}
```

### Кастомное обучение

```rust
use rust_rl_trading::agent::Agent;

// Обучающий цикл
for episode in 0..1000 {
    let mut state = env.reset();
    let mut done = false;

    while !done {
        let action = agent.select_action(&state, agent.get_epsilon());
        let result = env.step(action);

        agent.remember_transition(
            state.clone(),
            action,
            result.reward,
            result.state.clone(),
            result.done,
        );

        if agent.can_train() {
            agent.train_step();
        }

        state = result.state;
        done = result.done;
    }

    agent.decay_epsilon();
}
```

## Лицензия

MIT

## Отказ от ответственности

Это программное обеспечение предназначено только для образовательных целей. Торговля криптовалютами связана с высоким риском потери капитала. Авторы не несут ответственности за финансовые потери, понесённые при использовании данного программного обеспечения.
