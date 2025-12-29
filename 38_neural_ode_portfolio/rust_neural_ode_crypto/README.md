# Neural ODE для криптовалютного портфеля (Rust)

Модульная реализация Neural ODE для оптимизации криптовалютного портфеля на Rust с использованием данных биржи Bybit.

## Возможности

- **Bybit API клиент** — получение исторических свечей и текущих тикеров
- **ODE солверы** — Euler, RK4, Dopri5 (адаптивный шаг)
- **Neural ODE модель** — непрерывная динамика портфеля
- **Технические индикаторы** — RSI, MACD, Bollinger Bands, ATR и др.
- **Стратегия ребалансирования** — непрерывное управление весами
- **Бэктестинг** — оценка эффективности стратегии

## Установка

```bash
cd rust_neural_ode_crypto
cargo build --release
```

## Использование

### CLI

```bash
# Получить данные с Bybit
cargo run -- fetch -s BTCUSDT,ETHUSDT,SOLUSDT -i 60 -l 1000 -o data

# Запустить бэктест
cargo run -- backtest -d data/btcusdt.csv -v 100000 -t 0.02

# Демонстрация ODE солверов
cargo run -- demo -s all
```

### Примеры

```bash
# Получение данных с Bybit
cargo run --example fetch_bybit_data

# Обучение модели
cargo run --example train_portfolio_ode

# Симуляция live-ребалансирования
cargo run --example live_rebalancing
```

## Структура проекта

```
rust_neural_ode_crypto/
├── Cargo.toml              # Зависимости и конфигурация
├── README.md               # Документация
├── src/
│   ├── lib.rs              # Корень библиотеки
│   ├── main.rs             # CLI приложение
│   ├── data/               # Работа с данными
│   │   ├── mod.rs          # Модуль данных
│   │   ├── bybit.rs        # Bybit API клиент
│   │   ├── candles.rs      # Свечные данные OHLCV
│   │   └── features.rs     # Технические индикаторы
│   ├── ode/                # ODE солверы
│   │   ├── mod.rs          # Модуль солверов
│   │   ├── euler.rs        # Метод Эйлера
│   │   ├── rk4.rs          # Runge-Kutta 4
│   │   └── dopri5.rs       # Dormand-Prince 5(4)
│   ├── model/              # Нейросетевые модели
│   │   ├── mod.rs          # Модуль моделей
│   │   ├── network.rs      # MLP, активации
│   │   ├── portfolio.rs    # Neural ODE для портфеля
│   │   └── training.rs     # Обучение модели
│   └── strategy/           # Торговые стратегии
│       ├── mod.rs          # Модуль стратегий
│       ├── rebalancer.rs   # Непрерывный ребалансировщик
│       └── backtest.rs     # Бэктестинг
└── examples/
    ├── fetch_bybit_data.rs    # Получение данных
    ├── train_portfolio_ode.rs # Обучение модели
    └── live_rebalancing.rs    # Live ребалансирование
```

## API

### Получение данных

```rust
use neural_ode_crypto::data::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BybitClient::new();

    // Получить 1000 часовых свечей для BTCUSDT
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;

    // Получить текущий тикер
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("BTC: ${}", ticker.last_price);

    Ok(())
}
```

### ODE солверы

```rust
use neural_ode_crypto::ode::{ODESolver, Dopri5Solver, ClosureODE};
use ndarray::Array1;

// Определяем ODE: dz/dt = -z (экспоненциальный распад)
let ode = ClosureODE::new(
    |z: &Array1<f64>, _t: f64| -z.clone(),
    1,  // размерность
);

// Решаем с Dopri5
let solver = Dopri5Solver::default();
let z0 = Array1::from_vec(vec![1.0]);
let (times, states) = solver.solve(&ode, z0, (0.0, 2.0), 21);

// states содержит траекторию решения
```

### Neural ODE для портфеля

```rust
use neural_ode_crypto::model::NeuralODEPortfolio;
use neural_ode_crypto::data::Features;

// Создаём модель: 3 актива, 12 признаков, 16 скрытых
let model = NeuralODEPortfolio::new(3, 12, 16);

// Начальные веса
let weights = vec![0.4, 0.35, 0.25];

// Признаки рынка
let features = Features::new(3, 12);

// Получить траекторию весов
let trajectory = model.solve_trajectory(
    &weights,
    &features,
    (0.0, 1.0),  // временной интервал
    50,          // количество точек
);

// Получить целевые веса
let target = model.get_target_weights(&weights, &features, 0.5);
```

### Стратегия ребалансирования

```rust
use neural_ode_crypto::strategy::ContinuousRebalancer;
use neural_ode_crypto::model::NeuralODEPortfolio;

// Создаём ребалансировщик
let model = NeuralODEPortfolio::new(3, 12, 16);
let rebalancer = ContinuousRebalancer::new(model, 0.02)  // порог 2%
    .with_asset_names(vec!["BTC", "ETH", "SOL"])
    .with_transaction_cost(0.001);

// Проверяем необходимость ребалансирования
let decision = rebalancer.check_rebalance(&current_weights, &features);

if decision.should_rebalance {
    // Выполняем ребалансирование
    let result = rebalancer.execute_rebalance(
        &current_weights,
        &features,
        portfolio_value,
    );

    println!("Trades: {:?}", result.trades);
    println!("Cost: ${:.2}", result.transaction_cost);
}
```

## Тестирование

```bash
# Запустить все тесты
cargo test

# Запустить с выводом
cargo test -- --nocapture

# Запустить тесты модуля
cargo test data::
cargo test ode::
cargo test model::
cargo test strategy::
```

## Benchmark

```bash
# ODE солверы
cargo run --example demo_solvers

# Пример вывода:
# Euler:  Error 4.5e-03, Time 0.1ms
# RK4:    Error 2.1e-09, Time 0.2ms
# Dopri5: Error 1.2e-11, Time 0.3ms
```

## Зависимости

- `tokio` — асинхронный runtime
- `reqwest` — HTTP клиент для Bybit API
- `ndarray` — N-мерные массивы (аналог NumPy)
- `serde` — сериализация
- `clap` — парсинг CLI
- `tracing` — логирование

## Ограничения

- Обучение использует простой эволюционный алгоритм (не backprop)
- Для production рекомендуется использовать `tch-rs` (PyTorch bindings)
- Bybit API имеет rate limits

## Лицензия

MIT

## Связанные главы

- [Глава 17: Deep Learning](../../17_deep_learning)
- [Глава 19: RNN](../../19_recurrent_neural_nets)
- [Глава 22: Deep RL](../../22_deep_reinforcement_learning)
- [Глава 25: Diffusion Models (Rust пример)](../../25_diffusion_models_for_trading/rust_diffusion_crypto)
