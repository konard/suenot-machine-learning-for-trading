# Карты значимости для трейдинга - Реализация на Rust

Эта директория содержит реализацию вычисления карт значимости для торговли криптовалютами на языке Rust.

## Возможности

- **Клиент API Bybit**: Получение данных OHLCV с биржи Bybit
- **Нейронная сеть**: Простая полносвязная сеть для прогнозирования цен
- **Вычисление значимости**: Несколько методов на основе градиентов
- **Торговая стратегия**: Генерация сигналов с фильтрацией по значимости
- **Бэктестинг**: Фреймворк для оценки производительности

## Структура проекта

```
rust_examples/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Точка входа библиотеки
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs        # Клиент API Bybit
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs    # Предобработка данных
│   │   └── features.rs     # Технические индикаторы
│   ├── models/
│   │   ├── mod.rs
│   │   └── network.rs      # Нейронная сеть
│   └── saliency/
│       ├── mod.rs
│       └── gradient.rs     # Вычисление значимости
└── examples/
    ├── fetch_data.rs       # Пример получения данных
    ├── saliency_trading.rs # Пример анализа значимости
    └── backtest.rs         # Пример бэктестинга
```

## Начало работы

### Требования

- Rust 1.70 или новее
- Менеджер пакетов Cargo

### Сборка

```bash
cargo build --release
```

### Запуск примеров

#### Получение рыночных данных
```bash
cargo run --example fetch_data
```

#### Вычисление карт значимости
```bash
cargo run --example saliency_trading
```

#### Запуск бэктеста
```bash
cargo run --example backtest
```

## Использование API

### Создание торговой сети

```rust
use saliency_maps_trading::models::network::TradingNetwork;

let network = TradingNetwork::new(
    30,           // длина последовательности
    5,            // количество признаков (OHLCV)
    &[64, 32]     // размеры скрытых слоёв
);
```

### Вычисление карт значимости

```rust
use saliency_maps_trading::saliency::gradient::SaliencyComputer;
use ndarray::Array2;

let computer = SaliencyComputer::new(network);
let input = Array2::from_shape_fn((30, 5), |_| 0.5);

// Ванильный градиент
let vanilla = computer.vanilla_gradient(&input);

// Градиент × Вход
let grad_input = computer.gradient_x_input(&input);

// Интегрированные градиенты
let integrated = computer.integrated_gradients(&input, 50);

// SmoothGrad
let smooth = computer.smoothgrad(&input, 0.1, 20);
```

### Генерация торговых сигналов

```rust
use saliency_maps_trading::saliency::gradient::SaliencyTrader;

let trader = SaliencyTrader::new(
    network,
    0.55,          // минимальная уверенность
    0.3,           // минимальная концентрация
    vec![3, 4],    // интерпретируемые признаки (Close, Volume)
);

let (signal, position_size) = trader.generate_signal(&input);
// signal: 1 (лонг), -1 (шорт), или 0 (без сделки)
```

## Методы значимости

| Метод | Описание | Применение |
|-------|----------|------------|
| Ванильный градиент | Базовый анализ чувствительности | Быстрый обзор |
| Градиент × Вход | Масштабирование по величине входа | Лучшая атрибуция |
| Интегрированные градиенты | Атрибуция на основе пути | Теоретические гарантии |
| SmoothGrad | Усреднённые по шуму градиенты | Более чистые визуализации |

## Зависимости

- `reqwest` - HTTP-клиент
- `serde` - Сериализация
- `ndarray` - N-мерные массивы
- `chrono` - Работа с датами
- `rand` - Генерация случайных чисел

## Лицензия

MIT License
