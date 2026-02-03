# Глава 129: MambaTS для Прогнозирования Временных Рядов

MambaTS (Mamba для временных рядов) — это инновационная архитектура, использующая селективную модель пространства состояний Mamba для долгосрочного прогнозирования временных рядов. Сочетая эффективность моделирования последовательностей с линейной сложностью и адаптивные механизмы отбора, MambaTS достигает передовых результатов в задачах прогнозирования финансовых временных рядов при сохранении вычислительной эффективности.

## Содержание

1. [Введение в MambaTS](#введение-в-mambats)
2. [Теоретические основы](#теоретические-основы)
   - [Модели пространства состояний (SSM)](#модели-пространства-состояний-ssm)
   - [От S4 к Mamba](#от-s4-к-mamba)
   - [Архитектура MambaTS](#архитектура-mambats)
3. [Ключевые компоненты](#ключевые-компоненты)
   - [Сканирование с учётом переменных](#сканирование-с-учётом-переменных)
   - [Патчирование временного разрешения](#патчирование-временного-разрешения)
   - [Смешивание каналов](#смешивание-каналов)
4. [Реализация](#реализация)
   - [Реализация на Python](#реализация-на-python)
   - [Реализация на Rust](#реализация-на-rust)
5. [Применение на финансовых рынках](#применение-на-финансовых-рынках)
6. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
7. [Ссылки](#ссылки)

---

## Введение в MambaTS

Традиционные модели на основе трансформеров для прогнозирования временных рядов сталкиваются со значительными проблемами при работе с длинными последовательностями из-за квадратичной сложности механизма внимания O(n²). MambaTS решает эту проблему, используя селективную модель пространства состояний Mamba, которая достигает линейной сложности O(n), сохраняя при этом высокие возможности моделирования.

### Почему MambaTS для трейдинга?

Финансовые временные ряды обладают несколькими характеристиками, которые делают MambaTS особенно подходящим:

1. **Долгосрочные зависимости**: Цены акций и рыночные индикаторы часто демонстрируют паттерны, охватывающие сотни временных шагов
2. **Мультимасштабные паттерны**: Рынки проявляют паттерны на разных временных масштабах (внутридневные, дневные, недельные)
3. **Множество переменных**: Трейдинг требует моделирования корреляций между различными активами и признаками
4. **Требования реального времени**: Низкая задержка вывода критична для алгоритмического трейдинга

---

## Теоретические основы

### Модели пространства состояний (SSM)

Модели пространства состояний представляют динамическую систему через скрытое состояние, эволюционирующее во времени:

```
h'(t) = Ah(t) + Bx(t)    # Эволюция состояния
y(t)  = Ch(t) + Dx(t)    # Выход
```

Где:
- `h(t)` — скрытое состояние
- `x(t)` — вход
- `y(t)` — выход
- `A`, `B`, `C`, `D` — обучаемые параметры

Для дискретных последовательностей это становится:

```
h[k] = Āh[k-1] + B̄x[k]
y[k] = Ch[k] + Dx[k]
```

### От S4 к Mamba

Модель **Structured State Space (S4)** представила эффективное вычисление SSM через:
- Диагональную структуру матрицы A
- Эффективное обучение на основе свёртки

**Mamba** улучшает S4, вводя **селективные пространства состояний**:
- Зависящие от входа параметры B, C и Δ (шаг дискретизации)
- Контекстно-зависимое рассуждение через механизм отбора
- Аппаратно-эффективный алгоритм параллельного сканирования

```python
# Упрощённый механизм отбора Mamba
def selective_ssm(x, A, B, C, delta):
    # B, C, delta теперь являются функциями входа x
    B = linear_B(x)
    C = linear_C(x)
    delta = softplus(linear_delta(x))

    # Дискретизация с зависящим от входа шагом
    A_bar = exp(delta * A)
    B_bar = delta * B

    # Селективное сканирование
    h = parallel_scan(A_bar, B_bar, x)
    y = einsum('bln,bln->bl', C, h)
    return y
```

### Архитектура MambaTS

MambaTS расширяет Mamba для многомерных временных рядов через несколько инноваций:

```
Вход: X ∈ R^(B×L×C)  [батч, длина_последовательности, каналы]
          ↓
┌─────────────────────────────────────┐
│    Эмбеддинг временных патчей       │
│  Разбиение последовательности       │
│  X_patch ∈ R^(B×N×P×C)              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Сканирование с учётом переменных │
│  Обработка каждой переменной SSM    │
│  Захват кросс-переменных паттернов  │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Слои энкодера Mamba (×N)         │
│  • Селективный SSM                  │
│  • Gated Linear Units               │
│  • Layer Normalization              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│    Голова предсказания              │
│  Проекция на горизонт прогноза      │
│  Y ∈ R^(B×H×C)                      │
└─────────────────────────────────────┘
```

---

## Ключевые компоненты

### Сканирование с учётом переменных

MambaTS обрабатывает многомерные временные ряды, учитывая как временные, так и межпеременные зависимости:

```python
class VariableAwareScanning(nn.Module):
    """
    Сканирование по временному и переменному измерениям
    для захвата сложных корреляций
    """
    def __init__(self, d_model, n_variables):
        super().__init__()
        self.temporal_mamba = MambaBlock(d_model)
        self.variable_mamba = MambaBlock(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: [batch, seq_len, n_vars, d_model]
        B, L, V, D = x.shape

        # Временное сканирование
        x_temp = rearrange(x, 'b l v d -> (b v) l d')
        h_temp = self.temporal_mamba(x_temp)
        h_temp = rearrange(h_temp, '(b v) l d -> b l v d', b=B, v=V)

        # Сканирование по переменным
        x_var = rearrange(x, 'b l v d -> (b l) v d')
        h_var = self.variable_mamba(x_var)
        h_var = rearrange(h_var, '(b l) v d -> b l v d', b=B, l=L)

        # Слияние
        return self.fusion(torch.cat([h_temp, h_var], dim=-1))
```

### Патчирование временного разрешения

Вместо обработки отдельных временных шагов MambaTS группирует шаги в патчи:

```python
class TemporalPatchEmbedding(nn.Module):
    """
    Преобразует временной ряд в патчи для эффективной обработки
    """
    def __init__(self, patch_size, d_model, n_variables):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * n_variables, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        B, L, V = x.shape

        # Создание патчей
        n_patches = L // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]
        x = x.reshape(B, n_patches, self.patch_size, V)
        x = x.reshape(B, n_patches, self.patch_size * V)

        # Проекция и добавление позиции
        x = self.projection(x)
        positions = torch.arange(n_patches, device=x.device)
        x = x + self.position_embedding(positions)

        return x
```

### Смешивание каналов

MambaTS использует стратегию смешивания каналов для моделирования зависимостей между различными финансовыми переменными:

```python
class ChannelMixing(nn.Module):
    """
    Смешивает информацию по каналам (переменным)
    """
    def __init__(self, n_channels, expansion_factor=2):
        super().__init__()
        hidden_dim = n_channels * expansion_factor
        self.fc1 = nn.Linear(n_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch, seq_len, n_channels]
        return self.fc2(self.activation(self.fc1(x)))
```

---

## Реализация

### Реализация на Python

Реализация на Python использует PyTorch и включает:

- `python/model.py` — архитектура модели MambaTS
- `python/train.py` — цикл обучения со смешанной точностью
- `python/backtest.py` — фреймворк для бэктестинга
- `python/data_loader.py` — загрузка финансовых данных

**Быстрый старт:**

```python
from model import MambaTS
from data_loader import FinancialDataLoader

# Инициализация модели
model = MambaTS(
    n_variables=10,        # OHLCV + технические индикаторы
    d_model=256,           # Скрытая размерность
    n_layers=4,            # Количество слоёв Mamba
    patch_size=16,         # Размер временного патча
    forecast_horizon=24    # Предсказание на 24 шага вперёд
)

# Загрузка данных
loader = FinancialDataLoader(
    symbols=['BTCUSDT', 'ETHUSDT'],
    interval='1h',
    lookback=720,          # 30 дней часовых данных
    source='bybit'
)

# Обучение
train_loader, val_loader = loader.get_dataloaders(batch_size=32)
trainer = MambaTSTrainer(model, lr=1e-4)
trainer.fit(train_loader, val_loader, epochs=100)
```

### Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительный вывод для продакшн торговых систем:

- `src/model/` — компоненты модели MambaTS
- `src/data/` — загрузка и предобработка данных
- `src/api/` — интеграция с Bybit API
- `examples/` — примеры использования

**Быстрый старт:**

```rust
use mambats_time_series::{MambaTS, BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Инициализация клиента Bybit
    let client = BybitClient::new();

    // Получение исторических данных
    let candles = client
        .get_klines("BTCUSDT", Interval::OneHour, start, end)
        .await?;

    // Загрузка обученной модели
    let model = MambaTS::load("models/mambats_btc.bin")?;

    // Предсказание
    let features = prepare_features(&candles);
    let prediction = model.predict(&features)?;

    println!("Прогнозируемое изменение цены: {:.4}%", prediction * 100.0);
    Ok(())
}
```

---

## Применение на финансовых рынках

### 1. Предсказание направления цены

MambaTS отлично справляется с предсказанием роста или падения цен:

```python
# Бинарная классификация для направления цены
model = MambaTS(
    n_variables=20,
    d_model=128,
    n_layers=3,
    output_dim=2,  # Вверх/Вниз
    task='classification'
)
```

### 2. Прогнозирование волатильности

Предсказание будущей волатильности для ценообразования опционов и управления рисками:

```python
# Регрессия для волатильности
model = MambaTS(
    n_variables=10,
    d_model=256,
    n_layers=4,
    output_dim=1,  # Одно значение волатильности
    task='regression'
)
```

### 3. Мультигоризонтное прогнозирование

Предсказание цен на несколько временных горизонтов:

```python
# Многошаговое прогнозирование
model = MambaTS(
    n_variables=5,
    d_model=256,
    n_layers=4,
    forecast_horizons=[1, 6, 12, 24],  # 1ч, 6ч, 12ч, 24ч
    task='multi_horizon'
)
```

---

## Фреймворк для бэктестинга

Глава включает комплексный фреймворк бэктестинга для оценки торговых стратегий:

```python
from backtest import Backtester, Strategy

class MambaTSStrategy(Strategy):
    def __init__(self, model, threshold=0.02):
        self.model = model
        self.threshold = threshold

    def generate_signals(self, data):
        predictions = self.model.predict(data)

        signals = np.zeros(len(predictions))
        signals[predictions > self.threshold] = 1   # Лонг
        signals[predictions < -self.threshold] = -1  # Шорт

        return signals

# Запуск бэктеста
backtester = Backtester(
    strategy=MambaTSStrategy(model),
    initial_capital=100000,
    commission=0.001
)

results = backtester.run(test_data)
print(f"Коэффициент Шарпа: {results.sharpe_ratio:.2f}")
print(f"Максимальная просадка: {results.max_drawdown:.2%}")
print(f"Общая доходность: {results.total_return:.2%}")
```

### Метрики производительности

| Метрика | Описание |
|---------|----------|
| Коэффициент Шарпа | Доходность с поправкой на риск (цель > 1.5) |
| Коэффициент Сортино | Доходность с учётом нисходящего риска |
| Макс. просадка | Наибольшее падение от пика до дна |
| Винрейт | Процент прибыльных сделок |
| Профит-фактор | Валовая прибыль / Валовой убыток |

---

## Структура директории

```
129_mambats_time_series/
├── README.md                 # Английская документация
├── README.ru.md              # Этот файл
├── README.specify.md         # Техническое задание
├── readme.simple.md          # Упрощённое объяснение (англ.)
├── readme.simple.ru.md       # Упрощённое объяснение (рус.)
├── python/
│   ├── __init__.py
│   ├── model.py              # Модель MambaTS
│   ├── mamba_block.py        # Базовые компоненты Mamba
│   ├── train.py              # Скрипт обучения
│   ├── backtest.py           # Фреймворк бэктестинга
│   ├── data_loader.py        # Загрузка финансовых данных
│   └── requirements.txt      # Зависимости Python
├── src/                      # Исходный код Rust
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── mamba.rs          # Блок Mamba
│   │   └── mambats.rs        # Полная модель
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs      # Предобработка данных
│   │   └── features.rs       # Технические индикаторы
│   └── api/
│       ├── mod.rs
│       └── bybit.rs          # Клиент Bybit API
├── examples/
│   ├── train_model.rs        # Пример обучения
│   ├── predict.rs            # Пример вывода
│   └── backtest.rs           # Пример бэктестинга
└── Cargo.toml                # Зависимости Rust
```

---

## Ссылки

1. **MambaTS: Improved Selective State Space Models for Long-term Forecasting**
   - URL: https://arxiv.org/abs/2405.16440
   - Год: 2024

2. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - Авторы: Albert Gu, Tri Dao
   - URL: https://arxiv.org/abs/2312.00752
   - Год: 2023

3. **Efficiently Modeling Long Sequences with Structured State Spaces (S4)**
   - Авторы: Albert Gu et al.
   - URL: https://arxiv.org/abs/2111.00396
   - Год: 2021

4. **Are Transformers Effective for Time Series Forecasting?**
   - URL: https://arxiv.org/abs/2205.13504
   - Год: 2022

5. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**
   - URL: https://arxiv.org/abs/2012.07436
   - Год: 2020

---

## Начало работы

### Настройка Python

```bash
cd python
pip install -r requirements.txt

# Обучение модели
python train.py --data bybit --symbol BTCUSDT --epochs 100

# Запуск бэктеста
python backtest.py --model models/mambats.pt --test-period 2024-01-01:2024-06-01
```

### Настройка Rust

```bash
# Сборка
cargo build --release

# Запуск примеров
cargo run --example train_model
cargo run --example predict
cargo run --example backtest
```

---

## Лицензия

Данная реализация предоставляется в образовательных целях как часть книги "Machine Learning for Trading".
