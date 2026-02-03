# Жидкостные машины состояний для трейдинга: резервуарные вычисления для финансовых временных рядов

Жидкостные машины состояний (Liquid State Machines, LSM) — это мощная форма резервуарных вычислений, вдохновлённая вычислительными принципами биологических нейронных сетей. Первоначально разработанные Вольфгангом Маассом в 2002 году, LSM используют рекуррентную сеть из спайковых нейронов в качестве «жидкости», которая преобразует временные входные паттерны в богатые многомерные представления, легко считываемые простыми линейными классификаторами.

В контексте алгоритмической торговли LSM предлагают уникальные преимущества для обработки финансовых временных рядов. Присущая резервуару временная память естественным образом захватывает динамику рыночных движений, что делает LSM особенно подходящими для таких задач, как предсказание цен, прогнозирование волатильности и генерация торговых сигналов.

Ключевые преимущества LSM для трейдинга:
- **Временная обработка**: Естественная обработка последовательных данных без явной инженерии признаков для временных зависимостей
- **Быстрое обучение**: Только выходной слой требует обучения, резервуар остаётся фиксированным
- **Вычислительная эффективность**: Разреженные, событийно-управляемые вычисления снижают требования к обработке
- **Биологическая правдоподобность**: Вдохновлены нейронными цепями, обрабатывающими информацию в реальном времени в мозге
- **Устойчивость**: Динамика резервуара обеспечивает естественную регуляризацию против переобучения

## Содержание

1. [Введение в резервуарные вычисления](#введение-в-резервуарные-вычисления)
    * [От традиционных RNN к резервуарам](#от-традиционных-rnn-к-резервуарам)
    * [Парадигма резервуарных вычислений](#парадигма-резервуарных-вычислений)
    * [LSM и Echo State Networks](#lsm-и-echo-state-networks)
2. [Архитектура Liquid State Machine](#архитектура-liquid-state-machine)
    * [Спайковые нейронные сети](#спайковые-нейронные-сети)
    * [Жидкостный резервуар](#жидкостный-резервуар)
    * [Механизмы считывания](#механизмы-считывания)
3. [Математические основы](#математические-основы)
    * [Свойство разделения](#свойство-разделения)
    * [Свойство аппроксимации](#свойство-аппроксимации)
    * [Нейроны с утечкой и интеграцией](#нейроны-с-утечкой-и-интеграцией)
4. [LSM для финансовых временных рядов](#lsm-для-финансовых-временных-рядов)
    * [Кодирование рыночных данных в спайковые последовательности](#кодирование-рыночных-данных-в-спайковые-последовательности)
    * [Предсказание движения цены](#предсказание-движения-цены)
    * [Прогнозирование волатильности](#прогнозирование-волатильности)
5. [Примеры кода](#примеры-кода)
    * [Реализация на Python](#реализация-на-python)
    * [Реализация на Rust](#реализация-на-rust)
6. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
    * [Проектирование стратегии](#проектирование-стратегии)
    * [Метрики производительности](#метрики-производительности)
7. [Продвинутые темы](#продвинутые-темы)
    * [Оптимизация резервуара](#оптимизация-резервуара)
    * [Гибридные архитектуры](#гибридные-архитектуры)
8. [Литература](#литература)

## Введение в резервуарные вычисления

### От традиционных RNN к резервуарам

Традиционные рекуррентные нейронные сети (RNN), такие как LSTM и GRU, достигли замечательных успехов в задачах моделирования последовательностей. Однако они сталкиваются со значительными проблемами:

1. **Сложность обучения**: Обратное распространение ошибки во времени (BPTT) вычислительно затратно
2. **Затухающие/взрывающиеся градиенты**: Сложно захватить долгосрочные зависимости
3. **Чувствительность к гиперпараметрам**: Требуется тщательная настройка для стабильного обучения

Резервуарные вычисления предлагают элегантную альтернативу, разделяя рекуррентную динамику и процесс обучения:

```
Традиционная RNN:        Вход → [Обучаемый рекуррентный слой] → Выход
                                (затратное обучение BPTT)

Резервуарные вычисления: Вход → [Фиксированный резервуар] → [Обучаемый выход]
                                (случайная иниц.)        (простое линейное обучение)
```

### Парадигма резервуарных вычислений

Ключевая идея резервуарных вычислений заключается в том, что случайно инициализированная рекуррентная сеть может служить мощным временным экстрактором признаков без какого-либо обучения. Резервуар преобразует входные последовательности в многомерные траектории состояний, из которых простой линейный выходной слой может извлечь желаемые выходы.

Три основных компонента:
1. **Входной слой**: Кодирует внешние сигналы в резервуар
2. **Резервуар**: Рекуррентная сеть с фиксированными случайными весами
3. **Выходной слой**: Обученная линейная комбинация состояний резервуара

Резервуар должен удовлетворять двум ключевым свойствам:
- **Свойство разделения**: Разные входные последовательности должны производить разные состояния резервуара
- **Свойство аппроксимации**: Похожие входные последовательности должны производить похожие состояния

### LSM и Echo State Networks

Существуют две основные семьи резервуарных вычислений:

| Аспект | Liquid State Machines | Echo State Networks |
|--------|----------------------|---------------------|
| **Модель нейрона** | Спайковая (биологическая) | Частотная (непрерывная) |
| **Представление времени** | Явное (время спайков) | Неявное (эволюция состояния) |
| **Вычисления** | Событийно-управляемые (разреженные) | Непрерывные (плотные) |
| **Биологическая правдоподобность** | Высокая | Средняя |
| **Реализация** | Более сложная | Проще |
| **Применение** | Нейроморфное оборудование | Общего назначения |

Для торговых приложений оба подхода жизнеспособны. LSM превосходят в сценариях, требующих точного временного разрешения и энергоэффективности, в то время как ESN предлагают более простую реализацию и интеграцию с существующими ML-пайплайнами.

## Архитектура Liquid State Machine

### Спайковые нейронные сети

LSM используют спайковые нейронные сети (SNN) в качестве резервуара. В отличие от традиционных искусственных нейронов, которые выдают непрерывные значения, спайковые нейроны общаются через дискретные события, называемые спайками:

```
Традиционный нейрон: выход = активация(сумма(веса * входы))
                     → Непрерывное значение (например, 0.73)

Спайковый нейрон:    выход = последовательность_спайков во времени
                     → Дискретные события: |  |   | |    |
                                           t₁ t₂ t₃t₄  t₅
```

Временной паттерн спайков кодирует информацию:
- **Частотное кодирование**: Информация в частоте генерации (спайков в секунду)
- **Временное кодирование**: Информация в точном времени спайков
- **Популяционное кодирование**: Информация распределена между несколькими нейронами

### Жидкостный резервуар

«Жидкость» в LSM относится к динамическому, постоянно меняющемуся состоянию рекуррентной сети — подобно тому, как волны распространяются по воде:

```
Входной сигнал → [Жидкостный резервуар] → Траектория состояний
   x(t)              (SNNs)                   s(t)

              ┌─────────────────┐
   Вход  →    │  ○──○     ○──○  │   → Состояние
              │   ╲ ╱ ╲   ╱ ╲   │
              │    ○───○───○    │
              │   ╱ ╲   ╲ ╱ ╲   │
              │  ○───○   ○───○  │
              └─────────────────┘
```

Ключевые параметры резервуара:
- **Размер**: Количество нейронов (обычно 100-1000 для трейдинга)
- **Связность**: Вероятность связей между нейронами (10-30%)
- **Спектральный радиус**: Наибольшее собственное значение матрицы весов (контролирует стабильность)
- **Скорость утечки**: Как быстро затухают состояния нейронов
- **Временные константы**: Мембранные и синаптические временные константы

### Механизмы считывания

Выходной слой извлекает релевантную для задачи информацию из состояний резервуара:

```python
# Простой линейный выход
предсказание = W_out @ состояние_резервуара

# Где W_out обучается с помощью ridge-регрессии:
W_out = (S.T @ S + lambda * I)^(-1) @ S.T @ Y
# S: собранные состояния резервуара
# Y: целевые выходы
# lambda: параметр регуляризации
```

Продвинутые подходы к выходному слою:
1. **Усреднение состояний**: Усреднение состояний резервуара по временному окну
2. **Выборка состояний**: Выборка состояний в определённые моменты времени
3. **Спайковый выход**: Использование количества спайков или времени напрямую
4. **Множественные выходы**: Разные выходы для разных задач

## Математические основы

### Свойство разделения

Свойство разделения гарантирует, что разные входные потоки производят различимые состояния резервуара. Для входов u(·) и v(·) и состояний резервуара x^u(t) и x^v(t):

```
d(x^u(t), x^v(t)) ≥ α · d(u, v)
```

Где d(·,·) — метрика расстояния и α > 0 — константа разделения.

В терминах трейдинга, если два рыночных условия различны, резервуар должен производить различимо разные состояния — позволяя выходному слою правильно их классифицировать.

### Свойство аппроксимации

Свойство аппроксимации обеспечивает непрерывность — похожие входы должны производить похожие состояния:

```
d(u, v) < ε  ⟹  d(x^u(t), x^v(t)) < δ(ε)
```

Где δ(ε) → 0 при ε → 0.

Это свойство критически важно для обобщения: если модель видит слегка отличающиеся рыночные условия от тренировочных, она всё равно должна производить разумные предсказания.

### Нейроны с утечкой и интеграцией

Наиболее распространённая модель нейрона в LSM — это нейрон с утечкой и интеграцией (Leaky Integrate-and-Fire, LIF):

```
Динамика мембранного потенциала:
τ_m · dV/dt = -(V - V_rest) + R · I(t)

Где:
- τ_m: мембранная временная константа
- V: мембранный потенциал
- V_rest: потенциал покоя
- R: мембранное сопротивление
- I(t): входной ток

Генерация спайка:
Если V ≥ V_threshold:
    испустить спайк
    V = V_reset
```

Дискретная аппроксимация для реализации:

```python
# Обновление LIF-нейрона
V[t] = V[t-1] + dt/tau_m * (-(V[t-1] - V_rest) + R * I[t])

if V[t] >= V_threshold:
    spike = 1
    V[t] = V_reset
else:
    spike = 0
```

## LSM для финансовых временных рядов

### Кодирование рыночных данных в спайковые последовательности

Финансовые данные должны быть преобразованы в спайковые последовательности для обработки LSM. Распространённые схемы кодирования:

**1. Частотное кодирование**
```python
def rate_encode(value, min_val, max_val, max_rate):
    """Преобразование непрерывного значения в частоту спайков."""
    normalized = (value - min_val) / (max_val - min_val)
    rate = normalized * max_rate
    # Генерация пуассоновской последовательности спайков с этой частотой
    return np.random.poisson(rate * dt)
```

**2. Временное кодирование (время до первого спайка)**
```python
def temporal_encode(value, min_val, max_val, time_window):
    """Большие значения производят более ранние спайки."""
    normalized = (value - min_val) / (max_val - min_val)
    spike_time = time_window * (1 - normalized)
    return spike_time
```

**3. Популяционное кодирование**
```python
def population_encode(value, centers, widths):
    """Распределение значения по популяции нейронов."""
    rates = np.exp(-0.5 * ((value - centers) / widths)**2)
    return rates
```

Для торговых приложений часто лучше работает комбинация:
- **Доходности цены**: Популяционное кодирование с гауссовскими рецептивными полями
- **Объём**: Частотное кодирование (больше объём → выше частота)
- **Технические индикаторы**: Временное кодирование для пороговых значений (RSI перепроданность/перекупленность)

### Предсказание движения цены

LSM для предсказания движения цены в следующем периоде:

```python
# Кодирование признаков
features = [
    encode_returns(returns[-20:]),      # Недавние доходности
    encode_volume(volume[-20:]),        # Профиль объёма
    encode_rsi(rsi),                    # Уровень RSI
    encode_macd(macd, signal),          # Пересечение MACD
]

# Обработка через резервуар
spike_input = concatenate_encodings(features)
for t in range(time_steps):
    reservoir.step(spike_input[t])
    states.append(reservoir.get_state())

# Предсказание выходного слоя
final_state = aggregate_states(states)
prediction = readout.predict(final_state)
# Выход: вероятность движения ВВЕРХ/ВНИЗ
```

### Прогнозирование волатильности

LSM могут захватывать сложную динамику волатильности:

```python
# Определение режима волатильности
# Кодирование реализованной волатильности на нескольких масштабах
vol_5d = encode_volatility(returns.rolling(5).std())
vol_20d = encode_volatility(returns.rolling(20).std())
vol_60d = encode_volatility(returns.rolling(60).std())

# LSM естественно захватывает кластеризацию волатильности
# и переходы между режимами через жидкостную динамику
reservoir_state = lsm.process([vol_5d, vol_20d, vol_60d])

# Предсказание будущего режима волатильности
volatility_forecast = readout.predict(reservoir_state)
```

## Примеры кода

### Реализация на Python

Ноутбук [01_liquid_state_machine_trading.ipynb](python/notebooks/01_liquid_state_machine_trading.ipynb) предоставляет полное руководство.

Основные модули Python:
- `python/lsm_core.py`: Базовая реализация LSM с LIF-нейронами
- `python/encoders.py`: Схемы спайкового кодирования для финансовых данных
- `python/data_loader.py`: Загрузка данных из Yahoo Finance и Bybit
- `python/backtest.py`: Фреймворк бэктестинга для LSM-стратегий

```python
# Пример: LSM для предсказания цены
import numpy as np
from lsm_core import LiquidStateMachine
from encoders import RateEncoder, PopulationEncoder
from data_loader import load_crypto_data

# Загрузка данных BTC/USDT с Bybit
data = load_crypto_data('BTCUSDT', source='bybit')

# Создание кодировщиков
price_encoder = PopulationEncoder(n_neurons=50, min_val=-0.05, max_val=0.05)
volume_encoder = RateEncoder(max_rate=100)

# Инициализация LSM
lsm = LiquidStateMachine(
    n_excitatory=800,
    n_inhibitory=200,
    connectivity=0.1,
    tau_m=20.0,  # мембранная временная константа (мс)
    tau_s=5.0,   # синаптическая временная константа (мс)
)

# Обработка данных и обучение выходного слоя
states = []
for t in range(len(data) - 1):
    # Кодирование текущих признаков
    spike_input = np.concatenate([
        price_encoder.encode(data['returns'].iloc[t]),
        volume_encoder.encode(data['volume'].iloc[t])
    ])

    # Шаг резервуара
    lsm.step(spike_input)
    states.append(lsm.get_state())

# Обучение выходного слоя с ridge-регрессией
states = np.array(states)
targets = (data['returns'].iloc[1:] > 0).astype(int).values
lsm.train_readout(states[:-1], targets[:-1])

# Получение предсказаний
predictions = lsm.predict(states)
```

### Реализация на Rust

Реализация на Rust в `rust/` предоставляет высокопроизводительный LSM для продакшен-трейдинга:

```rust
// rust/src/lib.rs - Базовая реализация LSM

use ndarray::{Array1, Array2};
use rand::Rng;

/// Нейрон с утечкой и интеграцией
pub struct LIFNeuron {
    pub membrane_potential: f64,
    pub threshold: f64,
    pub reset: f64,
    pub tau_m: f64,
    pub refractory_time: f64,
    pub last_spike: f64,
}

impl LIFNeuron {
    pub fn new(tau_m: f64, threshold: f64) -> Self {
        LIFNeuron {
            membrane_potential: 0.0,
            threshold,
            reset: 0.0,
            tau_m,
            refractory_time: 2.0,
            last_spike: f64::NEG_INFINITY,
        }
    }

    pub fn step(&mut self, input_current: f64, dt: f64, t: f64) -> bool {
        // Проверка рефрактерного периода
        if t - self.last_spike < self.refractory_time {
            return false;
        }

        // Интеграция с утечкой
        let dv = dt / self.tau_m * (-self.membrane_potential + input_current);
        self.membrane_potential += dv;

        // Генерация спайка
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset;
            self.last_spike = t;
            return true;
        }

        false
    }
}

/// Резервуар Liquid State Machine
pub struct LSMReservoir {
    neurons: Vec<LIFNeuron>,
    weights: Array2<f64>,
    input_weights: Array2<f64>,
    state: Array1<f64>,
}

impl LSMReservoir {
    pub fn new(n_neurons: usize, n_inputs: usize, connectivity: f64) -> Self {
        let mut rng = rand::thread_rng();

        // Инициализация нейронов
        let neurons: Vec<LIFNeuron> = (0..n_neurons)
            .map(|_| LIFNeuron::new(20.0 + rng.gen::<f64>() * 10.0, 1.0))
            .collect();

        // Инициализация разреженной матрицы весов
        let mut weights = Array2::zeros((n_neurons, n_neurons));
        for i in 0..n_neurons {
            for j in 0..n_neurons {
                if rng.gen::<f64>() < connectivity {
                    weights[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        // Нормализация спектрального радиуса
        let spectral_radius = 0.9;
        // ... код нормализации спектрального радиуса ...

        // Входные веса
        let input_weights = Array2::from_shape_fn(
            (n_neurons, n_inputs),
            |_| rng.gen_range(-0.5..0.5)
        );

        LSMReservoir {
            neurons,
            weights,
            input_weights,
            state: Array1::zeros(n_neurons),
        }
    }

    pub fn step(&mut self, input: &Array1<f64>, dt: f64, t: f64) -> Array1<f64> {
        let n = self.neurons.len();
        let mut spikes = Array1::zeros(n);

        // Вычисление входных токов
        let input_current = self.input_weights.dot(input);
        let recurrent_current = self.weights.dot(&self.state);
        let total_current = &input_current + &recurrent_current;

        // Обновление каждого нейрона
        for i in 0..n {
            if self.neurons[i].step(total_current[i], dt, t) {
                spikes[i] = 1.0;
            }
        }

        self.state = spikes.clone();
        spikes
    }

    pub fn get_state(&self) -> &Array1<f64> {
        &self.state
    }
}
```

Запуск примера на Rust:
```bash
cd rust
cargo run --example lsm_trading --release
```

## Фреймворк бэктестинга

### Проектирование стратегии

Торговая стратегия на основе LSM:

```python
class LSMTradingStrategy:
    def __init__(self, lsm, lookback=20, threshold=0.6):
        self.lsm = lsm
        self.lookback = lookback
        self.threshold = threshold

    def generate_signal(self, market_data):
        # Кодирование недавних рыночных данных
        encoded = self.encode_features(market_data[-self.lookback:])

        # Обработка через LSM
        for spike_input in encoded:
            self.lsm.step(spike_input)

        # Получение предсказания
        state = self.lsm.get_state()
        prob_up = self.lsm.readout.predict(state)

        # Генерация сигнала
        if prob_up > self.threshold:
            return 1  # Лонг
        elif prob_up < (1 - self.threshold):
            return -1  # Шорт
        else:
            return 0  # Нейтрально
```

### Метрики производительности

Ключевые метрики для оценки LSM торговых стратегий:

```python
def evaluate_strategy(returns, positions):
    metrics = {}

    # Доходности стратегии
    strategy_returns = returns * positions.shift(1)

    # Коэффициент Шарпа (годовой)
    metrics['sharpe'] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    # Коэффициент Сортино
    downside_std = strategy_returns[strategy_returns < 0].std()
    metrics['sortino'] = strategy_returns.mean() / downside_std * np.sqrt(252)

    # Максимальная просадка
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()

    # Доля выигрышных сделок
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (positions.diff() != 0).sum()
    metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0

    return metrics
```

## Продвинутые темы

### Оптимизация резервуара

Хотя резервуар обычно случайный и фиксированный, несколько подходов к оптимизации могут улучшить производительность:

**1. Внутренняя пластичность**
Адаптация параметров отдельных нейронов для максимизации передачи информации:

```python
def intrinsic_plasticity_update(neuron, target_rate=0.1, learning_rate=0.001):
    """Корректировка порога нейрона для достижения целевой частоты генерации."""
    actual_rate = neuron.spike_count / neuron.time_window
    error = target_rate - actual_rate
    neuron.threshold -= learning_rate * error
```

**2. Оптимизация топологии**
Проектирование связности резервуара на основе знаний финансовой предметной области:

```python
def create_financial_topology(n_neurons):
    """Создание резервуара со структурой, отражающей рыночные связи."""
    # Группировка нейронов по классам активов
    equity_neurons = n_neurons // 3
    bond_neurons = n_neurons // 3
    commodity_neurons = n_neurons - equity_neurons - bond_neurons

    # Сильные внутриклассовые, слабые межклассовые связи
    weights = sparse_block_diagonal(
        [create_block(equity_neurons, 0.3),
         create_block(bond_neurons, 0.3),
         create_block(commodity_neurons, 0.3)]
    )
    # Добавление межклассовых связей
    add_sparse_connections(weights, density=0.05)
    return weights
```

**3. Эволюционная оптимизация**
Эволюция параметров резервуара для конкретных торговых задач:

```python
def evolve_reservoir_params(fitness_function, generations=100):
    """Использование генетического алгоритма для оптимизации параметров резервуара."""
    population = initialize_population()

    for gen in range(generations):
        fitness = [fitness_function(params) for params in population]
        population = select_and_mutate(population, fitness)

    return best_individual(population)
```

### Гибридные архитектуры

Комбинирование LSM с другими подходами:

**1. LSM + Внимание**
Использование механизма внимания для взвешивания состояний резервуара:

```python
class LSMWithAttention:
    def __init__(self, lsm, attention_dim=64):
        self.lsm = lsm
        self.query = nn.Linear(lsm.n_neurons, attention_dim)
        self.key = nn.Linear(lsm.n_neurons, attention_dim)
        self.value = nn.Linear(lsm.n_neurons, attention_dim)

    def forward(self, states):
        # states: (T, N) - временные шаги x нейроны
        Q = self.query(states)
        K = self.key(states)
        V = self.value(states)

        attention = softmax(Q @ K.T / sqrt(attention_dim))
        output = attention @ V
        return output
```

**2. Ансамбль из нескольких резервуаров**
Комбинирование нескольких LSM с разными параметрами:

```python
class EnsembleLSM:
    def __init__(self, n_reservoirs=5, **params):
        self.reservoirs = [
            LiquidStateMachine(**varied_params(params, i))
            for i in range(n_reservoirs)
        ]

    def predict(self, input_sequence):
        predictions = []
        for reservoir in self.reservoirs:
            state = reservoir.process(input_sequence)
            pred = reservoir.readout.predict(state)
            predictions.append(pred)

        # Ансамблевое усреднение
        return np.mean(predictions)
```

## Литература

1. **Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations**
   - Авторы: Wolfgang Maass, Thomas Natschläger, Henry Markram
   - URL: https://www.mitpressjournals.org/doi/10.1162/089976602760407955
   - Год: 2002
   - Основополагающая статья LSM, представляющая парадигму жидкостных вычислений

2. **Liquid State Machines: Motivation, Theory, and Applications**
   - Авторы: Hesham Mostafa и др.
   - URL: https://arxiv.org/abs/2008.00925
   - Год: 2020
   - Полный обзор теории и применений LSM

3. **Echo State Networks: A Brief Tutorial**
   - Автор: Herbert Jaeger
   - URL: https://www.ai.rug.nl/minds/uploads/ESNTutorialRev.pdf
   - Год: 2007
   - Туториал по связанному подходу резервуарных вычислений

4. **Reservoir Computing Approaches to Recurrent Neural Network Training**
   - Авторы: Mantas Lukoševičius, Herbert Jaeger
   - URL: https://www.sciencedirect.com/science/article/pii/S1574013709000173
   - Год: 2009
   - Обзор методов резервуарных вычислений

5. **Spiking Neural Networks for Financial Time Series Prediction**
   - Авторы: Различные
   - URL: https://arxiv.org/search/?query=spiking+neural+networks+financial
   - Современные исследования SNN для финансов

## Источники данных

- **Yahoo Finance / yfinance**: Исторические цены акций, индексы и ETF
- **Bybit API**: Данные криптовалютного рынка (OHLCV, стакан заявок, сделки)
- **Binance API**: Альтернативный источник крипто-данных
- **LOBSTER**: Высокочастотные данные книги лимитных заявок
- **Kaggle**: Различные финансовые датасеты для экспериментов

## Библиотеки и инструменты

### Python
- `numpy`, `scipy`: Численные вычисления
- `brian2`: Симулятор спайковых нейронных сетей
- `norse`: SNN-библиотека на основе PyTorch
- `reservoirpy`: Библиотека резервуарных вычислений
- `pandas`: Обработка данных
- `yfinance`: API данных Yahoo Finance
- `ccxt`: API криптовалютных бирж
- `backtrader`: Фреймворк бэктестинга

### Rust
- `ndarray`: N-мерные массивы
- `polars`: Быстрые DataFrames
- `rand`: Генерация случайных чисел
- `rayon`: Параллельная обработка
- `reqwest`: HTTP-клиент для API-запросов
- `serde`: Сериализация/десериализация
