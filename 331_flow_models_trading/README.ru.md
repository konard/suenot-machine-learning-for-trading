# Глава 331: Торговля с использованием Flow-моделей

## Обзор

Flow-модели (нормализующие потоки) представляют собой мощный класс генеративных моделей, которые изучают **обратимые преобразования** между сложными распределениями данных и простыми базовыми распределениями (обычно гауссовскими). В отличие от других генеративных моделей (VAE, GAN), flow-модели обеспечивают **точное вычисление правдоподобия** и **идеальную реконструкцию**, что делает их идеальными для финансовых приложений, где точная оценка плотности и обнаружение аномалий критически важны.

## Почему Flow-модели для торговли?

### Проблемы традиционных подходов

Традиционные модели для прогнозирования рынка сталкиваются с трудностями:

- **Предположения о распределении**: Рынки не следуют гауссовскому распределению
- **Смена режимов**: Внезапные изменения рыночной динамики
- **Обнаружение аномалий**: Выявление необычных рыночных условий
- **Количественная оценка неопределенности**: Понимание уверенности прогноза

### Решение с помощью Flow-моделей

Flow-модели решают эти проблемы:

```
Традиционный подход: Предположить p(x) ~ Гауссиан → Подогнать параметры → Предсказать

Flow-модель: Изучить точное преобразование f: x → z
где:
  x = сложное распределение рыночных данных
  z = простое гауссовское распределение
  f обратима (можно идти в обе стороны)
  p(x) = p(z) |det(df/dx)|  ← ТОЧНОЕ правдоподобие!
```

## Техническая архитектура

### 1. Основные концепции Flow-моделей

```
Цепочка Flow-преобразований:
x ↔ h₁ ↔ h₂ ↔ ... ↔ hₙ ↔ z

где:
├── x = наблюдаемые данные (рыночные признаки)
├── z = латентное пространство (гауссовское)
├── Каждый шаг ОБРАТИМ
└── Определитель Якобиана вычислим

Ключевые свойства:
├── Точное правдоподобие: log p(x) = log p(z) + Σ log|det(∂hᵢ/∂hᵢ₋₁)|
├── Идеальная реконструкция: x = f⁻¹(f(x))
├── Эффективная генерация: z ~ N(0,I) → x = f⁻¹(z)
└── Обнаружение аномалий: Низкое p(x) = необычное состояние рынка
```

### 2. Популярные архитектуры Flow

#### NICE (Non-linear Independent Components Estimation)

```python
# Аддитивный coupling-слой
def nice_forward(x, mask):
    x1, x2 = x * mask, x * (1 - mask)
    y1 = x1
    y2 = x2 + neural_net(x1)  # Аддитивное преобразование
    return y1 + y2

# Обратное преобразование тривиально!
def nice_inverse(y, mask):
    y1, y2 = y * mask, y * (1 - mask)
    x1 = y1
    x2 = y2 - neural_net(y1)  # Просто вычитаем
    return x1 + x2
```

#### RealNVP (Real-valued Non-Volume Preserving)

```python
# Аффинный coupling-слой
def realnvp_forward(x, mask):
    x1, x2 = x * mask, x * (1 - mask)
    s, t = scale_translate_net(x1)  # Выход: масштаб и сдвиг
    y1 = x1
    y2 = x2 * exp(s) + t  # Аффинное преобразование
    log_det = sum(s)  # Лог-детерминант - просто сумма масштабов
    return y1 + y2, log_det
```

#### Glow (Generative Flow)

```
Glow-блок:
├── ActNorm: Обучаемая нормализация активаций
├── 1x1 Свертка: Обучаемая перестановка
└── Аффинный Coupling: Преобразование в стиле RealNVP

Многомасштабная архитектура:
Уровень 1: [Flow-блок × K] → Разделение
Уровень 2: [Flow-блок × K] → Разделение
Уровень L: [Flow-блок × K] → Финальное z
```

### 3. Непрерывные нормализующие потоки (CNF)

```
Формулировка нейронного ODE:
dz/dt = f(z(t), t; θ)

Ключевые преимущества:
├── Произвольная архитектура (нет ограничений обратимости)
├── Эффективное по памяти обучение (метод сопряженных)
└── Гладкие преобразования

Flow Matching (современный подход):
├── Более простая функция потерь
├── Лучшая стабильность
└── Быстрая сходимость
```

## Архитектура модели для торговли

```
┌─────────────────────────────────────────────────────────────────┐
│              СИСТЕМА ТОРГОВЛИ НА FLOW-МОДЕЛЯХ                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ВХОДНОЙ СЛОЙ                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Рыночные признаки (на каждый момент времени):            │   │
│  │   - Дисбаланс потока заявок (OFI)                        │   │
│  │   - Профиль объема (объемы bid/ask)                      │   │
│  │   - Доходность (разные таймфреймы)                       │   │
│  │   - Микроструктурные признаки (спред, глубина)           │   │
│  │   - Технические индикаторы (RSI, MACD, Bollinger)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ЭНКОДЕР (Опциональное кондиционирование)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Временное кодирование для рыночного контекста            │   │
│  │ Переменные кондиционирования режима                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  БЛОКИ FLOW-ПРЕОБРАЗОВАНИЯ (×N)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Нормализация активаций (ActNorm)                    │   │   │
│  │ │   - Инициализация на основе данных                  │   │   │
│  │ │   - Обучаемые масштаб и сдвиг                       │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Слой перестановки                                   │   │   │
│  │ │   - 1x1 Свертка (обучаемая)                        │   │   │
│  │ │   - или Фиксированная перестановка                  │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Аффинный Coupling-слой                              │   │   │
│  │ │   - Разделение входа: [x₁, x₂]                     │   │   │
│  │ │   - Преобразование: y₂ = x₂ * exp(s(x₁)) + t(x₁)  │   │   │
│  │ │   - Объединение: [x₁, y₂]                          │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ЛАТЕНТНОЕ ПРОСТРАНСТВО                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ z ~ N(0, I) - Гауссовское латентное пространство         │   │
│  │   - Определение режима через кластеризацию               │   │
│  │   - Обнаружение аномалий через правдоподобие             │   │
│  │   - Оценка плотности для риска                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ВЫХОДНЫЕ ГОЛОВЫ                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Оценка правдоподобия: log p(x) для обнаружения аномалий  │   │
│  │ Латентный режим: кластеризация z для состояния рынка     │   │
│  │ Условная генерация: Семплирование будущих сценариев      │   │
│  │ Прогноз потока заявок: Направление/величина след. тика   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Торговые приложения

### 1. Прогнозирование потока заявок

```python
class OrderFlowPredictor:
    """Прогноз потока заявок с использованием условной flow-модели"""

    def __init__(self, flow_model, context_encoder):
        self.flow = flow_model
        self.encoder = context_encoder

    def predict(self, market_context, num_samples=1000):
        # Кодирование рыночного контекста
        context = self.encoder(market_context)

        # Семплирование из латентного пространства
        z = torch.randn(num_samples, self.flow.latent_dim)

        # Генерация прогнозов потока заявок
        predictions = self.flow.inverse(z, context)

        # Вычисление статистик
        mean_flow = predictions.mean(dim=0)
        std_flow = predictions.std(dim=0)

        return {
            'expected_flow': mean_flow,
            'uncertainty': std_flow,
            'samples': predictions
        }
```

### 2. Моделирование рыночной микроструктуры

```python
class MicrostructureFlow:
    """Моделирование динамики стакана с помощью нормализующих потоков"""

    def compute_likelihood(self, order_book_state):
        """Вычисление лог-правдоподобия конфигурации стакана"""
        z, log_det = self.flow.forward(order_book_state)
        log_pz = self.base_dist.log_prob(z).sum(dim=-1)
        log_px = log_pz + log_det
        return log_px

    def detect_anomaly(self, order_book_state, threshold=-10.0):
        """Обнаружение необычных конфигураций стакана"""
        log_px = self.compute_likelihood(order_book_state)
        return log_px < threshold

    def simulate_book_evolution(self, initial_state, steps=100):
        """Симуляция будущих состояний стакана"""
        states = [initial_state]
        for _ in range(steps):
            # Кодирование текущего состояния в латентное
            z, _ = self.flow.forward(states[-1])

            # Добавление небольшого шума для эволюции
            z_next = z + 0.01 * torch.randn_like(z)

            # Декодирование в следующее состояние
            next_state = self.flow.inverse(z_next)
            states.append(next_state)

        return torch.stack(states)
```

### 3. Определение режимов в латентном пространстве

```python
class RegimeDetector:
    """Определение рыночных режимов в латентном пространстве flow-модели"""

    def __init__(self, flow_model, n_regimes=4):
        self.flow = flow_model
        self.n_regimes = n_regimes
        self.clusterer = GaussianMixture(n_components=n_regimes)

    def fit_regimes(self, historical_data):
        """Обучение кластеров режимов на латентных представлениях"""
        z_latent, _ = self.flow.forward(historical_data)
        self.clusterer.fit(z_latent.detach().numpy())

        # Маркировка режимов на основе характеристик
        self.regime_labels = self._analyze_regimes(historical_data, z_latent)

    def detect_current_regime(self, current_data):
        """Определение текущего рыночного режима"""
        z, _ = self.flow.forward(current_data)
        regime = self.clusterer.predict(z.detach().numpy())
        probs = self.clusterer.predict_proba(z.detach().numpy())

        return {
            'regime': regime[0],
            'label': self.regime_labels[regime[0]],
            'confidence': probs.max(),
            'regime_probs': dict(zip(self.regime_labels, probs[0]))
        }

    def _analyze_regimes(self, data, z_latent):
        """Анализ характеристик режимов"""
        labels = self.clusterer.predict(z_latent.detach().numpy())
        regime_labels = []

        for i in range(self.n_regimes):
            mask = labels == i
            regime_data = data[mask]

            volatility = regime_data.std()
            trend = regime_data.mean()

            if volatility > 0.02 and trend > 0:
                regime_labels.append("Высокая вол. Бычий")
            elif volatility > 0.02 and trend < 0:
                regime_labels.append("Высокая вол. Медвежий")
            elif volatility <= 0.02 and trend > 0:
                regime_labels.append("Низкая вол. Бычий")
            else:
                regime_labels.append("Низкая вол. Медвежий")

        return regime_labels
```

### 4. Flow Matching для торговли

```python
class FlowMatchingTrader:
    """Современный подход flow matching для торговых сигналов"""

    def __init__(self, vector_field_net):
        self.v_net = vector_field_net  # Нейросеть для векторного поля

    def flow_matching_loss(self, x0, x1):
        """
        Функция потерь flow matching
        x0: шумовые семплы (базовое распределение)
        x1: семплы данных (рыночные признаки)
        """
        # Случайное время
        t = torch.rand(x0.shape[0], 1)

        # Интерполяция между шумом и данными
        xt = (1 - t) * x0 + t * x1

        # Целевая скорость (оптимальный транспорт)
        ut = x1 - x0

        # Предсказанная скорость
        vt = self.v_net(xt, t)

        # MSE-потери
        loss = ((vt - ut) ** 2).mean()
        return loss

    def sample(self, num_samples, steps=100):
        """Генерация семплов через интегрирование ODE"""
        # Начинаем с шума
        x = torch.randn(num_samples, self.dim)

        # Интегрируем ODE
        dt = 1.0 / steps
        for t in torch.linspace(0, 1, steps):
            v = self.v_net(x, t.expand(num_samples, 1))
            x = x + v * dt

        return x
```

## Торговая стратегия

### Генерация сигналов

```python
class FlowTradingStrategy:
    def __init__(self, flow_model, regime_detector):
        self.flow = flow_model
        self.regime_detector = regime_detector
        self.anomaly_threshold = -15.0

    def generate_signal(self, market_data):
        """Генерация торгового сигнала с использованием flow-модели"""

        # 1. Вычисление правдоподобия
        log_likelihood = self.flow.log_prob(market_data)

        # 2. Определение режима
        regime_info = self.regime_detector.detect_current_regime(market_data)

        # 3. Проверка на аномалию
        is_anomaly = log_likelihood < self.anomaly_threshold

        # 4. Генерация сигнала на основе режима и условий
        if is_anomaly:
            return Signal("СНИЗИТЬ_ЭКСПОЗИЦИЮ", confidence=0.9,
                         reason="Обнаружено аномальное состояние рынка")

        regime = regime_info['label']
        confidence = regime_info['confidence']

        if regime == "Высокая вол. Бычий" and confidence > 0.7:
            return Signal("LONG", confidence=confidence * 0.8,
                         reason=f"Режим высокой волатильности бычий")
        elif regime == "Высокая вол. Медвежий" and confidence > 0.7:
            return Signal("SHORT", confidence=confidence * 0.8,
                         reason=f"Режим высокой волатильности медвежий")
        elif regime in ["Низкая вол. Бычий", "Низкая вол. Медвежий"]:
            return Signal("НЕЙТРАЛЬНО", confidence=confidence * 0.5,
                         reason=f"Режим низкой волатильности - меньше возможностей")

        return Signal("УДЕРЖИВАТЬ", confidence=0.5, reason="Неопределенный режим")
```

### Управление рисками

```python
class FlowRiskManager:
    """Управление рисками с использованием оценок плотности flow-модели"""

    def __init__(self, flow_model):
        self.flow = flow_model

    def compute_var(self, portfolio, confidence=0.95, num_samples=10000):
        """Вычисление Value-at-Risk с использованием flow-модели"""
        # Семплирование из flow-модели
        samples = self.flow.sample(num_samples)

        # Вычисление доходности портфеля для каждого семпла
        portfolio_returns = (samples * portfolio.weights).sum(dim=-1)

        # VaR при заданной доверительности
        var = torch.quantile(portfolio_returns, 1 - confidence)

        return var.item()

    def stress_test(self, portfolio, scenario_likelihood_threshold=-20.0):
        """Генерация стресс-сценариев из областей низкого правдоподобия"""
        # Поиск областей низкого правдоподобия в латентном пространстве
        z_extreme = torch.randn(1000, self.flow.latent_dim) * 3  # Далеко от среднего

        # Преобразование в пространство данных
        extreme_scenarios = self.flow.inverse(z_extreme)

        # Вычисление правдоподобий
        log_probs = self.flow.log_prob(extreme_scenarios)

        # Выбор экстремальных, но правдоподобных сценариев
        mask = log_probs > scenario_likelihood_threshold
        stress_scenarios = extreme_scenarios[mask]

        # Вычисление влияния на портфель
        impacts = []
        for scenario in stress_scenarios:
            impact = (scenario * portfolio.weights).sum()
            impacts.append(impact.item())

        return {
            'scenarios': stress_scenarios,
            'impacts': impacts,
            'worst_case': min(impacts),
            'expected_shortfall': np.mean(sorted(impacts)[:int(len(impacts)*0.05)])
        }
```

## Ключевые компоненты

### 1. Аффинный Coupling-слой

```python
class AffineCoupling(nn.Module):
    """Аффинный coupling-слой для RealNVP/Glow"""

    def __init__(self, dim, hidden_dim=256, mask_type='checkerboard'):
        super().__init__()
        self.dim = dim
        self.mask = self._create_mask(dim, mask_type)

        # Сети масштабирования и сдвига
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()  # Ограниченный масштаб для стабильности
        )

        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self, x):
        """Прямой проход: x -> z"""
        x1, x2 = x[:, :self.dim//2], x[:, self.dim//2:]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        log_det = s.sum(dim=-1)

        return torch.cat([y1, y2], dim=-1), log_det

    def inverse(self, y):
        """Обратный проход: z -> x"""
        y1, y2 = y[:, :self.dim//2], y[:, self.dim//2:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=-1)
```

### 2. ActNorm (Нормализация активаций)

```python
class ActNorm(nn.Module):
    """Нормализация активаций с инициализацией на основе данных"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x):
        """Инициализация на основе данных"""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)

            self.bias.data = -mean
            self.scale.data = 1.0 / (std + 1e-6)
            self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        y = (x + self.bias) * self.scale
        log_det = torch.log(torch.abs(self.scale)).sum() * x.shape[0]

        return y, log_det

    def inverse(self, y):
        x = y / self.scale - self.bias
        return x
```

### 3. Полная Flow-модель

```python
class NormalizingFlow(nn.Module):
    """Полная модель нормализующего потока"""

    def __init__(self, dim, num_layers=8, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(ActNorm(dim))
            self.layers.append(AffineCoupling(dim, hidden_dim))
            if i < num_layers - 1:
                self.layers.append(Permutation(dim))

        self.base_dist = torch.distributions.Normal(
            torch.zeros(dim), torch.ones(dim)
        )

    def forward(self, x):
        """Преобразование данных в латентное пространство"""
        log_det_total = 0
        z = x

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z):
        """Преобразование латентного в пространство данных"""
        x = z

        for layer in reversed(self.layers):
            x = layer.inverse(x)

        return x

    def log_prob(self, x):
        """Вычисление лог-правдоподобия данных"""
        z, log_det = self.forward(x)
        log_pz = self.base_dist.log_prob(z).sum(dim=-1)
        log_px = log_pz + log_det
        return log_px

    def sample(self, num_samples):
        """Генерация семплов из модели"""
        z = self.base_dist.sample((num_samples,))
        x = self.inverse(z)
        return x
```

## Детали реализации

### Требования к данным

```
Рыночные данные для Flow-моделей:
├── Высокочастотные данные (предпочтительно тиковые)
│   └── Поток заявок, сделки, котировки
├── Снимки стакана заявок
│   └── Многоуровневые bid/ask с объемами
├── Данные об объемах
│   └── Декомпозиция покупок/продаж
└── Производные признаки
    ├── Дисбаланс потока заявок (OFI)
    ├── Отклонение цены от VWAP
    ├── Динамика спреда
    └── Дисбаланс глубины

Инженерия признаков:
├── Временные признаки
│   ├── Доходность на разных масштабах (1с, 10с, 1м, 5м)
│   └── Оценки волатильности
├── Микроструктурные признаки
│   ├── Bid-ask спред (б.п.)
│   ├── Дисбаланс глубины (L1-L5)
│   └── Частота поступления заявок
└── Производные сигналы
    ├── VPIN (Синхронизированный по объему PIN)
    └── Оценки лямбда Кайла
```

### Конфигурация обучения

```yaml
model:
  type: "realnvp"  # или "glow", "continuous_flow"
  input_dim: 32
  num_flow_layers: 8
  hidden_dim: 256
  activation: "relu"
  use_actnorm: true
  permutation: "learnable_1x1"

training:
  batch_size: 256
  learning_rate: 0.0001
  weight_decay: 0.00001
  max_epochs: 200
  gradient_clip: 1.0
  warmup_steps: 1000

regularization:
  spectral_norm: true
  weight_decay: 0.00001

data:
  sequence_length: 100
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  normalize: "standard"  # или "minmax", "robust"
```

## Ключевые метрики

### Производительность модели

- **Отрицательное лог-правдоподобие (NLL)**: Основная функция потерь (меньше - лучше)
- **Биты на измерение**: Нормализованный NLL для сравнения между размерностями
- **Ошибка реконструкции**: Должна быть ~0 для обратимых потоков
- **Качество семплов**: Визуальная и статистическая оценка

### Торговая производительность

- **Коэффициент Шарпа**: Доходность с поправкой на риск (цель > 2.0)
- **Коэффициент Сортино**: Доходность с поправкой на риск падения
- **Максимальная просадка**: Наибольшее падение от пика
- **Точность обнаружения аномалий**: Для необычных рыночных событий
- **Точность определения режимов**: Правильная идентификация состояний рынка

## Преимущества Flow-моделей

| Аспект | Традиционные модели | Flow-модели |
|--------|---------------------|-------------|
| Правдоподобие | Приближенное (VAE) или нет (GAN) | Точное вычисление |
| Реконструкция | С потерями | Идеальная (обратимая) |
| Обнаружение аномалий | Порог по признакам | Принципиальная оценка плотности |
| Неопределенность | Часто отсутствует | Естественная из плотности |
| Интерпретируемость | Черный ящик | Структура латентного пространства |
| Качество семплов | Коллапс мод (GAN) | Стабильное обучение |

## Сравнение с другими подходами

### vs. VAE

- **VAE**: Приближенный постериор, обучение ELBO, потери реконструкции
- **Flow**: Точное правдоподобие, идеальная реконструкция, нет отдельного энкодера

### vs. GAN

- **GAN**: Нет плотности, коллапс мод, состязательное обучение
- **Flow**: Точная плотность, стабильное обучение, не нужен дискриминатор

### vs. Диффузионные модели

- **Диффузия**: Медленное семплирование, нет точного правдоподобия, сильная генерация
- **Flow**: Быстрое семплирование, точное правдоподобие, более простая архитектура

## Соображения для продакшена

```
Пайплайн инференса:
├── Сбор данных (Bybit WebSocket)
│   └── Стакан + сделки в реальном времени
├── Вычисление признаков
│   └── Поток заявок, микроструктурные признаки
├── Инференс Flow-модели
│   ├── Вычисление правдоподобия (обнаружение аномалий)
│   ├── Извлечение латентного представления (режим)
│   └── Генерация семплов (анализ сценариев)
├── Генерация сигналов
│   └── Комбинация режима + аномалии + прогноза
└── Исполнение ордеров
    └── Размер позиции с учетом риска

Бюджет задержки:
├── Сбор данных: ~5мс (WebSocket)
├── Вычисление признаков: ~2мс
├── Прямой проход flow: ~5мс (GPU)
├── Определение режима: ~1мс
├── Генерация сигнала: ~1мс
└── Всего: ~15мс (без исполнения)
```

## Структура директории

```
331_flow_models_trading/
├── README.md                    # Английская версия
├── README.ru.md                 # Этот файл (русская версия)
├── readme.simple.md             # Объяснение для начинающих (англ.)
├── readme.simple.ru.md          # Объяснение для начинающих (рус.)
├── python/                      # Python реализация
│   ├── requirements.txt        # Зависимости Python
│   ├── data_fetcher.py         # Данные Bybit через CCXT
│   ├── flow_model.py           # Основная flow-модель (NormalizingFlow, ActNorm и др.)
│   ├── trading_strategy.py     # Генерация сигналов и стратегия
│   └── backtest.py             # Комплексный фреймворк бэктестинга
└── rust_flow_models/           # Rust реализация
    ├── Cargo.toml
    ├── README.md               # Документация для Rust
    ├── src/
    │   ├── lib.rs              # Точка входа библиотеки
    │   ├── api/                # Клиент API Bybit
    │   │   ├── mod.rs
    │   │   ├── client.rs       # REST API клиент
    │   │   └── types.rs        # Типы данных
    │   ├── flow/               # Реализация flow-модели
    │   │   ├── mod.rs
    │   │   ├── config.rs       # Конфигурация модели
    │   │   ├── layers.rs       # Flow-слои (ActNorm, Coupling)
    │   │   ├── model.rs        # Модель NormalizingFlow
    │   │   ├── anomaly.rs      # Обнаружение аномалий
    │   │   └── regime.rs       # Определение режимов
    │   ├── features/           # Инженерия признаков
    │   │   ├── mod.rs
    │   │   ├── engine.rs       # Вычисление признаков
    │   │   └── indicators.rs   # Технические индикаторы
    │   ├── strategy/           # Торговая стратегия
    │   │   ├── mod.rs
    │   │   ├── signal.rs       # Типы сигналов
    │   │   └── flow_strategy.rs # Flow-стратегия
    │   └── backtest/           # Движок бэктестинга
    │       ├── mod.rs
    │       ├── engine.rs       # Выполнение бэктеста
    │       └── report.rs       # Отчеты о производительности
    └── examples/
        ├── fetch_market_data.rs  # Пример получения данных
        ├── train_flow_model.rs   # Пример обучения модели
        ├── anomaly_detection.rs  # Пример обнаружения аномалий
        ├── regime_detection.rs   # Пример определения режимов
        ├── backtest.rs           # Пример бэктестинга
        └── live_signals.rs       # Генерация сигналов в реальном времени
```

## Ссылки

1. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2014)
   - https://arxiv.org/abs/1410.8516

2. **Density estimation using Real-NVP** (Dinh et al., 2016)
   - https://arxiv.org/abs/1605.08803

3. **Glow: Generative Flow with Invertible 1x1 Convolutions** (Kingma & Dhariwal, 2018)
   - https://arxiv.org/abs/1807.03039

4. **Neural Ordinary Differential Equations** (Chen et al., 2018)
   - https://arxiv.org/abs/1806.07366

5. **Flow Matching for Generative Modeling** (Lipman et al., 2022)
   - https://arxiv.org/abs/2210.02747

6. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2021)
   - https://arxiv.org/abs/1912.02762

7. **Применение нормализующих потоков в финансах** (различные авторы)
   - Моделирование рыночной микроструктуры
   - Ценообразование опционов со сложными распределениями

## Уровень сложности

**Эксперт** - Требуется понимание:
- Теории вероятностей и оценки плотности
- Формулы замены переменных
- Архитектур нейронных сетей
- Рыночной микроструктуры
- Концепций высокочастотной торговли

## Отказ от ответственности

Эта глава предназначена **только для образовательных целей**. Торговля криптовалютами сопряжена со значительными рисками. Описанные здесь стратегии не были проверены в реальной торговле и должны быть тщательно протестированы перед любым реальным применением. Прошлые результаты не гарантируют будущих доходов.
