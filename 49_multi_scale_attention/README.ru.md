# Глава 49: Многомасштабное внимание для финансовых временных рядов

Эта глава исследует механизмы **многомасштабного внимания** (Multi-Scale Attention) для прогнозирования финансовых временных рядов. В отличие от одномасштабных подходов, обрабатывающих данные с одним временным разрешением, многомасштабное внимание захватывает паттерны на разных временных горизонтах одновременно — от краткосрочных тиковых колебаний до долгосрочной динамики трендов.

## Содержание

1. [Введение в многомасштабное внимание](#введение-в-многомасштабное-внимание)
    * [Почему несколько временных масштабов?](#почему-несколько-временных-масштабов)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с одномасштабными моделями](#сравнение-с-одномасштабными-моделями)
2. [Архитектура многомасштабного внимания](#архитектура-многомасштабного-внимания)
    * [Энкодеры для каждого масштаба](#энкодеры-для-каждого-масштаба)
    * [Многоуровневое внимание](#многоуровневое-внимание)
    * [Кросс-масштабное слияние](#кросс-масштабное-слияние)
    * [Иерархическая агрегация](#иерархическая-агрегация)
3. [Декомпозиция временных масштабов](#декомпозиция-временных-масштабов)
    * [Временное понижение частоты](#временное-понижение-частоты)
    * [Вейвлет-декомпозиция](#вейвлет-декомпозиция)
    * [Вариационная модовая декомпозиция (VMD)](#вариационная-модовая-декомпозиция-vmd)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных с многомасштабными признаками](#01-подготовка-данных-с-многомасштабными-признаками)
    * [02: Архитектура многомасштабного внимания](#02-архитектура-многомасштабного-внимания)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Многогоризонтное прогнозирование](#04-многогоризонтное-прогнозирование)
    * [05: Бэктестинг стратегии](#05-бэктестинг-стратегии)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в многомасштабное внимание

Финансовые рынки демонстрируют паттерны на множестве временных масштабов. Минутный трейдер видит шум там, где дневной трейдер видит тренды, а дневной трейдер видит шум там, где свинг-трейдер видит циклы. **Многомасштабное внимание** явно моделирует эти различные временные разрешения, позволяя единой модели понимать как краткосрочный моментум, так и долгосрочные тренды.

### Почему несколько временных масштабов?

Традиционные модели обрабатывают временные ряды с одним разрешением:

```
Ценовые данные (1-мин) → Модель → Прогноз

Проблемы:
- Короткое окно: Захватывает шум, пропускает тренды
- Длинное окно: Захватывает тренды, теряет мелкие детали
- Фиксированный масштаб: Не адаптируется к меняющимся рыночным условиям
```

Многомасштабное внимание обрабатывает несколько разрешений одновременно:

```
Данные (1-мин)   ]
Данные (5-мин)   ]  → Многомасштабное  → Прогноз
Данные (1-час)   ]    внимание          (понимая все паттерны!)
Данные (1-день)  ]

Преимущества:
- Короткие масштабы: Захватывают тайминг входа/выхода
- Средние масштабы: Захватывают внутридневные паттерны
- Длинные масштабы: Захватывают направление тренда
- Слияние: Умно комбинирует все insights
```

### Ключевые преимущества

1. **Иерархическое обнаружение паттернов**
   - Минутный уровень: Поток ордеров, эффекты микроструктуры
   - Часовой уровень: Сессионные паттерны, профили объёма
   - Дневной уровень: Следование тренду, циклы возврата к среднему
   - Недельный уровень: Макрорежимные изменения

2. **Адаптивный фокус**
   - Обучение тому, какие масштабы важны для каждого прогноза
   - Веса внимания показывают важность масштабов
   - Динамическое перевзвешивание на основе рыночных условий

3. **Устойчивые прогнозы**
   - Краткосрочный шум не доминирует
   - Долгосрочные тренды дают контекст
   - Кросс-масштабная валидация уменьшает ложные сигналы

4. **Многогоризонтное прогнозирование**
   - Единая модель для множества горизонтов прогноза
   - Согласованные сигналы на разных таймфреймах
   - Унифицированное управление рисками

### Сравнение с одномасштабными моделями

| Характеристика | LSTM | Transformer | TFT | Многомасштабное внимание |
|---------------|------|-------------|-----|--------------------------|
| Мультиразрешение | Нет | Нет | Ограничено | Полное |
| Масштаб-зависимое внимание | Нет | Нет | Нет | Да |
| Кросс-масштабное слияние | Нет | Нет | Нет | Да |
| Адаптивные веса масштабов | Нет | Нет | Частично | Да |
| Длинные последовательности | Плохо | O(L²) | O(L) | O(L) на масштаб |
| Интерпретируемость | Низкая | Средняя | Высокая | Очень высокая |

## Архитектура многомасштабного внимания

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       СЕТЬ МНОГОМАСШТАБНОГО ВНИМАНИЯ                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Исходный временной ряд                                                       │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    СЛОЙ ДЕКОМПОЗИЦИИ МАСШТАБОВ                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │ │
│  │  │Масштаб 1 │  │Масштаб 2 │  │Масштаб 3 │  │Масштаб 4 │                │ │
│  │  │ (1-мин)  │  │ (5-мин)  │  │ (1-час)  │  │ (1-день) │                │ │
│  │  │ L=1440   │  │ L=288    │  │ L=24     │  │ L=30     │                │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │ │
│  └───────│─────────────│─────────────│─────────────│───────────────────────┘ │
│          │             │             │             │                          │
│          ▼             ▼             ▼             ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Энкодер  │  │ Энкодер  │  │ Энкодер  │  │ Энкодер  │                      │
│  │(короткий)│  │(средний) │  │ (длинный)│  │ (тренд)  │                      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
│       │             │             │             │                             │
│       └──────────┬──┴──────────┬──┴──────────┬──┘                            │
│                  │             │             │                                │
│                  ▼             ▼             ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    КРОСС-МАСШТАБНОЕ СЛИЯНИЕ ВНИМАНИЯ                    │ │
│  │                                                                          │ │
│  │    Масштаб 1 ←──внимание──→ Масштаб 2 ←──внимание──→ Масштаб 3         │ │
│  │         └──────────────── внимание ─────────────────→ Масштаб 4          │ │
│  │                                                                          │ │
│  │    Обучается: "Краткосрочный моментум совпадает с долгосрочным трендом?"│ │
│  │               "Масштаб 2 подтверждает сигнал Масштаба 1?"                │ │
│  └──────────────────────────────────┬──────────────────────────────────────┘ │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    ИЕРАРХИЧЕСКАЯ АГРЕГАЦИЯ                               │ │
│  │                                                                          │ │
│  │    Взвешенная комбинация на основе выученной важности масштабов         │ │
│  │    α₁·Масштаб1 + α₂·Масштаб2 + α₃·Масштаб3 + α₄·Масштаб4                │ │
│  │                                                                          │ │
│  └──────────────────────────────────┬──────────────────────────────────────┘ │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         ГОЛОВА ПРОГНОЗИРОВАНИЯ                           │ │
│  │    • Краткосрочный прогноз (следующие 5 мин)                             │ │
│  │    • Среднесрочный прогноз (следующий 1 час)                             │ │
│  │    • Долгосрочный прогноз (следующий 1 день)                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Энкодеры для каждого масштаба

Каждый временной масштаб имеет свой собственный энкодер, оптимизированный для этого разрешения:

```python
class ScaleEncoder(nn.Module):
    """
    Энкодер для конкретного временного масштаба.

    Разные масштабы требуют разных архитектур:
    - Короткие масштабы (1-мин): CNN для локальных паттернов, dropout для шума
    - Длинные масштабы (1-день): Глубокое внимание для глобальных паттернов
    """
    def __init__(
        self,
        scale_name: str,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scale_name = scale_name

        # Позиционное кодирование для временного порядка
        self.pos_encoding = LearnablePositionalEncoding(d_model)

        # Проекция входных данных
        self.input_proj = nn.Linear(input_dim, d_model)

        # Слои Transformer энкодера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Выходная нормализация
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            x: [batch, seq_len, input_dim]
        Возвращает:
            encoded: [batch, seq_len, d_model]
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.output_norm(x)
```

**Конфигурации для разных масштабов:**

| Масштаб | Длина последовательности | n_layers | n_heads | Примечания |
|---------|-------------------------|----------|---------|------------|
| 1-мин | 1440 (1 день) | 2 | 4 | Фокус на локальных паттернах |
| 5-мин | 288 (1 день) | 3 | 8 | Баланс локального/глобального |
| 1-час | 168 (1 неделя) | 4 | 8 | Захват дневных паттернов |
| 1-день | 252 (1 год) | 4 | 8 | Долгосрочные тренды |

### Многоуровневое внимание

Ключевой механизм: внимание, работающее между масштабами:

```python
class MultiResolutionAttention(nn.Module):
    """
    Механизм внимания, запрашивающий информацию из разных временных масштабов.

    Запрос из одного масштаба может обращать внимание на ключи/значения
    из других масштабов, позволяя модели обнаруживать кросс-масштабные зависимости.
    """
    def __init__(self, d_model: int, n_heads: int, n_scales: int):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Отдельные проекции для каждого масштаба
        self.query_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])
        self.value_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])

        # Выходная проекция
        self.output_proj = nn.Linear(d_model * n_scales, d_model)

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Аргументы:
            scale_features: Список [batch, seq_len_i, d_model] для каждого масштаба
        Возвращает:
            fused: [batch, d_model] - объединённое представление
            attention_weights: [batch, n_scales, n_scales] при запросе
        """
        batch_size = scale_features[0].shape[0]

        # Вычисление запросов, ключей, значений для каждого масштаба
        # Используем последний временной шаг как запрос (самая свежая информация)
        queries = [
            self.query_projs[i](feat[:, -1, :])  # [batch, d_model]
            for i, feat in enumerate(scale_features)
        ]

        # Используем все временные шаги как ключи/значения
        keys = [
            self.key_projs[i](feat)  # [batch, seq_len, d_model]
            for i, feat in enumerate(scale_features)
        ]
        values = [
            self.value_projs[i](feat)  # [batch, seq_len, d_model]
            for i, feat in enumerate(scale_features)
        ]

        # Кросс-масштабное внимание: каждый масштаб обращает внимание на все масштабы
        attended_features = []
        attention_weights = []

        for i in range(self.n_scales):
            q = queries[i].unsqueeze(1)  # [batch, 1, d_model]

            # Внимание ко всем масштабам
            scale_attended = []
            scale_attention = []

            for j in range(self.n_scales):
                k = keys[j]  # [batch, seq_len_j, d_model]
                v = values[j]

                # Вычисление оценок внимания
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)  # [batch, 1, seq_len_j]

                # Применение к значениям
                context = torch.matmul(attn, v)  # [batch, 1, d_model]
                scale_attended.append(context.squeeze(1))

                # Сохранение для интерпретируемости
                scale_attention.append(attn.mean(dim=-1))

            # Объединение внимания от всех масштабов
            attended_features.append(torch.stack(scale_attended, dim=1).mean(dim=1))
            attention_weights.append(torch.stack(scale_attention, dim=-1))

        # Конкатенация и проекция
        fused = torch.cat(attended_features, dim=-1)  # [batch, d_model * n_scales]
        fused = self.output_proj(fused)  # [batch, d_model]

        if return_attention:
            attention_matrix = torch.stack(attention_weights, dim=1)
            return fused, attention_matrix

        return fused, None
```

### Кросс-масштабное слияние

Разные масштабы содержат взаимодополняющую информацию. Слой слияния обучается их комбинировать:

```python
class CrossScaleFusion(nn.Module):
    """
    Слияние информации из нескольких временных масштабов с обучаемыми весами.

    Использует внимание для определения, какие масштабы наиболее релевантны
    для текущей задачи прогнозирования.
    """
    def __init__(self, d_model: int, n_scales: int):
        super().__init__()

        # Обучаемые веса важности масштабов
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Гейт для каждого масштаба
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(n_scales)
        ])

        # Финальный слой слияния
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Аргументы:
            scale_features: Список [batch, d_model] от каждого масштаба
        Возвращает:
            fused: [batch, d_model]
        """
        # Вычисление динамических гейтов для каждого масштаба
        gates = []
        for i, feat in enumerate(scale_features):
            gate = self.scale_gates[i](feat)  # [batch, 1]
            gates.append(gate)

        # Комбинация с обучаемыми весами
        weights = torch.softmax(self.scale_weights, dim=0)
        weighted_features = []

        for i, feat in enumerate(scale_features):
            # Важность масштаба = обученный вес * динамический гейт
            importance = weights[i] * gates[i]
            weighted_features.append(importance * feat)

        # Конкатенация и слияние
        concat = torch.cat(weighted_features, dim=-1)
        fused = self.fusion(concat)

        if return_weights:
            return fused, weights, torch.cat(gates, dim=-1)

        return fused
```

### Иерархическая агрегация

Для очень длинных последовательностей иерархическая агрегация снижает вычислительные затраты:

```python
class HierarchicalAggregation(nn.Module):
    """
    Иерархическая агрегация для эффективной многомасштабной обработки.

    Вместо внимания ко всем токенам, агрегация в пирамиде:
    Уровень 0: Полное разрешение (все токены)
    Уровень 1: 2x понижение (средний пулинг)
    Уровень 2: 4x понижение
    ...
    """
    def __init__(
        self,
        d_model: int,
        n_levels: int = 4,
        pool_size: int = 2
    ):
        super().__init__()
        self.n_levels = n_levels
        self.pool_size = pool_size

        # Внимание на каждом иерархическом уровне
        self.level_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for _ in range(n_levels)
        ])

        # Слияние уровней
        self.level_fusion = nn.Sequential(
            nn.Linear(d_model * n_levels, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            x: [batch, seq_len, d_model]
        Возвращает:
            output: [batch, d_model]
        """
        level_outputs = []
        current = x

        for level in range(self.n_levels):
            # Self-attention на текущем уровне
            attn_out, _ = self.level_attention[level](current, current, current)

            # Последний токен как представление уровня
            level_outputs.append(attn_out[:, -1, :])

            # Понижение для следующего уровня
            if level < self.n_levels - 1:
                batch, seq_len, d = current.shape
                if seq_len >= self.pool_size:
                    # Средний пулинг
                    current = current.view(batch, seq_len // self.pool_size, self.pool_size, d)
                    current = current.mean(dim=2)

        # Слияние всех уровней
        fused = torch.cat(level_outputs, dim=-1)
        return self.level_fusion(fused)
```

## Декомпозиция временных масштабов

Перед подачей данных в многомасштабные энкодеры необходимо разложить временной ряд на разные масштабы.

### Временное понижение частоты

Простейший подход: пересэмплирование данных с разными интервалами:

```python
def temporal_downsample(
    df: pd.DataFrame,
    target_intervals: List[str] = ['1min', '5min', '1H', '1D']
) -> Dict[str, pd.DataFrame]:
    """
    Понижение частоты OHLCV данных до нескольких временных масштабов.

    Аргументы:
        df: DataFrame с 1-минутными OHLCV данными
        target_intervals: Список целевых разрешений

    Возвращает:
        Словарь, сопоставляющий интервал с DataFrame
    """
    result = {}

    for interval in target_intervals:
        if interval == '1min':
            result[interval] = df.copy()
        else:
            # Корректное пересэмплирование OHLCV данных
            resampled = df.resample(interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            result[interval] = resampled

    return result
```

### Вейвлет-декомпозиция

Вейвлеты естественным образом разлагают сигналы на разные частотные компоненты:

```python
import pywt
import numpy as np

def wavelet_decomposition(
    prices: np.ndarray,
    wavelet: str = 'db4',
    levels: int = 4
) -> Dict[str, np.ndarray]:
    """
    Декомпозиция ценового ряда с использованием вейвлет-преобразования.

    Возвращает аппроксимационные (тренд) и детальные (шум) коэффициенты
    на каждом уровне.

    Аргументы:
        prices: 1D numpy массив цен
        wavelet: Тип вейвлета (db4, haar, sym5 и др.)
        levels: Количество уровней декомпозиции

    Возвращает:
        Словарь с 'trend' и 'detail_i' для каждого уровня
    """
    # Выполнение вейвлет-декомпозиции
    coeffs = pywt.wavedec(prices, wavelet, level=levels)

    result = {
        'trend': coeffs[0]  # Аппроксимация (самая низкая частота)
    }

    # Детальные коэффициенты (от высокой к низкой частоте)
    for i, detail in enumerate(coeffs[1:], 1):
        result[f'detail_{i}'] = detail

    return result
```

### Вариационная модовая декомпозиция (VMD)

VMD обеспечивает адаптивную декомпозицию на частотные полосы:

```python
from vmdpy import VMD

def vmd_decomposition(
    prices: np.ndarray,
    n_modes: int = 4,
    alpha: float = 2000,
    tau: float = 0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7
) -> Dict[str, np.ndarray]:
    """
    Декомпозиция ценового ряда с использованием вариационной модовой декомпозиции.

    VMD адаптивно находит оптимальные центральные частоты,
    что делает его более подходящим для финансовых данных, чем вейвлеты.

    Аргументы:
        prices: 1D numpy массив цен
        n_modes: Количество мод для извлечения
        alpha: Параметр ограничения полосы пропускания
        tau: Допуск шума (0 для отсутствия шума)
        DC: Включить DC-компонент (0 или 1)
        init: Инициализация (1 = равномерная, 2 = случайная)
        tol: Допуск сходимости

    Возвращает:
        Словарь с mode_i для каждой разложенной моды
    """
    # Запуск VMD
    modes, freqs, _ = VMD(
        prices, alpha, tau, n_modes, DC, init, tol
    )

    result = {}
    for i, mode in enumerate(modes):
        # Меньший индекс = меньшая частота (тренд)
        # Больший индекс = большая частота (шум)
        result[f'mode_{i}'] = mode

    # Также возвращаем центральные частоты
    result['frequencies'] = freqs

    return result
```

## Практические примеры

### 01: Подготовка данных с многомасштабными признаками

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from datetime import datetime, timedelta

def fetch_bybit_klines(
    symbol: str = 'BTCUSDT',
    interval: str = '1',  # 1 минута
    limit: int = 1000
) -> pd.DataFrame:
    """
    Получение OHLCV данных из Bybit.

    Аргументы:
        symbol: Торговая пара (например, 'BTCUSDT')
        interval: Интервал свечи в минутах ('1', '5', '60', '240', 'D')
        limit: Количество свечей для получения

    Возвращает:
        DataFrame с OHLCV данными
    """
    url = 'https://api.bybit.com/v5/market/kline'

    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"Ошибка API: {data['retMsg']}")

    # Парсинг ответа
    klines = data['result']['list']
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Преобразование типов
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    df = df.set_index('timestamp').sort_index()

    return df


def create_multi_scale_features(
    df: pd.DataFrame,
    scales: List[str] = ['1min', '5min', '15min', '1H', '4H']
) -> Dict[str, pd.DataFrame]:
    """
    Создание DataFrame признаков для нескольких временных масштабов.

    Аргументы:
        df: 1-минутный OHLCV DataFrame
        scales: Список целевых временных масштабов

    Возвращает:
        Словарь, сопоставляющий масштаб с DataFrame признаков
    """
    result = {}

    for scale in scales:
        # Пересэмплирование при необходимости
        if scale == '1min':
            resampled = df.copy()
        else:
            resampled = df.resample(scale).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Расчёт признаков
        features = pd.DataFrame(index=resampled.index)

        # Доходности
        features['log_return'] = np.log(resampled['close'] / resampled['close'].shift(1))
        features['return_1'] = resampled['close'].pct_change(1)
        features['return_5'] = resampled['close'].pct_change(5)
        features['return_10'] = resampled['close'].pct_change(10)

        # Волатильность
        features['volatility_5'] = features['log_return'].rolling(5).std()
        features['volatility_20'] = features['log_return'].rolling(20).std()

        # Признаки объёма
        features['volume_ratio'] = resampled['volume'] / resampled['volume'].rolling(20).mean()
        features['volume_change'] = resampled['volume'].pct_change()

        # Ценовые признаки
        features['high_low_ratio'] = (resampled['high'] - resampled['low']) / resampled['close']
        features['close_position'] = (resampled['close'] - resampled['low']) / (resampled['high'] - resampled['low'] + 1e-8)

        # Скользящие средние
        for period in [5, 10, 20, 50]:
            ma = resampled['close'].rolling(period).mean()
            features[f'ma_{period}_ratio'] = resampled['close'] / ma - 1

        # RSI
        delta = resampled['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features['rsi'] = 100 - (100 / (1 + rs))

        # Нормализация RSI к [-1, 1]
        features['rsi_normalized'] = (features['rsi'] - 50) / 50

        # Удаление NaN и сохранение
        result[scale] = features.dropna()

    return result
```

### 02: Архитектура многомасштабного внимания

См. [python/model.py](python/model.py) для полной реализации.

### 03: Обучение модели

```python
# python/03_train.py

import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

def train_multi_scale_model(
    model: nn.Module,
    train_data: Tuple[Dict[str, np.ndarray], np.ndarray],
    val_data: Tuple[Dict[str, np.ndarray], np.ndarray],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Обучение модели многомасштабного внимания.

    Аргументы:
        model: Модель MultiScaleAttention
        train_data: (X_dict, y) для обучения
        val_data: (X_dict, y) для валидации
        epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Начальная скорость обучения
        device: Устройство для обучения

    Возвращает:
        Словарь истории обучения
    """
    model = model.to(device)

    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Функции потерь
    regression_loss = nn.MSELoss()
    direction_loss = nn.BCELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_direction_acc': []
    }

    # Цикл обучения...
    # (см. полную реализацию в python/train.py)

    return history
```

### 04: Многогоризонтное прогнозирование

```python
# python/04_prediction.py

def multi_horizon_predict(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Прогнозирование для нескольких горизонтов.

    Аргументы:
        model: Обученная модель MultiScaleAttention
        data: Словарь входных массивов по масштабам
        device: Используемое устройство

    Возвращает:
        Словарь с прогнозами для каждого горизонта
    """
    model.eval()
    model = model.to(device)

    # Преобразование в тензоры
    inputs = {
        k: torch.FloatTensor(v).to(device)
        for k, v in data.items()
    }

    with torch.no_grad():
        outputs = model(inputs, return_attention=True)

    return {
        'short_term': outputs['short_term'].cpu().numpy(),
        'medium_term': outputs['medium_term'].cpu().numpy(),
        'long_term': outputs['long_term'].cpu().numpy(),
        'direction': outputs['direction'].cpu().numpy(),
        'attention': outputs['attention'].cpu().numpy()
    }
```

### 05: Бэктестинг стратегии

```python
# python/05_backtest.py

from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Контейнер для результатов бэктестинга."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    equity_curve: pd.Series


def backtest_multi_scale_strategy(
    predictions: Dict[str, np.ndarray],
    prices: pd.DataFrame,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.6,
    position_sizing: str = 'confidence'
) -> BacktestResult:
    """
    Бэктестинг стратегии многомасштабного внимания.

    Аргументы:
        predictions: Прогнозы модели
        prices: DataFrame с OHLCV данными
        initial_capital: Начальный капитал
        transaction_cost: Стоимость транзакции (доля)
        confidence_threshold: Минимальная уверенность направления для торговли
        position_sizing: 'fixed', 'confidence' или 'kelly'

    Возвращает:
        BacktestResult с метриками производительности
    """
    # Реализация бэктестинга...
    # (см. полный код в python/backtest.py)

    return BacktestResult(...)
```

## Реализация на Rust

См. [rust/](rust/) для полной реализации на Rust.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── scales.rs       # Многомасштабная декомпозиция
│   ├── model/              # Многомасштабное внимание
│   │   ├── mod.rs
│   │   ├── encoder.rs      # Энкодеры для каждого масштаба
│   │   ├── attention.rs    # Многоуровневое внимание
│   │   ├── fusion.rs       # Кросс-масштабное слияние
│   │   └── network.rs      # Полная модель
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка данных Bybit
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Переход в проект Rust
cd rust

# Получение данных из Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 1

# Обучение модели
cargo run --example train -- --epochs 100 --batch-size 32

# Запуск бэктеста
cargo run --example backtest -- --model checkpoints/best_model.ot
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

### Быстрый старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск полного примера
python examples/example_usage.py

# Или пошагово:
# 1. Получение и подготовка данных
python data.py --symbol BTCUSDT --days 30

# 2. Обучение модели
python train.py --config configs/default.yaml

# 3. Запуск бэктеста
python backtest.py --model checkpoints/best_model.pt
```

## Лучшие практики

### Когда использовать многомасштабное внимание

**Хорошие случаи использования:**
- Многогоризонтное прогнозирование (прогноз нескольких таймфреймов)
- Волатильные рынки с паттернами на разных масштабах
- Ребалансировка портфеля на разных таймфреймах
- Управление рисками с многомасштабной волатильностью

**Не идеально для:**
- Высокочастотной торговли (слишком много накладных расходов)
- Единичных, очень коротких горизонтов прогноза
- Ограниченных данных (нужно достаточно для всех масштабов)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуемое | Примечания |
|----------|---------------|------------|
| `n_scales` | 3-5 | Больше масштабов = больше сложность |
| `d_model` | 64-256 | Масштабируйте с размером данных |
| `n_heads` | 4-8 | Должно делить d_model |
| `n_encoder_layers` | 2-4 | Глубже для длинных последовательностей |
| `lookback` | 24-168 на масштаб | Зависит от частоты данных |
| `dropout` | 0.1-0.3 | Выше для малых датасетов |

### Распространённые ошибки

1. **Выравнивание масштабов**: Убедитесь, что временные метки выровнены между масштабами
2. **Утечка данных**: Не используйте будущие данные при понижении частоты
3. **Несбалансированные масштабы**: Некоторые масштабы могут доминировать; используйте сбалансированные потери
4. **Переобучение**: Многомасштабные модели имеют много параметров; используйте регуляризацию

### Оптимизация производительности

1. **Память**: Используйте gradient checkpointing для длинных последовательностей
2. **Скорость**: Обрабатывайте масштабы параллельно, когда возможно
3. **Инференс**: Кэшируйте промежуточные представления
4. **Размер батча**: Балансируйте память и пропускную способность

## Ресурсы

### Научные работы

- [Multi-Scale Temporal Memory for Financial Time Series](https://arxiv.org/abs/2201.08586) — Основополагающая работа
- [VMD-MSANet: Multi-Scale Attention with VMD](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015267) — Интеграция VMD
- [MSTAN: Multi-Scale Temporal Attention Network](https://www.researchgate.net/publication/398598476_MSTAN_A_multi-scale_temporal_attention_network_for_stock_prediction) — Вариант для прогноза акций
- [Multi-Scale Temporal Neural Network](https://www.ijcai.org/proceedings/2025/0364.pdf) — IJCAI 2025
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальный Transformer

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Многогоризонтное прогнозирование
- [Глава 43: Stockformer Multivariate](../43_stockformer_multivariate) — Кросс-активное внимание
- [Глава 46: Temporal Attention Networks](../46_temporal_attention_networks) — Временное внимание
- [Глава 48: Positional Encoding Timeseries](../48_positional_encoding_timeseries) — Позиционные кодирования

### Реализации

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Библиотека временных рядов
- [Informer](https://github.com/zhouhaoyi/Informer2020) — Эффективные трансформеры
- [vmdpy](https://github.com/vrcarva/vmdpy) — Реализация VMD

---

## Уровень сложности

**Продвинутый**

Предварительные требования:
- Архитектура Transformer и механизмы внимания
- Анализ временных рядов и инженерия признаков
- Концепции многогоризонтного прогнозирования
- Фреймворки ML PyTorch/Rust
