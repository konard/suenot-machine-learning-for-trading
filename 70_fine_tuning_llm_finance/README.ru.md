# Глава 70: Fine-tuning LLM для финансов — LoRA, QLoRA и Prefix-Tuning

В этой главе рассматриваются **методы дообучения больших языковых моделей (LLM)** для финансовой сферы. Мы изучим методы параметрически-эффективного дообучения (PEFT), включая LoRA, QLoRA и prefix-tuning, и покажем, как адаптировать базовые модели для анализа финансовых настроений, прогнозирования рынка и генерации торговых сигналов.

<p align="center">
<img src="https://i.imgur.com/KLMnB8v.png" width="70%">
</p>

## Содержание

1. [Введение в дообучение LLM](#введение-в-дообучение-llm)
    * [Зачем дообучать для финансов?](#зачем-дообучать-для-финансов)
    * [Полное дообучение vs PEFT](#полное-дообучение-vs-peft)
    * [Ключевые методы PEFT](#ключевые-методы-peft)
2. [LoRA: Низкоранговая адаптация](#lora-низкоранговая-адаптация)
    * [Математические основы](#математические-основы)
    * [Детали реализации](#детали-реализации)
    * [Выбор гиперпараметров](#выбор-гиперпараметров)
3. [QLoRA: Квантованная LoRA](#qlora-квантованная-lora)
    * [4-битное квантование](#4-битное-квантование)
    * [Двойное квантование](#двойное-квантование)
    * [Эффективность памяти](#эффективность-памяти)
4. [Prefix-Tuning](#prefix-tuning)
    * [Мягкие промпты](#мягкие-промпты)
    * [Виртуальные токены](#виртуальные-токены)
    * [Сравнение с LoRA](#сравнение-с-lora)
5. [Финансовые применения](#финансовые-применения)
    * [Анализ настроений](#анализ-настроений)
    * [Прогнозирование рынка](#прогнозирование-рынка)
    * [Генерация торговых сигналов](#генерация-торговых-сигналов)
6. [Практические примеры](#практические-примеры)
    * [01: Дообучение для финансовых настроений](#01-дообучение-для-финансовых-настроений)
    * [02: Анализ крипторынка с данными Bybit](#02-анализ-крипторынка-с-данными-bybit)
    * [03: Бэктестинг дообученных моделей](#03-бэктестинг-дообученных-моделей)
7. [Реализация на Rust](#реализация-на-rust)
8. [Реализация на Python](#реализация-на-python)
9. [Лучшие практики](#лучшие-практики)
10. [Ресурсы](#ресурсы)

## Введение в дообучение LLM

Дообучение адаптирует предварительно обученные большие языковые модели к конкретным доменам или задачам. В финансах это позволяет моделям понимать специализированную терминологию, точно интерпретировать рыночные настроения и генерировать действенные торговые сигналы.

### Зачем дообучать для финансов?

Предобученные модели не обладают экспертизой в предметной области:

```
ПРОБЛЕМЫ ОБЩИХ LLM В ФИНАНСАХ:
┌──────────────────────────────────────────────────────────────────┐
│  1. ДОМЕННАЯ ТЕРМИНОЛОГИЯ                                         │
│     "Акция торгуется с forward P/E 25x при сильном FCF yield"    │
│     Общая LLM: Может неправильно интерпретировать показатели     │
│     Дообученная: Понимает метрики оценки в контексте            │
├──────────────────────────────────────────────────────────────────┤
│  2. НЮАНСЫ НАСТРОЕНИЙ                                             │
│     "Компания сохранила прогноз несмотря на макро-трудности"     │
│     Общая LLM: Нейтрально или негативно?                         │
│     Дообученная: Распознаёт как умеренно позитивное             │
├──────────────────────────────────────────────────────────────────┤
│  3. ВРЕМЕННЫЕ ПАТТЕРНЫ                                            │
│     "Превысили консенсус на 200 б.п., повысили годовой прогноз" │
│     Общая LLM: Может упустить контекст отчётного сезона         │
│     Дообученная: Понимает квартальные паттерны отчётности       │
├──────────────────────────────────────────────────────────────────┤
│  4. ОЦЕНКА ВЛИЯНИЯ НА РЫНОК                                       │
│     "ФРС сигнализирует ястребиный поворот, доходности растут"   │
│     Общая LLM: Может не связать с торговыми последствиями       │
│     Дообученная: Понимает межрыночные взаимосвязи              │
└──────────────────────────────────────────────────────────────────┘
```

### Полное дообучение vs PEFT

| Аспект | Полное дообучение | PEFT (LoRA/QLoRA) |
|--------|-------------------|-------------------|
| Обновляемые параметры | Все (миллиарды) | 0.1-1% от общего |
| Память GPU | 40-80GB+ на GPU | 4-16GB один GPU |
| Время обучения | Дни-недели | Часы-дни |
| Катастрофическое забывание | Высокий риск | Низкий риск |
| Хранение на задачу | Полная копия модели | Малые файлы адаптеров |
| Развёртывание | Сложное | Простая смена адаптеров |

### Ключевые методы PEFT

```
ЛАНДШАФТ ПАРАМЕТРИЧЕСКИ-ЭФФЕКТИВНОГО ДООБУЧЕНИЯ:
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                    МЕТОДЫ НА ОСНОВЕ АДАПТЕРОВ                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │    LoRA       │  │   QLoRA       │  │   AdaLoRA     │        │
│  │  Низкоранго-  │  │  4-бит квант  │  │  Адаптивное   │        │
│  │  вая адапт.   │  │  + LoRA       │  │  распред.ранга│        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                  │
│  Параметры: 0.1-1%  │  Память: 4-8GB  │  Сохраняет знания       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    МЕТОДЫ НА ОСНОВЕ ПРОМПТОВ                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Prefix-Tuning │  │ Prompt-Tuning │  │  P-Tuning v2  │        │
│  │  Виртуальные  │  │  Мягкие       │  │  Глубокое     │        │
│  │  токены       │  │  промпты      │  │  prompt-тюнинг│        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                  │
│  Параметры: <0.1%  │  Память: 2-4GB  │  Специфичные для задачи  │
└─────────────────────────────────────────────────────────────────┘
```

## LoRA: Низкоранговая адаптация

LoRA (Low-Rank Adaptation) — самый популярный метод PEFT, вводящий обучаемые низкоранговые матрицы, которые модифицируют поведение замороженных предобученных весов.

### Математические основы

Вместо обновления полной матрицы весов W ∈ ℝ^(d×k), LoRA обучает низкоранговое разложение:

```
МЕХАНИЗМ ОБНОВЛЕНИЯ ВЕСОВ LORA:
═══════════════════════════════════════════════════════════════════

Исходная матрица весов:    W₀ ∈ ℝ^(d×k)     (заморожена)
LoRA разложение:           ΔW = BA
  где: B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

Прямой проход:             h = W₀x + ΔWx = W₀x + BAx

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Вход x ───────┬──────────────────────────────┬─────── Выход h  │
│                │                              │                  │
│                ▼                              ▼                  │
│        ┌──────────────┐              ┌──────────────┐           │
│        │W₀ (заморож.) │              │   BA (LoRA)  │           │
│        │   d × k      │              │   d × r × k  │           │
│        └──────────────┘              └──────────────┘           │
│                │                              │                  │
│                └────────────┬─────────────────┘                  │
│                             ▼                                    │
│                       h = W₀x + αBAx                            │
│                       (α = коэффициент масштаб.)                │
└─────────────────────────────────────────────────────────────────┘

Пример сокращения параметров:
  Исходно: d=4096, k=4096 → 16.7М параметров
  LoRA r=8: (4096×8) + (8×4096) = 65K параметров (0.4%)
  LoRA r=16: (4096×16) + (16×4096) = 131K параметров (0.8%)
```

### Детали реализации

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Реализация слоя LoRA для дообучения финансовых LLM.

    Этот слой добавляет обучаемые низкоранговые матрицы к замороженным
    предобученным весам, обеспечивая эффективную адаптацию к финансовым
    задачам, таким как анализ настроений.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Замороженный предобученный вес (симуляция, обычно из базовой модели)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # Обучаемые матрицы LoRA
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)

        # Инициализация A по Kaiming, B нулями (начинаем с оригинала)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Исходное преобразование (заморожено)
        result = x @ self.weight.T

        # Адаптация LoRA
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return result + self.scaling * lora_result


class FinancialSentimentLoRA(nn.Module):
    """
    Классификатор финансовых настроений на основе LoRA-адаптированного
    трансформера.

    Классифицирует финансовый текст по категориям настроений:
    - Бычий (позитивный прогноз рынка)
    - Медвежий (негативный прогноз рынка)
    - Нейтральный (нет явного направленного сигнала)
    """

    def __init__(
        self,
        base_dim: int = 768,
        lora_rank: int = 8,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # LoRA-адаптированная проекция внимания
        self.query_lora = LoRALayer(base_dim, base_dim, lora_rank)
        self.value_lora = LoRALayer(base_dim, base_dim, lora_rank)

        # Классификационная голова (полностью обучаемая)
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim // 2, num_classes)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Применяем преобразования LoRA
        query = self.query_lora(hidden_states)
        value = self.value_lora(hidden_states)

        # Простая агрегация (mean pooling)
        pooled = hidden_states.mean(dim=1)

        # Классификация
        logits = self.classifier(pooled)
        return logits
```

### Выбор гиперпараметров

Оптимальные гиперпараметры для финансового дообучения:

| Гиперпараметр | Рекомендуемый диапазон | Финансовые задачи |
|---------------|------------------------|-------------------|
| Ранг (r) | 4-64 | 8-16 для настроений, 16-32 для генерации |
| Альфа (α) | r до 2r | Обычно хорошо работает 2×ранг |
| Скорость обучения | 1e-4 до 3e-4 | Ниже для больших моделей |
| Dropout | 0.05-0.1 | 0.1 для малых датасетов |
| Целевые модули | q_proj, v_proj | Добавьте k_proj, o_proj для сложных задач |
| Шаги прогрева | 5-10% | Критично для стабильности |

```
РУКОВОДСТВО ПО ВЫБОРУ РАНГА ДЛЯ ФИНАНСОВЫХ ЗАДАЧ:
═══════════════════════════════════════════════════════════════════

┌─────────────────┬──────────┬────────────────────────────────────┐
│ Задача          │ Ранг     │ Обоснование                        │
├─────────────────┼──────────┼────────────────────────────────────┤
│ Бин. настроения │ r=4-8    │ Простая классиф., низкий ранг     │
│ Многоклассовая  │ r=8-16   │ Больше нюансов требует ёмкости    │
│ Распозн. сущн.  │ r=16-32  │ Точное определение границ          │
│ Генерация текста│ r=32-64  │ Сложное выходное пространство     │
│ Мультизадачность│ r=64+    │ Множество целей для баланса       │
└─────────────────┴──────────┴────────────────────────────────────┘

Размер обучающих данных vs Ранг:
  < 1K примеров   → r=4-8   (предотвращение переобучения)
  1K-10K примеров → r=8-16  (сбалансированная ёмкость)
  10K+ примеров   → r=16-32 (можно использовать больше параметров)
```

## QLoRA: Квантованная LoRA

QLoRA сочетает 4-битное квантование с LoRA, позволяя дообучать большие модели на потребительском железе.

### 4-битное квантование

```
СХЕМА КВАНТОВАНИЯ QLORA:
═══════════════════════════════════════════════════════════════════

Базовая модель (FP16):       16 бит на параметр
После квантования NF4:        4 бита на параметр  (4× сжатие)

┌─────────────────────────────────────────────────────────────────┐
│              Тип данных NormalFloat4 (NF4)                       │
│                                                                  │
│  Значения распределены по квантилям нормального распределения:  │
│                                                                  │
│  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,   │
│    0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, │
│    1.0]                                                          │
│                                                                  │
│  Почему NF4?                                                     │
│  - Веса нейросетей следуют ~нормальному распределению           │
│  - NF4 оптимально покрывает это распределение                   │
│  - Лучшее сохранение качества vs равномерное квантование        │
└─────────────────────────────────────────────────────────────────┘

Сравнение памяти (модель 7B параметров):
  FP32: 28 ГБ
  FP16: 14 ГБ
  INT8:  7 ГБ
  NF4:   3.5 ГБ  ← QLoRA работает здесь
```

### Двойное квантование

```python
# Конфигурация QLoRA для финансового дообучения
from transformers import BitsAndBytesConfig

qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Использовать 4-битное квантование
    bnb_4bit_quant_type="nf4",      # Квантование NormalFloat4
    bnb_4bit_use_double_quant=True,  # Двойное квантование для констант
    bnb_4bit_compute_dtype=torch.bfloat16  # Вычисления в BF16
)

# Двойное квантование экономит дополнительную память:
# Константы квантования (32-бит) → тоже квантуются (8-бит)
# Экономит ~0.37 бит на параметр в среднем
```

### Эффективность памяти

```
СРАВНЕНИЕ РАСХОДА ПАМЯТИ (Дообучение модели 7B):
═══════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│ Метод            │ Модель│ Оптимиз.│ Градиенты │ Всего        │
├──────────────────┼───────┼─────────┼───────────┼──────────────┤
│ Полное FP16      │ 14ГБ  │ 28ГБ    │ 14ГБ      │ ~56ГБ        │
│ Полное + ZeRO-3  │ 5ГБ   │ 9ГБ     │ 5ГБ       │ ~19ГБ/GPU    │
│ LoRA FP16        │ 14ГБ  │ 0.1ГБ   │ 0.05ГБ    │ ~15ГБ        │
│ QLoRA NF4        │ 3.5ГБ │ 0.1ГБ   │ 0.05ГБ    │ ~4ГБ         │
│ QLoRA + Gradient │ 3.5ГБ │ 0.1ГБ   │ ~0ГБ*     │ ~4ГБ         │
│ Checkpointing    │       │         │           │              │
└────────────────────────────────────────────────────────────────┘

* Gradient checkpointing обменивает память на вычисления

Требования к оборудованию:
  Полное дообучение 7B:  4× A100 80GB
  LoRA 7B:               1× A100 40GB
  QLoRA 7B:              1× RTX 3090/4090 (24GB)
  QLoRA 7B + 8bit opt:   1× RTX 3080 (10GB)
```

## Prefix-Tuning

Prefix-tuning добавляет обучаемые непрерывные векторы (префиксы) к входу, направляя поведение модели без изменения весов.

### Мягкие промпты

```
МЕХАНИЗМ PREFIX-TUNING:
═══════════════════════════════════════════════════════════════════

Традиционный промптинг (дискретный):
  Вход: "Классифицируй настроение: [текст]" → Жёстко заданные токены

Prefix-Tuning (непрерывный):
  Вход: [P₁, P₂, ..., Pₘ] + [токены текста] → Обученные эмбеддинги

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Префикс (обучаемый)        │ Входной текст (заморож. код.)  ││
│  │ [P₁] [P₂] [P₃] ... [Pₘ]   │ [CLS] Акция ... [SEP]          ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌────────────────┐           ┌────────────────────┐            │
│  │  Prefix MLP    │           │  Заморож. тело LLM │            │
│  │  (обучаемый)   │           │  (без градиентов)  │            │
│  └────────────────┘           └────────────────────┘            │
│           │                              │                       │
│           └──────────────────────────────┘                       │
│                       │                                          │
│                       ▼                                          │
│            ┌────────────────────┐                               │
│            │   Выход / Потери   │                               │
│            └────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

Параметры префикса:
  m = длина префикса (обычно 10-100 токенов)
  Каждый токен префикса имеет размерность d (hidden size)
  Всего параметров: m × d × num_layers (для глубокого префикса)
```

### Виртуальные токены

```python
import torch
import torch.nn as nn

class PrefixTuningLayer(nn.Module):
    """
    Реализация prefix-tuning для адаптации финансовых LLM.

    Использует обучаемые эмбеддинги префиксов для управления
    поведением модели в финансовых задачах без изменения
    весов базовой модели.
    """

    def __init__(
        self,
        num_prefix_tokens: int = 20,
        hidden_dim: int = 768,
        num_layers: int = 12,
        prefix_projection: bool = True,
        prefix_hidden_dim: int = 512
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if prefix_projection:
            # Двухэтапный: эмбеддинг → MLP → префикс
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                prefix_hidden_dim
            )
            self.prefix_mlp = nn.Sequential(
                nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(prefix_hidden_dim, num_layers * 2 * hidden_dim)
            )
        else:
            # Прямые параметры префикса
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                num_layers * 2 * hidden_dim
            )
            self.prefix_mlp = nn.Identity()

    def forward(self, batch_size: int) -> tuple:
        """
        Генерация пар ключ-значение префикса для всех слоёв.

        Returns:
            Кортеж (prefix_keys, prefix_values) для каждого слоя
        """
        prefix_tokens = torch.arange(self.num_prefix_tokens).unsqueeze(0)
        prefix_tokens = prefix_tokens.expand(batch_size, -1)

        # Получаем эмбеддинги префикса и проецируем
        prefix_embeds = self.prefix_embedding(prefix_tokens)
        prefix_output = self.prefix_mlp(prefix_embeds)

        # Reshape в (batch, layers, 2, num_prefix, hidden)
        prefix_output = prefix_output.view(
            batch_size,
            self.num_prefix_tokens,
            self.num_layers,
            2,
            self.hidden_dim
        )
        prefix_output = prefix_output.permute(2, 3, 0, 1, 4)

        # Разделяем на ключи и значения для каждого слоя
        prefix_keys = prefix_output[:, 0]   # (layers, batch, prefix, hidden)
        prefix_values = prefix_output[:, 1]

        return prefix_keys, prefix_values
```

### Сравнение с LoRA

| Аспект | LoRA | Prefix-Tuning |
|--------|------|---------------|
| Где применяется | Матрицы весов | Входная последовательность |
| Параметры | ~0.5% модели | ~0.1% модели |
| Влияние на длину посл. | Нет | Уменьшает эффективный контекст |
| Мультизадачность | Отдельные адаптеры | Отдельные префиксы |
| Качество генерации | Лучше | Может влиять на беглость |
| Классификация | Хорошо | Очень хорошо |
| Лучше для | Общей адаптации | Специфичного управления задачей |

## Финансовые применения

### Анализ настроений

Дообучение для финансовых настроений с LoRA:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def create_financial_sentiment_model():
    """
    Создание LoRA-адаптированной модели для анализа финансовых настроений.

    Метки:
        0: Медвежий (негативное настроение, сигнал продажи)
        1: Нейтральный (нет явного направления)
        2: Бычий (позитивное настроение, сигнал покупки)
    """
    # Загрузка базовой модели
    model_name = "ProsusAI/finbert"  # или "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Конфигурация LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                          # Ранг
        lora_alpha=16,                # Масштабирование
        lora_dropout=0.1,             # Dropout
        target_modules=["query", "value"],  # Применяем к Q и V проекциям
        bias="none"
    )

    # Создание PEFT модели
    peft_model = get_peft_model(model, lora_config)

    # Вывод обучаемых параметров
    peft_model.print_trainable_parameters()
    # Вывод: trainable params: 294,912 || all params: 109,777,923 || trainable%: 0.27%

    return peft_model, tokenizer


# Пример формата обучающих данных
financial_examples = [
    {
        "text": "Apple превзошла ожидания по прибыли, повысила дивиденды на 10%",
        "label": 2  # Бычий
    },
    {
        "text": "ФРС сигнализирует агрессивное повышение ставок, рынки падают",
        "label": 0  # Медвежий
    },
    {
        "text": "Компания сохранила прогноз на Q4 на фоне смешанных экономических сигналов",
        "label": 1  # Нейтральный
    },
    {
        "text": "Bitcoin взлетел выше $50K на институциональных покупках",
        "label": 2  # Бычий (крипто)
    },
    {
        "text": "Bybit сообщает о рекордных объёмах торгов на фоне роста волатильности BTC",
        "label": 1  # Нейтральный (рыночная активность, не направление)
    }
]
```

### Прогнозирование рынка

```python
class MarketDirectionPredictor(nn.Module):
    """
    Дообученная LLM для прогнозирования направления рынка.

    Комбинирует текстовые сигналы (новости, настроения) с числовыми
    признаками (цена, объём) для прогноза направления на следующий день.
    """

    def __init__(
        self,
        text_model,           # LoRA-адаптированный трансформер
        numerical_features: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.text_encoder = text_model

        # Процессор числовых признаков
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Слой слияния
        text_dim = text_model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # Вниз, Боковик, Вверх
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        # Кодирование текста
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_embedding = text_output.hidden_states[-1][:, 0]  # CLS токен

        # Кодирование числовых признаков
        num_embedding = self.numerical_encoder(numerical_features)

        # Слияние и прогноз
        combined = torch.cat([text_embedding, num_embedding], dim=-1)
        logits = self.fusion(combined)

        return logits
```

### Генерация торговых сигналов

```python
class LLMTradingSignalGenerator:
    """
    Генерация торговых сигналов с помощью дообученной LLM.

    Комбинирует анализ настроений с оценкой уверенности
    для получения действенных торговых сигналов.
    """

    def __init__(
        self,
        sentiment_model,
        tokenizer,
        confidence_threshold: float = 0.7
    ):
        self.model = sentiment_model
        self.tokenizer = tokenizer
        self.threshold = confidence_threshold
        self.label_map = {0: "ПРОДАВАТЬ", 1: "ДЕРЖАТЬ", 2: "ПОКУПАТЬ"}

    def generate_signal(self, text: str) -> dict:
        """
        Генерация торгового сигнала из текста.

        Args:
            text: Финансовая новость или аналитический текст

        Returns:
            Словарь с сигналом, уверенностью и сырыми оценками
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Получение предсказаний
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        confidence, prediction = probs.max(dim=-1)

        # Генерация сигнала
        signal = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": self.label_map[prediction.item()],
            "confidence": confidence.item(),
            "scores": {
                "медвежий": probs[0, 0].item(),
                "нейтральный": probs[0, 1].item(),
                "бычий": probs[0, 2].item()
            },
            "actionable": confidence.item() >= self.threshold
        }

        return signal

    def aggregate_signals(self, signals: list) -> dict:
        """
        Агрегация множества сигналов в композитный сигнал.

        Использует голосование, взвешенное по уверенности.
        """
        if not signals:
            return {"signal": "ДЕРЖАТЬ", "confidence": 0.0}

        weighted_scores = {"ПРОДАВАТЬ": 0, "ДЕРЖАТЬ": 0, "ПОКУПАТЬ": 0}
        total_weight = 0

        for sig in signals:
            weight = sig["confidence"]
            weighted_scores[sig["prediction"]] += weight
            total_weight += weight

        # Нормализация
        for key in weighted_scores:
            weighted_scores[key] /= total_weight

        # Получение финального сигнала
        final_signal = max(weighted_scores, key=weighted_scores.get)

        return {
            "signal": final_signal,
            "confidence": weighted_scores[final_signal],
            "score_breakdown": weighted_scores,
            "num_sources": len(signals)
        }
```

## Практические примеры

### 01: Дообучение для финансовых настроений

См. `python/examples/01_sentiment_finetuning.py` для полной реализации.

```python
# Быстрый старт
from python.trainer import FineTuningTrainer
from python.data_loader import load_financial_phrasebank

# Загрузка данных
train_data, val_data = load_financial_phrasebank()

# Создание тренера с LoRA
trainer = FineTuningTrainer(
    model_name="ProsusAI/finbert",
    method="lora",
    lora_rank=8,
    learning_rate=2e-4
)

# Обучение
trainer.train(train_data, val_data, epochs=3)

# Оценка
metrics = trainer.evaluate(val_data)
print(f"Точность: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### 02: Анализ крипторынка с данными Bybit

См. `python/examples/02_crypto_analysis.py` для полной реализации.

```python
# Анализ настроений криптовалют с данными Bybit
from python.data_loader import BybitDataLoader
from python.signals import CryptoSignalGenerator

# Инициализация загрузчика Bybit
bybit = BybitDataLoader()

# Получение последних рыночных данных
btc_data = bybit.get_klines(
    symbol="BTCUSDT",
    interval="1h",
    limit=1000
)

# Загрузка дообученной модели
signal_gen = CryptoSignalGenerator.from_pretrained(
    "outputs/crypto_sentiment_model"
)

# Генерация сигналов из новостей
news_texts = [
    "Киты Bitcoin накапливают позиции пока цена консолидируется у поддержки",
    "Регуляторные опасения давят на настроения крипторынка",
    "Bybit запускает новые бессрочные контракты со сниженными комиссиями"
]

signals = signal_gen.batch_signals(news_texts)
composite = signal_gen.aggregate_signals(signals)

print(f"Композитный сигнал: {composite['signal']}")
print(f"Уверенность: {composite['confidence']:.2%}")
```

### 03: Бэктестинг дообученных моделей

См. `python/examples/03_backtest.py` для полной реализации.

```python
# Бэктест торговых сигналов LLM
from python.backtest import LLMBacktester
from python.data_loader import YahooFinanceLoader

# Загрузка исторических данных
yahoo = YahooFinanceLoader()
spy_data = yahoo.get_daily("SPY", start="2020-01-01", end="2024-01-01")

# Инициализация бэктестера с дообученной моделью
backtester = LLMBacktester(
    model_path="outputs/sentiment_model",
    initial_capital=100000,
    position_size=0.1,
    confidence_threshold=0.7
)

# Запуск бэктеста с данными новостей
results = backtester.run(
    price_data=spy_data,
    news_data=news_headlines,  # Исторические заголовки новостей
    signal_aggregation="confidence_weighted"
)

# Вывод метрик
print(f"Общая доходность: {results['total_return']:.2%}")
print(f"Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
print(f"Макс. просадка: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

## Реализация на Rust

Реализация на Rust обеспечивает высокопроизводительный инференс для продакшн-развёртывания. См. директорию `rust/` для полного кода.

```rust
//! Дообучение финансовых LLM - реализация на Rust
//!
//! Этот крейт обеспечивает эффективный инференс для дообученных моделей,
//! разработанный для низколатентной генерации торговых сигналов.

use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Module};
use serde::{Deserialize, Serialize};

/// Слой LoRA для эффективной адаптации модели
pub struct LoraLayer {
    lora_a: Tensor,      // (rank, in_features)
    lora_b: Tensor,      // (out_features, rank)
    scaling: f64,
    rank: usize,
}

impl LoraLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let lora_a = vb.get((rank, in_features), "lora_a")?;
        let lora_b = vb.get((out_features, rank), "lora_b")?;

        Ok(Self {
            lora_a,
            lora_b,
            scaling: alpha / rank as f64,
            rank,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Вычисляем BA @ x с масштабированием
        let intermediate = x.matmul(&self.lora_a.t()?)?;
        let output = intermediate.matmul(&self.lora_b.t()?)?;
        output.affine(self.scaling, 0.0)
    }
}

/// Торговый сигнал, сгенерированный дообученной моделью
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub direction: SignalDirection,
    pub confidence: f64,
    pub scores: SentimentScores,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SignalDirection {
    Buy,    // Покупать
    Hold,   // Держать
    Sell,   // Продавать
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    pub bullish: f64,   // Бычий
    pub neutral: f64,   // Нейтральный
    pub bearish: f64,   // Медвежий
}

/// Высокопроизводительный генератор сигналов для продакшна
pub struct SignalGenerator {
    model: FineTunedModel,
    tokenizer: tokenizers::Tokenizer,
    confidence_threshold: f64,
}

impl SignalGenerator {
    pub fn from_pretrained(path: &str) -> anyhow::Result<Self> {
        let model = FineTunedModel::load(path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(
            format!("{}/tokenizer.json", path)
        )?;

        Ok(Self {
            model,
            tokenizer,
            confidence_threshold: 0.7,
        })
    }

    pub fn generate(&self, text: &str) -> anyhow::Result<TradingSignal> {
        // Токенизация
        let encoding = self.tokenizer.encode(text, true)?;
        let tokens = encoding.get_ids();

        // Создание тензора
        let device = Device::Cpu;
        let input_ids = Tensor::new(tokens, &device)?;

        // Прямой проход
        let logits = self.model.forward(&input_ids)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;

        // Извлечение предсказаний
        let probs_vec: Vec<f64> = probs.to_vec1()?;
        let (max_idx, max_prob) = probs_vec.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let direction = match max_idx {
            0 => SignalDirection::Sell,
            1 => SignalDirection::Hold,
            _ => SignalDirection::Buy,
        };

        Ok(TradingSignal {
            direction,
            confidence: *max_prob,
            scores: SentimentScores {
                bearish: probs_vec[0],
                neutral: probs_vec[1],
                bullish: probs_vec[2],
            },
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
}
```

## Реализация на Python

Реализация на Python включает полные пайплайны обучения и оценки. См. директорию `python/` для полного кода.

**Основные модули:**

| Модуль | Описание |
|--------|----------|
| `model.py` | Реализации LoRA, QLoRA и prefix-tuning |
| `trainer.py` | Цикл обучения с ранней остановкой и чекпоинтами |
| `data_loader.py` | Загрузчики данных Yahoo Finance и Bybit |
| `signals.py` | Генерация и агрегация торговых сигналов |
| `backtest.py` | Фреймворк бэктестинга для LLM сигналов |
| `evaluate.py` | Метрики оценки (accuracy, F1, Sharpe и др.) |

## Лучшие практики

### Рекомендации по обучению

```
ЛУЧШИЕ ПРАКТИКИ ДООБУЧЕНИЯ:
═══════════════════════════════════════════════════════════════════

1. ПОДГОТОВКА ДАННЫХ
   ✓ Балансируйте классы (оверсэмплинг миноритарных, focal loss)
   ✓ Очищайте финансовый жаргон консистентно
   ✓ Включайте временной контекст в текст
   ✓ Разделяйте train/val/test по времени (без утечки будущего)

2. ВЫБОР ГИПЕРПАРАМЕТРОВ
   ✓ Начинайте с r=8 для LoRA, увеличивайте при недообучении
   ✓ Используйте alpha = 2 × rank как базу
   ✓ Скорость обучения: 1e-4 до 3e-4 для адаптеров
   ✓ Размер батча: 8-32 (накапливайте градиенты при ограничении GPU)

3. РЕГУЛЯРИЗАЦИЯ
   ✓ LoRA dropout: 0.05-0.1
   ✓ Weight decay: 0.01-0.1
   ✓ Ранняя остановка по validation loss
   ✓ Обрезка градиентов: max_norm=1.0

4. ОЦЕНКА
   ✓ Используйте временное разделение train/val/test
   ✓ Сообщайте и классификационные И торговые метрики
   ✓ Тестируйте на разных рыночных режимах
   ✓ Рассчитывайте статистическую значимость

5. РАЗВЁРТЫВАНИЕ
   ✓ Квантуйте модель для инференса (INT8)
   ✓ Батчируйте предсказания когда возможно
   ✓ Мониторьте латентность предсказаний
   ✓ Реализуйте пороги уверенности
```

### Типичные ошибки

```
ТИПИЧНЫЕ ОШИБКИ, КОТОРЫХ СЛЕДУЕТ ИЗБЕГАТЬ:
═══════════════════════════════════════════════════════════════════

❌ Использование будущих данных в обучении
   → Всегда используйте строгое временное разделение

❌ Игнорирование дисбаланса классов
   → Финансовые настроения часто смещены; используйте взвешенный loss

❌ Чрезмерная опора на accuracy
   → Используйте F1, precision, recall для несбалансированных данных

❌ Отсутствие out-of-sample тестирования
   → Тестируйте на отложенных временных периодах

❌ Игнорирование транзакционных издержек
   → Включайте издержки в метрики бэктеста

❌ Переобучение на конкретный рыночный режим
   → Валидируйте на бычьих/медвежьих/боковых рынках

❌ Использование слишком высокого ранга LoRA
   → Может переобучиться на малых датасетах; начинайте с r=4-8

❌ Не отслеживание забывания
   → Периодически проверяйте базовые способности
```

## Ресурсы

### Научные работы

1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   - https://arxiv.org/abs/2106.09685

2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
   - https://arxiv.org/abs/2305.14314

3. **Prefix-Tuning: Optimizing Continuous Prompts** (Li & Liang, 2021)
   - https://arxiv.org/abs/2101.00190

4. **FinBERT: Financial Sentiment Analysis** (Araci, 2019)
   - https://arxiv.org/abs/1908.10063

5. **BloombergGPT: A Large Language Model for Finance** (Wu et al., 2023)
   - https://arxiv.org/abs/2303.17564

### Датасеты

| Датасет | Описание | Размер |
|---------|----------|--------|
| Financial PhraseBank | Размеченные по настроениям финансовые новости | 4,840 предложений |
| FiQA | Вопросно-ответные пары по финансам | 17,000+ пар |
| SemEval-2017 Task 5 | Настроения в финансовых микроблогах | 2,000+ текстов |
| Crypto Sentiment | Настроения криптовалют в Twitter | 10,000+ твитов |

### Инструменты и библиотеки

- [HuggingFace PEFT](https://github.com/huggingface/peft) - Параметрически-эффективное дообучение
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-битное квантование
- [Candle](https://github.com/huggingface/candle) - Rust ML фреймворк
- [yfinance](https://github.com/ranaroussi/yfinance) - Данные Yahoo Finance
- [ccxt](https://github.com/ccxt/ccxt) - Данные криптобирж (Bybit)

### Структура директории

```
70_fine_tuning_llm_finance/
├── README.md              # Английская документация
├── README.ru.md           # Этот файл (Русский перевод)
├── readme.simple.md       # Упрощённое объяснение для начинающих
├── readme.simple.ru.md    # Упрощённое объяснение (Русский)
├── python/
│   ├── __init__.py
│   ├── model.py           # Реализации LoRA/QLoRA/Prefix
│   ├── trainer.py         # Пайплайн обучения
│   ├── data_loader.py     # Загрузчики Yahoo Finance и Bybit
│   ├── signals.py         # Генерация сигналов
│   ├── backtest.py        # Фреймворк бэктестинга
│   ├── evaluate.py        # Метрики оценки
│   ├── requirements.txt   # Python зависимости
│   └── examples/
│       ├── 01_sentiment_finetuning.py
│       ├── 02_crypto_analysis.py
│       └── 03_backtest.py
└── rust/
    ├── Cargo.toml
    ├── README.md
    └── src/
        ├── lib.rs         # Корень библиотеки
        ├── lora.rs        # Реализация LoRA
        ├── model.rs       # Загрузка модели
        ├── signals.rs     # Генерация сигналов
        ├── data.rs        # Загрузка данных
        └── bin/
            ├── sentiment.rs
            └── backtest.rs
```
