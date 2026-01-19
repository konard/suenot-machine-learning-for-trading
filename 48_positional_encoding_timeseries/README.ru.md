# Глава 48: Позиционное кодирование для временных рядов

Эта глава исследует методы **позиционного кодирования** (Positional Encoding), специально разработанные для временных рядов и финансовых данных. В отличие от стандартных NLP-трансформеров, временные ряды требуют специализированных кодировок, которые захватывают временные паттерны, периодичность и динамику финансовых рынков.

<p align="center">
<img src="https://i.imgur.com/p2VxZcR.png" width="70%">
</p>

## Содержание

1. [Введение в позиционное кодирование](#введение-в-позиционное-кодирование)
    * [Почему важна позиция](#почему-важна-позиция)
    * [Проблемы временных рядов](#проблемы-временных-рядов)
    * [Типы позиционного кодирования](#типы-позиционного-кодирования)
2. [Синусоидальное позиционное кодирование](#синусоидальное-позиционное-кодирование)
    * [Математические основы](#математические-основы)
    * [Реализация](#реализация-синусоидального)
    * [Адаптация для временных рядов](#адаптация-для-временных-рядов)
3. [Обучаемое позиционное кодирование](#обучаемое-позиционное-кодирование)
    * [Обучаемые эмбеддинги](#обучаемые-эмбеддинги)
    * [Преимущества для финансовых данных](#преимущества-для-финансовых-данных)
4. [Относительное позиционное кодирование](#относительное-позиционное-кодирование)
    * [Относительное внимание Шоу](#относительное-внимание-шоу)
    * [Кодирование в стиле XLNet](#кодирование-в-стиле-xlnet)
5. [Вращательное позиционное кодирование (RoPE)](#вращательное-позиционное-кодирование-rope)
    * [Математическая формулировка](#математическая-формулировка-rope)
    * [RoPE для временных рядов](#rope-для-временных-рядов)
6. [Временные кодировки для финансов](#временные-кодировки-для-финансов)
    * [Календарные признаки](#календарные-признаки)
    * [Кодирование торговых сессий](#кодирование-торговых-сессий)
    * [Многомасштабное временное кодирование](#многомасштабное-временное-кодирование)
7. [Практические примеры](#практические-примеры)
8. [Реализация на Rust](#реализация-на-rust)
9. [Реализация на Python](#реализация-на-python)
10. [Лучшие практики](#лучшие-практики)
11. [Ресурсы](#ресурсы)

## Введение в позиционное кодирование

Трансформеры обрабатывают последовательности без врождённого понимания порядка. В отличие от RNN, которые обрабатывают токены последовательно, механизм self-attention обрабатывает все позиции одинаково. **Позиционное кодирование** внедряет информацию о позиции в модель.

### Почему важна позиция

Для временных рядов позиция несёт критически важную информацию:

```
Без позиции:   [100, 105, 103, 108, 102] = Значения цены (неупорядоченное множество)
С позицией:    t=1: 100 → t=2: 105 → t=3: 103 → t=4: 108 → t=5: 102

Последовательность рассказывает историю:
- Цена ВЫРОСЛА со 100 до 105 (+5%)
- Затем УПАЛА до 103 (-2%)
- Затем ВЫРОСЛА до 108 (+5%)
- Затем УПАЛА до 102 (-6%)
```

**Позиция определяет смысл**:
- `[100, 105]` = бычий тренд (цена растёт)
- `[105, 100]` = медвежий тренд (цена падает)

### Проблемы временных рядов

Финансовые временные ряды имеют уникальные характеристики:

| Проблема | Описание | Решение |
|----------|----------|---------|
| Переменная длина | Разные горизонты прогнозирования | Относительные кодировки |
| Множественные масштабы | Минуты, часы, дни, недели | Многомасштабное кодирование |
| Периодичность | Дневные/недельные/месячные паттерны | Синусоидальное кодирование |
| Нестационарность | Смена рыночных режимов | Обучаемое + контекстное кодирование |
| Пропущенные данные | Праздники, разрывы | Маскированное позиционное кодирование |

### Типы позиционного кодирования

```
┌────────────────────────────────────────────────────────────────┐
│              ТИПЫ ПОЗИЦИОННОГО КОДИРОВАНИЯ                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. СИНУСОИДАЛЬНОЕ (Фиксированное)                             │
│     ├── Нет обучаемых параметров                               │
│     ├── Экстраполирует на неизвестные длины                    │
│     └── PE(pos, 2i) = sin(pos / 10000^(2i/d))                 │
│                                                                 │
│  2. ОБУЧАЕМОЕ (Trainable)                                      │
│     ├── Таблица эмбеддингов для каждой позиции                 │
│     ├── Адаптируется к паттернам данных                        │
│     └── Ограничено длиной обучающей последовательности         │
│                                                                 │
│  3. ОТНОСИТЕЛЬНОЕ (Shaw, T5)                                   │
│     ├── Кодирует расстояние между токенами                     │
│     ├── Хорошо для разных длин последовательностей             │
│     └── att(i,j) зависит от (i-j)                             │
│                                                                 │
│  4. ВРАЩАТЕЛЬНОЕ (RoPE)                                        │
│     ├── Вращает векторы query/key                              │
│     ├── Относительная позиция через вращение                   │
│     └── Используется в LLaMA, GPT-NeoX                        │
│                                                                 │
│  5. ВРЕМЕННОЕ (Специфичное для временных рядов)                │
│     ├── Календарные признаки (день, месяц, год)                │
│     ├── Индикаторы торговых сессий                             │
│     └── Мультичастотные компоненты                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Синусоидальное позиционное кодирование

Оригинальный Transformer использует синусоидальные функции для кодирования позиции:

### Математические основы

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Где:
- pos = позиция в последовательности (0, 1, 2, ...)
- i = индекс измерения (0, 1, ..., d_model/2 - 1)
- d_model = размерность эмбеддинга
```

**Почему синус/косинус?**
1. **Ограниченные значения**: Всегда между [-1, 1]
2. **Уникальность**: Каждая позиция имеет уникальную кодировку
3. **Относительная позиция**: `PE(pos+k)` может быть представлена как линейная функция от `PE(pos)`
4. **Экстраполяция**: Работает для последовательностей длиннее обучающих

### Реализация синусоидального кодирования

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Стандартное синусоидальное позиционное кодирование из 'Attention Is All You Need'

    Для временных рядов расширяем:
    - Опциональное масштабирование для разных временных шкал
    - Параметр температуры для контроля частоты
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Создаём матрицу позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Частотные термы
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(temperature) / d_model)
        )

        # Чередуем sin и cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Регистрируем как буфер (не параметр)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор [batch, seq_len, d_model]
        Returns:
            Тензор с позиционным кодированием [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Адаптация для временных рядов

Для финансовых временных рядов адаптируем синусоидальное кодирование:

```python
class TimeSeriesSinusoidalEncoding(nn.Module):
    """
    Синусоидальное кодирование, адаптированное для временных рядов
    с множественными частотами

    Захватывает:
    - Внутридневные паттерны (часовые циклы)
    - Дневные паттерны (открытие/закрытие рынка)
    - Недельные паттерны (эффект понедельника)
    - Месячные паттерны (ребалансировка в конце месяца)
    """

    def __init__(
        self,
        d_model: int,
        frequencies: list = [24, 24*7, 24*30, 24*365],  # Периоды для часовых данных
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.frequencies = frequencies
        self.dropout = nn.Dropout(p=dropout)

        # Распределяем измерения по частотам
        dims_per_freq = d_model // (len(frequencies) * 2)
        self.dims_per_freq = dims_per_freq

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Входной тензор [batch, seq_len, d_model]
            timestamps: Опциональные абсолютные временные метки [batch, seq_len]
        Returns:
            Тензор с позиционным кодированием [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        if timestamps is None:
            # Используем последовательные позиции
            positions = torch.arange(seq_len, device=device).float()
        else:
            positions = timestamps.float()

        pe = torch.zeros(seq_len, d_model, device=device)

        dim_idx = 0
        for freq in self.frequencies:
            # Нормализуем позицию к периоду частоты
            pos_normalized = positions / freq

            for i in range(self.dims_per_freq):
                # Множественные гармоники на каждую частоту
                harmonic = 2 ** i
                pe[:, dim_idx] = torch.sin(2 * math.pi * harmonic * pos_normalized)
                pe[:, dim_idx + 1] = torch.cos(2 * math.pi * harmonic * pos_normalized)
                dim_idx += 2

        x = x + pe.unsqueeze(0)
        return self.dropout(x)
```

## Обучаемое позиционное кодирование

Вместо фиксированных функций — обучаем позиционные эмбеддинги на данных:

### Обучаемые эмбеддинги

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование с использованием таблицы эмбеддингов

    Преимущества:
    - Адаптируется к специфичным паттернам датасета
    - Может обучать асимметричные зависимости
    - Хорошо работает с последовательностями фиксированной длины

    Недостатки:
    - Не может экстраполировать за пределы обучающей длины
    - Больше параметров для оптимизации
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        init_std: float = 0.02
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Инициализируем малыми значениями
        nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор [batch, seq_len, d_model]
        Returns:
            Тензор с позиционным кодированием [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)  # [seq_len, d_model]

        x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)
```

### Преимущества для финансовых данных

Обучаемые кодировки могут захватывать:
- **Смещение к недавнему**: Недавние цены важнее
- **Асимметричный lookback**: Разные веса для разных лагов
- **Нелинейное затухание**: Кастомные паттерны внимания во времени

## Относительное позиционное кодирование

Кодирует *расстояние* между позициями, а не абсолютные позиции:

### Относительное внимание Шоу

```python
class RelativePositionalEncoding(nn.Module):
    """
    Относительное позиционное кодирование от Shaw et al.

    Вместо добавления позиции к входу, модифицируем скоры внимания:
    attention(Q, K) = softmax((Q @ K^T + Q @ R^T) / sqrt(d))

    Где R — эмбеддинг относительной позиции
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_relative_position = max_relative_position

        # Эмбеддинги относительной позиции: [-max_pos, ..., 0, ..., max_pos]
        self.relative_embedding = nn.Embedding(
            2 * max_relative_position + 1,
            self.head_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q: Query тензор [batch, n_heads, seq_len, head_dim]
            k: Key тензор [batch, n_heads, seq_len, head_dim]
            v: Value тензор [batch, n_heads, seq_len, head_dim]
        Returns:
            Выход внимания [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, head_dim = q.shape
        device = q.device

        # Стандартные QK скоры внимания
        qk_scores = torch.matmul(q, k.transpose(-2, -1))

        # Вычисляем индексы относительных позиций
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.clamp(
            -self.max_relative_position,
            self.max_relative_position
        )
        relative_positions = relative_positions + self.max_relative_position

        # Получаем относительные эмбеддинги
        rel_emb = self.relative_embedding(relative_positions)

        # Вычисляем относительные скоры внимания: Q @ R^T
        q_expanded = q.unsqueeze(3)
        rel_emb_expanded = rel_emb.unsqueeze(0).unsqueeze(0)

        relative_scores = torch.matmul(
            q_expanded,
            rel_emb_expanded.transpose(-2, -1)
        ).squeeze(-2)

        # Комбинируем скоры
        scores = (qk_scores + relative_scores) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применяем внимание к values
        output = torch.matmul(attn_weights, v)

        return output
```

## Вращательное позиционное кодирование (RoPE)

RoPE кодирует позицию путём вращения векторов query и key:

### Математическая формулировка RoPE

```
Для query q и key k на позициях m и n:
RoPE(q, m) = R_m @ q
RoPE(k, n) = R_n @ k

Где R — матрица вращения:
R_m = [cos(mθ₁)  -sin(mθ₁)    0        0     ...
       sin(mθ₁)   cos(mθ₁)    0        0     ...
         0          0      cos(mθ₂) -sin(mθ₂) ...
         0          0      sin(mθ₂)  cos(mθ₂) ...
        ...       ...       ...      ...     ...]

Результат: (R_m @ q)^T @ (R_n @ k) = q^T @ R_{n-m} @ k

Скор внимания зависит от относительной позиции (n-m)!
```

### RoPE для временных рядов

```python
class RotaryPositionalEncoding(nn.Module):
    """
    Вращательное позиционное кодирование (RoPE) для временных рядов

    Ключевая идея: Вращаем векторы query/key на зависящий от позиции угол
    Результат: Скоры внимания естественно кодируют относительную позицию

    Преимущества для временных рядов:
    - Эффективно обрабатывает длинные последовательности
    - Естественное затухание для далёких позиций
    - Работает с последовательностями переменной длины
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 8192,
        base: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_len = max_len
        self.base = base

        # Предвычисляем частоты
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer('inv_freq', inv_freq)

        # Предвычисляем кэш sin/cos
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Предвычисляем значения cos и sin"""
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)

        # Стекаем sin и cos для вращения
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Вращаем половину скрытых измерений"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor = None
    ) -> tuple:
        """
        Применяем вращательные эмбеддинги к queries и keys

        Args:
            q: Query тензор [batch, n_heads, seq_len, head_dim]
            k: Key тензор [batch, n_heads, seq_len, head_dim]
            positions: Опциональные индексы позиций [batch, seq_len]

        Returns:
            Кортеж повёрнутых (query, key) тензоров
        """
        batch, n_heads, seq_len, head_dim = q.shape

        if positions is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[positions].unsqueeze(1)
            sin = self.sin_cached[positions].unsqueeze(1)

        # Применяем вращение
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot
```

## Временные кодировки для финансов

Специализированные кодировки, захватывающие паттерны финансовых рынков:

### Календарные признаки

```python
class CalendarEncoding(nn.Module):
    """
    Кодирование календарных признаков, важных для финансовых рынков

    Признаки:
    - День недели (эффект понедельника)
    - Месяц (эффект января, ребалансировка в конце месяца)
    - Квартал (сезоны отчётности)
    - Год (для осознания режимов)
    - Торговая сессия (рабочие часы)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # Распределение измерений
        self.d_dayofweek = d_model // 8
        self.d_month = d_model // 8
        self.d_quarter = d_model // 16
        self.d_hour = d_model // 8
        self.d_session = d_model // 16

        # Эмбеддинги
        self.dayofweek_emb = nn.Embedding(7, self.d_dayofweek)      # Пн-Вс
        self.month_emb = nn.Embedding(12, self.d_month)             # Янв-Дек
        self.quarter_emb = nn.Embedding(4, self.d_quarter)          # Q1-Q4
        self.hour_emb = nn.Embedding(24, self.d_hour)               # 0-23
        self.session_emb = nn.Embedding(4, self.d_session)          # Пре/Основная/Пост/Закрыто

        # Проекция в размерность модели
        total_dim = (self.d_dayofweek + self.d_month + self.d_quarter +
                     self.d_hour + self.d_session)
        self.proj = nn.Linear(total_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dayofweek: torch.Tensor,
        month: torch.Tensor,
        quarter: torch.Tensor,
        hour: torch.Tensor,
        session: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Все входы: [batch, seq_len]
        Returns:
            Календарное кодирование [batch, seq_len, d_model]
        """
        dow = self.dayofweek_emb(dayofweek)
        mon = self.month_emb(month)
        qtr = self.quarter_emb(quarter)
        hr = self.hour_emb(hour)
        ses = self.session_emb(session)

        combined = torch.cat([dow, mon, qtr, hr, ses], dim=-1)
        out = self.proj(combined)

        return self.dropout(out)
```

### Кодирование торговых сессий

```python
class MarketSessionEncoding(nn.Module):
    """
    Кодирование информации о торговых сессиях

    Для крипто (24/7): Паттерны времени суток
    Для акций: Пре-маркет, основная сессия, после закрытия, закрыто
    """

    def __init__(
        self,
        d_model: int,
        market_type: str = 'crypto',  # 'crypto' или 'stock'
        dropout: float = 0.1
    ):
        super().__init__()
        self.market_type = market_type

        if market_type == 'crypto':
            # 24-часовой цикл с Азиатской/Европейской/Американской сессиями
            self.session_emb = nn.Embedding(3, d_model // 3)  # Азия/Европа/США
            self.hour_emb = nn.Embedding(24, d_model // 3)
            self.proj = nn.Linear(2 * (d_model // 3), d_model)
        else:
            # Сессии фондового рынка
            self.session_emb = nn.Embedding(4, d_model // 2)
            self.time_in_session = nn.Embedding(100, d_model // 2)
            self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _get_crypto_session(self, hour: torch.Tensor) -> torch.Tensor:
        """Отображение часа в крипто торговую сессию"""
        # Азия: 0-8 UTC, Европа: 8-16 UTC, США: 16-24 UTC
        session = torch.zeros_like(hour)
        session[(hour >= 0) & (hour < 8)] = 0   # Азия
        session[(hour >= 8) & (hour < 16)] = 1  # Европа
        session[(hour >= 16) & (hour < 24)] = 2 # США
        return session

    def forward(
        self,
        hour: torch.Tensor,
        session: torch.Tensor = None,
        time_in_session: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hour: Час дня [batch, seq_len]
            session: Торговая сессия (для акций) [batch, seq_len]
            time_in_session: Минуты в сессии [batch, seq_len]
        """
        if self.market_type == 'crypto':
            crypto_session = self._get_crypto_session(hour)
            ses_emb = self.session_emb(crypto_session)
            hr_emb = self.hour_emb(hour)
            combined = torch.cat([ses_emb, hr_emb], dim=-1)
        else:
            ses_emb = self.session_emb(session)
            time_emb = self.time_in_session(time_in_session.clamp(0, 99))
            combined = torch.cat([ses_emb, time_emb], dim=-1)

        out = self.proj(combined)
        return self.dropout(out)
```

### Многомасштабное временное кодирование

```python
class MultiScaleTemporalEncoding(nn.Module):
    """
    Кодирование времени на множественных масштабах для комплексного
    временного представления

    Масштабы:
    - Микро: Внутри торговой сессии (минуты)
    - Внутридневной: Часы внутри дня
    - Дневной: Паттерны дней
    - Недельный: Паттерны недель
    - Месячный: Паттерны месяцев
    """

    def __init__(
        self,
        d_model: int,
        time_scales: list = ['minute', 'hour', 'day', 'week', 'month'],
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_scales = time_scales

        # Измерение на масштаб
        d_per_scale = d_model // len(time_scales)
        self.d_per_scale = d_per_scale

        # Эмбеддинги для каждого масштаба
        self.scale_embeddings = nn.ModuleDict()
        scale_sizes = {
            'minute': 60,
            'hour': 24,
            'day': 31,
            'week': 7,
            'month': 12
        }

        for scale in time_scales:
            self.scale_embeddings[scale] = nn.Embedding(
                scale_sizes[scale],
                d_per_scale
            )

        # Финальная проекция
        self.proj = nn.Linear(d_per_scale * len(time_scales), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, timestamps: dict) -> torch.Tensor:
        """
        Args:
            timestamps: Dict с ключами для каждого масштаба
                       например, {'minute': [batch, seq], 'hour': [batch, seq], ...}
        Returns:
            Многомасштабное временное кодирование [batch, seq_len, d_model]
        """
        embeddings = []

        for scale in self.time_scales:
            if scale in timestamps:
                emb = self.scale_embeddings[scale](timestamps[scale])
                embeddings.append(emb)
            else:
                batch, seq_len = next(iter(timestamps.values())).shape
                embeddings.append(
                    torch.zeros(batch, seq_len, self.d_per_scale,
                               device=next(iter(timestamps.values())).device)
                )

        combined = torch.cat(embeddings, dim=-1)
        out = self.proj(combined)

        return self.dropout(out)
```

## Практические примеры

См. папку [python/examples/](python/examples/) для полных примеров:

1. **01_compare_encodings.py** — Сравнение методов кодирования
2. **02_crypto_prediction.py** — Предсказание цен криптовалют (Bybit)
3. **03_stock_prediction.py** — Прогнозирование фондового рынка
4. **04_backtesting.py** — Бэктестинг торговых стратегий

## Реализация на Rust

См. [rust_positional_encoding](rust_positional_encoding/) для полной реализации на Rust.

```
rust_positional_encoding/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты
│   ├── api/                # Клиент Bybit API
│   ├── data/               # Обработка данных
│   ├── encoding/           # Реализации позиционного кодирования
│   ├── model/              # Модель трансформера
│   └── strategy/           # Торговая стратегия
└── examples/
    ├── compare_encodings.rs
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust_positional_encoding

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h

# Сравнить методы кодирования
cargo run --example compare_encodings

# Обучить модель
cargo run --example train -- --epochs 100 --encoding rope

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

```
python/
├── positional_encoding.py  # Все реализации кодирования
├── model.py                # Модель трансформера
├── data.py                 # Загрузка данных Bybit
├── strategy.py             # Торговая стратегия
├── train.py                # Скрипт обучения
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_compare_encodings.py
    ├── 02_crypto_prediction.py
    ├── 03_stock_prediction.py
    └── 04_backtesting.py
```

### Быстрый старт (Python)

```bash
# Установить зависимости
pip install -r requirements.txt

# Сравнить методы кодирования
python examples/01_compare_encodings.py

# Обучить модель предсказания криптовалют
python train.py --symbol BTCUSDT --encoding rope

# Запустить бэктест
python examples/04_backtesting.py
```

## Лучшие практики

### Выбор метода кодирования

| Случай использования | Рекомендуемое кодирование | Причина |
|---------------------|--------------------------|---------|
| Последовательности фиксированной длины | Синусоидальное или обучаемое | Просто и эффективно |
| Последовательности переменной длины | RoPE или относительное | Работает с разными длинами |
| Длинные последовательности (>512) | RoPE | Лучшая экстраполяция |
| Зависящие от календаря паттерны | Календарное + Синусоидальное | Захватывает рыночные эффекты |
| Крипто рынки 24/7 | RoPE + Сессионное | Осознание непрерывного времени |
| Фондовые рынки | Календарное + Сессия рынка | Паттерны торговых часов |

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуется | Заметки |
|----------|--------------|---------|
| `d_model` | 64-256 | Больше для сложных паттернов |
| `dropout` | 0.1-0.2 | Выше для малых датасетов |
| `max_len` | 2x длины обучения | Позволяет экстраполяцию |
| `temperature` (синусоидальное) | 10000 | Стандартное значение |
| `base` (RoPE) | 10000 | Можно настроить для более длинных последовательностей |

### Типичные ошибки

1. **Обучаемые эмбеддинги фиксированной длины**: Не могут экстраполировать
2. **Игнорирование календарных признаков**: Пропуск рыночных паттернов
3. **Избыточная инженерия**: Простое синусоидальное часто достаточно
4. **Не масштабирование позиций**: Нормализуйте для длинных последовательностей

## Ресурсы

### Научные работы

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальный Transformer с синусоидальным кодированием
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Статья о RoPE
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — Относительное кодирование Шоу
- [Transformer-XL](https://arxiv.org/abs/1901.02860) — Сегментная рекуррентность
- [Informer](https://arxiv.org/abs/2012.07436) — Трансформеры для временных рядов

### Реализации

- [PyTorch Transformers](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [x-transformers](https://github.com/lucidrains/x-transformers)

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers)
- [Глава 43: Stockformer Multivariate](../43_stockformer_multivariate)
- [Глава 47: Cross-Attention Multi-Asset](../47_cross_attention_multi_asset)
- [Глава 51: Linformer Long Sequences](../51_linformer_long_sequences)

---

## Уровень сложности

**Средний — Продвинутый**

Предварительные требования:
- Основы архитектуры Transformer
- Механизм self-attention
- Базовые знания временных рядов
- Библиотеки ML на PyTorch/Rust
