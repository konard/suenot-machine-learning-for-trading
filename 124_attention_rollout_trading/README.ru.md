# Глава 124: Attention Rollout для Трейдинга

## Введение

Attention Rollout — это мощный метод интерпретируемости для моделей на основе трансформеров, который отслеживает поток внимания через несколько слоёв. В торговых приложениях этот метод помогает объяснить, почему модель принимает конкретные решения, позволяя трейдерам понять, какие исторические паттерны, временные периоды или признаки больше всего влияют на сигналы покупки/продажи.

В этой главе рассматриваются теоретические основы attention rollout, его математическая формулировка и практические реализации на Python и Rust для анализа финансовых рынков.

## Содержание

1. [Теоретические основы](#теоретические-основы)
2. [Математическая формулировка](#математическая-формулировка)
3. [Алгоритм Attention Rollout](#алгоритм-attention-rollout)
4. [Применение в трейдинге](#применение-в-трейдинге)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры](#практические-примеры)
8. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
9. [Продвинутые темы](#продвинутые-темы)
10. [Список литературы](#список-литературы)

---

## Теоретические основы

### Что такое Attention Rollout?

Attention Rollout, представленный Abnar & Zuidema (2020), — это метод количественной оценки потока информации в архитектурах трансформеров. В отличие от сырых весов внимания, которые показывают только паттерны конкретного слоя, attention rollout рекурсивно комбинирует матрицы внимания по всем слоям, чтобы выявить совокупное влияние входных токенов на конечный результат.

### Почему Attention Rollout для трейдинга?

Традиционные модели типа "чёрный ящик" несут значительные риски в финансовых приложениях:

1. **Соответствие нормативам**: Финансовые организации должны объяснять решения моделей
2. **Управление рисками**: Понимание причин прогноза краха критически важно
3. **Валидация стратегии**: Подтверждение использования разумных рыночных индикаторов
4. **Отладка**: Выявление случаев, когда модели полагаются на ложные корреляции

```
┌─────────────────────────────────────────────────────────────────┐
│                   ТРАНСФОРМЕР ДЛЯ ТРЕЙДИНГА                     │
├─────────────────────────────────────────────────────────────────┤
│  Вход: [Цена_t-5, Цена_t-4, Цена_t-3, Цена_t-2, Цена_t-1]      │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Внимание слоя 1: Какие прошлые цены важны?             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Внимание слоя 2: Уточнённое распознавание паттернов    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Внимание слоя N: Финальное взвешивание решения         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Выход: Сигнал ПОКУПКА/ПРОДАЖА + Объяснение Attention Rollout  │
└─────────────────────────────────────────────────────────────────┘
```

### Сравнение с другими методами интерпретируемости

| Метод | Преимущества | Недостатки | Лучше всего для |
|-------|--------------|------------|-----------------|
| **Attention Rollout** | Улавливает многослойный поток | Предполагает линейную комбинацию | Моделей последовательностей |
| **Attention Flow** | Теоретико-графовое обоснование | Высокая вычислительная сложность | Глубокого анализа |
| **Градиентные методы** | Не зависят от модели | Могут быть шумными | Любых дифференцируемых моделей |
| **SHAP** | Теоретическое обоснование | Медленно для трансформеров | Важности признаков |
| **LIME** | Локальные объяснения | Ошибки аппроксимации | Уровня экземпляров |

---

## Математическая формулировка

### Однонаправленное внимание

Для одной головы внимания веса вычисляются как:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Где:
- Q: Матрица запросов (n × d_k)
- K: Матрица ключей (n × d_k)
- V: Матрица значений (n × d_v)
- d_k: Размерность ключей

Матрица весов внимания A:

```
A = softmax(QK^T / √d_k)
```

### Агрегация многоголового внимания

Для многоголового внимания с h головами агрегируем веса:

```
A_combined = (1/h) Σ_{i=1}^{h} A_i
```

Или с использованием взвешивания по важности голов:

```
A_combined = Σ_{i=1}^{h} w_i · A_i,  где Σw_i = 1
```

### Формула Attention Rollout

Ключевая идея attention rollout — учёт остаточных соединений. На каждом слое эффективное внимание становится:

```
Ã_l = 0.5 · I + 0.5 · A_l
```

Где I — единичная матрица (представляющая остаточное соединение).

Матрица rollout R после L слоёв вычисляется рекурсивно:

```
R_1 = Ã_1
R_l = Ã_l · R_{l-1}  для l = 2, ..., L
```

Финальная матрица rollout R_L показывает совокупное внимание от каждой входной позиции к выходу.

### Нормализованный Rollout

Для обеспечения корректного распределения вероятностей:

```
R̂_L = R_L / Σ_j R_L[i,j]
```

Каждая строка суммируется до 1, представляя распределение внимания.

---

## Алгоритм Attention Rollout

### Псевдокод алгоритма

```
Алгоритм: Attention Rollout
Вход: Матрицы внимания A_1, A_2, ..., A_L из L слоёв
Выход: Матрица rollout R, показывающая поток внимания от входа к выходу

1. Инициализация: R ← I (единичная матрица)
2. Для l = 1 до L:
   a. Если многоголовое: A_l ← mean(A_l, axis=heads)
   b. Добавить остаточное: Ã_l ← 0.5 · I + 0.5 · A_l
   c. Накопить: R ← Ã_l · R
3. Нормализовать строки: R ← R / row_sum(R)
4. Вернуть R
```

### Вычислительная сложность

- Время: O(L · n²) для L слоёв и длины последовательности n
- Память: O(n²) для хранения матриц внимания

Для трейдинга с типичными длинами последовательностей (50-200 временных шагов) это высокоэффективно.

---

## Применение в трейдинге

### 1. Атрибуция признаков для прогнозирования цен

Понимание того, какие исторические цены влияют на прогнозы:

```
Входная последовательность: [День-10, День-9, День-8, ..., День-1, День-0]
                              ↓        ↓       ↓            ↓       ↓
Веса Rollout:               [0.05,   0.08,   0.15,  ...,  0.25,   0.20]

Интерпретация: Дни -1 и -2 имеют наибольшее влияние на прогноз
```

### 2. Анализ внимания для множества активов

Для портфельных моделей, обрабатывающих несколько активов:

```
┌─────────────────────────────────────────────────┐
│  Активы: [AAPL, GOOGL, MSFT, AMZN, TSLA]       │
│                                                 │
│  Attention Rollout для прогноза AAPL:          │
│  AAPL  ████████████████████  0.35              │
│  GOOGL ██████████            0.20              │
│  MSFT  ████████              0.18              │
│  AMZN  ██████                0.15              │
│  TSLA  ██████                0.12              │
│                                                 │
│  → Прогноз AAPL сильно зависит от              │
│    себя и аналогов в технологическом секторе   │
└─────────────────────────────────────────────────┘
```

### 3. Обнаружение временных паттернов

Определение важных временных окон:

```
Определение рыночного режима через Attention Rollout:

Бычий рынок: Внимание сконцентрировано на недавнем импульсе
  [0.05, 0.08, 0.12, 0.20, 0.25, 0.30] → Смещение к недавнему

Медвежий рынок: Внимание распределено по истории
  [0.15, 0.18, 0.17, 0.16, 0.18, 0.16] → Равномерное внимание

Всплеск волатильности: Внимание на конкретных событиях
  [0.05, 0.40, 0.05, 0.05, 0.40, 0.05] → Фокус на событиях
```

---

## Реализация на Python

### Зависимости

```python
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
transformers>=4.10.0
yfinance>=0.1.70
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
requests>=2.26.0
```

### Основной модуль Attention Rollout

```python
"""
attention_rollout.py - Реализация Attention Rollout для трейдинга
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn


class AttentionRollout:
    """
    Вычисление attention rollout для моделей трансформеров.

    Attention rollout отслеживает поток внимания через слои трансформера,
    предоставляя интерпретируемые объяснения для прогнозов модели.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_name: str = "attn",
        head_fusion: str = "mean",
        discard_ratio: float = 0.0
    ):
        """
        Инициализация AttentionRollout.

        Args:
            model: Модель трансформера PyTorch
            attention_layer_name: Шаблон имени для слоёв внимания
            head_fusion: Метод объединения голов ('mean', 'max', 'min')
            discard_ratio: Доля наименьших весов внимания для отбрасывания
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions: List[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Регистрация хуков для захвата весов внимания."""
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                module.register_forward_hook(self._attention_hook)

    def _attention_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: Tuple
    ) -> None:
        """Функция хука для захвата весов внимания."""
        if isinstance(output, tuple):
            attention = output[1] if len(output) > 1 else output[0]
        else:
            attention = output
        self.attentions.append(attention.detach())

    def _fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Объединение нескольких голов внимания в одну матрицу.

        Args:
            attention: Тензор формы (batch, heads, seq_len, seq_len)

        Returns:
            Объединённое внимание формы (batch, seq_len, seq_len)
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Неизвестный метод объединения голов: {self.head_fusion}")

    def compute_rollout(
        self,
        input_tensor: torch.Tensor,
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Вычисление attention rollout для данного входа.

        Args:
            input_tensor: Входной тензор для модели
            start_layer: Слой для начала вычисления rollout

        Returns:
            Матрица rollout формы (seq_len, seq_len)
        """
        self.attentions = []

        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attentions:
            raise RuntimeError("Веса внимания не захвачены. Проверьте имя слоя.")

        batch_size = self.attentions[0].shape[0]
        seq_len = self.attentions[0].shape[-1]

        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = rollout.to(self.attentions[0].device)

        for attention in self.attentions[start_layer:]:
            attention = self._fuse_heads(attention)
            identity = torch.eye(seq_len).unsqueeze(0).to(attention.device)
            attention = 0.5 * attention + 0.5 * identity
            rollout = torch.bmm(attention, rollout)

        rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        return rollout.cpu().numpy()

    def get_input_attribution(
        self,
        input_tensor: torch.Tensor,
        output_position: int = -1
    ) -> np.ndarray:
        """
        Получение оценок атрибуции для входных позиций.

        Args:
            input_tensor: Входной тензор
            output_position: Позиция для получения атрибуции (-1 для последней)

        Returns:
            Оценки атрибуции для каждой входной позиции
        """
        rollout = self.compute_rollout(input_tensor)
        attribution = rollout[0, output_position, :]
        return attribution


class TradingAttentionRollout(AttentionRollout):
    """
    Специализированный attention rollout для торговых приложений.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.feature_names = feature_names

    def analyze_temporal_importance(
        self,
        input_tensor: torch.Tensor,
        timestamps: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Анализ важности временных периодов для прогноза."""
        attribution = self.get_input_attribution(input_tensor)

        if timestamps is None:
            timestamps = [f"t-{i}" for i in range(len(attribution)-1, -1, -1)]

        return dict(zip(timestamps, attribution))

    def detect_attention_regime(
        self,
        input_tensor: torch.Tensor,
        threshold_recent: float = 0.6
    ) -> str:
        """
        Определение рыночного режима на основе паттерна внимания.

        Returns:
            Определённый режим: 'momentum', 'mean_reversion', или 'mixed'
        """
        attribution = self.get_input_attribution(input_tensor)
        seq_len = len(attribution)

        recent_window = seq_len // 4
        recent_attention = attribution[-recent_window:].sum()

        if recent_attention > threshold_recent:
            return "momentum"
        elif recent_attention < 1 - threshold_recent:
            return "mean_reversion"
        else:
            return "mixed"
```

### Модель трансформера для трейдинга

```python
"""
model.py - Модель трансформера для трейдинга с извлечением внимания
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование для последовательностей."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Многоголовое внимание с сохранением весов."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.out_linear(context)
        return output, self.attention_weights


class TradingTransformer(nn.Module):
    """
    Модель трансформера для торговых прогнозов с извлечением внимания.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        n_classes: int = 3
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        self.attention_maps: list = []

    def _make_layer(self, d_model, n_heads, d_ff, dropout):
        return nn.ModuleDict({
            'attn': MultiHeadAttention(d_model, n_heads, dropout),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model)
        })

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        self.attention_maps = []

        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)

        for layer in self.layers:
            attn_out, attn_weights = layer['attn'](x, x, x, mask)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
            self.attention_maps.append(attn_weights)

        x = x[:, -1, :]
        logits = self.classifier(x)

        return logits, self.attention_maps
```

### Модуль загрузки данных

```python
"""
data_loader.py - Утилиты загрузки данных для фондовых и криптовалютных рынков
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import requests


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Загрузка данных акций с использованием yfinance.

    Args:
        symbol: Тикер акции
        start_date: Дата начала (YYYY-MM-DD)
        end_date: Дата окончания (YYYY-MM-DD)
        interval: Интервал данных (1d, 1h и т.д.)

    Returns:
        DataFrame с данными OHLCV
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    return df


def load_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 200
) -> pd.DataFrame:
    """
    Загрузка данных криптовалют с биржи Bybit.

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал свечи (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit: Количество свечей (максимум 200)

    Returns:
        DataFrame с данными OHLCV
    """
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Ошибка API Bybit: {data['retMsg']}")

    candles = data["result"]["list"]

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    lookback: int = 20,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготовка признаков для модели трансформера.

    Args:
        df: DataFrame с данными OHLCV
        feature_columns: Столбцы для использования как признаки
        lookback: Количество временных шагов для просмотра назад
        normalize: Нормализовать ли признаки

    Returns:
        X: Последовательности признаков (n_samples, lookback, n_features)
        y: Целевые метки (n_samples,)
    """
    if feature_columns is None:
        feature_columns = ["open", "high", "low", "close", "volume"]

    df = add_technical_indicators(df)
    df["returns"] = df["close"].pct_change().shift(-1)

    df["label"] = 1  # hold
    df.loc[df["returns"] > 0.01, "label"] = 2  # buy
    df.loc[df["returns"] < -0.01, "label"] = 0  # sell

    if normalize:
        for col in feature_columns:
            if col in df.columns:
                df[col] = (df[col] - df[col].rolling(lookback).mean()) / (
                    df[col].rolling(lookback).std() + 1e-8
                )

    df = df.dropna()

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df[feature_columns].iloc[i-lookback:i].values)
        y.append(df["label"].iloc[i])

    return np.array(X), np.array(y)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление технических индикаторов в DataFrame."""

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    df["volatility"] = df["close"].pct_change().rolling(20).std()

    return df
```

### Модуль бэктестинга

```python
"""
backtest.py - Фреймворк бэктестинга с анализом внимания
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch


@dataclass
class BacktestResult:
    """Результаты бэктестинга торговой стратегии."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    returns: np.ndarray
    equity_curve: np.ndarray
    attention_analysis: Optional[Dict] = None


class AttentionBacktester:
    """
    Фреймворк бэктестинга с анализом attention rollout.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attention_rollout,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ):
        self.model = model
        self.attention_rollout = attention_rollout
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.6
    ) -> BacktestResult:
        """Запуск бэктеста на исторических данных."""
        self.model.eval()

        n_samples = len(X)
        returns = np.zeros(n_samples)
        attention_patterns = []

        capital = self.initial_capital
        equity_curve = [capital]
        current_position = 0
        n_trades = 0
        wins = 0

        for i in range(n_samples - 1):
            x_tensor = torch.FloatTensor(X[i:i+1])

            with torch.no_grad():
                logits, _ = self.model(x_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            attribution = self.attention_rollout.get_input_attribution(x_tensor)
            attention_patterns.append(attribution)

            pred_class = np.argmax(probs)
            confidence = probs[pred_class]

            if confidence > threshold:
                if pred_class == 2:
                    new_position = 1
                elif pred_class == 0:
                    new_position = -1
                else:
                    new_position = current_position
            else:
                new_position = 0

            price_return = (prices[i+1] - prices[i]) / prices[i]

            if new_position != current_position:
                n_trades += 1
                trade_cost = self.transaction_cost * abs(new_position - current_position)
            else:
                trade_cost = 0

            position_return = current_position * price_return - trade_cost
            returns[i] = position_return

            if position_return > 0:
                wins += 1

            capital *= (1 + position_return)
            equity_curve.append(capital)
            current_position = new_position

        equity_curve = np.array(equity_curve)

        return BacktestResult(
            total_return=(capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio=self._calculate_sharpe(returns),
            sortino_ratio=self._calculate_sortino(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=wins / max(n_trades, 1),
            n_trades=n_trades,
            returns=returns,
            equity_curve=equity_curve,
            attention_analysis=self._analyze_attention_patterns(attention_patterns, returns)
        )

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / downside.std()

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def _analyze_attention_patterns(self, attention_patterns: List, returns: np.ndarray) -> Dict:
        attention_matrix = np.array(attention_patterns)

        winning_mask = returns > 0
        losing_mask = returns < 0

        avg_winning = attention_matrix[winning_mask[:-1]].mean(axis=0)
        avg_losing = attention_matrix[losing_mask[:-1]].mean(axis=0)

        return {
            "avg_winning_attention": avg_winning.tolist(),
            "avg_losing_attention": avg_losing.tolist(),
            "recent_bias_winning": float(avg_winning[-5:].sum()),
            "recent_bias_losing": float(avg_losing[-5:].sum())
        }


def print_backtest_report(result: BacktestResult) -> None:
    """Вывод форматированного отчёта бэктестинга."""
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
    print("="*60)
    print(f"Общая доходность:     {result.total_return*100:>10.2f}%")
    print(f"Коэффициент Шарпа:    {result.sharpe_ratio:>10.2f}")
    print(f"Коэффициент Сортино:  {result.sortino_ratio:>10.2f}")
    print(f"Макс. просадка:       {result.max_drawdown*100:>10.2f}%")
    print(f"Доля прибыльных:      {result.win_rate*100:>10.2f}%")
    print(f"Количество сделок:    {result.n_trades:>10d}")
    print("="*60)
```

---

## Реализация на Rust

### Cargo.toml

```toml
[package]
name = "attention_rollout_trading"
version = "0.1.0"
edition = "2021"
description = "Attention Rollout для интерпретируемых торговых моделей"
license = "MIT"

[dependencies]
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
ndarray-stats = "0.5"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "blocking"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"

[[example]]
name = "trading_example"
path = "examples/trading_example.rs"
```

### Основная библиотека (src/lib.rs)

```rust
//! Библиотека Attention Rollout Trading
//!
//! Реализация вычисления attention rollout для интерпретируемых
//! торговых моделей на Rust для высокой производительности.

use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Ошибки при вычислении attention rollout
#[derive(Error, Debug)]
pub enum AttentionRolloutError {
    #[error("Передан пустой список внимания")]
    EmptyAttentionList,

    #[error("Несоответствие размерностей: ожидалось {expected}, получено {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Недопустимый индекс слоя: {0}")]
    InvalidLayerIndex(usize),

    #[error("Ошибка вычисления: {0}")]
    ComputationError(String),
}

/// Метод объединения нескольких голов внимания
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeadFusion {
    Mean,
    Max,
    Min,
}

/// Конфигурация для вычисления attention rollout
#[derive(Debug, Clone)]
pub struct AttentionRolloutConfig {
    pub head_fusion: HeadFusion,
    pub discard_ratio: f64,
    pub add_residual: bool,
    pub residual_weight: f64,
}

impl Default for AttentionRolloutConfig {
    fn default() -> Self {
        Self {
            head_fusion: HeadFusion::Mean,
            discard_ratio: 0.0,
            add_residual: true,
            residual_weight: 0.5,
        }
    }
}

/// Движок вычисления Attention Rollout
pub struct AttentionRollout {
    config: AttentionRolloutConfig,
}

impl AttentionRollout {
    pub fn new() -> Self {
        Self {
            config: AttentionRolloutConfig::default(),
        }
    }

    pub fn with_config(config: AttentionRolloutConfig) -> Self {
        Self { config }
    }

    fn fuse_heads(&self, attention: &Array3<f64>) -> Array2<f64> {
        match self.config.head_fusion {
            HeadFusion::Mean => attention.mean_axis(Axis(0)).unwrap(),
            HeadFusion::Max => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut max_val = f64::NEG_INFINITY;
                        for h in 0..attention.shape()[0] {
                            max_val = max_val.max(attention[[h, i, j]]);
                        }
                        result[[i, j]] = max_val;
                    }
                }
                result
            }
            HeadFusion::Min => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut min_val = f64::INFINITY;
                        for h in 0..attention.shape()[0] {
                            min_val = min_val.min(attention[[h, i, j]]);
                        }
                        result[[i, j]] = min_val;
                    }
                }
                result
            }
        }
    }

    pub fn compute_rollout(
        &self,
        attentions: &[Array3<f64>],
        start_layer: usize,
    ) -> Result<Array2<f64>, AttentionRolloutError> {
        if attentions.is_empty() {
            return Err(AttentionRolloutError::EmptyAttentionList);
        }

        if start_layer >= attentions.len() {
            return Err(AttentionRolloutError::InvalidLayerIndex(start_layer));
        }

        let seq_len = attentions[0].shape()[1];
        let mut rollout = Array2::eye(seq_len);

        for attention in attentions.iter().skip(start_layer) {
            let mut fused = self.fuse_heads(attention);

            if self.config.add_residual {
                let identity = Array2::eye(seq_len);
                let w = self.config.residual_weight;
                fused = &fused * (1.0 - w) + &identity * w;
            }

            rollout = fused.dot(&rollout);
        }

        for mut row in rollout.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(rollout)
    }

    pub fn get_input_attribution(
        &self,
        attentions: &[Array3<f64>],
        output_position: usize,
    ) -> Result<Array1<f64>, AttentionRolloutError> {
        let rollout = self.compute_rollout(attentions, 0)?;
        Ok(rollout.row(output_position).to_owned())
    }
}

/// Определённый режим внимания
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AttentionRegime {
    Momentum,
    MeanReversion,
    Mixed,
}

/// Анализатор attention rollout для трейдинга
pub struct TradingAttentionAnalyzer {
    rollout: AttentionRollout,
    momentum_threshold: f64,
}

impl TradingAttentionAnalyzer {
    pub fn new() -> Self {
        Self {
            rollout: AttentionRollout::new(),
            momentum_threshold: 0.6,
        }
    }

    pub fn analyze(
        &self,
        attentions: &[Array3<f64>],
    ) -> Result<(AttentionRegime, f64), AttentionRolloutError> {
        let seq_len = attentions[0].shape()[1];
        let attribution = self.rollout.get_input_attribution(attentions, seq_len - 1)?;

        let recent_window = seq_len / 4;
        let recent_bias: f64 = attribution
            .slice(ndarray::s![seq_len - recent_window..])
            .sum();

        let regime = if recent_bias > self.momentum_threshold {
            AttentionRegime::Momentum
        } else if recent_bias < 1.0 - self.momentum_threshold {
            AttentionRegime::MeanReversion
        } else {
            AttentionRegime::Mixed
        };

        Ok((regime, recent_bias))
    }
}

/// Модуль загрузки данных Bybit
pub mod data {
    use super::*;
    use reqwest::blocking::Client;

    #[derive(Debug, Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    struct BybitResult {
        list: Vec<Vec<String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    pub fn fetch_bybit_candles(
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, anyhow::Error> {
        let client = Client::new();
        let url = format!(
            "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            symbol, interval, limit
        );

        let response: BybitResponse = client.get(&url).send()?.json()?;

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|row| Candle {
                timestamp: row[0].parse().unwrap_or(0),
                open: row[1].parse().unwrap_or(0.0),
                high: row[2].parse().unwrap_or(0.0),
                low: row[3].parse().unwrap_or(0.0),
                close: row[4].parse().unwrap_or(0.0),
                volume: row[5].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(candles)
    }
}

/// Утилиты бэктестинга
pub mod backtest {
    pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / std
    }

    pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let downside_var: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside.len() as f64;
        let downside_std = downside_var.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / downside_std
    }

    pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_dd = 0.0;
        let mut peak = equity_curve[0];

        for &value in equity_curve.iter() {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}
```

---

## Практические примеры

### Пример 1: Прогнозирование фондового рынка

```python
"""
Пример: Использование attention rollout для интерпретации прогнозов акций
"""

import torch
import numpy as np
from attention_rollout import TradingAttentionRollout
from model import TradingTransformer
from data_loader import load_stock_data, prepare_features

# Загрузка данных
df = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
X, y = prepare_features(df, lookback=20)

# Инициализация модели
model = TradingTransformer(input_dim=X.shape[2], d_model=128, n_heads=8, n_layers=4)

# Инициализация attention rollout
rollout = TradingAttentionRollout(model, attention_layer_name="attn", head_fusion="mean")

# Получение прогноза и атрибуции
sample_idx = 100
x_sample = torch.FloatTensor(X[sample_idx:sample_idx+1])

with torch.no_grad():
    logits, _ = model(x_sample)
    prediction = torch.argmax(logits, dim=-1).item()

attribution = rollout.get_input_attribution(x_sample)

# Интерпретация результатов
labels = {0: "ПРОДАЖА", 1: "УДЕРЖАНИЕ", 2: "ПОКУПКА"}
print(f"Прогноз: {labels[prediction]}")
print(f"\nНаиболее влиятельные временные периоды:")
top_5_idx = np.argsort(attribution)[-5:][::-1]
for idx in top_5_idx:
    print(f"  t-{19-idx}: {attribution[idx]:.4f}")
```

### Пример 2: Криптотрейдинг с данными Bybit

```python
"""
Пример: Криптовалютный трейдинг с анализом внимания на Bybit
"""

from data_loader import load_bybit_data, prepare_features
from attention_rollout import TradingAttentionRollout
from model import TradingTransformer
from backtest import AttentionBacktester, print_backtest_report

# Загрузка данных Bybit
df = load_bybit_data(symbol="BTCUSDT", interval="60", limit=200)
print(f"Загружено {len(df)} часовых свечей")

# Подготовка признаков
X, y = prepare_features(df, lookback=24)
prices = df["close"].values[-len(X)-1:-1]

# Инициализация модели
model = TradingTransformer(input_dim=X.shape[2], d_model=64, n_heads=4, n_layers=3)

# Инициализация attention rollout
rollout = TradingAttentionRollout(model)

# Запуск бэктеста с анализом внимания
backtester = AttentionBacktester(
    model=model,
    attention_rollout=rollout,
    initial_capital=10000.0,
    transaction_cost=0.001
)

result = backtester.run_backtest(X, prices, threshold=0.6)
print_backtest_report(result)
```

---

## Список литературы

1. **Abnar, S., & Zuidema, W. (2020)**. "Quantifying Attention Flow in Transformers." *ACL 2020*. [https://arxiv.org/abs/2005.00928](https://arxiv.org/abs/2005.00928)

2. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *NeurIPS 2017*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

3. **Chefer, H., Gur, S., & Wolf, L. (2021)**. "Transformer Interpretability Beyond Attention Visualization." *CVPR 2021*. [https://arxiv.org/abs/2012.09838](https://arxiv.org/abs/2012.09838)

4. **Ding, Q., et al. (2020)**. "Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction." *IJCAI 2020*. [https://www.ijcai.org/proceedings/2020/640](https://www.ijcai.org/proceedings/2020/640)

5. **Zhang, L., et al. (2022)**. "Transformer-based Stock Trend Prediction with Attention Analysis." *Expert Systems with Applications*.

---

## Запуск примеров

### Настройка Python

```bash
cd 124_attention_rollout_trading/python
pip install -r requirements.txt

python -c "
from attention_rollout import AttentionRollout
from model import TradingTransformer
import torch

model = TradingTransformer(input_dim=5, d_model=64, n_heads=4, n_layers=2)
rollout = AttentionRollout(model, attention_layer_name='attn')

x = torch.randn(1, 20, 5)
attribution = rollout.get_input_attribution(x)
print('Оценки атрибуции:', attribution)
"
```

### Настройка Rust

```bash
cd 124_attention_rollout_trading/rust
cargo build --release
cargo run --example trading_example
```

---

## Заключение

Attention Rollout предоставляет мощный инструмент для понимания моделей трейдинга на основе трансформеров:

1. **Интерпретируемые прогнозы** — понимание причин сигналов покупки/продажи
2. **Управление рисками** — валидация областей фокуса модели перед торговлей
3. **Разработка стратегий** — обнаружение временных паттернов, которые изучает модель
4. **Отладка** — выявление случаев опоры на ложные корреляции

Комбинируя attention rollout с надёжным бэктестингом, трейдеры могут строить более надёжные и прибыльные алгоритмические торговые системы.
