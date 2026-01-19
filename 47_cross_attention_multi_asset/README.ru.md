# Глава 47: Cross-Attention для мультиактивной торговли

В этой главе рассматриваются **механизмы Cross-Attention** для моделирования взаимосвязей между несколькими финансовыми активами одновременно. В отличие от традиционного прогнозирования отдельных активов, cross-attention позволяет модели улавливать межактивные зависимости, корреляции и опережающе-запаздывающие связи, которые критически важны для управления портфелем и мультиактивных торговых стратегий.

<p align="center">
<img src="https://i.imgur.com/YqN3rZm.png" width="70%">
</p>

## Содержание

1. [Введение в Cross-Attention](#введение-в-cross-attention)
    * [Почему Cross-Attention для мультиактивной торговли?](#почему-cross-attention-для-мультиактивной-торговли)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими подходами](#сравнение-с-другими-подходами)
2. [Архитектура Cross-Attention](#архитектура-cross-attention)
    * [Механизм Query-Key-Value](#механизм-query-key-value)
    * [Многоголовый Cross-Attention](#многоголовый-cross-attention)
    * [Временной Cross-Attention](#временной-cross-attention)
    * [Иерархический Cross-Attention](#иерархический-cross-attention)
3. [Математические основы](#математические-основы)
    * [Вычисление оценок внимания](#вычисление-оценок-внимания)
    * [Cross-Attention vs Self-Attention](#cross-attention-vs-self-attention)
    * [Масштабированное скалярное произведение](#масштабированное-скалярное-произведение)
4. [Представление данных](#представление-данных)
    * [Мультиактивная инженерия признаков](#мультиактивная-инженерия-признаков)
    * [Данные фондового рынка](#данные-фондового-рынка)
    * [Данные криптовалютных рынков (Bybit)](#данные-криптовалютных-рынков-bybit)
5. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Модель Cross-Attention](#02-модель-cross-attention)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Мультиактивное прогнозирование](#04-мультиактивное-прогнозирование)
    * [05: Бэктестинг портфеля](#05-бэктестинг-портфеля)
6. [Реализация на Rust](#реализация-на-rust)
7. [Реализация на Python](#реализация-на-python)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в Cross-Attention

Cross-attention — это механизм внимания, при котором запросы (queries) поступают из одной последовательности (или актива), а ключи (keys) и значения (values) — из другой. В мультиактивной торговле это позволяет каждому активу «обращать внимание» на другие активы, обучаясь определять, какие активы предоставляют прогностическую информацию для других.

### Почему Cross-Attention для мультиактивной торговли?

Традиционные подходы обрабатывают каждый актив независимо:

```
Актив A → Модель_A → Прогноз_A
Актив B → Модель_B → Прогноз_B
Актив C → Модель_C → Прогноз_C
```

Cross-attention моделирует все активы совместно:

```
┌─────────────────────────────────────────────────┐
│           Cross-Attention Network                │
│                                                  │
│   Актив A ←→ Актив B ←→ Актив C                 │
│      ↑           ↑           ↑                   │
│      └───────────┴───────────┘                   │
│       Двунаправленное внимание                   │
│                                                  │
│                    ↓                             │
│   [Прогноз_A, Прогноз_B, Прогноз_C]             │
└─────────────────────────────────────────────────┘
```

**Ключевая идея**: Финансовые рынки взаимосвязаны. Когда Bitcoin движется, Ethereum часто следует за ним. Когда цены на нефть растут, акции авиакомпаний обычно падают. Cross-attention явно моделирует эти зависимости.

### Ключевые преимущества

1. **Обучение межактивным зависимостям**
   - Улавливает корреляции между различными классами активов
   - Моделирует опережающе-запаздывающие связи (например, BTC опережает альткоины)
   - Обучается изменяющимся во времени взаимосвязям

2. **Интерпретируемость на основе внимания**
   - Веса внимания показывают, какие активы влияют на прогнозы
   - Визуализация потока межактивной информации
   - Определение лидеров и последователей рынка

3. **Оптимизация на уровне портфеля**
   - Прямая оптимизация коэффициента Шарпа вместо отдельных прогнозов
   - Обучение оптимальным весам распределения активов
   - Учёт преимуществ диверсификации

4. **Адаптивное определение режимов**
   - Паттерны внимания меняются при разных рыночных режимах
   - Обнаружение разрыва корреляций во время кризисов
   - Адаптация к структурным изменениям рынка

### Сравнение с другими подходами

| Характеристика | Одноактивный LSTM | Мультиактивный RNN | Self-Attention | Cross-Attention |
|----------------|-------------------|---------------------|----------------|-----------------|
| Межактивное моделирование | ✗ | Ограниченно | Неявное | ✓ Явное |
| Двунаправленное влияние | ✗ | ✗ | ✓ | ✓ |
| Асимметричные связи | ✗ | ✗ | ✗ | ✓ |
| Определение lead-lag | ✗ | ✗ | Ограниченно | ✓ |
| Интерпретируемость | ✗ | ✗ | ✓ | ✓ |
| Оптимизация портфеля | ✗ | ✗ | ✗ | ✓ |

## Архитектура Cross-Attention

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    МУЛЬТИАКТИВНАЯ МОДЕЛЬ CROSS-ATTENTION                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ BTC     │  │ ETH     │  │ SOL     │  │ AAPL    │                     │
│  │ (Query) │  │ (Query) │  │ (Query) │  │ (Query) │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │            │                            │
│       ▼            ▼            ▼            ▼                            │
│  ┌──────────────────────────────────────────────────┐                    │
│  │            Слой Token Embedding                   │                    │
│  │    (1D-CNN или линейная проекция для каждого)    │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │          Временной Self-Attention                 │                    │
│  │  (Моделирование временных паттернов актива)      │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │      Межактивный Cross-Attention                  │                    │
│  │                                                   │                    │
│  │   Q(BTC) → K,V(ETH), K,V(SOL), K,V(AAPL)         │                    │
│  │   Q(ETH) → K,V(BTC), K,V(SOL), K,V(AAPL)         │                    │
│  │   ...                                            │                    │
│  │                                                   │                    │
│  │   Изучает: "BTC опережает ETH с весом 0.7"       │                    │
│  │            "ETH опережает SOL с весом 0.5"       │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │         Стек энкодеров (N слоёв)                  │                    │
│  │    Временной Attention + Межактивный Attention    │                    │
│  └───────────────────────┬──────────────────────────┘                    │
│                          │                                                │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │              Головы прогнозирования               │                    │
│  │   • Прогноз доходности (регрессия)               │                    │
│  │   • Прогноз направления (классификация)          │                    │
│  │   • Веса портфеля (softmax/tanh)                 │                    │
│  └──────────────────────────────────────────────────┘                    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Механизм Query-Key-Value

В cross-attention один актив генерирует запросы, а другие активы предоставляют ключи и значения:

```python
class CrossAssetAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_assets):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Отдельные проекции для каждой роли
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query_asset, key_value_assets):
        """
        Аргументы:
            query_asset: [batch, seq_len, d_model] - Актив для прогноза
            key_value_assets: [batch, n_other_assets, seq_len, d_model]

        Возвращает:
            context: [batch, seq_len, d_model] - Представление с вниманием
            attention: [batch, n_heads, seq_len, n_other_assets]
        """
        batch, seq_len, d_model = query_asset.shape
        n_other = key_value_assets.shape[1]

        # Проецируем запросы от целевого актива
        Q = self.query_proj(query_asset)

        # Проецируем ключи и значения от других активов
        K = self.key_proj(key_value_assets.view(-1, seq_len, d_model))
        V = self.value_proj(key_value_assets.view(-1, seq_len, d_model))

        # Изменяем форму для многоголового внимания
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, n_other, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch, n_other, seq_len, self.n_heads, self.head_dim)

        # Cross-attention: каждая позиция запроса обращает внимание
        # на все позиции всех других активов
        K_last = K[:, :, -1, :, :].transpose(1, 2)
        V_last = V[:, :, -1, :, :].transpose(1, 2)

        # Оценки внимания
        scores = torch.matmul(Q, K_last.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)

        # Взвешенные значения
        context = torch.matmul(attention, V_last)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.output_proj(context), attention
```

### Многоголовый Cross-Attention

Несколько голов внимания улавливают различные типы межактивных связей:

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Многоголовый cross-attention с разными головами, специализирующимися на:
    - Связях на основе корреляции
    - Опережающе-запаздывающих связях
    - Перетоке волатильности
    - Группировках по секторам/отраслям
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_query, x_key_value, mask=None):
        """
        Аргументы:
            x_query: [batch, n_query_assets, seq_len, d_model]
            x_key_value: [batch, n_kv_assets, seq_len, d_model]

        Возвращает:
            output: [batch, n_query_assets, seq_len, d_model]
            attention: [batch, n_heads, n_query_assets, n_kv_assets]
        """
        batch, n_q, seq_len, d_model = x_query.shape
        n_kv = x_key_value.shape[1]

        # Агрегируем временное измерение для межактивного внимания
        q = x_query.mean(dim=2)
        k = x_key_value.mean(dim=2)
        v = x_key_value.mean(dim=2)

        # Проецируем
        Q = self.W_q(q).view(batch, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(k).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Оценки внимания: [batch, n_heads, n_q, n_kv]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Взвешенная сумма: [batch, n_heads, n_q, head_dim]
        context = torch.matmul(attention, V)

        # Изменяем форму и проецируем
        context = context.transpose(1, 2).contiguous().view(batch, n_q, d_model)
        output = self.W_o(context)

        # Транслируем обратно на длину последовательности
        output = output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        output = self.layer_norm(x_query + output)

        return output, attention
```

### Временной Cross-Attention

Улавливает опережающе-запаздывающие связи во времени:

```python
class TemporalCrossAttention(nn.Module):
    """
    Cross-attention с учётом временных сдвигов между активами.

    Пример: BTC в момент t-1 прогнозирует ETH в момент t
    """

    def __init__(self, d_model, n_heads, max_lag=5):
        super().__init__()
        self.max_lag = max_lag
        self.attention = MultiHeadCrossAttention(d_model, n_heads)

        # Обучаемые веса задержки
        self.lag_weights = nn.Parameter(torch.ones(max_lag + 1) / (max_lag + 1))

    def forward(self, x_query, x_key_value):
        """
        Аргументы:
            x_query: [batch, n_q, seq_len, d_model]
            x_key_value: [batch, n_kv, seq_len, d_model]

        Возвращает:
            output: Представление с вниманием и временным выравниванием
            attention: Веса межактивного внимания для каждой задержки
        """
        batch, n_q, seq_len, d_model = x_query.shape
        n_kv = x_key_value.shape[1]

        outputs = []
        attentions = []

        # Вычисляем внимание при разных задержках
        for lag in range(self.max_lag + 1):
            if lag == 0:
                kv_lagged = x_key_value
            else:
                # Сдвигаем key_value назад на lag шагов
                kv_lagged = F.pad(x_key_value[:, :, :-lag], (0, 0, lag, 0))

            out, attn = self.attention(x_query, kv_lagged)
            outputs.append(out)
            attentions.append(attn)

        # Взвешенная комбинация по задержкам
        lag_weights = F.softmax(self.lag_weights, dim=0)
        output = sum(w * o for w, o in zip(lag_weights, outputs))

        return output, torch.stack(attentions, dim=1)
```

### Иерархический Cross-Attention

Моделирует связи на нескольких уровнях (активы, секторы, рынки):

```python
class HierarchicalCrossAttention(nn.Module):
    """
    Трёхуровневая иерархия:
    1. Уровень активов: Связи отдельных активов
    2. Уровень секторов: Связи секторов/отраслей
    3. Уровень рынков: Межрыночные связи (крипто vs акции)
    """

    def __init__(self, d_model, n_heads, sector_mapping, market_mapping):
        super().__init__()
        self.sector_mapping = sector_mapping  # asset_id -> sector_id
        self.market_mapping = market_mapping  # asset_id -> market_id

        # Внимание на уровне активов
        self.asset_attention = MultiHeadCrossAttention(d_model, n_heads)

        # Внимание на уровне секторов
        self.sector_attention = MultiHeadCrossAttention(d_model, n_heads // 2)

        # Внимание на уровне рынков
        self.market_attention = MultiHeadCrossAttention(d_model, n_heads // 4)

        # Объединение иерархий
        self.combine = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        """
        Аргументы:
            x: [batch, n_assets, seq_len, d_model]

        Возвращает:
            output: Иерархически обработанное представление
        """
        # Cross-attention на уровне активов
        asset_out, _ = self.asset_attention(x, x)

        # Агрегация по секторам
        sector_repr = self._aggregate_to_sectors(x)
        sector_out, _ = self.sector_attention(sector_repr, sector_repr)
        sector_out = self._broadcast_from_sectors(sector_out, x.shape)

        # Агрегация по рынкам
        market_repr = self._aggregate_to_markets(x)
        market_out, _ = self.market_attention(market_repr, market_repr)
        market_out = self._broadcast_from_markets(market_out, x.shape)

        # Объединение всех уровней
        combined = torch.cat([asset_out, sector_out, market_out], dim=-1)
        return self.combine(combined)
```

## Математические основы

### Вычисление оценок внимания

Оценка внимания между запрашивающим активом $i$ и ключевым активом $j$:

$$\text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

Где:
- $Q_i \in \mathbb{R}^{T \times d_k}$ — представления запросов для актива $i$
- $K_j \in \mathbb{R}^{T \times d_k}$ — представления ключей для актива $j$
- $V_j \in \mathbb{R}^{T \times d_v}$ — представления значений для актива $j$
- $d_k$ — размерность ключей (масштабирующий фактор)

### Cross-Attention vs Self-Attention

| Аспект | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| Источник Q, K, V | Одна последовательность | Q из одной, K/V из другой |
| Применение | Временные паттерны | Межактивные связи |
| Симметрия | Симметричный | Может быть асимметричным |
| Сложность | $O(T^2)$ | $O(T^2 \cdot N)$ для N активов |

### Масштабированное скалярное произведение

Для мультиактивных сценариев с $N$ активами:

$$\text{MultiAssetAttention}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

Где каждая голова $i$ вычисляет:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

## Представление данных

### Мультиактивная инженерия признаков

```python
def create_multi_asset_features(df_dict: dict, lookback: int = 100) -> np.ndarray:
    """
    Создание тензора признаков для нескольких активов.

    Аргументы:
        df_dict: Словарь, отображающий символ актива на DataFrame с OHLCV
        lookback: Количество исторических временных шагов

    Возвращает:
        features: [n_samples, n_assets, lookback, n_features]
    """
    features = []

    for symbol, df in df_dict.items():
        asset_features = []

        # Ценовые признаки
        asset_features.append(np.log(df['close'] / df['close'].shift(1)))  # Лог-доходности
        asset_features.append((df['close'] - df['open']) / df['open'])     # Внутридневная доходность
        asset_features.append((df['high'] - df['low']) / df['close'])      # Диапазон

        # Объёмные признаки
        asset_features.append(df['volume'] / df['volume'].rolling(20).mean())  # Относительный объём

        # Технические индикаторы
        asset_features.append(compute_rsi(df['close'], 14))
        asset_features.append(compute_macd(df['close']))

        features.append(np.column_stack(asset_features))

    return np.stack(features, axis=1)  # [time, n_assets, n_features]
```

### Данные фондового рынка

```python
import yfinance as yf

def fetch_stock_data(symbols: list, start: str, end: str) -> dict:
    """
    Получение данных акций из Yahoo Finance.

    Аргументы:
        symbols: Список тикеров (например, ['AAPL', 'GOOGL', 'MSFT'])
        start: Дата начала (YYYY-MM-DD)
        end: Дата окончания (YYYY-MM-DD)

    Возвращает:
        Словарь, отображающий символ на DataFrame
    """
    data = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval='1h')
        df.columns = df.columns.str.lower()
        data[symbol] = df

    return data

# Пример использования
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
stock_data = fetch_stock_data(stock_symbols, '2023-01-01', '2024-01-01')
```

### Данные криптовалютных рынков (Bybit)

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class BybitDataLoader:
    """Загрузка криптовалютных данных с биржи Bybit."""

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 60 минут = 1 час
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получение данных kline/свечей с Bybit.

        Аргументы:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Интервал свечей (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Количество свечей (макс 1000)

        Возвращает:
            DataFrame с данными OHLCV
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        response = self.session.get(self.BASE_URL, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Ошибка API: {data['retMsg']}")

        klines = data['result']['list']

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        return df.sort_values('timestamp').reset_index(drop=True)

    def fetch_multi_asset(self, symbols: list, **kwargs) -> dict:
        """Получение данных для нескольких активов."""
        return {symbol: self.fetch_klines(symbol, **kwargs) for symbol in symbols}

# Пример использования
loader = BybitDataLoader()
crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
crypto_data = loader.fetch_multi_asset(crypto_symbols, interval='60', limit=1000)
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler

def prepare_cross_attention_data(
    asset_data: Dict[str, pd.DataFrame],
    lookback: int = 168,  # 7 дней почасовых данных
    horizon: int = 24,    # 24 часа вперёд
    features: List[str] = ['log_return', 'volume_ratio', 'volatility', 'rsi']
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Подготовка данных для мультиактивной модели cross-attention.

    Возвращает:
        X: [n_samples, n_assets, lookback, n_features]
        y: [n_samples, n_assets] - Будущие доходности
        symbols: Список символов активов
    """
    symbols = list(asset_data.keys())
    n_assets = len(symbols)

    # Вычисляем признаки для каждого актива
    processed = {}
    for symbol, df in asset_data.items():
        feat = pd.DataFrame(index=df.index)

        feat['log_return'] = np.log(df['close'] / df['close'].shift(1))
        feat['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        feat['volatility'] = feat['log_return'].rolling(20).std()
        feat['rsi'] = compute_rsi(df['close'], 14)

        processed[symbol] = feat

    # Выравниваем временные метки
    common_idx = processed[symbols[0]].index
    for symbol in symbols[1:]:
        common_idx = common_idx.intersection(processed[symbol].index)

    # Создаём последовательности
    X, y = [], []
    for i in range(lookback, len(common_idx) - horizon):
        x_sample = []
        y_sample = []

        for symbol in symbols:
            df = processed[symbol].loc[common_idx]
            x_sample.append(df.iloc[i-lookback:i][features].values)
            y_sample.append(df.iloc[i+horizon]['log_return'])

        X.append(np.stack(x_sample, axis=0))
        y.append(np.array(y_sample))

    return np.array(X), np.array(y), symbols

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Вычисление индекса относительной силы (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 02: Модель Cross-Attention

Смотрите [python/model.py](python/model.py) для полной реализации.

### 03: Обучение модели

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_cross_attention_model(
    model: nn.Module,
    train_data: tuple,
    val_data: tuple,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cuda'
):
    """
    Обучение модели cross-attention.

    Аргументы:
        model: Модель CrossAttentionMultiAsset
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Создаём загрузчики данных
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Функция потерь и оптимизатор
    if model.output_type == 'regression':
        criterion = nn.MSELoss()
    elif model.output_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # portfolio
        criterion = lambda pred, ret: -torch.mean(torch.sum(pred * ret, dim=-1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)

            if model.output_type == 'classification':
                predictions = predictions.view(-1, 3)
                batch_y = (batch_y > 0).long().view(-1)

            loss = criterion(predictions, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(X_val).to(device)
            val_y = torch.FloatTensor(y_val).to(device)
            val_pred = model(val_x)

            if model.output_type == 'classification':
                val_pred = val_pred.view(-1, 3)
                val_y = (val_y > 0).long().view(-1)

            val_loss = criterion(val_pred, val_y).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f"Эпоха {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, "
                  f"Val Loss = {val_loss:.6f}")

    return model
```

### 04: Мультиактивное прогнозирование

```python
# python/predict.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict_and_visualize(
    model,
    X: np.ndarray,
    symbols: list,
    device: str = 'cuda'
):
    """
    Создание прогнозов и визуализация паттернов внимания.
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        predictions, attentions = model(x, return_attention=True)

    predictions = predictions.cpu().numpy()

    # Визуализация межактивного внимания
    if attentions:
        cross_attn = attentions[-1]['cross_asset']
        avg_attn = cross_attn.mean(dim=[0, 1]).cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attn,
            xticklabels=symbols,
            yticklabels=symbols,
            annot=True,
            fmt='.2f',
            cmap='Blues'
        )
        plt.title('Веса межактивного внимания')
        plt.xlabel('Ключ (Исходный актив)')
        plt.ylabel('Запрос (Целевой актив)')
        plt.tight_layout()
        plt.savefig('cross_attention_heatmap.png', dpi=150)
        plt.close()

    return predictions
```

### 05: Бэктестинг портфеля

```python
# python/backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List

class CrossAttentionBacktest:
    """
    Бэктестинг стратегии портфеля cross-attention.
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 24  # Часы
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        X: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Запуск бэктеста на тестовых данных.

        Аргументы:
            X: [n_samples, n_assets, lookback, n_features]
            returns: [n_samples, n_assets] - Фактические будущие доходности
            timestamps: DatetimeIndex для результатов

        Возвращает:
            DataFrame с метриками портфеля во времени
        """
        import torch

        self.model.eval()
        n_samples, n_assets, _, _ = X.shape

        capital = self.initial_capital
        positions = np.zeros(n_assets)

        results = []

        for i in range(0, n_samples, self.rebalance_freq):
            # Получаем прогнозы модели (веса портфеля)
            with torch.no_grad():
                x = torch.FloatTensor(X[i:i+1])
                weights = self.model(x).numpy().flatten()

            # Нормализуем веса
            if self.model.output_type == 'regression':
                weights = np.clip(weights, -1, 1)
                weights = weights / (np.abs(weights).sum() + 1e-8)

            # Вычисляем транзакционные издержки
            position_change = np.abs(weights - positions).sum()
            costs = position_change * self.transaction_cost * capital

            # Вычисляем доходность за период
            period_returns = returns[i:min(i+self.rebalance_freq, n_samples)]

            for j, ret in enumerate(period_returns):
                portfolio_return = np.sum(positions * ret)
                capital = capital * (1 + portfolio_return)

                if j == 0:
                    capital -= costs

                results.append({
                    'timestamp': timestamps[i+j] if i+j < len(timestamps) else None,
                    'capital': capital,
                    'return': portfolio_return,
                    'positions': positions.copy(),
                    'weights': weights.copy()
                })

            # Обновляем позиции
            positions = weights

        return pd.DataFrame(results)

    def compute_metrics(self, results: pd.DataFrame) -> Dict:
        """Вычисление метрик производительности."""
        returns = results['return'].values

        # Коэффициент Шарпа (годовой для почасовых данных)
        sharpe = np.sqrt(365 * 24) * returns.mean() / (returns.std() + 1e-8)

        # Коэффициент Сортино
        downside = returns[returns < 0]
        sortino = np.sqrt(365 * 24) * returns.mean() / (downside.std() + 1e-8)

        # Максимальная просадка
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Общая доходность
        total_return = (results['capital'].iloc[-1] / self.initial_capital - 1) * 100

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown * 100,
            'volatility': returns.std() * np.sqrt(365 * 24) * 100,
            'win_rate': (returns > 0).mean() * 100
        }
```

## Реализация на Rust

Смотрите [rust/](rust/) для полной реализации на Rust с использованием ML-фреймворка `candle`.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Экспорты библиотеки
│   ├── model/                 # Реализация модели
│   │   ├── mod.rs
│   │   ├── attention.rs       # Слои cross-attention
│   │   ├── embedding.rs       # Token embeddings
│   │   └── cross_attention.rs # Основная модель
│   ├── data/                  # Обработка данных
│   │   ├── mod.rs
│   │   ├── bybit.rs          # Клиент API Bybit
│   │   ├── features.rs       # Инженерия признаков
│   │   └── dataset.rs        # Датасет для обучения
│   └── strategy/             # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs        # Генерация сигналов
│       └── backtest.rs       # Движок бэктестинга
└── examples/
    ├── fetch_data.rs         # Загрузка данных с Bybit
    ├── train.rs              # Обучение модели
    └── backtest.rs           # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Перейдите в проект Rust
cd rust

# Загрузите данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT

# Обучите модель
cargo run --release --example train -- --epochs 50 --batch-size 32

# Запустите бэктест
cargo run --release --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── model.py                  # Модель cross-attention
├── data.py                   # Загрузка данных (Bybit + Yahoo Finance)
├── features.py               # Инженерия признаков
├── train.py                  # Скрипт обучения
├── backtest.py               # Утилиты бэктестинга
├── requirements.txt          # Зависимости
└── examples/
    ├── 01_data_preparation.py
    ├── 02_model_training.py
    ├── 03_prediction.py
    └── 04_backtesting.py
```

### Быстрый старт (Python)

```bash
# Установите зависимости
pip install -r requirements.txt

# Запустите примеры
python examples/01_data_preparation.py
python examples/02_model_training.py
python examples/03_prediction.py
python examples/04_backtesting.py
```

## Лучшие практики

### Когда использовать Cross-Attention

**Хорошие сценарии применения:**
- Торговля коррелированными классами активов (крипто, технологические акции, товары)
- Оптимизация портфеля по нескольким активам
- Обнаружение опережающе-запаздывающих связей
- Мультиактивное управление рисками

**Не идеально для:**
- Прогнозирования одного актива (используйте более простые модели)
- Очень краткосрочного прогнозирования (проблемы с задержкой)
- Некоррелированных активов (cross-attention не поможет)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуемое | Примечания |
|----------|---------------|------------|
| `d_model` | 64-128 | Под вычислительный бюджет |
| `n_heads` | 4-8 | Больше голов для большего числа активов |
| `n_layers` | 2-4 | Глубже для сложных связей |
| `dropout` | 0.1-0.2 | Выше для малых датасетов |
| `lookback` | 168 (7 дней почасовых) | Соответствует горизонту прогноза |

### Типичные ошибки

1. **Коллапс корреляции**: Всё внимание уходит на один доминирующий актив
   - Решение: Используйте dropout, регуляризацию внимания

2. **Переобучение на межактивных паттернах**: Модель запоминает ложные корреляции
   - Решение: Больше данных, более простая модель, регуляризация

3. **Игнорирование смены режимов**: Межактивные связи меняются со временем
   - Решение: Скользящие окна обучения, определение режимов

4. **Вычислительные затраты**: O(N² * T²) для N активов, T временных шагов
   - Решение: Разреженное внимание, эффективные реализации

## Ресурсы

### Научные статьи

- [Portfolio Transformer for Attention-Based Asset Allocation](https://arxiv.org/abs/2206.03246) — Сквозная оптимизация портфеля с механизмом внимания
- [Attention-Based Ensemble Learning for Portfolio Optimisation](https://arxiv.org/abs/2404.08935) — Фреймворк MASAAT с мультиагентным вниманием
- [Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks](https://arxiv.org/abs/2407.15532) — Управление портфелем на основе GAT
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальная статья о Transformer

### Реализации

- [PyTorch Multi-Head Attention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Candle ML Framework (Rust)](https://github.com/huggingface/candle)

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Многогоризонтное прогнозирование
- [Глава 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-ticker attention
- [Глава 44: ProbSparse Attention](../44_probsparse_attention) — Эффективные механизмы внимания
- [Глава 46: Temporal Attention Networks](../46_temporal_attention_networks) — Временное внимание

---

## Уровень сложности

**Продвинутый**

Предварительные требования:
- Архитектура Transformer и механизмы внимания
- Теория мультиактивного портфеля
- Прогнозирование временных рядов
- ML-библиотеки PyTorch или Rust
