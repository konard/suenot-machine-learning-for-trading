# Глава 55: FNet — Преобразование Фурье для эффективного смешивания токенов

Эта глава посвящена **FNet** — инновационной архитектуре, которая заменяет механизмы самовнимания (self-attention) операциями преобразования Фурье, достигая вычислительной сложности O(n log n) при сохранении конкурентоспособной производительности для прогнозирования финансовых временных рядов.

<p align="center">
<img src="https://i.imgur.com/7vKwPzN.png" width="70%">
</p>

## Содержание

1. [Введение в FNet](#введение-в-fnet)
    * [Зачем заменять внимание?](#зачем-заменять-внимание)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с трансформерами](#сравнение-с-трансформерами)
2. [Архитектура FNet](#архитектура-fnet)
    * [Слой преобразования Фурье](#слой-преобразования-фурье)
    * [Полносвязная сеть](#полносвязная-сеть)
    * [Полная архитектура](#полная-архитектура)
3. [Математические основы](#математические-основы)
    * [Дискретное преобразование Фурье](#дискретное-преобразование-фурье)
    * [2D преобразование Фурье в FNet](#2d-преобразование-фурье-в-fnet)
    * [Вычислительная сложность](#вычислительная-сложность)
4. [FNet для трейдинга](#fnet-для-трейдинга)
    * [Адаптация для временных рядов](#адаптация-для-временных-рядов)
    * [Мультиактивное прогнозирование](#мультиактивное-прогнозирование)
    * [Генерация сигналов](#генерация-сигналов)
5. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Архитектура FNet](#02-архитектура-fnet)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Торговая стратегия](#04-торговая-стратегия)
    * [05: Бэктестинг](#05-бэктестинг)
6. [Реализация на Rust](#реализация-на-rust)
7. [Реализация на Python](#реализация-на-python)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в FNet

FNet (Fourier Network) — это революционная архитектура, представленная Google Research в 2021 году, которая бросает вызов доминированию механизмов внимания в моделях Transformer. Вместо вычисления дорогостоящих квадратичных матриц внимания FNet использует **быстрое преобразование Фурье (БПФ)** для смешивания представлений токенов.

### Зачем заменять внимание?

Стандартное самовнимание имеет сложность O(L²), где L — длина последовательности:

```
Стандартное внимание:
- Q, K, V проекции: O(L × d × d)
- Вычисление QK^T: O(L² × d)  ← Узкое место!
- Softmax + умножение на V: O(L² × d)

Для L=512, d=768: ~200M операций на слой
Для L=2048: ~3.2B операций на слой (в 16 раз больше!)
```

FNet заменяет это на БПФ:

```
Слой Фурье в FNet:
- 2D БПФ: O(L × d × log(L × d))
- Взятие действительной части: O(L × d)

Для L=512, d=768: ~3.6M операций (в 55 раз быстрее!)
Для L=2048: ~19M операций (в 168 раз быстрее!)
```

### Ключевые преимущества

1. **Скорость**: На 80% быстрее обучение на GPU, на 70% быстрее на TPU
2. **Эффективность памяти**: Нет матриц внимания размера O(L²)
3. **Простота**: Нет обучаемых параметров в слое смешивания
4. **Длинные последовательности**: Линейное масштабирование с длиной
5. **Конкурентная точность**: 92-97% от производительности BERT на GLUE

### Сравнение с трансформерами

| Характеристика | Transformer | FNet | Преимущество |
|----------------|-------------|------|--------------|
| Смешивание токенов | Self-Attention | БПФ | FNet (скорость) |
| Сложность | O(L²) | O(L log L) | FNet |
| Параметры | Обучаемые Q,K,V | Нет | FNet (проще) |
| Скорость GPU | Базовая | На 80% быстрее | FNet |
| GLUE Score | 100% | 92-97% | Transformer |
| Длинные последов. | Медленно | Быстро | FNet |
| Интерпретируемость | Веса внимания | Частотный анализ | Разные |

## Архитектура FNet

### Слой преобразования Фурье

Ядро FNet удивительно простое:

```python
class FourierTransformLayer(nn.Module):
    """
    Заменяет самовнимание 2D преобразованием Фурье.

    Преобразование Фурье смешивает токены по двум измерениям:
    1. Измерение последовательности (по временным шагам)
    2. Скрытое измерение (по признакам)
    """

    def __init__(self):
        super().__init__()
        # Нет обучаемых параметров!

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]

        # Применяем 2D БПФ
        # БПФ по измерению последовательности смешивает временную информацию
        # БПФ по скрытому измерению смешивает признаковую информацию
        x_fft = torch.fft.fft2(x.float())

        # Берём действительную часть (отбрасываем мнимую)
        return x_fft.real
```

**Почему это работает?**

1. **Преобразование Фурье как глобальное смешивание**: Каждый выходной токен содержит информацию от ВСЕХ входных токенов (через представление в частотной области)
2. **Периодические паттерны**: Финансовые данные часто имеют периодические компоненты (дневные, недельные, месячные циклы)
3. **Эффективность**: Алгоритм БПФ вычисляет преобразование за O(n log n) вместо наивного O(n²)

### Полносвязная сеть

После смешивания Фурье FNet использует стандартную полносвязную сеть:

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Расширяем, активируем, проецируем обратно
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### Полная архитектура

```
┌──────────────────────────────────────────────────────────────────────┐
│                              FNet                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────┐          │
│  │                   Входное эмбеддинг                      │          │
│  │   Эмбеддинг токенов + Позиционное кодирование            │          │
│  └────────────────────────────┬───────────────────────────┘          │
│                               │                                        │
│           ┌───────────────────┴───────────────────┐                   │
│           │         Блок энкодера FNet            │ × N               │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │   Слой преобразования Фурье      │  │                   │
│           │  │     FFT2D → Действительная часть  │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓ + Остаточное соединение │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │      Нормализация слоя           │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓                         │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │      Полносвязная сеть          │  │                   │
│           │  │   Linear → GELU → Linear         │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           │             ↓ + Остаточное соединение │                   │
│           │  ┌─────────────────────────────────┐  │                   │
│           │  │      Нормализация слоя           │  │                   │
│           │  └─────────────────────────────────┘  │                   │
│           └───────────────────┬───────────────────┘                   │
│                               │                                        │
│  ┌────────────────────────────┴───────────────────────────┐          │
│  │                    Выходная голова                      │          │
│  │   Пулинг → Linear → Прогноз                            │          │
│  └────────────────────────────────────────────────────────┘          │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Математические основы

### Дискретное преобразование Фурье

Дискретное преобразование Фурье (ДПФ) преобразует последовательность из временной области в частотную:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i \cdot kn / N}$$

Где:
- $x_n$ — входная последовательность в позиции n
- $X_k$ — k-я частотная компонента
- $N$ — длина последовательности
- $i$ — мнимая единица

### 2D преобразование Фурье в FNet

FNet применяет 2D БПФ к входному тензору:

```python
def fnet_fourier_transform(x):
    """
    Применяет 2D преобразование Фурье к входному тензору.

    Args:
        x: Входной тензор размера [batch, seq_len, hidden_dim]

    Returns:
        Действительная часть выхода 2D БПФ
    """
    # БПФ по двум последним измерениям
    # Измерение -2: измерение последовательности (временное смешивание)
    # Измерение -1: скрытое измерение (признаковое смешивание)
    return torch.fft.fft2(x).real
```

2D БПФ можно разложить на:

1. **БПФ по измерению последовательности**: Смешивает информацию по временным шагам
   - Захватывает временные паттерны и периодичность
   - Каждая позиция учится от всех других позиций

2. **БПФ по скрытому измерению**: Смешивает информацию по признакам
   - Комбинирует разные признаковые представления
   - Создаёт более богатые представления

### Вычислительная сложность

**Стандартное самовнимание:**
```
Сложность: O(L² × d)
Где L = длина последовательности, d = скрытое измерение

Память: O(L²) для матрицы внимания
```

**Преобразование Фурье FNet:**
```
Сложность: O(L × d × log(L × d))
≈ O(L × d × log(L)) для типичных случаев

Память: O(L × d) - матрица внимания не нужна
```

**Анализ ускорения:**

| Длина последовательности | Операции внимания | Операции БПФ | Ускорение |
|--------------------------|-------------------|--------------|-----------|
| 128 | 12.6M | 0.8M | 15x |
| 512 | 201.3M | 3.6M | 55x |
| 1024 | 805.3M | 7.8M | 103x |
| 2048 | 3221.2M | 16.5M | 195x |

## FNet для трейдинга

### Адаптация для временных рядов

Адаптация FNet для финансовых временных рядов требует нескольких модификаций:

```python
class FNetForTrading(nn.Module):
    """
    FNet, адаптированный для прогнозирования финансовых временных рядов.

    Модификации оригинального FNet:
    1. Временное позиционное кодирование (учитывает торговые часы, дни)
    2. Многомасштабный анализ Фурье (дневные, недельные, месячные паттерны)
    3. Каузальное маскирование для прогнозирования в реальном времени
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_type: str = 'regression'
    ):
        super().__init__()

        # Входной эмбеддинг
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Блоки энкодера FNet
        self.encoder_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Выходная голова
        self.output_head = self._create_output_head(d_model, output_type)

    def forward(self, x, return_frequencies=False):
        # x: [batch, seq_len, n_features]

        # Проекция в измерение модели
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Применяем блоки энкодера FNet
        frequency_maps = []
        for block in self.encoder_blocks:
            x, freq = block(x, return_frequencies=True)
            frequency_maps.append(freq)

        # Генерируем прогнозы
        output = self.output_head(x)

        if return_frequencies:
            return output, frequency_maps
        return output
```

### Мультиактивное прогнозирование

FNet может прогнозировать несколько активов одновременно:

```python
class MultiFNet(nn.Module):
    """
    Мультиактивный FNet для портфельного прогнозирования.

    Использует отдельные слои Фурье для:
    1. Временного смешивания (по времени)
    2. Смешивания активов (по разным инструментам)
    """

    def __init__(self, n_assets, n_features, d_model):
        super().__init__()

        # Эмбеддинг для каждого актива
        self.asset_embeddings = nn.ModuleList([
            nn.Linear(n_features, d_model)
            for _ in range(n_assets)
        ])

        # Временной FNet (внутри каждого актива)
        self.temporal_fnet = FNetEncoder(d_model, n_layers=2)

        # Межактивный FNet (между активами)
        self.cross_asset_fnet = FNetEncoder(d_model, n_layers=2)

        # Выходные головы для каждого актива
        self.prediction_heads = nn.ModuleList([
            nn.Linear(d_model, 1)
            for _ in range(n_assets)
        ])

    def forward(self, x):
        # x: [batch, seq_len, n_assets, n_features]
        batch_size, seq_len, n_assets, _ = x.shape

        # Создаём эмбеддинг для каждого актива отдельно
        asset_features = []
        for i in range(n_assets):
            asset_x = self.asset_embeddings[i](x[:, :, i, :])
            asset_features.append(asset_x)

        # Складываем: [batch, seq_len, n_assets, d_model]
        x = torch.stack(asset_features, dim=2)

        # Применяем временной FNet к каждому активу
        temporal_outputs = []
        for i in range(n_assets):
            temp_out = self.temporal_fnet(x[:, :, i, :])
            temporal_outputs.append(temp_out)
        x = torch.stack(temporal_outputs, dim=2)

        # Применяем межактивный FNet
        # Преобразуем: [batch * seq_len, n_assets, d_model]
        x_reshaped = x.view(batch_size * seq_len, n_assets, -1)
        x_cross = self.cross_asset_fnet(x_reshaped)
        x = x_cross.view(batch_size, seq_len, n_assets, -1)

        # Генерируем прогнозы для каждого актива
        predictions = []
        for i in range(n_assets):
            pred = self.prediction_heads[i](x[:, -1, i, :])
            predictions.append(pred)

        return torch.cat(predictions, dim=1)
```

### Генерация сигналов

Генерация торговых сигналов из прогнозов FNet:

```python
class FNetSignalGenerator:
    """
    Генерация торговых сигналов из прогнозов FNet.
    """

    def __init__(self, model, threshold=0.0, confidence_threshold=0.6):
        self.model = model
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, x, return_confidence=False):
        """
        Генерация торговых сигналов.

        Args:
            x: Входные признаки [batch, seq_len, n_features]
            return_confidence: Возвращать ли оценки уверенности

        Returns:
            signals: Торговые сигналы (-1, 0, 1) для (шорт, ожидание, лонг)
            confidence: Опциональные оценки уверенности
        """
        self.model.eval()
        with torch.no_grad():
            # Получаем прогнозы и частотные карты
            predictions, freq_maps = self.model(x, return_frequencies=True)

            # Вычисляем уверенность из стабильности частот
            confidence = self._calculate_confidence(freq_maps)

            # Генерируем сигналы
            signals = torch.zeros_like(predictions)

            # Сигнал лонг: прогноз выше порога И высокая уверенность
            long_mask = (predictions > self.threshold) & (confidence > self.confidence_threshold)
            signals[long_mask] = 1.0

            # Сигнал шорт: прогноз ниже -порога И высокая уверенность
            short_mask = (predictions < -self.threshold) & (confidence > self.confidence_threshold)
            signals[short_mask] = -1.0

        if return_confidence:
            return signals, confidence
        return signals
```

## Практические примеры

### 01: Подготовка данных

```python
# python/data_loader.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests

class BybitDataLoader:
    """
    Загрузчик данных для криптовалютных данных Bybit.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 1 час
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получение данных свечей с Bybit.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Интервал свечи в минутах
            limit: Количество свечей для получения

        Returns:
            DataFrame с OHLCV данными
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"Ошибка API: {data['retMsg']}")

        # Парсим данные свечей
        klines = data["result"]["list"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Конвертируем типы
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        return df.sort_values("timestamp").reset_index(drop=True)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт признаков для модели FNet.
        """
        df = df.copy()

        # Логарифмические доходности
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Волатильность (скользящее стандартное отклонение за 20 периодов)
        df["volatility"] = df["log_return"].rolling(20).std()

        # Изменение объёма
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Ценовой моментум
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Позиция в полосах Боллинджера
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - sma_20) / (2 * std_20 + 1e-8)

        return df.dropna()
```

### 02: Архитектура FNet

См. [python/model.py](python/model.py) для полной реализации.

### 03: Обучение модели

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import FNet
from data_loader import BybitDataLoader, create_sequences

def train_fnet(
    symbols: list = ["BTCUSDT", "ETHUSDT"],
    seq_len: int = 168,
    horizon: int = 24,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Обучение модели FNet на криптовалютных данных.
    """
    print(f"Обучение на {device}")

    # Загрузка и подготовка данных
    loader = BybitDataLoader()
    all_X, all_y = [], []

    feature_cols = [
        "log_return", "volatility", "volume_ratio",
        "momentum_5", "momentum_20", "rsi", "bb_position"
    ]

    for symbol in symbols:
        print(f"Загрузка данных для {symbol}...")
        df = loader.fetch_klines(symbol, interval="60", limit=2000)
        df = loader.prepare_features(df)

        X, y = create_sequences(df, feature_cols, "log_return", seq_len, horizon)
        all_X.append(X)
        all_y.append(y)

    # Объединяем данные
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Нормализация
    X_mean, X_std = X.mean(axis=(0, 1)), X.std(axis=(0, 1))
    X = (X - X_mean) / (X_std + 1e-8)

    # Разделение train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Создаём датасеты
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Инициализация модели
    model = FNet(
        n_features=len(feature_cols),
        d_model=256,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=seq_len,
        output_dim=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_fnet_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print(f"Обучение завершено. Лучший val loss: {best_val_loss:.6f}")
    return model
```

### 04: Торговая стратегия

```python
# python/strategy.py

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class FNetTradingStrategy:
    """
    Торговая стратегия на основе прогнозов FNet.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.001,
        position_size: float = 1.0,
        stop_loss: float = 0.02,
        take_profit: float = 0.04
    ):
        self.model = model
        self.threshold = threshold
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signal(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Генерация торгового сигнала из прогноза модели.

        Args:
            x: Входные признаки [1, seq_len, n_features]

        Returns:
            signal: Торговый сигнал (-1=шорт, 0=ожидание, 1=лонг)
            confidence: Уверенность прогноза
        """
        self.model.eval()
        with torch.no_grad():
            prediction, freq_maps = self.model(x, return_frequencies=True)
            pred_value = prediction.item()

            # Расчёт уверенности из стабильности частот
            confidence = self._calculate_confidence(freq_maps)

        # Генерация сигнала
        if pred_value > self.threshold and confidence > 0.5:
            return 1, confidence
        elif pred_value < -self.threshold and confidence > 0.5:
            return -1, confidence
        else:
            return 0, confidence
```

### 05: Бэктестинг

```python
# python/backtest.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import FNet
from data_loader import BybitDataLoader, create_sequences
from strategy import FNetTradingStrategy, Backtester

def run_full_backtest():
    """
    Запуск полного бэктеста торговой стратегии FNet.
    """
    # Загрузка модели
    feature_cols = [
        "log_return", "volatility", "volume_ratio",
        "momentum_5", "momentum_20", "rsi", "bb_position"
    ]

    model = FNet(
        n_features=len(feature_cols),
        d_model=256,
        n_layers=4,
        d_ff=1024,
        output_dim=1
    )
    model.load_state_dict(torch.load("best_fnet_model.pt", map_location="cpu"))
    model.eval()

    # Загрузка тестовых данных
    loader = BybitDataLoader()
    df = loader.fetch_klines("BTCUSDT", interval="60", limit=2000)
    df = loader.prepare_features(df)

    # Создание последовательностей
    X, y = create_sequences(df, feature_cols, "log_return", seq_len=168, horizon=24)

    # Используем последние 20% как тест
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:]
    prices_test = df["close"].values[168 + 23 + test_start:][:len(X_test)]

    # Нормализация с использованием статистики обучения
    X_mean = X[:test_start].mean(axis=(0, 1))
    X_std = X[:test_start].std(axis=(0, 1))
    X_test = (X_test - X_mean) / (X_std + 1e-8)

    # Создание стратегии и бэктестера
    strategy = FNetTradingStrategy(
        model=model,
        threshold=0.001,
        position_size=1.0,
        stop_loss=0.02,
        take_profit=0.04
    )

    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    # Запуск бэктеста
    results = backtester.run_backtest(strategy, X_test, prices_test)

    # Вывод метрик
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*50)
    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results
```

## Реализация на Rust

См. [rust_fnet](rust_fnet/) для полной реализации на Rust.

```
rust_fnet/
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
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура FNet
│   │   ├── mod.rs
│   │   ├── fourier.rs      # Слой преобразования Фурье
│   │   ├── encoder.rs      # Стек энкодера
│   │   └── fnet.rs         # Полная модель
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
# Переход в папку Rust-проекта
cd rust_fnet

# Загрузка данных с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Обучение модели
cargo run --example train -- --epochs 100 --batch-size 32

# Запуск бэктеста
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

```
python/
├── __init__.py             # Экспорты пакета
├── model.py                # Реализация модели FNet
├── data_loader.py          # Загрузка данных Bybit
├── train.py                # Скрипт обучения
├── strategy.py             # Торговая стратегия
├── backtest.py             # Утилиты бэктестинга
├── requirements.txt        # Зависимости
└── example_usage.py        # Полный пример
```

### Быстрый старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка данных и обучение
python train.py --symbols BTCUSDT,ETHUSDT --epochs 100

# Запуск бэктеста
python backtest.py --model best_fnet_model.pt
```

## Лучшие практики

### Когда использовать FNet

**Хорошие случаи использования:**
- Прогнозирование длинных последовательностей (>512 токенов)
- Прогнозирование в реальном времени (важна скорость инференса)
- Обнаружение периодических паттернов (дневные/недельные циклы)
- Среды с ограниченными ресурсами
- Управление мультиактивным портфелем

**Не идеально для:**
- Задач, требующих интерпретируемых весов внимания
- Очень коротких последовательностей (<64 токенов)
- Когда критически важна максимальная точность (используйте Transformer)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуемое | Примечания |
|----------|---------------|------------|
| `d_model` | 256 | Баланс между ёмкостью и скоростью |
| `n_layers` | 4-6 | Больше для сложных паттернов |
| `d_ff` | 4 × d_model | Стандартный коэффициент расширения |
| `dropout` | 0.1-0.2 | Выше для малых датасетов |
| `seq_len` | 168-336 | 1-2 недели почасовых данных |

### Распространённые ошибки

1. **Отсутствие нормализации входов**: БПФ чувствительно к масштабу входных данных
2. **Игнорирование фазовой информации**: Только действительная часть теряет направленную информацию
3. **Слишком короткие последовательности**: БПФ нужно достаточно данных для обнаружения паттернов
4. **Переобучение на шуме**: Используйте правильную регуляризацию

## Ресурсы

### Научные работы

- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) — Оригинальная статья FNet (2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальный Transformer
- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) — Связанная работа по Фурье в нейросетях

### Реализации

- [Google Research FNet](https://github.com/google-research/google-research/tree/master/f_net) — Официальная реализация
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/fnet) — FNet в библиотеке transformers
- [PyTorch FFT Documentation](https://pytorch.org/docs/stable/fft.html) — Операции БПФ в PyTorch

### Связанные главы

- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Другой вариант эффективного внимания
- [Глава 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Внимание с локально-чувствительным хешированием
- [Глава 58: Flash Attention Trading](../58_flash_attention_trading) — Память-эффективное внимание

---

## Уровень сложности

**Средний**

Необходимые знания:
- Понимание основ преобразования Фурье
- Основы нейронных сетей
- Концепции прогнозирования временных рядов
- Опыт программирования на Python/Rust
