# Глава 85: Zero-Shot Трейдинг

## Обзор

Zero-shot трейдинг представляет собой парадигмальный сдвиг в алгоритмической торговле, позволяющий моделям делать прогнозы по совершенно новым активам, рынкам или режимам **без каких-либо примеров для обучения**. В отличие от few-shot обучения, требующего небольшого набора поддержки, zero-shot обучение использует перенос знаний и семантическое понимание для немедленной генерализации на невиданные ранее сценарии.

Этот подход особенно эффективен для криптовалютных рынков, где постоянно появляются новые токены, или для адаптации к внезапным сменам рыночных режимов, когда исторические паттерны становятся неактуальными.

## Содержание

1. [Введение](#введение)
2. [Теоретические основы](#теоретические-основы)
3. [Zero-Shot vs Few-Shot обучение](#zero-shot-vs-few-shot-обучение)
4. [Архитектура](#архитектура)
5. [Стратегия реализации](#стратегия-реализации)
6. [Интеграция с Bybit](#интеграция-с-bybit)
7. [Торговая стратегия](#торговая-стратегия)
8. [Управление рисками](#управление-рисками)
9. [Метрики производительности](#метрики-производительности)
10. [Ссылки](#ссылки)

---

## Введение

### Проблема Zero-Shot в трейдинге

Традиционное машинное обучение для трейдинга следует предсказуемому паттерну:
1. Собрать исторические данные по целевому активу
2. Обучить модель на этих данных
3. Делать прогнозы для того же актива

Но что происходит, когда:
- Новая криптовалюта листится без исторических данных?
- Рыночный режим резко меняется, делая исторические паттерны недействительными?
- Вы хотите торговать в совершенно новом сегменте рынка?

**Zero-shot трейдинг** решает эти проблемы, обучаясь **переносимым представлениям**, которые обобщаются на разные активы и рыночные условия без специфичного обучения на целевых данных.

### Почему Zero-Shot для трейдинга?

```
+-------------------------------------------------------------------------+
|                    Проблема Zero-Shot Трейдинга                           |
+-------------------------------------------------------------------------+
|                                                                           |
|   Традиционный подход:               Zero-Shot подход:                    |
|   --------------------              --------------------                  |
|                                                                           |
|   Новый актив листится:              Новый актив листится:                |
|   "Ждем 6 месяцев данных"            "Торгуем сразу!"                     |
|   "Затем обучаем модель"             "Используем перенос знаний"          |
|   "Затем начинаем торговать"                                              |
|                                                                           |
|   Обнаружена смена режима:           Обнаружена смена режима:             |
|   "Модель сломалась"                 "Адаптируемся через семантику"       |
|   "Переобучаем с нуля"               "Продолжаем торговать"               |
|                                                                           |
|   Рыночный крах:                     Рыночный крах:                       |
|   "Исторические паттерны не работают" "Используем кросс-рыночные          |
|   "Понесены большие убытки"           инварианты"                         |
|                                       "Робастные прогнозы продолжаются"   |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Ключевые преимущества

| Аспект | Традиционный ML | Few-Shot | Zero-Shot |
|--------|-----------------|----------|-----------|
| Требования к данным | 1000+ примеров | 5-20 примеров | 0 примеров |
| Адаптация к новому активу | Полное переобучение | Нужны примеры | Мгновенная |
| Обработка смены режима | Плохая | Умеренная | Отличная |
| Вычислительные затраты | Высокие | Низкие | Очень низкие |
| Время до первой сделки | Дни/недели | Часы | Секунды |

---

## Теоретические основы

### Фреймворк Zero-Shot обучения

Zero-shot обучение работает путем отображения входных данных (рыночных данных) и выходов (прогнозов) в общее семантическое пространство эмбеддингов, где отношения могут быть перенесены.

### Математическая формулировка

**Функции эмбеддинга**:

Пусть $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$ - энкодер рыночных данных, отображающий рыночные признаки в эмбеддинги.

Пусть $g_\phi: \mathcal{A} \rightarrow \mathbb{R}^d$ - энкодер атрибутов, отображающий атрибуты актива/режима в то же пространство эмбеддингов.

**Функция совместимости**:

$$F(x, a) = f_\theta(x)^T g_\phi(a)$$

Она измеряет совместимость между рыночными данными $x$ и атрибутами $a$.

**Zero-Shot прогноз**:

Для нового целевого класса $c$ с атрибутами $a_c$:

$$\hat{y} = \arg\max_{c \in \mathcal{C}_{new}} F(x, a_c) = \arg\max_{c} f_\theta(x)^T g_\phi(a_c)$$

### Перенос на основе атрибутов

Ключевая идея в том, что активы/режимы можно описать **семантическими атрибутами**:

```
+-------------------------------------------------------------------------+
|                    Описание атрибутов актива                              |
+-------------------------------------------------------------------------+
|                                                                           |
|   Bitcoin (BTC):                                                          |
|   - Тип актива: Криптовалюта                                             |
|   - Рыночная капитализация: Большая                                       |
|   - Волатильность: Высокая                                               |
|   - Корреляция с: Технологическими акциями, Risk-on активами             |
|   - Типичный дневной диапазон: 3-5%                                      |
|   - Часы торговли: 24/7                                                   |
|   - Ликвидность: Высокая                                                 |
|                                                                           |
|   Новый альткоин (неизвестный):                                          |
|   - Тип актива: Криптовалюта  <-- То же!                                 |
|   - Рыночная капитализация: Маленькая                                    |
|   - Волатильность: Очень высокая                                         |
|   - Корреляция с: BTC, Risk-on активы  <-- Похоже!                       |
|   - Типичный дневной диапазон: 10-20%                                    |
|   - Часы торговли: 24/7  <-- То же!                                      |
|   - Ликвидность: Низкая                                                  |
|                                                                           |
|   Сопоставляя атрибуты, модель переносит знания BTC на новый альткоин    |
|                                                                           |
+-------------------------------------------------------------------------+
```

### GMM мета-обучение для Zero-Shot прогнозирования

На основе недавних исследований (Liu et al., 2025), мощный подход использует:

1. **Обученные эмбеддинги**: Нейросеть обучается встраивать временные ряды в латентное пространство
2. **GMM кластеризация**: Gaussian Mixture Models мягко кластеризуют эмбеддинги в латентные режимы
3. **Двойное обучение задач**:
   - **Внутрикластерные задачи**: Изучение паттернов внутри похожих активов/режимов
   - **Межкластерные задачи**: Изучение переносимых паттернов между разными кластерами
4. **Поиск сложных задач**: Фокус на трудных межкластерных переносах для усиления обобщения

```
+-------------------------------------------------------------------------+
|                    GMM-based Zero-Shot архитектура                        |
+-------------------------------------------------------------------------+
|                                                                           |
|   Вход: Временной ряд рынка                                              |
|   [цена, объем, волатильность, ...]                                      |
|            |                                                              |
|            v                                                              |
|   +------------------+                                                    |
|   | Энкодер временных|                                                    |
|   | рядов f_theta    |                                                    |
|   +------------------+                                                    |
|            |                                                              |
|            v                                                              |
|   Эмбеддинг z в R^d                                                       |
|            |                                                              |
|            v                                                              |
|   +------------------+                                                    |
|   | GMM кластеризация|-----> K латентных кластеров (режимов)              |
|   +------------------+       c1, c2, ..., cK                              |
|            |                                                              |
|            v                                                              |
|   +------------------+     +------------------+                           |
|   | Внутрикластерные |     | Межкластерные    |                          |
|   | мета-задачи      |     | мета-задачи      |                          |
|   | (один режим)     |     | (кросс-режим)    |                          |
|   +------------------+     +------------------+                           |
|            |                        |                                     |
|            v                        v                                     |
|   +------------------------------------------+                           |
|   |    Комбинированное мета-обучение         |                           |
|   | Изучает локальные и глобальные паттерны  |                           |
|   +------------------------------------------+                           |
|            |                                                              |
|            v                                                              |
|   Zero-Shot прогноз для нового ряда                                       |
|                                                                           |
+-------------------------------------------------------------------------+
```

---

## Zero-Shot vs Few-Shot обучение

### Сравнительная структура

```
+-------------------------------------------------------------------------+
|               Zero-Shot vs Few-Shot для трейдинга                         |
+-------------------------------------------------------------------------+
|                                                                           |
|   Few-Shot обучение (напр., Prototypical Networks):                      |
|   -------------------------------------------------                       |
|   - Дано: 5-20 примеров нового актива/режима                             |
|   - Метод: Вычисляем прототип, классифицируем по расстоянию              |
|   - Сильная сторона: Может адаптироваться к новым паттернам              |
|   - Слабая сторона: Нужны хотя бы некоторые примеры                      |
|                                                                           |
|   Zero-Shot обучение:                                                     |
|   -------------------                                                     |
|   - Дано: Семантическое описание нового актива/режима                    |
|   - Метод: Сопоставление через общее пространство эмбеддингов            |
|   - Сильная сторона: Примеры вообще не нужны                             |
|   - Слабая сторона: Ограничено качеством атрибутов                       |
|                                                                           |
|   Гибридный подход (рекомендуется):                                      |
|   ----------------------------------                                      |
|   - Старт: Zero-shot прогноз для немедленной торговли                    |
|   - Развитие: Собираем примеры со временем                               |
|   - Улучшение: Переход к few-shot по мере накопления данных              |
|   - Лучшее из обоих миров!                                               |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Когда использовать каждый подход

| Сценарий | Рекомендуемый подход |
|----------|----------------------|
| Листинг нового токена | Zero-shot |
| Flash crash (внезапный режим) | Zero-shot |
| Новый рынок (forex в crypto) | Zero-shot, затем few-shot |
| Актив с 1 неделей данных | Few-shot |
| Актив с 1+ месяцем данных | Традиционный или few-shot |
| Кросс-активная стратегия | Zero-shot для инициализации |

---

## Архитектура

### Сеть Zero-Shot трейдинга

```
+-------------------------------------------------------------------------+
|                    Архитектура Zero-Shot трейдинга                        |
+-------------------------------------------------------------------------+
|                                                                           |
|   ЭНКОДЕР РЫНОЧНЫХ ДАННЫХ (f_theta)                                      |
|   ==================================                                      |
|   Вход: [цена, объем, волатильность, индикаторы]                         |
|   Форма: (batch, длина_последовательности, признаки)                     |
|                                                                           |
|   +-----------------------+                                               |
|   | Временной эмбеддинг   |                                               |
|   | - Conv1D слои         |                                               |
|   | - Позиционное кодир.  |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | Transformer Encoder   |                                               |
|   | - Self-attention      |                                               |
|   | - Feed-forward        |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | Проекционная голова   |                                               |
|   | - Линейные слои       |                                               |
|   | - L2 нормализация     |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   Рыночный эмбеддинг: z_market в R^d                                      |
|                                                                           |
|   ЭНКОДЕР АТРИБУТОВ (g_phi)                                              |
|   =========================                                               |
|   Вход: [тип_актива, класс_волатильности, корреляция, ...]               |
|                                                                           |
|   +-----------------------+                                               |
|   | Эмбеддинг атрибутов   |                                               |
|   | - Категориальный эмб. |                                               |
|   | - Числовое масштабир. |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   +-----------------------+                                               |
|   | MLP проекция          |                                               |
|   | - Скрытые слои        |                                               |
|   | - L2 нормализация     |                                               |
|   +-----------------------+                                               |
|            |                                                              |
|            v                                                              |
|   Эмбеддинг атрибутов: z_attr в R^d                                       |
|                                                                           |
|   ОЦЕНКА СОВМЕСТИМОСТИ                                                    |
|   ====================                                                    |
|   score = z_market . z_attr (скалярное произведение)                     |
|   prediction = softmax(scores по возможным классам)                      |
|                                                                           |
+-------------------------------------------------------------------------+
```

### Семантические атрибуты для трейдинга

```
+-------------------------------------------------------------------------+
|                    Атрибуты для Zero-Shot трейдинга                       |
+-------------------------------------------------------------------------+
|                                                                           |
|   Атрибуты уровня актива:                                                |
|   -----------------------                                                 |
|   - тип_актива: [crypto, акция, forex, товар]                           |
|   - класс_капитализации: [большая, средняя, малая, микро]               |
|   - режим_волатильности: [низкий, средний, высокий, экстремальный]      |
|   - класс_ликвидности: [высоколиквидный, ликвидный, неликвидный]        |
|   - сектор: [defi, layer1, layer2, meme, gaming, ...]                   |
|   - корреляция_btc: непрерывная [-1, 1]                                  |
|   - корреляция_sp500: непрерывная [-1, 1]                                |
|   - бета: непрерывная мера риска                                         |
|                                                                           |
|   Атрибуты уровня режима:                                                |
|   ------------------------                                                |
|   - тренд: [сильный_рост, слабый_рост, боковик, слабое_падение,         |
|             сильное_падение]                                              |
|   - состояние_волатильности: [сужающаяся, стабильная, расширяющаяся]    |
|   - профиль_объема: [накопление, распределение, нейтральный]            |
|   - настроение_рынка: [эйфория, оптимизм, нейтральное, страх, паника]   |
|   - ставка_финансирования: непрерывная                                   |
|   - тренд_открытого_интереса: [растет, плоский, падает]                 |
|                                                                           |
|   Временные атрибуты:                                                    |
|   --------------------                                                    |
|   - время_суток: [азиатская_сессия, европейская_сессия, us_сессия]      |
|   - день_недели: [понедельник, ..., пятница, выходные]                  |
|   - рыночное_событие: [отчетность, fomc, экспирация, обычный]           |
|                                                                           |
+-------------------------------------------------------------------------+
```

---

## Стратегия реализации

### Реализация на Python

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class MarketEncoder(nn.Module):
    """
    Кодирует данные временных рядов рынка в эмбеддинги.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Извлечение временных признаков
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Transformer для моделирования последовательности
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Проекция в пространство эмбеддингов
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Тензор рыночных данных формы (batch, seq_len, features)

        Returns:
            Тензор эмбеддинга формы (batch, embed_dim)
        """
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Извлечение признаков сверткой
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # x: (batch, hidden, seq_len) -> (batch, seq_len, hidden)
        x = x.transpose(1, 2)

        # Transformer кодирование
        x = self.transformer(x)

        # Глобальный средний пулинг
        x = x.mean(dim=1)

        # Проекция в пространство эмбеддингов
        x = self.projection(x)

        # L2 нормализация
        x = F.normalize(x, p=2, dim=1)

        return x


class AttributeEncoder(nn.Module):
    """
    Кодирует атрибуты актива/режима в эмбеддинги.
    """

    def __init__(
        self,
        categorical_dims: Dict[str, int],  # {имя_атрибута: число_категорий}
        numerical_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.categorical_dims = categorical_dims
        self.numerical_dim = numerical_dim

        # Категориальные эмбеддинги
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cats, hidden_dim // len(categorical_dims))
            for name, num_cats in categorical_dims.items()
        })

        # Обработка числовых признаков
        self.num_mlp = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Комбинированная проекция
        total_dim = (hidden_dim // len(categorical_dims)) * len(categorical_dims) + hidden_dim // 2
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(
        self,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            categorical_attrs: Dict, отображающий имена атрибутов в индексы категорий
            numerical_attrs: Тензор формы (batch, numerical_dim)

        Returns:
            Тензор эмбеддинга формы (batch, embed_dim)
        """
        # Эмбеддинг категориальных атрибутов
        cat_embeds = []
        for name in self.categorical_dims.keys():
            cat_embeds.append(self.cat_embeddings[name](categorical_attrs[name]))
        cat_embed = torch.cat(cat_embeds, dim=1)

        # Обработка числовых атрибутов
        num_embed = self.num_mlp(numerical_attrs)

        # Комбинирование и проекция
        combined = torch.cat([cat_embed, num_embed], dim=1)
        x = self.projection(combined)

        # L2 нормализация
        x = F.normalize(x, p=2, dim=1)

        return x


class ZeroShotTradingModel(nn.Module):
    """
    Полная модель zero-shot трейдинга с энкодерами рынка и атрибутов.
    """

    def __init__(
        self,
        market_input_dim: int,
        categorical_dims: Dict[str, int],
        numerical_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 0.1
    ):
        super().__init__()

        self.market_encoder = MarketEncoder(
            input_dim=market_input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim
        )

        self.attribute_encoder = AttributeEncoder(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )

        self.temperature = temperature

    def encode_market(self, market_data: torch.Tensor) -> torch.Tensor:
        """Кодирует рыночные данные в пространство эмбеддингов."""
        return self.market_encoder(market_data)

    def encode_attributes(
        self,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> torch.Tensor:
        """Кодирует атрибуты в пространство эмбеддингов."""
        return self.attribute_encoder(categorical_attrs, numerical_attrs)

    def compute_compatibility(
        self,
        market_embed: torch.Tensor,
        attr_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет оценки совместимости между рыночными и атрибутными эмбеддингами.
        """
        if attr_embed.dim() == 2:
            scores = torch.matmul(market_embed, attr_embed.T) / self.temperature
        else:
            scores = torch.bmm(
                attr_embed,
                market_embed.unsqueeze(-1)
            ).squeeze(-1) / self.temperature

        return scores

    def forward(
        self,
        market_data: torch.Tensor,
        categorical_attrs: Dict[str, torch.Tensor],
        numerical_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход для zero-shot прогноза.

        Returns:
            (оценки_совместимости, вероятности_предсказанных_классов)
        """
        market_embed = self.encode_market(market_data)
        attr_embed = self.encode_attributes(categorical_attrs, numerical_attrs)

        scores = self.compute_compatibility(market_embed, attr_embed)
        probs = F.softmax(scores, dim=-1)

        return scores, probs
```

---

## Интеграция с Bybit

### Получение данных для Zero-Shot трейдинга

```python
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class BybitZeroShotClient:
    """
    Клиент Bybit для сбора данных zero-shot трейдинга.
    Получает данные и вычисляет атрибуты для множества активов.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Получает OHLCV данные свечей с Bybit.
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        klines = data["result"]["list"]

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")

        return df.sort_values("timestamp").reset_index(drop=True)

    async def compute_asset_attributes(
        self,
        symbol: str,
        reference_symbols: List[str] = ["BTCUSDT", "ETHUSDT"]
    ) -> Dict:
        """
        Вычисляет семантические атрибуты актива для zero-shot обучения.

        Возвращает атрибуты: класс волатильности, корреляция и т.д.
        """
        # Получаем данные для целевого и эталонных активов
        tasks = [self.fetch_klines(symbol, "60", 168)]
        tasks.extend([self.fetch_klines(ref, "60", 168) for ref in reference_symbols])

        results = await asyncio.gather(*tasks)
        target_df = results[0]
        ref_dfs = results[1:]

        # Вычисляем доходности
        target_returns = target_df["close"].pct_change().dropna()
        ref_returns = [df["close"].pct_change().dropna() for df in ref_dfs]

        # Класс волатильности
        annualized_vol = target_returns.std() * np.sqrt(24 * 365)
        if annualized_vol < 0.3:
            volatility_class = "low"
        elif annualized_vol < 0.6:
            volatility_class = "medium"
        elif annualized_vol < 1.0:
            volatility_class = "high"
        else:
            volatility_class = "extreme"

        # Корреляции с эталонными активами
        min_len = min(len(target_returns), min(len(r) for r in ref_returns))
        correlations = {}
        for ref_name, ref_ret in zip(reference_symbols, ref_returns):
            corr = np.corrcoef(
                target_returns.iloc[-min_len:],
                ref_ret.iloc[-min_len:]
            )[0, 1]
            correlations[f"corr_{ref_name}"] = corr

        # Класс рыночной капитализации (аппроксимация по объему)
        avg_volume = target_df["turnover"].mean()
        if avg_volume > 1e9:
            market_cap_class = "large"
        elif avg_volume > 1e8:
            market_cap_class = "medium"
        elif avg_volume > 1e7:
            market_cap_class = "small"
        else:
            market_cap_class = "micro"

        # Определение тренда
        sma_20 = target_df["close"].rolling(20).mean().iloc[-1]
        sma_50 = target_df["close"].rolling(50).mean().iloc[-1]
        current_price = target_df["close"].iloc[-1]

        if current_price > sma_20 > sma_50:
            trend = "strong_up"
        elif current_price > sma_20:
            trend = "weak_up"
        elif current_price < sma_20 < sma_50:
            trend = "strong_down"
        elif current_price < sma_20:
            trend = "weak_down"
        else:
            trend = "sideways"

        return {
            "asset_type": "crypto",
            "volatility_class": volatility_class,
            "market_cap_class": market_cap_class,
            "trend": trend,
            "annualized_vol": annualized_vol,
            **correlations
        }
```

---

## Торговая стратегия

### Zero-Shot торговля на основе режимов

```python
class ZeroShotTradingStrategy:
    """
    Торговая стратегия с использованием zero-shot прогнозирования режима.
    """

    def __init__(
        self,
        model: ZeroShotTradingModel,
        regime_attributes: Dict[str, Tuple[Dict, np.ndarray]],
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            model: Обученная zero-shot модель
            regime_attributes: Dict, отображающий имена режимов в атрибуты
            confidence_threshold: Минимальная уверенность для торговли
        """
        self.model = model
        self.regime_attributes = regime_attributes
        self.confidence_threshold = confidence_threshold

        # Предвычисляем эмбеддинги режимов
        self._precompute_regime_embeddings()

    def predict_regime(
        self,
        market_data: torch.Tensor
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Прогнозирует рыночный режим с помощью zero-shot вывода.

        Returns:
            (предсказанный_режим, уверенность, все_вероятности_режимов)
        """
        self.model.eval()

        with torch.no_grad():
            market_embed = self.model.encode_market(market_data)

            similarities = {}
            for regime_name, regime_embed in self.regime_embeddings.items():
                sim = F.cosine_similarity(
                    market_embed,
                    regime_embed.unsqueeze(0)
                ).item()
                similarities[regime_name] = sim

        # Конвертируем в вероятности через softmax
        sim_values = list(similarities.values())
        exp_sims = np.exp(np.array(sim_values) / self.model.temperature)
        probs = exp_sims / exp_sims.sum()

        regime_probs = dict(zip(similarities.keys(), probs))

        predicted_regime = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[predicted_regime]

        return predicted_regime, confidence, regime_probs

    def generate_signal(
        self,
        market_data: torch.Tensor,
        current_position: float = 0.0
    ) -> Dict:
        """
        Генерирует торговый сигнал на основе zero-shot прогноза режима.
        """
        regime, confidence, regime_probs = self.predict_regime(market_data)

        # Маппинг режима в действие
        regime_actions = {
            "strong_uptrend": {"action": "long", "base_size": 1.0},
            "weak_uptrend": {"action": "long", "base_size": 0.5},
            "sideways": {"action": "neutral", "base_size": 0.0},
            "weak_downtrend": {"action": "short", "base_size": 0.5},
            "strong_downtrend": {"action": "short", "base_size": 1.0},
        }

        signal = regime_actions.get(regime, {"action": "neutral", "base_size": 0.0})

        # Корректировка размера на основе уверенности
        if confidence < self.confidence_threshold:
            signal["base_size"] *= 0.5

        # Вычисление целевой позиции
        if signal["action"] == "long":
            target_position = signal["base_size"] * confidence
        elif signal["action"] == "short":
            target_position = -signal["base_size"] * confidence
        else:
            target_position = 0.0

        position_change = target_position - current_position

        return {
            "regime": regime,
            "confidence": confidence,
            "regime_probabilities": regime_probs,
            "action": signal["action"],
            "target_position": target_position,
            "position_change": position_change,
            "reasoning": f"Zero-shot обнаружил режим {regime} с уверенностью {confidence:.1%}"
        }
```

---

## Управление рисками

### Специфичные риски Zero-Shot

```
+-------------------------------------------------------------------------+
|                    Риски Zero-Shot трейдинга                              |
+-------------------------------------------------------------------------+
|                                                                           |
|   1. Риск несоответствия атрибутов                                       |
|   =================================                                       |
|   Риск: Атрибуты актива вычислены неправильно                            |
|   Смягчение:                                                              |
|   - Использование нескольких методов оценки атрибутов                    |
|   - Требование порога уверенности атрибутов                              |
|   - Сравнение с похожими активами                                         |
|                                                                           |
|   2. Риск сдвига распределения                                           |
|   ============================                                            |
|   Риск: Новый актив фундаментально отличается от обучающих               |
|   Смягчение:                                                              |
|   - Мониторинг расстояний эмбеддингов до обучающего распределения        |
|   - Отметка выбросов для ручной проверки                                 |
|   - Использование квантификации неопределенности                         |
|                                                                           |
|   3. Прогнозы с низкой уверенностью                                      |
|   ==================================                                      |
|   Риск: Модель не уверена, но все равно торгует                          |
|   Смягчение:                                                              |
|   - Строгие пороги уверенности (напр., >60%)                            |
|   - Размер позиции пропорционален уверенности                            |
|   - Запрет торговли ниже минимальной уверенности                         |
|                                                                           |
|   4. Риск перехода режимов                                               |
|   =========================                                               |
|   Риск: Режим меняется быстрее, чем обнаруживается                       |
|   Смягчение:                                                              |
|   - Непрерывный мониторинг режима                                         |
|   - Стоп-лосс всегда активен                                             |
|   - Лимиты максимального времени удержания                               |
|                                                                           |
+-------------------------------------------------------------------------+
```

---

## Метрики производительности

### Система оценки

| Метрика | Описание | Цель |
|---------|----------|------|
| Zero-Shot точность | Классификация режима на невиданных активах | >60% |
| Скорость адаптации | Время до достижения 70% точности на новом активе | <24 часов |
| Коэффициент переноса | Производительность на новом активе / обученном | >0.7 |
| Sharpe Ratio | Доходность с поправкой на риск | >1.5 |
| Max Drawdown | Максимальная просадка | <15% |
| Sortino Ratio | Доходность с поправкой на downside риск | >2.0 |
| Win Rate | Процент прибыльных сделок | >55% |
| Profit Factor | Валовая прибыль / Валовый убыток | >1.3 |

---

## Ссылки

### Научные статьи

1. **Adapting to the Unknown: Robust Meta-Learning for Zero-Shot Financial Time Series Forecasting**
   - Liu, Ma, Zhang (2025)
   - URL: https://arxiv.org/abs/2504.09664
   - Ключевой вклад: GMM-based мета-обучение для zero-shot прогнозирования

2. **Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly**
   - Xian et al. (2018)
   - Ключевой вклад: Всесторонний бенчмарк и оценка ZSL

### Связанные главы

- [Глава 81: MAML для трейдинга](../81_maml_for_trading/)
- [Глава 82: Reptile алгоритм для трейдинга](../82_reptile_algorithm_trading/)
- [Глава 83: Prototypical Networks для финансов](../83_prototypical_networks_finance/)
- [Глава 84: Matching Networks для финансов](../84_matching_networks_finance/)
- [Глава 86: Few-Shot прогнозирование рынка](../86_few_shot_market_prediction/)

### Библиотеки и инструменты

- **PyTorch**: Фреймворк глубокого обучения
- **Bybit API**: Данные криптовалютных рынков
- **scikit-learn**: GMM и кластеризация
- **pandas/numpy**: Обработка данных

---

## Структура директории

```
85_zero_shot_trading/
├── README.md                    # Английская версия
├── README.ru.md                 # Этот файл
├── readme.simple.md             # Упрощенное объяснение (English)
├── readme.simple.ru.md          # Упрощенное объяснение (Russian)
├── Cargo.toml                   # Конфигурация Rust проекта
├── src/                         # Исходный код Rust
│   ├── lib.rs                   # Корень библиотеки
│   ├── model/                   # Реализации моделей
│   ├── data/                    # Обработка данных и клиент Bybit
│   ├── training/                # Логика обучения
│   ├── strategy/                # Торговая стратегия
│   └── backtest/                # Движок бэктестинга
├── python/                      # Реализация на Python
│   └── zero_shot_trading.py     # Основной Python модуль
└── examples/                    # Примеры скриптов
    ├── basic_zero_shot.rs       # Базовый пример Rust
    ├── multi_asset.rs           # Пример мульти-актив
    └── trading_strategy.rs      # Полный пример стратегии
```

---

## Быстрый старт

### Python

```python
import asyncio
from zero_shot_trading import (
    ZeroShotTradingModel,
    ZeroShotTradingStrategy,
    BybitZeroShotClient,
    prepare_features
)

async def main():
    # Инициализация модели
    model = ZeroShotTradingModel(
        market_input_dim=15,
        categorical_dims={
            "asset_type": 4,
            "volatility_class": 4,
            "market_cap_class": 4,
            "trend": 5
        },
        numerical_dim=3,
        embed_dim=64
    )

    # Загрузка предобученных весов
    model.load_state_dict(torch.load("zero_shot_trading.pth"))

    # Определение атрибутов режимов
    regime_attributes = {
        "strong_uptrend": (
            {"trend": 0, "volatility_class": 2},
            np.array([0.7, 0.8, 0.6])
        ),
        # ... другие режимы
    }

    # Создание стратегии
    strategy = ZeroShotTradingStrategy(
        model=model,
        regime_attributes=regime_attributes,
        confidence_threshold=0.6
    )

    # Получение данных для нового актива
    async with BybitZeroShotClient() as client:
        df = await client.fetch_klines("NEWTOKEN", "60", 100)
        attrs = await client.compute_asset_attributes("NEWTOKEN")

    # Подготовка признаков и прогноз
    features = prepare_features(df)
    signal = strategy.generate_signal(features)

    print(f"Режим: {signal['regime']}")
    print(f"Уверенность: {signal['confidence']:.1%}")
    print(f"Действие: {signal['action']}")
    print(f"Обоснование: {signal['reasoning']}")

asyncio.run(main())
```

### Rust

```rust
use zero_shot_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Инициализация модели
    let model = ZeroShotModel::load("model.bin")?;

    // Получение данных с Bybit
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "1h", 100).await?;

    // Вычисление атрибутов
    let attrs = compute_asset_attributes(&klines)?;

    // Генерация признаков и прогноз
    let features = prepare_features(&klines);
    let prediction = model.predict_regime(&features, &attrs)?;

    println!("Предсказанный режим: {:?}", prediction.regime);
    println!("Уверенность: {:.1}%", prediction.confidence * 100.0);

    Ok(())
}
```

---

*Эта глава является частью серии "Машинное обучение для трейдинга". По вопросам или предложениям, пожалуйста, откройте issue на GitHub.*
