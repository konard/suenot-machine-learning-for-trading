# Глава 356: DenseNet для Криптовалютной Торговли

## Обзор

DenseNet (Densely Connected Convolutional Networks — плотно связанные свёрточные сети) представляет прорывную архитектуру, в которой каждый слой соединён с каждым последующим слоем напрямую. Для торговых приложений эта плотная связность обеспечивает исключительное переиспользование признаков, улучшенный поток градиентов и замечательно эффективное использование параметров — критически важные преимущества при извлечении сложных паттернов из данных криптовалютных рынков.

## Торговая Стратегия

**Основная концепция:** Применение паттерна плотной связности DenseNet для извлечения иерархических признаков из криптовалютных данных OHLCV и динамики ордербука. Архитектура обрабатывает временные ряды как структурированные входные данные, где признаки ранних слоёв (краткосрочные паттерны) напрямую участвуют в предсказаниях поздних слоёв (долгосрочные тренды).

**Ключевые преимущества для трейдинга:**
1. **Переиспользование признаков** — Рано обнаруженные паттерны (уровни поддержки/сопротивления, свечные формации) остаются доступными для всех последующих слоёв
2. **Поток градиентов** — Глубокие сети эффективно обучаются без затухания градиентов
3. **Эффективность параметров** — Меньше параметров чем у ResNet при сохранении производительности
4. **Неявный ансамбль** — Плотные связи создают эффект неявного глубокого наблюдения

**Преимущество:** Архитектура DenseNet естественно захватывает мультимасштабные временные зависимости в рыночных данных, от микроструктуры на уровне тиков до дневных трендов.

## Техническая Спецификация

### Компоненты Архитектуры

```
┌─────────────────────────────────────────────────────────────────┐
│                  Архитектура DenseNet для Трейдинга             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Вход: OHLCV + Технические Индикаторы + Признаки Ордербука      │
│         [batch_size, sequence_length, features]                  │
│                           │                                      │
│                           ▼                                      │
│              ┌─────────────────────────┐                        │
│              │   Начальная Свёртка     │                        │
│              │      (7x1, stride 2)    │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │    Плотный Блок 1       │ ◄── Growth Rate k=32   │
│              │    (6 плотных слоёв)    │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │   Переходный Слой 1     │ ◄── Сжатие θ=0.5       │
│              │  (1x1 conv + AvgPool)   │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │    Плотный Блок 2       │                        │
│              │   (12 плотных слоёв)    │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │   Переходный Слой 2     │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │    Плотный Блок 3       │                        │
│              │   (24 плотных слоя)     │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │ Глобальное Усреднение   │                        │
│              └───────────┬─────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌─────────────────────────┐                        │
│              │   Торговая Голова       │                        │
│              │ (Позиция: -1, 0, +1)    │                        │
│              └─────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Механизм Плотного Блока

```python
class DenseLayer:
    """
    Единичный слой внутри Плотного Блока
    H_l(x) = BN → ReLU → Conv(3x1) → Dropout

    Каждый слой получает ВСЕ предыдущие карты признаков на вход
    """
    def __init__(self, in_channels, growth_rate, dropout=0.2):
        self.bn = BatchNorm1d(in_channels)
        self.conv = Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x — это конкатенация всех предыдущих выходов
        out = self.dropout(self.conv(relu(self.bn(x))))
        return out


class DenseBlock:
    """
    Плотный Блок: каждый слой соединён со всеми предыдущими

    Слой 0: x_0 → H_0(x_0) = y_0
    Слой 1: [x_0, y_0] → H_1([x_0, y_0]) = y_1
    Слой 2: [x_0, y_0, y_1] → H_2([x_0, y_0, y_1]) = y_2
    ...
    """
    def __init__(self, n_layers, in_channels, growth_rate):
        self.layers = []
        for i in range(n_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Конкатенируем ВСЕ предыдущие карты признаков
            concat_features = torch.cat(features, dim=1)
            new_features = layer(concat_features)
            features.append(new_features)
        return torch.cat(features, dim=1)
```

### Переходный Слой

```python
class TransitionLayer:
    """
    Уменьшает карты признаков между Плотными Блоками
    Коэффициент сжатия θ уменьшает каналы: out = floor(θ * in_channels)
    """
    def __init__(self, in_channels, compression=0.5):
        out_channels = int(in_channels * compression)
        self.bn = BatchNorm1d(in_channels)
        self.conv = Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(relu(self.bn(x))))
```

### Торговый DenseNet

```python
class DenseNetTrader:
    """
    Архитектура DenseNet, адаптированная для криптовалютной торговли
    """
    def __init__(
        self,
        input_features: int = 32,      # OHLCV + индикаторы + ордербук
        sequence_length: int = 128,     # Окно ретроспективы
        growth_rate: int = 32,          # k: новых признаков на слой
        block_config: tuple = (6, 12, 24),  # Слоёв на блок
        compression: float = 0.5,       # θ: уменьшение каналов
        num_classes: int = 3,           # Long, Hold, Short
        dropout: float = 0.2
    ):
        # Начальная свёртка
        self.init_conv = Conv1d(input_features, 64, kernel_size=7, stride=2, padding=3)
        self.init_bn = BatchNorm1d(64)
        self.init_pool = MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Плотные Блоки и Переходы
        self.blocks = []
        num_features = 64

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, compression)
                self.blocks.append(trans)
                num_features = int(num_features * compression)

        # Финальные слои
        self.final_bn = BatchNorm1d(num_features)
        self.global_pool = AdaptiveAvgPool1d(1)

        # Торговая голова
        self.classifier = Sequential(
            Linear(num_features, 128),
            ReLU(),
            Dropout(dropout),
            Linear(128, num_classes)
        )

        # Голова управления рисками
        self.risk_head = Sequential(
            Linear(num_features, 64),
            ReLU(),
            Linear(64, 2)  # Предсказание волатильности, Размер позиции
        )

    def forward(self, x):
        # x: [batch, features, sequence]

        # Начальная обработка
        x = self.init_pool(relu(self.init_bn(self.init_conv(x))))

        # Плотные блоки
        for block in self.blocks:
            x = block(x)

        # Финальная обработка
        x = relu(self.final_bn(x))
        x = self.global_pool(x).squeeze(-1)

        # Торговые выходы
        position = self.classifier(x)
        risk = self.risk_head(x)

        return {
            'position': softmax(position, dim=-1),
            'volatility': relu(risk[:, 0]),
            'position_size': sigmoid(risk[:, 1])
        }
```

### Извлечение Признаков для Крипты

```python
class CryptoFeatureExtractor:
    """
    Извлечение признаков из криптовалютных данных Bybit
    """
    def __init__(self, sequence_length=128):
        self.seq_len = sequence_length

    def extract(self, ohlcv_data, orderbook_data):
        features = {}

        # Ценовые признаки
        features['returns'] = np.log(ohlcv_data['close'] / ohlcv_data['close'].shift(1))
        features['high_low_range'] = (ohlcv_data['high'] - ohlcv_data['low']) / ohlcv_data['close']
        features['close_position'] = (ohlcv_data['close'] - ohlcv_data['low']) / (ohlcv_data['high'] - ohlcv_data['low'])

        # Объёмные признаки
        features['volume_ma_ratio'] = ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()
        features['volume_std'] = ohlcv_data['volume'].rolling(20).std() / ohlcv_data['volume'].rolling(20).mean()

        # Признаки волатильности
        features['realized_vol'] = features['returns'].rolling(20).std() * np.sqrt(365 * 24)
        features['parkinson_vol'] = self._parkinson_volatility(ohlcv_data)

        # Моментум индикаторы
        features['rsi'] = self._compute_rsi(ohlcv_data['close'], 14)
        features['macd'], features['macd_signal'] = self._compute_macd(ohlcv_data['close'])

        # Признаки ордербука
        features['bid_ask_spread'] = (orderbook_data['best_ask'] - orderbook_data['best_bid']) / orderbook_data['mid_price']
        features['order_imbalance'] = (orderbook_data['bid_volume'] - orderbook_data['ask_volume']) / (orderbook_data['bid_volume'] + orderbook_data['ask_volume'])
        features['depth_imbalance'] = self._compute_depth_imbalance(orderbook_data)

        # Ставка финансирования (бессрочные фьючерсы)
        features['funding_rate'] = orderbook_data.get('funding_rate', 0)

        # Динамика открытого интереса
        features['oi_change'] = orderbook_data['open_interest'].pct_change()

        return self._normalize_and_stack(features)

    def _parkinson_volatility(self, ohlcv, window=20):
        """Оценка волатильности по максимуму-минимуму"""
        log_hl = np.log(ohlcv['high'] / ohlcv['low'])
        return np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean())

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _compute_depth_imbalance(self, orderbook, levels=5):
        """Вычисление дисбаланса глубины ордербука по нескольким уровням"""
        bid_depth = sum(orderbook[f'bid_vol_{i}'] for i in range(levels))
        ask_depth = sum(orderbook[f'ask_vol_{i}'] for i in range(levels))
        return (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
```

### Мультимасштабный DenseNet

```python
class MultiScaleDenseNet:
    """
    DenseNet с входами нескольких временных масштабов
    для комплексного анализа рынка
    """
    def __init__(self, scales=[1, 5, 15, 60]):  # 1м, 5м, 15м, 1ч
        self.scales = scales
        self.encoders = {
            scale: DenseNetEncoder(growth_rate=24)
            for scale in scales
        }

        # Слой слияния
        total_features = len(scales) * 256
        self.fusion = Sequential(
            Linear(total_features, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(0.3),
            Linear(512, 256)
        )

        # Мультизадачные головы
        self.direction_head = Linear(256, 3)  # Long, Hold, Short
        self.magnitude_head = Linear(256, 1)  # Ожидаемая доходность
        self.confidence_head = Linear(256, 1) # Уверенность предсказания

    def forward(self, multi_scale_data):
        """
        multi_scale_data: словарь {масштаб: тензор}
        """
        # Кодируем каждый временной масштаб
        encoded = []
        for scale in self.scales:
            x = multi_scale_data[scale]
            enc = self.encoders[scale](x)
            encoded.append(enc)

        # Конкатенируем и объединяем
        combined = torch.cat(encoded, dim=-1)
        fused = self.fusion(combined)

        return {
            'direction': softmax(self.direction_head(fused), dim=-1),
            'magnitude': self.magnitude_head(fused),
            'confidence': sigmoid(self.confidence_head(fused))
        }
```

### Пайплайн Обучения

```python
class DenseNetTradingTrainer:
    """
    Пайплайн обучения с правильными функциями потерь для трейдинга
    """
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            features, labels = batch
            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self._compute_loss(outputs, labels)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(train_loader)

    def _compute_loss(self, outputs, labels):
        """
        Мультизадачный лосс, объединяющий:
        1. Классификация направления (кросс-энтропия)
        2. Предсказание доходности (MSE взвешенный по уверенности)
        3. Максимизация коэффициента Шарпа (кастомный)
        """
        # Лосс направления
        direction_loss = cross_entropy(
            outputs['position'],
            labels['direction'],
            weight=torch.tensor([1.0, 0.5, 1.0])  # Взвешиваем классы действий
        )

        # Лосс предсказания доходности (только для не-hold позиций)
        mask = labels['direction'] != 1
        if mask.any():
            pred_returns = outputs['magnitude'][mask]
            true_returns = labels['future_return'][mask]
            return_loss = mse_loss(pred_returns, true_returns)
        else:
            return_loss = 0

        # Лосс на основе Шарпа
        sharpe_loss = self._sharpe_loss(outputs, labels)

        # Комбинируем лоссы
        total_loss = direction_loss + 0.5 * return_loss + 0.3 * sharpe_loss

        return total_loss

    def _sharpe_loss(self, outputs, labels, eps=1e-8):
        """
        Дифференцируемый лосс коэффициента Шарпа
        """
        # Предсказанные позиции (-1, 0, 1)
        positions = outputs['position'][:, 2] - outputs['position'][:, 0]

        # Доходности стратегии
        strategy_returns = positions * labels['future_return']

        # Отрицательный Шарп (для минимизации)
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std() + eps

        return -mean_return / std_return
```

### Интеграция Бэктестинга

```python
class DenseNetBacktester:
    """
    Фреймворк бэктестинга для торговых сигналов DenseNet
    """
    def __init__(self, model, fee_rate=0.0004):  # Комиссия тейкера Bybit
        self.model = model
        self.fee_rate = fee_rate

    def run_backtest(self, data, initial_capital=10000):
        self.model.eval()

        results = {
            'timestamps': [],
            'positions': [],
            'equity': [],
            'returns': [],
            'trades': []
        }

        capital = initial_capital
        position = 0
        entry_price = 0

        for i in range(len(data) - 1):
            features = self._prepare_features(data, i)

            with torch.no_grad():
                output = self.model(features.unsqueeze(0))

            signal = self._get_signal(output)
            current_price = data['close'].iloc[i]
            next_price = data['close'].iloc[i + 1]

            # Исполняем сделку если сигнал изменился
            if signal != position:
                # Закрываем существующую позицию
                if position != 0:
                    pnl = position * (current_price - entry_price) / entry_price
                    fee = abs(position) * self.fee_rate * 2
                    capital *= (1 + pnl - fee)

                    results['trades'].append({
                        'exit_time': data.index[i],
                        'exit_price': current_price,
                        'pnl': pnl - fee
                    })

                # Открываем новую позицию
                if signal != 0:
                    entry_price = current_price
                    results['trades'].append({
                        'entry_time': data.index[i],
                        'entry_price': current_price,
                        'direction': signal
                    })

                position = signal

            # Переоценка по рынку
            if position != 0:
                unrealized_pnl = position * (current_price - entry_price) / entry_price
                equity = capital * (1 + unrealized_pnl)
            else:
                equity = capital

            results['timestamps'].append(data.index[i])
            results['positions'].append(position)
            results['equity'].append(equity)

        return self._compute_metrics(results)

    def _get_signal(self, output, threshold=0.4):
        """Конвертируем выход модели в торговый сигнал"""
        probs = output['position'][0]
        confidence = output['confidence'][0].item()

        if confidence < 0.3:
            return 0  # Hold если не уверены

        if probs[2] > threshold:
            return 1   # Long
        elif probs[0] > threshold:
            return -1  # Short
        return 0       # Hold

    def _compute_metrics(self, results):
        """Вычисление метрик торговой эффективности"""
        equity = np.array(results['equity'])
        returns = np.diff(equity) / equity[:-1]

        metrics = {
            'total_return': (equity[-1] / equity[0]) - 1,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365 * 24),
            'max_drawdown': self._max_drawdown(equity),
            'win_rate': self._win_rate(results['trades']),
            'profit_factor': self._profit_factor(results['trades']),
            'num_trades': len([t for t in results['trades'] if 'pnl' in t])
        }

        return metrics

    def _max_drawdown(self, equity):
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return np.max(drawdown)

    def _win_rate(self, trades):
        closed_trades = [t for t in trades if 'pnl' in t]
        if not closed_trades:
            return 0
        wins = sum(1 for t in closed_trades if t['pnl'] > 0)
        return wins / len(closed_trades)

    def _profit_factor(self, trades):
        closed_trades = [t for t in trades if 'pnl' in t]
        profits = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        return profits / (losses + 1e-8)
```

### Ключевые Метрики

- **Производительность модели:** Accuracy, F1-score, AUC-ROC для предсказания направления
- **Торговая эффективность:** Коэффициент Шарпа, Коэффициент Сортино, Максимальная просадка
- **Метрики эффективности:** Количество параметров, Время инференса, Использование памяти
- **Метрики риска:** VaR, Ожидаемый дефицит, Винрейт

### Варианты DenseNet для Трейдинга

| Вариант | Блоки | Growth Rate | Параметры | Применение |
|---------|-------|-------------|-----------|------------|
| DenseNet-Tiny | (4, 4, 4) | 16 | ~0.5M | HFT в реальном времени |
| DenseNet-Small | (6, 12, 8) | 24 | ~1.5M | Внутридневная торговля |
| DenseNet-Medium | (6, 12, 24) | 32 | ~4M | Свинг-трейдинг |
| DenseNet-Large | (6, 12, 32, 24) | 48 | ~12M | Исследования |

### Зависимости

```toml
[dependencies]
# Ядро ML
torch = ">=2.0.0"
numpy = ">=1.23.0"

# Обработка данных
pandas = ">=2.0.0"
polars = ">=0.19.0"

# API Bybit
pybit = ">=5.6.0"

# Технический анализ
ta-lib = ">=0.4.26"
pandas-ta = ">=0.3.14"

# Визуализация
matplotlib = ">=3.6.0"
plotly = ">=5.18.0"
```

## Ожидаемые Результаты

1. **Торговая модель DenseNet** — Оптимизированная архитектура для предсказания цен криптовалют
2. **Мультимасштабное извлечение признаков** — Комплексное представление рынка
3. **Пайплайн торговли в реальном времени** — Интеграция с биржей Bybit
4. **Фреймворк бэктестинга** — Строгая оценка эффективности
5. **Модуль управления рисками** — Автоматизация размера позиции и стоп-лоссов

## Ссылки

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Оригинальная статья DenseNet
- [Deep Learning for Financial Time Series](https://arxiv.org/abs/1901.08280)
- [Machine Learning for Asset Managers](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545)
- [Документация API Bybit](https://bybit-exchange.github.io/docs/)

## Уровень Сложности

⭐⭐⭐⭐ (Продвинутый)

**Требуемые знания:** Основы глубокого обучения, Архитектуры CNN, Анализ временных рядов, Микроструктура криптовалютных рынков
