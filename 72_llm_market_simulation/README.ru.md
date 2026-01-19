# Глава 72: Симуляция рынка с помощью LLM — Тестирование финансовых теорий с AI-агентами

## Обзор

Симуляция рынка с помощью LLM (Large Language Models) использует большие языковые модели в качестве гетерогенных торговых агентов в реалистичных симулированных рынках акций. Этот подход позволяет тестировать финансовые теории, изучать рыночную динамику и исследовать поведение агентов без затрат на реальные рынки и ограничений, связанных с участием людей.

<p align="center">
<img src="https://i.imgur.com/YwKvZ3Q.png" width="70%">
</p>

## Содержание

1. [Что такое симуляция рынка с LLM](#что-такое-симуляция-рынка-с-llm)
   * [Мотивация и предпосылки](#мотивация-и-предпосылки)
   * [Ключевые концепции](#ключевые-концепции)
   * [Типы агентов](#типы-агентов)
2. [Фреймворк симуляции](#фреймворк-симуляции)
   * [Механика рынка](#механика-рынка)
   * [Реализация книги ордеров](#реализация-книги-ордеров)
   * [Принятие решений агентами](#принятие-решений-агентами)
3. [Стратегии агентов](#стратегии-агентов)
   * [Стоимостные инвесторы](#стоимостные-инвесторы)
   * [Моментум-трейдеры](#моментум-трейдеры)
   * [Маркет-мейкеры](#маркет-мейкеры)
4. [Рыночные явления](#рыночные-явления)
   * [Ценообразование](#ценообразование)
   * [Пузыри и обвалы](#пузыри-и-обвалы)
   * [Динамика ликвидности](#динамика-ликвидности)
5. [Примеры кода](#примеры-кода)
   * [Python реализация](#python-реализация)
   * [Rust реализация](#rust-реализация)
6. [Бэктестинг и анализ](#бэктестинг-и-анализ)
7. [Ресурсы](#ресурсы)

## Что такое симуляция рынка с LLM

Симуляция рынка с LLM создаёт искусственный финансовый рынок, где AI-агенты (работающие на больших языковых моделях) торгуют ценными бумагами на основе своих назначенных стратегий, доступной информации и рыночных условий. В отличие от традиционных агентных моделей с жёстко заданными правилами, LLM-агенты могут рассуждать о сложных сценариях и адаптировать своё поведение.

### Мотивация и предпосылки

Традиционные финансовые симуляции имеют ограничения:

1. **Жёсткие правила**: Классические агентные модели основаны на предопределённых правилах, которые не улавливают реальную сложность рынка
2. **Эксперименты с людьми**: Дорогие, трудоёмкие и сложно воспроизводимые
3. **Исторические данные**: Ограничены прошлыми событиями, не позволяют тестировать гипотетические сценарии

Симуляция на основе LLM предлагает:
- **Гибкое рассуждение**: Агенты могут интерпретировать сложные рыночные сценарии
- **Естественный язык**: Стратегии можно определять на обычном языке
- **Масштабируемость**: Быстрое тестирование тысяч сценариев
- **Воспроизводимость**: Точно такие же условия можно повторить

### Ключевые концепции

```
Компоненты симуляции рынка LLM:
├── Рыночная среда
│   ├── Книга ордеров (лимитные ордера, рыночные ордера)
│   ├── Механизм ценообразования
│   ├── Процесс дивидендов/фундаментальной стоимости
│   └── Распределение информации
├── LLM-агенты
│   ├── Промпт стратегии (стоимость, моментум, маркет-мейкер)
│   ├── Информационный набор (цены, новости, приватная инфо)
│   ├── Функция принятия решений (структурированный вывод)
│   └── Состояние портфеля (наличные, активы)
└── Движок симуляции
    ├── Временные шаги (дискретные или непрерывные)
    ├── Сопоставление ордеров
    ├── Расчёты
    └── Сбор метрик
```

### Типы агентов

Симуляция поддерживает несколько архетипов агентов:

| Тип агента | Стратегия | Используемая информация | Поведение |
|------------|-----------|------------------------|-----------|
| Стоимостной инвестор | Покупать недооценённое, продавать переоценённое | Фундаментальные показатели, дивиденды | Долгосрочный, контртрендовый |
| Моментум-трейдер | Следовать за трендами | История цен, объём | Краткосрочный, трендовый |
| Маркет-мейкер | Обеспечивать ликвидность | Книга ордеров, спред | Котировки bid/ask, управление запасами |
| Шумовой трейдер | Случайная торговля | Нет специфической | Добавляет рыночный шум |
| Информированный трейдер | Торговля на приватной инфо | Приватные сигналы | Стратегический тайминг |

## Фреймворк симуляции

### Механика рынка

Симуляция реализует реалистичную рыночную механику:

```python
class MarketEnvironment:
    """
    Симулированная рыночная среда с книгой ордеров и ценообразованием
    """
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        self.order_book = OrderBook(tick_size)
        self.current_price = initial_price
        self.fundamental_value = initial_price
        self.price_history = [initial_price]
        self.time_step = 0

    def submit_order(self, agent_id: str, order: Order) -> OrderResult:
        """
        Обработать ордер от агента

        Args:
            agent_id: Уникальный идентификатор агента
            order: Объект ордера (рыночный или лимитный)

        Returns:
            OrderResult с информацией об исполнении
        """
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(agent_id, order)
        else:
            return self._add_limit_order(agent_id, order)

    def update_fundamental(self, dividend: float = None, news: str = None):
        """Обновить фундаментальную стоимость на основе дивидендов или новостей"""
        if dividend:
            # Корректировка приведённой стоимости
            self.fundamental_value += dividend * 10  # Простой множитель

    def step(self):
        """Продвинуть симуляцию на один временной шаг"""
        self.time_step += 1
        self._update_mid_price()
        self.price_history.append(self.current_price)
```

### Реализация книги ордеров

Реалистичная книга лимитных ордеров с приоритетом цена-время:

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import heapq

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    order_type: OrderType
    side: Side
    quantity: int
    price: Optional[float] = None  # None для рыночных ордеров
    agent_id: str = ""
    timestamp: int = 0

@dataclass
class OrderResult:
    filled_quantity: int
    average_price: float
    remaining_quantity: int
    status: str  # "filled", "partial", "pending"

class OrderBook:
    """
    Книга лимитных ордеров с приоритетом цена-время
    """
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids = []  # Max heap (отрицательные цены)
        self.asks = []  # Min heap
        self.order_id = 0

    def add_limit_order(self, order: Order) -> OrderResult:
        """Добавить лимитный ордер в книгу"""
        self.order_id += 1

        if order.side == Side.BUY:
            # Проверка на немедленное сопоставление с asks
            filled_qty, avg_price = self._match_against_asks(order)
            if filled_qty < order.quantity:
                # Добавить остаток на сторону bid
                remaining = order.quantity - filled_qty
                heapq.heappush(self.bids,
                    (-order.price, self.order_id, remaining, order.agent_id))
                return OrderResult(filled_qty, avg_price, remaining, "partial")
            return OrderResult(filled_qty, avg_price, 0, "filled")
        else:
            # Проверка на немедленное сопоставление с bids
            filled_qty, avg_price = self._match_against_bids(order)
            if filled_qty < order.quantity:
                remaining = order.quantity - filled_qty
                heapq.heappush(self.asks,
                    (order.price, self.order_id, remaining, order.agent_id))
                return OrderResult(filled_qty, avg_price, remaining, "partial")
            return OrderResult(filled_qty, avg_price, 0, "filled")

    def get_best_bid(self) -> Optional[float]:
        """Получить лучшую цену bid"""
        if self.bids:
            return -self.bids[0][0]
        return None

    def get_best_ask(self) -> Optional[float]:
        """Получить лучшую цену ask"""
        if self.asks:
            return self.asks[0][0]
        return None

    def get_spread(self) -> Optional[float]:
        """Получить спред bid-ask"""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None
```

### Принятие решений агентами

LLM-агенты принимают решения через структурированные промпты и вызовы функций:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMAgent(ABC):
    """
    Базовый класс для торговых агентов на основе LLM
    """
    def __init__(self, agent_id: str, initial_cash: float,
                 strategy_prompt: str, llm_client):
        self.agent_id = agent_id
        self.cash = initial_cash
        self.holdings = 0
        self.strategy_prompt = strategy_prompt
        self.llm = llm_client
        self.trade_history = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Вернуть системный промпт для этого типа агента"""
        pass

    def make_decision(self, market_state: Dict[str, Any]) -> Order:
        """
        Использовать LLM для принятия торгового решения

        Args:
            market_state: Текущая рыночная информация

        Returns:
            Объект Order, представляющий решение
        """
        # Построить контекст для LLM
        context = self._build_context(market_state)

        # Запрос к LLM со структурированным выводом
        response = self.llm.create_completion(
            system=self.get_system_prompt(),
            user=context,
            functions=[{
                "name": "submit_order",
                "description": "Отправить торговый ордер",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "quantity": {"type": "integer", "minimum": 0},
                        "order_type": {"type": "string", "enum": ["market", "limit"]},
                        "limit_price": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["action", "quantity", "order_type", "reasoning"]
                }
            }]
        )

        return self._parse_response(response)


class ValueInvestorAgent(LLMAgent):
    """
    Стоимостной инвестор, покупающий недооценённые активы
    """
    def get_system_prompt(self) -> str:
        return """Вы агент-стоимостной инвестор в симулированном фондовом рынке.

Ваша стратегия:
1. Сравнивайте текущую цену с фундаментальной стоимостью
2. Покупайте, когда цена значительно ниже фундаментальной (>10% дисконт)
3. Продавайте, когда цена значительно выше фундаментальной (>10% премия)
4. Будьте терпеливы — не торгуйте на малых отклонениях
5. Учитывайте текущую аллокацию портфеля

Управление рисками:
- Никогда не инвестируйте более 30% наличных в одну сделку
- Поддерживайте резервы наличных для возможностей
- Учитывайте транзакционные издержки

Объясните ваши рассуждения перед принятием решения."""


class MomentumTraderAgent(LLMAgent):
    """
    Моментум-трейдер, следующий за трендами
    """
    def get_system_prompt(self) -> str:
        return """Вы агент-моментум трейдер в симулированном фондовом рынке.

Ваша стратегия:
1. Анализируйте недавние ценовые тренды (последние 5-10 периодов)
2. Покупайте при восходящем моментуме (растущие цены с объёмом)
3. Продавайте при нисходящем моментуме или развороте тренда
4. Используйте скользящие средние для определения трендов
5. Быстро фиксируйте убытки, давайте прибыли расти

Управление рисками:
- Установите мысленный стоп-лосс на 5% ниже входа
- Фиксируйте прибыль при 10-15% роста
- Не идите против тренда

Объясните ваши рассуждения перед принятием решения."""


class MarketMakerAgent(LLMAgent):
    """
    Маркет-мейкер, обеспечивающий ликвидность
    """
    def get_system_prompt(self) -> str:
        return """Вы агент-маркет мейкер в симулированном фондовом рынке.

Ваша стратегия:
1. Обеспечивайте ликвидность, выставляя ордера bid и ask
2. Зарабатывайте на спреде bid-ask
3. Управляйте риском запасов — избегайте накопления больших позиций
4. Корректируйте котировки на основе условий рынка и запасов

Управление рисками:
- Поддерживайте запасы близко к нейтральным
- Расширяйте спреды при высокой волатильности
- Уменьшайте размер в неопределённых рынках

Объясните ваши рассуждения перед принятием решения."""
```

## Стратегии агентов

### Стоимостные инвесторы

Стоимостные инвесторы сравнивают рыночные цены с фундаментальной стоимостью:

```python
def value_investor_decision_logic(
    current_price: float,
    fundamental_value: float,
    cash: float,
    holdings: int,
    discount_threshold: float = 0.10,
    premium_threshold: float = 0.10,
    max_position_pct: float = 0.30
) -> Dict[str, Any]:
    """
    Логика принятия решений стоимостного инвестора

    Args:
        current_price: Текущая рыночная цена
        fundamental_value: Оценка внутренней стоимости
        cash: Доступные наличные
        holdings: Текущие активы
        discount_threshold: Покупать при таком дисконте к стоимости
        premium_threshold: Продавать при такой премии к стоимости
        max_position_pct: Максимальный процент наличных на сделку

    Returns:
        Словарь решения с действием и параметрами
    """
    portfolio_value = cash + holdings * current_price

    # Вычисление разрыва стоимости
    value_gap = (fundamental_value - current_price) / fundamental_value

    if value_gap > discount_threshold:
        # Цена ниже фундаментальной — возможность ПОКУПКИ
        max_spend = cash * max_position_pct
        quantity = int(max_spend / current_price)

        if quantity > 0:
            return {
                "action": "buy",
                "quantity": quantity,
                "order_type": "limit",
                "limit_price": current_price * 0.99,
                "reasoning": f"Цена ${current_price:.2f} на {value_gap*100:.1f}% ниже фундаментальной ${fundamental_value:.2f}"
            }

    elif value_gap < -premium_threshold and holdings > 0:
        # Цена выше фундаментальной — возможность ПРОДАЖИ
        sell_quantity = min(holdings, int(holdings * 0.5))

        return {
            "action": "sell",
            "quantity": sell_quantity,
            "order_type": "limit",
            "limit_price": current_price * 1.01,
            "reasoning": f"Цена ${current_price:.2f} на {-value_gap*100:.1f}% выше фундаментальной ${fundamental_value:.2f}"
        }

    return {
        "action": "hold",
        "quantity": 0,
        "order_type": "market",
        "reasoning": f"Цена ${current_price:.2f} близка к фундаментальной ${fundamental_value:.2f}, действия не требуются"
    }
```

### Моментум-трейдеры

Моментум-трейдеры выявляют и следуют за ценовыми трендами:

```python
import numpy as np

def momentum_trader_decision_logic(
    price_history: List[float],
    current_price: float,
    cash: float,
    holdings: int,
    short_window: int = 5,
    long_window: int = 20,
    entry_threshold: float = 0.02,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15
) -> Dict[str, Any]:
    """
    Логика принятия решений моментум-трейдера

    Использует стратегию пересечения скользящих средних
    """
    if len(price_history) < long_window:
        return {
            "action": "hold",
            "quantity": 0,
            "order_type": "market",
            "reasoning": "Недостаточно истории цен для анализа моментума"
        }

    # Вычисление скользящих средних
    prices = np.array(price_history[-long_window:])
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices)

    # Сигнал моментума
    momentum = (short_ma - long_ma) / long_ma

    # Ценовой тренд (недавняя доходность)
    recent_return = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

    if momentum > entry_threshold and recent_return > 0:
        # Бычий моментум — ПОКУПКА
        max_spend = cash * 0.25
        quantity = int(max_spend / current_price)

        if quantity > 0:
            return {
                "action": "buy",
                "quantity": quantity,
                "order_type": "market",
                "reasoning": f"Бычий моментум: короткая MA ({short_ma:.2f}) > длинная MA ({long_ma:.2f})"
            }

    elif momentum < -entry_threshold and holdings > 0:
        # Медвежий моментум — ПРОДАЖА
        return {
            "action": "sell",
            "quantity": holdings,
            "order_type": "market",
            "reasoning": f"Медвежий моментум: короткая MA ({short_ma:.2f}) < длинная MA ({long_ma:.2f})"
        }

    return {
        "action": "hold",
        "quantity": 0,
        "order_type": "market",
        "reasoning": f"Нет явного сигнала моментума. Короткая MA: {short_ma:.2f}, Длинная MA: {long_ma:.2f}"
    }
```

### Маркет-мейкеры

Маркет-мейкеры обеспечивают ликвидность, котируя обе стороны:

```python
def market_maker_decision_logic(
    current_price: float,
    best_bid: float,
    best_ask: float,
    inventory: int,
    cash: float,
    volatility: float,
    target_spread_bps: int = 50,  # 0.5%
    max_inventory: int = 100
) -> List[Dict[str, Any]]:
    """
    Логика котирования маркет-мейкера

    Возвращает котировки bid и ask
    """
    orders = []

    # Корректировка спреда на волатильность и запасы
    base_spread = current_price * (target_spread_bps / 10000)
    volatility_adj = 1 + volatility * 2
    inventory_adj = abs(inventory) / max_inventory

    effective_spread = base_spread * volatility_adj * (1 + inventory_adj)
    half_spread = effective_spread / 2

    # Смещение котировок на основе запасов
    inventory_skew = (inventory / max_inventory) * half_spread * 0.5

    bid_price = current_price - half_spread - inventory_skew
    ask_price = current_price + half_spread - inventory_skew

    # Размер котировки на основе запасов
    base_size = 10
    bid_size = int(base_size * (1 - inventory / max_inventory)) if inventory < max_inventory else 0
    ask_size = int(base_size * (1 + inventory / max_inventory)) if inventory > -max_inventory else 0

    if bid_size > 0:
        orders.append({
            "action": "buy",
            "quantity": bid_size,
            "order_type": "limit",
            "limit_price": bid_price,
            "reasoning": f"Маркет-мейкинг bid: ликвидность по ${bid_price:.2f}"
        })

    if ask_size > 0:
        orders.append({
            "action": "sell",
            "quantity": ask_size,
            "order_type": "limit",
            "limit_price": ask_price,
            "reasoning": f"Маркет-мейкинг ask: ликвидность по ${ask_price:.2f}"
        })

    return orders
```

## Рыночные явления

### Ценообразование

Симулированные LLM-рынки демонстрируют реалистичное ценообразование:

```python
class PriceDiscoveryAnalyzer:
    """
    Анализ эффективности ценообразования в симулированных рынках
    """
    def __init__(self, market: MarketEnvironment):
        self.market = market

    def calculate_efficiency(self) -> Dict[str, float]:
        """
        Расчёт метрик эффективности ценообразования
        """
        prices = np.array(self.market.price_history)
        fundamental = self.market.fundamental_value

        # Ошибка отслеживания
        deviations = prices - fundamental
        tracking_error = np.std(deviations)

        # Скорость возврата к среднему (период полураспада)
        if len(prices) > 20:
            log_prices = np.log(prices / fundamental)
            autocorr = np.corrcoef(log_prices[:-1], log_prices[1:])[0, 1]
            half_life = -np.log(2) / np.log(autocorr) if autocorr > 0 else float('inf')
        else:
            half_life = None

        return {
            "tracking_error": tracking_error,
            "half_life": half_life,
            "final_deviation_pct": (prices[-1] - fundamental) / fundamental * 100
        }
```

### Пузыри и обвалы

Симуляция может генерировать и изучать рыночные пузыри:

```python
def detect_bubble(prices: List[float], fundamental_value: float,
                  bubble_threshold: float = 0.50) -> Dict[str, Any]:
    """
    Обнаружение формирования пузыря в ценовом ряде

    Args:
        prices: Исторические цены
        fundamental_value: Истинная фундаментальная стоимость
        bubble_threshold: Процент выше фундаментальной для квалификации как пузырь

    Returns:
        Словарь с анализом пузыря
    """
    prices = np.array(prices)
    deviation = (prices - fundamental_value) / fundamental_value

    # Найти периоды пузыря
    bubble_mask = deviation > bubble_threshold

    if not any(bubble_mask):
        return {"bubble_detected": False}

    return {
        "bubble_detected": True,
        "max_deviation": float(np.max(deviation)),
        "time_in_bubble_pct": float(np.mean(bubble_mask) * 100)
    }
```

## Примеры кода

### Python реализация

Python-реализация предоставляет полный фреймворк симуляции:

```
python/
├── market/
│   ├── __init__.py
│   ├── order_book.py      # Книга лимитных ордеров
│   ├── environment.py     # Рыночная среда
│   └── matching.py        # Движок сопоставления ордеров
├── agents/
│   ├── __init__.py
│   ├── base.py           # Базовый LLM-агент
│   ├── value.py          # Стоимостной инвестор
│   ├── momentum.py       # Моментум-трейдер
│   └── market_maker.py   # Маркет-мейкер
├── simulation/
│   ├── __init__.py
│   ├── engine.py         # Движок симуляции
│   ├── scenarios.py      # Готовые сценарии
│   └── metrics.py        # Метрики производительности
├── data/
│   ├── __init__.py
│   ├── bybit.py          # Загрузчик данных Bybit
│   └── yahoo.py          # Данные Yahoo Finance
└── examples/
    ├── basic_simulation.py
    ├── bubble_formation.py
    └── multi_agent.py
```

### Rust реализация

Rust-реализация обеспечивает высокопроизводительную симуляцию:

```
rust_llm_market/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главная библиотека
│   ├── main.rs             # CLI интерфейс
│   ├── market/
│   │   ├── mod.rs
│   │   ├── order_book.rs   # Реализация книги ордеров
│   │   └── matching.rs     # Движок сопоставления
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── traits.rs       # Трейт агента
│   │   ├── value.rs        # Стоимостной инвестор
│   │   └── momentum.rs     # Моментум-трейдер
│   ├── simulation/
│   │   ├── mod.rs
│   │   ├── engine.rs       # Движок симуляции
│   │   └── config.rs       # Конфигурация
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Клиент Bybit API
│   │   └── types.rs        # Типы данных
│   └── metrics/
│       ├── mod.rs
│       └── performance.rs  # Метрики производительности
└── examples/
    ├── run_simulation.rs
    └── fetch_data.rs
```

### Быстрый старт

**Python:**
```bash
cd 72_llm_market_simulation/python

# Установка зависимостей
pip install -r requirements.txt

# Запуск базовой симуляции
python examples/basic_simulation.py

# Запуск с несколькими агентами
python examples/multi_agent.py --agents 10 --steps 1000
```

**Rust:**
```bash
cd 72_llm_market_simulation/rust_llm_market

# Сборка и запуск
cargo build --release
cargo run --release -- --agents 10 --steps 1000

# Запуск примеров
cargo run --example run_simulation
```

## Бэктестинг и анализ

### Ключевые метрики

| Категория | Метрика | Описание |
|-----------|---------|----------|
| Доходность | CAGR | Среднегодовой темп роста |
| | Общая доходность | Кумулятивная доходность |
| Риск | Волатильность | Стандартное отклонение доходности |
| | Макс. просадка | Наибольшее падение от пика |
| | VaR | Стоимость под риском |
| Эффективность | Sharpe Ratio | Доходность с поправкой на риск |
| | Sortino Ratio | Доходность с поправкой на нисходящий риск |
| Качество рынка | Спред | Спред bid-ask |
| | Глубина | Глубина книги ордеров |

### Пример результатов

| Тип агента | CAGR | Волатильность | Sharpe | Макс. просадка |
|------------|------|---------------|--------|----------------|
| Стоимостной инвестор | 12% | 15% | 0.80 | -18% |
| Моментум-трейдер | 18% | 25% | 0.72 | -30% |
| Маркет-мейкер | 8% | 8% | 1.00 | -10% |
| Комбинированный | 15% | 12% | 1.25 | -15% |

*Примечание: Результаты симуляции не представляют реальную торговую производительность*

## Ресурсы

### Академические статьи

- [Can Large Language Models Trade? Testing Financial Theories with LLM Agents](https://arxiv.org/abs/2504.10789) (Lopez-Lira, 2025)
- [Agent-Based Models of Financial Markets](https://www.annualreviews.org/doi/abs/10.1146/annurev.financial.1.1.61) (LeBaron, 2006)
- [Market Microstructure Theory](https://www.amazon.com/Market-Microstructure-Theory-Maureen-OHara/dp/0631207619) (O'Hara, 1995)

### Книги

- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) (Marcos Lopez de Prado)
- [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715) (Stefan Jansen)
- [Trading and Exchanges](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708) (Larry Harris)

### Связанные главы

- [Глава 64: Multi-Agent LLM Trading](../64_multi_agent_llm_trading) - Мульти-агентные системы
- [Глава 65: RAG for Trading](../65_rag_for_trading) - Retrieval-augmented generation
- [Глава 22: Deep Reinforcement Learning](../22_deep_reinforcement_learning) - RL для трейдинга

## Зависимости

### Python

```
openai>=1.0.0          # LLM API клиент
anthropic>=0.18.0      # Альтернативный LLM
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.28.0
aiohttp>=3.8.0         # Async HTTP
pydantic>=2.0.0        # Валидация данных
```

### Rust

```toml
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
ndarray = "0.16"
rand = "0.8"
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
```

## Уровень сложности

Продвинутый

**Требуется понимание:**
- Большие языковые модели и промптинг
- Рыночная микроструктура
- Агентное моделирование
- Механика книги ордеров
- Торговые стратегии
- Асинхронное программирование (для вызовов LLM)
