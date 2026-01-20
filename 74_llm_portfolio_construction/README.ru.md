# Глава 74: Построение портфеля с помощью LLM

## Обзор

Большие языковые модели (LLM) могут революционизировать построение портфелей, анализируя разнообразные источники данных — новости, отчёты о доходах, рыночные комментарии и фундаментальные данные — для генерации интеллектуальных рекомендаций по распределению активов. Эта глава исследует использование LLM для построения и управления инвестиционными портфелями, сочетая понимание естественного языка с количественными методами оптимизации.

## Торговая стратегия

**Основная концепция:** LLM обрабатывают финансовые документы, новостные настроения и рыночные данные для генерации весов портфеля, рекомендаций по активам и сигналов ребалансировки.

**Сигналы входа:**
- Длинная позиция: Позитивные настроения + благоприятные фундаментальные показатели, выявленные LLM
- Увеличение веса: LLM идентифицирует недооценённые активы с катализаторами роста
- Снижение веса: LLM обнаруживает ухудшающиеся фундаментальные показатели или негативные настроения

**Преимущество:** LLM могут синтезировать огромные объёмы неструктурированных данных (звонки о доходах, новости, отчёты SEC) в действенные портфельные рекомендации быстрее, чем человеческие аналитики, выявляя тонкие паттерны и межактивные взаимосвязи.

## Техническая спецификация

### Ключевые компоненты

1. **Конвейер загрузки данных** - Сбор рыночных данных, новостей и фундаментальных данных
2. **Движок анализа LLM** - Генерация оценок активов и рекомендаций
3. **Оптимизатор портфеля** - Преобразование инсайтов LLM в оптимальные веса
4. **Управление рисками** - Построение портфеля на основе ограничений
5. **Фреймворк бэктестинга** - Валидация производительности стратегии

### Архитектура

```
                    ┌─────────────────────┐
                    │   Источники данных  │
                    │ (Новости, отчёты,   │
                    │  рыночные данные)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Обработчик текста │
                    │ (Суммаризация,      │
                    │  извлечение)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Портфельный       │
                    │   движок LLM        │
                    │ (Анализ + Рек.)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Оптимизатор портфеля│
                    │ (Mean-Variance,     │
                    │  Risk Parity)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Слой исполнения   │
                    │ (Ордера + Ребаланс) │
                    └─────────────────────┘
```

### Требования к данным

```
Рыночные данные:
├── OHLCV ценовые данные (Bybit для крипты, Yahoo для акций)
├── Объём торгов и метрики ликвидности
├── Волатильность и корреляционные данные
└── Бенчмарк-индексы

Фундаментальные данные:
├── Отчёты о доходах и прогнозы
├── Отчёты SEC (10-K, 10-Q, 8-K)
├── Оценки и ревизии аналитиков
└── Финансовые коэффициенты и метрики

Альтернативные данные:
├── Новостные статьи и заголовки
├── Настроения в социальных сетях
├── Транскрипты звонков о доходах
└── Макроэкономические индикаторы
```

### Подходы к портфелю на основе LLM

LLM может использоваться несколькими способами для построения портфеля:

| Подход | Описание | Применение |
|--------|----------|------------|
| **Прямое распределение** | LLM выдаёт веса портфеля напрямую | Простой, интерпретируемый |
| **Скоринг + Оптимизация** | LLM оценивает активы, оптимизатор устанавливает веса | Сочетание инсайтов LLM с математикой |
| **Мульти-агентный ансамбль** | Несколько персон LLM голосуют за распределение | Робастный, разнообразные перспективы |
| **С усилением RAG** | LLM извлекает релевантные данные перед решением | Доступ к информации в реальном времени |

### Инженерия промптов для построения портфеля

```python
PORTFOLIO_CONSTRUCTION_PROMPT = """
Вы количественный портфельный менеджер. Проанализируйте следующие активы и рыночные условия.

Активы для рассмотрения:
{asset_list}

Последние рыночные данные:
{market_data}

Новости и настроения:
{news_summary}

Текущий портфель:
{current_portfolio}

На основе этой информации предоставьте:

1. Оценки активов (шкала 1-10):
   - Фундаментальная оценка: Качество финансов и бизнеса
   - Оценка моментума: Ценовой тренд и технические индикаторы
   - Оценка настроений: Новостные и социальные настроения
   - Оценка риска: Волатильность и риск снижения

2. Рекомендуемые веса портфеля (должны суммироваться до 100%):
   - Для каждого актива укажите целевой вес и обоснование

3. Действия по ребалансировке:
   - Какие сделки выполнить
   - Порядок приоритета сделок
   - Соображения по рискам

4. Уровень уверенности: (низкий/средний/высокий)

Вывод в формате JSON.
"""
```

### Ключевые метрики

**Производительность портфеля:**
- Коэффициент Шарпа (риск-скорректированная доходность)
- Коэффициент Сортино (скорректированный на нисходящий риск)
- Максимальная просадка
- Коэффициент Кальмара
- Информационный коэффициент относительно бенчмарка

**Метрики качества LLM:**
- Точность рекомендаций
- Ранговая корреляция (Спирмен)
- Hit rate по прогнозам направления
- Эффективность оборота

### Зависимости

```python
# Python зависимости
openai>=1.0.0           # OpenAI API клиент
anthropic>=0.5.0        # Claude API клиент
transformers>=4.30.0    # HuggingFace модели
torch>=2.0.0            # PyTorch
pandas>=2.0.0           # Работа с данными
numpy>=1.24.0           # Численные вычисления
yfinance>=0.2.0         # Данные акций
scipy>=1.10.0           # Оптимизация
cvxpy>=1.4.0            # Выпуклая оптимизация
requests>=2.28.0        # HTTP клиент
```

```rust
// Rust зависимости
reqwest = "0.12"        // HTTP клиент
serde = "1.0"           // Сериализация
tokio = "1.0"           // Асинхронный рантайм
ndarray = "0.16"        // Массивы
polars = "0.46"         // DataFrames
```

## Реализация на Python

### Структуры данных портфеля

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np

class AssetClass(Enum):
    EQUITY = "equity"      # Акции
    CRYPTO = "crypto"      # Криптовалюта
    BOND = "bond"          # Облигации
    COMMODITY = "commodity" # Товары

@dataclass
class Asset:
    """Представляет торгуемый актив."""
    symbol: str
    name: str
    asset_class: AssetClass
    current_price: float
    market_cap: Optional[float] = None

@dataclass
class AssetScore:
    """Оценки актива, сгенерированные LLM."""
    symbol: str
    fundamental_score: float  # 1-10
    momentum_score: float     # 1-10
    sentiment_score: float    # 1-10
    risk_score: float         # 1-10 (выше = больше риск)
    overall_score: float      # Взвешенная комбинация
    reasoning: str            # Объяснение LLM
    confidence: str           # низкий/средний/высокий

    @property
    def composite_score(self) -> float:
        """Вычисление взвешенной композитной оценки."""
        # Выше лучше, поэтому инвертируем оценку риска
        weights = {
            'fundamental': 0.30,
            'momentum': 0.25,
            'sentiment': 0.25,
            'risk': 0.20
        }
        return (
            weights['fundamental'] * self.fundamental_score +
            weights['momentum'] * self.momentum_score +
            weights['sentiment'] * self.sentiment_score +
            weights['risk'] * (10 - self.risk_score)  # Инвертируем риск
        )

@dataclass
class Portfolio:
    """Представляет распределение портфеля."""
    weights: Dict[str, float]  # символ -> вес
    cash_weight: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        # Нормализуем веса до суммы 1
        total = sum(self.weights.values()) + self.cash_weight
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
            self.cash_weight = self.cash_weight / total

    def get_weight(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "cash_weight": self.cash_weight,
            "timestamp": self.timestamp
        }
```

### Движок портфеля на LLM

```python
import json
from typing import List, Dict, Tuple
import openai

class LLMPortfolioEngine:
    """Движок построения портфеля на основе LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_assets(
        self,
        assets: List[Asset],
        market_data: Dict,
        news_data: List[str]
    ) -> List[AssetScore]:
        """Анализ активов и генерация оценок с помощью LLM."""

        # Подготовка информации об активах
        asset_info = self._format_assets(assets)
        market_summary = self._format_market_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = self._build_analysis_prompt(asset_info, market_summary, news_summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Вы количественный аналитик, специализирующийся на построении портфелей."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_scores(result)

    def generate_portfolio(
        self,
        scores: List[AssetScore],
        constraints: Dict = None
    ) -> Portfolio:
        """Генерация весов портфеля из оценок активов."""

        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.02,
                "max_assets": 10,
                "min_score": 5.0
            }

        # Фильтрация активов по минимальной оценке
        valid_scores = [s for s in scores if s.composite_score >= constraints["min_score"]]

        # Сортировка по композитной оценке
        valid_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Берём топ N активов
        selected = valid_scores[:constraints["max_assets"]]

        # Вычисление весов пропорционально оценкам
        total_score = sum(s.composite_score for s in selected)

        weights = {}
        for score in selected:
            raw_weight = score.composite_score / total_score
            # Применение ограничений
            weight = max(constraints["min_weight"],
                        min(constraints["max_weight"], raw_weight))
            weights[score.symbol] = weight

        # Нормализация
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        return Portfolio(weights=weights)

    def _build_analysis_prompt(
        self,
        assets: str,
        market: str,
        news: str
    ) -> str:
        return f"""Проанализируйте следующие активы для построения портфеля.

АКТИВЫ:
{assets}

РЫНОЧНЫЕ УСЛОВИЯ:
{market}

ПОСЛЕДНИЕ НОВОСТИ:
{news}

Для каждого актива предоставьте оценки (1-10) и анализ:
- fundamental_score: Качество бизнеса и финансов
- momentum_score: Сила ценового тренда
- sentiment_score: Новостные и рыночные настроения
- risk_score: Волатильность и риск снижения (10 = наивысший риск)
- reasoning: Краткое объяснение
- confidence: низкий/средний/высокий

Верните JSON с массивом "scores", содержащим объекты для каждого актива."""

    def _format_assets(self, assets: List[Asset]) -> str:
        lines = []
        for a in assets:
            lines.append(f"- {a.symbol}: {a.name} ({a.asset_class.value}), Цена: ${a.current_price:.2f}")
        return "\n".join(lines)

    def _format_market_data(self, data: Dict) -> str:
        lines = []
        for symbol, info in data.items():
            lines.append(f"- {symbol}: Доходность 7д: {info.get('return_7d', 0):.1%}, Волатильность: {info.get('volatility', 0):.1%}")
        return "\n".join(lines)

    def _format_news(self, news: List[str]) -> str:
        return "\n".join([f"- {n}" for n in news[:10]])

    def _parse_scores(self, data: Dict) -> List[AssetScore]:
        scores = []
        for item in data.get("scores", []):
            scores.append(AssetScore(
                symbol=item.get("symbol", ""),
                fundamental_score=float(item.get("fundamental_score", 5)),
                momentum_score=float(item.get("momentum_score", 5)),
                sentiment_score=float(item.get("sentiment_score", 5)),
                risk_score=float(item.get("risk_score", 5)),
                overall_score=float(item.get("overall_score", 5)),
                reasoning=item.get("reasoning", ""),
                confidence=item.get("confidence", "средний")
            ))
        return scores
```

### Оптимизатор Mean-Variance

```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

class MeanVarianceOptimizer:
    """Оптимизация портфеля по среднему-дисперсии с интеграцией оценок LLM."""

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        target_volatility: Optional[float] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        llm_scores: Optional[np.ndarray] = None,
        constraints: Dict = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Оптимизация весов портфеля.

        Args:
            expected_returns: Ожидаемые доходности для каждого актива
            covariance_matrix: Ковариационная матрица доходностей
            llm_scores: Опциональные композитные оценки LLM для смешивания
            constraints: Ограничения портфеля

        Returns:
            Кортеж (веса, метрики)
        """
        n_assets = len(expected_returns)

        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.0,
                "long_only": True
            }

        # Смешивание оценок LLM с ожидаемыми доходностями
        if llm_scores is not None:
            # Нормализация оценок LLM на схожую шкалу с доходностями
            normalized_scores = (llm_scores - llm_scores.mean()) / llm_scores.std()
            blend_weight = 0.3  # Вес оценок LLM
            adjusted_returns = (
                (1 - blend_weight) * expected_returns +
                blend_weight * normalized_scores * 0.01  # Масштабирующий фактор
            )
        else:
            adjusted_returns = expected_returns

        # Начальное приближение: равные веса
        x0 = np.ones(n_assets) / n_assets

        # Целевая функция: максимизация Sharpe ratio (минимизация отрицательного Sharpe)
        def neg_sharpe(weights):
            port_return = np.dot(weights, adjusted_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol

        # Ограничения
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Веса суммируются до 1
        ]

        # Границы
        if constraints["long_only"]:
            bounds = [(constraints["min_weight"], constraints["max_weight"])
                     for _ in range(n_assets)]
        else:
            bounds = [(-constraints["max_weight"], constraints["max_weight"])
                     for _ in range(n_assets)]

        # Оптимизация
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        weights = result.x

        # Вычисление метрик
        port_return = np.dot(weights, adjusted_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        metrics = {
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
            "optimization_success": result.success
        }

        return weights, metrics

    def risk_parity(
        self,
        covariance_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Распределение портфеля по паритету рисков.

        Каждый актив вносит равный вклад в риск портфеля.
        """
        n_assets = covariance_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_contrib = risk_budget * port_vol
            return np.sum((risk_contrib - target_contrib) ** 2)

        x0 = np.ones(n_assets) / n_assets
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        result = minimize(
            risk_contribution_error,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        weights = result.x
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        metrics = {
            "volatility": port_vol,
            "optimization_success": result.success
        }

        return weights, metrics
```

### Фреймворк бэктестинга

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class BacktestResult:
    """Результаты бэктестинга портфеля."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
Результаты бэктестинга портфеля
===============================
Общая доходность:      {self.total_return:.2%}
Годовая доходность:    {self.annualized_return:.2%}
Волатильность:         {self.volatility:.2%}
Коэффициент Шарпа:     {self.sharpe_ratio:.2f}
Коэффициент Сортино:   {self.sortino_ratio:.2f}
Максимальная просадка: {self.max_drawdown:.2%}
Коэффициент Кальмара:  {self.calmar_ratio:.2f}
Win Rate:              {self.win_rate:.2%}
Количество сделок:     {self.num_trades}
"""

class PortfolioBacktester:
    """Бэктестинг портфельных стратегий на основе LLM."""

    def __init__(
        self,
        initial_capital: float = 100000,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005  # 0.05%
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run(
        self,
        price_data: pd.DataFrame,
        portfolio_weights: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Запуск бэктеста с заданными весами портфеля.

        Args:
            price_data: DataFrame с ценами активов (колонки = символы)
            portfolio_weights: Dict, отображающий даты на DataFrame весов
            start_date: Дата начала бэктеста
            end_date: Дата окончания бэктеста

        Returns:
            BacktestResult с метриками производительности
        """
        # Фильтрация данных
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        prices = price_data.loc[mask].copy()

        # Инициализация
        capital = self.initial_capital
        portfolio_values = [capital]
        dates = [prices.index[0]]
        current_weights = {}
        num_trades = 0

        # Определение дат ребалансировки
        rebalance_dates = self._get_rebalance_dates(prices.index)

        for i in range(1, len(prices)):
            date = prices.index[i]
            prev_date = prices.index[i-1]

            # Вычисление доходностей
            daily_returns = (prices.iloc[i] / prices.iloc[i-1]) - 1

            # Проверка на ребалансировку
            if date in rebalance_dates and str(date) in portfolio_weights:
                new_weights = portfolio_weights[str(date)]

                # Вычисление оборота и издержек
                turnover = self._calculate_turnover(current_weights, new_weights)
                cost = turnover * (self.transaction_cost + self.slippage)
                capital *= (1 - cost)
                num_trades += sum(1 for s in new_weights if new_weights.get(s, 0) != current_weights.get(s, 0))

                current_weights = new_weights

            # Вычисление доходности портфеля
            port_return = sum(
                current_weights.get(symbol, 0) * daily_returns.get(symbol, 0)
                for symbol in current_weights
            )

            capital *= (1 + port_return)
            portfolio_values.append(capital)
            dates.append(date)

        # Вычисление метрик
        returns = pd.Series(portfolio_values).pct_change().dropna()

        total_return = (capital / self.initial_capital) - 1
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino (нисходящее отклонение)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino = annualized_return / downside_std

        # Максимальная просадка
        cumulative = pd.Series(portfolio_values)
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            num_trades=num_trades,
            portfolio_values=portfolio_values,
            dates=[str(d) for d in dates]
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Получение дат ребалансировки на основе частоты."""
        if self.rebalance_frequency == "daily":
            return set(dates)
        elif self.rebalance_frequency == "weekly":
            # Ребалансировка по понедельникам
            return set(dates[dates.dayofweek == 0])
        elif self.rebalance_frequency == "monthly":
            # Ребалансировка в первый торговый день месяца
            return set(dates.to_series().groupby(dates.to_period('M')).first())
        return set()

    def _calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """Вычисление оборота портфеля."""
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in all_symbols
        ) / 2
        return turnover
```

## Реализация на Rust

См. директорию `rust_llm_portfolio/` для полной реализации на Rust, которая включает:

- **Загрузка данных** с Bybit и Yahoo Finance
- **Интеграция с LLM API** (совместимо с OpenAI)
- **Алгоритмы оптимизации портфеля**
- **Фреймворк бэктестинга**
- **Вычисление метрик производительности**

### Быстрый старт (Rust)

```bash
cd rust_llm_portfolio

# Сборка проекта
cargo build --release

# Загрузка рыночных данных
cargo run --example fetch_data

# Запуск анализа портфеля
cargo run --example analyze_portfolio -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Бэктестинг стратегии
cargo run --example backtest -- --start 2024-01-01 --end 2024-06-01
```

## Ожидаемые результаты

1. **Конвейер анализа LLM** - Сквозная система для скоринга активов
2. **Построение портфеля** - Оптимизированные веса на основе инсайтов LLM
3. **Управление рисками** - Построение портфеля на основе ограничений
4. **Результаты бэктестинга** - Валидация исторической производительности
5. **Стратегия ребалансировки** - Правила динамической корректировки портфеля

## Примеры использования

### Криптовалютный портфель

```python
# Пример: Построение крипто-портфеля с LLM
assets = [
    Asset("BTCUSDT", "Bitcoin", AssetClass.CRYPTO, 65000),
    Asset("ETHUSDT", "Ethereum", AssetClass.CRYPTO, 3200),
    Asset("SOLUSDT", "Solana", AssetClass.CRYPTO, 140),
    Asset("BNBUSDT", "Binance Coin", AssetClass.CRYPTO, 580),
]

# Получение оценок LLM
scores = engine.analyze_assets(assets, market_data, news)

# Генерация портфеля
portfolio = engine.generate_portfolio(scores, constraints={
    "max_weight": 0.40,  # Макс 40% в одном активе
    "min_weight": 0.05,  # Мин 5% аллокация
    "max_assets": 5
})
```

### Портфель акций

```python
# Пример: Построение диверсифицированного портфеля акций
assets = [
    Asset("AAPL", "Apple Inc", AssetClass.EQUITY, 185),
    Asset("MSFT", "Microsoft", AssetClass.EQUITY, 420),
    Asset("GOOGL", "Alphabet", AssetClass.EQUITY, 175),
    Asset("NVDA", "NVIDIA", AssetClass.EQUITY, 880),
    Asset("AMZN", "Amazon", AssetClass.EQUITY, 185),
]

# Анализ с секторальными ограничениями
scores = engine.analyze_assets(assets, market_data, news)
portfolio = engine.generate_portfolio(scores, constraints={
    "max_weight": 0.25,
    "min_weight": 0.05,
    "sector_limits": {"tech": 0.60}  # Макс 60% в техе
})
```

### Мульти-агентный ансамбль

```python
# Пример: Использование нескольких персон LLM для робастного распределения
personas = [
    "value_investor",     # Фокус на фундаментале
    "momentum_trader",    # Фокус на трендах
    "risk_manager",       # Фокус на снижении рисков
    "contrarian"          # Противоположность консенсусу
]

ensemble_weights = {}
for persona in personas:
    scores = engine.analyze_with_persona(assets, persona)
    weights = engine.generate_portfolio(scores)
    ensemble_weights[persona] = weights

# Агрегация: усреднение весов по персонам
final_weights = aggregate_portfolios(ensemble_weights)
```

## Лучшие практики

1. **Инженерия промптов** - Тестируйте промпты для согласованного, действенного вывода
2. **Калибровка оценок** - Валидируйте оценки LLM по историческим результатам
3. **Настройка ограничений** - Используйте разумные лимиты позиций и диверсификацию
4. **Регулярная валидация** - Часто проводите бэктестинг на данных вне выборки
5. **Человеческий надзор** - Проверяйте рекомендации LLM перед исполнением
6. **Управление затратами** - Кэшируйте ответы LLM для снижения затрат на API
7. **Резервная логика** - Имейте правила на основе правил как резерв при отказе LLM

## Ссылки

- [Large Language Models in Equity Markets](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) - Комплексный обзор применения LLM в инвестировании в акции
- [LLM Agents for Investment Management](https://dl.acm.org/doi/10.1145/3768292.3770387) - Обзор агентных подходов
- [FolioLLM: Portfolio Construction with LLMs](https://web.stanford.edu/class/cs224n/final-reports/256938687.pdf) - Исследование Stanford по распределению ETF
- [Persona-Based LLM Ensembles](https://arxiv.org/html/2411.19515v1) - Исследование University of Tokyo по ансамблевым методам
- [From Text to Returns](https://arxiv.org/abs/2512.05907) - Оптимизация взаимных фондов с LLM
- [BloombergGPT](https://arxiv.org/abs/2303.17564) - Большая языковая модель для финансов
- [FinGPT](https://arxiv.org/abs/2306.06031) - Финансовая LLM с открытым исходным кодом

## Уровень сложности

Эксперт

Необходимые знания: промптинг LLM, оптимизация портфеля, количественные финансы, интеграция API, методология бэктестинга
