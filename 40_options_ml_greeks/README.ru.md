# Глава 40: Предсказание греков опционов с помощью ML — Дельта-нейтральная торговля волатильностью

## Обзор

Опционное ценообразование зависит от **подразумеваемой волатильности (Implied Volatility, IV)**, которая часто отличается от **реализованной волатильности (Realized Volatility, RV)**. Модели машинного обучения способны предсказывать будущую реализованную волатильность лучше, чем это отражено в текущей подразумеваемой волатильности. Это создаёт возможности для торговли волатильностью с дельта-хеджированными позициями.

### Ключевые концепции

| Термин | Определение |
|--------|-------------|
| **Implied Volatility (IV)** | Волатильность, заложенная в цену опциона рынком |
| **Realized Volatility (RV)** | Фактическая историческая волатильность актива |
| **Volatility Risk Premium (VRP)** | Разница между IV и RV (обычно IV > RV) |
| **Delta-neutral** | Позиция, нечувствительная к малым движениям цены базового актива |
| **Straddle** | Одновременная покупка/продажа call и put с одним страйком |

## Торговая стратегия

### Суть стратегии

**Основная идея:** Предсказание будущей Realized Volatility (RV) и сравнение с текущей Implied Volatility (IV).

- **Продажа опционов** когда IV > predicted RV (волатильность переоценена — собираем премию)
- **Покупка опционов** когда IV < predicted RV (волатильность недооценена — ожидаем большее движение)
- **Delta-hedge:** Поддержание дельта-нейтральной позиции для изоляции ставки на волатильность

### Сигналы на вход

| Сигнал | Условие | Действие |
|--------|---------|----------|
| **Sell Straddle** | IV > predicted RV + threshold | Продаём волатильность |
| **Buy Straddle** | IV < predicted RV - threshold | Покупаем волатильность |
| **Delta-hedge** | |Δ| > threshold | Корректируем позицию в базовом активе |

### Преимущество (Edge)

Наше преимущество заключается в **лучшем предсказании RV**, чем то, что заложено в ценах опционов. Рынок опционов систематически переоценивает волатильность (VRP положительная в среднем), но бывают периоды, когда она недооценена.

## Греки опционов

### Что такое греки?

**Греки** — это коэффициенты чувствительности цены опциона к различным параметрам:

| Грек | Символ | Чувствительность к | Интерпретация |
|------|--------|-------------------|---------------|
| **Delta (Δ)** | δ | Цене базового актива | Насколько изменится цена опциона при изменении цены актива на $1 |
| **Gamma (Γ)** | γ | Изменению дельты | Скорость изменения дельты (кривизна) |
| **Theta (Θ)** | θ | Времени | Потеря стоимости опциона за день |
| **Vega (ν)** | ν | Волатильности | Изменение цены при изменении IV на 1% |
| **Rho (ρ)** | ρ | Процентной ставке | Редко используется для краткосрочных опционов |

### Формула Блэка-Шоулза

Классическая модель ценообразования европейских опционов:

```
Call = S × N(d₁) - K × e^(-rT) × N(d₂)
Put  = K × e^(-rT) × N(-d₂) - S × N(-d₁)

где:
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

S = текущая цена актива
K = страйк опциона
T = время до экспирации (в годах)
r = безрисковая ставка
σ = волатильность
N() = функция нормального распределения
```

## Техническая спецификация

### План ноутбуков

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_options_basics.ipynb` | Основы опционов, греки, расчёт по Блэку-Шоулзу |
| 2 | `02_data_collection.ipynb` | Сбор данных: цепочки опционов, исторические IV и RV |
| 3 | `03_iv_term_structure.ipynb` | Анализ временной структуры IV и улыбки волатильности |
| 4 | `04_rv_forecasting.ipynb` | ML модели для предсказания Realized Volatility |
| 5 | `05_iv_rv_spread.ipynb` | Анализ спреда IV-RV (Volatility Risk Premium) |
| 6 | `06_var_swap_replication.ipynb` | Ценообразование и репликация variance swaps |
| 7 | `07_straddle_strategy.ipynb` | Выбор страддлов и определение размера позиции |
| 8 | `08_delta_hedging.ipynb` | Динамическое дельта-хеджирование |
| 9 | `09_gamma_scalping.ipynb` | Гамма-скальпинг для длинной волатильности |
| 10 | `10_backtesting.ipynb` | Полный бэктест с атрибуцией P&L по грекам |
| 11 | `11_risk_management.ipynb` | Лимиты по веге, хвостовые риски |

## Расчёт греков на Python

### Модель Блэка-Шоулза

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Расчёт цены опциона по модели Блэка-Шоулза

    Параметры:
    ----------
    S : float - Текущая цена базового актива
    K : float - Страйк-цена опциона
    T : float - Время до экспирации в годах
    r : float - Безрисковая процентная ставка
    sigma : float - Волатильность (стандартное отклонение)
    option_type : str - 'call' или 'put'

    Возвращает:
    -----------
    float - Теоретическая цена опциона
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Расчёт подразумеваемой волатильности из цены опциона
    методом Брента (бинарный поиск)

    Параметры:
    ----------
    price : float - Рыночная цена опциона
    S, K, T, r : float - Параметры опциона
    option_type : str - 'call' или 'put'

    Возвращает:
    -----------
    float - Подразумеваемая волатильность
    """
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - price

    try:
        iv = brentq(objective, 0.01, 5.0)
    except ValueError:
        iv = np.nan

    return iv

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Расчёт всех греков опциона

    Возвращает:
    -----------
    dict - Словарь с delta, gamma, theta, vega, rho
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Delta - чувствительность к цене
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma - скорость изменения дельты
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta - временной распад (в день)
    if option_type == 'call':
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

    # Vega - чувствительность к волатильности (на 1% изменения IV)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho - чувствительность к процентной ставке
    if option_type == 'call':
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
```

### Пример использования

```python
# Параметры опциона
S = 100      # Цена актива $100
K = 100      # Страйк $100 (ATM)
T = 30/365   # 30 дней до экспирации
r = 0.05     # 5% годовая ставка
sigma = 0.20 # 20% волатильность

# Расчёт цены
call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

print(f"Call цена: ${call_price:.2f}")
print(f"Put цена: ${put_price:.2f}")

# Расчёт греков
greeks = calculate_greeks(S, K, T, r, sigma, 'call')
print(f"\nГреки call опциона:")
print(f"  Delta: {greeks['delta']:.4f}")
print(f"  Gamma: {greeks['gamma']:.4f}")
print(f"  Theta: ${greeks['theta']:.4f}/день")
print(f"  Vega:  ${greeks['vega']:.4f}/1% IV")
```

## Предсказание реализованной волатильности

### Модель предсказания RV

```python
import pandas as pd
from lightgbm import LGBMRegressor

class RealizedVolatilityPredictor:
    """
    Предсказание будущей реализованной волатильности
    с использованием машинного обучения
    """

    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )

        # Признаки для модели
        self.feature_names = [
            # Историческая волатильность разных окон
            'rv_5d', 'rv_10d', 'rv_20d', 'rv_60d',

            # Волатильность волатильности
            'rv_20d_std', 'rv_20d_skew',

            # Подразумеваемая волатильность
            'iv_atm', 'iv_25d_put', 'iv_25d_call',
            'iv_skew', 'iv_term_slope',

            # Рыночные условия
            'return_5d', 'return_20d',
            'volume_ratio', 'gap_frequency',

            # VIX (для акций) или криптовол
            'vix_level', 'vix_percentile', 'vix_term_structure'
        ]

    def calculate_realized_vol(self, returns, window=20):
        """
        Расчёт реализованной волатильности (годовая)

        RV = std(returns) * sqrt(252)  для акций
        RV = std(returns) * sqrt(365)  для крипты
        """
        return returns.rolling(window).std() * np.sqrt(365)  # для крипты

    def prepare_features(self, data):
        """
        Создание матрицы признаков из сырых данных
        """
        features = pd.DataFrame(index=data.index)

        # Историческая RV на разных окнах
        for window in [5, 10, 20, 60]:
            features[f'rv_{window}d'] = self.calculate_realized_vol(
                data['returns'], window
            )

        # Волатильность волатильности (vol of vol)
        features['rv_20d_std'] = features['rv_20d'].rolling(20).std()
        features['rv_20d_skew'] = features['rv_20d'].rolling(20).skew()

        # IV признаки (должны быть в данных)
        if 'iv_atm' in data.columns:
            features['iv_atm'] = data['iv_atm']
            features['iv_skew'] = data['iv_25d_put'] - data['iv_25d_call']
            features['iv_term_slope'] = data['iv_3m'] - data['iv_1m']

        # Признаки доходности
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)

        # Объём
        if 'volume' in data.columns:
            features['volume_ratio'] = (
                data['volume'] / data['volume'].rolling(20).mean()
            )

        return features.dropna()

    def train(self, X, y):
        """
        Обучение модели
        y = реализованная волатильность за следующие N дней
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Предсказание будущей RV"""
        return self.model.predict(X)

    def feature_importance(self):
        """Важность признаков"""
        return pd.DataFrame({
            'feature': self.feature_names[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
```

## Анализ волатильностной премии (VRP)

```python
class VolatilityRiskPremium:
    """
    Анализ премии за волатильностный риск (IV - RV)

    VRP > 0: IV переоценивает будущую волатильность (продаём опционы)
    VRP < 0: IV недооценивает волатильность (покупаем опционы)
    """

    def __init__(self, lookback=20):
        self.lookback = lookback

    def calculate_vrp(self, iv_series, rv_series):
        """
        Расчёт VRP
        IV - форвардно-смотрящая (текущая)
        RV - будущая реализованная (сдвигаем назад для расчёта)
        """
        future_rv = rv_series.shift(-self.lookback)
        vrp = iv_series - future_rv
        return vrp

    def vrp_statistics(self, vrp):
        """
        Статистика VRP для торговых сигналов
        """
        return {
            'mean': vrp.mean(),
            'std': vrp.std(),
            'current_zscore': (vrp.iloc[-1] - vrp.mean()) / vrp.std(),
            'pct_positive': (vrp > 0).mean(),  # Как часто IV > RV
            'avg_when_positive': vrp[vrp > 0].mean(),
            'avg_when_negative': vrp[vrp < 0].mean()
        }

    def trading_signal(self, current_iv, predicted_rv, threshold=0.02):
        """
        Генерация торгового сигнала

        Параметры:
        ----------
        current_iv : float - Текущая подразумеваемая волатильность
        predicted_rv : float - Предсказанная реализованная волатильность
        threshold : float - Минимальный спред для входа (2% по умолчанию)

        Возвращает:
        -----------
        dict - Сигнал и размер edge
        """
        spread = current_iv - predicted_rv

        if spread > threshold:
            return {
                'action': 'sell_volatility',
                'edge': spread,
                'reason': 'IV переоценена относительно прогноза RV'
            }
        elif spread < -threshold:
            return {
                'action': 'buy_volatility',
                'edge': -spread,
                'reason': 'IV недооценена относительно прогноза RV'
            }
        else:
            return {
                'action': 'no_trade',
                'edge': 0,
                'reason': 'Спред в пределах threshold'
            }
```

## Дельта-хеджирование

```python
class DeltaHedger:
    """
    Поддержание дельта-нейтральной позиции

    Цель: Изолировать ставку на волатильность от
    направленного движения цены базового актива
    """

    def __init__(self, hedge_threshold=0.05, transaction_cost=0.001):
        self.hedge_threshold = hedge_threshold  # Порог для ребалансировки
        self.transaction_cost = transaction_cost  # Комиссия 0.1%
        self.position = {'options': [], 'underlying': 0}
        self.hedge_history = []

    def calculate_portfolio_delta(self, options_positions, spot_price):
        """
        Расчёт суммарной дельты портфеля
        """
        total_delta = 0

        for opt in options_positions:
            opt_greeks = calculate_greeks(
                S=spot_price,
                K=opt['strike'],
                T=opt['tte'],
                r=opt['rate'],
                sigma=opt['iv'],
                option_type=opt['type']
            )
            # 1 контракт = 1 единица базового актива для крипты
            total_delta += opt['quantity'] * opt_greeks['delta']

        # Добавляем дельту базового актива (дельта = 1)
        total_delta += self.position['underlying']

        return total_delta

    def hedge_delta(self, current_delta, spot_price, timestamp):
        """
        Выполнение хедж-сделки для нейтрализации дельты
        """
        if abs(current_delta) > self.hedge_threshold:
            # Продаём базовый актив для снижения положительной дельты
            # Покупаем для снижения отрицательной
            hedge_quantity = -current_delta

            trade = {
                'timestamp': timestamp,
                'type': 'hedge',
                'quantity': hedge_quantity,
                'price': spot_price,
                'cost': abs(hedge_quantity) * spot_price * self.transaction_cost
            }

            self.position['underlying'] += hedge_quantity
            self.hedge_history.append(trade)

            return trade

        return None

    def get_hedge_pnl(self):
        """
        P&L от хеджирующих сделок
        """
        total_cost = sum(h['cost'] for h in self.hedge_history)

        # Реализованный P&L от закрытых позиций
        realized_pnl = 0
        for i, h in enumerate(self.hedge_history[:-1]):
            next_h = self.hedge_history[i+1]
            if h['quantity'] * next_h['quantity'] < 0:  # Противоположные направления
                realized_pnl += (next_h['price'] - h['price']) * min(
                    abs(h['quantity']), abs(next_h['quantity'])
                )

        return {
            'transaction_costs': total_cost,
            'realized_pnl': realized_pnl,
            'num_hedges': len(self.hedge_history)
        }
```

## Стратегия Straddle

```python
class StraddleStrategy:
    """
    Торговля волатильностью через ATM страддлы

    Straddle = Call + Put с одинаковым страйком и экспирацией

    - Long straddle: Ставка на рост волатильности
    - Short straddle: Ставка на снижение волатильности
    """

    def __init__(self, rv_predictor, min_edge=0.02, max_vega=10000):
        self.rv_predictor = rv_predictor
        self.min_edge = min_edge  # Минимальный edge для входа
        self.max_vega = max_vega  # Максимальная вега экспозиция

    def select_straddle(self, options_chain, spot_price, predicted_rv):
        """
        Выбор оптимального страддла

        Параметры:
        ----------
        options_chain : DataFrame - Цепочка опционов
        spot_price : float - Текущая цена актива
        predicted_rv : float - Предсказанная RV

        Возвращает:
        -----------
        dict или None - Параметры сделки
        """
        # Находим ATM страйк (ближайший к текущей цене)
        strikes = options_chain['strike'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))

        # Получаем ATM call и put
        call = options_chain[
            (options_chain['strike'] == atm_strike) &
            (options_chain['type'] == 'call')
        ].iloc[0]

        put = options_chain[
            (options_chain['strike'] == atm_strike) &
            (options_chain['type'] == 'put')
        ].iloc[0]

        # Средняя IV страддла
        straddle_iv = (call['iv'] + put['iv']) / 2

        # Edge = IV - predicted RV
        edge = straddle_iv - predicted_rv

        if abs(edge) > self.min_edge:
            direction = 'sell' if edge > 0 else 'buy'

            # Размер позиции на основе вега-лимита
            straddle_vega = call['vega'] + put['vega']
            max_contracts = self.max_vega / straddle_vega if straddle_vega > 0 else 0

            return {
                'strike': atm_strike,
                'expiry': call['expiry'],
                'iv': straddle_iv,
                'predicted_rv': predicted_rv,
                'edge': edge,
                'direction': direction,
                'contracts': min(max_contracts, 10),
                'call_price': call['price'],
                'put_price': put['price'],
                'straddle_price': call['price'] + put['price'],
                'total_vega': straddle_vega,
                'total_gamma': call['gamma'] + put['gamma'],
                'total_theta': call['theta'] + put['theta']
            }

        return None

    def pnl_attribution(self, entry, exit_data, daily_moves):
        """
        Атрибуция P&L по грекам

        Параметры:
        ----------
        entry : dict - Параметры входа
        exit_data : dict - Параметры выхода
        daily_moves : list - Дневные движения цены

        Возвращает:
        -----------
        dict - Разбивка P&L по источникам
        """
        multiplier = 1 if entry['direction'] == 'buy' else -1
        contracts = entry['contracts']

        # Theta P&L (временной распад)
        days_held = len(daily_moves)
        theta_pnl = entry['total_theta'] * days_held * contracts * multiplier

        # Gamma P&L (от реализованных движений)
        # P&L = 0.5 * Gamma * (move^2) для каждого дня
        gamma_pnl = 0.5 * entry['total_gamma'] * sum(
            m**2 for m in daily_moves
        ) * contracts * multiplier

        # Vega P&L (изменение IV)
        iv_change = exit_data['iv'] - entry['iv']
        vega_pnl = entry['total_vega'] * iv_change * 100 * contracts * multiplier

        # Delta P&L (должен быть ~0 если хеджировали)
        delta_pnl = 0  # Предполагаем идеальный хедж

        return {
            'theta_pnl': theta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'delta_pnl': delta_pnl,
            'total_pnl': theta_pnl + gamma_pnl + vega_pnl + delta_pnl,
            'attribution': {
                'theta': theta_pnl / (abs(theta_pnl) + abs(gamma_pnl) + abs(vega_pnl) + 0.0001),
                'gamma': gamma_pnl / (abs(theta_pnl) + abs(gamma_pnl) + abs(vega_pnl) + 0.0001),
                'vega': vega_pnl / (abs(theta_pnl) + abs(gamma_pnl) + abs(vega_pnl) + 0.0001)
            }
        }
```

## Гамма-скальпинг

```python
class GammaScalper:
    """
    Гамма-скальпинг для длинной волатильности

    Стратегия: Купить straddle + скальпировать гамму

    Когда цена растёт → дельта увеличивается → продаём базовый актив
    Когда цена падает → дельта уменьшается → покупаем базовый актив

    Прибыль = Gamma * (реализованное движение)^2 - Theta decay
    """

    def __init__(self, rebalance_threshold=0.10):
        self.rebalance_threshold = rebalance_threshold
        self.scalp_history = []

    def should_scalp(self, current_delta, previous_delta):
        """
        Определяем, нужно ли скальпировать
        """
        delta_change = current_delta - previous_delta
        return abs(delta_change) > self.rebalance_threshold

    def execute_scalp(self, delta_change, spot_price, timestamp):
        """
        Выполняем скальп-сделку
        """
        # Продаём когда дельта выросла, покупаем когда упала
        trade_quantity = -delta_change

        trade = {
            'timestamp': timestamp,
            'quantity': trade_quantity,
            'price': spot_price,
            'delta_before': delta_change,
            'type': 'sell' if trade_quantity < 0 else 'buy'
        }

        self.scalp_history.append(trade)
        return trade

    def calculate_scalping_pnl(self):
        """
        Расчёт P&L от скальпинга

        Принцип: Покупаем дёшево (когда цена упала),
                 продаём дорого (когда цена выросла)
        """
        if len(self.scalp_history) < 2:
            return 0

        total_pnl = 0
        position = 0
        avg_cost = 0

        for trade in self.scalp_history:
            if trade['type'] == 'buy':
                # Обновляем средневзвешенную стоимость
                total_cost = position * avg_cost + abs(trade['quantity']) * trade['price']
                position += abs(trade['quantity'])
                avg_cost = total_cost / position if position > 0 else 0
            else:
                # Фиксируем прибыль
                pnl = abs(trade['quantity']) * (trade['price'] - avg_cost)
                total_pnl += pnl
                position -= abs(trade['quantity'])

        return total_pnl
```

## Ключевые метрики

### Метрики предсказания
- **RMSE** — среднеквадратичная ошибка предсказания RV
- **IC** — информационный коэффициент (корреляция прогноза с результатом)
- **Hit Rate** — доля верных предсказаний направления

### Торговые метрики
- **Sharpe Ratio** — доходность с поправкой на риск
- **Win Rate** — процент прибыльных сделок
- **Avg P&L** — средний P&L на сделку
- **Max Drawdown** — максимальная просадка

### Метрики греков
- **Theta Capture** — сколько тета-распада удалось собрать
- **Gamma Scalping P&L** — прибыль от гамма-скальпинга
- **Vega Exposure** — экспозиция к изменению IV

### Метрики риска
- **Max Loss** — максимальный убыток
- **Tail Risk** — риск экстремальных событий
- **Vega Concentration** — концентрация веги по экспирациям

## Зависимости

```python
# requirements.txt
py_vollib>=1.0.1      # Ценообразование опционов
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
lightgbm>=4.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Ожидаемые результаты

1. **Опционный движок** — расчёт цен и греков
2. **Модель предсказания RV** — точнее наивного подхода (историческая RV)
3. **Анализ VRP** — когда IV систематически переоценивает RV
4. **Стратегия страддлов** — с дельта-хеджированием
5. **Атрибуция P&L** — разбивка по грекам
6. **Результат:** Положительный edge от неправильного ценообразования волатильности

## Риски и ограничения

### Основные риски

| Риск | Описание | Митигация |
|------|----------|-----------|
| **Gap Risk** | Резкий гэп при открытии | Избегать позиций через выходные |
| **Vega Risk** | Неожиданный рост IV | Лимиты на вегу, хедж VIX |
| **Model Risk** | Ошибка предсказания RV | Cross-validation, ensemble |
| **Liquidity Risk** | Широкие спреды | Торговать только ликвидные страйки |
| **Execution Risk** | Слипpage при хеджировании | Лимиты на размер позиции |

### Особенности крипторынка

- **24/7 торговля** — нет overnight gaps
- **Высокая волатильность** — больше opportunities
- **Ликвидность** — может быть ограничена на дальних страйках
- **Ставка фондирования** — влияет на put-call parity

## Литература

- [Volatility Trading](https://www.amazon.com/Volatility-Trading-Euan-Sinclair/dp/0470181990) (Euan Sinclair)
- [Option Volatility and Pricing](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774) (Sheldon Natenberg)
- [Forecasting Realized Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1502915)
- [The Variance Risk Premium](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1316046)
- [Dynamic Hedging](https://www.amazon.com/Dynamic-Hedging-Managing-Vanilla-Options/dp/0471152803) (Nassim Taleb)

## Уровень сложности

⭐⭐⭐⭐⭐ (Эксперт)

**Требуемые знания:**
- Теория опционов и греки
- Модели волатильности
- Дельта-хеджирование
- Машинное обучение
- Торговля деривативами
