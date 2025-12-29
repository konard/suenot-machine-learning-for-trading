# Chapter 40: Options Greeks Prediction — Delta-Neutral Volatility Trading

## Overview

Options pricing зависит от implied volatility, которая часто отличается от realized volatility. ML модели могут предсказывать будущую realized volatility лучше, чем подразумевает текущая IV. Это создает возможности для volatility trading с delta-hedged positions.

## Trading Strategy

**Суть стратегии:** Предсказание Realized Volatility (RV) vs Implied Volatility (IV). Продажа опционов когда IV > predicted RV (собираем премию). Покупка опционов когда IV < predicted RV.

**Сигнал на вход:**
- Sell Straddle: IV > predicted RV + threshold (volatility overpriced)
- Buy Straddle: IV < predicted RV - threshold (volatility underpriced)
- Delta-hedge: Maintain delta-neutral позицию

**Edge:** Лучшее предсказание RV чем implied в options prices

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_options_basics.ipynb` | Greeks, Black-Scholes, IV calculation |
| 2 | `02_data_collection.ipynb` | Options chains, historical IV, RV data |
| 3 | `03_iv_term_structure.ipynb` | IV term structure и skew analysis |
| 4 | `04_rv_forecasting.ipynb` | ML модели для Realized Volatility |
| 5 | `05_iv_rv_spread.ipynb` | Analysis of IV-RV spread (VRP) |
| 6 | `06_var_swap_replication.ipynb` | Variance swap pricing и replication |
| 7 | `07_straddle_strategy.ipynb` | Straddle selection и sizing |
| 8 | `08_delta_hedging.ipynb` | Dynamic delta hedging |
| 9 | `09_gamma_scalping.ipynb` | Gamma scalping for long volatility |
| 10 | `10_backtesting.ipynb` | Full backtest с Greeks P&L attribution |
| 11 | `11_risk_management.ipynb` | Vega limits, tail risk |

### Options Fundamentals

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes option pricing
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
    Calculate implied volatility from option price
    """
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - price

    try:
        iv = brentq(objective, 0.01, 5.0)
    except ValueError:
        iv = np.nan

    return iv

def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (per day)
    theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    theta_put = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    theta = theta_call if option_type == 'call' else theta_put

    # Vega (per 1% move in IV)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }
```

### Realized Volatility Prediction

```python
class RealizedVolatilityPredictor:
    """
    Predict future realized volatility
    """
    def __init__(self):
        self.model = LGBMRegressor(n_estimators=100, max_depth=6)

        self.features = [
            # Historical volatility
            'rv_5d', 'rv_10d', 'rv_20d', 'rv_60d',

            # Volatility of volatility
            'rv_20d_std', 'rv_20d_skew',

            # Implied volatility
            'iv_atm', 'iv_25d_put', 'iv_25d_call',
            'iv_skew', 'iv_term_slope',

            # Market conditions
            'return_5d', 'return_20d',
            'volume_ratio', 'gap_frequency',

            # VIX features
            'vix_level', 'vix_percentile', 'vix_term_structure'
        ]

    def calculate_realized_vol(self, returns, window=20):
        """
        Calculate realized volatility (annualized)
        """
        return returns.rolling(window).std() * np.sqrt(252)

    def prepare_features(self, data):
        """
        Create feature matrix
        """
        features = pd.DataFrame()

        # Historical RV at different windows
        for window in [5, 10, 20, 60]:
            features[f'rv_{window}d'] = self.calculate_realized_vol(data['returns'], window)

        # Vol of vol
        features['rv_20d_std'] = features['rv_20d'].rolling(20).std()
        features['rv_20d_skew'] = features['rv_20d'].rolling(20).skew()

        # IV features
        features['iv_atm'] = data['iv_atm']
        features['iv_skew'] = data['iv_25d_put'] - data['iv_25d_call']
        features['iv_term_slope'] = data['iv_3m'] - data['iv_1m']

        # ... more features

        return features

    def train(self, X, y):
        """
        Train model to predict future RV
        y = realized volatility over next N days
        """
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

### IV-RV Spread Analysis

```python
class VolatilityRiskPremium:
    """
    Analyze Volatility Risk Premium (IV - RV)
    """
    def __init__(self, lookback=20):
        self.lookback = lookback

    def calculate_vrp(self, iv_series, rv_series):
        """
        Calculate Volatility Risk Premium
        """
        # IV is forward-looking, RV is backward-looking
        # Compare IV today with RV over next N days
        future_rv = rv_series.shift(-self.lookback)
        vrp = iv_series - future_rv

        return vrp

    def vrp_statistics(self, vrp):
        """
        VRP statistics for trading signals
        """
        return {
            'mean': vrp.mean(),
            'std': vrp.std(),
            'current_zscore': (vrp.iloc[-1] - vrp.mean()) / vrp.std(),
            'pct_positive': (vrp > 0).mean(),  # How often IV > RV
            'avg_when_positive': vrp[vrp > 0].mean(),
            'avg_when_negative': vrp[vrp < 0].mean()
        }

    def trading_signal(self, current_iv, predicted_rv, threshold=0.02):
        """
        Generate trading signal based on IV vs predicted RV
        """
        spread = current_iv - predicted_rv

        if spread > threshold:
            return {'action': 'sell_vol', 'edge': spread}
        elif spread < -threshold:
            return {'action': 'buy_vol', 'edge': -spread}
        else:
            return {'action': 'none', 'edge': 0}
```

### Delta Hedging

```python
class DeltaHedger:
    """
    Maintain delta-neutral position
    """
    def __init__(self, hedge_frequency='daily', hedge_threshold=0.05):
        self.hedge_frequency = hedge_frequency
        self.hedge_threshold = hedge_threshold
        self.position = {'options': {}, 'stock': 0}

    def calculate_portfolio_delta(self, options_positions, spot_price):
        """
        Calculate total portfolio delta
        """
        total_delta = 0

        for opt in options_positions:
            opt_greeks = greeks(
                S=spot_price,
                K=opt['strike'],
                T=opt['tte'],
                r=opt['rate'],
                sigma=opt['iv'],
                option_type=opt['type']
            )
            total_delta += opt['quantity'] * opt_greeks['delta'] * 100  # 100 shares per contract

        # Add stock delta
        total_delta += self.position['stock']

        return total_delta

    def hedge_delta(self, current_delta, spot_price):
        """
        Execute hedge trade to neutralize delta
        """
        if abs(current_delta) > self.hedge_threshold:
            # Sell stock to reduce positive delta, buy to reduce negative
            hedge_shares = -current_delta

            trade = {
                'type': 'stock',
                'shares': hedge_shares,
                'price': spot_price,
                'cost': abs(hedge_shares) * spot_price * 0.001  # Assume 10bps cost
            }

            self.position['stock'] += hedge_shares

            return trade

        return None
```

### Straddle Strategy

```python
class StraddleStrategy:
    """
    ATM Straddle volatility trading
    """
    def __init__(self, rv_predictor, min_edge=0.02, max_vega=10000):
        self.rv_predictor = rv_predictor
        self.min_edge = min_edge
        self.max_vega = max_vega

    def select_straddle(self, options_chain, spot_price, predicted_rv):
        """
        Select optimal straddle based on IV vs predicted RV
        """
        # Find ATM options
        atm_strike = round(spot_price / 5) * 5  # Round to nearest $5

        call = options_chain[(options_chain['strike'] == atm_strike) &
                            (options_chain['type'] == 'call')].iloc[0]
        put = options_chain[(options_chain['strike'] == atm_strike) &
                           (options_chain['type'] == 'put')].iloc[0]

        # Calculate combined IV (straddle IV)
        straddle_iv = (call['iv'] + put['iv']) / 2

        # Edge = IV - predicted RV
        edge = straddle_iv - predicted_rv

        if abs(edge) > self.min_edge:
            direction = 'sell' if edge > 0 else 'buy'

            # Size based on vega limit
            straddle_vega = call['vega'] + put['vega']
            max_contracts = self.max_vega / (straddle_vega * 100)

            return {
                'strike': atm_strike,
                'expiry': call['expiry'],
                'iv': straddle_iv,
                'predicted_rv': predicted_rv,
                'edge': edge,
                'direction': direction,
                'contracts': min(max_contracts, 10),  # Max 10 contracts
                'straddle_price': call['price'] + put['price']
            }

        return None

    def pnl_attribution(self, entry, exit, realized_moves):
        """
        Attribute P&L to Greeks
        """
        dt = (exit['date'] - entry['date']).days

        # Theta P&L (time decay)
        theta_pnl = entry['theta'] * dt * entry['contracts'] * 100

        # Gamma P&L (realized moves)
        gamma_pnl = 0.5 * entry['gamma'] * sum(m**2 for m in realized_moves) * entry['contracts'] * 100

        # Vega P&L (IV change)
        iv_change = exit['iv'] - entry['iv']
        vega_pnl = entry['vega'] * iv_change * 100 * entry['contracts'] * 100

        # Delta P&L (should be small if hedged)
        delta_pnl = entry['delta'] * (exit['spot'] - entry['spot']) * entry['contracts'] * 100

        return {
            'theta': theta_pnl,
            'gamma': gamma_pnl,
            'vega': vega_pnl,
            'delta': delta_pnl,
            'total': theta_pnl + gamma_pnl + vega_pnl + delta_pnl
        }
```

### Key Metrics

- **Prediction:** RV forecast accuracy, IC, RMSE
- **Trading:** Sharpe, Win rate, Avg P&L per trade
- **Greeks:** Theta capture, Gamma scalping P&L, Vega exposure
- **Risk:** Max loss, Tail events, Vega concentration

### Dependencies

```python
py_vollib>=1.0.1      # Options pricing
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
lightgbm>=4.0.0
matplotlib>=3.6.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **Options pricing engine** с Greeks calculation
2. **RV prediction model** лучше чем naive (past RV)
3. **VRP analysis** — когда IV consistently overprices RV
4. **Straddle strategy** с delta hedging
5. **P&L attribution** по Greeks
6. **Results:** Positive edge from volatility mispricing

## References

- [Volatility Trading](https://www.amazon.com/Volatility-Trading-Euan-Sinclair/dp/0470181990) (Sinclair)
- [Option Volatility and Pricing](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774) (Natenberg)
- [Forecasting Realized Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1502915)
- [The Variance Risk Premium](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1316046)

## Difficulty Level

⭐⭐⭐⭐⭐ (Expert)

Требуется понимание: Options theory, Greeks, Volatility modeling, Delta hedging, Derivatives trading
