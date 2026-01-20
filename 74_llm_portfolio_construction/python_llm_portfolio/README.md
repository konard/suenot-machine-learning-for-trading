# LLM Portfolio Construction - Python Implementation

This module provides Python implementations for LLM-based portfolio construction, supporting both cryptocurrency (Bybit) and stock market (Yahoo Finance) data.

## Installation

```bash
pip install -r requirements.txt
```

## Components

### 1. Data Clients

- **`bybit_client.py`**: Fetches cryptocurrency market data from Bybit exchange
- **`stock_client.py`**: Fetches stock market data using yfinance

### 2. Portfolio Engine

- **`portfolio_engine.py`**: Core portfolio construction engine with:
  - LLM-based asset scoring
  - Score-weighted portfolio generation
  - Mean-variance optimization
  - Risk parity allocation

### 3. Backtesting

- **`backtester.py`**: Portfolio backtesting framework with:
  - Multiple rebalancing frequencies
  - Transaction costs and slippage
  - Performance metrics calculation
  - Strategy comparison

## Quick Start

### Crypto Portfolio

```python
from bybit_client import BybitClient, PortfolioDataFetcher
from portfolio_engine import LLMPortfolioEngine, Asset, AssetClass

# Fetch data
client = BybitClient()
fetcher = PortfolioDataFetcher(client)
data = fetcher.fetch_portfolio_data(["BTCUSDT", "ETHUSDT", "SOLUSDT"], days=30)

# Create assets
assets = [
    Asset("BTCUSDT", "Bitcoin", AssetClass.CRYPTO, 65000),
    Asset("ETHUSDT", "Ethereum", AssetClass.CRYPTO, 3200),
    Asset("SOLUSDT", "Solana", AssetClass.CRYPTO, 140),
]

# Analyze with LLM (mock mode for demo)
engine = LLMPortfolioEngine()
scores = engine.analyze_assets_mock(assets, {}, [])
portfolio = engine.generate_portfolio(scores)
print(portfolio)
```

### Stock Portfolio

```python
from stock_client import StockClient, StockPortfolioDataFetcher
from portfolio_engine import LLMPortfolioEngine, Asset, AssetClass

# Fetch data
client = StockClient()
fetcher = StockPortfolioDataFetcher(client)
data = fetcher.fetch_portfolio_data(["AAPL", "MSFT", "GOOGL"], period="1y")

# Create assets
assets = [
    Asset("AAPL", "Apple Inc", AssetClass.EQUITY, 185),
    Asset("MSFT", "Microsoft", AssetClass.EQUITY, 420),
    Asset("GOOGL", "Alphabet", AssetClass.EQUITY, 175),
]

# Generate portfolio
engine = LLMPortfolioEngine()
scores = engine.analyze_assets_mock(assets, {}, [])
portfolio = engine.generate_portfolio(scores)
print(portfolio)
```

### Backtesting

```python
from backtester import PortfolioBacktester, StrategyComparison
import pandas as pd

# Load your price data
price_df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# Backtest equal weight
backtester = PortfolioBacktester(
    initial_capital=100000,
    rebalance_frequency="weekly"
)

weights = {"BTCUSDT": 0.33, "ETHUSDT": 0.33, "SOLUSDT": 0.34}
result = backtester.run_static_weights(price_df, weights)
print(result.summary())
```

## LLM Integration

To use real LLM analysis (not mock), set your API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

engine = LLMPortfolioEngine(api_key=os.environ["OPENAI_API_KEY"])
scores = engine.analyze_assets(assets, market_data, news_headlines)
```

## Performance Metrics

The backtester calculates:

- Total Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Turnover

## License

MIT
