# LLM Risk Assessment - Python Implementation

A Python implementation for LLM-based risk assessment in trading.

## Features

- **Multiple LLM Providers** - Support for OpenAI, Anthropic, and local models
- **Bybit Integration** - Fetch cryptocurrency market data
- **Risk Scoring** - Multi-dimensional risk assessment
- **Trading Strategies** - Risk-based signal generation
- **Backtesting** - Strategy validation framework

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Risk Assessment

```python
from risk_assessment import LLMRiskAssessor, LLMProvider

# Initialize assessor (uses OPENAI_API_KEY environment variable)
assessor = LLMRiskAssessor(provider=LLMProvider.OPENAI)

# Analyze text
text = "Bitcoin faces regulatory scrutiny as SEC announces investigation"
score = assessor.assess(text)

print(f"Overall Risk: {score.overall()}/10")
print(f"Risk Level: {score.risk_level.name}")
print(f"Position Multiplier: {score.position_multiplier()}")
```

### Fetch Market Data

```python
from bybit_client import BybitClient

client = BybitClient()

# Fetch recent klines
klines = client.fetch_klines("BTCUSDT", "60", 100)

# Fetch historical data
historical = client.fetch_historical_klines("BTCUSDT", "D", 30)

# Convert to DataFrame
df = client.to_dataframe(klines)
print(df.tail())
```

### Trading Strategy

```python
from trading_strategy import RiskBasedTrader, Backtester

# Create trader
trader = RiskBasedTrader(
    risk_threshold_long=4.0,
    risk_threshold_short=7.0,
    max_position_size=0.5,
)

# Generate signal
signal = trader.generate_signal(risk_score, current_price)

if signal.signal_type.value == "long":
    print(f"Buy with {signal.position_size:.1%} of capital")
```

### Backtesting

```python
from trading_strategy import Backtester, RiskBasedTrader

backtester = Backtester(
    initial_capital=10000.0,
    commission_rate=0.001,
)

result = backtester.run(prices, risk_scores, trader)
result.print_summary()
```

## Modules

### risk_assessment.py

Core risk assessment functionality:
- `RiskScore` - Multi-dimensional risk score dataclass
- `RiskLevel` - Risk level enumeration
- `LLMRiskAssessor` - LLM-based risk analyzer

### bybit_client.py

Market data fetching:
- `OHLCV` - Candlestick data structure
- `BybitClient` - Bybit API client
- `MarketDataAggregator` - Multi-symbol data fetcher

### trading_strategy.py

Trading strategy components:
- `TradingSignal` - Signal with position sizing
- `RiskBasedTrader` - Risk-based signal generator
- `Backtester` - Strategy backtesting framework
- `BacktestResult` - Performance metrics

## Configuration

Set environment variables for API access:

```bash
export OPENAI_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Examples

Run the demo scripts:

```bash
# Risk assessment demo
python risk_assessment.py

# Bybit client demo
python bybit_client.py

# Trading strategy demo
python trading_strategy.py
```

## Risk Dimensions

| Dimension | Description |
|-----------|-------------|
| Market Risk | Exposure to market volatility |
| Credit Risk | Counterparty default probability |
| Liquidity Risk | Ability to trade without impact |
| Operational Risk | System/process failures |
| Regulatory Risk | Legal/compliance exposure |
| Sentiment Risk | Market perception shifts |

## License

MIT License
