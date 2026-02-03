# Chapter 117: Concept Bottleneck Trading - Simple Guide

## What is This?

Imagine you have a robot that trades for you. Most robots are "black boxes" - they make decisions but can't explain why. That's scary when real money is involved!

**Concept Bottleneck Models (CBM)** fix this by making the robot think in human terms first.

## How Does It Work?

Think of it like this:

```
Market Data → "Is the market trending up?" → "Buy" or "Sell"
              "Is volatility high?"
              "Is there strong momentum?"
```

Instead of:
```
Market Data → [Magic Black Box] → "Buy" or "Sell"
```

## Simple Example

**Without CBM (Black Box):**
- Robot says: "Buy Bitcoin"
- You ask: "Why?"
- Robot says: "Because math"

**With CBM:**
- Robot says: "Buy Bitcoin"
- You ask: "Why?"
- Robot says: "Because trend is UP, volatility is LOW, momentum is POSITIVE"
- You can now agree or disagree!

## The Concepts We Use

| Concept | What It Means |
|---------|---------------|
| Trend | Is the price going up or down? |
| Volatility | How much is the price jumping around? |
| Momentum | Is the movement speeding up or slowing down? |
| Volume | Are many people trading? |

## Why Use This for Trading?

1. **You Understand Decisions**: Know exactly why trades happen
2. **You Can Intervene**: Override bad predictions
3. **You Can Debug**: Find out why you lost money
4. **Regulators Like It**: Easier to explain to compliance

## Quick Start

### For Beginners (Python)

```python
# Load some Bitcoin price data
from data_loader import BybitDataLoader
loader = BybitDataLoader()
prices = loader.fetch_klines("BTCUSDT", "1h", 100)

# Create and use the model
from model import ConceptBottleneckTrader
trader = ConceptBottleneckTrader()
signal, concepts = trader.predict(prices)

print(f"Signal: {signal}")  # Buy, Sell, or Hold
print(f"Why: {concepts}")   # The reasoning
```

### What You'll See

```
Signal: BUY
Why: {
  "trend": "bullish",
  "volatility": "low",
  "momentum": "positive",
  "volume": "above_average"
}
```

## Files in This Chapter

```
117_concept_bottleneck_trading/
├── python/           <- Python code (beginner-friendly)
│   ├── model.py      <- The main model
│   ├── concepts.py   <- How we calculate concepts
│   └── backtest.py   <- Test your strategy
└── rust_examples/    <- Rust code (for speed)
```

## Next Steps

1. Read the full README.md for technical details
2. Try the Python examples in the `python/` folder
3. Run a backtest to see how well it works
4. Experiment with different concepts!

## Summary

CBM = Make AI trading decisions understandable by humans.

Instead of trusting a black box, you can see exactly what the model "thinks" about the market before it trades.
