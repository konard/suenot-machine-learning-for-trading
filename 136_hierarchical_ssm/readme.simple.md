# Chapter 136: Hierarchical SSM — Explained Simply

## What Is This?

Imagine you're trying to predict the weather. You could look at:
- **Right now**: Is it cloudy? Is it windy? (seconds/minutes)
- **Today's trend**: Has it been getting warmer or cooler all day? (hours)
- **This week**: Is a cold front moving in? (days)
- **This season**: Are we heading into winter? (weeks/months)

Each "zoom level" tells you something different, and the best weather forecast uses ALL of them together.

**Hierarchical SSM (HiSS)** does exactly this, but for stock and crypto prices. It looks at price data at multiple zoom levels simultaneously to make better predictions.

## The "State Space Model" Part

A State Space Model (SSM) is like a smart filter that watches a stream of numbers and remembers the important stuff.

Think of it like a person watching a stock ticker:
- They see each new price (input)
- They keep a mental note of the trend (hidden state)
- They predict what comes next (output)

The "state space" is like their mental notebook — it stores the patterns they've noticed.

## The "Hierarchical" Part

"Hierarchical" means "layered" or "stacked" — like a pyramid.

**Real-life analogy**: Think of how a company works:
- **Workers** (Level 0) see day-to-day details
- **Managers** (Level 1) see weekly patterns
- **Executives** (Level 2) see monthly/yearly trends

Each level summarizes information from below and passes it up. Decisions are best when ALL levels contribute.

HiSS does the same thing with price data:

```
Level 0: Looks at every single price tick     → "Price just jumped!"
Level 1: Looks at hourly summaries            → "We're in an uptrend today"
Level 2: Looks at daily summaries             → "This week is bearish overall"
```

## Why Does This Help in Trading?

### The Problem with "Flat" Models

A regular model that looks at only one time scale is like a person who can only see one thing:
- Only ticks? They panic at every small move and miss the big picture
- Only daily? They miss fast opportunities and react too slowly

### The HiSS Solution

HiSS combines all scales, so it can say:
> "The daily trend is up (Level 2), the hourly momentum confirms it (Level 1), and there's a good entry point right now (Level 0) → **BUY**"

Or:
> "The daily trend is up (Level 2), but hourly shows weakening (Level 1), and there's selling pressure (Level 0) → **WAIT**"

## How It Works (Step by Step)

1. **Collect data**: Get price and volume data for a stock or crypto
2. **Level 0 processes it**: The finest level sees every data point
3. **Compress and send up**: Average groups of points together (like making an hourly summary from minutes)
4. **Level 1 processes summaries**: Finds medium-term patterns
5. **Compress and send up again**: Make daily summaries from hourly
6. **Level 2 processes**: Finds long-term trends
7. **Combine all levels**: Merge insights from all zoom levels
8. **Make prediction**: "Price will go up/down, by roughly this much, with this much uncertainty"

## A Day in the Life of HiSS

```
9:00 AM  - HiSS sees opening prices
           Level 2 says: "Weekly trend is bullish"
           Level 1 says: "Morning session starting neutral"
           Level 0 says: "Normal opening activity"
           Decision: HOLD

10:30 AM - Big volume spike
           Level 2 says: "Weekly trend still bullish"
           Level 1 says: "Strong buying pressure building"
           Level 0 says: "Breakout pattern detected!"
           Decision: BUY

2:00 PM  - Prices pulling back
           Level 2 says: "Weekly trend intact"
           Level 1 says: "Normal afternoon pullback"
           Level 0 says: "Some profit-taking"
           Decision: HOLD (pullback is normal)
```

## Key Takeaway

**HiSS is like having three traders working together** — one watches every tick, one watches hourly charts, and one watches daily charts. They share their insights and make better decisions together than any one of them could alone.

## Want to Try It?

### Python (easier to start)
```python
from python.model import HierarchicalSSM

# Create a model with 3 zoom levels
model = HierarchicalSSM(
    input_dim=8,
    hidden_dim=64,
    output_dim=3,
    num_levels=3,
    downsample_factors=[4, 4]
)
```

### Rust (faster for real trading)
```bash
cargo run --example basic_hiss
```

Both do the same thing — the Python version is great for learning and experiments, while the Rust version is fast enough for real-time trading.
