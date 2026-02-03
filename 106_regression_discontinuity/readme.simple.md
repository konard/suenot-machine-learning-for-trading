# Chapter 106: Regression Discontinuity — Simple Explanation

## What Is This?

Imagine you are in school, and the rule is: "Students with grades 70 or higher pass, students below 70 fail."

Now think about a student who scored 71 and another who scored 69. Are they really that different in ability? Probably not — the difference is just 2 points, which could be luck on one question.

But their outcomes are VERY different: one passes, one fails. This sharp change at a specific number (70) is what we call a **discontinuity**.

**Regression Discontinuity Design (RDD)** is a method that uses these sharp cutoffs to understand cause and effect. In trading, many such cutoffs exist, and they create predictable price movements we can trade on!

## A Real-Life Analogy

### The Birthday Party Example

Imagine a birthday party where:
- Kids age 10 and above get a big slice of cake
- Kids under 10 get a small slice

Now look at a kid who just turned 10 yesterday and one who turns 10 tomorrow. They're basically the same age! But one gets a big slice and one gets a small slice.

If we compare ONLY kids very close to age 10, we can see the pure effect of the "big slice" rule — because the kids are so similar in age that any difference in how full they feel must be due to the cake size, not their age.

### In Trading Terms

In the stock market, there are similar rules:
- Companies with market cap rank 1-1000 are in the Russell 1000 index
- Companies with rank 1001-3000 are in the Russell 2000 index

A company ranked 999 and one ranked 1001 are almost identical in size. But one is in Russell 1000 and one is in Russell 2000. Because of this, their stock prices move differently!

Index funds that track Russell 2000 MUST buy the company at rank 1001 but NOT the one at rank 999. This buying pressure causes the stock price to go up.

## How Does It Work?

### Step 1: Find the Threshold

First, we need to identify where the "cutoff" or "threshold" is:

```
Examples of thresholds in trading:
- Market cap rank 1000 (Russell index)
- RSI indicator = 30 (oversold signal)
- Stock price = $100 (round number)
- Funding rate = +1% (crypto perpetual futures)
```

### Step 2: Look at Units Just Above and Below

We compare things that are VERY close to the threshold:

```
Just below:  rank 950-999   (NOT in Russell 2000)
Just above:  rank 1001-1050 (IN Russell 2000)
```

Because these companies are so similar in size, any difference in their stock returns must be caused by the index membership, not by size differences.

### Step 3: Measure the Jump

We measure how different the outcomes are right at the threshold:

```
Companies just ABOVE rank 1000: Average return = +5%
Companies just BELOW rank 1000: Average return = +1%
───────────────────────────────────────────────────
The "jump" at the threshold = 5% - 1% = 4%
```

This 4% jump is the CAUSAL effect of being added to the index!

### Step 4: Trade on It

If we know the effect ahead of time, we can profit:

```
1. Before reconstitution: Buy stocks that will be added
2. Hold through the event
3. After index funds buy: Sell for profit
```

## Why Is This Useful for Trading?

### Causal vs. Correlation

Most trading "signals" are just correlations that might be fake:
- "Stocks that went up yesterday tend to go up today" — Is this real or coincidence?

RDD signals are CAUSAL:
- "Stocks added to an index go up BECAUSE of the forced buying" — This is a real effect!

### Predictable Timing

Many RDD effects have predictable timing:
- Russell reconstitution: Every June
- Technical indicator thresholds: When indicator crosses level
- Earnings announcements: Scheduled dates

### Real Examples in Markets

| Threshold | What Happens | Effect |
|---|---|---|
| Russell 2000 addition | Index funds must buy | +5% price increase |
| RSI crosses below 30 | Traders buy "oversold" stocks | Price bounces |
| Stock hits $100 | Stop-loss orders trigger | Price breaks or reverses |
| Crypto funding rate > 50% | Leveraged longs close | Price drops |

## Simple Example

### Trading the RSI Threshold

RSI (Relative Strength Index) is a popular indicator that goes from 0 to 100.

Many traders believe:
- RSI < 30 = "oversold" = BUY signal
- RSI > 70 = "overbought" = SELL signal

Let's use RDD to check if this is real:

```
Step 1: Collect data on Bitcoin hourly candles
Step 2: For each candle, record RSI and next-day return
Step 3: Compare returns when RSI is 28-29 vs. 31-32

If RSI "oversold" signal is real:
  Returns when RSI = 28-29: Higher (strong buy signal)
  Returns when RSI = 31-32: Lower (weaker signal)

We should see a JUMP in returns right at RSI = 30!
```

### The Russell Index Trade

Every May, we know:
- Companies near rank 1000 might switch indexes
- If moved to Russell 2000, they get buying pressure
- If moved to Russell 1000, they lose buying pressure

**Trading strategy:**
```
1. In May: Identify companies ranked 980-1020
2. Predict which will be added/removed
3. Buy predicted additions, short predicted deletions
4. Hold until late June (reconstitution)
5. Exit after index funds finish rebalancing
```

Historical results: ~5% return for additions, ~-3% for deletions!

## Key Terms

| Term | Simple Explanation |
|---|---|
| **Running Variable** | The number that determines treatment (like market cap rank) |
| **Cutoff/Threshold** | The specific number where the rule kicks in (like rank 1000) |
| **Treatment** | What happens when you cross the threshold (like being added to index) |
| **Bandwidth** | How close to the threshold we look (like "within 50 ranks") |
| **Sharp RDD** | When the rule is 100% followed (above = treated, below = not) |
| **Fuzzy RDD** | When the rule is mostly but not always followed |
| **Treatment Effect** | The size of the jump at the threshold |
| **Local Effect** | The effect only applies to things near the threshold |

## Comparison with Other Methods

| Method | Finds Causation? | Easy to Use? | Many Opportunities? |
|---|---|---|---|
| Correlation trading | No | Yes | Yes |
| Machine learning | No | No | Yes |
| A/B testing | Yes | Hard to do | No |
| **RDD Trading** | **Yes** | **Medium** | **Limited** |

## What the Code Does

### Python Code
- `rdd_model.py`: Estimates the size of the jump at the threshold
- `data_loader.py`: Downloads stock and crypto price data
- `backtest.py`: Tests the strategy on historical data

### Rust Code
- Same functions as Python, but runs much faster
- Good for real-time monitoring of thresholds
- Used in production trading systems

## Limitations

### Only Works Near the Threshold

RDD only tells us what happens to things VERY CLOSE to the cutoff. It doesn't tell us:
- What happens far from the threshold
- What would happen if we moved the threshold

### Need Enough Data

We need many observations near the threshold. If only 5 companies are ranked 990-1010 each year, we might not have enough data.

### No Manipulation

If companies can easily manipulate their rank, RDD breaks down. For example, if a company knows it's at rank 1005, it might try to reduce its market cap to stay in Russell 2000. This would mess up our analysis.

## Summary

Regression Discontinuity Design is like finding places where the rules of the game create unfair advantages. Just like a kid who turns 10 gets more cake, a stock that crosses into an index gets more buyers.

By identifying these thresholds and trading around them, we can profit from predictable, CAUSAL effects — not just random correlations that might disappear tomorrow.

The key insight: **When rules create sharp cutoffs, prices must adjust. If we know the cutoff in advance, we can trade on it.**
