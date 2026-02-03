# Chapter 121: Layer-wise Relevance Propagation (LRP) - Simple Guide

## What is LRP? A Simple Explanation

Imagine you have a robot that predicts whether to buy or sell stocks. The robot looks at many things: price, volume, market trends, etc. Then it says "BUY!" But... **why** does it say buy?

**LRP is like asking the robot to explain its decision.**

```
Without LRP:                    With LRP:
┌─────────────┐                ┌─────────────┐
│   Robot     │                │   Robot     │
│             │                │             │
│  "BUY!" ???│       VS       │  "BUY!"     │
│             │                │  Because:   │
│ (no reason) │                │  - Price +40%│
│             │                │  - Volume+35%│
│             │                │  - RSI    +25%│
└─────────────┘                └─────────────┘
```

## The Detective Analogy

Think of LRP like a detective solving a case:

1. **The Crime (Output)**: The robot made a prediction
2. **The Evidence (Input)**: Price, volume, indicators
3. **The Investigation (LRP)**: Figure out which evidence led to the verdict

```
The Investigation:

   VERDICT: "BUY"
       │
       ▼
   ┌───────┐  "Who contributed?"
   │  LRP  │◄─────────────────────
   └───────┘
       │
       ▼
   ┌─────────────────────────────┐
   │ Price momentum:    40%      │
   │ Trading volume:    35%      │
   │ RSI indicator:     25%      │
   │ ─────────────────────────   │
   │ Total:            100%      │
   └─────────────────────────────┘
```

## How Does It Work?

### Step 1: The Robot Makes a Prediction

The neural network looks at data and outputs a prediction score.

```
Input Data          Neural Network           Output
┌──────────┐       ┌──────────────┐        ┌──────┐
│ Price    │──────▶│              │───────▶│      │
│ Volume   │──────▶│   Layers     │───────▶│ 0.85 │ = "BUY"
│ RSI      │──────▶│              │───────▶│      │
│ MACD     │──────▶│              │        └──────┘
└──────────┘       └──────────────┘
```

### Step 2: LRP Goes Backwards

Now LRP takes the output (0.85) and traces back through each layer to find out which inputs were responsible.

```
Input Data          Neural Network           Output
┌──────────┐       ┌──────────────┐        ┌──────┐
│ Price    │◀──────│              │◀───────│      │
│ Volume   │◀──────│   Relevance  │◀───────│ 0.85 │
│ RSI      │◀──────│   flows back │◀───────│      │
│ MACD     │◀──────│              │        └──────┘
└──────────┘       └──────────────┘
   │
   ▼
Price:  0.34 (40%)
Volume: 0.30 (35%)
RSI:    0.21 (25%)
```

### Step 3: The Conservation Rule

**Important rule**: The total relevance stays the same!

```
At output:  0.85 (100%)
At input:   0.34 + 0.30 + 0.21 = 0.85 (100%)

Nothing is lost, nothing is created!
(Like energy in physics)
```

## A Pizza Analogy

Imagine you and 3 friends order a pizza. The pizza costs $20.

**LRP is like figuring out how much each person contributed:**

```
Pizza Cost: $20
┌─────────────────────────────────┐
│ Alice paid:  $8  (40%)          │
│ Bob paid:    $7  (35%)          │
│ Carol paid:  $5  (25%)          │
│ ─────────────────────────────   │
│ Total:      $20 (100%)          │
└─────────────────────────────────┘
```

In LRP:
- **Pizza cost** = Model's prediction (output)
- **Friends** = Input features
- **Each person's payment** = How much each feature contributed

## Why is LRP Useful in Trading?

### 1. Understanding Decisions

```
Before LRP:
Trader: "Why did the model say SELL?"
Model: "..."

After LRP:
Trader: "Why did the model say SELL?"
Model: "Because volume dropped 45% and RSI shows overbought!"
```

### 2. Finding Problems

If LRP shows the model focuses on wrong things:

```
BAD (Model learned wrong patterns):
┌────────────────────────────────┐
│ Day of week:     60%           │  <-- This shouldn't matter!
│ Hour of day:     30%           │  <-- This shouldn't matter!
│ Actual price:    10%           │  <-- Too low!
└────────────────────────────────┘

GOOD (Model learned right patterns):
┌────────────────────────────────┐
│ Price momentum:  45%           │  <-- Makes sense!
│ Volume change:   35%           │  <-- Makes sense!
│ Volatility:      20%           │  <-- Makes sense!
└────────────────────────────────┘
```

### 3. Building Trust

When a model can explain itself, traders trust it more:

```
"I recommend buying BTCUSDT because:
 - Strong upward momentum (45% weight)
 - Increasing volume (30% weight)
 - RSI recovering from oversold (25% weight)"
```

## Simple Code Example

Here's a very simple example of how LRP works:

```python
# Simple LRP example

# Imagine a tiny neural network
# Input: [price_change, volume_change]
# Output: prediction score

# Forward pass (making prediction)
price_change = 0.05    # 5% price increase
volume_change = 0.10   # 10% volume increase

# Weights (how much the network cares about each input)
weight_price = 0.6
weight_volume = 0.4

# Prediction
prediction = price_change * weight_price + volume_change * weight_volume
# prediction = 0.05 * 0.6 + 0.10 * 0.4 = 0.03 + 0.04 = 0.07

print(f"Prediction: {prediction}")  # 0.07

# LRP backward pass (explaining the prediction)
# How much did each input contribute?

# Price contribution
price_relevance = (price_change * weight_price) / prediction * prediction
# = 0.03 / 0.07 * 0.07 = 0.03 (43%)

# Volume contribution
volume_relevance = (volume_change * weight_volume) / prediction * prediction
# = 0.04 / 0.07 * 0.07 = 0.04 (57%)

print(f"Price contributed: {price_relevance:.2f} ({price_relevance/prediction*100:.0f}%)")
print(f"Volume contributed: {volume_relevance:.2f} ({volume_relevance/prediction*100:.0f}%)")

# Check: relevances sum to prediction!
print(f"Total: {price_relevance + volume_relevance:.2f}")  # 0.07
```

Output:
```
Prediction: 0.07
Price contributed: 0.03 (43%)
Volume contributed: 0.04 (57%)
Total: 0.07
```

## Different LRP Rules

LRP has different "rules" for tracing back through the network. Think of them as different investigation techniques:

### LRP-0: Basic Rule
The simplest method. Like dividing equally based on contribution.

### LRP-epsilon: Stable Rule
Adds a small safety number to avoid math problems. Like having a backup plan.

### LRP-gamma: Positive Focus Rule
Focuses more on positive evidence. Like a detective focusing on what points TO the suspect, not away.

```
┌─────────────────────────────────────────────────┐
│ Rule Comparison:                                │
│                                                 │
│ LRP-0:       Simple, can be unstable            │
│ LRP-epsilon: Stable, good for most cases        │
│ LRP-gamma:   Focus on positive evidence         │
│                                                 │
│ Best practice: Use different rules for          │
│ different layers!                               │
└─────────────────────────────────────────────────┘
```

## Real-World Trading Example

Let's say a model predicts BTC price direction:

```
Today's Market Data:
┌─────────────────────────────────┐
│ BTC Price Change:    +2.5%      │
│ Trading Volume:      High       │
│ RSI:                 45         │
│ MACD:                Bullish    │
│ Market Sentiment:    Positive   │
└─────────────────────────────────┘

Model Prediction: BUY (confidence: 78%)

LRP Explanation:
┌─────────────────────────────────┐
│ MACD Bullish:        35%        │
│ Price momentum:      28%        │
│ Volume increase:     22%        │
│ RSI normal:          10%        │
│ Sentiment positive:   5%        │
│ ─────────────────────────────   │
│ Total:              100%        │
└─────────────────────────────────┘

Interpretation:
"The model recommends BUY mainly because MACD
shows bullish signal (35%) and price has
positive momentum (28%). Volume increase (22%)
provides additional confirmation."
```

## Summary

| Concept | Simple Explanation |
|---------|-------------------|
| **LRP** | A way to explain neural network decisions |
| **Relevance** | How much each input contributed to the output |
| **Conservation** | All relevances add up to the output value |
| **Rules** | Different methods for calculating relevance |
| **Purpose** | Understanding, debugging, and trusting AI |

## Key Takeaways

1. **LRP explains WHY** a neural network made a prediction
2. **Relevance is conserved** - the sum of all input relevances equals the output
3. **Different rules** exist for different situations
4. **Trading benefits** include understanding, debugging, and regulatory compliance
5. **Trust increases** when models can explain themselves

## What's Next?

After understanding this simple guide:
1. Read the main [README.md](README.md) for technical details
2. Try the [Python examples](python/)
3. Explore the [Rust implementation](rust_lrp/)

---

**Difficulty Level: Beginner**

No advanced math required - just curiosity and basic understanding of how neural networks make predictions!
