# Mediation Analysis for Finance - A Simple Guide

## What is Mediation Analysis? (The "Middle Man" Story)

Imagine you're a detective trying to understand how things influence each other. Sometimes, the connection isn't direct - there's a "middle man" involved!

### Real Life Example: The Pizza Party Effect

Let's say you notice that when the sun is shining, ice cream shops make more money. But WHY?

```
Direct thinking:
    Sunny Weather ───────────────► Ice Cream Sales

    "Sun makes people buy ice cream"

But the REAL story might be:

    Sunny Weather ────► More People Outside ────► Ice Cream Sales
         │                                              ▲
         └──────── (maybe some direct effect too) ──────┘

    The "middle man" is "More People Outside"!
```

This "middle man" is called a **MEDIATOR** - something that helps explain HOW one thing affects another.

## Why Should Traders Care?

### The Stock Market Detective Story

Imagine you discover that good news about a company makes its stock price go up. But you want to know HOW this happens:

```
Simple view:
    Good News ──────────────────► Stock Price ↑

    "Good news = price up. Done!"

Detective view:
    Good News ────► People Get Excited ────► They Buy ────► Stock Price ↑
         │              (Sentiment)           (Volume)           ▲
         │                                                       │
         └──────────── some people react instantly ──────────────┘

    Now we understand the MECHANISM!
```

### Why This Matters for Trading

1. **Better Timing**: If you know the middle step (people getting excited → buying), you can watch for when this chain is working or broken

2. **Smarter Signals**: Not all "good news" works the same way
   - If sentiment increases but volume doesn't → chain might be broken
   - If volume increases without sentiment → different mechanism at work

3. **Risk Management**: Different pathways have different risks!

## The Three Main Characters

Every mediation story has three main characters:

```
┌──────────────────────────────────────────────────────────────────┐
│                   THE MEDIATION CAST                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  🎬 X = The TREATMENT (The Thing That Starts It All)             │
│      Examples: News event, earnings report, market crash         │
│                                                                   │
│  🔄 M = The MEDIATOR (The Middle Man)                            │
│      Examples: Sentiment, trading volume, analyst opinions       │
│                                                                   │
│  🎯 Y = The OUTCOME (What We Want to Predict)                    │
│      Examples: Stock returns, price movement, volatility         │
│                                                                   │
│  The Story: X ───► M ───► Y                                      │
│             │             ▲                                       │
│             └─────────────┘                                       │
│             (and maybe X affects Y directly too!)                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Breaking Down the Effects

Think of it like splitting a pie:

```
TOTAL EFFECT PIE = DIRECT SLICE + INDIRECT SLICE

┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│    Total Effect = 100%                                           │
│    ┌─────────────────────────────────────────┐                   │
│    │                                         │                   │
│    │   DIRECT          │    INDIRECT         │                   │
│    │   40%             │    60%              │                   │
│    │                   │                     │                   │
│    │   X goes          │    X goes through   │                   │
│    │   straight to Y   │    M to reach Y     │                   │
│    │                   │                     │                   │
│    └───────────────────┴─────────────────────┘                   │
│                                                                   │
│    In formulas:  Total = Direct + Indirect                       │
│                  100%  =  40%   +   60%                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## A Trading Example: Bitcoin → Ethereum

Let's see a real crypto example:

```
Question: Does Bitcoin price movement affect Ethereum price?
          And if so, HOW?

Setup:
    X = Bitcoin returns (did BTC go up or down?)
    M = Market sentiment (are people excited or scared?)
    Y = Ethereum returns (what happens to ETH?)

The Story:

    BTC Price ↑ ────► Crypto Optimism ↑ ────► ETH Price ↑
         │                                         ▲
         │                                         │
         └─── direct correlation (same market) ────┘

What we might find:
    - Total effect: When BTC goes up 1%, ETH goes up 0.8%
    - Direct effect: 0.3% (they move together because same market)
    - Indirect effect: 0.5% (via sentiment spreading)

    Proportion mediated: 0.5 / 0.8 = 62.5%

    Meaning: Most of how BTC affects ETH is through sentiment!
```

## Types of Mediation (With Pizza Examples!)

### 1. Full Mediation (The Pizza Delivery Story)

```
Your hunger doesn't magically make pizza appear at your door:

    Hunger ────► You Order Pizza ────► Pizza Arrives

    The "ordering" is the ONLY way hunger leads to pizza.
    This is FULL MEDIATION - the middle step is EVERYTHING.

    In trading: Insider info → Trading activity → Price change
                (The info only matters because someone trades on it)
```

### 2. Partial Mediation (The Rainy Day Story)

```
Rain affects your mood in two ways:

    Rain ────► You Stay Inside ────► Feel Bored
      │                                   ▲
      └───── (the sound is annoying) ─────┘

    Rain affects your mood BOTH:
    - Through staying inside (indirect)
    - AND directly (the sound, the gloom)

    In trading: News → Volume + Sentiment → Price
                (News has direct and indirect effects)
```

### 3. No Mediation (The Ice in Your Drink Story)

```
Wanting a cold drink leads to ice AND a cold drink,
but ice doesn't CAUSE the coldness of your drink choice:

    Want Cold Drink ────────────────► Choose Cold Drink
           │
           └──────► Add Ice
                    (related but not on the path)

    In trading: Market crash → You panic + Prices fall
                (Your panic didn't cause the fall)
```

## The Classic Method: Baron & Kenny's Four Steps

Think of it as a recipe:

```
┌──────────────────────────────────────────────────────────────────┐
│              THE FOUR-STEP MEDIATION RECIPE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: CHECK THE MAIN CONNECTION                               │
│          Does X actually predict Y?                              │
│          "Does good news → higher returns?"                      │
│          If NO → Stop! There's nothing to mediate.               │
│                                                                   │
│  Step 2: CHECK THE FIRST LINK                                    │
│          Does X predict M?                                       │
│          "Does good news → positive sentiment?"                  │
│          If NO → M isn't a mediator for X.                       │
│                                                                   │
│  Step 3: CHECK THE SECOND LINK                                   │
│          Does M predict Y (when we also include X)?              │
│          "Does sentiment → returns (beyond just news)?"          │
│          If NO → M doesn't add anything.                         │
│                                                                   │
│  Step 4: COMPARE THE EFFECTS                                     │
│          Did X's effect on Y get smaller when we added M?        │
│          "Is news effect smaller when we account for sentiment?" │
│          If YES → Mediation confirmed!                           │
│                                                                   │
│          If X's effect becomes ZERO → Full mediation             │
│          If X's effect gets smaller but not zero → Partial       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Simple Python Example

Here's a basic example you can understand:

```python
import numpy as np

# Create some fake but realistic data
np.random.seed(42)
n = 100  # 100 days of data

# X = Market news sentiment score (-1 to 1)
X = np.random.randn(n) * 0.5

# M = Trading volume (affected by news)
# When news is strong, more people trade
M = 0.4 * X + np.random.randn(n) * 0.3

# Y = Stock returns
# Affected by BOTH news directly AND volume
Y = 0.3 * X + 0.5 * M + np.random.randn(n) * 0.2

# Now let's find the effects!

# STEP 1: Total effect (just X predicting Y)
# Y = c * X + error
from numpy.linalg import lstsq
X_design = np.column_stack([np.ones(n), X])
beta_total = lstsq(X_design, Y, rcond=None)[0]
total_effect = beta_total[1]
print(f"Total Effect (c): {total_effect:.3f}")

# STEP 2: X predicting M
beta_a = lstsq(X_design, M, rcond=None)[0]
a_path = beta_a[1]
print(f"a path (X → M): {a_path:.3f}")

# STEP 3: Both X and M predicting Y
XM_design = np.column_stack([np.ones(n), X, M])
beta_direct = lstsq(XM_design, Y, rcond=None)[0]
direct_effect = beta_direct[1]  # c'
b_path = beta_direct[2]         # b
print(f"Direct Effect (c'): {direct_effect:.3f}")
print(f"b path (M → Y): {b_path:.3f}")

# Calculate indirect effect
indirect_effect = a_path * b_path
print(f"\nIndirect Effect (a × b): {indirect_effect:.3f}")

# Check: Total should equal Direct + Indirect
print(f"\nVerification:")
print(f"  Total:            {total_effect:.3f}")
print(f"  Direct + Indirect: {direct_effect + indirect_effect:.3f}")

# Proportion mediated
prop_mediated = indirect_effect / total_effect
print(f"\nProportion Mediated: {prop_mediated:.1%}")
```

Expected output:
```
Total Effect (c): 0.495
a path (X → M): 0.412
Direct Effect (c'): 0.287
b path (M → Y): 0.503

Indirect Effect (a × b): 0.207

Verification:
  Total:            0.495
  Direct + Indirect: 0.494

Proportion Mediated: 41.9%
```

## How to Use This in Trading

### The Mediation-Aware Trading Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│             TRADING WITH MEDIATION INSIGHTS                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SITUATION 1: Mediator Channel is ACTIVE                         │
│  ─────────────────────────────────────────────                   │
│  Signal:  Good news appeared                                      │
│  Check:   Is sentiment rising? Is volume increasing?             │
│  Result:  If YES → Expect FULL EFFECT (direct + indirect)        │
│  Action:  Take a STRONGER position                               │
│                                                                   │
│  SITUATION 2: Mediator Channel is BLOCKED                        │
│  ─────────────────────────────────────────────                   │
│  Signal:  Good news appeared                                      │
│  Check:   Sentiment flat? Volume normal?                         │
│  Result:  If YES → Expect ONLY DIRECT EFFECT                     │
│  Action:  Take a SMALLER position (or wait)                      │
│                                                                   │
│  SITUATION 3: Mediator Active Without Treatment                  │
│  ─────────────────────────────────────────────                   │
│  Signal:  No big news                                            │
│  Check:   But sentiment/volume spiking anyway?                   │
│  Result:  Something else is going on!                            │
│  Action:  Investigate before trading                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Simple Trading Rules

```python
def should_trade(news_signal, sentiment_change, volume_change,
                 mediation_threshold=0.3):
    """
    Simple mediation-aware trading decision

    Args:
        news_signal: Strength of news (-1 to 1)
        sentiment_change: How much sentiment moved
        volume_change: How much volume changed from normal
        mediation_threshold: How much of effect comes from mediator

    Returns:
        decision: 'strong_long', 'weak_long', 'strong_short',
                  'weak_short', or 'no_trade'
    """

    # Is the mediator channel active?
    mediator_active = (abs(sentiment_change) > 0.5 and
                      abs(volume_change) > 0.3)

    # Base signal from news
    if abs(news_signal) < 0.2:
        return 'no_trade'  # Not enough signal

    if news_signal > 0:
        if mediator_active:
            return 'strong_long'  # Full effect expected
        else:
            return 'weak_long'    # Only direct effect
    else:
        if mediator_active:
            return 'strong_short'
        else:
            return 'weak_short'

# Example usage
decision = should_trade(
    news_signal=0.7,      # Positive news
    sentiment_change=0.8,  # Sentiment jumped
    volume_change=0.5      # Volume up
)
print(f"Trading decision: {decision}")
# Output: Trading decision: strong_long
```

## Key Takeaways

### Remember These Points:

```
1️⃣  MEDIATION = Understanding the "how" and "why"
    Not just "X predicts Y" but "X → M → Y"

2️⃣  THREE CHARACTERS: Treatment (X), Mediator (M), Outcome (Y)

3️⃣  TWO EFFECTS: Direct (X straight to Y) and Indirect (X through M to Y)

4️⃣  TRADING USE: Watch if the mediator channel is active or blocked

5️⃣  CAUTION: Correlation ≠ Causation! Mediation assumes causal structure.
```

### Common Mistakes to Avoid

```
❌ DON'T: Assume every correlated variable is a mediator
   → Just because A, B, and C correlate doesn't mean B mediates A→C

❌ DON'T: Ignore the order of events
   → The mediator MUST come BETWEEN treatment and outcome in time

❌ DON'T: Over-interpret small effects
   → Statistical significance doesn't mean trading significance

❌ DON'T: Forget about other explanations
   → There might be hidden factors affecting everything

✅ DO: Think about the mechanism first
   → "WHY would X affect M, and WHY would M affect Y?"

✅ DO: Test your assumptions
   → Run the analysis, check the numbers, be skeptical

✅ DO: Consider timing
   → How fast does each step happen? Seconds? Days? Weeks?
```

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────┐
│                    MEDIATION CHEAT SHEET                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FORMULAS:                                                        │
│    Total Effect:     c  (from Y ~ X)                             │
│    Direct Effect:    c' (from Y ~ X + M)                         │
│    Indirect Effect:  a × b                                       │
│    Proportion Mediated: (a × b) / c                              │
│                                                                   │
│  PATHS:                                                           │
│    a = X → M  (treatment affects mediator)                       │
│    b = M → Y  (mediator affects outcome, controlling for X)      │
│    c = X → Y  (total effect)                                     │
│    c' = X → Y (direct effect, with M in model)                   │
│                                                                   │
│  INTERPRETATIONS:                                                 │
│    Full mediation:    c' ≈ 0 (all effect through M)             │
│    Partial mediation: c' < c but c' ≠ 0 (both paths work)       │
│    No mediation:      a×b ≈ 0 (M doesn't carry the effect)      │
│                                                                   │
│  TRADING INSIGHT:                                                 │
│    Mediator ACTIVE:   Expect full effect → Larger position       │
│    Mediator BLOCKED:  Expect direct only → Smaller position      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

*This is a simplified guide to help you understand mediation analysis. For the full technical details, mathematical proofs, and production code, see the main README.md file.*
