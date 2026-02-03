# Counterfactual Trading — Explained Simply!

## What is a Counterfactual?

A counterfactual is just a fancy word for "what if?"

```
What actually happened:
  You bought Apple stock → Made $100

Counterfactual question:
  "What if I had NOT bought Apple stock?"
  Answer: You would have made $0 (no position)
```

The difference ($100 - $0 = $100) is called the **treatment effect** — the real impact of your trading decision.

---

## Why Should Traders Care?

### The Problem with Traditional Analysis

Imagine you bought a stock and made 10% profit. Was that because:
- Your analysis was brilliant?
- The whole market went up 12%?
- You just got lucky?

Traditional analysis can't separate these!

```
Traditional view:
  "I bought stock → I made money → I'm a good trader!"

Reality might be:
  "Market went up 15% → I made 10% → I actually underperformed!"
```

### Counterfactual Analysis Fixes This

```
What happened:
  I traded → Made 10%

Counterfactual:
  If I hadn't traded → Would have made 0%

My actual contribution: 10% - 0% = +10% (Good!)

---

What happened:
  I traded → Made 10%

Counterfactual:
  If I had just held the index → Would have made 15%

My actual contribution: 10% - 15% = -5% (Bad!)
```

---

## The Core Idea: Two Parallel Worlds

Think of it like a movie where you can see two timelines:

```
TIMELINE A (What Happened):
────────────────────────────────────────
You → Buy Stock → Price Goes Up → Profit $100

TIMELINE B (Counterfactual):
────────────────────────────────────────
You → Don't Buy → Price Goes Up → Profit $0
      (same you)   (same market)

Your true contribution = $100 - $0 = $100
```

The key insight: Everything else (the market, the economy, other investors) stays the same. Only YOUR action is different.

---

## A Simple Trading Example

### Scenario: Should I Have Made That Trade?

```
Morning:
  - Bitcoin is at $50,000
  - Your model says "BUY"
  - You buy 1 BTC

Evening:
  - Bitcoin is at $51,000
  - You sell for $1,000 profit

Question: Was the model signal valuable?
```

### Traditional Analysis
```
"I followed the signal and made $1,000. Signal = Good!"
```

### Counterfactual Analysis
```
What if I had ignored the signal?

Scenario 1: I would have held my existing BTC
  → Still made $1,000
  → Signal value = $0 (I would have made same amount)

Scenario 2: I would have stayed in cash
  → Made $0
  → Signal value = $1,000 (Signal was valuable!)

Scenario 3: I would have bought more aggressively
  → Made $2,000
  → Signal value = -$1,000 (Signal held me back!)
```

---

## The Fundamental Problem

**We can never observe both worlds at the same time!**

```
You can only live ONE timeline:

✓ You bought the stock and saw what happened
✗ You CAN'T also not-buy and see that outcome

This is called the "Fundamental Problem of Causal Inference"
```

### So How Do We Solve It?

We estimate the counterfactual using:
1. **Similar situations** from the past
2. **Statistical models** that predict outcomes
3. **Smart math** that combines multiple approaches

---

## Three Methods to Estimate Counterfactuals

### Method 1: Find Similar Situations (Matching)

```
Your trade:
  Features: RSI=25, Trend=Up, Volatility=Low
  Action: Bought
  Outcome: +2%

Find similar situations where you DIDN'T trade:
  Similar trade 1: RSI=27, Trend=Up, Vol=Low, No Buy → +1.5%
  Similar trade 2: RSI=24, Trend=Up, Vol=Low, No Buy → +1.8%
  Similar trade 3: RSI=26, Trend=Up, Vol=Low, No Buy → +1.6%

Estimated counterfactual: ~1.6%

Your treatment effect: 2% - 1.6% = +0.4% (You added value!)
```

### Method 2: Build a Prediction Model

```
Train a model to predict: Outcome = f(Features, Action)

For your trade:
  Predicted outcome if traded: 2%
  Predicted outcome if not traded: 1.5%

Your treatment effect: 2% - 1.5% = +0.5%
```

### Method 3: Doubly Robust (Best of Both)

```
Combines matching AND prediction models
More accurate than either alone
Gives confidence intervals

Result: Treatment effect = 0.45% ± 0.1%
```

---

## Real-World Application: Strategy Attribution

### The Big Question

"My strategy made 50% this year. How much was skill vs luck?"

### Counterfactual Decomposition

```
Total Return: +50%

Breakdown:
├── Market Return (if I did nothing): +35%
├── Strategy Alpha (my decisions): +18%
└── Unexplained/Luck: -3%

TRUE skill contribution: 18%
```

### Why This Matters

```
Traditional thinking:
  "50% return! I'm amazing!"

Counterfactual thinking:
  "Market gave 35% for free. I only added 18%.
   But 18% IS still good! Just not as amazing as 50%."
```

---

## Regret Analysis: Learning from Mistakes

### What is Regret?

```
Regret = What you COULD have made - What you DID make

Positive regret = You made a mistake
Zero regret = You made the optimal choice
```

### Example

```
Day 1:
  You: Bought Stock A → Made +2%
  Counterfactual: Could have bought Stock B → Would make +5%
  Regret: 5% - 2% = 3%

Day 2:
  You: Sold early → Made +1%
  Counterfactual: If held longer → Would make +4%
  Regret: 4% - 1% = 3%

Day 3:
  You: Bought at the dip → Made +3%
  Counterfactual: If didn't buy → Would make +0%
  Regret: 0% (You did better!)

Total Regret: 3% + 3% + 0% = 6%
```

### Using Regret to Improve

```
High regret situations → Study what went wrong
Zero regret situations → Reinforce those patterns
```

---

## The Trading Decision Framework

### Before Trading

```
1. What do I expect to happen if I trade?
2. What do I expect if I DON'T trade?
3. Is the difference worth the risk?

Expected Treatment Effect = E[Trade] - E[No Trade]

If Expected Effect > Transaction Costs → Trade!
```

### After Trading

```
1. What actually happened?
2. Estimate: What would have happened otherwise?
3. Calculate: Was my decision valuable?

Actual Treatment Effect = Actual - Counterfactual

If Effect > 0 → Good decision!
If Effect < 0 → Learn from it
```

---

## Common Mistakes to Avoid

### Mistake 1: Ignoring the Market

```
Wrong: "I made 20% this year!"
Right: "I made 20%, but market made 25%. I underperformed."
```

### Mistake 2: Cherry-Picking Comparisons

```
Wrong: "I beat my neighbor who doesn't invest!"
Right: "I should compare to a reasonable alternative (index fund)"
```

### Mistake 3: Not Accounting for Risk

```
Wrong: "Strategy A made more than B, so A is better"
Right: "Strategy A made more but took 3x risk.
       Risk-adjusted, B is better."
```

### Mistake 4: Confusing Correlation with Causation

```
Wrong: "I always trade when RSI < 30 and make money.
       RSI < 30 causes profits!"

Right: "RSI < 30 might just correlate with market bottoms.
       Need to check if MY TRADE adds value beyond the bounce."
```

---

## Quick Python Example

```python
# Simple counterfactual analysis
from model import CounterfactualEstimator

# Your trading data
features = get_market_features()  # RSI, volume, etc.
treatment = get_trade_signals()   # 1 = traded, 0 = didn't
outcome = get_returns()           # Actual returns

# Fit the model
estimator = CounterfactualEstimator()
estimator.fit(features, treatment, outcome)

# Analyze a specific trade
result = estimator.estimate_counterfactual(
    features=today_features,
    treatment=1,  # You traded
    observed_outcome=0.02  # Made 2%
)

print(f"You made: {result.observed_outcome:.2%}")
print(f"If you hadn't traded: {result.counterfactual_outcome:.2%}")
print(f"Your contribution: {result.treatment_effect:.2%}")
```

---

## Key Takeaways

1. **Counterfactual = "What if?"**
   - Compare what happened to what WOULD have happened

2. **Separate skill from luck**
   - Market gains aren't your skill
   - Strategy alpha IS your skill

3. **Use regret to learn**
   - Positive regret = opportunity to improve
   - Zero regret = good decisions

4. **Three estimation methods**
   - Matching (find similar situations)
   - Prediction (model the outcomes)
   - Doubly robust (combine both)

5. **Apply before AND after trading**
   - Before: Should I trade?
   - After: Was my decision good?

---

## The Bottom Line

**Traditional analysis asks:** "Did I make money?"

**Counterfactual analysis asks:** "Did I make money BECAUSE of my decision, or DESPITE it?"

```
Making money when the market goes up: Easy
Making money when the market goes down: Hard
Making MORE money than doing nothing: The real test!
```

The trader who understands counterfactuals knows the difference between being skilled and being lucky. And that knowledge is the first step to consistent performance.

---

## Try It Yourself!

```bash
# Python
cd 110_counterfactual_trading/python
pip install -r requirements.txt
python model.py      # Basic counterfactual estimation
python backtest.py   # Full backtesting with attribution

# Rust (faster!)
cd 110_counterfactual_trading/rust
cargo run --example counterfactual_demo
```
