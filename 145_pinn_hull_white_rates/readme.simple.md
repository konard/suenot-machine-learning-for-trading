# Chapter 145: Hull-White Interest Rates Explained Simply

## Imagine a Spring Pulling Back

Let's understand Hull-White interest rates through a simple analogy!

---

## The Spring Analogy

### What happens when you pull a spring?

Imagine you have a spring attached to a wall. There is a ball on the spring.

```
Wall  =====[spring]=====O  <-- ball
                         |
                    Normal position
```

When you pull the ball away from its normal position:

```
Wall  =====[spring]================O  <-- pulled far right
                                    |
                              "Rates too high!"
```

The spring pulls it back:

```
Wall  =====[spring]=====O  <-- back to normal
                         |
                    "Ahh, comfortable"
```

And if you push it the other way:

```
Wall  ==O  <-- pushed far left
         |
    "Rates too low!"
```

The spring pulls it back again!

**Interest rates work exactly like this spring.** They tend to come back to a "normal" level over time.

---

## What Are Interest Rates?

### The Simple Version

When you put money in a bank, the bank pays you for letting them use your money. That payment is the **interest rate**.

```
You: "Here's $100, bank!"
Bank: "Thanks! I'll pay you 3% per year."
After 1 year: You get $103
```

### Why Do Rates Change?

Interest rates go up and down based on many things:

```
Rates go UP when:
  - Economy is growing fast
  - Inflation is rising
  - Central bank wants to slow spending

Rates go DOWN when:
  - Economy is slowing
  - Inflation is falling
  - Central bank wants to encourage spending
```

### The Key Insight: Rates Come Back

Here is the crucial observation: rates do not keep going up forever or down forever. They always **come back** to some normal level, just like our spring!

```
Rate
  |
8%|        *
  |       * *
6%|      *   *              *
  |     *     *            * *
4%|- - *- - - -*--- - - -*-*- -*- - - - - (long-run average)
  |  *         *        *       *
2%| *           *      *         *
  |              *    *
0%|               *  *
  |                **
  +-----------------------------------------> Time
```

This behavior is called **mean reversion**.

---

## What is the Hull-White Model?

### In Plain Words

The Hull-White model is a mathematical formula that says:

> "Interest rates behave like a ball on a spring. They bounce around randomly,
> but they always get pulled back toward a normal level."

### The Formula (Do Not Panic!)

```
Change in rate = [pull toward normal] + [random wobble]

Or more precisely:
  dr = [theta(t) - a * r] * dt + sigma * dW
```

Let's break this down:

```
dr          = How much the rate changes in a tiny moment
theta(t)    = Where the spring's "rest position" is (can change over time)
a           = How STRONG the spring is (how fast rates come back)
r           = Where the rate is RIGHT NOW
sigma       = How wildly the rate wobbles (volatility)
dW          = Random push (like wind blowing the ball)
```

### Real-Life Example

```
Today's rate: 5%
Normal rate:  3%
Spring strength (a): 0.1

The spring is pulling DOWN because 5% is above 3%:
  Pull = 0.1 * (5% - 3%) = 0.2% per year downward

Plus some random wobble:
  Wobble = maybe +0.1% or -0.1% (unpredictable)

Expected rate next year: approximately 4.8% (pulled down from 5%)
```

---

## What is a PINN?

### The Problem

We want to know: "If rates are at 5% today, how much should a bond cost?"

There is a complicated equation (PDE) that gives the answer, but solving it perfectly is hard.

### The Traditional Way

Old method: Build a big grid (like graph paper) and calculate the answer at every point.

```
r
|  .  .  .  .  .  .
|  .  .  .  .  .  .
|  .  .  .  .  .  .
|  .  .  .  .  .  .
+---.---.---.---.---> t
Each dot needs a calculation!
```

Problem: Need thousands of dots. Slow!

### The PINN Way

A **Physics-Informed Neural Network** is like a smart student:

1. Learn the physics rules (the PDE equation)
2. Learn from some known answers (market data)
3. Figure out the answer everywhere, not just at grid points

```
Input: (rate, time) ---> [Neural Network] ---> Output: bond price

The network is TRAINED to follow the physics rules:
  - The equation must be satisfied (physics loss)
  - At maturity, bond price = $1 (boundary condition)
  - Match the real yield curve (data loss)
```

Think of it like this:

```
Traditional method: Calculate every single point on a map
PINN method: Learn the PATTERN and predict any point instantly
```

---

## Why Do We Care About Bond Prices?

### Bonds Are Like IOUs

A bond is a promise: "I will pay you $100 in 5 years."

How much would you pay TODAY for that promise?

```
If rates are HIGH (say 10%):
  $100 in 5 years is worth only about $62 today
  (You could put $62 in the bank at 10% and get $100 in 5 years)

If rates are LOW (say 1%):
  $100 in 5 years is worth about $95 today
  (You'd need $95 in the bank at 1% to get $100 in 5 years)
```

So: **when rates go up, bond prices go down**, and vice versa!

```
Rates UP    -->  Bond prices DOWN
Rates DOWN  -->  Bond prices UP
```

### The Yield Curve

Different bonds have different time horizons. The "yield curve" shows rates for different times:

```
Rate (%)
  |
5%|  *
  |    *
4%|      *  *  *  *  *  *
  |
3%|
  +--+-+-+-+-+-+-+-+-+----> Years
  0  1 2 3 5 7 10   30

Short-term: might be high (central bank policy)
Long-term: more stable (market expectations)
```

---

## What About Crypto?

### Funding Rates = Crypto Interest Rates

In crypto markets, perpetual futures contracts have "funding rates" -- these are like interest rates:

```
Positive funding rate (+0.01%):
  "Long traders pay short traders every 8 hours"
  (Everyone is bullish, so longs pay a premium)

Negative funding rate (-0.01%):
  "Short traders pay long traders every 8 hours"
  (Everyone is bearish)
```

### They Act Like Springs Too!

```
Funding Rate
     |
+0.1%|     *           *
     |    * *         * *
  0% |---*---*---*---*---*--- (normal level)
     |         * * *
-0.1%|
     +-------------------------> Time
```

When funding gets too high, arbitrageurs bring it back. Just like the spring!

### Using Hull-White for Crypto

We can use the same model for crypto funding rates:

```
Traditional Finance:
  r = Federal Funds Rate
  Traded: Treasury bonds, interest rate swaps

Crypto Finance:
  r = Bybit Funding Rate
  Traded: Perpetual futures, lending rates

Same math, different markets!
```

---

## The Trading Strategy

### Mean Reversion Trading

If we know rates come back to normal, we can trade on this:

```
Step 1: Estimate the "normal" level
Step 2: When rate is FAR ABOVE normal --> bet it will come down
Step 3: When rate is FAR BELOW normal --> bet it will go up
Step 4: Close the bet when rate returns to normal

          Entry: rate too high!
              |
Rate          v
  |        * *
  |       *   *
  |------*-----*------  <-- Normal level
  |                *
  |               * *
  |              *       <-- Entry: rate too low!
  |
  +-------|-----|-------> Time
          ^     ^
          Sell  Buy back (profit!)
```

### Real Example with Bybit

```
Bybit BTCUSDT funding rate: +0.03% (very high!)
Normal funding rate: +0.005%

Strategy: SHORT the perpetual (collect the high funding)
Expect: Funding will drop back to 0.005%
Result: Collect funding payments until rate normalizes
```

---

## What Are Derivatives?

### Caps, Floors, and Swaptions (Simple Explanation)

**Cap**: Insurance that rates will not go above a certain level.

```
"I have a loan at floating rate. I'm worried rates will skyrocket!"
"Buy a cap at 5%. If rates go above 5%, the cap pays you the difference."

Rate goes to 7%?  Cap pays you 2% (7% - 5%)
Rate stays at 3%? Cap pays nothing (you don't need it)
```

**Floor**: Insurance that rates will not go below a certain level.

```
"I have savings at floating rate. I'm worried rates will crash!"
"Buy a floor at 2%. If rates drop below 2%, the floor pays you."
```

**Swaption**: The right (not obligation) to enter a swap in the future.

```
"I might want to lock in a fixed rate in 1 year."
"Buy a swaption. In 1 year, you can choose to enter the swap or not."
```

---

## Putting It All Together

```
1. Get data:
   - Treasury yields (traditional)
   - Bybit funding rates (crypto)

2. Calibrate Hull-White:
   - How strong is the spring? (mean reversion speed)
   - How much does it wobble? (volatility)

3. Train the PINN:
   - Learn the physics (PDE)
   - Fit to market data

4. Use the trained PINN:
   - Price bonds instantly
   - Build yield curves
   - Price derivatives (caps, floors, swaptions)
   - Compute risk measures (duration, VaR)

5. Trade:
   - Mean reversion on funding rates
   - Duration management
   - Derivatives hedging
```

---

## Summary

| Concept | Simple Explanation |
|---------|-------------------|
| Hull-White Model | Rates are like a ball on a spring -- they bounce around but always come back |
| Mean Reversion | The spring pulls rates back to normal |
| PINN | A smart neural network that learns physics rules |
| Bond Pricing | How much is a future payment worth today? |
| Yield Curve | Interest rates for different time periods |
| Funding Rate | Crypto's version of interest rates |
| Cap | Insurance against high rates |
| Floor | Insurance against low rates |

---

## Try It Yourself!

```python
# Install
pip install torch numpy scipy matplotlib requests

# Run
cd 145_pinn_hull_white_rates/python
python train.py          # Train the PINN
python analytical.py     # See analytical prices
python data_loader.py    # Fetch Bybit data
python backtest.py       # Test a trading strategy
```

Remember: Interest rates are just a ball on a spring. The spring always pulls back!
