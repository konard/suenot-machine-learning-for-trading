# Chapter 105: Difference-in-Differences — Explained Simply

## What Is This?

Imagine you want to know if a new coffee shop in your neighborhood actually brought more people to the area, or if people were already coming more often anyway.

You could compare:
- **Your street** (got the coffee shop) - Before vs After
- **The next street over** (no coffee shop) - Before vs After

If both streets got more crowded by the same amount, then it wasn't the coffee shop — it was something else (like summer starting). But if YOUR street got MORE crowded than the other street, the difference is probably because of the coffee shop!

**Difference-in-Differences (DiD)** does exactly this, but for stocks and cryptocurrencies. It helps us figure out if an event (like a new regulation or exchange listing) actually caused a price change, or if the market was moving that way anyway.

## The Key Idea

DiD uses a simple formula:

```
Real Effect = (Change in affected group) - (Change in unaffected group)
```

**Real-life example**: Did a new tax on tech companies hurt their stock prices?

- **Tech stocks** dropped 5% after the tax was announced
- **Non-tech stocks** dropped 3% during the same period (general market downturn)
- **Real effect of the tax** = 5% - 3% = 2% drop

Without the comparison group, you'd think the tax caused a 5% drop. But 3% was happening anyway!

## The "Parallel Trends" Rule

For DiD to work, both groups should move together BEFORE the event happens.

**Good example for DiD:**
```
Before the event:
  Tech stocks:     100 → 102 → 104 → 106  (growing +2 each period)
  Non-tech stocks: 100 → 102 → 104 → 106  (growing +2 each period)
                                     ↓ EVENT HAPPENS
After the event:
  Tech stocks:     106 → 104 → 102        (now falling!)
  Non-tech stocks: 106 → 108 → 110        (still growing)

The divergence = effect of the event
```

**Bad example for DiD:**
```
Before the event:
  Tech stocks:     100 → 105 → 112 → 118  (growing fast)
  Non-tech stocks: 100 → 101 → 102 → 103  (growing slow)
                                     ↓ EVENT HAPPENS
After the event:
  ???

We can't use DiD because they weren't moving together before!
```

## How It Helps in Trading

### The Problem with Simple Analysis

If Bitcoin gets listed on a new exchange and the price goes up 10%, is that because of the listing? Or was Bitcoin already going up?

A simple "before vs after" comparison gets fooled by market trends.

### The DiD Solution

Compare Bitcoin (got listed) with Ethereum (didn't get listed):

```
                    Before Listing    After Listing    Change
Bitcoin (listed)         $30,000         $33,000      +$3,000 (+10%)
Ethereum (not listed)    $2,000          $2,100       +$100 (+5%)

DiD Effect: 10% - 5% = 5% real listing premium!
```

The market was already going up (+5%), but Bitcoin got an EXTRA 5% boost from the listing.

## Trading Strategy: Step by Step

### 1. Spot an Event
- A new regulation is announced
- A crypto gets listed on a major exchange
- The Fed changes interest rates
- A company gets sued

### 2. Identify Groups
- **Treated**: Assets directly affected
- **Control**: Similar assets NOT affected

### 3. Check Parallel Trends
Look at the past 30+ days. Were both groups moving together?

### 4. Calculate the Effect
After the event, measure how much the treated group moved BEYOND what the control group moved.

### 5. Trade on It
- If DiD shows a big positive effect → Consider buying (the effect might continue)
- If DiD shows a big negative effect → Consider shorting (the effect might continue)
- If DiD shows no significant effect → Don't trade (it was just noise)

## A Day in the Life of DiD Trading

```
9:00 AM  - News Alert: "SEC announces new rules for DeFi platforms"

           Identify groups:
           - Treated: DeFi tokens (affected by SEC)
           - Control: Layer-1 coins like Bitcoin, Ethereum (not directly affected)

10:00 AM - Check pre-event data:
           Both groups moved together over the past month ✓
           Parallel trends assumption satisfied!

11:00 AM - Calculate DiD after market opens:
           DeFi tokens: -8%
           Control tokens: -2%
           DiD Effect: -8% - (-2%) = -6% (regulatory penalty)

           Signal: SELL DeFi tokens

3:00 PM  - Update analysis with more data:
           Effect is statistically significant (p < 0.05)
           Maintain short position

Next Day - Effect stabilizes around -5%
           Take profit and close position
```

## Common Pitfalls

### 1. Bad Control Group
**Wrong**: Comparing Bitcoin to Apple stock after crypto regulation
**Right**: Comparing Bitcoin to Ethereum after a Bitcoin-specific event

### 2. Ignoring Pre-trends
Always check that your groups moved together BEFORE the event. If they didn't, your DiD estimate is meaningless.

### 3. Too Short a Window
You need enough data before AND after the event. A few hours isn't enough — aim for days or weeks.

### 4. Multiple Events
If two events happen at once, you can't separate their effects. Wait for "clean" events.

## Key Takeaway

**DiD is like having a "what would have happened" machine.** Instead of guessing, you use a comparison group to estimate what the affected assets would have done WITHOUT the event. The difference is your trading signal.

## Want to Try It?

### Python (easier to start)
```python
from python.model import DifferenceInDifferences

# Create a DiD model
model = DifferenceInDifferences(
    treatment_col='treated',
    time_col='post_treatment',
    outcome_col='return',
)

# Fit and get results
results = model.fit(your_data)
print(f"DiD Effect: {results.did_estimate:.2%}")
print(f"p-value: {results.p_value:.4f}")
```

### Rust (faster for real trading)
```bash
cargo run --example basic_did
```

Both do the same thing — Python is great for learning and testing ideas, while Rust is fast enough for live trading systems.

## Quick Reference Card

| Step | What to Do |
|------|------------|
| 1. Event | Find a market event that affects some assets but not others |
| 2. Groups | Identify treated (affected) and control (unaffected) assets |
| 3. Pre-check | Verify both groups had similar trends before the event |
| 4. Calculate | DiD = (Treated change) - (Control change) |
| 5. Test | Check if the effect is statistically significant |
| 6. Trade | Long if DiD > 0, Short if DiD < 0, Skip if not significant |
