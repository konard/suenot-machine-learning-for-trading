# Causal Factor Discovery — Explained Simply!

## What's the Problem with Correlation?

Imagine you notice that every time you see more fire trucks in your neighborhood, there's more damage to buildings. Should you conclude that fire trucks CAUSE damage?

Of course not! Both fire trucks AND damage are caused by the same thing: a fire.

```
WRONG conclusion:
Fire Trucks ───────────→ Building Damage
            "causes"

RIGHT understanding:
              Fire
             /    \
            ↓      ↓
     Fire Trucks  Building Damage
```

This is called a **spurious correlation** — two things that appear connected but don't actually cause each other.

---

## Why Does This Matter for Trading?

In trading, we're constantly looking for "factors" that predict stock returns. The problem? Many correlations we find are spurious:

### The Ice Cream Problem in Finance

```
Factor: "Stocks go up when ice cream sales are high!"

Reality:
- Ice cream sales ↑ when it's summer
- People are happier in summer
- Happy people invest more
- Stocks go up

The ice cream isn't causing stock rises — SUMMER is causing both!
```

If you trade based on ice cream sales, your strategy will fail when:
- Summer is unusually cold
- People's investing behavior changes
- Any other confounding factor changes

---

## What is Causal Factor Discovery?

Instead of just looking for correlations, causal factor discovery tries to figure out the **actual cause-and-effect relationships**.

### The Detective Analogy

Think of yourself as a detective solving a mystery:

**Correlation Detective** (Traditional):
- "The suspect was seen near the crime scene"
- "The suspect has a motive"
- "Therefore, the suspect is guilty!"
- *Problem: Could be coincidence!*

**Causal Detective** (Our Approach):
- "Did the suspect have opportunity?"
- "Is there a chain of events connecting them to the crime?"
- "Can we rule out other explanations?"
- "Can we establish a causal mechanism?"
- *Result: Much more reliable conclusion*

---

## How Does It Work? A Simple Example

Imagine we have three market factors:
- **Interest Rates** (set by central banks)
- **Bond Prices** (affected by interest rates)
- **Stock Returns** (what we want to predict)

### Step 1: Look at Correlations

```
Interest Rates ←──→ Bond Prices    (correlated)
Bond Prices ←──→ Stock Returns     (correlated)
Interest Rates ←──→ Stock Returns  (correlated)
```

Hmm, everything is correlated with everything! Which one actually CAUSES stock returns?

### Step 2: Test for Independence

We ask: "If we know Interest Rates, does knowing Bond Prices tell us anything NEW about Stock Returns?"

```
Test: Stock Returns ⊥ Bond Prices | Interest Rates

Translation: "Are stock returns independent of bond prices,
              once we already know interest rates?"
```

If yes → Bond Prices doesn't DIRECTLY cause stock returns; it's just a middleman.

### Step 3: Build the Causal Graph

After testing all combinations, we discover:

```
Interest Rates
      |
      ↓
Bond Prices ←───── (some other factors)
      |
      ↓
Stock Returns

The TRUE causal chain!
```

Now we know: Interest Rates → Bond Prices → Stock Returns

---

## The PC Algorithm: Finding Causes Step by Step

The main algorithm we use is called **PC** (after its inventors, Peter and Clark).

### Think of it Like Sculpting

**Start**: Assume EVERYTHING is connected (a big messy blob)

```
Step 0: Complete Graph
A ─── B ─── C
 \   / \   /
  \ /   \ /
   D ─── E
```

**Then**: Remove connections we can prove are NOT causal

```
Step 1: Test direct pairs
"Is A independent of E?" → YES → Remove A─E
"Is B independent of D?" → YES → Remove B─D
```

```
Step 2: Test with conditions
"Is A independent of C, given B?" → YES → Remove A─C
```

**Finally**: Figure out the direction of arrows (what causes what)

```
Final Graph:
A → B → C
    ↓
    D → E
```

---

## Real-World Trading Example

Let's say we're trying to predict Bitcoin returns, and we have these potential factors:

1. **Trading Volume** - How much BTC is traded
2. **Volatility** - How much price swings
3. **Social Media Buzz** - Twitter mentions of Bitcoin
4. **Google Searches** - People searching "Bitcoin"
5. **Stock Market (S&P 500)** - General market sentiment

### Traditional Approach (Correlation-Based)

```
All factors correlated with returns:
- Volume: r = 0.15
- Volatility: r = 0.20
- Social Media: r = 0.18
- Google Searches: r = 0.12
- S&P 500: r = 0.08

Decision: Use ALL factors in model!
```

Problem: Many of these might be spurious or redundant.

### Causal Approach

```
Causal Discovery finds:

Social Media Buzz
       ↓
Google Searches → Trading Volume
                       ↓
                   Volatility
                       ↓
                  BTC Returns
                       ↑
                   S&P 500

Actual causal factors for returns:
1. Volatility (direct cause)
2. S&P 500 (direct cause)

NOT causal (spurious):
- Social Media (causes searches, not returns)
- Google Searches (causes volume, not returns)
- Volume (causes volatility, but not directly returns)
```

### The Benefit

When market conditions change:
- **Correlation model**: Breaks down because spurious correlations disappear
- **Causal model**: Still works because TRUE causes are stable

---

## Why Causal Relationships Are More Stable

### The Weather Analogy

**Correlation-based prediction:**
- "When my neighbor washes his car, it rains the next day"
- Works sometimes... until it doesn't!

**Causal-based prediction:**
- "When humidity is high and a cold front approaches, it rains"
- Works consistently because it's the actual mechanism

### In Trading Terms

| Situation | Correlation Factor | Causal Factor |
|-----------|-------------------|---------------|
| Bull Market | Factor X predicts returns | Causal Factor Y predicts returns |
| Bear Market | Factor X stops working | Causal Factor Y still works |
| Market Crash | All correlations break | Causal relationships hold |
| New Regime | Need to find new correlations | Same causes still apply |

---

## Key Concepts Simplified

### 1. Confounding
```
                  Hidden Factor
                   /        \
                  ↓          ↓
          Factor A          Factor B
               "appear correlated"

Reality: A doesn't cause B, they just share a common cause!
```

### 2. Reverse Causation
```
What we think:   News → Stock Movement
Reality:         Stock Movement → News Coverage

Reporters write about stocks BECAUSE they moved,
not the other way around!
```

### 3. Selection Bias
```
We only study successful traders...
All successful traders drink coffee...
Conclusion: Coffee makes traders successful?

NO! We ignored all the failed traders who also drink coffee.
```

---

## The Bottom Line

**Traditional factor models ask:** "What is correlated with returns?"
- Easy to find
- Lots of false signals
- Breaks down during market stress

**Causal factor models ask:** "What actually CAUSES returns?"
- Harder to find
- Fewer factors, but more reliable
- Stable across market regimes

```
Traditional:
100 correlated factors → 90 are spurious → Strategy fails

Causal:
100 potential factors → 5 causal factors → Strategy is robust
```

---

## Try It Yourself!

### Quick Python Example

```python
# Traditional: Find correlated factors
correlations = data.corr()['returns'].sort_values()
top_correlated = correlations.tail(10)  # Top 10 correlated

# Causal: Find actual causes
from causal_discovery import CausalFactorModel

model = CausalFactorModel()
model.fit(factors, returns)
causal_factors = model.causal_factors  # Usually much fewer!

print(f"Correlated factors: {len(top_correlated)}")
print(f"Causal factors: {len(causal_factors)}")
# Output: "Correlated: 10, Causal: 3"
```

### Run the Examples

```bash
# Python version
cd 109_causal_factor_discovery/python
pip install -r requirements.txt
python model.py

# Rust version (faster!)
cd 109_causal_factor_discovery
cargo run --example discover_factors
```

---

## Key Takeaways

1. **Correlation ≠ Causation** — Just because two things move together doesn't mean one causes the other

2. **Spurious factors hurt performance** — Trading on false signals leads to losses

3. **Causal factors are stable** — True causes work across different market conditions

4. **Less is more** — 3 causal factors beat 30 correlated factors

5. **The PC algorithm** — A systematic way to discover what causes what

6. **Always test for confounders** — Ask "Is there a hidden factor causing both?"

---

## One Final Analogy

**Correlation-based trading** is like driving by looking in the rearview mirror — you can see where you've been, but not where you're going.

**Causal-based trading** is like understanding how the engine works — you know WHY the car moves and can predict what happens when conditions change.

Which would you trust to get you where you want to go?
