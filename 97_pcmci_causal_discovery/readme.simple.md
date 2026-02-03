# PCMCI Causal Discovery: The Detective That Finds Real Causes

Have you ever noticed that when ice cream sales go up, the number of people drowning also goes up? Does that mean ice cream causes drowning? Of course not! But a computer looking at raw data might think so. That is the problem PCMCI was built to solve.

In this chapter, we will learn how PCMCI works like a detective, sorting out what REALLY causes what, especially when it comes to the stock market and crypto trading.

---

## What Is Causal Discovery?

Imagine you are a scientist looking at data. You notice two things that always happen together:

- Ice cream sales go up.
- More people drown in swimming pools.

```
  Ice Cream Sales        Drownings
       ^                     ^
       |                     |
       +--- Both go UP ------+
```

A lazy scientist might say: "Ice cream causes drowning! Ban ice cream!"

But a SMART scientist asks: "Wait, is there something ELSE causing both of these?" And the answer is: **summer heat**!

```
              SUMMER HEAT
              /         \
             v           v
     Ice Cream Sales   Drownings
```

When it is hot outside, people buy more ice cream AND people swim more (so more accidents happen). The heat is the REAL cause. Ice cream and drowning just happen to occur at the same time. They are CORRELATED but one does NOT cause the other.

This difference between "things that happen together" (correlation) and "things that actually make other things happen" (causation) is one of the most important ideas in all of science.

**Causal discovery** is the science of figuring out which things ACTUALLY cause other things, rather than just happening at the same time.

---

## The Simple Analogy: PCMCI Is a Detective

Think of PCMCI as a detective investigating a crime scene.

**Lazy Detective:** "The suspect was seen near the crime scene. Case closed!"

**PCMCI Detective:** "The suspect was seen near the crime scene. But wait... let me check if there is a BETTER explanation. Let me test every possibility and rule out the fakes."

```
  LAZY DETECTIVE           PCMCI DETECTIVE
  +-----------+            +-----------+
  | Sees two  |            | Sees two  |
  | things    |            | things    |
  | together  |            | together  |
  +-----+-----+           +-----+-----+
        |                        |
        v                        v
  "They must               "Let me check ALL
   be related!"             possible explanations,
                            THEN decide."
```

PCMCI does not jump to conclusions. It carefully checks every relationship, tests alternatives, and only keeps the connections that are REAL.

---

## Why Does This Matter for Trading?

If you want to make money in the stock market, you need to know what CAUSES stock prices to move. Here is why this is tricky:

**Example 1: Fake Connection**
Every time a YouTuber posts, Bitcoin goes up. Is the YouTuber causing it? No! Both happen because of a big news event. The news is the real cause.

**Example 2: Real Connection**
When the US dollar gets stronger, gold goes down. This is a REAL causal relationship because gold is priced in dollars.

```
  FAKE RELATIONSHIP:
  YouTuber posts --> Bitcoin up?    NO!
  Big news event --> YouTuber posts
  Big news event --> Bitcoin up     YES! News is the real cause.

  REAL RELATIONSHIP:
  Dollar stronger --> Gold cheaper   YES! This is real causation.
```

If you build a trading strategy based on FAKE relationships, you will lose money. If you find REAL relationships, you have an edge. That is why PCMCI matters for traders.

---

## How PCMCI Works (Kid-Friendly Version)

PCMCI stands for **Peter-Clark Momentary Conditional Independence**. That sounds complicated, but it works in just two main steps. Think of it like cleaning your room in two passes.

### Step 1: The PC Phase (Find Suspects)

Imagine you are in a room full of 100 people, and you need to figure out who is friends with whom. In the first step, you just LOOK at everyone and notice who hangs out together.

This is the **PC phase**. PCMCI looks at all the data and finds every pair of things that MIGHT be related. It casts a wide net.

```
  ALL POSSIBLE CONNECTIONS:

  A ---?--- B
  |         |
  ?         ?
  |         |
  C ---?--- D ---?--- E

  "Hmm, lots of potential connections here.
   Let me investigate each one..."
```

But here is the smart part: even in this first step, PCMCI starts eliminating fake connections. If A and C only look related because they are BOTH connected to B, PCMCI notices that and removes the fake link.

It is like noticing that two kids in school are not actually friends with each other. They just both happen to be friends with the same popular kid.

```
  BEFORE:                    AFTER PC PHASE:
  A --- B --- C              A --- B --- C
  |           |
  +-----------+              (fake link removed!)
  (fake link)
```

### Step 2: The MCI Phase (Test Each Suspect)

Now PCMCI has a shorter list of "suspects" (potential causal links). In the second step, it tests each one more carefully.

For each potential connection, PCMCI asks: "If I account for EVERYTHING else that is happening, does this connection still hold?"

This is like a detective saying: "OK, the suspect has a motive. But does the suspect still look guilty when I consider the alibi, the other evidence, and the timeline?"

```
  MCI TEST FOR EACH LINK:

  Is A --> B real?

  Control for C: Still real?  YES --> KEEP IT
  Control for D: Still real?  YES --> KEEP IT
  Control for E: Still real?  YES --> CONFIRMED REAL!

  Is D --> E real?

  Control for A: Still real?  NO  --> REMOVE IT (was fake!)
```

The "Momentary" part means PCMCI also considers TIME. It checks if something that happened YESTERDAY causes something TODAY, which is perfect for time-series data like stock prices.

### Step 3: Build the Causal Map

After both phases, PCMCI gives you a clean map showing only the REAL causal connections:

```
  FINAL CAUSAL MAP:

  A -------> B -------> E
             |
             v
             C

  (Only the real connections survive!)
```

This map tells you: A causes B, B causes both C and E. Everything else was just noise or coincidence.

---

## Real Example with Stocks

Let us say you are tracking five things over the past year:

1. Gold price
2. Mining stock prices
3. The US Dollar index
4. Stock market fear index (VIX)
5. Social media buzz about gold

A lazy analysis might find that ALL of these are connected:

```
  CORRELATION VIEW (everything looks connected):

  Gold ------ Mining Stocks
   |    \    /     |
   |     \  /      |
   |      \/       |
  Dollar  VIX   Social Media
```

But PCMCI digs deeper and finds the REAL picture:

```
  PCMCI CAUSAL VIEW (only real causes):

  Dollar ------> Gold -------> Mining Stocks
                   |
  Market Crisis -> VIX
                   |
                   v
              Social Media Buzz
```

What PCMCI discovered:
- Dollar REALLY causes Gold to move (dollar up means gold down).
- Gold REALLY causes Mining Stocks to move (miners depend on gold price).
- Market crises cause the VIX (fear index) to spike.
- VIX causes social media buzz (scared people start posting).
- Gold and VIX looked connected but are NOT directly causing each other.

Now you know to watch the Dollar to predict Gold, instead of wasting time on social media posts!

---

## How Traders Use This

Here are the practical ways traders use PCMCI:

**1. Finding Lead-Lag Relationships**
PCMCI discovers that asset A moves BEFORE asset B. Watch A to predict B!

```
  Oil price (cause) --[1 day delay]--> Airline stocks (effect)
```

**2. Building Better Portfolios**
Know the REAL causal structure so you avoid putting money in things that crash together.

**3. Filtering Out Noise**
The market is full of false signals. PCMCI helps you focus on what actually matters.

**4. Crypto Markets**
PCMCI can reveal that Bitcoin truly leads altcoins, or that on-chain metrics predict price moves.

```
  BTC price      --[2 hours]--> ETH price        (real lead)
  Exchange volume --[1 day]---> Price volatility  (real cause)
  Twitter hype   --[???]------> Price             (often NOT real!)
```

---

## Simple Code Example

Here is a basic Python example showing how you would use PCMCI. You do not need to understand every line, just the overall idea.

```python
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Imagine we have daily data for 3 things:
# Column 0: Gold price changes
# Column 1: Dollar index changes
# Column 2: Mining stock changes
np.random.seed(42)

# Create some fake data where Dollar causes Gold,
# and Gold causes Mining stocks
T = 500  # 500 days of data
dollar  = np.random.randn(T)
gold    = np.zeros(T)
mining  = np.zeros(T)

for t in range(1, T):
    gold[t]   = -0.6 * dollar[t-1] + 0.3 * np.random.randn()
    mining[t] =  0.5 * gold[t-1]   + 0.3 * np.random.randn()

# Stack into one array
data = np.column_stack([dollar, gold, mining])
var_names = ["Dollar", "Gold", "Mining"]

# Set up PCMCI
dataframe = pp.DataFrame(data, var_names=var_names)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())

# Run PCMCI (check up to 3 days back in time)
results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.05)

# Print the results
pcmci.print_significant_links(
    p_matrix=results["p_matrix"],
    val_matrix=results["val_matrix"],
    alpha_level=0.05
)

# Expected output:
# Dollar --(-0.6)--> Gold    (lag 1)  REAL!
# Gold   --(+0.5)--> Mining  (lag 1)  REAL!
# No direct link from Dollar to Mining (PCMCI is smart enough
# to see that Dollar affects Mining ONLY through Gold)
```

What this code does:
1. Creates fake data where Dollar affects Gold, and Gold affects Mining.
2. Feeds it to PCMCI.
3. PCMCI correctly discovers the chain: Dollar causes Gold causes Mining.
4. It does NOT falsely claim Dollar directly causes Mining.

---

## Key Takeaways

- **Correlation is not causation.** Two things moving together does not mean one causes the other.
- **PCMCI is a two-step detective.** First it finds suspects (PC phase), then it tests them (MCI phase).
- **Time matters.** PCMCI checks if past events cause future events, perfect for trading.
- **It removes confounders.** If two stocks move together because of a hidden third factor, PCMCI figures that out.
- **Traders use it to find real edges.** Knowing what REALLY causes price moves beats chasing false patterns.
- **It works on any time-series data.** Stocks, crypto, economic indicators, weather, health data, and more.

---

## Fun Facts About Causality

**1. The Turkey Problem**
A turkey is fed every day for 1,000 days. It concludes: "The farmer must care about me!" On day 1,001 (Thanksgiving), the turkey discovers it was wrong. This is from philosopher Bertrand Russell and shows why blindly trusting patterns is dangerous.

**2. The Pirate-Temperature Connection**
There is a strong correlation between the decline in pirates and the rise in global temperatures. Pirates prevent global warming? Nope! This is a famous "spurious correlation" -- statistically real but meaningfully nonsense.

**3. Granger Causality Was Not Enough**
Before PCMCI, traders used "Granger causality." It was simpler but could not handle many variables and confounders at the same time. PCMCI fixed this by combining graph-based methods with time-series analysis.

**4. The Name "Peter-Clark"**
The "PC" comes from Peter Spirtes and Clark Glymour, who developed the original PC algorithm. The "MCI" part (Momentary Conditional Independence) was added by Jakob Runge in 2019 for time-series data.

**5. Domino Chains**
PCMCI can find "domino chains" in the market:

```
  Oil shock -> Transport costs rise -> Inflation -> Rate hikes -> Stocks drop
```

Finding these chains early gives traders a head start.

---

## Think About It

Next time someone tells you "studies show X is linked to Y," ask yourself the PCMCI question: "Is X really causing Y, or is there something else causing both?" That one question can save you from bad decisions, whether in trading, science, or everyday life.

PCMCI gives computers the ability to ask that question automatically, and that is what makes it such a powerful tool for anyone working with data.
