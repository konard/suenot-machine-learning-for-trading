# TCN for Trading: Simple Explanation for Beginners

## Imagine You're a Weather Forecaster

### The Problem: Predicting Tomorrow's Weather

Imagine you're trying to predict tomorrow's weather. You look out the window and see:
- Today it's sunny
- Yesterday it rained
- The day before was cloudy
- Last week was cold...

**Question:** How do you use ALL this past information to predict tomorrow?

This is exactly what TCN (Temporal Convolutional Network) does, but for cryptocurrency prices!

---

## What is TCN in Simple Terms?

### Analogy: The Detective with a Magnifying Glass

Imagine a detective examining footprints in the sand:

```
🔍 Small Magnifying Glass (zoom = 1)
   - Sees only 2-3 footprints
   - Notices: "This footprint is deeper than the next one"
   - Finds SMALL details

🔍 Medium Magnifying Glass (zoom = 2)
   - Sees 5-6 footprints
   - Notices: "The person was speeding up"
   - Finds MEDIUM patterns

🔍 Large Magnifying Glass (zoom = 4)
   - Sees 10-12 footprints
   - Notices: "The person walked in a circle!"
   - Finds LARGE patterns
```

**TCN works just like this detective!** It looks at price charts with "magnifying glasses" of different sizes:
- Small zoom: sees minute-by-minute changes
- Medium zoom: sees hourly trends
- Large zoom: sees daily and weekly patterns

---

## Why is TCN Better Than Other Methods?

### Comparison with a Human Memory

**Regular Neural Networks (RNN/LSTM):**
```
Like reading a book ONE WORD at a time:
📖 "The" → remember
📖 "cat" → remember, but forget some of "The"
📖 "sat" → remember, but forget even more...
📖 "on" → already forgot what came at the beginning!

Problem: By the end of a long book, you forget the beginning!
```

**TCN:**
```
Like looking at the ENTIRE PAGE at once:
📖 [The cat sat on the mat]
   ↓ ↓ ↓ ↓ ↓ ↓
   See EVERYTHING at the same time!

Advantage: Never forget anything important!
```

---

## How Does TCN Look at Data?

### Analogy: Building a Pyramid of Observations

Imagine you're building a pyramid of knowledge about the market:

```
Level 4 (sees 15+ days):     [📊 Long-term trend]
                               ↑       ↑
Level 3 (sees 7 days):    [📈 Weekly]  [📉 Weekly]
                            ↑   ↑       ↑   ↑
Level 2 (sees 3 days):  [📈][📉]     [📈][📉]
                          ↑ ↑ ↑ ↑     ↑ ↑ ↑ ↑
Level 1 (sees 1 day):  [D][D][D][D] [D][D][D][D]
                         ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
Raw data:             Mon Tue Wed Thu Fri Sat Sun Mon...
```

**How it works:**
1. **Level 1:** Looks at what happened today and yesterday
2. **Level 2:** Combines level 1 observations, sees 3-day patterns
3. **Level 3:** Combines level 2, sees weekly patterns
4. **Level 4:** Combines level 3, sees long-term trends

Each level "skips" some days (this is called **dilation**), which allows seeing more!

---

## Real Life Example: Predicting Bitcoin Price

### Step 1: Collecting Data

```
Date        | Price   | Volume    | What happened?
------------|---------|-----------|------------------
Dec 1       | $40,000 | Low       | Calm market
Dec 2       | $40,500 | Medium    | Slight growth
Dec 3       | $41,000 | High      | Buyers are coming!
Dec 4       | $40,800 | Medium    | Small pullback
Dec 5       | $41,500 | Very high | Breakout!
Dec 6       | ???     | ???       | What's next?
```

### Step 2: TCN Analysis

```
🔍 Small magnifying glass (today + yesterday):
   "Dec 5 we saw a breakout on high volume"
   Signal: Bullish!

🔍 Medium magnifying glass (last 3 days):
   "Overall upward trend with one pullback"
   Signal: Bullish with caution

🔍 Large magnifying glass (entire week):
   "Started at 40K, now 41.5K, +3.75%"
   Signal: Confirmed bullish trend
```

### Step 3: TCN Prediction

```
Combining all magnifying glasses:
- Short-term: ⬆️ bullish
- Medium-term: ⬆️ bullish
- Long-term: ⬆️ bullish

TCN Prediction: Price likely to RISE
Confidence: 75%
Suggestion: BUY
```

---

## Why is This Called "Causal" Convolution?

### Analogy: Time Machine Rule

Imagine you have a time machine, but with one rule: **you can NOT look into the future!**

```
❌ WRONG (looking into future):
   To predict Dec 5, I'll look at Dec 6 prices...
   This is cheating! In real life, we don't know the future!

✅ CORRECT (causal):
   To predict Dec 5, I'll only look at Dec 1, 2, 3, 4
   Only use information we ALREADY KNOW!
```

**TCN follows this rule:** When predicting the future, it only uses PAST data!

---

## What is "Dilation"?

### Analogy: Jumping Over Steps

Imagine climbing stairs:

```
Regular walk (dilation = 1):
Step 1 → Step 2 → Step 3 → Step 4
Very slow, see every step

Skip one (dilation = 2):
Step 1 → Step 3 → Step 5 → Step 7
Faster! Cover more ground!

Skip three (dilation = 4):
Step 1 → Step 5 → Step 9 → Step 13
Very fast! See the big picture!
```

**In TCN:**
- Dilation = 1: Look at every price bar
- Dilation = 2: Look at every other bar
- Dilation = 4: Look at every 4th bar
- And so on...

This way, TCN can see both:
- Tiny movements (dilation = 1)
- Huge trends (large dilation)

---

## How Does Trading with TCN Work?

### Step-by-Step Process

```
1. 📥 GET DATA
   Connect to exchange (Bybit)
   Download: prices, volumes, order book

2. 📊 PREPARE FEATURES
   Calculate indicators: RSI, MACD, moving averages
   Normalize everything to same scale

3. 🧠 TCN THINKS
   Looks at last 100-1000 price bars
   Analyzes patterns at all scales
   Makes a prediction

4. 📢 GENERATE SIGNAL
   If confidence > 70% UP → BUY
   If confidence > 70% DOWN → SELL
   Otherwise → WAIT

5. 💰 EXECUTE TRADE
   Calculate position size
   Set stop-loss (safety net!)
   Set take-profit (lock in gains!)

6. 🔄 REPEAT
   Go back to step 1
   Keep learning and improving!
```

---

## Simple Code Example

### Reading Cryptocurrency Data

```rust
// Super simple: getting Bitcoin price from Bybit
async fn get_bitcoin_price() -> f64 {
    let client = BybitClient::new();
    let ticker = client.get_ticker("BTCUSDT").await?;

    println!("Bitcoin price right now: ${}", ticker.last_price);

    ticker.last_price
}
```

### Making a Simple Prediction

```rust
// TCN looks at data and says what might happen
fn predict_next_move(prices: &[f64]) -> Prediction {
    // TCN analyzes the prices...
    let tcn = TCN::new();
    let prediction = tcn.forward(prices);

    // If prediction > 0.5, price likely to go UP
    // If prediction < 0.5, price likely to go DOWN

    if prediction > 0.6 {
        Prediction::Buy("TCN thinks price will rise!")
    } else if prediction < 0.4 {
        Prediction::Sell("TCN thinks price will fall!")
    } else {
        Prediction::Wait("TCN isn't sure, better to wait")
    }
}
```

---

## Fun Facts About TCN

### Why is TCN like a Swiss Army Knife?

```
🔧 Tool 1: Pattern Recognition
   Finds hidden patterns humans might miss

🔧 Tool 2: Multi-scale Analysis
   Sees small and big picture simultaneously

🔧 Tool 3: Fast Processing
   Analyzes thousands of prices in milliseconds

🔧 Tool 4: No Forgetting
   Remembers important events from long ago
```

### TCN vs Human Trader

| Aspect | Human | TCN |
|--------|-------|-----|
| Speed | Slow (seconds) | Fast (milliseconds) |
| Emotions | Gets scared, greedy | No emotions |
| Memory | Forgets things | Remembers everything |
| Patterns | Sees some | Sees many more |
| Tiredness | Gets tired | Works 24/7 |
| Mistakes | Makes emotional mistakes | Makes data-based mistakes |

---

## Key Terms Dictionary

| Term | Simple Meaning |
|------|---------------|
| **TCN** | A smart computer that looks at price history to guess the future |
| **Convolution** | Like a magnifying glass sliding over data |
| **Dilation** | Skipping steps to see bigger patterns |
| **Causal** | Only looking at the past, never the future |
| **Receptive Field** | How far back in time the network can "see" |
| **Residual Connection** | A shortcut that helps information flow better |

---

## Summary: What Did We Learn?

1. **TCN is like a detective** with multiple magnifying glasses
2. **It looks at patterns** at different time scales simultaneously
3. **It never cheats** by looking at future data
4. **It's fast** because it processes everything in parallel
5. **It never forgets** important patterns from the past
6. **For trading:** TCN helps predict if prices will go up or down

---

## What's Next?

If you're interested in trying TCN for trading:

1. **Learn Rust basics** - our code is written in Rust for speed
2. **Understand charts** - know what candlesticks and indicators mean
3. **Start small** - test with paper trading first (no real money!)
4. **Be patient** - ML takes time to learn

Remember: **No prediction is 100% accurate!** Always use risk management.

---

## Questions Kids Might Ask

**Q: Can TCN predict lottery numbers?**
A: No! Lottery is random. TCN works with patterns, and lottery has no patterns.

**Q: Will I become rich using TCN?**
A: Maybe, maybe not. Trading is risky. TCN helps, but nothing is guaranteed.

**Q: Is this like a video game?**
A: Kind of! But with real money. Games have clear rules, markets don't.

**Q: Can I run this on my computer?**
A: Yes! The Rust code in this chapter can run on any modern computer.

---

Good luck with your learning journey!
