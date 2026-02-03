# Chapter 119: Rule Extraction Trading — Simple Explanation

## What Is This?

Imagine you have a very smart friend who can predict whether stock prices will go up or down. But when you ask "Why do you think so?", they just say "I don't know, I just feel it." That's frustrating, right?

**Rule extraction** is like teaching that smart friend to explain their thinking using simple "if-then" rules:

- "IF the price dropped 3 days in a row AND volume is increasing, THEN it will probably go up"
- "IF RSI is above 70 AND price hit resistance, THEN it will probably go down"

This way, you can understand, verify, and trust their predictions.

## A Real-Life Analogy

### The Expert Chef Problem

Imagine a master chef who makes amazing dishes. You want to learn their secrets, but they just say "I add ingredients until it feels right."

**Rule extraction** is like watching the chef carefully and writing down rules:

```
IF sauce is too thick THEN add water
IF pasta is still hard THEN cook 2 more minutes
IF garlic is browning THEN reduce heat immediately
```

Now anyone can follow these rules to cook well, even without the chef's "intuition."

### In Trading Terms

A neural network is like the intuitive chef. It makes good predictions but can't explain why.

Rule extraction turns it into a cookbook:

```
IF moving_average_20 > moving_average_50 THEN market is bullish
IF volatility > 2% AND volume_spike THEN expect big move
IF RSI < 30 AND price_near_support THEN BUY signal
```

## Why Do We Need This?

### Problem: The Black Box

Modern AI models (neural networks, deep learning) are like black boxes:

```
[Input: Price data] → [??? Magic ???] → [Output: Buy/Sell]
```

We can't see what's happening inside. This creates problems:

1. **Trust**: Would you bet money on something you don't understand?
2. **Debugging**: If it's wrong, how do you fix it?
3. **Regulators**: Banks must explain their trading decisions
4. **Learning**: You can't improve what you don't understand

### Solution: Extract Rules

Rule extraction opens the black box:

```
[Input: Price data] → [Clear Rules] → [Output: Buy/Sell]
                           ↓
                    "IF price > SMA_20 AND RSI < 70
                     THEN BUY with 65% confidence"
```

Now you can:
- Understand why decisions are made
- Find and fix problems
- Explain to regulators
- Learn what patterns matter

## How Does It Work?

### Method 1: Watch and Learn (Pedagogical)

We treat the AI model as a teacher and learn from its examples:

```
Step 1: Show the AI 1000 market situations
Step 2: Record what the AI predicts for each
Step 3: Find patterns in when it says "BUY" vs "SELL"
Step 4: Write these patterns as simple rules

Example:
- AI said "BUY" when price was above SMA in 80% of cases
- AI said "SELL" when RSI was above 80 in 75% of cases
- Therefore: Rule 1: IF price > SMA THEN lean BUY
             Rule 2: IF RSI > 80 THEN lean SELL
```

### Method 2: Look Inside (Decompositional)

We look at the AI's internal structure (weights and connections):

```
Neural Network Internal:
   Input: [RSI, Volume, Price Change]
              ↓
   Neuron 1: 0.8×RSI + 0.3×Volume - 0.5×Change > 0.5?
              ↓
   Simplified Rule: IF RSI is high AND Volume is medium THEN ...
```

This is more precise but requires understanding the AI's architecture.

### Method 3: Combine Both (Eclectic)

Use internal structure as a guide, then verify with examples.

## Simple Example

### Starting Point: A Trained Model

Let's say we have a neural network that predicts Bitcoin price direction:

```
Input: [RSI, MACD, Volume Change, Volatility, SMA Ratio]
Output: UP (1) or DOWN (0)
Accuracy: 58% (pretty good for markets!)
```

But it's a black box. We want rules.

### Step 1: Generate Examples

```python
# Feed many scenarios to the model
scenarios = generate_market_scenarios(1000)
predictions = model.predict(scenarios)

# Results look like:
# Scenario 1: RSI=45, MACD=0.5, ... → Model says: UP
# Scenario 2: RSI=75, MACD=-0.2, ... → Model says: DOWN
# ...
```

### Step 2: Find Patterns

```python
# Use a decision tree to find patterns
tree = DecisionTree(max_depth=5)
tree.fit(scenarios, predictions)

# The tree discovers:
# - When RSI > 65, model usually says DOWN
# - When MACD > 0.3 AND RSI < 50, model usually says UP
# - When Volatility > 0.03, model is uncertain
```

### Step 3: Extract Rules

```
Rule 1: IF RSI > 65 THEN SELL (confidence: 72%)
Rule 2: IF MACD > 0.3 AND RSI < 50 THEN BUY (confidence: 68%)
Rule 3: IF Volatility > 0.03 THEN HOLD (uncertain)
Rule 4: IF SMA_Ratio > 1.02 AND Volume_Change > 0 THEN BUY (confidence: 61%)
```

### Step 4: Trade Using Rules

```
Current market: RSI=42, MACD=0.45, Vol=0.02, SMA_Ratio=1.03, Vol_Change=0.1

Check rules:
- Rule 1: RSI=42 not > 65 → doesn't apply
- Rule 2: MACD=0.45 > 0.3 AND RSI=42 < 50 → APPLIES! → BUY
- Rule 3: Vol=0.02 not > 0.03 → doesn't apply
- Rule 4: SMA_Ratio=1.03 > 1.02 AND Vol_Change=0.1 > 0 → APPLIES! → BUY

Decision: BUY (2 rules agree)
Explanation: "Buying because MACD is positive with low RSI,
              and price is above moving average with increasing volume"
```

## Key Concepts

| Term | Simple Meaning |
|------|----------------|
| **Black Box** | A model where you can't see how it makes decisions |
| **Rule Extraction** | Getting simple if-then rules from a complex model |
| **Fidelity** | How closely the rules match what the original model does |
| **Coverage** | What percentage of situations the rules can handle |
| **Pedagogical** | Learning rules by watching the model's inputs and outputs |
| **Decompositional** | Learning rules by looking inside the model |

## Measuring Rule Quality

### Fidelity: Do rules match the model?

```
Model predictions:   [BUY, SELL, BUY, BUY, SELL, BUY]
Rule predictions:    [BUY, SELL, BUY, BUY, BUY,  BUY]
                                          ↑
                                    Mismatch!

Fidelity = 5/6 = 83% (rules match model 83% of the time)
```

Higher fidelity means rules accurately represent what the model learned.

### Coverage: Do rules handle all cases?

```
Total scenarios: 100
Scenarios where at least one rule applies: 92

Coverage = 92% (rules can handle 92% of situations)
```

Higher coverage means fewer "I don't know" situations.

### Complexity: Are rules simple enough?

```
Simple rule: IF RSI > 70 THEN SELL
Complex rule: IF RSI > 68.5 AND MACD < -0.234 AND Volume > 1.5×Avg
              AND Volatility BETWEEN 0.015 AND 0.028 THEN SELL
```

Simpler rules are easier to understand and less likely to overfit.

## Trading Strategy with Rules

### Signal Generation

```
For each new market data point:
    1. Calculate features (RSI, MACD, etc.)
    2. Check each rule
    3. Count how many rules say BUY vs SELL
    4. Generate signal based on majority

Example:
    Features: RSI=35, MACD=0.2, Volume_spike=True

    Rule 1 (BUY): RSI < 40 → Matches! (+1 BUY)
    Rule 2 (SELL): MACD < 0 → Doesn't match
    Rule 3 (BUY): Volume_spike = True → Matches! (+1 BUY)

    Score: BUY=2, SELL=0 → Signal: BUY
```

### Why This Is Powerful

1. **Explainability**: "I bought because RSI was oversold and volume spiked"
2. **Adjustability**: Don't trust Rule 3? Just remove it
3. **Debugging**: Strategy losing money? Check which rules are failing
4. **Regulatory compliance**: Can show exactly why each trade was made

## What the Code Does

### Python Code
- `rule_extractor.py`: Extracts rules from neural networks and gradient boosting
- `bybit_data.py`: Fetches cryptocurrency data from Bybit exchange
- `stock_data.py`: Gets stock market data from Yahoo Finance
- `backtest.py`: Tests how well the rules would have traded historically
- `strategy.py`: Implements the trading strategy using extracted rules

### Rust Code
- Same functionality as Python, but runs much faster
- Used in production when speed matters (real-time trading)
- Can process millions of data points per second

## Comparison

| Approach | Explainability | Accuracy | Speed | Best For |
|----------|---------------|----------|-------|----------|
| Neural Network (black box) | None | High | Medium | When accuracy is all that matters |
| Decision Tree (simple) | Excellent | Medium | Fast | Quick, interpretable models |
| **Rule Extraction** | **Excellent** | **High** | **Fast** | **Best of both worlds** |

## Summary

Rule extraction is like having a translator who can explain what a genius is thinking in simple terms:

1. **Train a powerful but opaque model** (neural network)
2. **Extract simple rules** that approximate its behavior
3. **Trade using the rules** with full explainability
4. **Debug and improve** by examining which rules work

The result: You get the prediction power of complex AI with the transparency of simple rules.

## Glossary

| Term | What It Means |
|------|---------------|
| **Neural Network** | A complex AI that learns patterns but can't explain them |
| **Decision Tree** | A simple model that makes decisions using if-then rules |
| **TREPAN** | An algorithm that creates a decision tree mimicking a neural network |
| **Fidelity** | How accurately rules copy the original model's behavior |
| **RSI** | Relative Strength Index - measures if something is overbought/oversold |
| **MACD** | Moving Average Convergence Divergence - shows trend direction |
| **Backtest** | Testing a strategy on historical data to see if it would have worked |
| **Bybit** | A cryptocurrency exchange where we get price data |
