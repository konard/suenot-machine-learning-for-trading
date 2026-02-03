# Attention Rollout Trading - Simple Guide

## What is Attention Rollout?

Imagine you're a detective trying to figure out why a trading AI made a specific decision. Attention Rollout is like a "thought tracker" that shows you exactly what the AI was "looking at" when it decided to buy or sell.

Think of it like this: when you decide to buy a stock, you might think about yesterday's price, last week's trend, and some news you read. Attention Rollout shows you the AI's version of this thought process.

```
Your Brain When Trading:
"Hmm, the price dropped yesterday... but it's been going up all week..."
     [Yesterday: 30%] [This Week: 50%] [News: 20%]
                           ↓
                    Decision: BUY

AI's Brain (shown by Attention Rollout):
"Processing historical data..."
     [Day -5: 5%] [Day -4: 8%] [Day -3: 12%] [Day -2: 25%] [Day -1: 50%]
                                      ↓
                               Signal: BUY
```

## Why Should Traders Care?

### The Black Box Problem

Most AI trading systems are "black boxes" - they give you a prediction, but you have no idea why:

```
Traditional AI:
┌─────────────────┐
│    INPUT        │
│  Price Data     │──────────?????????────────▶  BUY!
│  Volume, etc.   │
└─────────────────┘
      What happens inside? No one knows!
```

### Attention Rollout Opens the Box

```
With Attention Rollout:
┌─────────────────┐
│    INPUT        │    "I'm looking mostly at
│  Price Data     │──▶  the last 2 days because  ──▶  BUY!
│  Volume, etc.   │     there's a clear uptrend"
└─────────────────┘
      Now you can see the AI's reasoning!
```

## A Restaurant Analogy

Imagine you ask a friend to pick a restaurant:

**Without Attention Rollout:**
Friend: "Let's go to Mario's Italian!"
You: "Why?"
Friend: "I don't know, it just feels right."

**With Attention Rollout:**
Friend: "Let's go to Mario's Italian!"
You: "Why?"
Friend: "Well, I thought about the rating (40% of my decision), the distance (30%), and that you mentioned liking pasta last week (30%)."

Now you understand their reasoning and can trust the decision more!

## The Core Idea

Transformer models (the AI behind modern trading systems) work by "paying attention" to different parts of the input data. But they have multiple layers, and each layer looks at things differently.

```
Layer 1: "Looking at everything equally"
   [Day-5] [Day-4] [Day-3] [Day-2] [Day-1]
     20%    20%     20%     20%     20%

Layer 2: "Starting to focus on recent days"
   [Day-5] [Day-4] [Day-3] [Day-2] [Day-1]
     10%    15%     20%     25%     30%

Layer 3: "Really focusing on yesterday"
   [Day-5] [Day-4] [Day-3] [Day-2] [Day-1]
      5%    10%     15%     25%     45%

Attention Rollout combines all layers to show:
"Overall, yesterday mattered most for this prediction"
```

## A Simple Trading Example

Let's say you have an AI predicting Bitcoin prices:

### Input Data (Last 5 Days)
```
Day -5: $40,000 (small drop)
Day -4: $39,500 (continued drop)
Day -3: $38,000 (bigger drop)
Day -2: $39,000 (recovery started)
Day -1: $41,000 (strong recovery)
```

### AI Prediction: BUY

### Attention Rollout Shows:
```
What the AI focused on:

Day -5: ██               8%
Day -4: ███             12%
Day -3: ████            15%
Day -2: ██████████      35%  ← Recovery start!
Day -1: ████████        30%  ← Strong recovery!

The AI learned that recovery patterns matter most!
```

## Three Types of Attention Patterns

### 1. Momentum Pattern
The AI focuses on recent data - it's following the trend.

```
[Old] [...]  [...] [Recent] [Latest]
  5%   10%   15%    30%      40%

Interpretation: "The current trend will continue"
Good for: Trending markets
```

### 2. Mean Reversion Pattern
The AI looks at historical averages - it expects prices to return to normal.

```
[Old] [...]  [...] [Recent] [Latest]
 25%   20%   20%    20%      15%

Interpretation: "Current price is abnormal, will revert"
Good for: Ranging markets
```

### 3. Event-Focused Pattern
The AI zeroes in on specific important moments.

```
[Old] [EVENT] [...] [Recent] [Latest]
  5%   60%    10%    15%      10%

Interpretation: "That specific event is key"
Good for: News-driven trading
```

## How Attention Rollout Helps Your Trading

### 1. Validate Model Decisions

Before trusting a BUY signal, check what the model focused on:

```
Good Sign:
- Model focused on relevant technical patterns
- Attention spread reasonably across important periods
- Recent price action has significant weight

Red Flag:
- Model focused on irrelevant old data
- Attention concentrated on a single random point
- Recent important events ignored
```

### 2. Understand Market Regimes

```
Bull Market Attention:    Bear Market Attention:
[....] [....] [RECENT]    [HISTORY] [....] [....]
         ↓                     ↓
  Momentum trading        Cautious, looking back
```

### 3. Debug Poor Performance

When your model loses money, attention rollout shows why:

```
Losing Trade Analysis:
"The model was looking at the wrong things!"

Expected: Focus on yesterday's breakout
Actual:   Focus on data from 2 weeks ago

Fix: Retrain with emphasis on recent patterns
```

## Common Mistakes to Avoid

### Mistake 1: Ignoring Attention Entropy

Entropy measures how "spread out" the attention is.

```
Low Entropy (Concentrated):    High Entropy (Spread):
[█] [█] [█] [█] [████████]    [██] [██] [██] [██] [██]

Concentrated might mean:       Spread might mean:
- Clear signal                 - Uncertain
- Or overfitting!              - Or comprehensive analysis
```

### Mistake 2: Not Comparing Winning vs Losing Trades

Always analyze both!

```
Winning Trades Attention:      Losing Trades Attention:
Focus on recent momentum       Focus on old, irrelevant data

This insight helps improve your model!
```

### Mistake 3: Forgetting About Multiple Heads

Transformer models have multiple "attention heads" - like having multiple experts. Some heads might be wrong!

```
Head 1: Looking at volume       → Useful
Head 2: Looking at old prices   → Not useful
Head 3: Looking at momentum     → Useful
Head 4: Looking at noise        → Harmful

Attention Rollout averages these, but you can also
analyze each head separately for deeper insights.
```

### Mistake 4: Trusting Without Verification

Attention shows correlation, not causation!

```
Model focuses on Monday data → Predicts Tuesday price
                   ↓
Does Monday actually predict Tuesday?
Or is the model learning spurious patterns?

Always backtest to verify!
```

## Quick Python Example

```python
# Simple example of using attention rollout
import torch
from attention_rollout import AttentionRollout
from model import TradingTransformer

# Create model and rollout analyzer
model = TradingTransformer(input_dim=5, d_model=64, n_heads=4, n_layers=2)
rollout = AttentionRollout(model)

# Your price data (20 days of: open, high, low, close, volume)
price_data = torch.randn(1, 20, 5)  # In practice, use real data!

# Get prediction and see what the model focused on
prediction, _ = model(price_data)
attention_scores = rollout.get_input_attribution(price_data)

# Print which days mattered most
print("Day importance for prediction:")
for i, score in enumerate(attention_scores):
    bar = "█" * int(score * 50)
    print(f"Day {-20+i}: {bar} {score:.2%}")
```

Output might look like:
```
Day importance for prediction:
Day -20: █ 2%
Day -19: █ 3%
...
Day -2: █████████████ 26%
Day -1: ██████████████████ 35%
```

## Key Takeaways

1. **Attention Rollout** shows you what your trading AI focuses on
2. **Interpretability** helps you trust (or distrust) model predictions
3. **Different patterns** indicate different market strategies
4. **Comparing trades** reveals what works and what doesn't
5. **Always verify** - attention shows correlation, not causation

## The Bottom Line

Attention Rollout turns your black-box trading AI into a transparent partner. Instead of blindly following signals, you can understand why the model makes each prediction and make better trading decisions.

## Try It Yourself!

```bash
# Install dependencies
cd 124_attention_rollout_trading/python
pip install -r requirements.txt

# Run the example
python -c "
from model import TradingTransformer
from attention_rollout import AttentionRollout
import torch

model = TradingTransformer(input_dim=5, d_model=64, n_heads=4, n_layers=2)
rollout = AttentionRollout(model, attention_layer_name='attn')

x = torch.randn(1, 20, 5)
scores = rollout.get_input_attribution(x)
print('Attention scores:', scores.round(3))
"
```

Now you can see inside your trading AI's mind!
