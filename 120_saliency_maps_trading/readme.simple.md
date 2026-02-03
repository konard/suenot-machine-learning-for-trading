# Saliency Maps: Simple Explanation for Beginners

Imagine you have a smart robot that predicts whether stock prices will go up or down. But here's the problem: the robot just says "up" or "down" without explaining WHY. Saliency maps are like X-ray glasses that let you see WHAT the robot is looking at when making its decision!

## What is a Saliency Map?

### Analogy with Highlighting

Imagine you're taking a test and you highlight the most important parts of a question before answering. A saliency map does the same thing for a computer program!

When a trading model looks at price data and makes a prediction, the saliency map highlights which parts of the data the model paid most attention to.

### Example with Weather Prediction

Let's say you want to predict if it will rain tomorrow. You look at:
- Temperature today
- Cloud coverage
- Wind direction
- Humidity
- Day of the week

You know that "day of the week" doesn't really matter for rain. But what if the computer thought Tuesday always means rain? That would be wrong!

A **saliency map** shows you what the computer thinks is important:
- Temperature: **HIGH importance** (correct!)
- Cloud coverage: **HIGH importance** (correct!)
- Day of the week: LOW importance (good, it learned correctly!)

## Why Do We Need Saliency Maps in Trading?

### The Problem with "Black Box" Models

Neural networks are often called "black boxes" because:
1. They take in data
2. Magic happens inside
3. Out comes a prediction

But in trading, we NEED to understand why because:
- **Money is at stake**: We don't want to bet money on wrong reasons
- **Markets change**: If the model focuses on wrong things, it will fail
- **Regulations**: Banks and funds must explain their trading decisions

### Saliency Maps = Transparency

With saliency maps, we can:
- Check if the model makes sense
- Find mistakes before losing money
- Build trust in automated trading

## How Saliency Maps Work

### The Basic Idea: Gradients

Think of it like this:

1. The model makes a prediction: "Price will go UP"
2. We ask: "If I change each input a tiny bit, how much does the prediction change?"
3. If changing one input changes the prediction a lot, that input is IMPORTANT

This "how much it changes" is called the **gradient**.

### Visual Example

Imagine the model looks at 5 days of stock data:

```
Day 1: Price = $100 (changed a lot when modified) --> IMPORTANT!
Day 2: Price = $102 (barely changed when modified) --> not important
Day 3: Price = $101 (barely changed when modified) --> not important
Day 4: Price = $105 (changed a lot when modified) --> IMPORTANT!
Day 5: Price = $103 (medium change when modified) --> somewhat important
```

The saliency map would show Days 1 and 4 as "bright" (important) and Days 2 and 3 as "dim" (not important).

## Types of Saliency Methods

### 1. Vanilla Gradient - The Simple One

Just look at how much the prediction changes when you wiggle each input.

**Analogy**: Like poking different keys on a piano to see which ones make the most sound.

### 2. Gradient x Input - The Scaled One

Multiply the gradient by the actual input value.

**Analogy**: Not just how sensitive a light switch is, but also whether it's actually turned on.

### 3. Integrated Gradients - The Path One

Instead of just looking at the final point, trace a path from "nothing" to the actual input.

**Analogy**: Instead of just measuring how far you jumped, measure every step of your running start.

### 4. SmoothGrad - The Average One

Add a little bit of random noise and average the results.

**Analogy**: Like taking multiple photos and averaging them to remove blur.

## Saliency Maps in Trading: A Simple Example

### Step 1: Train a Model

We teach a computer to predict if tomorrow's price will be higher or lower based on:
- Last 30 days of prices
- Trading volume
- Technical indicators (like moving averages)

### Step 2: Make a Prediction

The model says: "78% chance price goes UP tomorrow"

### Step 3: Compute the Saliency Map

We look at what the model focused on:

```
Last 30 days importance:
Day -30: [=====     ] 50%
Day -29: [===       ] 30%
Day -28: [==        ] 20%
...
Day -3:  [========  ] 80%  <-- Model thinks this day is important!
Day -2:  [=======   ] 70%
Day -1:  [==========] 100% <-- Yesterday is most important!
```

### Step 4: Check if it Makes Sense

If the model thinks Day -15 is super important but ignores yesterday, something might be wrong!

## Using Saliency Maps for Better Trading

### Strategy 1: Only Trade When It Makes Sense

```
IF model prediction is "BUY"
AND saliency shows focus on recent prices (not random old data)
AND saliency shows focus on volume (a good indicator)
THEN execute the trade
ELSE skip this signal
```

### Strategy 2: Adjust Position Size

```
IF saliency is concentrated on few clear factors
   --> Trade with larger size (more confident)
IF saliency is spread everywhere
   --> Trade with smaller size (less confident)
```

### Strategy 3: Detect Regime Changes

If the model suddenly starts focusing on different things than before, the market might be changing!

```
Last month: Model focused on volume and momentum
This week:  Model focuses on volatility and correlations
            --> Market regime might be changing!
```

## Key Takeaways

| Concept | Simple Explanation |
|---------|-------------------|
| **Saliency Map** | Shows what the model pays attention to |
| **Gradient** | How much prediction changes when input changes |
| **Vanilla Gradient** | Simple sensitivity check |
| **Gradient x Input** | Sensitivity times actual value |
| **Integrated Gradients** | Path from nothing to actual input |
| **SmoothGrad** | Average over noisy versions |

## Fun Analogy: The Detective Model

Imagine your trading model is a detective solving a case (predicting price direction).

- **Evidence (Input)**: Last month's prices, volumes, news
- **Verdict (Output)**: "Price will go UP"
- **Saliency Map**: Detective's notes showing which evidence was most convincing

Without saliency maps, the detective just says "guilty" without explaining why. With saliency maps, we can see:
- "The fingerprint on Day 5 was crucial" (recent price pattern)
- "The witness statement from Day 20 didn't matter" (old data)
- "The motive became clear on Day 28" (volume spike)

## What's Next?

Now that you understand the basics, you can:
1. Look at the Jupyter notebooks to see real examples
2. Try running the Python code on real market data
3. Build your own saliency-based trading strategy!

Good luck on your journey into explainable AI for trading!
