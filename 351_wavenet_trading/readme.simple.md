# WaveNet: How a Computer Learns to Predict the Future

## What is WaveNet?

Imagine you have a magical friend who can listen to music and predict what note comes next. That's basically what **WaveNet** does!

WaveNet was created by Google's DeepMind to help computers generate human-like speech. But clever scientists discovered it can also predict prices in the stock market!

---

## The Magic of Looking Back

### The Detective Story

Imagine you're a detective trying to predict what your friend will have for lunch tomorrow. You could:

1. **Look at yesterday**: "She had pizza yesterday, maybe pizza again?"
2. **Look at the whole week**: "Hmm, she had pizza on Monday, salad on Tuesday, pizza on Wednesday..."
3. **Look at the whole month**: "I see a pattern! She loves pizza on Mondays and Fridays!"

The more you look back, the better you can predict!

**WaveNet is like a super-detective** that can look back at THOUSANDS of days all at once to find patterns!

---

## The Telephone Game with a Twist

### Normal Neural Networks (Like a Telephone Game)

Remember the telephone game? You whisper a message from person to person:

```
Person 1 → Person 2 → Person 3 → Person 4 → Person 5
"Apple"   "Apple"    "Ample"    "Temple"   "Simple"
```

By the end, the message gets distorted! This is the problem with normal neural networks - information gets "lost" as it travels through many steps.

### WaveNet (Like a Group Chat)

WaveNet is different! It's like having a group chat where everyone can read ALL previous messages at once:

```
Person 1: "Apple"
Person 2: (sees Person 1) "Apple pie?"
Person 3: (sees Person 1, 2) "Apple pie recipe!"
Person 4: (sees ALL) "Apple pie recipe with cinnamon!"
```

No information is lost because everyone can see everything!

---

## The Skip and Jump Pattern

### What are "Dilated" Convolutions?

Imagine you're reading a very long book and you want to understand the whole story quickly. You could:

1. **Read every word** (takes forever!)
2. **Read every other word** (faster, but might miss details)
3. **Smart skipping**: Read every word on page 1, every 2nd word on page 2, every 4th word on page 3...

This "smart skipping" is called **dilation**!

### Visual Example

```
Looking at today's weather to predict tomorrow:

Normal way:  [Today] [Yesterday] [2 days ago] [3 days ago]
              ●────────●──────────●────────────●

Dilated way: [Today] [Yesterday] [3 days ago] [7 days ago] [15 days ago]
              ●────────●───────────────●────────────────────●──────────────────●
```

With dilation, you can see MUCH farther back while doing less work!

---

## Real-Life Examples

### Example 1: The Ice Cream Shop

Imagine you own an ice cream shop and want to predict how many ice creams you'll sell tomorrow.

**Without WaveNet (looking at just yesterday)**:
- Yesterday you sold 50 ice creams
- Prediction: "Maybe 50 again?"

**With WaveNet (looking at patterns)**:
- Yesterday (Sunday): 50 ice creams
- Last Sunday: 55 ice creams
- Two weeks ago (Sunday): 48 ice creams
- Plus: It was hot last month, sales went up!
- Plus: School holiday starts tomorrow!

WaveNet sees ALL these patterns and predicts: "70 ice creams!"

### Example 2: The Music Predictor

You're listening to "Twinkle Twinkle Little Star":

```
Twinkle, twinkle, little star...
```

What comes next? If you've heard the song before, you know it's "how I wonder what you are!"

WaveNet works the same way - it learns patterns from history and predicts what comes next!

### Example 3: The Price Guesser

Bitcoin prices are like a roller coaster:

```
Monday:    $50,000  (went up)
Tuesday:   $51,000  (went up again)
Wednesday: $50,500  (went down a little)
Thursday:  $52,000  (up!)
Friday:    ???
```

WaveNet looks at the pattern and thinks:
- "Prices have been going up overall"
- "There was a small dip mid-week"
- "Similar pattern happened last month, and it went up on Friday!"

Prediction: "Probably around $52,500!"

---

## How WaveNet Sees Patterns

### The Expanding View

Think of it like looking through different binoculars:

| Layer | What You See | Real Life Example |
|-------|--------------|-------------------|
| Layer 1 | Yesterday | "What happened yesterday?" |
| Layer 2 | Last 2 days | "What happened this week?" |
| Layer 3 | Last 4 days | "What was the trend?" |
| Layer 4 | Last 8 days | "What about last week?" |
| Layer 5 | Last 16 days | "Any monthly patterns?" |
| Layer 6 | Last 32 days | "What about last month?" |

Each layer sees farther and farther back!

### The Magic Number

With just **10 layers**, WaveNet can see back **1024 time steps**!

That's like being able to remember:
- 1024 hours = **42 days** of hourly data
- 1024 minutes = **17 hours** of minute-by-minute data
- 1024 days = **almost 3 years** of daily data!

---

## The Gate Keeper

### What are Gated Activations?

Imagine you have a smart filter for your brain that decides:
1. **What to remember** (important stuff)
2. **What to forget** (not useful)

WaveNet uses two filters:
- **Filter 1 (tanh)**: "What is interesting?"
- **Filter 2 (sigmoid)**: "How important is it?"

Together, they work like this:

```
Information comes in
        ↓
Filter 1: "This price jump is interesting!"
Filter 2: "And it's VERY important!"
        ↓
Result: "REMEMBER THIS STRONGLY!"
```

Or:

```
Information comes in
        ↓
Filter 1: "This tiny change is boring..."
Filter 2: "And it's not important"
        ↓
Result: "Forget about it"
```

---

## Why This Matters for Trading

### The Advantage

| Old Way | WaveNet Way |
|---------|-------------|
| Looks at recent data | Sees far into the past |
| Slow to train | Fast to train |
| Forgets old patterns | Remembers everything |
| Needs many layers | Works with fewer layers |

### Real Trading Application

1. **Morning**: WaveNet looks at all Bitcoin prices for the past month
2. **Analysis**: Finds patterns like "prices usually dip at 3 PM on Fridays"
3. **Prediction**: "Today is Friday, price might dip at 3 PM"
4. **Action**: "Maybe wait until 3 PM to buy!"

---

## A Day in the Life of WaveNet

```
8:00 AM - WaveNet wakes up and reads all price history
8:01 AM - Finds 5 interesting patterns:
          - Weekly cycle (prices up on Mondays)
          - Monthly pattern (dip at end of month)
          - Similar price movement to January 2024
          - Volume increasing (people are buying!)
          - News sentiment is positive
8:02 AM - Makes prediction: "Price will go UP by 2%"
8:03 AM - Trading system: "BUY!"
```

---

## Fun Facts About WaveNet

1. **Made for Music**: WaveNet was originally made to generate human voice! Google uses it in Google Assistant.

2. **Super Memory**: Can "remember" thousands of data points at once.

3. **Fast Learner**: Trains much faster than older models because it can look at everything in parallel.

4. **Not Perfect**: Like any prediction, it can be wrong! Markets are unpredictable.

---

## Summary: WaveNet in 5 Points

1. **It's a Pattern Finder**: Like a detective looking for clues in price history

2. **It Skips Smartly**: Uses "dilated" convolutions to see far back efficiently

3. **It Remembers Everything**: Uses skip connections so no information is lost

4. **It Filters Smartly**: Uses "gates" to focus on important patterns

5. **It Predicts Fast**: Can make predictions quickly because it processes data in parallel

---

## What's Next?

Now that you understand WaveNet, you can:

1. **Look at the code**: Check out the `rust/` folder for working examples
2. **Run experiments**: Fetch real Bitcoin data and make predictions
3. **Learn more**: Read the main README.md for technical details

---

## Quick Quiz!

**Question 1**: Why is WaveNet better than looking at just yesterday's price?
<details>
<summary>Answer</summary>
Because it can see patterns from weeks or months ago, not just yesterday!
</details>

**Question 2**: What does "dilated" mean?
<details>
<summary>Answer</summary>
It means skipping some data points to see farther back without doing more work!
</details>

**Question 3**: Why is WaveNet faster than old neural networks?
<details>
<summary>Answer</summary>
Because it looks at all data at once (in parallel), instead of one step at a time!
</details>

---

*Remember: Even the smartest computer can't predict the future perfectly. WaveNet is a tool, not a crystal ball!*
