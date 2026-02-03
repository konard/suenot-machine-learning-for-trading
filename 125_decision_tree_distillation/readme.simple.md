# Decision Tree Distillation — Explained Simply!

## What is Decision Tree Distillation?

Imagine you have a genius trading expert who makes great decisions, but can't explain WHY they make them. Decision Tree Distillation is like hiring a translator who watches the expert and writes down simple rules that capture their wisdom.

```
The Expert (Complex Model):
  "I just... know when to buy. It's intuition."
  (Actually using 500 rules nobody can understand)

The Translator (Distilled Tree):
  "Ah! The expert buys when:
   - RSI is below 30, AND
   - MACD is turning up, AND
   - Volume is higher than normal"
```

Now you have clear rules you can understand, verify, and trust!

---

## Why Should Traders Care?

### Problem: Black Box Models

Modern AI models (neural networks, ensembles) often make better predictions than simple rules. But they're like a mysterious oracle:

```
YOU: "Why should I buy Apple stock?"

BLACK BOX MODEL: "Buy. Confidence: 73%"

YOU: "But WHY?"

BLACK BOX MODEL: "..."
```

### Solution: Distillation

Distillation extracts readable rules from the black box:

```
YOU: "Why should I buy Apple stock?"

DISTILLED TREE:
  "Because:
   ✓ RSI is 28 (oversold)
   ✓ MACD crossed above signal line
   ✓ Volume is 1.5x average

   This pattern historically leads to price increases."

YOU: "Now I understand! And I agree with this logic."
```

---

## The Core Idea: Teacher and Student

Think of it like school:

```
STEP 1: The Teacher Learns
─────────────────────────────────────────
A complex AI model studies millions of trades.
It becomes really good at predicting, but it's complicated.

    Complex Model (Teacher)
    ├── 500 decision trees
    ├── Each tree has 15 levels
    └── Total: Thousands of rules


STEP 2: The Student Watches
─────────────────────────────────────────
A simple decision tree watches the teacher make predictions.
It doesn't learn from the market - it learns from the TEACHER.

    Teacher predicts: "BUY" (73% confident)
    Student notes: "When RSI=28, MACD=positive, Volume=high → BUY (73%)"


STEP 3: The Student Summarizes
─────────────────────────────────────────
After watching thousands of predictions, the student creates
a simple summary of the teacher's wisdom.

    Simple Tree (Student)
    └── Only 5 levels
    └── ~30 rules total
    └── Captures 85-95% of teacher's behavior
```

---

## Real Trading Example

### The Setup

```
You have a powerful AI model that predicts crypto prices.
It works great, but your compliance team asks:
"How does it decide when to buy Bitcoin?"

You have no idea. The model uses:
- 50 input features
- 3 neural network layers
- 128 neurons each
- Millions of parameters

Nobody can explain it!
```

### The Distillation Process

```
Step 1: Let the complex model make 10,000 predictions
────────────────────────────────────────────────────
Input Features         → Complex Model → Prediction
RSI=25, MACD=0.5, ...  →    [????]    → BUY (78%)
RSI=72, MACD=-0.3, ... →    [????]    → SELL (65%)
RSI=45, MACD=0.1, ...  →    [????]    → HOLD (52%)
...


Step 2: Train a simple decision tree on these predictions
───────────────────────────────────────────────────────────
The tree learns: "When the complex model sees X, it predicts Y"


Step 3: Extract the rules
─────────────────────────
IF RSI < 30:
    IF MACD > 0:
        → BUY (confidence: 0.75)
    ELSE:
        → HOLD (confidence: 0.55)
ELIF RSI > 70:
    IF Volume_spike:
        → SELL (confidence: 0.72)
    ELSE:
        → HOLD (confidence: 0.58)
ELSE:
    → HOLD (confidence: 0.51)
```

### The Result

Now you can tell your compliance team:

```
"Our AI primarily uses RSI to identify overbought/oversold conditions,
confirmed by MACD momentum. Volume spikes trigger stronger sell signals.
Here are the exact thresholds: RSI < 30 for buy, RSI > 70 for sell."
```

---

## The Magic of Soft Labels

Here's a crucial detail that makes distillation powerful:

### Hard Labels (Traditional)
```
Price went up   → Label: "BUY" (or 1)
Price went down → Label: "SELL" (or 0)

Problem: A 51% up move and 99% up move both become "BUY"
```

### Soft Labels (Distillation)
```
Teacher predicts 51% chance of up → Label: 0.51
Teacher predicts 99% chance of up → Label: 0.99

Benefit: Student learns CONFIDENCE, not just direction!
```

Why this matters for trading:

```
Hard Labels:
  Student thinks: "Both cases are BUY, trade the same way"

Soft Labels:
  Student thinks: "0.51 = barely confident, small position
                   0.99 = very confident, full position"
```

---

## Visual Guide to Distillation

```
BEFORE DISTILLATION:

    Your Model
    ┌─────────────────────────────────────────────────┐
    │  Neural Network + Random Forest + XGBoost       │
    │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
    │  │Tree1│  │Tree2│  │Tree3│  │ ... │  │T500 │   │
    │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
    │     └────────┴────────┴────────┴────────┘       │
    │                      ↓                          │
    │              [Magic Happens]                    │
    │                      ↓                          │
    │               Prediction: BUY                   │
    └─────────────────────────────────────────────────┘

    Why? No idea. 🤷


AFTER DISTILLATION:

    Your Simple Tree
    ┌─────────────────────────────────────────────────┐
    │                  RSI < 30?                      │
    │                 /        \                      │
    │               Yes         No                    │
    │               /             \                   │
    │         MACD > 0?        RSI > 70?              │
    │         /      \         /      \               │
    │       Yes      No      Yes      No              │
    │        ↓        ↓        ↓        ↓             │
    │      BUY     HOLD     SELL     HOLD             │
    │     (75%)   (55%)    (68%)    (51%)             │
    └─────────────────────────────────────────────────┘

    Why BUY? RSI < 30 AND MACD > 0. Clear! ✓
```

---

## The Trade-Off: Accuracy vs Clarity

You can't have it all. Deeper trees = more accurate but less interpretable:

```
Tree Depth 2:  ████░░░░░░ Accuracy: 60%  | Clarity: ⭐⭐⭐⭐⭐
Tree Depth 3:  █████░░░░░ Accuracy: 72%  | Clarity: ⭐⭐⭐⭐
Tree Depth 5:  ███████░░░ Accuracy: 85%  | Clarity: ⭐⭐⭐
Tree Depth 7:  █████████░ Accuracy: 92%  | Clarity: ⭐⭐
Tree Depth 10: ██████████ Accuracy: 96%  | Clarity: ⭐

Sweet spot for trading: Depth 4-6
- Accurate enough to be useful
- Simple enough to understand and trust
```

---

## Quick Code Example

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Step 1: Train complex teacher model
teacher = GradientBoostingClassifier(n_estimators=200, max_depth=10)
teacher.fit(X_train, y_train)

# Step 2: Get soft predictions from teacher
soft_labels = teacher.predict_proba(X_train)

# Step 3: Train simple student tree on soft labels
student = DecisionTreeClassifier(max_depth=5)
student.fit(X_train, soft_labels[:, 1])  # Use probability of positive class

# Step 4: Compare performance
teacher_accuracy = teacher.score(X_test, y_test)
student_accuracy = student.score(X_test, y_test)
fidelity = (teacher.predict(X_test) == student.predict(X_test)).mean()

print(f"Teacher accuracy: {teacher_accuracy:.1%}")
print(f"Student accuracy: {student_accuracy:.1%}")
print(f"Student mimics teacher: {fidelity:.1%} of the time")
```

---

## Common Mistakes to Avoid

### Mistake 1: Too Deep Tree
```
BAD:  Distill to depth-15 tree
      → Just as confusing as original model!

GOOD: Distill to depth-5 tree
      → Simple enough to understand and verify
```

### Mistake 2: Ignoring Fidelity
```
BAD:  "My distilled tree is 90% accurate!"
      (But it only agrees with teacher 60% of the time)

GOOD: Check BOTH accuracy AND fidelity
      "90% accurate AND agrees with teacher 85% of the time"
```

### Mistake 3: Using Hard Labels
```
BAD:  Train student on original 0/1 labels
      → Loses the teacher's confidence information

GOOD: Train student on teacher's probability predictions
      → Captures uncertainty and confidence
```

### Mistake 4: One Tree for All Markets
```
BAD:  Single distilled tree for trending AND ranging markets

GOOD: Distill separate trees for each market regime
      → Trending regime: momentum rules
      → Ranging regime: mean-reversion rules
```

---

## Key Takeaways

1. **What it is**: Converting a complex "black box" model into simple, readable rules

2. **Why it matters**: Compliance, trust, debugging, and understanding your trading system

3. **How it works**: Train a simple decision tree to mimic the complex model's predictions

4. **The trade-off**: Simpler trees are more understandable but less accurate

5. **Soft labels**: Use probability predictions, not just 0/1 labels

6. **Fidelity**: Measure how well the simple tree mimics the complex model

---

## The Bottom Line

> Decision Tree Distillation turns your mysterious AI trading model into a set of clear rules that humans can read, understand, verify, and trust.

---

## Try It Yourself

```bash
cd 125_decision_tree_distillation/python
pip install -r requirements.txt
python -c "
from model import DistillationModel
from data_loader import load_stock_data

# Load Apple stock data
data = load_stock_data('AAPL', period='1y')
print('Loaded', len(data), 'data points')

# The model.py file has a complete example!
print('See model.py for the full distillation example')
"
```
