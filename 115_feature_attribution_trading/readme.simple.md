# Feature Attribution Trading - Simple Guide

## What is Feature Attribution?

Feature attribution is like asking your trading AI: "Hey, WHY did you decide to buy?" and getting a clear answer.

Think of it like this: your friend says "I think Bitcoin will go up." You ask "Why?" A good answer: "Because trading volume increased 50%, price broke above $50K, and big investors are buying."

Feature attribution does exactly this for AI models - it shows which pieces of information (features) were most important in making a decision.

```
Without Attribution:          With Attribution:
AI says: "BUY"                AI says: "BUY because..."
You ask: "Why?"               - Price momentum: 40%
AI says: "Just trust me"      - Volume spike: 35%
                              - RSI signal: 25%
```

## Why It's Important

| Without Attribution | With Attribution |
|---------------------|------------------|
| AI is a "black box" | AI decisions are transparent |
| Hard to trust the AI | You understand the reasoning |
| Can't fix bad decisions | Can identify and fix problems |
| Regulators are unhappy | Meets compliance requirements |

## Simple Example

Imagine an AI looking at Apple stock and deciding to BUY:

```
Input Features:           Contribution to BUY decision:
-----------------         ----------------------------
Price went up 2%    ---->  [========]     +0.35
Volume doubled      ---->  [======]       +0.28
RSI is at 65        ---->  [===]          +0.15
News sentiment      ---->  [==]           +0.12
Interest rates      ---->  [-]            -0.05
                          -----------------
                    TOTAL: +0.85 --> BUY signal!
```

## How It Works (Simple Version)

1. **Train your AI model** - The AI learns patterns from historical data
2. **Make a prediction** - AI says BUY, SELL, or HOLD
3. **Run attribution** - Ask "which features mattered most?"
4. **Get importance scores** - Each feature gets a score
5. **Visualize results** - See a clear breakdown of the decision

```
[Market Data] --> [AI Model] --> [Prediction: BUY]
                      |
                      v
              [Attribution Method]
                      |
                      v
              Price: 35% | Volume: 30% | RSI: 20% | News: 15%
```

## Trading Use Cases

| Use Case | How Attribution Helps |
|----------|----------------------|
| Risk Management | See which features drive risky trades |
| Strategy Improvement | Find which signals work best |
| Debugging | Figure out why a trade went wrong |
| Compliance | Explain decisions to regulators |
| Feature Selection | Remove features that don't help |

## Quick Start

### Python
```bash
cd 115_feature_attribution_trading/python
pip install -r requirements.txt
python model.py
```

### Rust
```bash
cd 115_feature_attribution_trading
cargo run --release
```

## Example Output

```
Loading BTC/USDT data...

Model Prediction: BUY (confidence: 78%)

Feature Attribution Analysis:
Feature              | Contribution | Direction
---------------------|--------------|----------
price_momentum_5d    |    0.32      |  BUY
volume_change        |    0.25      |  BUY
rsi_14               |    0.18      |  BUY
macd_signal          |    0.12      |  BUY
bollinger_position   |   -0.08      |  SELL

Top 3 reasons for BUY signal:
  1. Strong 5-day price momentum (+0.32)
  2. Volume increased significantly (+0.25)
  3. RSI indicates upward momentum (+0.18)
```

## Attribution Methods Comparison

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| SHAP | Slow | Very High | Deep analysis |
| LIME | Medium | High | Quick explanations |
| Permutation | Fast | Medium | Simple models |
| Integrated Gradients | Medium | High | Neural networks |

## Files in This Chapter

```
115_feature_attribution_trading/
├── README.md              # Full technical documentation
├── readme.simple.md       # This beginner guide
├── python/
│   ├── model.py           # Main attribution model
│   ├── train.py           # Training script
│   ├── backtest.py        # Test strategies
│   └── notebooks/
└── rust/
    └── src/lib.rs         # Fast Rust implementation
```

## Key Terms

- **Feature**: A piece of information the AI uses (price, volume, RSI)
- **Attribution**: How much each feature contributed to a decision
- **SHAP**: Popular method using game theory to explain AI
- **LIME**: Explains AI by testing similar examples nearby
- **Black Box**: An AI model you can't understand or explain

## When to Use Feature Attribution

**Use it when:**
- You need to explain trades to clients or regulators
- You want to understand why your strategy wins or loses
- You're debugging unexpected trading behavior
- You want to find and remove weak features

**Skip it when:**
- You're just prototyping quickly
- Speed is critical and explanations can wait

## Learn More

- Full documentation: [README.md](README.md)
- XAI Survey paper: [arxiv.org/abs/2407.15909](https://arxiv.org/abs/2407.15909)
- SHAP documentation: [github.com/slundberg/shap](https://github.com/slundberg/shap)
