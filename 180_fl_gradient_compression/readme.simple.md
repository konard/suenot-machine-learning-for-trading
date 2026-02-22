# TGC: The "Decisive Trader" Analogy

Imagine you have a group of analysts monitoring the market. At the end of the day, you ask them to send a report on where the price should move.

### 1. Normal Report (Dense FL)
An analyst writes a long letter: "I think the price will rise by 0.1245%. Although, maybe by 0.1246%..." You receive thousands of such letters. This overloads your inbox (the network).

### 2. Ternary Report (Ternary FL)
You give your analysts a strict condition: "Don't write me long letters. Send only one of three signals":
- **+1**: "Buy" (Price will rise significantly).
- **-1**: "Sell" (Price will fall significantly).
- **0**: "Wait" (Changes are too small to pay attention to).

### Adaptive Threshold
But what counts as a "significant" change?
- If the market is in a storm, we lower the bar: even a small movement could be important.
- If the market is calm, we raise the bar: we only react to truly major news.

**The Result**: You only get the most important signals. Your inbox (internet channel) is 95% free, yet you know all the key sentiments of your analyst team. The model becomes resistant to minor market "noise."
