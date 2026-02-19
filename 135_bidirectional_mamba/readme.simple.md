# Bidirectional Mamba - Explained Simply!

## Normal Mamba vs Bidirectional Mamba

Imagine you are trying to understand a movie plot by watching scene summaries on your phone.

**Normal Mamba (The Forward Watcher):**
You watch the movie starting from Scene 1 to Scene 100.
You do this using Mamba’s famous super-memory — efficiently crunching through the scenes without forgetting the overall plot. But what if Scene 10 was a subtle clue that only made sense after you saw the twist in Scene 99? 
Since Normal Mamba only goes Left-to-Right, it processes the clue early on and might drop it from its limited working memory by the time it reaches Scene 99.

**Bidirectional Mamba (The Detective Reviewing Evidence):**
Bidirectional Mamba doesn't just watch the movie; it watches the movie **forwards** AND **backwards** at the same time.
- **Pass 1 (Forward):** Scene 1 → Scene 100
- **Pass 2 (Backward):** Scene 100 → Scene 1 
By watching it backwards, the model sees the twist at Scene 99 first! By the time it reaches the clue in Scene 10, it instantly connects the two items together. The final "understanding" of the movie is a combination of both passes. 

---

## Wait, isn't looking backwards cheating in Trading?

Good question! In Trading, looking at the future is called **Data Leakage** and it ruins backtests. 

If we are at Day 100, we NEVER scan from Day 101. 
Instead, we define a "lookback window", like the past 50 days (Days 50 to 100). 
Since all days from 50 to 100 have already happened, we are 100% allowed to scan those days backward (from 100 to 50) in memory. This helps the AI fuse the current market state directly into the older data without memory dilution!

## Why is it so fast?

A standard Transformer model (like ChatGPT) uses $O(N^2)$ math. For 10,000 prices, that means 100,000,000 calculations.
Bidirectional Mamba does it twice, but uses $O(N)$ math. For 10,000 prices, it does $10,000 \text{ (forward)} + 10,000 \text{ (backward)} = 20,000$ calculations. 
It captures the entire bidirectional context globally but runs **massively faster** than Transformers for long time-series.

## Code Examples

We implemented Custom Bidirectional State-Space blocks in both Python and Rust:
- `python/model.py`: The forward and reverse scan mechanism in PyTorch.
- `python/train.py`: Using Bidirectional scans to predict market moves accurately.
- `rust/src/lib.rs`: Raw mathematically structured SSD algorithms in Rust.
