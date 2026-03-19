# Chapter 267: Quote Update Prediction -- Explained Simply!

## What Is This About?

Imagine you are at a lemonade stand where the price changes every few seconds. Sometimes it goes up because lots of people want lemonade. Sometimes it drops because nobody is buying. **Quote update prediction** is like being able to guess whether the price is about to go up or down *before* it actually changes.

## The Lemonade Stand Analogy

Picture two lines at the stand:

- **The BUY line** (called the "bid"): people waiting to buy lemonade.
- **The SELL line** (called the "ask"): the stand owner who sets the selling price.

If the BUY line suddenly gets really, really long -- lots of thirsty people! -- you can bet the price is about to go up, because demand is high. If the BUY line shrinks and nobody is buying, the price will probably drop.

That is basically **order-book imbalance**: counting how many people are in each line and seeing which side is bigger.

## The Mid-Price: Finding the "Fair" Price

The buyer says: "I will pay $1.00."
The seller says: "I want $1.02."

The **mid-price** is right in the middle: $1.01. It is our best guess at the "true" price.

But what if there are 100 buyers and only 2 sellers? The real fair price is probably closer to $1.02, because all those buyers will push the price up. That adjusted guess is called the **micro-price**.

## How Do We Predict the Next Change?

We collect eight clues (called "features") from the lemonade stand:

1. **Who is in line?** -- Are there more buyers or sellers? (imbalance)
2. **How deep is the crowd?** -- Even people further back in line matter. (depth imbalance)
3. **How big is the gap?** -- The difference between the buy and sell price. (spread)
4. **Which way did it just move?** -- Did the price just go up or down? (recent return)
5. **What does the micro-price say?** -- Our smart fair-price guess. (micro-mid diff)
6. **Are buyers showing up or leaving?** -- Is the buy line growing? (bid size change)
7. **Are sellers showing up or leaving?** -- Is the sell line growing? (ask size change)
8. **How fast are things changing?** -- Time since the last update. (time delta)

## The World's Simplest Prediction Machine

We take those eight clues and multiply each one by a number (a "weight") that says how important it is. Then we add them all up:

```
Score = (weight1 x clue1) + (weight2 x clue2) + ... + (weight8 x clue8) + bias
```

- If the **score is positive** --> the price will probably go **UP**.
- If the **score is negative** --> the price will probably go **DOWN**.
- If the **score is near zero** --> it is a toss-up.

This is called a **linear model**, and it is super fast -- a computer can do it in about 80 *billionths* of a second!

## How Does the Machine Learn the Weights?

Think of it like tuning a guitar:

1. The machine looks at what happened in the past.
2. It makes a prediction.
3. If the prediction was **wrong**, it adjusts the weights a tiny bit.
4. It keeps going through examples, getting a little better each time.

This process is called **training**, and the specific method is called **stochastic gradient descent** (a fancy name for "learn from your mistakes, one example at a time").

## What About Bybit?

Bybit is a real cryptocurrency exchange where people trade Bitcoin, Ethereum, and other digital coins. Our program:

1. Asks Bybit: "Hey, what does the order book look like right now?"
2. Bybit sends back the list of buyers and sellers with their prices and amounts.
3. Our program computes the eight clues.
4. The linear model gives a score.
5. We decide: buy, sell, or wait.

It is like having a super-fast assistant who watches the lemonade stand and whispers in your ear: "Prices are about to go up -- buy now!"

## Why Rust?

Rust is a programming language that is:

- **Super fast** -- almost as fast as the computer can possibly go.
- **Safe** -- it catches mistakes before the program even runs.
- **Great for trading** -- when every nanosecond counts, Rust delivers.

## Fun Facts

- In real markets, quote updates happen **thousands of times per second**.
- Our model can make a prediction in about **80 nanoseconds** -- that is 80 billionths of a second. In that time, light only travels about 24 metres (less than a swimming pool length).
- The most important clue is usually the **imbalance** -- just knowing who is in line tells you a lot!

## Key Takeaways (The Short Version)

1. **Prices change in tiny jumps** called quote updates.
2. **Counting buyers vs sellers** (imbalance) is the best clue for what happens next.
3. **A simple formula** (linear model) is fast enough for real trading.
4. **Rust makes it blazing fast** -- fast enough to beat most competitors.
5. **Bybit gives us real data** so we can test our ideas on actual markets.

## Try It Yourself!

If you have Rust installed, you can run the trading example:

```bash
cd rust
cargo run --example trading_example
```

Watch as it fetches live Bitcoin order-book data from Bybit and predicts where the price is headed next!
