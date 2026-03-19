# InfoNCE Trading - Simple Explanation

## What is InfoNCE? A Matching Game!

Imagine you are playing a matching game with cards. Each card has a picture of the weather on it -- sunny, rainy, snowy, cloudy, and so on. Your job is to look at one card and figure out which other card shows the **same kind of weather**.

That is basically what InfoNCE does, but instead of weather cards, it uses **market cards** -- little snapshots of what the stock or crypto market looks like at a certain moment.

## How Does the Game Work?

Let's say you have three cards in front of you:

1. **Card A** (your question card): Shows a market going up with lots of trading happening
2. **Card B** (the matching card): Shows a market also going up with lots of trading -- it is from just a few minutes later
3. **Card C** (the trick card): Shows a market going down with very little trading -- it is from a completely different day

Your job is to figure out that **Card A matches Card B**, not Card C. They look similar because they came from similar market moments!

## The Scoring System

The game has a scoring system called **temperature**. Think of it like adjusting the difficulty:

- **Cold temperature** (hard mode): You have to find a nearly perfect match. Even small differences count!
- **Hot temperature** (easy mode): Close enough is good enough. The game is more forgiving.

## Why Is This Useful for Trading?

Once the computer gets really good at this matching game, something amazing happens: it learns to **understand** the market!

It is like how you learn to recognize weather patterns. After seeing hundreds of sunny days and rainy days, you don't just memorize each day -- you learn what "sunny" and "rainy" actually **mean**. You can recognize a sunny day you've never seen before.

The same thing happens here. The computer learns to recognize market patterns:
- "This looks like a calm, boring market"
- "This looks like a wild, crazy market"
- "This looks like the market is about to make a big move"

## The Secret Sauce: No Cheating Required!

Here is the coolest part: the computer **never needs to be told what will happen next**. It doesn't need labels like "price will go up" or "price will go down." It just learns by playing the matching game over and over.

This is like learning to recognize animals just by seeing which photos were taken at the same zoo exhibit, without anyone telling you "that's a lion" or "that's an elephant." Eventually, you'd figure out on your own that lions look different from elephants!

## A Real Example

Here is how it works with real crypto data:

1. **Get the cards**: Download Bitcoin price data from Bybit (a crypto exchange) -- hundreds of little snapshots showing the price, volume, and other details.

2. **Make pairs**: Take snapshots from similar moments (positive pairs) and mix in snapshots from random other moments (negative pairs).

3. **Play the game**: The computer tries to match the right pairs. When it gets it wrong, it adjusts its strategy.

4. **Learn**: After thousands of rounds, the computer becomes really good at understanding what makes market moments similar or different.

5. **Use the knowledge**: Now you can ask the computer: "Have you ever seen a market that looks like today?" and it can find the most similar historical moments!

## The Math (Super Simple Version)

The score for matching two cards is calculated like this:

```
Score = How similar are they? / Temperature
```

Then the computer tries to make the score **high** for matching cards and **low** for non-matching cards. That is the entire idea!

## Summary

- InfoNCE is like a matching game for market snapshots
- The computer learns what makes markets "similar" without being told the answers
- Temperature controls how picky the matching is
- After training, the computer understands market patterns and can find similar historical moments
- This knowledge can help traders understand what kind of market they are in right now
