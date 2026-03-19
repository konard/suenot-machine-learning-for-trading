# Hierarchical RL Trading - Explained Simply

## What is it?

Imagine a company with a boss (decides big goals) and workers (do daily tasks). The boss might say "We need to save money this month," and the workers figure out how to do that every day - maybe by turning off extra lights or finding cheaper supplies.

Hierarchical RL Trading works the same way! Instead of one robot trying to do everything, we have a team:

- **The Boss (Manager)**: Looks at the big picture. "The market is going up! Let's buy more!" or "Things look scary, let's be careful."
- **The Workers**: Handle the actual buying and selling. They figure out the best time to buy and the best price to get.

## Why is this better?

Think about playing a video game. If you had to think about every tiny movement AND the overall strategy at the same time, it would be really hard! But if one part of your brain handles "go to the castle" and another part handles "jump over this obstacle," it's much easier.

The same thing happens in trading:
- The boss thinks about **what** to do (big decisions, slow)
- The workers think about **how** to do it (small actions, fast)

## How does it work?

1. **The Boss looks at the weather** (market conditions) and decides: "It's sunny (bull market) - let's go outside (buy)!" or "It's stormy (bear market) - let's stay inside (sell)!"

2. **The Boss gives instructions** to the workers: "Try to buy some Bitcoin today"

3. **The Workers get a gold star** (reward) when they follow the boss's instructions well - even if the overall result isn't perfect yet

4. **Everyone learns together**: The boss learns which instructions work best, and the workers learn how to follow instructions better

## A Real Example

- Monday: The Boss sees Bitcoin prices going up steadily. Says "Buy mode!"
- Monday-Friday: The Worker buys small amounts throughout the week, picking good moments when the price dips a little
- Next Monday: The Boss sees prices starting to drop. Says "Careful mode!"
- The Worker stops buying and starts protecting the money

## Why do we use Rust?

Rust is like a super-fast race car that's also very safe. When trading, we need:
- **Speed**: Make decisions in milliseconds
- **Safety**: Never crash or make mistakes with money
- **Reliability**: Work 24/7 without problems

## The Cool Part

Just like in a real company, the boss and workers can get better at their jobs independently. A new worker can join and quickly learn to follow instructions, even if the boss's strategy changes. And the boss can try new strategies without worrying about the details of execution!
