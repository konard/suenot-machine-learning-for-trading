# Quantum Annealing Trading - Simple Explanation

## What is it?

Imagine you're rolling a ball down a bumpy mountain to find the lowest valley. Sometimes the ball gets stuck in a small dip. Quantum annealing is like giving the ball the magical ability to tunnel through hills to find the deepest valley!

## How does it work?

Think about picking your favorite team of superheroes. You have 20 heroes to choose from, but you can only pick 5 for your team. Some heroes work great together, and some don't. You want to find the BEST team of 5.

If you tried every possible team, that would take a really long time -- there are thousands of combinations! Instead, quantum annealing is like a magic trick:

1. **Start with a random team** - Just pick any 5 heroes
2. **Shake things up** - At first, shake the team A LOT, swapping heroes in and out randomly
3. **Shake less and less** - Over time, shake more gently, so you keep the good heroes and only swap out the bad ones
4. **The magic part** - Normal shaking (called "simulated annealing") can only swap one hero at a time. But quantum shaking can magically swap MULTIPLE heroes at once by "tunneling" through bad combinations to find really good ones!

## How is this used in trading?

In trading, the "heroes" are things like stocks or crypto coins. You want to pick the best combination that:
- Makes the most money (high returns)
- Doesn't lose too much money on bad days (low risk)
- Only picks a certain number of coins (you can't buy everything!)

This is exactly like the hero-picking problem! So we can use quantum annealing to find great investment portfolios.

## The cool part

Regular computers try to find the best answer by looking around nearby -- like a ball rolling downhill. But sometimes the best answer is on the OTHER side of a hill. Quantum computers can "tunnel" through the hill to find it!

Even though we don't all have quantum computers yet, we can simulate the tunneling effect on regular computers. It's not as fast as a real quantum computer, but it's still pretty clever!

## What did we build?

We built a program in Rust that:
1. Gets real crypto prices from Bybit (a crypto exchange)
2. Figures out which coins go up and down together
3. Uses both regular and quantum-style solving to pick the best coins
4. Compares which method found a better team of coins!
