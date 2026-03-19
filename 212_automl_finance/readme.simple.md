# AutoML Finance - Explained Simply

Imagine a robot chef that not only cooks meals but also invents new recipes, picks the best ingredients, and figures out the perfect cooking temperature — all by itself! AutoML is like a robot that builds AI models without humans telling it exactly how.

## What is AutoML?

When people want to teach a computer to predict things (like whether a stock price will go up or down), they usually have to make a LOT of choices:

- **What clues to look at** (like how the price has been moving lately)
- **What type of brain to use** (simple math? a decision tree? a neural network?)
- **How to tune the brain** (how fast should it learn? how complicated should it be?)

That's a LOT of decisions! And each one matters. It's like trying to bake a cake but having to choose from 100 types of flour, 50 sweeteners, and 200 different oven temperatures. How do you find the best combination?

## The Robot Chef Approach

AutoML is like having a super-smart robot chef that:

1. **Tries lots of ingredients** — It automatically tests hundreds of different clues from the data
2. **Tests many recipes** — It tries simple recipes AND complicated ones to see what works best
3. **Adjusts the temperature** — It tweaks all the little settings to find the sweet spot
4. **Combines the winners** — It takes the best recipes and mixes them together for an even better result!

## How Does It Pick the Best?

Imagine you're trying to guess which flavor of ice cream your friend likes best. You could:

- **Random guessing**: Just try flavors randomly (this actually works okay!)
- **Smart guessing**: If they liked strawberry, maybe try raspberry next (this is called "Bayesian optimization")
- **Quick elimination**: Give them tiny tastes first, and only give full scoops of the ones they seem to like (this is called "Hyperband")

## Why Is It Tricky with Money?

Using AutoML for trading has some special challenges:

- **The rules keep changing**: Imagine if your friend's favorite ice cream flavor changed every week! Markets change like that too.
- **No peeking at the future**: You can't use tomorrow's information to make today's prediction. That would be cheating!
- **Don't be fooled by luck**: If you try a million recipes, some will seem great just by accident. You need to make sure a recipe is ACTUALLY good, not just lucky.

## The Cool Part

The best thing about AutoML is that it can discover combinations that humans would never think of! Just like a robot chef might invent a delicious recipe that no human cook ever imagined, AutoML can find trading strategies that human traders wouldn't have tried.

In this chapter, we build our own AutoML robot in Rust that can:
- Download real price data from Bybit (a cryptocurrency exchange)
- Automatically try different strategies
- Find the best one
- And combine the winners into a super-team!
