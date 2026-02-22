# PCL: The "Political Parties" Analogy

Imagine you are studying the views of millions of people (market data).

### 1. Simple Comparison (Instance-level Contrast)
You take two people and ask: "Do you agree with each other?". If it's the same person at different times, you try to make their answers similar. If they are different people, you just note that they are different. You see thousands of pairs, but you don't see the big picture of society.

### 2. PCL (Prototypical Contrast)
Instead of comparing everyone to everyone else, you define "Political Parties" (Prototypes). A prototype is an idealized representative of a group (e.g., "Liberal", "Conservative").

Now your task is:
1. Understand which "party" a given person is closest to (Clustering).
2. Make sure that the person shares the views of their party even more (Attraction to the prototype).
3. Make sure that the views of the "Liberal" and "Conservative" are as far apart as possible (Repulsion of prototypes).

In trading, prototypes are **reference market situations**. One prototype is the "Ideal level breakout," another is the "Fading trend." When the model sees a new situation, it doesn't just compare it to the previous minute, but relates it to all the known "ideals" it has learned from the entire dataset. This makes the model much more robust to random market fluctuations.
