# MoCo Trading - Explained Simply!

## The Detective and the Cold Case Files

Imagine a detective who is trying to identify a specific type of financial crime (a "pattern"). 

**SimCLR (the previous method)** is like a detective who only looks at the evidence they have on their desk *right now* (the current batch). If their desk is small, they might forget what other crimes look like.

**MoCo (Momentum Contrast)** is like a detective with a massive filing cabinet of "Cold Case Files" (the **Queue**). Every time a new case comes in, they compare it not just to the other cases on their desk, but to thousands of historical cases in the cabinet. 

## The "Consistent Partner" (Momentum Encoder)

In MoCo, we have two experts:
1. **The Lead Detective (Query Encoder)**: They are constantly learning and changing their mind based on new evidence.
2. **The Senior Partner (Momentum Encoder)**: They change their mind very slowly. They "take in" what the lead detective learns but keep a stable, long-term perspective. 

Because the Senior Partner is stable, the "files" in the cabinet stay organized and consistent, even as the Lead Detective gets smarter.

## Why this helps in the Market

1. **Long Memory**: The market has a "long memory." Patterns that happened weeks ago are still relevant. MoCo's queue allows the AI to remember those patterns without needing a super-computer.
2. **Harder to Fool**: By comparing a current price jump to a huge variety of historical jumps, the AI learns to distinguish a "genuine breakout" from a "fake-out" much more effectively.
3. **Smooth Learning**: Because the Senior Partner is slow to change, the AI doesn't "panic" and change its entire strategy just because the market was weird for a few minutes.

Check the `python/` folder to see how we build this "Detective with a Filing Cabinet"!
