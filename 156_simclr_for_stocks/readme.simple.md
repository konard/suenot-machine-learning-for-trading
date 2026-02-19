# SimCLR for Stocks - Explained Simply!

## Learning "Cousins"

Imagine you are looking at a photo of a golden retriever. Even if you make the photo black and white, crop it, or add some "grainy noise," you still know it's a golden retriever. Traditional AI needs you to write a label "Golden Retriever" on every photo. **SimCLR** doesn't.

SimCLR learns by taking one photo and creating two "edited" versions of it. It tells the AI: *"I don't know what this is, but I know these two edited photos are cousins (positive pair). Everything else in this pile of millions of photos are strangers (negative pairs)."*

Over time, the AI learns that "dog-like" things are cousins and "car-like" things are strangers.

## How we apply this to the Stock Market

In trading, we do the same with price patterns.

1. **The Original**: A specific 4-hour price movement.
2. **The Cousins (Augmentations)**: We take that movement and slightly change the scale, add some random "fuzz" (noise), or hide a small part of it. 
3. **The Task**: We tell the AI to learn that these "cousins" represent the same underlying market behavior, but they are different from a massive crash or a sideways chop.

## Why use this for Trading?

- **No more manual labeling**: You don't need a human to go through 10 years of charts and label "Double Bottoms." The AI finds the structures automatically.
- **Noise Resistance**: Because the AI was trained to recognize a pattern even when it was "fuzzy" or "stretched," it is much better at identifying patterns in the real, messy market.
- **Transfer Learning**: Once the AI understands the "language" of price movements, you can teach it to predict specific things (like "Will price go up?") very quickly with very little data.

Explore the `python/` folder to see how we create "price cousins" and train the AI to recognize them!
