# Asynchronous FL: The "Collective Puzzle" Analogy

Imagine 10 people are putting together one giant jigsaw puzzle, but they are in different cities and send pieces by mail.

### 1. How it works in Synchronous mode (Sync FL)
You cannot glue a single new piece until you receive mail from **all** 10 participants. If one person goes on vacation or their mail gets stuck, the whole team sits and waits. In trading, this means fast algorithms remain idle, waiting for slow ones.

### 2. How it works in Asynchronous mode (Async FL)
As soon as a piece arrives in the mail, you immediately glue it onto the puzzle. You don't need to wait for the others.

### The "Old News" Problem
However, there's a catch: if someone sends a piece they found a week ago, it might no longer fit what you have assembled in the meantime.

In our system, we say:
- "Oh, a fresh piece! Let's put it in the center."
- "Hmm, this piece is a bit old. We'll include it, but carefully, so it doesn't ruin what we've already built."

**The Result**: The puzzle (model) is assembled continuously and very quickly, never stopping for a second.
