# CPC: The "Detective" Analogy

Imagine you are watching a detective series, but the sound is off and you only see individual frames.

### 1. Standard Training (Predicting the Frame)
You are forced to predict, pixel by pixel, exactly what the next frame will look like. This is almost impossible — there are too many details (wallpaper color, dust in the air). In trading, this is like trying to predict the exact price to the cent in 1 minute.

### 2. CPC (Contrastive Predictive Coding)
Instead of drawing the next frame, you try to guess its "meaning."
- You see the hero reaching for a gun.
- You predict that in 5 seconds, the "semantic field" of the frame will contain a "shot" or a "threat."
- If a "shot" actually occurs in the future — your understanding of the current situation ($c_t$) was correct.

You contrast this real future frame with frames from other movies (negative samples). If you can distinguish the real future of this series from random clips of a comedy — it means your neural network understands the **logic of events**.

In trading, CPC teaches the model to understand the "plot" of the market: "We are currently in an accumulation phase, and the logical continuation is an upward impulse." It doesn't matter if the price is 100.05 or 100.07; what matters is that the **state** of the market will change in a predictable way.
