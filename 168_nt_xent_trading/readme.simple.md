# NT-Xent: The "Lens" Analogy

Imagine you are looking at a group of people in the fog and trying to distinguish twins from strangers.

### 1. Temperature ($\tau$) as Lens Focus
Temperature in NT-Xent works like the focus ring on a camera lens:

- **Low Temperature ($\tau = 0.05$)**: This is a "microscope." You see the slightest differences. The model becomes extremely picky. It ignores everyone who is even slightly different from the original and spends all its energy distinguishing "near-twins" from real twins. In trading, this helps catch subtle patterns, but there is a risk of mistaking random noise for an important signal.
- **High Temperature ($\tau = 1.0$)**: This is "soft focus." Everyone looks roughly the same. The model doesn't care how much a negative sample resembles the anchor — it penalizes everyone a little bit. Training is stable, but the embeddings end up "blurry" and indistinct.

### 2. Why is this needed?
Without temperature scaling, contrastive loss performs poorly because cosine similarity is always in the range $[-1, 1]$. This range is too narrow for the `exp()` function to create meaningful differences in probabilities. $\tau$ stretches this range, allowing the model to clearly say: "This sample is definitely an enemy, and this one is a friend."

In trading, the right choice of $\tau$ allows the model to ignore market noise (high-frequency fluctuations) while remaining sensitive to real structural price changes.
