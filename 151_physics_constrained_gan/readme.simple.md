# Physics-Constrained GAN for Trading -- Simplified

## What is it, in plain language?

Imagine you hire an artist to paint realistic landscapes. A regular artist (standard GAN) might paint something that *looks* pretty but has impossible physics -- water flowing uphill, shadows pointing toward the sun, trees floating in mid-air.

A **Physics-Constrained GAN** is like telling the artist: "Paint whatever you want, but gravity must work, light must come from the sun, and water must flow downhill." The paintings are still creative and varied, but they respect the laws of physics.

In finance, the "paintings" are **synthetic price charts**, and the "laws of physics" are well-known statistical properties of real markets:

- **Prices don't have a free lunch** (no-arbitrage / martingale property)
- **Calm days cluster together, and stormy days cluster together** (volatility clustering)
- **Extreme moves happen more often than a bell curve predicts** (fat tails)
- **Bad news creates more chaos than good news** (leverage effect)

## The Two Players: Generator and Discriminator

Think of it as a game between a **counterfeiter** and a **detective**:

```
COUNTERFEITER (Generator)
  "Here, look at this stock chart I made."
         |
         v
DETECTIVE (Discriminator)
  "Hmm... is this chart real or fake?"
```

- The **counterfeiter** keeps getting better at making fake charts
- The **detective** keeps getting better at spotting fakes
- Eventually, the counterfeiter produces charts so realistic that even the detective cannot tell

**The twist with physics constraints:** We also hire a **physics teacher** who checks every fake chart against financial laws. Even if the detective is fooled, the physics teacher will flag charts where "gravity doesn't work" (e.g., volatility never clusters, or extreme events never happen).

## The Kitchen Recipe Analogy

Think of baking a cake:

| Part | Baking | Physics-Constrained GAN |
|------|--------|------------------------|
| Raw ingredients | Random noise (z) | Random numbers |
| Recipe | Generator neural network | Transforms noise into returns |
| Taste tester | Discriminator | "Does this taste like real market data?" |
| Health inspector | Physics constraints | "Does this meet nutritional requirements?" |
| Final score | Loss function | Adversarial + Physics penalty |

A standard GAN only has the taste tester. If the cake tastes good, it passes -- even if it has zero nutritional value.

A physics-constrained GAN also has the health inspector. The cake must taste good AND be nutritionally sound.

## The Five Financial "Laws of Physics"

### 1. The Martingale Property (No Free Lunch)

**Analogy:** If you flip a fair coin, your expected winnings are always $0, no matter how many flips you have done before.

In markets, the *expected* future price equals today's price (adjusted for risk-free rate). Generated data must not contain systematic exploitable drift.

**Without constraint:** Generator might create charts where prices consistently go up by 1% every day -- free money!

**With constraint:** Generator learns that sometimes prices go up, sometimes down, and the average movement is near zero.

### 2. Volatility Clustering (Storms Travel in Packs)

**Analogy:** Weather. Storms do not alternate with sunshine every hour. Instead, you get a week of storms, then a week of calm weather. Market volatility works the same way.

**Without constraint:** Generator might alternate between big and small moves randomly, like flipping between stormy and calm every minute.

**With constraint:** Generator learns to produce sequences where big moves are followed by more big moves, and calm periods persist.

### 3. Fat Tails (Extreme Events are More Common Than You Think)

**Analogy:** Imagine rolling dice. With normal (Gaussian) dice, rolling a "100" is essentially impossible. But in financial markets, "100-rolls" happen far more often than the bell curve predicts. Think of Black Monday, the 2008 crash, or Bitcoin flash crashes.

**Without constraint:** Generator produces nice Gaussian-looking returns where extreme events almost never happen.

**With constraint:** Generator learns to occasionally produce extreme moves, matching the heavy tails we see in real data.

### 4. The Leverage Effect (Bad News Hits Harder)

**Analogy:** When a company's stock drops 20%, people panic and start selling more, creating even more volatility. But when it rises 20%, people celebrate quietly. Negative moves amplify future turbulence more than positive moves.

**Without constraint:** Generator treats upward and downward moves symmetrically.

**With constraint:** Generator learns that after a big drop, the next few periods should be more volatile.

### 5. Autocorrelation Structure (Returns Are Unpredictable, But Their Size Is Not)

**Analogy:** You cannot predict whether tomorrow will be rainy or sunny (returns have no autocorrelation). But if today was a hurricane, tomorrow is likely to be at least windy (absolute returns ARE autocorrelated).

**Without constraint:** Generator might produce returns where you can predict direction, or volatility jumps randomly.

**With constraint:** Generator learns: "no pattern in direction, but persistent pattern in magnitude."

## How It Works: The Training Loop

```
Step 1: Generate fake market data
  z (random noise) --[Generator]--> fake returns

Step 2: Score the fakes
  Detective score = "How real does this look?"
  Physics score  = "How well does this follow financial laws?"

Step 3: Total score
  Total = Detective score + Physics score

Step 4: Improve the generator
  Generator adjusts to: (a) fool the detective, (b) satisfy the physics teacher

Step 5: Improve the detective
  Detective gets better at spotting fakes

Repeat 500+ times.
```

## Why WGAN-GP? (Stable Training)

Regular GANs are notoriously unstable -- the counterfeiter and detective can get stuck in loops or one can overpower the other.

**WGAN-GP** (Wasserstein GAN with Gradient Penalty) is like adding a referee to the game who ensures fair play:

- Instead of "Is this real? Yes/No" (which can be unstable), it asks "How far is this from real?" (a smooth distance)
- The gradient penalty ensures the detective does not get too aggressive

## Conditional Generation: "What If" Scenarios

The most powerful feature: you can *ask* the generator for specific types of scenarios.

```
"Show me what Bitcoin might do in a crisis with extreme volatility"
  --> Generator produces 1000 realistic crisis paths

"Show me calm bull market scenarios for ETH"
  --> Generator produces 1000 gentle upward paths
```

This is like asking the artist: "Paint me a landscape, but make it winter" vs. "Paint me a landscape, but make it a tropical beach." The underlying skill is the same, but the conditions change the output.

## Real-World Uses

| Use Case | Description |
|----------|-------------|
| **Data augmentation** | Only 3 years of crypto data? Generate 100 years of realistic synthetic data |
| **Stress testing** | What happens to my portfolio in 10,000 different crash scenarios? |
| **Strategy testing** | Test my trading algorithm on synthetic data before risking real money |
| **Privacy** | Share realistic market data without revealing actual proprietary trades |

## Quick Code Example

```python
# Train the GAN
from model import PhysicsConstrainedGAN, GANConfig
from data_loader import BybitDataLoader

loader = BybitDataLoader(symbol="BTCUSDT", interval="1h")
data = loader.fetch_and_preprocess(days=365)

gan = PhysicsConstrainedGAN(GANConfig())
# ... train ...

# Generate crisis scenarios
crisis_paths = gan.generate(
    n_paths=1000,
    condition={'regime': 'crisis', 'volatility': 'extreme'},
    as_prices=True,
)

# How bad could it get?
worst_case = crisis_paths[:, -1].min()
print(f"Worst case: {worst_case:.2f}")
```

## Summary

| Concept | One-liner |
|---------|-----------|
| **GAN** | Counterfeiter vs. detective game for generating data |
| **Physics constraints** | Financial laws baked into the training objective |
| **Martingale** | No systematic free money in generated data |
| **Vol clustering** | Stormy days cluster together |
| **Fat tails** | Extreme events happen more than bell curve says |
| **Leverage effect** | Bad news creates extra chaos |
| **WGAN-GP** | Stable training with fair referee |
| **Conditional generation** | "Show me crisis scenarios" on demand |
