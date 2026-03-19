# VAE Volatility Surface — Explained Simply

## What Is a Volatility Surface?

Imagine you run a weather station, and you need to predict how "stormy" the financial markets will be. But instead of one forecast, you need a whole **map** — like a weather map that shows storm intensity across different locations and times.

In finance, this map is called a **volatility surface**. It tells traders how wild price swings are expected to be for options at different prices (strikes) and different time horizons (expirations).

Think of it like a topographic map of a mountain range:
- The **x-axis** is the strike price (how far from the current price)
- The **y-axis** is when the option expires (next week, next month, next year)
- The **height** at any point is the expected volatility (how bumpy the ride will be)

## The Problem: Missing Pieces

Here's the catch: we don't have data for every point on this map. Options only trade at certain strike prices and certain expiration dates. It's like having temperature readings from only 10 weather stations but needing a complete weather map for an entire country.

Traditional methods use mathematical formulas to "connect the dots," but these formulas make strong assumptions about the shape. What if the real shape doesn't match their assumptions?

## Enter the VAE: A Smart Artist

A **Variational Autoencoder (VAE)** is like training an artist who:

1. **Studies** thousands of real volatility surface maps
2. **Learns** the essential patterns — "most surfaces have this general shape, with these typical variations"
3. **Compresses** each surface into a small set of numbers (the "latent code") — like describing a face with just a few traits: "round, smiling, brown eyes"
4. **Reconstructs** complete surfaces from these compressed descriptions

### The Compression Game

Imagine describing a photograph using only 4 numbers. That sounds impossible! But if you've seen millions of photographs of the same type (say, landscapes), you start to notice patterns:
- Number 1 could represent "how sunny"
- Number 2 could represent "how mountainous"
- Number 3 could represent "how green"
- Number 4 could represent "time of day"

The VAE does the same thing with volatility surfaces:
- Number 1 might capture "overall volatility level"
- Number 2 might capture "skew" (are put options more expensive than calls?)
- Number 3 might capture "term structure" (is short-term vol higher than long-term?)
- Number 4 might capture "smile curvature" (how curved is the pattern?)

## No Cheating Allowed: Arbitrage Rules

There's an important rule: the generated surfaces must be **realistic**. In finance, an unrealistic surface would let someone make free money (called "arbitrage"). The VAE is trained with extra rules that say:

- **Rule 1 (Butterfly)**: If you look at three nearby strike prices, the middle one can't be cheaper than the average of the other two (that would be free money from a "butterfly spread")
- **Rule 2 (Calendar)**: Uncertainty can't decrease when you look further into the future (that would be free money from a "calendar spread")

It's like teaching an artist to draw landscapes where water always flows downhill — the laws of physics (or finance) must be obeyed.

## Why This Matters

1. **Pricing**: Banks and traders need complete surfaces to price complex options — the VAE fills in the gaps
2. **Risk Management**: "What if markets crash?" — sample scary surfaces from the VAE to test portfolios
3. **Finding Deals**: If the market surface looks different from the VAE's "fair" surface, maybe something is mispriced
4. **Understanding Regimes**: The latent code can reveal "we're in a stressed regime" vs "markets are calm"

## Try It Yourself

The Rust implementation in this chapter lets you:

1. **Generate** synthetic volatility surfaces using the SABR model
2. **Train** a VAE to learn the surface distribution
3. **Encode** a surface into a compact latent representation
4. **Decode** a latent code back into a complete surface
5. **Check** for arbitrage violations in generated surfaces
6. **Fetch** real BTC options data from Bybit to build actual surfaces
