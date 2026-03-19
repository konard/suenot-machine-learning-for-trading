# Chapter 237: Hierarchical VAE Trading

## 1. Introduction — Multi-Scale Latent Representations for Financial Markets

Financial markets operate simultaneously across multiple time scales. Intraday price fluctuations reflect microstructure noise and short-term order flow dynamics. Daily and weekly returns capture momentum effects, earnings surprises, and sector rotation. Monthly and quarterly patterns reveal macroeconomic cycles, credit conditions, and structural shifts in investor sentiment. A trading model that captures only one scale inevitably misses critical information available at others.

Standard Variational Autoencoders (VAEs) learn a single latent space that must compress all variation — from tick-level noise to macro regime shifts — into a flat representation. This forces the model to make uncomfortable trade-offs: a latent space large enough to capture macro structure wastes capacity on noise, while one tuned for fine-grained patterns cannot represent slow-moving trends.

A **Hierarchical VAE (HVAE)** addresses this fundamental limitation by organizing the latent space into multiple levels, each operating at a different scale of abstraction. Higher levels capture slow-moving, global structure (market regimes, trend direction, volatility regimes), while lower levels encode fast-moving, local detail (daily return patterns, short-term mean reversion, microstructure effects). The levels interact through a top-down generative pathway: high-level latent variables set the context, and lower levels refine the details within that context.

This architecture is inspired by the **Ladder VAE** (Sønderby et al., 2016), which demonstrated that hierarchical latent variables with top-down inference paths dramatically improve the expressiveness and training stability of deep generative models. In trading, this translates to models that can simultaneously reason about "what regime are we in?" (top level) and "what specific return pattern is happening today?" (bottom level), with coherent interaction between these scales.

In this chapter, we develop a complete Hierarchical VAE implementation in Rust, integrate it with live market data from Bybit, and demonstrate how multi-scale latent representations improve both the quality of generated scenarios and the usefulness of learned representations for trading decisions.

## 2. Mathematical Foundation

### The Hierarchical Generative Model

A standard VAE uses a single latent variable z and factorizes the joint distribution as:

```
p(x, z) = p(z) * p(x|z)
```

A Hierarchical VAE introduces L levels of latent variables z_1, z_2, ..., z_L, where z_L is the highest (most abstract) level. The generative model factorizes top-down:

```
p(x, z_1, ..., z_L) = p(z_L) * prod_{l=L-1}^{1} p(z_l | z_{l+1}) * p(x | z_1)
```

Each conditional p(z_l | z_{l+1}) is parameterized by a neural network that takes the higher-level latent variable and produces the parameters (mean, variance) of a Gaussian distribution for the current level. The prior at the top level p(z_L) is typically a standard Gaussian N(0, I).

### The Hierarchical Inference Model

The approximate posterior (encoder) mirrors the generative structure but operates bottom-up, then combines with top-down information:

**Bottom-up pass:** A deterministic encoder processes the input x through a series of layers, producing intermediate representations d_1, d_2, ..., d_L at each level.

**Top-down pass with inference:** Starting from the top level, at each level l:

```
q(z_l | z_{l+1}, x) = N(z_l; mu_q_l, sigma_q_l)
```

where the parameters mu_q_l and sigma_q_l are computed by combining:
- The bottom-up representation d_l (from the data)
- The top-down representation from z_{l+1} (from higher levels)

This combination is typically done through a precision-weighted merge:

```
mu_q_l = (mu_bu_l / sigma_bu_l^2 + mu_td_l / sigma_td_l^2) / (1/sigma_bu_l^2 + 1/sigma_td_l^2)
sigma_q_l^2 = 1 / (1/sigma_bu_l^2 + 1/sigma_td_l^2)
```

where mu_bu_l, sigma_bu_l come from the bottom-up path and mu_td_l, sigma_td_l from the top-down generative model.

### The Hierarchical ELBO

The Evidence Lower Bound for the hierarchical model decomposes as:

```
ELBO = E_q[log p(x|z_1)] - sum_{l=1}^{L} KL(q(z_l | z_{l+1}, x) || p(z_l | z_{l+1}))
```

This is the sum of a reconstruction term (how well the lowest-level latent reconstructs the input) and L KL divergence terms (one per level, measuring how much the inference model deviates from the generative prior at each level).

### Why Hierarchy Helps: The Information Allocation Argument

The key insight is that hierarchy allows the model to allocate information efficiently across levels. Without hierarchy, the KL penalty forces all information through a single bottleneck, leading to two failure modes:

1. **Posterior collapse:** The model ignores the latent variables entirely, relying solely on the decoder's capacity. The KL term drives q(z|x) toward p(z), making z uninformative.

2. **Information overload:** Too much information is packed into a single latent space, making the representations entangled and hard to use.

With hierarchy, each level can specialize:
- High levels capture broad structure with low KL cost (this information is shared across many data points)
- Low levels capture instance-specific detail with higher KL cost but correspondingly higher reconstruction benefit

This natural information allocation produces representations that are interpretable at each level — exactly what we need for multi-scale trading analysis.

### KL Annealing and Free Bits

Training hierarchical VAEs requires careful handling of the KL terms to prevent posterior collapse at individual levels. Two techniques are standard:

**KL Annealing:** A warmup coefficient beta increases from 0 to 1 over the first portion of training:

```
Loss = Reconstruction + beta * sum(KL_l)
```

This allows the model to first learn useful representations, then gradually regularize them.

**Free Bits:** A minimum KL value lambda is enforced per level:

```
KL_l_effective = max(KL_l, lambda)
```

This prevents any level from collapsing to the prior, ensuring all levels carry information.

## 3. Multi-Scale Financial Interpretation

### Level Mapping to Trading Horizons

In our implementation with L=3 levels:

**Level 3 (Top — Macro Scale):** Captures market regime and trend direction. Latent variables at this level change slowly and correspond to macro states: bull/bear markets, high/low volatility regimes, risk-on/risk-off environments. These variables have the highest autocorrelation and are most useful for strategic allocation decisions.

**Level 2 (Middle — Tactical Scale):** Captures medium-term patterns like momentum persistence, mean reversion cycles, and sector rotation. These variables change on a weekly-to-monthly basis and are relevant for tactical trading decisions: position sizing, sector tilts, and risk adjustment.

**Level 1 (Bottom — Execution Scale):** Captures day-to-day return patterns, short-term volatility clustering, and microstructure effects. These variables change rapidly and are most relevant for trade timing, execution optimization, and short-term hedging.

### Generation with Scale Control

The hierarchical structure enables a powerful form of controlled generation. By fixing the top-level latent variables and sampling the lower levels, we can generate diverse daily return scenarios that all share the same macro regime. This is far more nuanced than the single-regime conditioning of a Conditional VAE — we can specify not just "bear market" but a specific type of bear market (gradual decline vs. crash, sector-specific vs. broad-based).

Conversely, by fixing lower levels and varying the top, we can explore how the same local pattern plays out under different macro conditions — useful for regime-change stress testing.

## 4. Trading Applications

### Multi-Scale Risk Assessment

Traditional risk models operate at a single scale, typically daily returns. This misses important cross-scale interactions:

- A portfolio may appear safe at the daily level while accumulating dangerous macro exposures
- Short-term hedging may be adequate for normal conditions but fail during regime transitions

The HVAE's multi-scale representations enable risk assessment at each level independently and in combination:

1. **Macro risk (Level 3):** Probability of regime transition, expected duration of current regime
2. **Tactical risk (Level 2):** Momentum reversal probability, sector concentration risk
3. **Execution risk (Level 1):** Short-term volatility forecasting, liquidity risk

### Hierarchical Scenario Generation

For stress testing, the hierarchical structure allows generating scenarios at each level of the hierarchy:

1. Fix a stressed macro state at Level 3 (e.g., recession)
2. Sample diverse tactical patterns at Level 2 (different paths through the recession)
3. For each tactical path, generate multiple daily return sequences at Level 1

This produces a tree of scenarios with meaningful structure, rather than a flat collection of independent paths.

### Latent Factor Trading Signals

Each level of the hierarchy produces latent variables that can serve as trading signals:

- **Level 3 shifts** → Regime change signals for strategic rebalancing
- **Level 2 trends** → Momentum/mean-reversion signals for tactical trading
- **Level 1 patterns** → Short-term alpha signals for execution timing

The hierarchical structure ensures these signals are naturally orthogonal — each level captures variation not explained by levels above it.

## 5. Architecture Design

### Bottom-Up Encoder

The bottom-up encoder processes input features through a series of layers with increasing abstraction. Each layer produces a deterministic representation d_l that captures information at the corresponding scale:

```
d_1 = ReLU(W_1 * x + b_1)          # Local patterns
d_2 = ReLU(W_2 * d_1 + b_2)        # Medium-scale patterns
d_3 = ReLU(W_3 * d_2 + b_3)        # Global patterns
```

At each level, d_l is used to produce bottom-up parameters for the approximate posterior.

### Top-Down Decoder with Stochastic Layers

The top-down path starts from the top level and works downward. At each level:

1. Combine top-down context with bottom-up information
2. Sample latent variables from the merged distribution
3. Pass the sample to the next level down

```
z_3 ~ q(z_3 | d_3)                            # Top level: just bottom-up
z_2 ~ q(z_2 | merge(d_2, transform(z_3)))     # Middle: merge both paths
z_1 ~ q(z_1 | merge(d_1, transform(z_2)))     # Bottom: merge both paths
x_recon = decode(z_1)                          # Reconstruction from bottom level
```

### Merge Operation

The merge operation combines bottom-up and top-down Gaussian parameters using precision-weighted averaging. Given two Gaussian distributions N(mu_1, sigma_1^2) and N(mu_2, sigma_2^2), the merged distribution is:

```
precision = 1/sigma_1^2 + 1/sigma_2^2
mu_merged = (mu_1/sigma_1^2 + mu_2/sigma_2^2) / precision
sigma_merged^2 = 1 / precision
```

This ensures that the level with more precise (lower variance) information dominates the merged distribution — a principled way to combine data-driven and model-driven information.

## 6. Implementation Walkthrough with Rust

Our Rust implementation provides a complete Hierarchical VAE system with the following components:

### Core Architecture

The **HierarchicalEncoder** implements the bottom-up pass, transforming input features through multiple layers to produce deterministic representations at each level. Each level's representation is used to compute bottom-up mean and log-variance parameters for the approximate posterior.

The **HierarchicalDecoder** implements the top-down generative path. At each level (starting from the top), it computes top-down prior parameters, merges them with bottom-up parameters using precision-weighted averaging, samples latent variables, and passes them to the next level. The final reconstruction is produced from the bottom-level latent variables.

The **PrecisionWeightedMerge** operation combines bottom-up and top-down Gaussian distributions at each level, weighting each by its precision (inverse variance). This principled combination allows the model to balance data-driven evidence (bottom-up) with structural priors (top-down).

### Training Strategy

Training proceeds with:

1. **Bottom-up pass:** Encode input to produce deterministic representations at all levels
2. **Top-down inference:** Starting from the top, merge bottom-up and top-down information, sample latent variables at each level
3. **Reconstruction:** Decode from bottom-level latents
4. **Loss computation:** Reconstruction MSE + KL divergence at each level
5. **KL annealing:** Gradually increase the weight on KL terms over training epochs to prevent posterior collapse
6. **Parameter update:** Numerical gradient descent (demonstration; production should use autodiff)

### Per-Level Diagnostics

We monitor KL divergence at each level separately. Healthy training shows:
- Level 3 (top): Moderate KL — captures global regime information
- Level 2 (middle): Moderate KL — captures tactical patterns
- Level 1 (bottom): Higher KL — captures fine-grained detail

If any level shows near-zero KL, it indicates posterior collapse at that level, which can be addressed by increasing the free-bits threshold.

## 7. Bybit Data Integration

Our implementation fetches real market data from the Bybit API, specifically BTCUSDT kline (candlestick) data. The integration works as follows:

1. **API endpoint:** We use `https://api.bybit.com/v5/market/kline` to fetch historical OHLCV data.

2. **Data processing:** Raw kline data is converted to log returns, which serve as the HVAE's input features. We compute returns from close prices and normalize them for numerical stability.

3. **Multi-scale feature construction:** Each training sample consists of a window of normalized returns. The hierarchical VAE internally learns to decompose this window into multi-scale components — the architecture handles scale separation automatically through the hierarchy of latent variables.

4. **Regime evaluation:** After training, we detect market regimes using rolling statistics and evaluate whether the HVAE's top-level latent variables correlate with detected regimes, validating that the hierarchy has learned meaningful multi-scale structure.

This pipeline provides a fully automated workflow from raw market data to trained Hierarchical VAE, requiring no manual data preparation or explicit multi-scale feature engineering.

## 8. Key Takeaways

1. **Hierarchical VAEs organize latent space into multiple levels**, each capturing variation at a different scale of abstraction — from macro regime structure to fine-grained daily patterns.

2. **The hierarchical ELBO decomposes into per-level KL terms**, allowing the model to allocate information efficiently across scales. High levels capture broadly shared structure; low levels capture instance-specific detail.

3. **Precision-weighted merging** of bottom-up (data-driven) and top-down (model-driven) information at each level provides a principled mechanism for combining evidence across the hierarchy.

4. **Multi-scale generation** enables nuanced scenario construction: fix the macro regime at the top level, then generate diverse tactical and daily paths beneath it. This produces scenario trees rather than flat collections.

5. **KL annealing and free bits** are essential training techniques that prevent posterior collapse — a failure mode where individual hierarchy levels become uninformative.

6. **Trading applications span multiple horizons**: top-level latents provide regime signals for strategic allocation, middle-level latents inform tactical positioning, and bottom-level latents support execution timing.

7. **Rust implementation** provides the performance needed for real-time multi-scale analysis, with the Bybit integration enabling a complete pipeline from live data to hierarchical generation.

8. **Compared to flat VAEs and CVAEs**, the hierarchical architecture learns more structured representations without requiring explicit conditioning labels — the hierarchy discovers scale separation automatically from the data.
