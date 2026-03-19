# Chapter 236: Conditional VAE Trading

## 1. Introduction — CVAE: Conditioning Generation on Auxiliary Information

Variational Autoencoders (VAEs) have proven to be powerful generative models in quantitative finance, capable of learning latent representations of market data and generating synthetic scenarios for risk analysis, portfolio optimization, and strategy backtesting. However, standard VAEs treat all data as arising from a single generative process, ignoring the well-documented fact that financial markets exhibit distinct behavioral regimes. Bull markets, bear markets, sideways consolidation, high-volatility crises, and low-volatility grinds all produce return distributions with fundamentally different characteristics.

A **Conditional Variational Autoencoder (CVAE)** extends the standard VAE framework by incorporating auxiliary conditioning information — such as market regime labels, sector classifications, or macroeconomic indicators — directly into both the encoding and decoding processes. Rather than learning a single latent space that must accommodate all market conditions, the CVAE learns a *conditioned* latent space where the generative model can be steered by specifying the desired context.

This capability is transformative for trading applications. Instead of generating random market scenarios and hoping some of them represent stressed conditions, a trader can explicitly request: "Generate 10,000 realistic return paths conditioned on a bear market regime with rising volatility." The CVAE produces scenarios that are statistically faithful to historical bear market dynamics while providing the diversity needed for robust risk estimation.

In this chapter, we develop a complete CVAE implementation in Rust, integrate it with live market data from Bybit, and demonstrate how conditional generation dramatically improves the quality and controllability of synthetic financial data compared to standard VAEs.

## 2. Mathematical Foundation

### The CVAE Objective

The standard VAE maximizes the Evidence Lower Bound (ELBO):

```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
```

The CVAE extends this by conditioning every distribution on an auxiliary variable **c**:

```
ELBO(x, c) = E_q(z|x,c)[log p(x|z,c)] - KL(q(z|x,c) || p(z|c))
```

This objective involves three key components:

**Encoder q(z|x,c):** The recognition network now takes both the input data x and the condition c, and produces parameters of the approximate posterior distribution over latent variables. Formally:

```
q(z|x,c) = N(z; mu_phi(x,c), sigma_phi(x,c))
```

where `mu_phi` and `sigma_phi` are neural networks parameterized by phi that map the concatenation (or some other combination) of x and c to the mean and variance of a Gaussian distribution.

**Decoder p(x|z,c):** The generative network reconstructs the input from both the latent code z and the condition c:

```
p(x|z,c) = N(x; mu_theta(z,c), sigma_theta(z,c))
```

This ensures that the decoder knows which regime it should be generating for, preventing it from averaging across regimes.

**Conditional Prior p(z|c):** Unlike the standard VAE which uses a fixed prior N(0,I), the CVAE learns a condition-dependent prior:

```
p(z|c) = N(z; mu_prior(c), sigma_prior(c))
```

This is critical because different market regimes may require different regions of latent space. A bull market prior might center on a different location than a bear market prior, allowing the model to maintain regime-specific structure in the latent space.

### The KL Divergence Term

The KL divergence between two Gaussians q(z|x,c) and p(z|c) has a closed-form solution:

```
KL = 0.5 * sum(log(sigma_prior^2 / sigma_encoder^2) - 1 + (sigma_encoder^2 + (mu_encoder - mu_prior)^2) / sigma_prior^2)
```

This term regularizes the encoder to produce latent codes that are close to the regime-specific prior, rather than to a single global prior. The result is a latent space that is organized by regime, making conditional generation natural and effective.

### The Reparameterization Trick

Training requires backpropagation through the sampling operation. The reparameterization trick expresses the sample as a deterministic function of the parameters and an auxiliary noise variable:

```
z = mu_phi(x,c) + sigma_phi(x,c) * epsilon,   epsilon ~ N(0,I)
```

This allows gradients to flow through the sampling operation, enabling end-to-end training with standard optimization methods.

## 3. Conditioning Strategies

The way conditioning information is injected into the network architecture significantly impacts model performance. Three primary strategies exist:

### Concatenation

The simplest approach: concatenate the condition vector with the input at each relevant layer. For a one-hot regime vector r of dimension K and input x of dimension D, the encoder receives a vector of dimension D+K. This is straightforward to implement and works well when the conditioning space is low-dimensional.

**Advantages:** Simple, no additional parameters, works reliably.
**Disadvantages:** The condition can be "washed out" by the input in deep networks.

### FiLM (Feature-wise Linear Modulation)

FiLM layers learn to scale and shift intermediate activations based on the condition:

```
FiLM(h; c) = gamma(c) * h + beta(c)
```

where gamma and beta are learned functions of the condition c. This is more expressive than concatenation because it modulates the *processing* of information, not just the input.

**Advantages:** More expressive, condition influences all layers, better for complex conditioning.
**Disadvantages:** More parameters, slightly harder to implement.

### Attention-Based Conditioning

Cross-attention mechanisms allow the model to selectively attend to different aspects of the condition:

```
Attention(Q=h, K=c, V=c) = softmax(h * c^T / sqrt(d)) * c
```

This is most powerful when the condition is itself high-dimensional or sequential (e.g., a time series of macro indicators).

**Advantages:** Most expressive, handles complex conditions, selective attention.
**Disadvantages:** Highest computational cost, requires more data to train.

In our implementation, we use concatenation for regime conditioning (a small discrete set) and FiLM-style modulation for continuous macro indicators, balancing simplicity with expressiveness.

## 4. Trading Applications

### Regime-Conditional Scenario Generation

The primary application is generating synthetic market scenarios conditioned on specific regimes. A risk manager can:

1. **Stress testing:** Generate thousands of bear market scenarios to evaluate portfolio drawdown distributions under adverse conditions, without being limited to the handful of historical bear markets available.

2. **Regime-specific VaR:** Compute Value-at-Risk separately for each regime, then weight by the probability of being in each regime. This produces more accurate risk estimates than unconditional VaR.

3. **Strategy robustness:** Test a trading strategy against generated scenarios for each regime independently, identifying which market conditions the strategy is most vulnerable to.

### Sector-Specific Return Modeling

By conditioning on sector labels, the CVAE can generate return distributions that respect sector-specific dynamics:

- Technology stocks exhibit higher volatility and momentum effects
- Utilities show lower volatility and mean-reversion tendencies
- Financial stocks have fat tails correlated with credit conditions

A sector-conditioned CVAE captures these differences, enabling more realistic cross-sectional simulation.

### Macro-Conditioned Portfolio Simulation

Conditioning on macroeconomic indicators (interest rates, inflation, GDP growth, unemployment) allows the CVAE to generate market scenarios consistent with specific macro environments:

- "What does my portfolio look like if inflation rises to 5% and the Fed raises rates by 200bp?"
- "Generate scenarios consistent with a recession where unemployment reaches 7%."

This capability is invaluable for strategic asset allocation and long-term planning.

## 5. CVAE vs Standard VAE — Why Conditioning Matters

### Mode Collapse and Averaging

Standard VAEs suffer from mode averaging: when the data contains distinct clusters (regimes), the decoder learns to produce outputs that are an average of all modes. This results in generated samples that do not actually resemble any real market condition — they are too volatile for calm markets and not volatile enough for crisis markets.

The CVAE eliminates this problem by telling the decoder which mode to generate. The conditional prior ensures that the latent space is organized by regime, and the conditional decoder produces samples appropriate for the specified regime.

### Quantitative Improvements

In practice, conditioning provides several measurable improvements:

1. **Lower reconstruction error:** Per-regime reconstruction is more accurate because the decoder does not need to handle all regimes simultaneously.

2. **Better distributional match:** The Wasserstein distance between generated and real return distributions decreases significantly when conditioning is used, typically by 30-50%.

3. **More realistic tail behavior:** Regime-specific generation produces more accurate extreme return distributions, which is critical for risk management.

4. **Controllability:** The ability to specify the generation condition enables applications that are impossible with standard VAEs.

### The Cost of Conditioning

Conditioning requires labeled data — someone must identify regimes, sectors, or provide macro indicators. For market regimes, automated detection methods (trend following, volatility clustering, hidden Markov models) can produce labels at scale. For sector classification, standard industry taxonomies exist. Macro indicators are publicly available. The marginal cost of conditioning is therefore low relative to the benefits.

## 6. Implementation Walkthrough with Rust

Our Rust implementation provides a complete CVAE system with the following architecture:

### Core Components

The **ConditionEncoder** transforms raw conditioning information into a fixed-dimensional representation. For discrete conditions like market regimes (bull/bear/sideways), it uses one-hot encoding. For continuous conditions like macro indicators, it applies a learned linear transformation.

The **CVAEEncoder** takes the concatenation of input features and the encoded condition, and produces the mean and log-variance of the approximate posterior q(z|x,c). It uses a two-layer feedforward architecture with ReLU activations.

The **CVAEDecoder** takes the concatenation of latent samples and the encoded condition, and reconstructs the input. It mirrors the encoder architecture.

The **ConditionalPrior** maps the condition to a regime-specific prior distribution p(z|c), allowing different regimes to occupy different regions of latent space.

### Training Loop

Training proceeds by:

1. Encoding the input and condition to get posterior parameters
2. Sampling from the posterior using reparameterization
3. Decoding the sample with the condition to get reconstruction
4. Computing the conditional prior parameters
5. Calculating the ELBO loss (reconstruction + KL divergence)
6. Updating parameters via gradient descent

### Regime Detection

Our implementation includes a simple but effective regime detector based on rolling statistics:

- **Bull market:** Positive rolling mean return with below-average volatility
- **Bear market:** Negative rolling mean return with above-average volatility
- **Sideways:** Low absolute return with below-average volatility

This heuristic provides adequate labels for demonstrating the CVAE's capabilities, though production systems would benefit from more sophisticated approaches like hidden Markov models.

### Quality Metrics

We evaluate generated samples per regime using:

- **Mean and variance match:** How closely generated statistics match real data within each regime
- **Distributional distance:** Simplified Wasserstein-like distance between generated and real distributions
- **Tail accuracy:** How well extreme quantiles are captured

## 7. Bybit Data Integration

Our implementation fetches real market data from the Bybit API, specifically BTCUSDT kline (candlestick) data. The integration works as follows:

1. **API endpoint:** We use `https://api.bybit.com/v5/market/kline` to fetch historical OHLCV data.

2. **Data processing:** Raw kline data is converted to log returns, which serve as the CVAE's input features. We compute returns from close prices and normalize them for numerical stability.

3. **Regime labeling:** The fetched price data is processed through our regime detector to produce condition labels. Each time window is classified as bull, bear, or sideways based on rolling statistics.

4. **Feature construction:** Each training sample consists of a window of returns (e.g., 5 consecutive returns) paired with the regime label for that window.

This pipeline provides a fully automated workflow from raw market data to trained CVAE, requiring no manual data preparation.

## 8. Key Takeaways

1. **Conditional VAEs extend standard VAEs** by incorporating auxiliary information (regime, sector, macro) into the encoder, decoder, and prior, enabling controlled generation of market scenarios.

2. **The conditional ELBO** naturally decomposes into regime-specific reconstruction and KL terms, ensuring that each regime receives appropriate modeling capacity.

3. **Concatenation conditioning** is the simplest and most robust strategy for low-dimensional discrete conditions like market regimes. FiLM and attention-based methods are preferable for high-dimensional or continuous conditions.

4. **Regime-conditional generation** eliminates the mode-averaging problem of standard VAEs, producing scenarios that are faithful to specific market conditions rather than averaging across all conditions.

5. **Practical improvements** from conditioning include 30-50% better distributional match, more realistic tail behavior, and the critical ability to generate targeted scenarios for stress testing and risk management.

6. **Rust implementation** provides the performance needed for real-time scenario generation, with the Bybit integration enabling a complete pipeline from live data to conditional generation.

7. **Automated regime detection** based on rolling statistics provides adequate conditioning labels, making the CVAE pipeline fully automated from raw price data to conditional scenario generation.

8. **Trading applications** span stress testing, regime-specific VaR, strategy robustness evaluation, sector-specific modeling, and macro-conditioned portfolio simulation — all made possible by the ability to condition the generative process on relevant auxiliary information.
