# Chapter 240: VAE Volatility Surface

## Introduction

A **volatility surface** maps implied volatility as a function of strike price and time to expiration. It is one of the most critical objects in derivatives pricing and risk management. However, observed market quotes are sparse—options trade at discrete strikes and maturities, leaving large portions of the surface unobserved. Traditional interpolation methods (cubic splines, SABR, SVI) enforce parametric assumptions that may not capture the true data-generating process.

**Variational Autoencoders (VAEs)** offer a data-driven, non-parametric alternative. A VAE learns a low-dimensional latent representation of the volatility surface, enabling:

- **Surface completion**: filling in missing implied volatilities for unquoted strikes and maturities.
- **Arbitrage-free generation**: producing synthetic surfaces that respect no-arbitrage constraints (no calendar spread or butterfly arbitrage).
- **Regime-aware modeling**: encoding market regimes (calm, stressed, trending) in the latent space.
- **Scenario generation**: sampling new plausible surfaces for stress testing and risk analysis.

## Key Concepts

### Volatility Surface Representation

A volatility surface $\sigma(K, T)$ can be discretized on a grid of $N_K$ strikes and $N_T$ maturities:

$$\mathbf{S} \in \mathbb{R}^{N_K \times N_T}$$

where $S_{i,j} = \sigma(K_i, T_j)$ is the implied volatility at strike $K_i$ and maturity $T_j$.

For neural network input, the surface is often expressed in **moneyness** $m = K / F$ (where $F$ is the forward price) and **time to expiration** $\tau$, yielding a normalized representation that is comparable across different underlying assets and dates.

### VAE Architecture for Surfaces

A VAE consists of an **encoder** $q_\phi(z | \mathbf{S})$ and a **decoder** $p_\theta(\mathbf{S} | z)$:

**Encoder:**
$$\mu, \log\sigma^2 = f_\phi(\mathbf{S})$$
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Decoder:**
$$\hat{\mathbf{S}} = g_\theta(z)$$

The model is trained by maximizing the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}(\theta, \phi; \mathbf{S}) = \mathbb{E}_{q_\phi(z|\mathbf{S})}[\log p_\theta(\mathbf{S}|z)] - D_{KL}(q_\phi(z|\mathbf{S}) \| p(z))$$

The first term is the **reconstruction loss** (how well the decoded surface matches the input), and the second is the **KL divergence** regularizer that keeps the latent distribution close to a standard normal prior.

### No-Arbitrage Constraints

A valid implied volatility surface must satisfy:

1. **No butterfly arbitrage**: The call price must be convex in strike:
$$\frac{\partial^2 C}{\partial K^2} \geq 0$$

2. **No calendar spread arbitrage**: Total implied variance must be non-decreasing in maturity:
$$\sigma^2(K, T_1) \cdot T_1 \leq \sigma^2(K, T_2) \cdot T_2 \quad \text{for } T_1 < T_2$$

These constraints can be incorporated into the VAE loss as penalty terms:

$$\mathcal{L}_{total} = \mathcal{L}_{ELBO} + \lambda_{butterfly} \cdot \mathcal{L}_{butterfly} + \lambda_{calendar} \cdot \mathcal{L}_{calendar}$$

### Beta-VAE for Disentanglement

A $\beta$-VAE modifies the ELBO with a weighting factor $\beta > 1$ on the KL term:

$$\mathcal{L}_{\beta} = \mathbb{E}[\log p_\theta(\mathbf{S}|z)] - \beta \cdot D_{KL}(q_\phi(z|\mathbf{S}) \| p(z))$$

Higher $\beta$ encourages more disentangled latent dimensions, where individual latent variables correspond to interpretable surface characteristics:
- **Level**: overall volatility level (ATM vol)
- **Skew**: asymmetry between OTM puts and OTM calls
- **Term structure**: slope of the ATM volatility across maturities
- **Curvature (smile)**: convexity of the volatility smile

## ML Approaches

### Convolutional VAE

When the volatility surface is represented as a 2D grid (moneyness × maturity), convolutional layers are natural:

- **Encoder**: Conv2D layers with stride to downsample the surface, followed by fully connected layers to produce $\mu$ and $\log\sigma^2$.
- **Decoder**: Fully connected layers followed by transposed Conv2D layers to reconstruct the surface grid.

Convolutional architectures capture local structure—nearby strikes and maturities tend to have correlated volatilities.

### Conditional VAE (CVAE)

A Conditional VAE conditions on auxiliary information $c$ (e.g., spot price level, VIX, interest rate, market regime):

$$q_\phi(z | \mathbf{S}, c), \quad p_\theta(\mathbf{S} | z, c)$$

This allows generating surfaces conditioned on specific market states, which is valuable for scenario analysis: "What would the surface look like if VIX doubled?"

### Surface Completion

For incomplete surfaces (missing grid points), the VAE can be trained with a **masked reconstruction loss**:

$$\mathcal{L}_{recon} = \sum_{(i,j) \in \text{observed}} \left( S_{i,j} - \hat{S}_{i,j} \right)^2$$

At inference time, the decoder fills in all grid points, providing a complete surface from partial observations.

## Feature Engineering

Key features for conditioning the VAE:

| Feature | Description |
|---------|-------------|
| ATM implied vol | At-the-money volatility level |
| 25-delta skew | $\sigma_{25\Delta P} - \sigma_{25\Delta C}$ |
| 25-delta butterfly | $\frac{\sigma_{25\Delta P} + \sigma_{25\Delta C}}{2} - \sigma_{ATM}$ |
| Term structure slope | $\sigma_{ATM,6M} - \sigma_{ATM,1M}$ |
| VIX / DVOL | Market-wide volatility index |
| Realized vol ratio | $\sigma_{realized} / \sigma_{implied}$ |
| Spot return | Recent underlying return |

## Applications

1. **Options Pricing**: Generate complete surfaces from sparse market quotes to price exotic options.
2. **Risk Management**: Sample surfaces from the latent space for VaR and stress testing.
3. **Relative Value Trading**: Detect mispricings by comparing market-observed surface to VAE-reconstructed "fair" surface.
4. **Regime Detection**: Cluster latent representations to identify distinct volatility regimes.
5. **Hedging**: Use latent factors as hedging instruments (hedge against "skew moves" or "level shifts").

## Rust Implementation

The Rust implementation in `rust/` provides:

- **VolSurface**: Struct representing a discretized implied volatility surface on a moneyness × maturity grid.
- **VAEModel**: A VAE with configurable encoder/decoder architecture, including $\beta$-VAE support.
- **ArbitrageChecker**: Functions to verify butterfly and calendar spread no-arbitrage conditions.
- **SurfaceGenerator**: Synthetic volatility surface generation using the SABR model for training data.
- **BybitClient**: Fetches BTC options data from Bybit V5 API to construct real implied volatility surfaces.

## Bybit API Integration

The implementation fetches cryptocurrency options data from Bybit to construct real volatility surfaces:

- **Endpoint**: `/v5/market/tickers?category=option&baseCoin=BTC`
- **Data**: Implied volatilities, strikes, and expirations for BTC options
- **Processing**: Groups options by expiration, constructs surface grid, normalizes to moneyness

## References

1. **Variational Autoencoders for Completing Volatility Surfaces**
   - URL: https://www.mdpi.com/1911-8074/18/5/239
   - Year: 2025

2. **Controllable Generation of Implied Volatility Surfaces with VAEs**
   - URL: https://arxiv.org/abs/2509.01743
   - Year: 2025

3. **Deep Learning Volatility: A Deep Neural Network Perspective on Pricing and Calibration in (Rough) Volatility Models**
   - Bayer, Horvath, Muguruza, Stemper, Tobia (2019)

4. **Arbitrage-Free Regularization of Neural Network Models**
   - Ackerer, Tagasovska (2020)

5. **Generative Adversarial Networks for Financial Trading Strategies Fine-Tuning and Combination**
   - Koshiyama, Firoozye, Treleaven (2021)
