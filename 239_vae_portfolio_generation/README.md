# Chapter 239: VAE Portfolio Generation

## Introduction

Portfolio construction is one of the central problems in quantitative finance. Traditional approaches such as mean-variance optimization (Markowitz, 1952) rely on point estimates of expected returns and covariances, which are notoriously unstable and sensitive to estimation error. Variational Autoencoders (VAEs) offer a fundamentally different approach: instead of optimizing a single portfolio, we learn the entire distribution of viable portfolios and generate novel allocations by sampling from a structured latent space.

A VAE learns to compress high-dimensional portfolio weight vectors into a low-dimensional latent representation while simultaneously learning to reconstruct realistic portfolios from this compressed space. The key insight is that the latent space captures the essential structure of well-formed portfolios — diversification patterns, sector exposures, risk characteristics — in a compact, continuous representation. By sampling from this latent space, we can generate an unlimited number of synthetic portfolios that share the statistical properties of the training set but are genuinely novel.

This chapter presents a complete framework for VAE-based portfolio generation. We cover the mathematical foundations of VAEs, their application to portfolio weight vectors, and a working Rust implementation that connects to the Bybit cryptocurrency exchange for real-time portfolio generation.

## Key Concepts

### Variational Autoencoders

A Variational Autoencoder consists of two neural networks: an encoder $q_\phi(\mathbf{z}|\mathbf{x})$ that maps input data to a distribution in latent space, and a decoder $p_\theta(\mathbf{x}|\mathbf{z})$ that maps latent vectors back to the data space.

Given an input portfolio weight vector $\mathbf{x} \in \mathbb{R}^n$ (where $n$ is the number of assets), the encoder produces parameters of a Gaussian distribution in latent space:

$$\boldsymbol{\mu} = f_\mu(\mathbf{x}; \phi), \quad \log \boldsymbol{\sigma}^2 = f_\sigma(\mathbf{x}; \phi)$$

A latent vector is sampled using the reparameterization trick:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The decoder then reconstructs the portfolio weights:

$$\hat{\mathbf{x}} = g(\mathbf{z}; \theta)$$

### Training Objective (ELBO)

The VAE is trained by maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The first term is the reconstruction loss, encouraging the decoder to accurately reproduce input portfolios. The second term is the KL divergence between the approximate posterior and the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, which regularizes the latent space to be smooth and continuous.

For portfolio weights with a sum-to-one constraint, the reconstruction loss is typically the mean squared error:

$$\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

The KL divergence for Gaussian distributions has a closed-form solution:

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

where $d$ is the dimensionality of the latent space.

### Portfolio Constraints

Generated portfolios must satisfy several constraints to be practically useful:

1. **Budget constraint**: Weights must sum to one: $\sum_{i=1}^{n} w_i = 1$
2. **Long-only constraint** (optional): All weights non-negative: $w_i \geq 0$
3. **Position limits**: No single position exceeds a maximum: $w_i \leq w_{\max}$
4. **Minimum allocation**: Meaningful positions only: $w_i \geq w_{\min}$ or $w_i = 0$

These constraints are enforced through a post-processing step applied to the decoder output. The most common approach is softmax normalization:

$$w_i = \frac{e^{\hat{x}_i}}{\sum_{j=1}^{n} e^{\hat{x}_j}}$$

This ensures non-negative weights that sum to one. Position limits can be enforced by clipping and renormalizing.

### Latent Space Properties

A well-trained VAE produces a latent space with useful geometric properties:

- **Smoothness**: Nearby points in latent space correspond to similar portfolios. Moving along a direction in latent space produces a continuous transformation of portfolio characteristics.
- **Interpolation**: Linear interpolation between two latent vectors produces a smooth transition between the corresponding portfolios, enabling portfolio blending.
- **Disentanglement**: Different latent dimensions may capture different portfolio characteristics (risk level, sector exposure, concentration), allowing targeted manipulation.
- **Coverage**: Sampling from the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ generates diverse, realistic portfolios that span the space of viable allocations.

## Portfolio Generation Strategies

### Unconditional Generation

The simplest approach samples latent vectors from the prior and decodes them:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \hat{\mathbf{w}} = \text{softmax}(g(\mathbf{z}; \theta))$$

This generates portfolios drawn from the learned distribution without any specific targeting. It is useful for exploring the full space of viable allocations and for Monte Carlo simulation of portfolio outcomes.

### Conditional Generation

A Conditional VAE (CVAE) conditions generation on desired properties such as target risk level, sector exposure, or expected return:

$$q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c}), \quad p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})$$

where $\mathbf{c}$ is a conditioning vector. For example, $\mathbf{c}$ might encode a target volatility of 15% and maximum allocation to technology of 30%. The CVAE learns to generate portfolios that satisfy these conditions while maintaining the diversity and realism of unconditional generation.

### Latent Space Optimization

Instead of random sampling, we can optimize in the latent space to find portfolios with specific properties:

$$\mathbf{z}^* = \arg\max_{\mathbf{z}} \; \text{Sharpe}(\text{decode}(\mathbf{z})) \quad \text{s.t.} \quad \|\mathbf{z}\| \leq r$$

The constraint $\|\mathbf{z}\| \leq r$ keeps the search within the region of high-density latent vectors, ensuring that optimized portfolios remain realistic. This approach combines the generative power of the VAE with traditional portfolio optimization objectives.

### Portfolio Interpolation

Given two reference portfolios $\mathbf{w}_A$ and $\mathbf{w}_B$, we can generate a continuum of blended portfolios:

$$\mathbf{z}_\alpha = (1 - \alpha) \cdot \text{encode}(\mathbf{w}_A) + \alpha \cdot \text{encode}(\mathbf{w}_B), \quad \alpha \in [0, 1]$$

$$\mathbf{w}_\alpha = \text{decode}(\mathbf{z}_\alpha)$$

Unlike simple weight averaging, this interpolation in latent space respects the learned structure of viable portfolios, producing intermediate allocations that are more realistic and better diversified.

## ML Approaches

### VAE Architecture for Portfolios

The encoder and decoder are typically shallow feed-forward networks:

**Encoder:**
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\boldsymbol{\mu} = \mathbf{W}_\mu \mathbf{h}_2 + \mathbf{b}_\mu$$
$$\log \boldsymbol{\sigma}^2 = \mathbf{W}_\sigma \mathbf{h}_2 + \mathbf{b}_\sigma$$

**Decoder:**
$$\mathbf{h}_3 = \text{ReLU}(\mathbf{W}_3 \mathbf{z} + \mathbf{b}_3)$$
$$\mathbf{h}_4 = \text{ReLU}(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4)$$
$$\hat{\mathbf{x}} = \mathbf{W}_5 \mathbf{h}_4 + \mathbf{b}_5$$

For a universe of $n$ assets with latent dimension $d$, typical layer sizes are:
- Input: $n$ (number of assets)
- Hidden layers: $64 \to 32$
- Latent dimension: $d = 8$ to $16$
- Output: $n$ (followed by softmax for weight normalization)

### Loss Function Design

The total loss balances reconstruction and regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}$$

The hyperparameter $\beta$ controls the trade-off:
- $\beta < 1$: Emphasizes reconstruction accuracy, producing portfolios closer to the training set but with less smooth latent space.
- $\beta = 1$: Standard VAE objective (ELBO).
- $\beta > 1$: Emphasizes latent space regularity ($\beta$-VAE), producing smoother interpolations but potentially sacrificing reconstruction accuracy.

For portfolio generation, $\beta \in [0.1, 0.5]$ often works well because precise weight reconstruction is more important than latent space smoothness.

### Training Data Generation

Training portfolios can be generated from several sources:

1. **Random Dirichlet sampling**: $\mathbf{w} \sim \text{Dir}(\boldsymbol{\alpha})$ generates random long-only portfolios. The concentration parameter $\boldsymbol{\alpha}$ controls the diversity: small $\alpha$ produces concentrated portfolios; large $\alpha$ produces uniform allocations.

2. **Historical optimized portfolios**: Run mean-variance optimization with different parameter settings (target returns, risk aversion levels, constraint sets) across rolling windows to generate a diverse training set of optimized portfolios.

3. **Risk parity portfolios**: Generate portfolios where each asset contributes equally to total risk, using different covariance estimation methods and lookback periods.

4. **Bootstrapped portfolios**: Resample historical returns and optimize to produce portfolios robust to estimation uncertainty.

## Feature Engineering

### Asset Return Features

Input features for conditioning the VAE include:

- **Rolling returns**: Mean returns over 5, 20, 60 trading days
- **Rolling volatility**: Standard deviation of returns over matching windows
- **Correlation structure**: Pairwise correlations between assets, possibly compressed via PCA
- **Momentum scores**: Risk-adjusted momentum (return divided by volatility) for trend signals

### Risk Features

- **Portfolio volatility**: $\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$
- **Maximum drawdown**: Worst peak-to-trough decline over a lookback period
- **Value at Risk (VaR)**: The loss level that is not exceeded with a given probability
- **Conditional VaR (CVaR)**: Expected loss given that loss exceeds VaR

### Market Regime Features

- **VIX level or crypto volatility index**: Overall market fear gauge
- **Trend indicator**: Whether the broad market is in an uptrend or downtrend
- **Dispersion**: Cross-sectional standard deviation of asset returns (high dispersion favors stock picking)

## Applications

### Portfolio Diversification

VAE-generated portfolios provide a natural source of diversified allocations. By sampling many portfolios and selecting those with the best risk-return characteristics, investors can discover allocations that may not emerge from traditional optimization. The VAE's ability to capture complex dependencies between asset weights ensures that generated portfolios respect the correlation structure of the underlying assets.

### Stress Testing

By systematically exploring the latent space, risk managers can generate portfolios that span the full range of possible allocations and evaluate their performance under stress scenarios. This is more thorough than testing a single optimized portfolio because it reveals how different allocation strategies perform under extreme conditions.

### Robust Optimization

Instead of optimizing a single set of weights, we can generate many portfolios from the VAE, evaluate each under multiple return scenarios, and select the allocation that performs best in the worst case. This max-min approach produces portfolios that are robust to estimation error and model uncertainty.

### Alpha Discovery

The latent space can reveal structure in portfolio space that is not obvious from examining individual assets. Clusters in the latent space correspond to distinct portfolio styles, and the boundaries between clusters may represent novel allocation strategies that blend different investment approaches.

## Rust Implementation

Our Rust implementation provides a complete VAE portfolio generation toolkit with the following components:

### PortfolioVAE

The `PortfolioVAE` struct implements a feed-forward VAE with configurable encoder and decoder architectures. It supports training via stochastic gradient descent with the ELBO objective, generating portfolios by sampling from the latent space, and encoding/decoding existing portfolios. The softmax output layer ensures generated weights are valid probability distributions.

### PortfolioGenerator

The `PortfolioGenerator` struct wraps the VAE and provides high-level portfolio generation methods: unconditional sampling, interpolation between reference portfolios, and latent space optimization for target objectives. It also handles portfolio constraint enforcement (position limits, minimum allocation thresholds).

### PortfolioEvaluator

The `PortfolioEvaluator` struct computes standard portfolio metrics: annualized return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and diversification ratio. These metrics are used both for evaluating generated portfolios and for latent space optimization objectives.

### DirichletSampler

The `DirichletSampler` generates random training portfolios using the Dirichlet distribution. It supports configurable concentration parameters and the ability to generate portfolios with specific characteristics (concentrated, balanced, sector-tilted) for training data augmentation.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint for computing asset returns and covariances. The client supports multiple trading pairs simultaneously, enabling portfolio construction across the cryptocurrency universe.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain real-time market data for portfolio construction:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals for each asset in the portfolio universe. Used to compute return series, covariance matrices, and risk metrics.
- **Multiple symbols**: The client fetches data for all symbols in the portfolio universe (e.g., BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT) and aligns timestamps to construct a clean return matrix.

The Bybit API is well-suited for portfolio generation because it provides:
- Consistent historical data across many trading pairs
- Fine-grained intervals for intraday portfolio rebalancing
- Real-time data for live portfolio generation systems

## References

1. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *Proceedings of the International Conference on Learning Representations (ICLR)*.
2. Higgins, I., et al. (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
3. Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91.
4. Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models. *Advances in Neural Information Processing Systems (NeurIPS)*.
5. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. *Proceedings of the 31st International Conference on Machine Learning (ICML)*.
6. Cong, L. W., Tang, K., Wang, J., & Zhang, Y. (2022). Deep sequence modeling: Development and applications in asset pricing. *The Journal of Financial Data Science*, 3(1), 28-42.
