# Chapter 232: Factor VQ-VAE — Vector Quantized Variational Autoencoders for Discrete Factor Models

## 1. Introduction

Traditional factor models in finance rely on continuous latent representations to describe the forces driving asset returns. While powerful, continuous latent spaces introduce challenges: posterior collapse in standard VAEs, difficulty in interpretation, and a lack of natural clustering structure. Vector Quantized Variational Autoencoders (VQ-VAE), introduced by van den Oord et al. (2017), offer a compelling alternative by learning a **discrete** latent space through vector quantization.

In the context of financial markets, VQ-VAE provides a principled framework for discovering a finite set of market regimes — discrete "states of the world" that summarize complex, high-dimensional market behavior into a compact codebook. Each codebook entry can be thought of as a prototypical market condition: a growth regime, a crisis regime, a low-volatility carry regime, and so on. Rather than forcing a trader to pre-specify these regimes (as with Hidden Markov Models), VQ-VAE learns them directly from data.

This chapter develops the mathematical foundations of VQ-VAE, contrasts it with standard VAEs, and demonstrates a complete Rust implementation applied to cryptocurrency market data from Bybit. By the end, you will be able to train a VQ-VAE that assigns each trading day to one of K learned market regimes and use those assignments for regime-conditional portfolio construction.

## 2. Mathematical Foundation

### 2.1 Vector Quantization

Vector quantization (VQ) maps a continuous vector to its nearest entry in a finite codebook. Given an encoder output $z_e(x) \in \mathbb{R}^D$ and a codebook $\mathcal{C} = \{e_1, e_2, \ldots, e_K\}$ where each $e_k \in \mathbb{R}^D$, the quantization operation is:

$$z_q(x) = e_k, \quad \text{where } k = \arg\min_j \|z_e(x) - e_j\|_2$$

This operation selects the codebook entry closest to the encoder output in Euclidean distance. The index $k$ serves as a discrete latent code that compresses all information about the input $x$ into one of $K$ possible states.

### 2.2 Codebook Learning

The codebook entries must be learned alongside the encoder and decoder. The codebook loss encourages entries to move toward the encoder outputs assigned to them:

$$\mathcal{L}_{\text{codebook}} = \|\text{sg}[z_e(x)] - e_k\|_2^2$$

where $\text{sg}[\cdot]$ denotes the stop-gradient operator. This loss updates only the codebook vectors, not the encoder.

### 2.3 Commitment Loss

The commitment loss encourages the encoder to produce outputs that stay close to the codebook entries, preventing the encoder from growing unboundedly while the codebook tries to catch up:

$$\mathcal{L}_{\text{commit}} = \beta \|z_e(x) - \text{sg}[e_k]\|_2^2$$

where $\beta$ is a hyperparameter (typically 0.25). This loss updates only the encoder, not the codebook.

### 2.4 Straight-Through Estimator

The $\arg\min$ operation in quantization is non-differentiable. VQ-VAE uses the **straight-through estimator**: during the forward pass, the decoder receives the quantized vector $z_q(x)$; during the backward pass, gradients are copied directly from the decoder input to the encoder output, bypassing the quantization step:

$$z_q(x) = z_e(x) + \text{sg}[e_k - z_e(x)]$$

This expression is equivalent to $e_k$ in the forward pass (since $z_e(x) + (e_k - z_e(x)) = e_k$) but has the gradient of $z_e(x)$ in the backward pass (since the stop-gradient term contributes zero gradient).

### 2.5 EMA Codebook Update

Instead of optimizing the codebook loss with gradient descent, Exponential Moving Average (EMA) updates provide more stable training. For each codebook entry $e_k$, we maintain running statistics:

$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1 - \gamma) n_k^{(t)}$$

$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1 - \gamma) \sum_{i \in S_k} z_e(x_i)$$

$$e_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)}}$$

where $n_k^{(t)}$ is the number of encoder outputs assigned to entry $k$ in the current batch, $S_k$ is the set of those encoder outputs, and $\gamma$ is the decay rate (typically 0.99). This is equivalent to an online k-means update and tends to be more stable than gradient-based codebook learning.

### 2.6 Combined Loss

The total VQ-VAE loss is:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \beta \mathcal{L}_{\text{commit}}$$

where $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_2^2$ is the reconstruction loss and $\hat{x}$ is the decoder output.

## 3. VQ-VAE vs Standard VAE

### 3.1 Discrete vs Continuous Latent Space

A standard VAE learns a continuous latent distribution $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ and regularizes it toward a standard normal prior via the KL divergence. VQ-VAE instead learns a categorical distribution over $K$ codebook entries. The key differences are:

| Aspect | Standard VAE | VQ-VAE |
|--------|-------------|--------|
| Latent space | Continuous $\mathbb{R}^D$ | Discrete $\{1, \ldots, K\}$ |
| Prior | $\mathcal{N}(0, I)$ | Uniform or autoregressive |
| Regularization | KL divergence | Commitment loss |
| Sampling | Reparameterization trick | Codebook lookup |
| Interpretability | Difficult | Each code = a regime |

### 3.2 Avoiding Posterior Collapse

Posterior collapse is a well-known failure mode in VAEs where the decoder learns to ignore the latent code, producing the same output regardless of $z$. This happens when the KL term drives $q(z|x)$ toward the uninformative prior. VQ-VAE avoids this entirely because there is no KL term. The discrete bottleneck forces the model to use the codebook, and the commitment loss ensures the encoder actively selects meaningful codes.

### 3.3 Information-Theoretic Perspective

With $K$ codebook entries, VQ-VAE transmits exactly $\log_2 K$ bits of information through the bottleneck. For $K = 8$ (as in our trading application), this is 3 bits per observation — an extreme compression that forces the model to capture only the most salient features of market behavior. This compression acts as a powerful regularizer, preventing overfitting to noise.

## 4. Trading Applications

### 4.1 Discrete Market Regimes as Codebook Entries

Each of the $K$ codebook entries naturally corresponds to a market regime. After training, we can analyze what each regime captures by examining the characteristics of days assigned to each code:

- **Regime 0**: Low volatility, positive drift — calm bull market
- **Regime 1**: High volatility, negative returns — crisis/sell-off
- **Regime 2**: Mean-reverting, moderate volatility — ranging market
- **Regime 3**: Trending with expanding volatility — momentum regime
- ... and so on for all $K$ entries.

The number of codebook entries $K$ is a hyperparameter that controls the granularity of regime classification. Small $K$ (4-8) gives broad market states; large $K$ (32-64) gives fine-grained sub-regimes.

### 4.2 Clustering Market States

Unlike k-means clustering applied directly to features, VQ-VAE learns a nonlinear encoding before clustering. The encoder maps raw market features (returns, volatility, volume, correlations) into a representation space where Euclidean distance is meaningful. This allows VQ-VAE to discover regimes that would be invisible to linear methods.

The codebook usage frequency also provides valuable information. If a codebook entry is rarely used, it may correspond to a tail event. If usage suddenly shifts from one entry to another, this signals a regime change. Monitoring the time series of codebook assignments gives a real-time regime indicator.

### 4.3 Generating Regime-Specific Scenarios

Once trained, VQ-VAE enables regime-conditional scenario generation:

1. Select a target regime $k$
2. Feed the corresponding codebook entry $e_k$ to the decoder
3. Add small perturbations to generate diverse scenarios within that regime
4. Use the generated scenarios for stress testing or Monte Carlo simulation

This is particularly valuable for risk management: "What does my portfolio look like if we enter Regime 1 (crisis mode)?" The decoder produces realistic market scenarios consistent with historical crisis periods, without needing to hand-craft stress scenarios.

### 4.4 Portfolio Construction

Regime assignments enable conditional portfolio strategies:

- Estimate expected returns and covariance matrices for each regime separately
- Use the current regime assignment to select the appropriate estimates
- Construct mean-variance optimal portfolios conditional on the detected regime
- Transition probabilities between regimes inform position sizing and hedging

## 5. Hierarchical VQ-VAE

### 5.1 Multi-Scale Market Representations

Markets operate on multiple time scales simultaneously. Hierarchical VQ-VAE uses multiple levels of quantization to capture this:

- **Level 1 (coarse)**: Broad market regime (bull/bear/sideways) — changes over weeks/months
- **Level 2 (fine)**: Microstructure state (trending/mean-reverting/volatile) — changes over days
- **Level 3 (finest)**: Intraday pattern (momentum/reversal/breakout) — changes over hours

Each level has its own codebook. The coarse level captures slow-moving macro factors while finer levels capture fast-moving tactical signals. The decoder reconstructs the input from all levels, ensuring each level captures complementary information.

### 5.2 Architecture

In hierarchical VQ-VAE, the encoder produces representations at multiple resolutions. The top level encodes global structure; each subsequent level encodes the residual detail not captured by coarser levels. This decomposition naturally separates signal from noise: macro regime information (signal) is captured at coarse levels while day-to-day noise is relegated to fine levels.

### 5.3 Trading with Hierarchical Codes

The multi-level codes enable multi-horizon trading strategies:

- **Strategic allocation** based on Level 1 codes (monthly rebalancing)
- **Tactical tilts** based on Level 2 codes (weekly adjustments)
- **Execution timing** based on Level 3 codes (intraday decisions)

Each level of the hierarchy informs a different part of the investment process, creating a coherent multi-scale trading system.

## 6. Implementation Walkthrough

Our Rust implementation follows the modular structure of the VQ-VAE architecture:

### 6.1 Encoder

The encoder is a simple feedforward network that maps input features (returns, volatility, volume ratios, etc.) to a continuous embedding vector of dimension $D$:

```rust
// Encoder: input_dim -> hidden_dim -> embedding_dim
// Two linear layers with ReLU activation
let hidden = relu(input * W1 + b1);
let z_e = hidden * W2 + b2;
```

### 6.2 Vector Quantization Layer

The VQ layer finds the nearest codebook entry and computes the straight-through estimator:

```rust
// Find nearest codebook entry
let distances = codebook.iter()
    .map(|e| euclidean_distance(&z_e, e))
    .collect();
let k = argmin(&distances);

// Straight-through: forward uses e_k, backward uses z_e
let z_q = z_e + stop_gradient(codebook[k] - z_e);
```

### 6.3 EMA Codebook Update

Instead of gradient-based updates, we use exponential moving averages:

```rust
// Update counts and sums for each codebook entry
ema_count[k] = gamma * ema_count[k] + (1.0 - gamma) * batch_count[k];
ema_sum[k] = gamma * ema_sum[k] + (1.0 - gamma) * batch_sum[k];
codebook[k] = ema_sum[k] / ema_count[k];
```

### 6.4 Decoder

The decoder reconstructs the input from the quantized embedding:

```rust
// Decoder: embedding_dim -> hidden_dim -> input_dim
let hidden = relu(z_q * W3 + b3);
let x_hat = hidden * W4 + b4;
```

### 6.5 Training Loop

The training loop combines reconstruction loss, codebook loss, and commitment loss:

```rust
for epoch in 0..num_epochs {
    let z_e = encoder.forward(&x);
    let (z_q, indices) = vq_layer.quantize(&z_e);
    let x_hat = decoder.forward(&z_q);

    let recon_loss = mse(&x, &x_hat);
    let commit_loss = mse(&z_e, &stop_gradient(&z_q));

    vq_layer.ema_update(&z_e, &indices);
    // Backprop through recon_loss + beta * commit_loss
}
```

The full implementation is in `rust/src/lib.rs` with a trading example in `rust/examples/trading_example.rs`.

## 7. Bybit Data Integration

Our implementation fetches OHLCV data from Bybit's public REST API v5:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=D&limit=200
```

From raw OHLCV data, we compute features for each day:

1. **Log returns**: $r_t = \ln(C_t / C_{t-1})$
2. **Realized volatility**: rolling standard deviation of returns over a window
3. **Volume ratio**: $V_t / \bar{V}$ (volume relative to its moving average)
4. **Range**: $(H_t - L_t) / C_t$ (normalized daily range)
5. **Open-close spread**: $(C_t - O_t) / O_t$ (intraday return)

These features form the input vector for VQ-VAE. The model learns to compress these five dimensions into one of $K = 8$ discrete codes, each representing a distinct market regime.

After training, we analyze each regime by computing the average return, volatility, and volume for days assigned to each codebook entry. This produces an interpretable regime characterization that can inform trading decisions.

## 8. Key Takeaways

1. **VQ-VAE learns discrete market regimes** directly from data, without requiring manual specification of regime definitions or the number of states (beyond choosing $K$).

2. **The discrete bottleneck is a feature, not a limitation.** By forcing all information through $\log_2 K$ bits, VQ-VAE captures only the most important factors driving market behavior, acting as a powerful regularizer.

3. **EMA codebook updates** provide more stable training than gradient-based alternatives and are equivalent to online k-means on the encoder outputs.

4. **The straight-through estimator** enables end-to-end training despite the non-differentiable quantization step. Gradients flow from the decoder through to the encoder as if quantization were an identity function.

5. **Regime-conditional analysis** enables sophisticated trading strategies: estimate separate return distributions for each regime, construct regime-specific portfolios, and generate regime-conditional stress scenarios.

6. **Hierarchical VQ-VAE** captures multi-scale market structure, naturally separating slow-moving macro regimes from fast-moving microstructure states.

7. **Codebook usage patterns** themselves are informative: monitoring which codebook entries are active over time provides a real-time regime change detector.

8. **Compared to standard VAEs**, VQ-VAE avoids posterior collapse, produces interpretable discrete codes, and achieves sharper reconstructions — all desirable properties for financial applications where precision and interpretability matter.

9. **The Rust implementation** demonstrates that VQ-VAE can be implemented efficiently without deep learning frameworks, making it suitable for low-latency trading systems where every microsecond counts.

10. **Integration with Bybit data** shows the complete pipeline from raw market data to actionable regime classifications, closing the gap between academic methods and practical trading systems.
