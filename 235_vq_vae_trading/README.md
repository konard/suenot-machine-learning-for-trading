# Chapter 235: VQ-VAE Trading

## 1. Introduction

Vector Quantized Variational Autoencoders (VQ-VAE) represent a powerful approach to learning discrete representations of complex data. Originally introduced by van den Oord et al. (2017) for image and audio generation, VQ-VAE offers a compelling framework for financial markets: the ability to compress continuous, noisy market data into a finite set of discrete "market tokens."

The core insight is that financial markets, despite their apparent complexity, often exhibit recurring patterns — consolidation phases, breakouts, mean-reversion episodes, momentum surges, and panic sell-offs. VQ-VAE learns to identify these patterns automatically by building a codebook of prototypical market states. Each trading day (or window) gets mapped to its nearest codebook entry, effectively creating a discrete "language" of market behavior.

This discretization has profound implications. Once market data is tokenized, we can apply the full arsenal of discrete sequence modeling — language models, hidden Markov models, n-gram statistics — to predict future market states. We can detect anomalies as days that are far from all codebook entries. We can build interpretable pattern libraries where each codebook entry has a clear market meaning.

In this chapter, we develop a complete VQ-VAE system for trading in Rust, integrating with Bybit exchange data to tokenize cryptocurrency market behavior.

## 2. Mathematical Foundation

### 2.1 VQ-VAE Architecture

The VQ-VAE consists of three components:

**Encoder** $z_e(x) = f_\theta(x)$: Maps input $x \in \mathbb{R}^D$ to a continuous embedding $z_e \in \mathbb{R}^d$.

**Codebook** $\mathcal{E} = \{e_1, e_2, \ldots, e_K\} \subset \mathbb{R}^d$: A set of $K$ learnable embedding vectors (codebook entries).

**Decoder** $\hat{x} = g_\phi(z_q)$: Reconstructs the input from the quantized embedding $z_q$.

### 2.2 Nearest-Neighbor Quantization

The quantization step maps the encoder output to its nearest codebook entry:

$$z_q = e_k, \quad \text{where} \quad k = \arg\min_j \|z_e(x) - e_j\|_2$$

This is a hard assignment — each input maps to exactly one codebook entry. The index $k$ serves as the discrete representation (token) of the input.

### 2.3 Straight-Through Estimator

The argmin operation is non-differentiable. VQ-VAE uses the straight-through estimator to propagate gradients: during the forward pass, $z_q = e_k$ (the nearest codebook entry); during the backward pass, gradients flow directly from the decoder input to the encoder output, bypassing the quantization:

$$\nabla_{z_e} \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$$

This works because if the encoder and codebook are well-trained, $z_e \approx z_q$, so the gradients are approximately correct.

### 2.4 Loss Function

The VQ-VAE loss has three components:

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[z_e(x)] - e_k\|_2^2}_{\text{codebook}} + \underbrace{\beta \|z_e(x) - \text{sg}[e_k]\|_2^2}_{\text{commitment}}$$

where $\text{sg}[\cdot]$ denotes the stop-gradient operator:

- **Reconstruction loss**: Trains encoder and decoder to faithfully reconstruct the input.
- **Codebook loss**: Moves codebook entries toward encoder outputs (only updates codebook).
- **Commitment loss**: Encourages encoder outputs to stay close to codebook entries (only updates encoder). $\beta$ is typically set to 0.25.

### 2.5 Exponential Moving Average (EMA) Updates

Instead of optimizing the codebook loss with gradient descent, EMA updates are more stable:

$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1 - \gamma) n_k^{(t)}$$

$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1 - \gamma) \sum_{i \in S_k} z_e(x_i)$$

$$e_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)}}$$

where $n_k^{(t)}$ is the number of encoder outputs assigned to codebook entry $k$ at time $t$, $S_k$ is the set of those inputs, and $\gamma$ is the decay rate (typically 0.99).

### 2.6 Perplexity Metric

To measure codebook utilization, we compute the perplexity of the codebook usage distribution:

$$\text{Perplexity} = \exp\left(-\sum_{k=1}^{K} p_k \log p_k\right)$$

where $p_k$ is the fraction of inputs assigned to codebook entry $k$. A perplexity of $K$ means all codes are used equally; a low perplexity indicates codebook collapse.

## 3. VQ-VAE vs. Standard VAE

### 3.1 Discrete vs. Continuous Latents

Standard VAEs learn a continuous latent space $z \sim \mathcal{N}(\mu, \sigma^2)$. VQ-VAE learns a discrete latent space $z \in \{e_1, \ldots, e_K\}$. This distinction has major implications:

| Property | VAE | VQ-VAE |
|----------|-----|--------|
| Latent space | Continuous | Discrete |
| Regularization | KL divergence | Commitment loss |
| Posterior | Gaussian | Categorical (deterministic) |
| Generation | Sample from prior | Autoregressive over codes |
| Interpretability | Low | High (each code = pattern) |

### 3.2 Avoiding Posterior Collapse

A notorious problem with VAEs is posterior collapse: the decoder becomes so powerful that it ignores the latent code, and the encoder posterior collapses to the prior. This means the latent space carries no information.

VQ-VAE avoids this by construction. The decoder receives a discrete code from a finite codebook — if the decoder ignores this input, it can only produce the average output, resulting in high reconstruction loss. The discrete bottleneck forces meaningful information through the latent space.

### 3.3 Information Utilization

In standard VAEs, the "bits back" argument shows that the latent code carries at most $\text{KL}(q(z|x) \| p(z))$ nats of information. VQ-VAE's latent carries exactly $\log_2 K$ bits per code (when all codes are used equally). With $K = 512$ codebook entries, this is 9 bits per code — enough to distinguish 512 distinct market patterns.

## 4. Trading Applications

### 4.1 Tokenizing Market Data

The most direct application of VQ-VAE in trading is converting continuous market data into discrete tokens. Given a window of OHLCV data (e.g., 5 consecutive days), the VQ-VAE maps it to a single codebook index. This creates a sequence of tokens:

$$\text{market data} \rightarrow [t_{23}, t_{7}, t_{45}, t_{12}, t_{7}, t_{31}, \ldots]$$

where each $t_i$ is an index into the learned codebook. This sequence can then be analyzed with discrete sequence models.

### 4.2 Building "Market Language Models"

Once we have token sequences, we can train autoregressive models to predict the next token:

$$P(t_{n+1} | t_1, t_2, \ldots, t_n)$$

This is analogous to language modeling, where we predict the next word. The "vocabulary" is our codebook, and the "sentences" are sequences of market states. N-gram models, LSTMs, or Transformers can all be applied.

If the codebook has learned meaningful patterns (e.g., entry 23 = "strong bullish breakout"), then predicting "the next market state is entry 23 with 40% probability" provides directly actionable intelligence.

### 4.3 Pattern Library Construction

Each codebook entry can be visualized and interpreted by examining:
- The decoder output $g_\phi(e_k)$: what "ideal" pattern does this entry represent?
- The set of actual market days assigned to entry $k$: what real data looks like this?
- Statistics of each entry: average return, volatility, volume characteristics.

This creates an automatic, data-driven pattern library. Unlike hand-crafted technical patterns (head-and-shoulders, cup-and-handle), VQ-VAE discovers patterns that are statistically grounded and optimized for reconstruction fidelity.

### 4.4 Anomaly Detection via Codebook Distance

A powerful application is anomaly detection. For each input $x$, the distance to its nearest codebook entry measures how well-represented $x$ is:

$$d(x) = \min_k \|z_e(x) - e_k\|_2$$

If $d(x)$ is large, the input is unlike anything in the codebook — it represents a genuinely novel market condition. This can serve as an early warning system: when the market enters uncharted territory, the codebook distance spikes.

We can set anomaly thresholds based on historical distance distributions. Days exceeding the 95th or 99th percentile of codebook distances are flagged as anomalous, triggering risk reduction or special attention.

## 5. PixelCNN-Style Prior Over Discrete Codes

### 5.1 Autoregressive Prior

VQ-VAE trains the encoder-decoder without any prior over the latent codes. To generate new data or compute likelihoods, we need a separate prior model $p(z_1, z_2, \ldots, z_T)$ over the discrete code sequences.

PixelCNN (van den Oord et al., 2016) provides a natural autoregressive factorization:

$$p(z_1, \ldots, z_T) = \prod_{t=1}^{T} p(z_t | z_1, \ldots, z_{t-1})$$

Each factor $p(z_t | z_{<t})$ is modeled as a categorical distribution over $K$ codebook entries, parameterized by a causal neural network (masked convolutions or Transformer).

### 5.2 Application to Trading

In the trading context, this prior captures the temporal dynamics of market states. Training it on historical token sequences allows us to:

1. **Forecast**: Compute $P(z_{T+1} = k | z_1, \ldots, z_T)$ for each codebook entry $k$, giving a probabilistic forecast over future market states.
2. **Simulate**: Sample entire trajectories $z_1, \ldots, z_T$ from the prior, decode them, and obtain realistic synthetic market scenarios for risk analysis.
3. **Score**: Compute the log-likelihood $\log p(z_1, \ldots, z_T)$ of observed sequences to detect regime changes (sudden drops in likelihood).

### 5.3 Implementation Notes

For a practical trading system, a simple n-gram or small Transformer prior works well:
- **Bigram**: $p(z_t | z_{t-1})$ — a $K \times K$ transition matrix
- **Trigram**: $p(z_t | z_{t-2}, z_{t-1})$ — a $K \times K \times K$ tensor
- **Transformer**: Full context window, learned positional encodings

The bigram prior is especially interpretable: each entry in the transition matrix tells us the probability of transitioning from one market state to another.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete VQ-VAE system optimized for trading data. Here we walk through the key components.

### 6.1 Encoder

The encoder is a simple feedforward network that maps input features (OHLCV data) to a continuous embedding:

```rust
pub struct Encoder {
    pub weights1: Array2<f64>,  // input_dim x hidden_dim
    pub biases1: Array1<f64>,
    pub weights2: Array2<f64>,  // hidden_dim x embedding_dim
    pub biases2: Array1<f64>,
}
```

We use ReLU activations. The encoder compresses, say, a 5-day OHLCV window (25 features) into an embedding vector of dimension $d$ (e.g., 16).

### 6.2 Codebook and Quantization

The codebook stores $K$ embedding vectors. Quantization finds the nearest one:

```rust
pub fn quantize(&self, z_e: &Array1<f64>) -> (Array1<f64>, usize) {
    let mut min_dist = f64::MAX;
    let mut min_idx = 0;
    for (i, entry) in self.embeddings.iter().enumerate() {
        let dist = (z_e - entry).mapv(|x| x * x).sum();
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }
    (self.embeddings[min_idx].clone(), min_idx)
}
```

EMA updates move codebook entries toward their assigned encoder outputs, ensuring stable training without a separate optimizer for the codebook.

### 6.3 Loss Computation

The combined loss drives training:

```rust
pub fn compute_loss(
    &self, x: &Array1<f64>, x_hat: &Array1<f64>,
    z_e: &Array1<f64>, z_q: &Array1<f64>, beta: f64,
) -> (f64, f64, f64) {
    let recon_loss = (x - x_hat).mapv(|v| v * v).sum();
    let codebook_loss = (z_e - z_q).mapv(|v| v * v).sum(); // sg on z_e
    let commitment_loss = (z_e - z_q).mapv(|v| v * v).sum(); // sg on z_q
    (recon_loss, codebook_loss, beta * commitment_loss)
}
```

The stop-gradient operation is applied during backpropagation (not shown in the loss calculation itself).

### 6.4 Anomaly Detection

After training, anomaly detection is straightforward — compute the minimum codebook distance for each new data point and compare against historical thresholds:

```rust
pub fn anomaly_score(&self, x: &Array1<f64>) -> f64 {
    let z_e = self.encoder.forward(x);
    let (_, _, min_dist) = self.codebook.quantize_with_distance(&z_e);
    min_dist
}
```

## 7. Bybit Data Integration

Our implementation fetches real market data from Bybit's public API. The endpoint `https://api.bybit.com/v5/market/kline` provides historical OHLCV data for cryptocurrency pairs.

### 7.1 Data Fetching

We fetch daily candles for BTCUSDT and normalize the data:
- **Open, High, Low, Close**: Percentage changes from the previous close
- **Volume**: Log-transformed and z-scored

This normalization ensures that the VQ-VAE learns shape patterns rather than absolute price levels.

### 7.2 Windowing

We create sliding windows of consecutive days (e.g., 5 days), flattening each window into a single feature vector. Each window becomes one data point for the VQ-VAE:

$$x = [\Delta o_1, \Delta h_1, \Delta l_1, \Delta c_1, v_1, \ldots, \Delta o_5, \Delta h_5, \Delta l_5, \Delta c_5, v_5]$$

### 7.3 Analysis Pipeline

After training, the pipeline:
1. Assigns each window to a codebook entry
2. Computes codebook usage statistics and perplexity
3. Identifies anomalous windows with high codebook distances
4. Prints the token sequence for downstream modeling

## 8. Key Takeaways

1. **VQ-VAE discretizes market data** into a finite set of learned patterns (codebook entries), creating a "vocabulary" of market states.

2. **Discrete representations avoid posterior collapse** — a common failure mode of standard VAEs where the latent space is ignored.

3. **EMA codebook updates** are more stable than gradient-based optimization and don't require a separate learning rate.

4. **Codebook perplexity** measures how many codes are actively used. Low perplexity indicates codebook collapse; aim for perplexity close to $K$.

5. **Anomaly detection via codebook distance** provides a principled way to detect novel market conditions — days that don't match any learned pattern.

6. **Token sequences enable language modeling** — once market data is tokenized, autoregressive models can predict future market states.

7. **The pattern library is interpretable** — each codebook entry can be decoded and analyzed to understand what market pattern it represents.

8. **Commitment loss weight $\beta$** controls the trade-off between reconstruction quality and codebook utilization. Typical values range from 0.1 to 1.0.

9. **Codebook size $K$** determines the granularity of discretization. Too few entries lose detail; too many lead to underutilization. Start with 32-128 for daily market data.

10. **VQ-VAE is a foundation** for more advanced models. The discrete codes can feed into Transformers, HMMs, or reinforcement learning agents as compact state representations.

## References

- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. NeurIPS.
- van den Oord, A., et al. (2016). Conditional Image Generation with PixelCNN Decoders. NeurIPS.
- Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS.
- Roy, A., et al. (2018). Theory and Experiments on Vector Quantized Autoencoders. arXiv:1805.11063.
