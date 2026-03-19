# Chapter 282: Contrastive Self-Supervised Pretraining for Trading

## 1. Introduction

Financial time series present a fundamental challenge for supervised learning: labeled data is scarce, expensive to obtain, and often subjective. A human analyst might label a period as a "bull regime" while another calls it "consolidation." Contrastive self-supervised pretraining sidesteps this problem entirely by learning useful representations from raw, unlabeled market data.

The core insight is simple yet powerful: if we can teach a model to recognize when two differently-augmented views of the same market window are "the same" and when two different windows are "different," the model implicitly learns to capture the underlying market structure -- volatility regimes, trend patterns, mean-reversion dynamics -- without ever seeing a single label.

This chapter explores three major contrastive learning frameworks -- SimCLR, MoCo, and BYOL -- adapted specifically for financial time series. We develop augmentation strategies that preserve the semantic content of price data, implement the full pipeline in Rust for production-grade performance, and demonstrate downstream applications including few-shot regime detection and anomaly detection using real Bybit cryptocurrency data.

### Why Contrastive Learning for Finance?

Traditional financial ML workflows suffer from several problems that contrastive pretraining directly addresses:

1. **Label scarcity**: Market regime labels require expert annotation. Contrastive learning needs zero labels.
2. **Non-stationarity**: Markets change over time. Pretrained representations transfer better across regimes than hand-crafted features.
3. **Data efficiency**: After pretraining, downstream tasks can work with as few as 10-50 labeled examples.
4. **Feature quality**: Contrastive representations capture market microstructure that hand-designed features miss.

## 2. Mathematical Foundations

### 2.1 The Contrastive Learning Framework

Given an unlabeled dataset of market windows $\{x_1, x_2, \ldots, x_N\}$, contrastive learning proceeds in four steps:

1. **Augmentation**: For each sample $x_i$, generate two augmented views $\tilde{x}_i^{(1)}$ and $\tilde{x}_i^{(2)}$ using stochastic transformations $t \sim \mathcal{T}$.
2. **Encoding**: Pass both views through an encoder $f_\theta$ to obtain representations $h_i^{(1)} = f_\theta(\tilde{x}_i^{(1)})$ and $h_i^{(2)} = f_\theta(\tilde{x}_i^{(2)})$.
3. **Projection**: Map representations through a projection head $g_\phi$ to get $z_i^{(1)} = g_\phi(h_i^{(1)})$ and $z_i^{(2)} = g_\phi(h_i^{(2)})$.
4. **Contrastive loss**: Pull together positive pairs $(z_i^{(1)}, z_i^{(2)})$ while pushing apart negative pairs.

### 2.2 SimCLR: Simple Contrastive Learning of Representations

SimCLR uses the Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss. For a minibatch of $N$ samples yielding $2N$ augmented views, define the similarity:

$$\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\| \|z_j\|}$$

The NT-Xent loss for a positive pair $(i, j)$ is:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $\tau$ is a temperature hyperparameter controlling the sharpness of the distribution. The total loss averages over all positive pairs in the batch:

$$\mathcal{L}_{\text{SimCLR}} = \frac{1}{2N} \sum_{k=1}^{N} [\ell_{2k-1, 2k} + \ell_{2k, 2k-1}]$$

**Key properties:**
- Larger batch sizes provide more negative pairs, improving representation quality.
- Temperature $\tau$ controls the penalty for hard negatives: lower $\tau$ focuses on the hardest negatives.
- The projection head $g_\phi$ is discarded after pretraining; only $f_\theta$ is used downstream.

### 2.3 MoCo: Momentum Contrast

MoCo addresses SimCLR's need for large batch sizes by maintaining a momentum-updated queue of negative examples. It uses two encoders:

- **Query encoder** $f_q$ with parameters $\theta_q$, updated by gradient descent.
- **Key encoder** $f_k$ with parameters $\theta_k$, updated via exponential moving average:

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

where $m \in [0.99, 0.999]$ is the momentum coefficient. The InfoNCE loss is:

$$\mathcal{L}_{\text{MoCo}} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^- \in \text{queue}} \exp(q \cdot k^- / \tau)}$$

The queue stores encoded keys from previous minibatches, decoupling the number of negatives from the batch size. For financial data, this is particularly useful since we can maintain a queue spanning different market regimes.

### 2.4 BYOL: Bootstrap Your Own Latent

BYOL eliminates negative pairs entirely, using only positive pairs with an asymmetric architecture:

- **Online network**: encoder $f_\theta$, projector $g_\theta$, predictor $q_\theta$
- **Target network**: encoder $f_\xi$, projector $g_\xi$ (no predictor), updated via momentum

The loss is the mean squared error between the L2-normalized prediction and the target:

$$\mathcal{L}_{\text{BYOL}} = \left\| \frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|_2} - \frac{z_\xi}{\|z_\xi\|_2} \right\|_2^2$$

BYOL avoids collapse through the asymmetry between online and target networks. For financial applications, BYOL is appealing because it does not require careful negative sampling, which can be problematic when market windows overlap temporally.

### 2.5 Projection Head Architecture

The projection head $g_\phi$ is a 2-layer MLP:

$$g_\phi(h) = W_2 \cdot \sigma(W_1 \cdot h + b_1) + b_2$$

where $\sigma$ is a non-linearity (ReLU or GELU). Research shows that representations $h$ (before projection) outperform projections $z$ on downstream tasks. The projection head acts as a "information bottleneck" that strips away details not useful for the contrastive objective, preserving more general features in $h$.

## 3. Augmentation Strategies for Financial Time Series

The quality of contrastive representations depends critically on the augmentation strategy. Unlike images where rotations and color jitter are standard, financial time series require domain-specific augmentations that preserve market semantics.

### 3.1 Time Masking

Randomly mask contiguous segments of the time series by replacing them with zeros or the mean value:

$$\tilde{x}_t = \begin{cases} 0 & \text{if } t \in [t_s, t_s + L] \\ x_t & \text{otherwise} \end{cases}$$

where $t_s$ is a random start point and $L$ is the mask length (typically 5-20% of the window). This forces the encoder to infer masked dynamics from context, similar to masked language modeling.

### 3.2 Magnitude Scaling

Scale the values by a random factor drawn from a narrow range:

$$\tilde{x}_t = \alpha \cdot x_t, \quad \alpha \sim \text{Uniform}(0.8, 1.2)$$

This teaches the encoder to be invariant to absolute price levels while preserving relative patterns -- a desirable property since the same chart pattern at different price scales should have the same representation.

### 3.3 Gaussian Jittering

Add small Gaussian noise:

$$\tilde{x}_t = x_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

where $\sigma$ is calibrated to be small relative to the data's standard deviation (typically $\sigma = 0.01 \cdot \text{std}(x)$). This provides robustness to microstructure noise and bid-ask bounce effects.

### 3.4 Temporal Warping

Apply non-linear time distortion to simulate different sampling rates:

$$\tilde{x}_{t'} = x_{t'(t)}, \quad t'(t) = t + \delta \cdot \sin(2\pi t / T)$$

with interpolation for non-integer indices. This captures the intuition that the same market pattern can unfold at different speeds.

### 3.5 Channel Dropout

For multi-feature inputs (OHLCV), randomly drop one or more channels:

$$\tilde{x}_t^{(c)} = \begin{cases} 0 & \text{with probability } p \\ x_t^{(c)} & \text{otherwise} \end{cases}$$

This forces the encoder to not rely on any single feature, improving robustness.

### Composition Strategy

Effective augmentation composes multiple transforms. For financial data, a strong composition is:

$$\mathcal{T} = \text{TimeMask} \circ \text{MagnitudeScale} \circ \text{GaussianJitter}$$

Each augmentation is applied with probability 0.5, giving $2^3 = 8$ possible combinations.

## 4. Downstream Applications

### 4.1 Few-Shot Regime Detection

After pretraining, freeze the encoder $f_\theta$ and train a linear classifier on a small labeled dataset:

$$\hat{y} = \text{softmax}(W \cdot f_\theta(x) + b)$$

With as few as 20 labeled examples per regime, the contrastive encoder can detect bull/bear/sideways regimes with 75-85% accuracy, compared to 50-60% for a randomly-initialized network with the same amount of labeled data.

### 4.2 Anomaly Detection

Compute representations for a reference set of "normal" market windows. Flag new windows as anomalous if their representation falls far from the reference distribution:

$$\text{anomaly\_score}(x) = \min_{x_r \in \text{ref}} \|f_\theta(x) - f_\theta(x_r)\|_2$$

This captures flash crashes, unusual volatility expansions, and regime transitions without explicit anomaly labels.

### 4.3 Similarity Search

Find historical periods most similar to the current market state:

$$x^* = \arg\min_{x_h \in \text{history}} \|f_\theta(x_{\text{current}}) - f_\theta(x_h)\|_2$$

This enables analogy-based trading: "the current market looks most like March 2020" and act accordingly.

## 5. Rust Implementation

Our Rust implementation provides a complete contrastive learning pipeline suitable for production deployment. The key components are:

- **`Augmenter`**: Configurable augmentation pipeline supporting time masking, scaling, jittering, and composition.
- **`Encoder`**: A simple feedforward encoder that maps raw time series windows to fixed-size representations.
- **`ProjectionHead`**: Two-layer MLP for projecting representations into contrastive space.
- **`NtXentLoss`**: Differentiable NT-Xent contrastive loss computation.
- **`MomentumEncoder`**: MoCo-style momentum-updated encoder with a queue of negative examples.
- **`LinearProbe`**: Simple linear classifier for evaluating representation quality on downstream tasks.
- **`BybitClient`**: Async client for fetching real-time and historical kline data from the Bybit exchange.

The implementation uses `ndarray` for matrix operations, providing zero-copy views and efficient BLAS-backed linear algebra. All components are modular and composable.

### Architecture Decision: Why Not Autograd?

Rust lacks a mature automatic differentiation ecosystem comparable to PyTorch. Our implementation computes forward passes and losses analytically, with manual gradient derivations for the linear probe. For full training, one would interface with a Python training loop or use `tch-rs` (Rust bindings for libtorch). The Rust code here focuses on inference, data processing, and the mathematical primitives.

See `rust/src/lib.rs` for the full implementation with 5+ unit tests, and `rust/examples/trading_example.rs` for a complete end-to-end workflow using Bybit data.

## 6. Bybit Data Integration

We use the Bybit V5 API to fetch historical kline (candlestick) data. The endpoint:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=15&limit=200
```

Returns OHLCV data that we preprocess into windows for contrastive learning:

1. **Fetch** 200 klines of 15-minute BTCUSDT data.
2. **Normalize** using z-score normalization per window.
3. **Window** into overlapping segments of 20 bars each.
4. **Augment** each window to create positive pairs.
5. **Encode** and compute contrastive loss to verify the pipeline.

The Bybit client handles rate limiting and error recovery, making it suitable for continuous data collection in a live trading system.

## 7. Key Takeaways

1. **Contrastive pretraining learns market structure without labels.** By training a model to recognize augmented views of the same market window, we obtain representations that capture regimes, volatility patterns, and trend dynamics.

2. **Augmentation design is critical.** Financial augmentations must preserve semantic content: time masking, magnitude scaling, and Gaussian jitter are effective; random shuffling or large-scale warping destroys the signal.

3. **SimCLR is simplest, MoCo is memory-efficient, BYOL avoids negative sampling.** Choose based on your infrastructure: SimCLR for large-batch GPU training, MoCo for limited memory, BYOL when negative sampling is problematic.

4. **The projection head is used only during pretraining.** Downstream tasks use the encoder representations directly, which contain richer information than the projected space.

5. **Few-shot learning becomes viable.** With contrastive pretraining, regime detection works with 20-50 labeled examples, reducing the need for expensive expert annotation.

6. **Anomaly detection is a natural downstream task.** The learned representation space naturally clusters similar market states, making distance-based anomaly detection highly effective.

7. **Rust provides production-grade performance.** The modular implementation handles data fetching, augmentation, encoding, and evaluation with the safety and speed guarantees of Rust.

8. **Temperature tuning matters.** The NT-Xent temperature $\tau$ controls the difficulty of the contrastive task. For financial data, $\tau \in [0.05, 0.1]$ typically works well, focusing the loss on the hardest negatives.

---

## References

- Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR). ICML.
- He, K. et al. (2020). "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo). CVPR.
- Grill, J.-B. et al. (2020). "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (BYOL). NeurIPS.
- Yue, Z. et al. (2022). "TS2Vec: Towards Universal Representation of Time Series." AAAI.
- Eldele, E. et al. (2021). "Time-Series Representation Learning via Temporal and Contextual Contrasting." IJCAI.
