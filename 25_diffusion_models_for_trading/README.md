# Diffusion Models for Synthetic Time Series and Forecasting

This chapter introduces **diffusion models** for financial time series applications. Originally developed for image generation (Stable Diffusion, DALL-E), diffusion models have emerged as a powerful alternative to GANs for generating synthetic financial data and probabilistic forecasting.

<p align="center">
<img src="https://i.imgur.com/YqKxZ8N.png" width="70%">
</p>

## Content

1. [Diffusion Models: From Images to Time Series](#diffusion-models-from-images-to-time-series)
    * [The Intuition Behind Diffusion](#the-intuition-behind-diffusion)
    * [Forward Process: Adding Noise](#forward-process-adding-noise)
    * [Reverse Process: Denoising](#reverse-process-denoising)
    * [Noise Schedules](#noise-schedules)
2. [Key Architectures for Time Series](#key-architectures-for-time-series)
    * [TimeGrad: Autoregressive Diffusion](#timegrad-autoregressive-diffusion)
    * [CSDI: Conditional Score-based Diffusion](#csdi-conditional-score-based-diffusion)
    * [Diffusion-TS: Decomposed Representations](#diffusion-ts-decomposed-representations)
    * [TimeDiff and Recent Advances (2024-2025)](#timediff-and-recent-advances-2024-2025)
3. [Code Examples](#code-examples)
    * [01: Diffusion Fundamentals](#01-diffusion-fundamentals)
    * [02: DDPM Implementation from Scratch](#02-ddpm-implementation-from-scratch)
    * [03: TimeGrad for Cryptocurrency Forecasting](#03-timegrad-for-cryptocurrency-forecasting)
    * [04: CSDI for Imputation and Forecasting](#04-csdi-for-imputation-and-forecasting)
    * [05: Diffusion-TS for Synthetic Data Generation](#05-diffusion-ts-for-synthetic-data-generation)
    * [06: Diffusion vs GANs Comparison](#06-diffusion-vs-gans-comparison)
    * [07: Complete Bitcoin Forecasting Pipeline](#07-complete-bitcoin-forecasting-pipeline)
4. [Rust Implementation](#rust-implementation)
5. [Practical Considerations](#practical-considerations)
6. [Resources](#resources)

## Diffusion Models: From Images to Time Series

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. Unlike GANs, which learn through adversarial training, diffusion models learn to **denoise** data step by step.

### The Intuition Behind Diffusion

The key insight is that it's easier to learn small denoising steps than to generate data in one shot:

1. **Forward Process**: Gradually add Gaussian noise to data until it becomes pure noise
2. **Reverse Process**: Learn to undo each noise step, recovering the original data

This approach offers several advantages over GANs:
- **Training stability**: No adversarial dynamics, mode collapse is rare
- **Quality**: Often produces higher-quality samples
- **Uncertainty quantification**: Natural probabilistic interpretation

### Forward Process: Adding Noise

The forward diffusion process is a Markov chain that gradually adds Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Where:
- $x_0$ is the original data
- $x_t$ is the noised data at step $t$
- $\beta_t$ is the noise schedule (typically $\beta_t \in [0.0001, 0.02]$)
- $T$ is the total number of diffusion steps (typically 1000)

A key property allows direct sampling at any timestep:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s)$

### Reverse Process: Denoising

The reverse process learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

In practice, we train a neural network $\epsilon_\theta(x_t, t)$ to predict the noise added at each step. The training objective is:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### Noise Schedules

The choice of noise schedule $\beta_t$ significantly affects performance:

| Schedule | Formula | Characteristics |
|----------|---------|-----------------|
| **Linear** | $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$ | Simple, works well for images |
| **Cosine** | $\bar{\alpha}_t = \frac{f(t)}{f(0)}$, $f(t) = \cos^2(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2})$ | Better for smaller images/sequences |
| **Sigmoid** | $\beta_t = \sigma(-6 + 12\frac{t}{T})$ | Smooth transitions |

## Key Architectures for Time Series

### TimeGrad: Autoregressive Diffusion

[TimeGrad](https://arxiv.org/abs/2101.12072) (Rasul et al., 2021) combines autoregressive modeling with diffusion:

```
Input: x_{1:t} (historical observations)
├── RNN Encoder → Hidden state h_t
├── Diffusion Process conditioned on h_t
└── Output: p(x_{t+1:t+τ} | x_{1:t})
```

**Key features**:
- Uses RNN/LSTM to encode history
- Diffusion generates future conditioned on hidden state
- Autoregressive: generates one step at a time

**Limitations**: Cumulative errors in autoregressive generation, slow inference.

### CSDI: Conditional Score-based Diffusion

[CSDI](https://github.com/ermongroup/CSDI) (Tashiro et al., NeurIPS 2021) uses attention-based conditioning:

```
Input: Partially observed time series with mask
├── Temporal Attention (across time)
├── Feature Attention (across variables)
├── Score-based diffusion conditioned on observed values
└── Output: Imputed/forecasted values with uncertainty
```

**Key features**:
- Self-supervised training with random masking
- Handles both imputation and forecasting
- **40-65% improvement** over existing probabilistic methods
- Generates probabilistic forecasts (multiple samples)

### Diffusion-TS: Decomposed Representations

[Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) (ICLR 2024) introduces interpretable decomposition:

```
Input: Time series x
├── Encoder-Decoder Transformer
├── Decomposition:
│   ├── Trend: Polynomial regression
│   └── Seasonal: Fourier series
├── Diffusion in decomposed space
└── Output: Interpretable synthetic series
```

**Key features**:
- Reconstructs samples directly (not noise)
- Fourier-based loss for spectral accuracy
- Same architecture for generation, forecasting, imputation
- **State-of-the-art** on Stocks, Energy, ETTh datasets

### TimeDiff and Recent Advances (2024-2025)

Recent developments have addressed key limitations:

| Method | Innovation | Performance |
|--------|------------|-------------|
| **TimeDiff** | Conditional diffusion at past-future boundary | 9-47% improvement over baselines |
| **ARMD** | Auto-regressive moving diffusion | Best MSE on Exchange Rates |
| **SimDiff** (2025) | Simplified architecture, faster inference | Competitive with 10x fewer params |
| **MG-TSD** | Multi-granularity temporal structures | SOTA on long-term forecasting |
| **S²DBM** | Brownian Bridge dynamics | Natural boundary conditions |

## Code Examples

### 01: Diffusion Fundamentals
The notebook [01_diffusion_fundamentals.ipynb](01_diffusion_fundamentals.ipynb) covers:
- Forward and reverse diffusion processes
- Noise schedules visualization
- ELBO derivation and loss functions
- Simple 1D diffusion example

### 02: DDPM Implementation from Scratch
The notebook [02_ddpm_from_scratch.ipynb](02_ddpm_from_scratch.ipynb) implements:
- Complete DDPM training loop
- U-Net architecture for time series
- Sampling algorithms (DDPM, DDIM)
- Visualization of denoising process

### 03: TimeGrad for Cryptocurrency Forecasting
The notebook [03_timegrad_crypto.ipynb](03_timegrad_crypto.ipynb) demonstrates:
- TimeGrad architecture with RNN encoder
- Training on Bitcoin/Ethereum hourly data
- Probabilistic forecasting with confidence intervals
- Comparison with LSTM baselines

### 04: CSDI for Imputation and Forecasting
The notebook [04_csdi_imputation_forecasting.ipynb](04_csdi_imputation_forecasting.ipynb) shows:
- CSDI implementation with attention
- Missing data imputation in OHLCV series
- Probabilistic forecasting
- Evaluation metrics (CRPS, calibration)

### 05: Diffusion-TS for Synthetic Data Generation
The notebook [05_diffusion_ts_synthetic.ipynb](05_diffusion_ts_synthetic.ipynb) covers:
- Generating synthetic cryptocurrency data
- Trend-seasonal decomposition
- Quality evaluation (discriminative score, FID)
- Comparison with TimeGAN (Chapter 21)

### 06: Diffusion vs GANs Comparison
The notebook [06_diffusion_vs_gans.ipynb](06_diffusion_vs_gans.ipynb) compares:
- Training stability (diffusion vs GAN)
- Sample quality metrics
- Diversity vs fidelity trade-offs
- Computational requirements

### 07: Complete Bitcoin Forecasting Pipeline
The notebook [07_bitcoin_pipeline.ipynb](07_bitcoin_pipeline.ipynb) provides:
- End-to-end production pipeline
- Technical indicator features (RSI, volatility)
- Monte Carlo uncertainty estimation
- Backtesting with realistic constraints

## Rust Implementation

The [rust_diffusion_crypto](rust_diffusion_crypto/) directory contains a Rust implementation using the `tch-rs` (PyTorch bindings) and `burn` frameworks:

```
rust_diffusion_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── data/           # Bybit API client, preprocessing
│   ├── model/          # DDPM, U-Net, noise schedules
│   ├── training/       # Training loop, losses
│   └── utils/          # Config, checkpoints
└── examples/
    ├── fetch_data.rs
    ├── train_ddpm.rs
    └── forecast.rs
```

See [rust_diffusion_crypto/README.md](rust_diffusion_crypto/README.md) for detailed usage.

## Practical Considerations

### When to Use Diffusion Models

**Good use cases**:
- Synthetic data generation for backtesting
- Probabilistic forecasting with uncertainty
- Missing data imputation
- Scenario generation for risk analysis
- When you need diversity in generated samples

**Not ideal for**:
- Real-time/low-latency predictions (slow inference)
- Limited computational resources
- When model interpretability is critical
- Very short sequences (<24 timesteps)

### Computational Requirements

| Task | GPU Memory | Training Time | Inference Time |
|------|------------|---------------|----------------|
| Simple DDPM | 4GB | 2-4 hours | 100ms/sample |
| TimeGrad | 8GB | 8-12 hours | 500ms/sample |
| CSDI | 12GB | 12-24 hours | 200ms/sample |
| Diffusion-TS | 8GB | 6-10 hours | 150ms/sample |

### Optimization Techniques

1. **DDIM Sampling**: Reduce steps from 1000 to 50-100 with minimal quality loss
2. **Token Merging**: [tomesd](https://github.com/dbolya/tomesd) speeds up 1.24x
3. **Distillation**: Train smaller student models
4. **Caching**: Cache attention computations across steps

## Resources

### Papers

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239), Ho et al., 2020
- [TimeGrad: Autoregressive Denoising Diffusion for Time Series](https://arxiv.org/abs/2101.12072), Rasul et al., 2021
- [CSDI: Conditional Score-based Diffusion for Imputation](https://arxiv.org/abs/2107.03502), Tashiro et al., NeurIPS 2021
- [Diffusion-TS: Interpretable Diffusion for Time Series](https://openreview.net/forum?id=4h1apFjO99), ICLR 2024
- [Diffusion Models for Time Series Forecasting: A Survey](https://arxiv.org/abs/2401.03006), Meijer et al., 2024
- [Generation of Synthetic Financial Time Series by Diffusion](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2528697), 2025

### Implementations

- [ermongroup/CSDI](https://github.com/ermongroup/CSDI) - Official CSDI
- [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) - Diffusion-TS
- [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion) - TSDiff
- [GavinKerrworking/TimeGrad](https://github.com/GavinKerrworking/TimeGrad) - TimeGrad implementation

### Tutorials & Guides

- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) - Hugging Face
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Lilian Weng
- [Diffusion Models in Time-Series Forecasting](https://www.emergentmind.com/topics/diffusion-models-in-time-series-forecasting)

### Related Chapters

- [Chapter 19: RNN for Multivariate Time Series](../19_recurrent_neural_nets) - LSTM/GRU foundations
- [Chapter 20: Autoencoders for Risk Factors](../20_autoencoders_for_conditional_risk_factors) - Latent space models
- [Chapter 21: GANs for Synthetic Time Series](../21_gans_for_synthetic_time_series) - Alternative generative approach
