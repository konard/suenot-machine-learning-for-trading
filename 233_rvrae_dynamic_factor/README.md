# Chapter 233: RVRAE Dynamic Factor Model

## 1. Introduction

The Recurrent Variational Recurrent Autoencoder (RVRAE) represents a significant advance in dynamic factor modeling for financial markets. Traditional factor models assume that both factor loadings and the factors themselves are static over time, or at best change according to pre-specified windows. This assumption breaks down in real financial markets where regimes shift, correlations evolve, and the underlying drivers of asset returns are fundamentally non-stationary.

RVRAE combines the generative modeling power of Variational Autoencoders (VAEs) with the sequential processing capabilities of Recurrent Neural Networks (RNNs). The result is a model that learns time-varying latent factors directly from multivariate financial time series, capturing how the hidden structure of the market evolves through time. Unlike static PCA or standard VAE approaches that treat each observation independently, RVRAE respects the temporal ordering of data and models the dynamics of latent factors as a stochastic process.

The key insight is that financial markets are driven by a small number of latent factors (such as risk appetite, liquidity conditions, sector rotations) whose influence on individual assets changes over time. RVRAE learns both the factors and their time-varying loadings simultaneously, producing a trajectory of latent states z_t that represents the evolving market structure. This makes it particularly suited for dynamic portfolio construction, regime detection, and time-varying risk decomposition.

In this chapter, we develop the mathematical foundations of RVRAE, compare it against static factor approaches, explore its architecture in detail, and implement a complete system in Rust with integration to Bybit market data.

## 2. Mathematical Foundation

### 2.1 Sequential Variational Inference

The RVRAE builds on the Variational Recurrent Neural Network (VRNN) framework. Given a sequence of observations x = (x_1, x_2, ..., x_T), we seek to learn a generative model with latent variables z = (z_1, z_2, ..., z_T). The joint distribution factorizes as:

```
p(x_1:T, z_1:T) = prod_{t=1}^{T} p(x_t | z_{<=t}, x_{<t}) * p(z_t | x_{<t}, z_{<t})
```

The prior on z_t depends on all previous observations and latent states through a hidden state h_t maintained by a recurrent network:

```
p(z_t | x_{<t}, z_{<t}) = N(mu_prior_t, sigma_prior_t)
```

where mu_prior_t and sigma_prior_t are computed from h_{t-1}, the RNN hidden state that summarizes the history.

### 2.2 Recurrent Encoder

The encoder (inference network) produces an approximate posterior:

```
q(z_t | x_{<=t}, z_{<t}) = N(mu_enc_t, sigma_enc_t)
```

The encoder takes as input the current observation x_t concatenated with the RNN hidden state h_{t-1}:

```
h_enc_t = GRU_enc(x_t, h_{t-1})
mu_enc_t = W_mu * h_enc_t + b_mu
log_sigma_enc_t = W_sigma * h_enc_t + b_sigma
```

This is critical: the posterior at time t depends not just on x_t but on the entire history through h_{t-1}. This allows the model to leverage temporal context when inferring the current latent state.

### 2.3 Reparameterization Trick

At each timestep, we sample using the reparameterization trick:

```
z_t = mu_enc_t + sigma_enc_t * epsilon_t, where epsilon_t ~ N(0, I)
```

This enables backpropagation through the sampling operation. The sampled z_t is then fed into both the decoder and the RNN state update.

### 2.4 Recurrent Decoder

The decoder reconstructs x_t from z_t and the hidden state:

```
h_dec_t = GRU_dec(z_t, h_{t-1})
x_hat_t = W_dec * h_dec_t + b_dec
```

### 2.5 Sequential ELBO with KL Annealing

The training objective is the sequential Evidence Lower Bound (ELBO):

```
L = sum_{t=1}^{T} [ E_q[log p(x_t | z_t, h_{t-1})] - beta_t * KL(q(z_t | x_{<=t}, z_{<t}) || p(z_t | x_{<t}, z_{<t})) ]
```

The KL divergence between two Gaussians has a closed-form expression:

```
KL(q || p) = 0.5 * sum_j [ log(sigma_p_j / sigma_q_j) + (sigma_q_j^2 + (mu_q_j - mu_p_j)^2) / sigma_p_j^2 - 1 ]
```

The coefficient beta_t implements KL annealing, a technique where we gradually increase the weight on the KL term during training. This prevents the common failure mode of posterior collapse, where the model learns to ignore the latent variables. A typical schedule is:

```
beta_t = min(1.0, epoch / annealing_epochs)
```

Starting with beta near zero allows the model to first learn useful representations in z, then gradually enforces the variational constraint.

## 3. Dynamic vs Static Factors

### 3.1 Limitations of Static Factor Models

Traditional factor models like PCA or standard factor analysis assume:

```
x_t = Lambda * f_t + epsilon_t
```

where Lambda (the loading matrix) is constant over time. Even if f_t varies, the relationship between factors and observables remains fixed. This assumption fails when:

- Market regimes shift (e.g., risk-on vs risk-off environments)
- Correlations evolve (e.g., crisis-period correlation spikes)
- New market structures emerge (e.g., cryptocurrency market maturation)
- Sector rotations change which assets co-move

Rolling-window PCA partially addresses this but introduces window-length sensitivity and cannot model smooth transitions.

### 3.2 Time-Varying Factor Loadings

RVRAE implicitly learns time-varying factor loadings through the interaction between z_t and the decoder's hidden state. The reconstruction at time t depends on:

```
x_hat_t = f(z_t, h_{t-1})
```

Since h_{t-1} carries information about the entire history, the effective mapping from z_t to x_hat_t changes over time. This means the "loading" of each asset on each factor is itself a dynamic quantity, adapting to the current market state.

### 3.3 Regime-Adaptive Factors

The latent trajectory z_1, z_2, ..., z_T naturally captures regime changes. When the market undergoes a structural shift, the latent factors move to a different region of the latent space. This can be detected by monitoring:

- **Factor velocity**: ||z_t - z_{t-1}|| indicates how rapidly the market structure is changing
- **Factor acceleration**: changes in velocity signal regime transitions
- **Factor clustering**: periods where z_t clusters in specific regions indicate stable regimes

### 3.4 Capturing Non-Stationarity

The recurrent structure of RVRAE makes it inherently suited for non-stationary data. The hidden state h_t acts as a sufficient statistic for the history, allowing the model to adapt its behavior based on what it has observed. This contrasts with static models that must be re-estimated on new windows and cannot smoothly track gradual changes.

## 4. Trading Applications

### 4.1 Dynamic Risk Factor Decomposition

RVRAE enables decomposition of portfolio risk into time-varying factor contributions. At each time t, the latent z_t represents the current factor exposures. By analyzing how portfolio returns relate to changes in z_t, traders can understand:

- Which latent factors are currently driving portfolio P&L
- How factor exposures have shifted over recent periods
- Whether the portfolio is becoming more or less concentrated in specific factor bets

This is more informative than static risk decomposition because it reflects the current market structure rather than an average over some historical window.

### 4.2 Time-Varying Portfolio Construction

Dynamic factors enable genuinely adaptive portfolio construction. The process works as follows:

1. Train RVRAE on recent multi-asset return data
2. Extract the latent factor trajectory z_1:T
3. Estimate time-varying factor covariance from recent z_t values
4. Map factor covariance back to asset covariance through the decoder
5. Optimize portfolio weights using the time-varying covariance

This approach naturally adapts to changing correlation structures without requiring explicit regime detection or model switching.

### 4.3 Market Regime Tracking Through Latent Dynamics

The latent space of a trained RVRAE provides a low-dimensional representation of market state. Regime tracking involves:

- **Online inference**: as new data arrives, run the encoder forward to get z_t
- **Regime labeling**: cluster historical z_t trajectories into regimes
- **Transition detection**: monitor distance from current z_t to regime centroids
- **Predictive signals**: use z_t dynamics (velocity, direction) as alpha signals

The advantage over explicit regime models (like Hidden Markov Models) is that RVRAE learns the regime structure directly from data without requiring a pre-specified number of regimes or transition structure.

### 4.4 Factor Momentum and Mean Reversion

The temporal structure of z_t enables factor-level timing strategies:

- **Factor momentum**: when z_t shows persistent directional movement, tilt toward assets with positive loading on that factor
- **Factor mean reversion**: when z_t deviates far from its historical mean, expect reversion and position accordingly
- **Factor volatility timing**: when ||z_t - z_{t-1}|| is elevated, reduce overall exposure

## 5. RVRAE Architecture

### 5.1 Overall Structure

The RVRAE architecture consists of three interconnected components:

```
Input x_t ──> [GRU Encoder] ──> (mu_t, sigma_t)
                    |                    |
                    v                    v
              [Hidden State h_t]  [Reparameterize] ──> z_t
                    |                    |
                    v                    v
              [GRU Decoder] <──────────┘
                    |
                    v
              x_hat_t (reconstruction)
```

### 5.2 GRU Cell Details

We use Gated Recurrent Units (GRU) for both encoder and decoder. The GRU update equations are:

```
r_t = sigmoid(W_r * [h_{t-1}, x_t])     (reset gate)
u_t = sigmoid(W_u * [h_{t-1}, x_t])     (update gate)
h_tilde = tanh(W_h * [r_t * h_{t-1}, x_t])  (candidate)
h_t = (1 - u_t) * h_{t-1} + u_t * h_tilde   (new state)
```

GRUs are preferred over LSTMs here for their simpler parameterization (fewer parameters to train), which is beneficial given the typically limited financial training data.

### 5.3 Temporal KL Divergence

The KL term at each timestep measures how much the encoder's posterior deviates from the prior. In RVRAE, the prior is not a fixed N(0, I) but is itself conditioned on the history:

```
p(z_t | h_{t-1}) = N(mu_prior(h_{t-1}), sigma_prior(h_{t-1}))
```

This temporal prior allows the model to learn smooth latent dynamics. The KL between the posterior and this learned prior captures how "surprising" the current observation is relative to what the model expected based on history.

### 5.4 Training Procedure

1. **Forward pass**: Process sequence through encoder GRU, sample z_t at each step, reconstruct through decoder GRU
2. **Loss computation**: Sum reconstruction error (MSE) and KL divergence across timesteps, applying beta annealing
3. **Backpropagation through time**: Gradients flow through both the latent sampling and the recurrent connections
4. **Gradient clipping**: Essential for stable training of recurrent models on financial data

## 6. Implementation Walkthrough

The Rust implementation provides a complete RVRAE system with the following components:

### 6.1 GRU Cell

The `GRUCell` struct implements the gated recurrent unit with reset gate, update gate, and candidate computation. Weights are initialized with Xavier initialization for stable training.

### 6.2 Recurrent Encoder

The `RecurrentEncoder` processes input sequences through a GRU and outputs per-timestep mean and log-variance vectors for the approximate posterior distribution.

### 6.3 Recurrent Decoder

The `RecurrentDecoder` takes sampled latent vectors z_t and produces reconstructed observations, also using a GRU to maintain temporal coherence.

### 6.4 RVRAE Model

The top-level `RVRAE` struct coordinates the encoder, decoder, and prior network. It implements:

- `forward()`: full forward pass with sampling
- `compute_loss()`: sequential ELBO with KL annealing
- `extract_factors()`: produces the latent trajectory z_1:T for analysis
- `train()`: training loop with configurable epochs and learning rate

### 6.5 Factor Analysis

The `FactorDynamics` struct provides tools for analyzing the extracted factors:

- Autocorrelation computation for each factor
- Regime shift detection based on factor velocity
- Factor correlation analysis over time

See `rust/src/lib.rs` for the full implementation and `rust/examples/trading_example.rs` for a complete trading workflow.

## 7. Bybit Data Integration

The implementation fetches real market data from Bybit's public API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The data pipeline:
1. Fetch OHLCV kline data for multiple symbols (BTCUSDT, ETHUSDT)
2. Compute log returns from closing prices
3. Normalize returns to zero mean and unit variance
4. Format as multivariate time series matrix
5. Feed into RVRAE for training and factor extraction

Multi-asset data is aligned by timestamp and combined into a single observation matrix where each row is a timestep and each column is an asset's return.

## 8. Key Takeaways

1. **RVRAE captures time-varying market structure**: Unlike static factor models, RVRAE learns how latent factors evolve through time, producing a trajectory z_1:T that represents the changing market state.

2. **Sequential VAE architecture respects temporal ordering**: The recurrent encoder and decoder maintain hidden states that carry historical context, enabling the model to leverage temporal patterns in factor dynamics.

3. **KL annealing is critical for financial data**: Financial time series are noisy and high-dimensional relative to the number of training samples. KL annealing prevents posterior collapse and ensures the latent space captures meaningful structure.

4. **Dynamic factors enable adaptive trading strategies**: Time-varying factor decomposition supports regime-aware portfolio construction, dynamic risk management, and factor timing strategies that static models cannot provide.

5. **Factor velocity signals regime changes**: Monitoring the rate of change of latent factors provides an early warning system for market regime transitions, allowing traders to adapt before the shift is apparent in raw price data.

6. **Rust implementation provides production-grade performance**: The GRU-based architecture implemented in Rust delivers the computational efficiency needed for real-time factor extraction and portfolio rebalancing.

7. **Bybit integration enables cryptocurrency factor analysis**: The volatile, 24/7 crypto market is an ideal testbed for dynamic factor models, with frequent regime changes and evolving correlation structures.

8. **The temporal prior is key**: Using a learned, history-dependent prior rather than a fixed N(0,I) prior allows RVRAE to model smooth factor dynamics and distinguish between expected and surprising market movements.
