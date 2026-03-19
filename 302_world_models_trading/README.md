# Chapter 302: World Models for Trading

## Introduction

World Models, introduced by Ha and Schmidhuber (2018), represent a paradigm shift in reinforcement learning. Instead of learning a policy directly through interaction with the real environment, the agent first learns a compressed, generative model of the world. This internal model captures the essential dynamics of the environment, enabling the agent to "dream" --- to simulate future trajectories entirely inside its own imagination. The policy (controller) is then optimized within this dream environment, dramatically reducing the need for expensive real-world interaction.

In the context of financial markets, world models offer a compelling framework. Markets are noisy, non-stationary, and expensive to interact with (every trade incurs fees, slippage, and market impact). A world model trained on historical market data can serve as a high-fidelity simulator for strategy exploration, data augmentation, and risk-free experimentation. The agent learns a latent representation of market states, a predictive model of how those states evolve, and a lightweight controller that maps latent states to trading actions.

The original World Models architecture comprises three components:

1. **Vision (V) --- the VAE**: A Variational Autoencoder that compresses high-dimensional observations into a compact latent vector z.
2. **Memory (M) --- the MDN-RNN**: A Mixture Density Network combined with an LSTM that models the temporal dynamics of z, predicting the distribution of the next latent state.
3. **Controller (C)**: A simple linear mapping from the current latent state (and RNN hidden state) to actions, optimized via evolutionary strategies (CMA-ES).

This chapter presents the mathematical foundations, trading-specific adaptations, and a complete Rust implementation with Bybit integration.

---

## Mathematical Foundations

### Variational Autoencoder (VAE)

The VAE compresses a market observation vector x (e.g., OHLCV features, technical indicators) into a latent representation z of much lower dimensionality. The encoder outputs parameters of a Gaussian distribution:

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))
$$

The latent vector is sampled via the reparameterization trick:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The decoder reconstructs the observation from z:

$$
p_\theta(x|z)
$$

The VAE loss combines reconstruction error and KL divergence:

$$
\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

where p(z) = N(0, I) is the standard normal prior. The KL term regularizes the latent space, ensuring smooth interpolation and meaningful generation.

For trading, the observation x might include normalized OHLCV data, returns, volatility estimates, order book imbalance, and volume profiles. The latent z captures the "market regime" in a compressed form.

### MDN-RNN (Memory Model)

The MDN-RNN models temporal dynamics in latent space. An LSTM processes sequences of (z_t, a_t) pairs (latent state and action) and outputs parameters of a Gaussian mixture model for the next latent state:

$$
P(z_{t+1} | a_t, z_t, h_t) = \sum_{i=1}^{K} \pi_i \mathcal{N}(z_{t+1}; \mu_i, \sigma_i^2)
$$

where K is the number of mixture components, pi_i are mixing coefficients (summing to 1), and mu_i, sigma_i are the mean and standard deviation of each Gaussian component. The LSTM hidden state h_t encodes the history of observations.

The LSTM update equations are:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, z_t, a_t] + b_f)
$$
$$
i_t = \sigma(W_i \cdot [h_{t-1}, z_t, a_t] + b_i)
$$
$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, z_t, a_t] + b_c)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$
$$
o_t = \sigma(W_o \cdot [h_{t-1}, z_t, a_t] + b_o)
$$
$$
h_t = o_t \odot \tanh(c_t)
$$

The MDN output layer transforms h_t into mixture parameters:

$$
[\pi, \mu, \sigma] = W_{mdn} \cdot h_t + b_{mdn}
$$

The loss function is the negative log-likelihood of the observed next latent state under the predicted mixture:

$$
\mathcal{L}_{MDN} = -\log \sum_{i=1}^{K} \pi_i \mathcal{N}(z_{t+1}; \mu_i, \sigma_i^2)
$$

For trading, the MDN-RNN learns to predict how market regimes evolve over time, capturing transitions between trending, mean-reverting, and volatile states.

### Controller (CMA-ES Optimization)

The controller is deliberately simple --- a linear mapping:

$$
a_t = W_c \cdot [z_t, h_t] + b_c
$$

where a_t is the trading action (e.g., position size in [-1, 1]), z_t is the current latent state, and h_t is the RNN hidden state.

The controller parameters theta_c = {W_c, b_c} are optimized using Covariance Matrix Adaptation Evolution Strategy (CMA-ES), which maintains a multivariate Gaussian search distribution:

$$
\theta \sim \mathcal{N}(m, \sigma^2 C)
$$

At each generation, CMA-ES:
1. Samples lambda candidate solutions from the distribution
2. Evaluates each candidate's fitness (cumulative return in the dream environment)
3. Updates the mean m toward the best-performing candidates
4. Adapts the covariance matrix C and step size sigma

The fitness function for trading is typically the Sharpe ratio or cumulative PnL of the controller over a dream rollout:

$$
F(\theta_c) = \frac{1}{N} \sum_{n=1}^{N} R_n(\theta_c)
$$

where R_n is the return from the n-th dream episode.

---

## Dream Training

The key insight of world models is **dream training**: once the VAE and MDN-RNN are trained on real data, the controller can be trained entirely within the learned world model without any further interaction with real markets.

The dream rollout procedure:

1. Sample an initial latent state z_0 from the training distribution
2. At each step t:
   - The controller selects action a_t = C(z_t, h_t)
   - The MDN-RNN predicts the distribution of z_{t+1}
   - Sample z_{t+1} from the predicted mixture distribution
   - Compute reward r_t (e.g., simulated PnL based on latent-to-price mapping)
3. Accumulate total return over the episode

This approach offers several advantages for trading:

- **Speed**: Dream rollouts are orders of magnitude faster than backtesting on real data, since they operate in latent space without full market reconstruction.
- **Diversity**: The stochastic nature of the MDN sampling generates diverse market scenarios, including rare events the controller must handle.
- **Safety**: No real capital is at risk during training. The agent can explore aggressive strategies in dreams before cautious deployment.
- **Generalization**: The learned world model captures market dynamics that help the controller generalize to unseen conditions.

---

## Applications in Trading

### Fast Strategy Exploration

Traditional strategy optimization requires running backtests over long historical periods for each parameter configuration. With world models, each backtest runs in the dream environment at a fraction of the cost. CMA-ES can evaluate thousands of controller variants per generation, enabling rapid exploration of the strategy space.

### Market Hallucination for Data Augmentation

The trained MDN-RNN can generate synthetic market trajectories by sampling from its learned distribution. These "hallucinated" markets can:

- Augment limited training data for downstream models
- Generate stress-test scenarios (sampling from tail distributions)
- Create diverse training environments for robust controller optimization
- Simulate regime changes that are rare in historical data

### Regime Detection

The latent space z learned by the VAE naturally clusters market regimes. By analyzing the structure of the latent space (e.g., via clustering or visualization), traders can identify distinct market states such as low-volatility trending, high-volatility mean-reverting, or crash regimes.

### Transfer Learning

A world model trained on one market (e.g., BTCUSDT) can be fine-tuned for related markets, leveraging shared dynamics while adapting to market-specific characteristics.

---

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

- **VAE**: Encoder and decoder networks that compress market features into a latent space. The encoder outputs mean and log-variance, and the reparameterization trick enables gradient-like optimization.
- **MDN-RNN**: An LSTM cell combined with a mixture density output layer. The LSTM tracks temporal dependencies while the MDN captures multimodal distributions over future latent states.
- **Controller**: A linear policy mapping (z_t, h_t) to actions, optimized via a simplified CMA-ES implementation.
- **WorldModel**: Orchestrates the three components, providing methods for encoding observations, predicting next states, dream rollouts, and controller optimization.

Key design choices:
- `ndarray` is used for efficient matrix operations
- The VAE uses a single hidden layer for both encoder and decoder
- The MDN uses K=5 Gaussian mixture components by default
- CMA-ES operates on flattened controller parameters with diagonal covariance approximation

---

## Bybit Data Integration

The implementation fetches real OHLCV data from the Bybit API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The response is parsed into candlestick data, normalized, and converted into feature vectors for VAE training. Features include:
- Normalized returns (close-to-close)
- High-low range (volatility proxy)
- Volume (log-scaled)
- Open-close body ratio

---

## Key Takeaways

1. **World models separate representation learning from policy optimization**, allowing each component to be trained independently and efficiently.

2. **Dream training eliminates the need for expensive market interaction** during policy search, enabling rapid strategy exploration at minimal cost.

3. **The VAE latent space provides a natural market regime representation** that captures essential dynamics while discarding noise.

4. **MDN-RNN captures multimodal market dynamics**, modeling the uncertainty and regime-switching behavior inherent in financial time series.

5. **CMA-ES is well-suited for optimizing trading controllers** because it handles non-differentiable objectives (Sharpe ratio), is robust to noise, and works well in low-dimensional parameter spaces.

6. **Market hallucination generates synthetic data** for augmentation, stress testing, and robust strategy development.

7. **The approach scales naturally**: the VAE and MDN-RNN train on historical data once, and then thousands of controller variants can be evaluated cheaply in dream space.

8. **Rust provides performance advantages** for dream rollouts where millions of latent-space simulations are needed, and for real-time inference in production trading systems.

---

## References

- Ha, D., & Schmidhuber, J. (2018). "World Models." arXiv:1803.10122.
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
- Bishop, C. M. (1994). "Mixture Density Networks." Technical Report.
- Hansen, N. (2006). "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
