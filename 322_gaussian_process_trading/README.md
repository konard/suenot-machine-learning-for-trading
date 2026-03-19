# Chapter 322: Gaussian Process Trading

## 1. Introduction: Gaussian Processes for Uncertainty-Aware Trading

Financial markets are inherently uncertain, yet most machine learning models used in trading produce point predictions without any measure of confidence. A model that predicts Bitcoin will be worth $68,000 tomorrow is far less useful than one that says "I predict $68,000 with a 95% confidence interval of [$65,500, $70,500]." This distinction is exactly what Gaussian Processes (GPs) provide: principled uncertainty quantification alongside predictions.

A Gaussian Process is a non-parametric Bayesian approach to regression (and classification). Rather than learning a fixed set of weights as in neural networks, a GP defines a distribution over functions. Given observed data, we can compute the posterior distribution over functions, which gives us both a mean prediction and a variance (uncertainty) at every point. In trading, this means we can:

- **Size positions proportionally to confidence**: When the GP is highly certain about a price increase, allocate more capital. When uncertain, reduce exposure or stay flat.
- **Detect regime changes**: A sudden increase in posterior variance signals that the market is behaving differently from the training data, indicating a potential regime shift.
- **Set dynamic stop-losses and take-profits**: Confidence intervals provide natural levels for risk management.

GPs have a rich theoretical foundation rooted in Bayesian statistics and have been applied across science and engineering for decades. Their application to financial time series is particularly compelling because markets exhibit non-stationary behavior, complex correlations, and fat-tailed distributions -- all of which benefit from a model that honestly communicates its limitations.

In this chapter, we develop a complete Gaussian Process trading system in Rust, applying it to BTCUSDT data from the Bybit exchange. We cover the mathematical foundations, practical implementation details, and strategies for making GPs scalable enough for real-time trading.

## 2. Mathematical Foundations

### 2.1 Gaussian Process Definition

A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is fully specified by:

- A **mean function** m(x) = E[f(x)]
- A **covariance (kernel) function** k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]

We write: f(x) ~ GP(m(x), k(x, x'))

For simplicity, we typically assume m(x) = 0, since any non-zero mean can be absorbed into the kernel or handled via data preprocessing.

### 2.2 Kernel Functions

The kernel function encodes our prior beliefs about the function we are modeling -- its smoothness, periodicity, amplitude, and length scale. Three kernels are particularly useful for financial time series:

**Radial Basis Function (RBF) / Squared Exponential:**

k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))

where sigma_f^2 is the signal variance (controls amplitude) and l is the length scale (controls how quickly correlations decay with distance). The RBF kernel produces infinitely differentiable (very smooth) functions. In trading, this captures slow-moving trends.

**Matern 5/2 Kernel:**

k(x, x') = sigma_f^2 * (1 + sqrt(5) * r / l + 5 * r^2 / (3 * l^2)) * exp(-sqrt(5) * r / l)

where r = ||x - x'||. The Matern 5/2 kernel is twice differentiable, producing functions that are smooth but not unrealistically so. This is often a better default choice for financial data, which exhibits rougher behavior than the RBF kernel assumes.

**Periodic Kernel:**

k(x, x') = sigma_f^2 * exp(-2 * sin^2(pi * |x - x'| / p) / l^2)

where p is the period. This kernel captures repeating patterns, useful for modeling intraday seasonality, day-of-week effects, or monthly cycles in crypto markets.

Kernels can be composed by addition and multiplication. For example, an RBF + Periodic kernel can model a smooth trend with seasonal oscillations.

### 2.3 GP Posterior (Prediction)

Given training data X (n x d) and observations y (n x 1) with noise variance sigma_n^2, and test points X* (m x d), the GP posterior is:

**Posterior mean:**
mu* = K(X*, X) [K(X, X) + sigma_n^2 I]^(-1) y

**Posterior covariance:**
Sigma* = K(X*, X*) - K(X*, X) [K(X, X) + sigma_n^2 I]^(-1) K(X, X*)

where K(A, B) denotes the kernel matrix with entries k(a_i, b_j).

The key insight: the posterior mean provides our prediction, while the diagonal of the posterior covariance gives the uncertainty (variance) at each test point. Far from training data, the variance reverts to the prior variance sigma_f^2; near training data, the variance shrinks.

### 2.4 Numerical Stability: Cholesky Decomposition

Direct inversion of K + sigma_n^2 I is numerically unstable and computationally expensive (O(n^3)). Instead, we use the Cholesky decomposition:

L = cholesky(K(X, X) + sigma_n^2 I)

Then solve via forward/backward substitution:

alpha = L^T \ (L \ y)
mu* = K(X*, X) alpha

This is both faster and more numerically stable than explicit matrix inversion.

### 2.5 Log Marginal Likelihood

The log marginal likelihood (LML) is the key quantity for model selection and hyperparameter optimization:

log p(y | X, theta) = -0.5 * y^T alpha - sum(log(diag(L))) - (n/2) * log(2 * pi)

where theta represents the hyperparameters (sigma_f, l, sigma_n). Maximizing the LML balances model fit (first term) against model complexity (second term), providing automatic Occam's razor.

### 2.6 Hyperparameter Optimization

We optimize hyperparameters by maximizing the log marginal likelihood. While gradient-based methods (using partial derivatives of the LML with respect to hyperparameters) are efficient, a gradient-free approach using grid search or random search over log-space is simpler to implement and often sufficient:

1. Define ranges: log(l) in [-2, 2], log(sigma_f) in [-2, 2], log(sigma_n) in [-4, 0]
2. Evaluate LML at candidate points
3. Select the hyperparameters with highest LML

## 3. Applications in Trading

### 3.1 Price Prediction with Confidence Intervals

The most direct application: fit a GP to recent price data and predict future prices. The posterior mean gives the predicted price, and the posterior variance provides confidence intervals.

**Trading signals from GP predictions:**
- **Long signal**: predicted mean > current price + 2 * sigma (95% confidence the price will increase meaningfully)
- **Short signal**: predicted mean < current price - 2 * sigma
- **Position sizing**: inversely proportional to predicted variance. High confidence -> larger position; low confidence -> smaller position or flat.

### 3.2 Regime Detection via Uncertainty

When the GP's posterior variance suddenly increases for in-sample predictions, it indicates the recent data is inconsistent with the GP's learned model. This is a natural regime change detector:

- **Low variance period**: market is behaving consistently with the learned model. Continue trading.
- **High variance period**: market has shifted. Reduce position sizes, halt trading, or retrain the model.

This approach requires no labeled regime data -- it emerges naturally from the GP's Bayesian uncertainty.

### 3.3 Volatility Surface Modeling

GPs excel at modeling implied volatility surfaces for options pricing. The kernel structure naturally captures the smooth, non-parametric shape of the volatility surface across strikes and maturities.

## 4. Sparse GPs and Inducing Points for Scalability

The O(n^3) computational cost of standard GPs limits their applicability to datasets of a few thousand points. For larger datasets, Sparse GP approximations are essential.

### 4.1 The Inducing Point Approach

Instead of conditioning on all n training points, we select m << n inducing points Z = {z_1, ..., z_m} and approximate the posterior:

q(f*) = integral p(f* | u) q(u) du

where u = f(Z) are function values at inducing points. The key approximation methods include:

- **Subset of Regressors (SoR)**: Uses q(u) = p(u | y), giving O(nm^2) complexity.
- **Fully Independent Training Conditional (FITC)**: Adds a diagonal correction term for better uncertainty estimates.
- **Variational Free Energy (VFE)**: Optimizes inducing point locations jointly with hyperparameters via a variational bound.

### 4.2 Practical Considerations

For trading applications:
- Use m = 50-200 inducing points for datasets of 1000-10000 points
- Place inducing points using k-means on the training inputs as initialization
- Retrain periodically (e.g., daily) to adapt to market evolution

## 5. Rust Implementation Walkthrough

Our Rust implementation consists of several core components:

### 5.1 Kernel Trait and Implementations

We define a `Kernel` trait with an `evaluate` method, then implement `RBFKernel` and `Matern52Kernel`. Each kernel stores its hyperparameters (length scale l and signal variance sigma_f).

### 5.2 GP Struct

The `GaussianProcess` struct holds:
- Training data (X, y)
- Kernel choice and hyperparameters
- Noise variance sigma_n
- Cached Cholesky factor L and alpha vector (for efficient prediction after training)

### 5.3 Prediction Pipeline

1. Compute kernel matrix K(X, X) + sigma_n^2 I
2. Cholesky decompose -> L
3. Solve L alpha = y for alpha (forward/backward substitution)
4. For test points: mu* = k* . alpha, sigma*^2 = k** - v^T v where L v = k*

### 5.4 Bybit Integration

We fetch BTCUSDT kline (candlestick) data from the Bybit v5 API:
- Endpoint: `GET /v5/market/kline`
- Parameters: category=linear, symbol=BTCUSDT, interval=D (daily)
- Parse close prices and timestamps
- Normalize for GP input (scale to [0, 1] range for better conditioning)

### 5.5 Trading Signal Generation

Given GP predictions with uncertainty:
```
if predicted_mean > current_price + 2.0 * predicted_std {
    signal = "STRONG BUY"
} else if predicted_mean > current_price + predicted_std {
    signal = "BUY"
} else if predicted_mean < current_price - 2.0 * predicted_std {
    signal = "STRONG SELL"
} else if predicted_mean < current_price - predicted_std {
    signal = "SELL"
} else {
    signal = "HOLD"
}
```

## 6. Working with Bybit Crypto Data

### 6.1 Data Pipeline

Our implementation fetches real BTCUSDT daily candles from the Bybit exchange:

1. **API Request**: Query the `/v5/market/kline` endpoint for daily OHLCV data
2. **Feature Engineering**: Extract closing prices, compute log returns, and normalize timestamps
3. **GP Input Preparation**: Scale features to improve numerical conditioning of kernel matrices
4. **Train/Test Split**: Use the most recent N days as training, predict the next M days

### 6.2 Prediction with Uncertainty Bands

The GP produces predictions with uncertainty bands that naturally widen as we forecast further into the future. This is a critical advantage over point-prediction models:

- **1-day ahead**: Narrow confidence intervals (the GP has strong information)
- **5-day ahead**: Wider intervals (increasing uncertainty)
- **10-day ahead**: Very wide intervals (the GP honestly communicates it cannot predict far ahead)

This honest uncertainty quantification prevents overconfident trading decisions.

### 6.3 Example Output

```
=== Gaussian Process BTCUSDT Predictions ===
Current price: $67,245.30

Day +1: $67,512.40 +/- $423.18  [66,666.04 - 68,358.76]  -> HOLD
Day +2: $67,891.20 +/- $891.45  [66,108.30 - 69,674.10]  -> HOLD
Day +3: $68,234.10 +/- $1,534.22 [65,165.66 - 71,302.54] -> HOLD
Day +5: $68,890.50 +/- $2,891.33 [63,107.84 - 74,673.16] -> HOLD
```

Notice how the uncertainty grows with the prediction horizon -- a hallmark of honest probabilistic modeling.

### 6.4 Practical Tips for GP Trading

1. **Retrain frequently**: Financial data is non-stationary. Retrain the GP daily or when the LML drops significantly.
2. **Use log returns, not prices**: Log returns are closer to stationary and better suited as GP targets.
3. **Combine kernels**: A Matern52 + Periodic kernel can capture both trends and weekly cycles.
4. **Monitor posterior variance**: Rising in-sample variance is an early warning of regime change.
5. **Keep training windows small**: 30-100 data points often work better than thousands, as distant history may be irrelevant.

## 7. Key Takeaways

1. **Gaussian Processes provide both predictions and uncertainty estimates**, making them ideal for risk-aware trading systems. Unlike neural networks or tree-based models, GPs naturally quantify how confident they are.

2. **Kernel choice encodes market priors**: The RBF kernel assumes smooth functions; Matern 5/2 allows rougher behavior more consistent with financial data; periodic kernels capture seasonality. Composing kernels lets you model complex market structure.

3. **The log marginal likelihood enables principled model selection**: Hyperparameters are optimized by maximizing the LML, which automatically balances fit and complexity without needing a validation set.

4. **Cholesky decomposition is essential for numerical stability**: Never explicitly invert kernel matrices. Always use Cholesky factorization followed by forward/backward substitution.

5. **Sparse GP approximations enable scalability**: Inducing point methods reduce O(n^3) complexity to O(nm^2), making GPs practical for larger trading datasets.

6. **Uncertainty-based position sizing is a natural risk management tool**: Scale positions inversely with predicted variance. This automatically reduces exposure when the model is unsure.

7. **Regime detection emerges naturally from GP uncertainty**: Rising posterior variance on in-sample data signals that the market has changed, providing an automatic regime change detector without labeled data.

8. **GPs are most effective for short-term predictions**: The honest uncertainty bands widen rapidly with prediction horizon, making GPs best suited for intraday to multi-day forecasting rather than long-term predictions.

9. **Rust implementation provides the performance needed for real-time trading**: Matrix operations in Rust, combined with efficient Cholesky decomposition, make GP inference fast enough for live trading systems.

10. **Always combine GP predictions with traditional risk management**: GPs are a powerful tool, but should be part of a broader trading system that includes stop-losses, diversification, and maximum drawdown limits.
