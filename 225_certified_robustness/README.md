# Chapter 225: Certified Robustness

## Introduction

In machine learning for trading, models operate in adversarial environments where input data is inherently noisy, subject to manipulation, and prone to rapid distributional shifts. Traditional robustness evaluation relies on empirical testing: we attack a model with perturbations and observe whether predictions change. But empirical robustness offers no formal guarantees. A model that survives 10,000 adversarial attacks may fail catastrophically on attack 10,001.

**Certified robustness** provides a fundamentally different paradigm. Instead of testing against a finite set of perturbations, certified robustness delivers *provable mathematical guarantees* that a model's predictions will remain unchanged under any perturbation within a specified bound. For a trading system, this means we can formally prove that a buy/sell signal will not flip when market prices fluctuate within a defined noise envelope.

This distinction matters enormously in finance. Regulatory frameworks increasingly demand explainability and reliability from algorithmic trading systems. A certified robustness guarantee transforms model validation from a probabilistic exercise into a deterministic proof. When a regulator asks "How do you know your model won't make erratic decisions under market stress?", certified robustness provides a mathematical certificate rather than an empirical hope.

The core question certified robustness answers is: given an input `x` and a classifier `f`, what is the largest perturbation radius `r` such that for all `x'` within distance `r` of `x`, we have `f(x') = f(x)`? This radius is called the **certified radius**, and it provides a per-prediction guarantee of stability.

## Mathematical Foundation

### Randomized Smoothing

Randomized smoothing, introduced by Cohen et al. (2019), is the most scalable certified defense. The key idea is to convert any base classifier `f` into a smoothed classifier `g` by averaging over Gaussian noise:

```
g(x) = argmax_c P(f(x + epsilon) = c), where epsilon ~ N(0, sigma^2 * I)
```

The smoothed classifier `g` returns the class most likely to be predicted when Gaussian noise is added to the input. The remarkable property is that `g` is provably robust: if `g` classifies `x` as class `c_A` with probability `p_A`, and the runner-up class has probability `p_B`, then `g` is certifiably robust within radius:

```
R = (sigma / 2) * (Phi^{-1}(p_A) - Phi^{-1}(p_B))
```

where `Phi^{-1}` is the inverse of the standard normal CDF. In the two-class case, this simplifies to:

```
R = sigma * Phi^{-1}(p_A)
```

### Neyman-Pearson Lemma

The certified radius derivation relies on the Neyman-Pearson lemma from hypothesis testing. The lemma states that the likelihood ratio test is the most powerful test at any given significance level. In the context of certified robustness:

Given two hypotheses about the distribution of `f(x + epsilon)` — one centered at `x` and one centered at `x + delta` — the Neyman-Pearson lemma tells us the worst-case probability mass that can shift from class `c_A` to another class under perturbation `delta`. This provides the tightest possible bound on the certified radius.

Formally, if `P(f(x + epsilon) = c_A) >= p_A` where `epsilon ~ N(0, sigma^2 I)`, then for any `||delta||_2 <= R`:

```
P(f(x + delta + epsilon) = c_A) >= Phi(Phi^{-1}(p_A) - R/sigma)
```

The certified radius is the largest `R` for which this probability exceeds 0.5.

### Lipschitz Continuity

An alternative mathematical framework uses Lipschitz continuity. A function `f` is L-Lipschitz if:

```
||f(x) - f(x')|| <= L * ||x - x'||
```

If we can bound the Lipschitz constant of a neural network, we can directly certify robustness. For a network with weight matrices `W_1, ..., W_k`, the Lipschitz constant is bounded by:

```
L <= product(||W_i||_2, i=1..k)
```

where `||W_i||_2` is the spectral norm of `W_i`. This bound is often loose but computationally cheap to evaluate.

### Confidence Interval Estimation

In practice, we cannot compute `p_A` exactly — we estimate it by sampling. We draw `n` samples of `f(x + epsilon)` and use binomial proportion confidence intervals. The Clopper-Pearson interval gives us a lower bound `p_A_lower` on `p_A` at confidence level `1 - alpha`:

```
p_A_lower = Beta(alpha; k, n - k + 1)
```

where `k` is the number of samples classified as `c_A`. The certified radius is then computed using `p_A_lower` instead of `p_A`, ensuring the guarantee holds with probability at least `1 - alpha`.

## Certification Methods

### Randomized Smoothing (Cohen et al., 2019)

Randomized smoothing is the most practical method for certifying deep networks. The procedure has two phases:

1. **Prediction**: Sample `n_0` noisy copies of input, take majority vote to determine predicted class `c_A`.
2. **Certification**: Sample `n` noisy copies, count how many predict `c_A`, compute lower confidence bound `p_A_lower`, then compute certified radius `R = sigma * Phi^{-1}(p_A_lower)`.

Advantages: works with any base classifier, scales to large networks, provides `l_2` certified robustness.
Limitations: certification is probabilistic (holds with confidence `1 - alpha`), accuracy degrades with larger `sigma`, requires many forward passes.

### Interval Bound Propagation (IBP)

IBP propagates interval bounds layer by layer through a neural network. Given input bounds `[x_l, x_u]`:

For a linear layer `y = Wx + b`:
```
y_l = W_pos * x_l + W_neg * x_u + b
y_u = W_pos * x_u + W_neg * x_l + b
```

where `W_pos = max(W, 0)` and `W_neg = min(W, 0)`.

For ReLU: `y_l = max(x_l, 0)`, `y_u = max(x_u, 0)`.

IBP gives deterministic (not probabilistic) bounds but they are often very loose, leading to conservative certified radii.

### CROWN (Zhang et al., 2018)

CROWN (Convex Relaxation based perturbation analysis of Neural Networks) provides tighter bounds than IBP by using linear relaxations. Instead of propagating intervals, CROWN propagates linear bounds:

```
A_l * x + b_l <= f(x) <= A_u * x + b_u
```

For ReLU activations, CROWN uses the convex relaxation: when `x_l < 0 < x_u`, the ReLU is bounded by a linear upper bound connecting `(x_l, 0)` to `(x_u, x_u)` and a linear lower bound that is optimized.

CROWN provides tighter certified radii than IBP at higher computational cost, making it suitable for smaller networks common in trading applications.

### Linear Relaxation

Linear relaxation methods generalize the approach used in CROWN. They replace non-linear activation functions with linear upper and lower bounds, converting the certification problem into a linear program (LP). The LP can be solved efficiently, and its solution provides valid certified bounds.

For trading networks that are typically smaller than vision models, LP-based certification is computationally tractable and provides excellent bound quality.

## Trading Applications

### Guaranteed Prediction Stability Under Market Noise

Financial data is inherently noisy. Tick-level price data contains microstructure noise, bid-ask bounce, and measurement errors. A trading signal that flips between buy and sell due to a 0.01% price change is unreliable and dangerous.

Certified robustness allows us to guarantee that a trading signal is stable under realistic noise levels. For example, if we certify that a buy signal for BTCUSDT has a certified radius of 0.5 standard deviations, we know that any price movement within that envelope will not change the signal.

The practical workflow is:
1. Estimate the noise level in market data (e.g., microstructure noise variance)
2. Train a base classifier for trading signals
3. Apply randomized smoothing with `sigma` matched to the noise level
4. Certify each prediction — only execute trades with certified radius exceeding the noise level

This approach naturally filters out low-confidence predictions, improving the overall quality of trading signals.

### Regulatory Compliance for Model Robustness

Financial regulators (SEC, FCA, MAS) are increasingly scrutinizing algorithmic trading systems. MiFID II in Europe requires firms to have "effective systems and risk controls" for algorithmic trading. Certified robustness provides a quantitative framework for demonstrating model reliability:

- **Model validation**: Each prediction comes with a certified radius, providing an auditable measure of reliability.
- **Stress testing**: Instead of running scenarios, certified robustness provides worst-case guarantees analytically.
- **Documentation**: The mathematical certificate can be included in model documentation as proof of robustness.

### Certified Risk Bounds

Beyond signal stability, certified robustness can be applied to risk models. If a Value-at-Risk (VaR) model has a certified radius of `r` in input space, we can bound the worst-case change in VaR estimates under input perturbations. This gives risk managers formal guarantees about model sensitivity.

For portfolio optimization, certified robustness of the expected return predictions translates into bounds on how much the optimal portfolio can change under data perturbations, addressing the well-known instability of mean-variance optimization.

## Certification vs Empirical Robustness

| Aspect | Certified Robustness | Empirical Robustness |
|--------|---------------------|---------------------|
| Guarantee | Mathematical proof | Statistical evidence |
| Coverage | All perturbations within radius | Only tested perturbations |
| Computational cost | Higher (sampling or bound propagation) | Lower (finite attack set) |
| Tightness | May be conservative | Can be tight but incomplete |
| Scalability | Moderate (randomized smoothing scales well) | High |
| Regulatory value | Strong (provable) | Moderate (best-effort) |

**Advantages of certified robustness:**
- Provides provable guarantees, not just empirical evidence
- No adversary can find an attack within the certified radius
- Per-prediction certificates allow selective execution
- Strong regulatory and compliance value

**Limitations of certified robustness:**
- Certified accuracy is always lower than standard accuracy
- Bounds may be conservative (especially IBP)
- Randomized smoothing requires many forward passes
- Only certifies against `l_p` perturbations, not all possible attacks
- Trade-off between certified radius and accuracy (larger `sigma` gives larger radius but lower clean accuracy)

In practice, the best approach combines both: use certified robustness for formal guarantees and empirical robustness testing for practical validation.

## Implementation Walkthrough

Our Rust implementation provides the core building blocks for certified robustness in trading:

### Architecture

The implementation consists of several key components:

1. **Base Classifier (`SimpleNeuralNetwork`)**: A feedforward neural network with configurable architecture that serves as the base classifier for randomized smoothing.

2. **Randomized Smoothing (`SmoothedClassifier`)**: Wraps the base classifier and implements:
   - `predict()`: Majority vote over noisy samples for robust prediction
   - `certify()`: Computes certified radius using Neyman-Pearson bound
   - Configurable noise level (`sigma`) and sample counts

3. **IBP Certification (`IBPCertifier`)**: Implements interval bound propagation for deterministic certification of simple networks.

4. **Confidence Intervals**: Clopper-Pearson binomial confidence intervals for rigorous statistical bounds on class probabilities.

5. **Bybit Integration**: Fetches real BTCUSDT kline data via the Bybit public API for realistic feature construction.

### Key Implementation Details

The randomized smoothing certification follows this procedure:

```rust
// 1. Sample n noisy copies and count classifications
for _ in 0..n_samples {
    let noisy_input = add_gaussian_noise(&input, sigma);
    let prediction = base_classifier.predict(&noisy_input);
    counts[prediction] += 1;
}

// 2. Find top class and compute lower confidence bound
let top_class = argmax(&counts);
let p_lower = clopper_pearson_lower(counts[top_class], n_samples, alpha);

// 3. Compute certified radius
let certified_radius = sigma * normal_ppf(p_lower);
```

The certified radius `R` guarantees that no perturbation within `l_2` distance `R` can change the prediction.

### Running the Example

```bash
cd rust
cargo run --example trading_example
```

The trading example:
1. Fetches BTCUSDT data from Bybit
2. Constructs features (returns, volatility, momentum)
3. Trains a base classifier on trading signals
4. Applies randomized smoothing at different `sigma` values
5. Reports certified accuracy at various radii
6. Compares certified vs empirical robustness metrics

## Bybit Data Integration

The implementation fetches real market data from the Bybit V5 API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

Features extracted from the kline data:
- **Log returns**: `ln(close_t / close_{t-1})`
- **Volatility**: Rolling standard deviation of returns
- **Momentum**: Cumulative return over lookback window
- **Volume ratio**: Current volume relative to moving average

These features are normalized and fed into the base classifier. The certified radius is expressed in terms of the normalized feature space, allowing interpretation as "how much can features change before the signal flips?"

## Key Takeaways

1. **Certified robustness provides mathematical guarantees** that model predictions are stable under bounded perturbations, unlike empirical robustness which only tests finite attack sets.

2. **Randomized smoothing is the most practical method** for certifying deep networks, working with any base classifier and providing `l_2` certified robustness.

3. **The certified radius quantifies prediction reliability** — larger radius means the prediction is more stable, enabling selective trade execution based on certification level.

4. **IBP and CROWN provide deterministic bounds** suitable for smaller networks typical in trading, with different trade-offs between tightness and computational cost.

5. **Trading applications include signal stability, regulatory compliance, and risk bound certification** — all areas where provable guarantees are more valuable than empirical evidence.

6. **There is a fundamental trade-off** between certified radius and clean accuracy: increasing the noise level `sigma` enlarges the certifiable region but degrades base accuracy.

7. **Combining certified and empirical robustness** gives the strongest validation: certified methods for formal guarantees and empirical methods for practical stress testing.

8. **Per-prediction certificates enable intelligent filtering** — only execute trades where the certified radius exceeds the expected noise level in market data, naturally improving signal quality.
