# Chapter 222: Adversarial Attack Detection

## 1. Introduction - Detecting Adversarial Inputs in Trading Systems

Adversarial attacks represent one of the most insidious threats to machine learning systems deployed in financial markets. Unlike traditional cyberattacks that target infrastructure, adversarial attacks target the decision-making logic of ML models by crafting inputs that are nearly indistinguishable from legitimate data yet cause catastrophic misclassifications. In trading, this can manifest as subtly manipulated market data feeds, spoofed order books, or engineered price patterns designed to trick predictive models into making unprofitable or destructive trades.

The stakes in trading are uniquely high. A misclassified image in a computer vision system is an inconvenience; a misclassified trading signal can result in millions of dollars in losses within milliseconds. Furthermore, the adversarial threat in finance is not hypothetical -- market manipulation is well-documented and increasingly sophisticated. Spoofing, layering, and quote stuffing are all forms of adversarial behavior that directly target algorithmic trading systems.

Adversarial attack detection sits at the intersection of robust machine learning and market microstructure analysis. The goal is to build a secondary defense layer that monitors model inputs and outputs in real-time, flagging data points that exhibit characteristics consistent with adversarial perturbation. This chapter develops the mathematical foundations, implements practical detection methods in Rust for low-latency deployment, and demonstrates their application to cryptocurrency market data from Bybit.

## 2. Mathematical Foundation

### 2.1 Statistical Tests for Input Anomaly Detection

The foundation of adversarial detection rests on the observation that adversarial examples, while close to legitimate inputs in input space, often reside in low-density regions of the data manifold. Let $X$ be the input space and $p(x)$ the probability density of clean data. An adversarial example $x_{adv}$ is typically crafted such that:

$$\|x_{adv} - x\|_p \leq \epsilon$$

for some perturbation budget $\epsilon$ under $L_p$ norm, yet $f(x_{adv}) \neq f(x)$ where $f$ is the target model. The key insight for detection is that despite being close in input space, adversarial examples often have distinguishable statistical properties.

**Kernel Density Estimation (KDE):** Given a set of clean training examples $\{x_1, \ldots, x_n\}$, we estimate the density at a test point $x$ using:

$$\hat{p}(x) = \frac{1}{nh^d} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where $K$ is the kernel function (typically Gaussian), $h$ is the bandwidth, and $d$ is the dimensionality. Adversarial examples tend to have lower density estimates than clean examples.

### 2.2 Feature Squeezing

Feature squeezing reduces the search space available to an adversary by applying input transformations that "squeeze" out adversarial perturbations while preserving semantic content. Two primary squeezing operations are:

**Bit-depth reduction:** Reduces the precision of each feature value from $b$ bits to $b'$ bits ($b' < b$):

$$\text{squeeze}_{bit}(x) = \frac{\lfloor x \cdot 2^{b'} \rfloor}{2^{b'}}$$

**Spatial smoothing:** Applies a local averaging filter with window size $w$:

$$\text{squeeze}_{smooth}(x_i) = \frac{1}{|N(i)|} \sum_{j \in N(i)} x_j$$

The detection criterion compares model predictions on original and squeezed inputs:

$$D(x) = \mathbb{1}\left[\|f(x) - f(\text{squeeze}(x))\|_1 > \tau\right]$$

where $\tau$ is a detection threshold calibrated on clean validation data.

### 2.3 Input Transformation Detection

Beyond feature squeezing, input transformations such as adding small Gaussian noise, applying total variation minimization, or JPEG-style compression can reveal adversarial perturbations. The principle is that natural data is robust to these transformations while adversarial perturbations are fragile:

$$\Delta(x) = \|f(x) - f(T(x))\|$$

where $T$ is an input transformation. A large $\Delta$ suggests adversarial manipulation.

### 2.4 Model Uncertainty Under Attack

Bayesian uncertainty provides another detection signal. Under a Bayesian neural network or MC-Dropout approximation, the predictive uncertainty for input $x$ is:

$$\text{BU}(x) = \mathbb{H}[y|x] = -\sum_c p(y=c|x) \log p(y=c|x)$$

Adversarial examples often exhibit higher predictive uncertainty because they lie near decision boundaries. Combining KDE density with Bayesian uncertainty (the KD+BU approach) yields a powerful joint detector.

### 2.5 Local Intrinsic Dimensionality (LID)

LID characterizes the dimensional properties of data in the neighborhood of a reference point. For a point $x$ and its $k$-nearest neighbors at distances $r_1 \leq r_2 \leq \ldots \leq r_k$, the maximum likelihood estimate of LID is:

$$\widehat{\text{LID}}(x) = -\left(\frac{1}{k} \sum_{i=1}^{k} \log \frac{r_i}{r_k}\right)^{-1}$$

Adversarial examples tend to have higher LID values than clean examples, indicating that they occupy regions of the data space with unusual local geometry. This property is robust across different attack methods and model architectures.

## 3. Detection Methods

### 3.1 KD+BU: Kernel Density and Bayesian Uncertainty

The KD+BU detector combines two complementary signals. Kernel density estimates capture whether an input lies in a well-populated region of the training distribution. Bayesian uncertainty captures whether the model is confident in its prediction. The joint detection score is:

$$s(x) = \alpha \cdot \text{normalize}(-\log \hat{p}(x)) + (1 - \alpha) \cdot \text{normalize}(\text{BU}(x))$$

A threshold on $s(x)$ determines the detection decision. In trading, this translates to monitoring whether incoming market data features have reasonable density under the historical distribution while also checking if the model's trading signal is unusually uncertain.

### 3.2 LID-Based Detection

The LID detector operates in the feature space of intermediate model layers. For each layer $l$ of a deep network, we compute $\widehat{\text{LID}}_l(x)$ for the test input $x$. The LID vector across layers serves as a feature vector for a binary classifier (clean vs. adversarial). This approach is particularly effective because adversarial perturbations affect the geometry of representations differently at different network depths.

### 3.3 Feature Squeezing Detection

Feature squeezing is appealing for trading systems because of its simplicity and computational efficiency. The detector:
1. Maintains the original model $f$.
2. Applies one or more squeezing functions to the input.
3. Compares predictions on original vs. squeezed inputs.
4. Flags inputs where the prediction divergence exceeds the threshold.

For market data, bit-depth reduction simulates the effect of rounding prices to fewer decimal places, while spatial smoothing corresponds to moving average filtering of time-series features.

### 3.4 Reconstruction-Based Detection

An autoencoder trained on clean market data learns to reconstruct normal patterns. The reconstruction error serves as an anomaly score:

$$e(x) = \|x - \text{Dec}(\text{Enc}(x))\|^2$$

Adversarial examples, which contain perturbations not present in the training distribution, tend to produce higher reconstruction errors. The autoencoder effectively learns a manifold of normal market behavior, and deviations from this manifold signal potential attacks.

## 4. Trading Applications

### 4.1 Detecting Manipulated Market Data

Market data manipulation can take several forms that adversarial detection can address:

- **Price feed tampering:** An attacker modifies price data between the exchange and the trading system. Feature squeezing and reconstruction-based detectors can identify artificial perturbations in the price stream.
- **Synthetic market patterns:** Adversarially generated candlestick patterns designed to trigger false technical signals. LID-based detection can identify patterns that don't match the local geometry of real market data.
- **Cross-asset inconsistency:** Perturbations to one asset's data that don't match the correlation structure with related assets. Statistical detectors using joint density estimation can flag these anomalies.

### 4.2 Spoofing Detection

Spoofing involves placing large orders with the intent to cancel them before execution, creating a false impression of supply or demand. From an adversarial detection perspective, spoofing creates order book states that are "adversarial" to models relying on depth-of-book features:

- The order book snapshot appears to suggest strong buying/selling pressure.
- Feature squeezing (smoothing the order book depth profile) reveals the artificiality.
- LID analysis of order book feature vectors shows unusual local dimensionality.
- Reconstruction error is elevated because the autoencoder cannot reproduce the spoofed pattern from clean training data.

### 4.3 Anomalous Order Flow Identification

Adversarial detection methods can be repurposed for identifying unusual trading activity:

- Sudden changes in order arrival rate patterns that deviate from the learned density model.
- Trade size distributions that shift in ways inconsistent with natural market evolution.
- Temporal patterns of quote updates that indicate algorithmic manipulation.

The KD+BU approach is particularly effective here because it combines the statistical anomaly (low density of the observed pattern) with model uncertainty (the trading model is unsure how to interpret the unusual flow).

## 5. Real-Time Detection

### 5.1 Online Detection with Low Latency Requirements

Trading systems operate under extreme latency constraints, often requiring sub-millisecond decision making. Adversarial detection must be designed for this environment:

**Computational budget allocation:** The detection system must consume only a fraction of the total latency budget. In practice, this means:
- Feature squeezing: O(n) time, suitable for microsecond-scale detection.
- KDE evaluation: Requires approximate nearest neighbor structures (KD-trees, LSH) for O(log n) query time.
- LID estimation: Requires k-NN queries, mitigated by pre-computed spatial indices.
- Reconstruction error: Single forward pass through a compact autoencoder.

**Streaming detection:** Rather than evaluating each data point independently, maintain a sliding window of detection scores and apply CUSUM or EWMA control charts to detect sustained adversarial activity:

$$S_t = \max(0, S_{t-1} + s_t - \mu_0 - k)$$

where $s_t$ is the detection score at time $t$, $\mu_0$ is the expected score under clean data, and $k$ is the allowance parameter. An alarm triggers when $S_t > h$.

**Tiered detection:** Implement a cascade of detectors with increasing computational cost:
1. Fast feature squeezing check (microseconds).
2. Pre-computed density estimate check (tens of microseconds).
3. Full LID + reconstruction analysis (hundreds of microseconds).

Only inputs that pass the fast checks proceed to the more expensive detectors, minimizing average detection latency.

### 5.2 Adaptive Thresholds

Market conditions are non-stationary, so detection thresholds must adapt. During high-volatility regimes, natural data variability increases, and fixed thresholds produce excessive false positives. Adaptive thresholding uses a rolling estimate of the clean-data score distribution:

$$\tau_t = \hat{\mu}_t + z_\alpha \cdot \hat{\sigma}_t$$

where $\hat{\mu}_t$ and $\hat{\sigma}_t$ are exponentially weighted moving estimates of the score mean and standard deviation, and $z_\alpha$ controls the false positive rate.

## 6. Implementation Walkthrough with Rust

The implementation in `rust/src/lib.rs` provides a complete adversarial detection framework optimized for trading applications. The key design decisions:

**Why Rust?** Trading systems demand predictable low-latency performance. Rust's zero-cost abstractions, absence of garbage collection, and strong type system make it ideal for detection systems that must run in the hot path of a trading pipeline.

**Architecture:** The implementation is structured around several core components:

1. **`AdversarialDetector`** -- The main struct that orchestrates all detection methods. It maintains clean data statistics and provides a unified detection interface.

2. **Attack generation (`fgsm_attack`, `pgd_attack`)** -- Implementations of Fast Gradient Sign Method and Projected Gradient Descent for generating test adversarial examples. These are essential for validating detector performance.

3. **`FeatureSqueezingDetector`** -- Implements bit-depth reduction and spatial smoothing, comparing the L1 distance between original and squeezed inputs against a configurable threshold.

4. **`StatisticalDetector`** -- KDE-based density estimation using Gaussian kernels. Fits on clean training data and flags inputs with density below a learned threshold.

5. **`LIDEstimator`** -- Computes Local Intrinsic Dimensionality using the maximum likelihood estimator with k-nearest neighbors.

6. **`ReconstructionDetector`** -- Simple linear autoencoder that learns to reconstruct clean data patterns. High reconstruction error indicates potential adversarial input.

7. **`DetectionMetrics`** -- Computes true positive rate, false positive rate, and AUC-ROC for evaluating detector performance.

8. **`BybitClient`** -- Fetches real market data (klines) from the Bybit API for realistic testing.

### Usage Example

```rust
use adversarial_attack_detection::*;

// Fit detector on clean market data
let mut detector = AdversarialDetector::new(0.1, 5, 10);
detector.fit(&clean_data);

// Check new data point
let result = detector.detect(&new_input);
if result.is_adversarial {
    println!("Adversarial input detected! Score: {}", result.score);
    // Halt trading or switch to conservative mode
}
```

The implementation prioritizes clarity and correctness while maintaining performance suitable for trading applications. See `rust/src/lib.rs` for the complete source and `rust/examples/trading_example.rs` for a full walkthrough with Bybit data.

## 7. Bybit Data Integration

The implementation includes a `BybitClient` that fetches real-time and historical market data from the Bybit exchange API. The integration supports:

- **Kline (candlestick) data:** OHLCV data at configurable intervals, providing the feature vectors for detection.
- **Multiple intervals:** From 1-minute to daily candles, allowing multi-timescale detection.
- **Feature extraction:** Raw klines are converted to feature vectors including returns, volatility ratios, volume profiles, and technical indicators.

The Bybit API endpoint used is:

```
GET https://api.bybit.com/v5/market/kline
    ?category=linear
    &symbol=BTCUSDT
    &interval=5
    &limit=200
```

Data is normalized before being passed to detectors, using rolling statistics to handle non-stationarity. The `trading_example.rs` demonstrates the full pipeline from data fetching through detection.

### Data Pipeline

1. Fetch raw klines from Bybit API.
2. Compute derived features (log returns, realized volatility, volume ratios).
3. Normalize features using rolling mean and standard deviation.
4. Feed normalized features to the adversarial detection ensemble.
5. Aggregate detection scores across methods.
6. Apply adaptive thresholding for the final detection decision.

## 8. Key Takeaways

1. **Adversarial attacks are a real threat to trading systems.** Market manipulation is a form of adversarial attack, and ML-based trading systems are particularly vulnerable because they rely on statistical patterns that can be deliberately exploited.

2. **Multiple detection methods provide defense in depth.** No single detector is perfect. Feature squeezing catches perturbation-based attacks, KDE catches out-of-distribution inputs, LID detects unusual local geometry, and reconstruction error catches complex anomalies. Using them together dramatically improves robustness.

3. **Detection must operate within latency constraints.** Trading systems cannot afford to spend milliseconds on detection. The tiered cascade approach (fast checks first, expensive checks only when needed) balances detection power with latency requirements.

4. **Adaptive thresholds are essential for non-stationary markets.** Fixed detection thresholds produce unacceptable false positive rates during regime changes. Rolling statistics-based adaptive thresholds maintain consistent detection quality across market conditions.

5. **Rust provides the performance guarantees needed for production deployment.** The detection pipeline must be deterministic and fast. Rust's zero-cost abstractions and absence of garbage collection pauses make it the ideal choice for latency-sensitive detection.

6. **Bybit and other exchange APIs provide the data needed for realistic testing.** Validating detectors on real market data is essential because synthetic data may not capture the full complexity of market microstructure.

7. **Adversarial detection is complementary to model robustness.** Detection (identifying adversarial inputs) and robustness (making models resistant to adversarial inputs) are complementary strategies. A complete defense uses both.

8. **The same detection framework applies to multiple trading threats.** Spoofing detection, market manipulation detection, and anomalous order flow identification all benefit from the same underlying adversarial detection methods, making this a high-leverage investment in trading infrastructure.
