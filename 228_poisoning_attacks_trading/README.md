# Chapter 228: Poisoning Attacks in Trading

## 1. Introduction

Data poisoning represents one of the most insidious threats to machine learning systems in financial markets. Unlike adversarial attacks that manipulate inputs at inference time, poisoning attacks corrupt the training data itself, compromising the model's learned behavior from its foundation. In trading environments, where models are continuously retrained on streaming market data, the attack surface for data poisoning is vast and the consequences are severe.

A poisoning attack works by injecting carefully crafted malicious samples into the training dataset. The attacker's goal is to cause the resulting model to behave incorrectly — either degrading its overall performance (availability attacks) or causing specific, targeted misclassifications (integrity attacks). In the context of algorithmic trading, a successful poisoning attack could cause a model to systematically make losing trades, miss profitable opportunities, or execute trades that benefit the attacker's positions.

The threat model for trading systems is particularly concerning because training data often comes from external sources: exchange feeds, third-party data providers, alternative data vendors, and social media sentiment aggregators. Each of these channels represents a potential injection point for poisoned data. A compromised price feed, manipulated order book snapshots, or fabricated sentiment scores can all serve as vectors for poisoning attacks.

This chapter explores the mathematical foundations of data poisoning, catalogs attack types relevant to trading, demonstrates practical implementations in Rust, and presents defense mechanisms that trading systems should employ to maintain model integrity.

## 2. Mathematical Foundation

### 2.1 Bilevel Optimization for Poisoning

Data poisoning can be formally expressed as a bilevel optimization problem. The attacker seeks to find poisoned data points that maximize a malicious objective, subject to the constraint that the model is trained optimally on the combined clean and poisoned dataset.

The outer optimization (attacker's problem):

```
max_{D_p} L_atk(theta*(D_train ∪ D_p))
```

The inner optimization (learner's problem):

```
theta*(D) = argmin_{theta} L_train(theta, D)
```

Here, `D_p` is the set of poisoned samples, `D_train` is the clean training set, `L_atk` is the attacker's loss function (which they want to maximize), and `L_train` is the standard training loss. The key insight is that the attacker must anticipate how the learner will respond to the poisoned data — the attack is only effective if the resulting model parameters `theta*` serve the attacker's purpose.

### 2.2 Influence Functions

Influence functions provide a principled way to estimate the effect of adding or removing a single training point on model parameters without full retraining. For a training point `z = (x, y)`, the influence on model parameters is:

```
I(z) = -H_{theta}^{-1} * nabla_theta L(z, theta*)
```

where `H_{theta}` is the Hessian of the training loss at the optimal parameters. The influence on the loss at a test point `z_test` is:

```
I(z, z_test) = -nabla_theta L(z_test, theta*)^T * H_{theta}^{-1} * nabla_theta L(z, theta*)
```

This allows the attacker to identify which training points have the greatest impact on specific predictions, enabling targeted poisoning with minimal data modification.

### 2.3 Gradient-Based Poisoning

For differentiable models, the attacker can use gradient-based optimization to craft poisoned samples. The approach computes the gradient of the attacker's objective with respect to the poisoned data points:

```
nabla_{x_p} L_atk = nabla_{theta} L_atk * (d theta* / d x_p)
```

The implicit differentiation through the training process yields:

```
d theta* / d x_p = -H_{theta}^{-1} * nabla_{theta, x_p}^2 L_train
```

This gradient tells the attacker how to modify the poisoned samples to maximally degrade (or redirect) model performance. In practice, iterative methods like projected gradient ascent are used to generate poisoned samples while keeping them within plausible data ranges.

## 3. Poisoning Attack Types

### 3.1 Label Flipping

Label flipping is the simplest form of poisoning: the attacker changes the labels of existing training samples while keeping features intact. In a binary trading classifier (buy/sell), flipping a fraction of labels causes the model to learn incorrect associations between market patterns and trading signals.

The effectiveness of label flipping depends on:
- **Flip rate**: Higher rates cause more damage but are easier to detect
- **Strategic selection**: Flipping labels near the decision boundary is more effective than random flipping
- **Class targeting**: Flipping only one class can create systematic directional bias

### 3.2 Feature Manipulation

Feature manipulation modifies the input features of training samples. The attacker can:
- **Shift features**: Add systematic bias to numerical features (e.g., slightly inflate volume data)
- **Inject noise**: Add carefully calibrated noise that degrades model learning
- **Craft adversarial features**: Construct feature vectors that push the decision boundary in a desired direction

In trading, feature manipulation might involve corrupting technical indicators, altering price data within plausible ranges, or injecting synthetic order book patterns.

### 3.3 Backdoor (Trojan) Attacks

Backdoor attacks embed a hidden trigger pattern in the training data. During normal operation, the model behaves correctly. However, when the trigger pattern appears in the input, the model produces an attacker-specified output.

The attack involves:
1. Selecting a trigger pattern (e.g., a specific combination of technical indicator values)
2. Creating training samples with the trigger pattern embedded
3. Assigning these samples the attacker's desired label
4. Injecting them into the training set

In trading, a backdoor trigger could be a specific pattern in tick data or a combination of market microstructure features that the attacker can reproduce at inference time.

### 3.4 Clean-Label Attacks

Clean-label attacks are particularly dangerous because the poisoned samples have correct labels, making them hard to detect through label inspection. The attack works by modifying features so that the sample is correctly labeled but positioned in feature space to distort the decision boundary.

The attacker uses gradient-based optimization to find feature perturbations that:
- Keep the sample within the correct class region (maintaining its label)
- Push the model's learned boundary to misclassify specific target samples

## 4. Trading-Specific Poisoning Vectors

### 4.1 Corrupted Price Feeds

Price feed poisoning targets the most fundamental input to trading models. Attackers with access to data pipelines can introduce subtle modifications:

- **Micro-adjustments**: Shifting prices by fractions of a tick, staying within the bid-ask spread to avoid detection
- **Temporal poisoning**: Altering timestamps of trades to create false patterns in time-series features
- **Volume inflation**: Artificially increasing volume on specific candles to trigger volume-based signals
- **OHLCV manipulation**: Modifying open, high, low, close, or volume values to create false technical patterns

The Bybit API, like all exchange APIs, provides historical kline data that models train on. If an attacker could compromise the data pipeline between the API and the training system, they could inject poisoned candles that teach the model incorrect price action patterns.

### 4.2 Manipulated Historical Data

Historical data repositories are attractive targets because:
- They are often cached and reused across multiple training runs
- Modifications to historical data affect all future models trained on it
- Historical data is rarely re-validated after initial collection

An attacker could target survivorship bias in historical data, inject synthetic securities with crafted return profiles, or modify corporate action adjustments to corrupt fundamental features.

### 4.3 Poisoned Alternative Data

Alternative data sources (sentiment, satellite imagery, web traffic, social media) are particularly vulnerable to poisoning because:
- They lack the standardized validation of exchange data
- Ground truth is often subjective or delayed
- Data collection pipelines may be less secure

Sentiment poisoning through coordinated social media campaigns, fake news injection, or manipulation of review scores can corrupt NLP-based trading models during training.

## 5. Defenses Against Data Poisoning

### 5.1 Data Sanitization

Data sanitization removes suspicious samples before training:

- **Statistical outlier detection**: Remove samples whose features fall outside expected distributions using z-score thresholds, IQR-based methods, or Mahalanobis distance
- **Nearest neighbor analysis**: Flag samples whose labels disagree with their k-nearest neighbors
- **Spectral signatures**: Analyze the spectrum of the feature covariance matrix to detect clusters of poisoned points

### 5.2 Robust Statistics

Robust training methods reduce the impact of poisoned samples:

- **Trimmed loss**: Discard the highest-loss samples during training (they are likely poisoned)
- **Median-based estimators**: Replace mean-based aggregations with median-based alternatives
- **RANSAC-style training**: Train on random subsets and select the model with best validation performance
- **Influence-based pruning**: Use influence functions to identify and remove training points with outsized impact

### 5.3 Anomaly Detection in Training Data

Continuous monitoring of training data can detect poisoning in real-time:

- **Distribution shift detection**: Monitor feature distributions across training batches
- **Label consistency checks**: Cross-validate labels using multiple independent sources
- **Temporal coherence**: Verify that time-series data maintains expected autocorrelation and stationarity properties
- **Cross-source validation**: Compare data from multiple providers to detect discrepancies

## 6. Implementation Walkthrough

Our Rust implementation provides a complete framework for studying and defending against poisoning attacks in trading systems. The library is organized around several core components:

### 6.1 Data Structures

We define `Sample` as a pair of feature vector and label, and `Dataset` as a collection of samples. The `PoisonConfig` struct controls attack parameters including the poisoning rate, attack type, and target specifications.

### 6.2 Attack Implementation

The `LabelFlipAttack` randomly selects a fraction of training samples and flips their binary labels. This simulates an attacker who has write access to the label column of the training database.

The `FeaturePoisonAttack` injects crafted samples with adversarial feature vectors. We use a simplified gradient approximation to determine the most damaging feature modifications, then project them back into the valid feature range.

The `BackdoorAttack` embeds a trigger pattern (a specific combination of feature values) into a subset of training samples and assigns them the attacker's target label. At inference time, any sample containing the trigger will be classified according to the attacker's intent.

### 6.3 Defense Implementation

The `DataSanitizer` implements outlier-based defense. For each feature dimension, it computes the mean and standard deviation, then removes samples that exceed a configurable z-score threshold. This effectively strips the most extreme poisoned samples from the dataset.

The `PoisonDetector` uses loss-based inspection: after training on the full dataset, it computes per-sample loss and flags samples in the top percentile as potentially poisoned. These high-loss samples are the ones the model struggles to fit — often because they are mislabeled or adversarial.

### 6.4 Model Training and Evaluation

We implement a simple linear classifier with gradient descent training. While production trading models would use more sophisticated architectures, the linear model clearly demonstrates the impact of poisoning and the effectiveness of defenses. Model accuracy is measured on a held-out test set under four conditions:
1. Clean training data (baseline)
2. Poisoned training data (attack impact)
3. Sanitized training data (defense effectiveness)
4. Various poisoning rates (degradation curve)

## 7. Bybit Data Integration

The implementation includes a Bybit API client that fetches historical kline (candlestick) data for any trading pair. We use the public `/v5/market/kline` endpoint, which provides OHLCV data at configurable intervals.

The fetched data is transformed into a feature matrix suitable for our binary classifier. Features include:
- **Return**: `(close - open) / open`
- **Range**: `(high - low) / open`
- **Body ratio**: `|close - open| / (high - low)`
- **Volume change**: Relative volume compared to the moving average

The label is derived from the next candle's direction: 1 if the next close is higher than the current close (bullish), 0 otherwise (bearish).

This pipeline demonstrates a realistic scenario where an attacker could poison the training data by intercepting or modifying the API response before it reaches the model training system.

The trading example fetches BTCUSDT data from Bybit, constructs clean and poisoned datasets at 5%, 10%, and 20% poisoning rates, trains separate models on each, and reports the accuracy degradation. It then applies the data sanitization defense and shows the recovered accuracy.

## 8. Key Takeaways

1. **Data poisoning is a first-order threat** to ML-based trading systems. Unlike inference-time attacks, poisoning corrupts the model at its foundation, affecting all future predictions until the model is retrained on clean data.

2. **The bilevel optimization framework** provides a rigorous mathematical foundation for understanding poisoning attacks. The attacker optimizes poisoned data subject to the learner's optimal response.

3. **Multiple attack types exist** with different trade-offs: label flipping is simple but detectable; backdoor attacks are stealthy but require trigger control; clean-label attacks evade label inspection but require sophisticated optimization.

4. **Trading systems are uniquely vulnerable** due to their dependence on external data sources (exchange feeds, alternative data) and continuous retraining schedules that create ongoing injection opportunities.

5. **Defense requires multiple layers**: statistical outlier removal catches gross poisoning, loss-based inspection identifies subtle attacks, and cross-source validation provides an independent check.

6. **Influence functions** enable both attack and defense: attackers use them to identify high-impact poisoning points, while defenders use them to identify and remove suspicious training samples.

7. **Robust training methods** (trimmed loss, median estimators) provide inherent resilience against moderate poisoning rates without requiring explicit detection.

8. **Continuous monitoring** of training data distributions, label consistency, and model performance is essential for detecting poisoning attacks in production trading systems.

9. **The cost-benefit analysis** of poisoning defense must consider that even small accuracy degradations in trading can translate to significant financial losses, justifying substantial investment in data integrity.

10. **No single defense is sufficient**. A comprehensive anti-poisoning strategy combines data sanitization, robust training, anomaly detection, and operational security measures (access controls, data provenance tracking, pipeline integrity verification).
