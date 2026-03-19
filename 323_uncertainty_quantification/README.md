# Chapter 323: Uncertainty Quantification for Trading

## Introduction: Why Uncertainty Matters in Trading Decisions

Financial markets are inherently uncertain. Every trading decision involves a prediction about an unknowable future, yet most machine learning models produce point estimates that convey false precision. A model might predict that BTCUSDT will reach $72,000 tomorrow, but how confident is it? Is the model equally certain about this prediction as it was about yesterday's prediction that turned out to be correct? Without quantifying uncertainty, traders are flying blind.

Uncertainty quantification (UQ) transforms raw predictions into actionable intelligence by attaching confidence measures to every forecast. Instead of a single price target, UQ provides a distribution of possible outcomes. This enables risk-aware position sizing, dynamic stop-loss placement, and informed decisions about when to trade and when to stay on the sidelines. A model that says "I predict the price will go up, but I'm very unsure" communicates something fundamentally different from one that says "I predict the price will go up, and I'm highly confident."

In traditional finance, the concept of uncertainty has always been present through measures like volatility and Value at Risk. However, these approaches typically capture only one dimension of uncertainty -- the inherent randomness in market data. Modern machine learning methods allow us to decompose uncertainty into its constituent parts and estimate each one separately, providing a richer picture for decision-making.

The practical implications are significant. Research has shown that uncertainty-aware trading strategies can reduce maximum drawdown by 20-40% compared to strategies that ignore prediction confidence. By scaling position sizes inversely with uncertainty, traders naturally become more conservative during volatile or unusual market conditions where models are least reliable. This chapter explores the theory, methods, and Rust implementation of uncertainty quantification for cryptocurrency trading.

## Epistemic vs Aleatoric Uncertainty

Understanding the two fundamental types of uncertainty is crucial for building effective trading systems.

### Aleatoric Uncertainty (Data Noise)

Aleatoric uncertainty, also called data uncertainty or irreducible uncertainty, arises from inherent randomness in the data-generating process. In financial markets, this includes:

- **Microstructure noise**: Random fluctuations from order flow, bid-ask bounce, and market maker activity
- **News-driven jumps**: Unexpected events that no model could predict from historical data alone
- **Regime changes**: Structural shifts in market dynamics that fundamentally alter price behavior
- **Liquidity variations**: Random changes in market depth that affect price impact

Aleatoric uncertainty cannot be reduced by collecting more data or building better models. It represents a fundamental limit on predictability. A well-calibrated model should acknowledge this noise rather than trying to fit it.

In practice, aleatoric uncertainty can be estimated by having the model output not just a mean prediction but also a variance parameter. For a Gaussian output, the model predicts both mu (the expected value) and sigma squared (the variance), where sigma squared captures the data noise at each input point.

### Epistemic Uncertainty (Model Uncertainty)

Epistemic uncertainty, also called model uncertainty or reducible uncertainty, arises from limitations in the model itself. This includes:

- **Limited training data**: The model has not seen enough examples to learn the true relationship
- **Out-of-distribution inputs**: The current market conditions differ significantly from training data
- **Model misspecification**: The model architecture cannot capture the true complexity of the relationship
- **Parameter uncertainty**: Multiple parameter settings could explain the training data equally well

Unlike aleatoric uncertainty, epistemic uncertainty can be reduced by gathering more data, improving the model architecture, or using better training procedures. It is highest in regions of the input space that are underrepresented in the training data.

For trading, epistemic uncertainty is particularly valuable because it signals when the model is operating outside its area of competence. During a market regime the model has never seen, epistemic uncertainty should spike, warning the trader to reduce position sizes or avoid trading altogether.

### Separating the Two Types

The total predictive uncertainty is the sum of aleatoric and epistemic components:

```
Total Uncertainty = Aleatoric Uncertainty + Epistemic Uncertainty
```

Deep Ensembles and Bayesian methods naturally decompose uncertainty this way. The aleatoric component is captured by the mean variance predicted across ensemble members, while the epistemic component is captured by the variance of the means across ensemble members. This decomposition provides actionable information: high aleatoric uncertainty suggests the market itself is unpredictable at this point, while high epistemic uncertainty suggests the model needs more training data in this regime.

## Methods for Uncertainty Quantification

### Monte Carlo Dropout

Monte Carlo (MC) Dropout is the simplest approach to uncertainty quantification. During training, dropout randomly deactivates neurons to prevent overfitting. The key insight from Gal and Ghahramani (2016) is that keeping dropout active during inference and running the model multiple times produces different predictions each time. The spread of these predictions approximates Bayesian uncertainty.

**Algorithm:**
1. Train a neural network with dropout layers as usual
2. At inference time, keep dropout active (do not switch to eval mode)
3. Run the same input through the model T times (typically T = 50-100)
4. Compute the mean prediction as the final forecast
5. Compute the variance of predictions as the uncertainty estimate

**Advantages:**
- Simple to implement -- just keep dropout on during inference
- Works with any existing dropout-trained model
- Computationally cheap compared to full Bayesian inference

**Limitations:**
- Uncertainty estimates depend on dropout rate and placement
- May underestimate uncertainty compared to full Bayesian methods
- Quality of uncertainty depends on the specific dropout architecture

### Deep Ensembles

Deep Ensembles, introduced by Lakshminarayanan et al. (2017), train multiple neural networks independently with different random initializations. Each model converges to a different local minimum in the loss landscape, and the disagreement between models captures epistemic uncertainty.

**Algorithm:**
1. Train M independent models (typically M = 5-10) with different random seeds
2. Each model predicts both mean and variance (capturing aleatoric uncertainty)
3. The final prediction is the average of individual means
4. Epistemic uncertainty is the variance of the individual means
5. Aleatoric uncertainty is the average of individual variances
6. Total uncertainty is the sum of both components

**Advantages:**
- State-of-the-art uncertainty estimates in practice
- Naturally decomposes aleatoric and epistemic uncertainty
- Models can be trained in parallel
- No special architecture modifications needed

**Limitations:**
- Requires training multiple models (M times the computational cost)
- Storage requirements scale linearly with ensemble size

### Bayesian Neural Networks

Bayesian Neural Networks (BNNs) place probability distributions over network weights instead of learning point estimates. Instead of a single weight value, each weight is characterized by a mean and variance, representing belief about the true parameter value.

**Key concepts:**
- **Prior**: Initial belief about weight distributions (before seeing data)
- **Likelihood**: How well the weights explain the observed data
- **Posterior**: Updated belief after observing data (via Bayes' theorem)

Full Bayesian inference over neural network weights is intractable for large networks. Practical approximations include Variational Inference (learning an approximate posterior) and Stochastic Weight Averaging Gaussian (SWAG), which fits a Gaussian to the trajectory of SGD iterates.

## Conformal Prediction for Trading Signals

Conformal prediction provides distribution-free prediction intervals with guaranteed coverage. Unlike parametric methods, it makes no assumptions about the underlying data distribution.

**Algorithm (Split Conformal):**
1. Split data into training set and calibration set
2. Train the model on the training set
3. Compute nonconformity scores on calibration set: scores = |y_actual - y_predicted|
4. For desired coverage level (1 - alpha), find the (1 - alpha) quantile of scores
5. At test time, prediction interval = [y_pred - q, y_pred + q]

**For trading applications:**
- **Adaptive intervals**: Intervals widen during volatile periods and narrow during calm periods when using locally-weighted conformal methods
- **Coverage guarantee**: Over time, exactly (1 - alpha) fraction of true values fall within the interval
- **Signal generation**: Only trade when the prediction interval does not contain zero (confident directional signal)

Conformal prediction is particularly appealing for regulatory compliance because it provides statistical guarantees without distributional assumptions.

## Implementation with Rust

We implement the core uncertainty quantification methods in Rust for performance-critical trading applications. The implementation includes MC Dropout simulation, Deep Ensembles, conformal prediction, and Bybit API integration.

### Core Data Structures

```rust
pub struct PredictionWithUncertainty {
    pub mean: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub total_uncertainty: f64,
    pub confidence_interval: (f64, f64),
    pub confidence_level: f64,
}

pub struct EnsembleMember {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub seed: u64,
}
```

### MC Dropout Implementation

```rust
pub fn mc_dropout_predict(
    input: &[f64],
    weights: &[Vec<f64>],
    dropout_rate: f64,
    n_samples: usize,
) -> PredictionWithUncertainty {
    let mut predictions = Vec::with_capacity(n_samples);
    let mut rng = rand::thread_rng();

    for _ in 0..n_samples {
        // Apply random dropout mask
        let masked: Vec<f64> = weights.iter()
            .map(|w| {
                let dot: f64 = w.iter().zip(input.iter())
                    .map(|(wi, xi)| {
                        if rng.gen::<f64>() > dropout_rate {
                            wi * xi / (1.0 - dropout_rate)
                        } else {
                            0.0
                        }
                    })
                    .sum();
                dot
            })
            .collect();
        let pred: f64 = masked.iter().sum::<f64>() / masked.len() as f64;
        predictions.push(pred);
    }

    compute_uncertainty(&predictions, 0.95)
}
```

### Deep Ensemble Prediction

```rust
pub fn ensemble_predict(
    input: &[f64],
    members: &[EnsembleMember],
) -> PredictionWithUncertainty {
    let predictions: Vec<f64> = members.iter()
        .map(|m| {
            m.weights.iter()
                .map(|w| w.iter().zip(input).map(|(a, b)| a * b).sum::<f64>())
                .sum::<f64>() / m.weights.len() as f64
                + m.biases.iter().sum::<f64>() / m.biases.len() as f64
        })
        .collect();

    compute_uncertainty(&predictions, 0.95)
}
```

### Conformal Prediction

```rust
pub fn conformal_intervals(
    calibration_errors: &[f64],
    prediction: f64,
    alpha: f64,
) -> (f64, f64) {
    let mut sorted = calibration_errors.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize;
    let quantile = sorted[idx.min(sorted.len() - 1)];
    (prediction - quantile, prediction + quantile)
}
```

## Bybit Crypto Data Integration

The implementation fetches real OHLCV data from Bybit's public API for BTCUSDT and ETHUSDT daily candles. This data serves as the foundation for training ensemble models and generating uncertainty-aware predictions.

```rust
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp = reqwest::get(&url).await?.json::<BybitResponse>().await?;
    // Parse response into Candle structs
    Ok(candles)
}
```

The API returns standard OHLCV data which we transform into features for the ensemble model:

- **Returns**: Log returns over 1, 5, and 20 periods
- **Volatility**: Rolling standard deviation of returns
- **Volume ratio**: Current volume relative to moving average
- **Price position**: Where current price sits within recent high-low range

## Risk-Adjusted Trading with Uncertainty Estimates

Uncertainty estimates enable sophisticated risk management that goes beyond simple stop-losses.

### Position Sizing

The Kelly Criterion can be extended with uncertainty:

```
position_size = base_size * (1 - normalized_uncertainty)
```

When uncertainty is high, position sizes shrink. When the model is confident, positions approach the base size. This naturally implements a "trade less when unsure" policy.

### Signal Filtering

Only act on signals where the prediction interval has a clear directional bias:

```rust
fn should_trade(prediction: &PredictionWithUncertainty) -> bool {
    let (lower, upper) = prediction.confidence_interval;
    // Only trade if entire interval is on one side of zero
    (lower > 0.0 && upper > 0.0) || (lower < 0.0 && upper < 0.0)
}
```

### Dynamic Stop-Losses

Set stop-loss levels based on the width of the prediction interval:

```
stop_loss = entry_price - k * total_uncertainty
```

Where k is a risk tolerance parameter. Wider uncertainty leads to wider stops, preventing premature exit during genuinely uncertain conditions.

### Regime Detection

A sudden spike in epistemic uncertainty can signal a regime change. When the model encounters market conditions unlike anything in its training data, epistemic uncertainty rises sharply. This can trigger:

- Reduced position sizes across all strategies
- Increased hedging ratios
- Manual review alerts for the trading team

## Key Takeaways

1. **Uncertainty quantification transforms point predictions into probability distributions**, enabling risk-aware decision-making that accounts for model confidence.

2. **Aleatoric and epistemic uncertainty serve different purposes**: aleatoric uncertainty captures irreducible market noise, while epistemic uncertainty signals when the model is operating outside its competence zone.

3. **Deep Ensembles provide the best practical uncertainty estimates** by training multiple models and measuring their disagreement. They naturally decompose total uncertainty into its aleatoric and epistemic components.

4. **MC Dropout offers a lightweight alternative** that requires no architectural changes -- simply keep dropout active during inference and run multiple forward passes.

5. **Conformal prediction provides distribution-free guarantees** on prediction interval coverage, making it suitable for applications where statistical guarantees are required.

6. **Position sizing should be inversely proportional to uncertainty**. This naturally implements a "trade less when unsure" policy that reduces drawdowns during regime changes and unusual market conditions.

7. **Epistemic uncertainty spikes serve as regime change detectors**. When the model encounters market conditions far from its training distribution, epistemic uncertainty rises sharply, providing an early warning signal.

8. **Rust implementation enables real-time uncertainty quantification** with the performance needed for live trading systems, while Bybit API integration provides access to cryptocurrency market data for BTCUSDT and ETHUSDT pairs.

9. **Combining multiple UQ methods** (ensembles for prediction, conformal for calibration) provides the most robust uncertainty estimates for production trading systems.

10. **The value of uncertainty quantification compounds over time**. Even small improvements in position sizing and signal filtering from uncertainty awareness lead to significantly better risk-adjusted returns over hundreds or thousands of trades.
