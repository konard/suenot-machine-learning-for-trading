//! # Robust Trading Models
//!
//! A library for building ML trading models that are robust to market regime changes,
//! distribution shifts, and adversarial conditions.
//!
//! Key components:
//! - Standard model training (ERM baseline)
//! - DRO-inspired robust training with worst-case sample weighting
//! - Input noise injection for perturbation robustness
//! - Ensemble with diversity encouragement
//! - Market regime simulation and cross-regime evaluation
//! - Bybit API data fetching

use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Represents a single OHLCV candle from market data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    Crash,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Bull => write!(f, "Bull"),
            MarketRegime::Bear => write!(f, "Bear"),
            MarketRegime::Sideways => write!(f, "Sideways"),
            MarketRegime::Crash => write!(f, "Crash"),
        }
    }
}

/// Configuration for a market regime simulation.
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    pub regime: MarketRegime,
    pub drift: f64,
    pub volatility: f64,
    pub autocorrelation: f64,
}

/// Feature vector with label for supervised learning.
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: f64, // 1.0 for up, -1.0 for down
}

/// A simple linear model: prediction = sign(weights . features + bias).
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

/// An ensemble of linear models.
#[derive(Debug, Clone)]
pub struct EnsembleModel {
    pub models: Vec<LinearModel>,
}

/// Results from robustness evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessReport {
    pub clean_accuracy: f64,
    pub perturbed_accuracy: f64,
    pub regime_accuracies: Vec<(String, f64)>,
    pub avg_regime_accuracy: f64,
    pub worst_regime_accuracy: f64,
    pub robustness_gap: f64,
}

impl std::fmt::Display for RobustnessReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Clean accuracy:        {:.2}%", self.clean_accuracy * 100.0)?;
        writeln!(f, "  Perturbed accuracy:    {:.2}%", self.perturbed_accuracy * 100.0)?;
        for (regime, acc) in &self.regime_accuracies {
            writeln!(f, "  {:>10} accuracy:  {:.2}%", regime, acc * 100.0)?;
        }
        writeln!(f, "  Avg regime accuracy:   {:.2}%", self.avg_regime_accuracy * 100.0)?;
        writeln!(f, "  Worst regime accuracy: {:.2}%", self.worst_regime_accuracy * 100.0)?;
        writeln!(f, "  Robustness gap:        {:.2}%", self.robustness_gap * 100.0)?;
        Ok(())
    }
}

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Market Regime Simulator
// ---------------------------------------------------------------------------

/// Returns default configurations for all four market regimes.
pub fn default_regime_configs() -> Vec<RegimeConfig> {
    vec![
        RegimeConfig {
            regime: MarketRegime::Bull,
            drift: 0.001,
            volatility: 0.01,
            autocorrelation: 0.05,
        },
        RegimeConfig {
            regime: MarketRegime::Bear,
            drift: -0.001,
            volatility: 0.02,
            autocorrelation: -0.05,
        },
        RegimeConfig {
            regime: MarketRegime::Sideways,
            drift: 0.0,
            volatility: 0.005,
            autocorrelation: -0.3,
        },
        RegimeConfig {
            regime: MarketRegime::Crash,
            drift: -0.005,
            volatility: 0.05,
            autocorrelation: 0.1,
        },
    ]
}

/// Simulate price series for a given regime.
pub fn simulate_regime(config: &RegimeConfig, n_steps: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut prices = Vec::with_capacity(n_steps);
    let mut price = 100.0;
    let mut prev_return = 0.0;

    prices.push(price);
    for _ in 1..n_steps {
        // Box-Muller approximation using uniform samples
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen::<f64>();
        let normal: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();

        let ret = config.drift
            + config.autocorrelation * prev_return
            + config.volatility * normal;
        price *= (1.0 + ret).max(0.01);
        prices.push(price);
        prev_return = ret;
    }
    prices
}

/// Generate labeled samples from a price series.
pub fn prices_to_samples(prices: &[f64], lookback: usize) -> Vec<Sample> {
    let mut samples = Vec::new();
    if prices.len() < lookback + 2 {
        return samples;
    }

    // Compute log returns
    let log_returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    for i in lookback..log_returns.len() - 1 {
        let window = &log_returns[i - lookback..i];

        // Feature 1: momentum (sum of recent returns)
        let momentum: f64 = window.iter().sum();

        // Feature 2: volatility (std dev of recent returns)
        let mean = momentum / lookback as f64;
        let variance = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / lookback as f64;
        let volatility = variance.sqrt();

        // Feature 3: last return
        let last_return = log_returns[i];

        // Feature 4: mean reversion signal (deviation from rolling mean price)
        let avg_price: f64 = prices[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        let mean_rev = (prices[i] - avg_price) / avg_price;

        // Feature 5: range ratio
        let max_p = prices[i - lookback..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_p = prices[i - lookback..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let range_ratio = if max_p > 0.0 {
            (max_p - min_p) / max_p
        } else {
            0.0
        };

        // Label: future return direction
        let future_return = log_returns[i + 1];
        let label = if future_return >= 0.0 { 1.0 } else { -1.0 };

        samples.push(Sample {
            features: vec![momentum, volatility, last_return, mean_rev, range_ratio],
            label,
        });
    }
    samples
}

// ---------------------------------------------------------------------------
// Feature Engineering (robust)
// ---------------------------------------------------------------------------

/// Apply rank transformation to features (column-wise).
pub fn rank_transform(samples: &mut [Sample]) {
    if samples.is_empty() {
        return;
    }
    let n_features = samples[0].features.len();
    let n = samples.len();

    for feat_idx in 0..n_features {
        // Extract feature column with original indices
        let mut indexed: Vec<(usize, f64)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.features[feat_idx]))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks normalized to [-1, 1]
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            let normalized_rank = 2.0 * (rank as f64) / (n as f64 - 1.0).max(1.0) - 1.0;
            samples[*orig_idx].features[feat_idx] = normalized_rank;
        }
    }
}

/// Clip features to [-clip_val, clip_val].
pub fn clip_features(samples: &mut [Sample], clip_val: f64) {
    for sample in samples.iter_mut() {
        for feat in sample.features.iter_mut() {
            *feat = feat.clamp(-clip_val, clip_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Standard Model Training (ERM)
// ---------------------------------------------------------------------------

impl LinearModel {
    /// Create a new model with random weights.
    pub fn new_random(n_features: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let weights: Vec<f64> = (0..n_features)
            .map(|_| (rng.gen::<f64>() - 0.5) * 0.1)
            .collect();
        LinearModel {
            weights,
            bias: 0.0,
        }
    }

    /// Raw prediction score (before sign).
    pub fn score(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias
    }

    /// Binary prediction: +1 or -1.
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.score(features) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Compute accuracy on a dataset.
    pub fn accuracy(&self, samples: &[Sample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples
            .iter()
            .filter(|s| self.predict(&s.features) == s.label)
            .count();
        correct as f64 / samples.len() as f64
    }

    /// Train via standard ERM using stochastic gradient descent.
    /// Uses hinge loss: L = max(0, 1 - y * (w . x + b)).
    pub fn train_standard(
        samples: &[Sample],
        n_features: usize,
        learning_rate: f64,
        epochs: usize,
        seed: u64,
    ) -> Self {
        let mut model = LinearModel::new_random(n_features, seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 100);

        for _epoch in 0..epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for &idx in &indices {
                let sample = &samples[idx];
                let margin = sample.label * model.score(&sample.features);

                if margin < 1.0 {
                    // Hinge loss gradient
                    for (w, x) in model.weights.iter_mut().zip(sample.features.iter()) {
                        *w += learning_rate * sample.label * x;
                    }
                    model.bias += learning_rate * sample.label;
                }
            }
        }
        model
    }
}

// ---------------------------------------------------------------------------
// DRO-Inspired Robust Training (worst-case sample weighting)
// ---------------------------------------------------------------------------

/// Train a model using DRO-inspired worst-case sample weighting.
///
/// High-loss samples receive higher weights (CVaR-style), forcing the model
/// to pay attention to difficult cases.
pub fn train_robust_dro(
    samples: &[Sample],
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
    alpha: f64, // CVaR level: fraction of worst samples to focus on (e.g., 0.3)
    seed: u64,
) -> LinearModel {
    let mut model = LinearModel::new_random(n_features, seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 200);

    for _epoch in 0..epochs {
        // Compute losses for all samples
        let mut losses: Vec<(usize, f64)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let margin = s.label * model.score(&s.features);
                let loss = (1.0 - margin).max(0.0); // hinge loss
                (i, loss)
            })
            .collect();

        // Sort by loss descending
        losses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the worst alpha fraction
        let k = ((samples.len() as f64 * alpha).ceil() as usize).max(1);
        let worst_indices: Vec<usize> = losses.iter().take(k).map(|(i, _)| *i).collect();

        // Shuffle the worst-case samples
        let mut worst_shuffled = worst_indices;
        for i in (1..worst_shuffled.len()).rev() {
            let j = rng.gen_range(0..=i);
            worst_shuffled.swap(i, j);
        }

        // Update on worst-case samples with higher weight
        for &idx in &worst_shuffled {
            let sample = &samples[idx];
            let margin = sample.label * model.score(&sample.features);

            if margin < 1.0 {
                let weight = 1.0 / alpha; // up-weight worst-case samples
                for (w, x) in model.weights.iter_mut().zip(sample.features.iter()) {
                    *w += learning_rate * weight * sample.label * x;
                }
                model.bias += learning_rate * weight * sample.label;
            }
        }

        // Also do a standard pass with lower weight to maintain average performance
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        for &idx in &indices {
            let sample = &samples[idx];
            let margin = sample.label * model.score(&sample.features);
            if margin < 1.0 {
                for (w, x) in model.weights.iter_mut().zip(sample.features.iter()) {
                    *w += learning_rate * 0.5 * sample.label * x;
                }
                model.bias += learning_rate * 0.5 * sample.label;
            }
        }
    }
    model
}

// ---------------------------------------------------------------------------
// Input Noise Injection Training
// ---------------------------------------------------------------------------

/// Train a model with Gaussian noise injected into features at each step.
/// This encourages the model to be robust to input perturbations.
pub fn train_with_noise_injection(
    samples: &[Sample],
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
    noise_std: f64,
    seed: u64,
) -> LinearModel {
    let mut model = LinearModel::new_random(n_features, seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 300);

    for _epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let sample = &samples[idx];

            // Add Gaussian noise to features
            let noisy_features: Vec<f64> = sample
                .features
                .iter()
                .map(|x| {
                    let u1: f64 = rng.gen::<f64>().max(1e-10);
                    let u2: f64 = rng.gen::<f64>();
                    let normal: f64 =
                        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
                    x + noise_std * normal
                })
                .collect();

            let score: f64 = model
                .weights
                .iter()
                .zip(noisy_features.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + model.bias;
            let margin = sample.label * score;

            if margin < 1.0 {
                for (w, x) in model.weights.iter_mut().zip(noisy_features.iter()) {
                    *w += learning_rate * sample.label * x;
                }
                model.bias += learning_rate * sample.label;
            }
        }
    }
    model
}

// ---------------------------------------------------------------------------
// Ensemble with Diversity Encouragement
// ---------------------------------------------------------------------------

impl EnsembleModel {
    /// Train an ensemble with diversity encouragement.
    ///
    /// Each model is trained on a different bootstrap sample and with
    /// different random seeds. Diversity is encouraged by penalizing
    /// agreement with existing ensemble members on training data.
    pub fn train_diverse(
        samples: &[Sample],
        n_features: usize,
        n_models: usize,
        learning_rate: f64,
        epochs: usize,
        diversity_weight: f64,
        seed: u64,
    ) -> Self {
        let mut models = Vec::with_capacity(n_models);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 400);

        for m in 0..n_models {
            // Bootstrap sample
            let bootstrap: Vec<Sample> = (0..samples.len())
                .map(|_| {
                    let idx = rng.gen_range(0..samples.len());
                    samples[idx].clone()
                })
                .collect();

            // If we have previous models, create diversity-adjusted samples
            let adjusted_samples = if !models.is_empty() && diversity_weight > 0.0 {
                bootstrap
                    .iter()
                    .map(|s| {
                        // Check if existing ensemble agrees
                        let ensemble_pred: f64 = models
                            .iter()
                            .map(|model: &LinearModel| model.predict(&s.features))
                            .sum::<f64>()
                            / models.len() as f64;

                        // If ensemble is confident and correct, reduce label magnitude
                        // to encourage the new model to focus on samples where ensemble is weak
                        let agreement = ensemble_pred * s.label;
                        let adjusted_label = if agreement > 0.5 {
                            s.label * (1.0 - diversity_weight)
                        } else {
                            s.label * (1.0 + diversity_weight)
                        };

                        Sample {
                            features: s.features.clone(),
                            label: adjusted_label.signum(), // keep as +1/-1
                        }
                    })
                    .collect()
            } else {
                bootstrap
            };

            let model = LinearModel::train_standard(
                &adjusted_samples,
                n_features,
                learning_rate,
                epochs,
                seed + m as u64 * 1000,
            );
            models.push(model);
        }

        EnsembleModel { models }
    }

    /// Predict using majority vote.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let sum: f64 = self
            .models
            .iter()
            .map(|m| m.predict(features))
            .sum();
        if sum >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Compute accuracy on a dataset.
    pub fn accuracy(&self, samples: &[Sample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples
            .iter()
            .filter(|s| self.predict(&s.features) == s.label)
            .count();
        correct as f64 / samples.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Robustness Evaluation
// ---------------------------------------------------------------------------

/// Add Gaussian perturbation to sample features.
pub fn perturb_samples(samples: &[Sample], noise_std: f64, seed: u64) -> Vec<Sample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    samples
        .iter()
        .map(|s| {
            let noisy_features: Vec<f64> = s
                .features
                .iter()
                .map(|x| {
                    let u1: f64 = rng.gen::<f64>().max(1e-10);
                    let u2: f64 = rng.gen::<f64>();
                    let normal: f64 =
                        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
                    x + noise_std * normal
                })
                .collect();
            Sample {
                features: noisy_features,
                label: s.label,
            }
        })
        .collect()
}

/// Evaluate a model's robustness across multiple dimensions.
///
/// The `predict_fn` takes a feature slice and returns +1 or -1.
pub fn evaluate_robustness<F>(
    predict_fn: &F,
    test_samples: &[Sample],
    regime_configs: &[RegimeConfig],
    perturbation_std: f64,
    seed: u64,
) -> RobustnessReport
where
    F: Fn(&[f64]) -> f64,
{
    // Clean accuracy
    let clean_correct = test_samples
        .iter()
        .filter(|s| predict_fn(&s.features) == s.label)
        .count();
    let clean_accuracy = clean_correct as f64 / test_samples.len().max(1) as f64;

    // Perturbed accuracy
    let perturbed = perturb_samples(test_samples, perturbation_std, seed + 500);
    let perturbed_correct = perturbed
        .iter()
        .filter(|s| predict_fn(&s.features) == s.label)
        .count();
    let perturbed_accuracy = perturbed_correct as f64 / perturbed.len().max(1) as f64;

    // Per-regime accuracy
    let mut regime_accuracies = Vec::new();
    for config in regime_configs {
        let prices = simulate_regime(config, 500, seed + 600);
        let mut regime_samples = prices_to_samples(&prices, 20);
        rank_transform(&mut regime_samples);

        if regime_samples.is_empty() {
            regime_accuracies.push((config.regime.to_string(), 0.5));
            continue;
        }

        let correct = regime_samples
            .iter()
            .filter(|s| predict_fn(&s.features) == s.label)
            .count();
        let acc = correct as f64 / regime_samples.len() as f64;
        regime_accuracies.push((config.regime.to_string(), acc));
    }

    let avg_regime = regime_accuracies.iter().map(|(_, a)| a).sum::<f64>()
        / regime_accuracies.len().max(1) as f64;
    let worst_regime = regime_accuracies
        .iter()
        .map(|(_, a)| *a)
        .fold(f64::INFINITY, f64::min);

    let robustness_gap = clean_accuracy - worst_regime;

    RobustnessReport {
        clean_accuracy,
        perturbed_accuracy,
        regime_accuracies,
        avg_regime_accuracy: avg_regime,
        worst_regime_accuracy: worst_regime,
        robustness_gap,
    }
}

// ---------------------------------------------------------------------------
// Cross-Regime Generalization Metrics
// ---------------------------------------------------------------------------

/// Train on one set of regimes and evaluate on all regimes.
/// Returns a matrix of (train_regime, test_regime) -> accuracy.
pub fn cross_regime_matrix<F>(
    train_fn: &F,
    regime_configs: &[RegimeConfig],
    n_train: usize,
    n_test: usize,
    lookback: usize,
    seed: u64,
) -> Vec<Vec<(String, String, f64)>>
where
    F: Fn(&[Sample]) -> Box<dyn Fn(&[f64]) -> f64>,
{
    let mut results = Vec::new();

    for (i, train_config) in regime_configs.iter().enumerate() {
        let train_prices = simulate_regime(train_config, n_train, seed + i as u64 * 100);
        let mut train_samples = prices_to_samples(&train_prices, lookback);
        rank_transform(&mut train_samples);

        let predictor = train_fn(&train_samples);

        let mut row = Vec::new();
        for (j, test_config) in regime_configs.iter().enumerate() {
            let test_prices =
                simulate_regime(test_config, n_test, seed + 10000 + j as u64 * 100);
            let mut test_samples = prices_to_samples(&test_prices, lookback);
            rank_transform(&mut test_samples);

            let acc = if test_samples.is_empty() {
                0.5
            } else {
                let correct = test_samples
                    .iter()
                    .filter(|s| predictor(&s.features) == s.label)
                    .count();
                correct as f64 / test_samples.len() as f64
            };

            row.push((
                train_config.regime.to_string(),
                test_config.regime.to_string(),
                acc,
            ));
        }
        results.push(row);
    }
    results
}

// ---------------------------------------------------------------------------
// Bybit API Data Fetching
// ---------------------------------------------------------------------------

/// Fetch OHLCV candles from Bybit V5 API.
///
/// # Arguments
/// * `symbol` - Trading pair, e.g., "BTCUSDT"
/// * `interval` - Candle interval: "1", "5", "15", "60", "240", "D"
/// * `limit` - Number of candles (max 200)
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Convert Bybit candles to price vector (close prices).
pub fn candles_to_prices(candles: &[Candle]) -> Vec<f64> {
    candles.iter().map(|c| c.close).collect()
}

/// Compute features from candles with robust engineering.
pub fn candles_to_samples(candles: &[Candle], lookback: usize) -> Vec<Sample> {
    if candles.len() < lookback + 2 {
        return Vec::new();
    }

    let prices = candles_to_prices(candles);
    let log_returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

    let mut samples = Vec::new();

    for i in lookback..log_returns.len() - 1 {
        let window = &log_returns[i - lookback..i];

        // Momentum
        let momentum: f64 = window.iter().sum();

        // Volatility
        let mean = momentum / lookback as f64;
        let variance = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / lookback as f64;
        let volatility = variance.sqrt();

        // Last return
        let last_return = log_returns[i];

        // Volume ratio
        let avg_volume: f64 = candles[i - lookback..i].iter().map(|c| c.volume).sum::<f64>()
            / lookback as f64;
        let volume_ratio = if avg_volume > 0.0 {
            candles[i].volume / avg_volume
        } else {
            1.0
        };

        // High-low range
        let range = (candles[i].high - candles[i].low) / candles[i].close.max(1e-10);

        // Label
        let future_return = log_returns[i + 1];
        let label = if future_return >= 0.0 { 1.0 } else { -1.0 };

        samples.push(Sample {
            features: vec![momentum, volatility, last_return, volume_ratio, range],
            label,
        });
    }

    samples
}

/// Label market regimes from candle data based on rolling statistics.
pub fn label_regimes(candles: &[Candle], lookback: usize) -> Vec<(usize, MarketRegime)> {
    if candles.len() < lookback + 1 {
        return Vec::new();
    }

    let prices = candles_to_prices(candles);
    let log_returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

    // Compute rolling return and volatility
    let mut rolling_stats: Vec<(f64, f64)> = Vec::new(); // (return, vol)
    for i in lookback..log_returns.len() {
        let window = &log_returns[i - lookback..i];
        let ret: f64 = window.iter().sum();
        let mean = ret / lookback as f64;
        let var = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / lookback as f64;
        rolling_stats.push((ret, var.sqrt()));
    }

    // Compute median volatility for thresholding
    let mut vols: Vec<f64> = rolling_stats.iter().map(|(_, v)| *v).collect();
    vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_vol = if vols.is_empty() {
        0.0
    } else {
        vols[vols.len() / 2]
    };
    let high_vol_threshold = if vols.is_empty() {
        0.0
    } else {
        vols[(vols.len() * 3) / 4]
    };

    rolling_stats
        .iter()
        .enumerate()
        .map(|(i, (ret, vol))| {
            let regime = if *ret < 0.0 && *vol > high_vol_threshold {
                MarketRegime::Crash
            } else if *ret < 0.0 && *vol > median_vol {
                MarketRegime::Bear
            } else if *ret > 0.0 && *vol <= median_vol {
                MarketRegime::Bull
            } else {
                MarketRegime::Sideways
            };
            (i + lookback, regime)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Utility: combine training approaches for a fully robust model
// ---------------------------------------------------------------------------

/// Train a fully robust model combining DRO + noise injection + ensemble.
pub fn train_fully_robust(
    samples: &[Sample],
    n_features: usize,
    seed: u64,
) -> EnsembleModel {
    let n_models = 5;
    let mut models = Vec::with_capacity(n_models);

    // Model 1-2: DRO with different alpha levels
    models.push(train_robust_dro(samples, n_features, 0.01, 20, 0.2, seed));
    models.push(train_robust_dro(samples, n_features, 0.01, 20, 0.4, seed + 1));

    // Model 3-4: Noise injection with different noise levels
    models.push(train_with_noise_injection(
        samples, n_features, 0.01, 20, 0.1, seed + 2,
    ));
    models.push(train_with_noise_injection(
        samples, n_features, 0.01, 20, 0.2, seed + 3,
    ));

    // Model 5: Standard (for diversity)
    models.push(LinearModel::train_standard(
        samples, n_features, 0.01, 20, seed + 4,
    ));

    EnsembleModel { models }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_regime_produces_correct_length() {
        for config in default_regime_configs() {
            let prices = simulate_regime(&config, 100, 42);
            assert_eq!(prices.len(), 100);
            assert!(prices.iter().all(|p| *p > 0.0), "All prices should be positive");
        }
    }

    #[test]
    fn test_simulate_bull_has_upward_drift() {
        let config = RegimeConfig {
            regime: MarketRegime::Bull,
            drift: 0.001,
            volatility: 0.005,
            autocorrelation: 0.0,
        };
        // Run multiple simulations and check average trend
        let mut up_count = 0;
        for seed in 0..20 {
            let prices = simulate_regime(&config, 200, seed);
            if prices.last().unwrap() > prices.first().unwrap() {
                up_count += 1;
            }
        }
        assert!(up_count > 10, "Bull regime should trend up more often than not");
    }

    #[test]
    fn test_simulate_crash_has_downward_drift() {
        let config = RegimeConfig {
            regime: MarketRegime::Crash,
            drift: -0.005,
            volatility: 0.02,
            autocorrelation: 0.0,
        };
        let mut down_count = 0;
        for seed in 0..20 {
            let prices = simulate_regime(&config, 200, seed);
            if prices.last().unwrap() < prices.first().unwrap() {
                down_count += 1;
            }
        }
        assert!(down_count > 10, "Crash regime should trend down more often than not");
    }

    #[test]
    fn test_prices_to_samples() {
        let prices = simulate_regime(&default_regime_configs()[0], 100, 42);
        let samples = prices_to_samples(&prices, 10);
        assert!(!samples.is_empty());
        assert_eq!(samples[0].features.len(), 5);
        for s in &samples {
            assert!(s.label == 1.0 || s.label == -1.0);
        }
    }

    #[test]
    fn test_rank_transform() {
        let mut samples = vec![
            Sample { features: vec![10.0, 1.0], label: 1.0 },
            Sample { features: vec![20.0, 3.0], label: -1.0 },
            Sample { features: vec![30.0, 2.0], label: 1.0 },
        ];
        rank_transform(&mut samples);
        // After rank transform, values should be in [-1, 1]
        for s in &samples {
            for f in &s.features {
                assert!(*f >= -1.0 && *f <= 1.0, "Rank transform should normalize to [-1, 1]");
            }
        }
    }

    #[test]
    fn test_clip_features() {
        let mut samples = vec![Sample {
            features: vec![5.0, -5.0, 0.5],
            label: 1.0,
        }];
        clip_features(&mut samples, 2.0);
        assert_eq!(samples[0].features, vec![2.0, -2.0, 0.5]);
    }

    #[test]
    fn test_linear_model_predict() {
        let model = LinearModel {
            weights: vec![1.0, -1.0],
            bias: 0.0,
        };
        assert_eq!(model.predict(&[1.0, 0.0]), 1.0);
        assert_eq!(model.predict(&[-1.0, 0.0]), -1.0);
        assert_eq!(model.predict(&[0.0, 1.0]), -1.0);
    }

    #[test]
    fn test_standard_training_runs() {
        let prices = simulate_regime(&default_regime_configs()[0], 200, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let model = LinearModel::train_standard(&samples, 5, 0.01, 10, 42);
        let acc = model.accuracy(&samples);
        // Just verify it runs and produces a reasonable accuracy
        assert!(acc > 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_dro_training_runs() {
        let prices = simulate_regime(&default_regime_configs()[0], 200, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let model = train_robust_dro(&samples, 5, 0.01, 10, 0.3, 42);
        let acc = model.accuracy(&samples);
        assert!(acc > 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_noise_injection_training_runs() {
        let prices = simulate_regime(&default_regime_configs()[0], 200, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let model = train_with_noise_injection(&samples, 5, 0.01, 10, 0.1, 42);
        let acc = model.accuracy(&samples);
        assert!(acc > 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_ensemble_training_runs() {
        let prices = simulate_regime(&default_regime_configs()[0], 200, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let ensemble = EnsembleModel::train_diverse(&samples, 5, 3, 0.01, 10, 0.3, 42);
        assert_eq!(ensemble.models.len(), 3);
        let acc = ensemble.accuracy(&samples);
        assert!(acc > 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_fully_robust_training() {
        let prices = simulate_regime(&default_regime_configs()[0], 200, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let ensemble = train_fully_robust(&samples, 5, 42);
        assert_eq!(ensemble.models.len(), 5);
        let acc = ensemble.accuracy(&samples);
        assert!(acc > 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_perturbation_changes_samples() {
        let samples = vec![Sample {
            features: vec![1.0, 2.0, 3.0],
            label: 1.0,
        }];
        let perturbed = perturb_samples(&samples, 0.5, 42);
        assert_eq!(perturbed.len(), 1);
        assert_eq!(perturbed[0].label, 1.0); // label unchanged
        // At least one feature should differ
        let any_different = perturbed[0]
            .features
            .iter()
            .zip(samples[0].features.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(any_different, "Perturbation should change at least one feature");
    }

    #[test]
    fn test_robustness_evaluation() {
        let prices = simulate_regime(&default_regime_configs()[0], 300, 42);
        let mut samples = prices_to_samples(&prices, 10);
        rank_transform(&mut samples);

        let model = LinearModel::train_standard(&samples, 5, 0.01, 10, 42);
        let configs = default_regime_configs();

        let report = evaluate_robustness(
            &|features: &[f64]| model.predict(features),
            &samples,
            &configs,
            0.1,
            42,
        );

        assert!(report.clean_accuracy >= 0.0 && report.clean_accuracy <= 1.0);
        assert!(report.perturbed_accuracy >= 0.0 && report.perturbed_accuracy <= 1.0);
        assert_eq!(report.regime_accuracies.len(), 4);
        assert!(report.worst_regime_accuracy <= report.avg_regime_accuracy);
    }

    #[test]
    fn test_label_regimes() {
        // Create fake candles
        let mut candles = Vec::new();
        for i in 0..100 {
            let price = 100.0 + (i as f64) * 0.5; // upward trend
            candles.push(Candle {
                timestamp: i as u64 * 3600000,
                open: price - 0.1,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 1000.0,
            });
        }

        let regimes = label_regimes(&candles, 10);
        assert!(!regimes.is_empty());
        // Upward trend with low vol should be mostly Bull
        let bull_count = regimes.iter().filter(|(_, r)| *r == MarketRegime::Bull).count();
        assert!(
            bull_count > regimes.len() / 3,
            "Uptrending data should have many Bull labels"
        );
    }

    #[test]
    fn test_candles_to_samples() {
        let mut candles = Vec::new();
        for i in 0..50 {
            let price = 100.0 + (i as f64) * 0.1;
            candles.push(Candle {
                timestamp: i as u64 * 3600000,
                open: price - 0.1,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 1000.0 + (i as f64) * 10.0,
            });
        }

        let samples = candles_to_samples(&candles, 10);
        assert!(!samples.is_empty());
        assert_eq!(samples[0].features.len(), 5);
    }

    #[test]
    fn test_cross_regime_matrix() {
        let configs = default_regime_configs();
        let matrix = cross_regime_matrix(
            &|samples: &[Sample]| {
                let model = LinearModel::train_standard(samples, 5, 0.01, 5, 42);
                Box::new(move |features: &[f64]| model.predict(features))
            },
            &configs,
            200,
            100,
            10,
            42,
        );

        assert_eq!(matrix.len(), 4);
        for row in &matrix {
            assert_eq!(row.len(), 4);
            for (_, _, acc) in row {
                assert!(*acc >= 0.0 && *acc <= 1.0);
            }
        }
    }
}
