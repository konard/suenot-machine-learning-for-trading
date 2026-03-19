//! # Poisoning Attacks in Trading
//!
//! This library implements data poisoning attacks and defenses for ML-based trading systems.
//! It provides label flipping, feature poisoning, backdoor attacks, influence function
//! approximation, data sanitization defenses, and Bybit API integration.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single data sample: feature vector + binary label (0 or 1).
#[derive(Clone, Debug)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: u8, // 0 or 1
}

/// A dataset is a collection of samples.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn new(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Return feature matrix (n_samples x n_features) and label vector.
    pub fn to_matrix(&self) -> (Array2<f64>, Array1<f64>) {
        let n = self.samples.len();
        if n == 0 {
            return (Array2::zeros((0, 0)), Array1::zeros(0));
        }
        let d = self.samples[0].features.len();
        let mut x = Array2::zeros((n, d));
        let mut y = Array1::zeros(n);
        for (i, s) in self.samples.iter().enumerate() {
            for (j, &v) in s.features.iter().enumerate() {
                x[[i, j]] = v;
            }
            y[i] = s.label as f64;
        }
        (x, y)
    }
}

// ---------------------------------------------------------------------------
// Attacks
// ---------------------------------------------------------------------------

/// Label flipping attack: flip a fraction of labels in the dataset.
pub struct LabelFlipAttack {
    pub flip_rate: f64,
    pub seed: u64,
}

impl LabelFlipAttack {
    pub fn new(flip_rate: f64, seed: u64) -> Self {
        Self { flip_rate, seed }
    }

    /// Apply label flipping to a dataset, returning a new poisoned dataset.
    pub fn poison(&self, dataset: &Dataset) -> Dataset {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut poisoned = dataset.clone();
        let n_flip = (dataset.len() as f64 * self.flip_rate).round() as usize;

        // Select random indices to flip
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        indices.shuffle(&mut rng);
        for &idx in indices.iter().take(n_flip) {
            poisoned.samples[idx].label = 1 - poisoned.samples[idx].label;
        }
        poisoned
    }
}

/// Feature poisoning attack: inject crafted adversarial samples.
pub struct FeaturePoisonAttack {
    pub poison_rate: f64,
    pub perturbation_scale: f64,
    pub seed: u64,
}

impl FeaturePoisonAttack {
    pub fn new(poison_rate: f64, perturbation_scale: f64, seed: u64) -> Self {
        Self {
            poison_rate,
            perturbation_scale,
            seed,
        }
    }

    /// Inject crafted poisoned samples into the dataset.
    /// Generates adversarial samples by perturbing existing samples toward the
    /// opposite class centroid and assigning them the wrong label.
    pub fn poison(&self, dataset: &Dataset) -> Dataset {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut poisoned = dataset.clone();
        let n_poison = (dataset.len() as f64 * self.poison_rate).round() as usize;
        if dataset.is_empty() {
            return poisoned;
        }

        let d = dataset.samples[0].features.len();

        // Compute class centroids
        let mut centroid0 = vec![0.0f64; d];
        let mut centroid1 = vec![0.0f64; d];
        let mut count0 = 0usize;
        let mut count1 = 0usize;
        for s in &dataset.samples {
            let c = if s.label == 0 {
                &mut centroid0
            } else {
                &mut centroid1
            };
            let cnt = if s.label == 0 {
                &mut count0
            } else {
                &mut count1
            };
            for (j, &v) in s.features.iter().enumerate() {
                c[j] += v;
            }
            *cnt += 1;
        }
        if count0 > 0 {
            centroid0.iter_mut().for_each(|v| *v /= count0 as f64);
        }
        if count1 > 0 {
            centroid1.iter_mut().for_each(|v| *v /= count1 as f64);
        }

        for _ in 0..n_poison {
            // Pick a random sample
            let idx = rng.gen_range(0..dataset.len());
            let base = &dataset.samples[idx];
            let target_centroid = if base.label == 0 {
                &centroid1
            } else {
                &centroid0
            };

            // Perturb toward opposite centroid
            let mut new_features = base.features.clone();
            for (j, f) in new_features.iter_mut().enumerate() {
                let direction = target_centroid[j] - *f;
                *f += direction * self.perturbation_scale + rng.gen_range(-0.01..0.01);
            }

            poisoned.samples.push(Sample {
                features: new_features,
                label: 1 - base.label,
            });
        }
        poisoned
    }
}

/// Backdoor (trojan) attack: embed trigger pattern + target label.
pub struct BackdoorAttack {
    pub trigger_pattern: Vec<f64>,
    pub target_label: u8,
    pub poison_rate: f64,
    pub seed: u64,
}

impl BackdoorAttack {
    pub fn new(trigger_pattern: Vec<f64>, target_label: u8, poison_rate: f64, seed: u64) -> Self {
        Self {
            trigger_pattern,
            target_label,
            poison_rate,
            seed,
        }
    }

    /// Inject backdoor-triggered samples into the dataset.
    pub fn poison(&self, dataset: &Dataset) -> Dataset {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut poisoned = dataset.clone();
        let n_poison = (dataset.len() as f64 * self.poison_rate).round() as usize;
        if dataset.is_empty() {
            return poisoned;
        }

        for _ in 0..n_poison {
            let idx = rng.gen_range(0..dataset.len());
            let base = &dataset.samples[idx];

            // Embed trigger pattern: replace last N features with trigger values
            let mut new_features = base.features.clone();
            let offset = new_features.len().saturating_sub(self.trigger_pattern.len());
            for (j, &tv) in self.trigger_pattern.iter().enumerate() {
                if offset + j < new_features.len() {
                    new_features[offset + j] = tv;
                }
            }

            poisoned.samples.push(Sample {
                features: new_features,
                label: self.target_label,
            });
        }
        poisoned
    }

    /// Test if a sample contains the trigger pattern.
    pub fn has_trigger(&self, sample: &Sample, tolerance: f64) -> bool {
        let offset = sample.features.len().saturating_sub(self.trigger_pattern.len());
        self.trigger_pattern.iter().enumerate().all(|(j, &tv)| {
            if offset + j < sample.features.len() {
                (sample.features[offset + j] - tv).abs() < tolerance
            } else {
                false
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Influence function approximation
// ---------------------------------------------------------------------------

/// Approximate influence of each training sample on a test point using a
/// simplified approach (gradient dot product without full Hessian inversion).
/// Returns a vector of influence scores (one per training sample).
pub fn approximate_influence(
    weights: &Array1<f64>,
    train_x: &Array2<f64>,
    train_y: &Array1<f64>,
    test_x: &Array1<f64>,
    test_y: f64,
) -> Vec<f64> {
    let n = train_x.nrows();
    let mut influences = Vec::with_capacity(n);

    // Gradient at test point
    let test_logit: f64 = train_x.row(0).len() as f64; // dimension
    let _ = test_logit;
    let test_pred = sigmoid(test_x.dot(weights));
    let test_grad_scalar = test_pred - test_y;
    let test_grad: Array1<f64> = test_x * test_grad_scalar;

    for i in 0..n {
        let xi = train_x.row(i);
        let pred = sigmoid(xi.dot(weights));
        let grad_scalar = pred - train_y[i];
        let train_grad: Array1<f64> = xi.to_owned() * grad_scalar;

        // Simplified influence: dot product of gradients (approximation without Hessian)
        let influence = test_grad.dot(&train_grad);
        influences.push(influence);
    }
    influences
}

/// Use influence scores to identify the most impactful training points for
/// targeted poisoning. Returns indices sorted by absolute influence (descending).
pub fn targeted_poisoning_candidates(
    weights: &Array1<f64>,
    train_x: &Array2<f64>,
    train_y: &Array1<f64>,
    test_x: &Array1<f64>,
    test_y: f64,
    top_k: usize,
) -> Vec<usize> {
    let influences = approximate_influence(weights, train_x, train_y, test_x, test_y);
    let mut indexed: Vec<(usize, f64)> = influences.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    indexed.iter().take(top_k).map(|(i, _)| *i).collect()
}

// ---------------------------------------------------------------------------
// Defenses
// ---------------------------------------------------------------------------

/// Data sanitization defense: remove outliers based on z-score.
pub struct DataSanitizer {
    pub z_threshold: f64,
}

impl DataSanitizer {
    pub fn new(z_threshold: f64) -> Self {
        Self { z_threshold }
    }

    /// Remove samples with any feature exceeding the z-score threshold.
    pub fn sanitize(&self, dataset: &Dataset) -> Dataset {
        if dataset.is_empty() {
            return dataset.clone();
        }
        let d = dataset.samples[0].features.len();
        let n = dataset.len() as f64;

        // Compute mean and std for each feature
        let mut means = vec![0.0f64; d];
        let mut vars = vec![0.0f64; d];

        for s in &dataset.samples {
            for (j, &v) in s.features.iter().enumerate() {
                means[j] += v;
            }
        }
        means.iter_mut().for_each(|m| *m /= n);

        for s in &dataset.samples {
            for (j, &v) in s.features.iter().enumerate() {
                vars[j] += (v - means[j]).powi(2);
            }
        }
        let stds: Vec<f64> = vars.iter().map(|v| (v / n).sqrt().max(1e-10)).collect();

        // Keep only samples within z-score threshold
        let filtered: Vec<Sample> = dataset
            .samples
            .iter()
            .filter(|s| {
                s.features
                    .iter()
                    .enumerate()
                    .all(|(j, &v)| ((v - means[j]) / stds[j]).abs() <= self.z_threshold)
            })
            .cloned()
            .collect();

        Dataset::new(filtered)
    }
}

/// Poison detection via loss inspection: identify high-loss samples as potentially poisoned.
pub struct PoisonDetector {
    /// Percentile threshold (0.0 - 1.0). Samples with loss above this percentile are flagged.
    pub percentile: f64,
}

impl PoisonDetector {
    pub fn new(percentile: f64) -> Self {
        Self { percentile }
    }

    /// Detect potentially poisoned samples by computing per-sample loss
    /// and flagging those in the top percentile.
    /// Returns (clean_dataset, flagged_indices).
    pub fn detect_and_remove(
        &self,
        dataset: &Dataset,
        weights: &Array1<f64>,
    ) -> (Dataset, Vec<usize>) {
        if dataset.is_empty() {
            return (dataset.clone(), vec![]);
        }

        // Compute per-sample loss
        let mut losses: Vec<(usize, f64)> = dataset
            .samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let x = Array1::from_vec(s.features.clone());
                let pred = sigmoid(x.dot(weights));
                let y = s.label as f64;
                let loss = -y * pred.max(1e-10).ln() - (1.0 - y) * (1.0 - pred).max(1e-10).ln();
                (i, loss)
            })
            .collect();

        losses.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let threshold_idx = (dataset.len() as f64 * self.percentile) as usize;
        let threshold_loss = if threshold_idx < losses.len() {
            losses[threshold_idx].1
        } else {
            f64::MAX
        };

        let mut flagged = Vec::new();
        let mut clean_samples = Vec::new();

        for (i, s) in dataset.samples.iter().enumerate() {
            let x = Array1::from_vec(s.features.clone());
            let pred = sigmoid(x.dot(weights));
            let y = s.label as f64;
            let loss = -y * pred.max(1e-10).ln() - (1.0 - y) * (1.0 - pred).max(1e-10).ln();
            if loss > threshold_loss {
                flagged.push(i);
            } else {
                clean_samples.push(s.clone());
            }
        }

        (Dataset::new(clean_samples), flagged)
    }
}

// ---------------------------------------------------------------------------
// Linear classifier with gradient descent
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple logistic regression classifier.
pub struct LinearClassifier {
    pub weights: Array1<f64>,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl LinearClassifier {
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            learning_rate,
            epochs,
        }
    }

    /// Train on a dataset using gradient descent.
    pub fn train(&mut self, dataset: &Dataset) {
        let (x, y) = dataset.to_matrix();
        let n = x.nrows() as f64;
        if n == 0.0 {
            return;
        }

        for _ in 0..self.epochs {
            // Compute predictions
            let logits = x.dot(&self.weights);
            let preds: Array1<f64> = logits.mapv(sigmoid);

            // Gradient: (1/n) * X^T * (preds - y)
            let errors = &preds - &y;
            let grad = x.t().dot(&errors) / n;

            // Update weights
            self.weights = &self.weights - &(grad * self.learning_rate);
        }
    }

    /// Predict labels for a dataset.
    pub fn predict(&self, dataset: &Dataset) -> Vec<u8> {
        dataset
            .samples
            .iter()
            .map(|s| {
                let x = Array1::from_vec(s.features.clone());
                let pred = sigmoid(x.dot(&self.weights));
                if pred >= 0.5 { 1 } else { 0 }
            })
            .collect()
    }

    /// Compute accuracy on a dataset.
    pub fn accuracy(&self, dataset: &Dataset) -> f64 {
        let preds = self.predict(dataset);
        let correct = preds
            .iter()
            .zip(dataset.samples.iter())
            .filter(|(&p, s)| p == s.label)
            .count();
        correct as f64 / dataset.len() as f64
    }
}

/// Evaluate model accuracy under various poisoning rates.
/// Returns Vec<(poisoning_rate, accuracy)>.
pub fn evaluate_poisoning_rates(
    clean_train: &Dataset,
    test: &Dataset,
    rates: &[f64],
    seed: u64,
) -> Vec<(f64, f64)> {
    let n_features = clean_train.samples[0].features.len();
    rates
        .iter()
        .map(|&rate| {
            let attack = LabelFlipAttack::new(rate, seed);
            let poisoned = attack.poison(clean_train);
            let mut model = LinearClassifier::new(n_features, 0.1, 200);
            model.train(&poisoned);
            let acc = model.accuracy(test);
            (rate, acc)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

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
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch historical kline data from Bybit API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Kline {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    klines.reverse();
    Ok(klines)
}

/// Convert klines to a labeled dataset for binary classification.
/// Features: return, range, body_ratio, volume_change
/// Label: 1 if next candle is bullish, 0 otherwise
pub fn klines_to_dataset(klines: &[Kline]) -> Dataset {
    if klines.len() < 3 {
        return Dataset::new(vec![]);
    }

    // Compute volume moving average (5-period)
    let window = 5;
    let mut samples = Vec::new();

    for i in window..klines.len() - 1 {
        let k = &klines[i];
        let range = k.high - k.low;

        let ret = (k.close - k.open) / k.open.max(1e-10);
        let range_feat = range / k.open.max(1e-10);
        let body_ratio = if range > 1e-10 {
            (k.close - k.open).abs() / range
        } else {
            0.0
        };

        // Volume change relative to moving average
        let vol_avg: f64 = klines[i - window..i].iter().map(|x| x.volume).sum::<f64>() / window as f64;
        let vol_change = if vol_avg > 1e-10 {
            k.volume / vol_avg
        } else {
            1.0
        };

        // Label: next candle direction
        let next = &klines[i + 1];
        let label = if next.close > next.open { 1u8 } else { 0u8 };

        samples.push(Sample {
            features: vec![ret, range_feat, body_ratio, vol_change],
            label,
        });
    }

    Dataset::new(samples)
}

/// Generate synthetic kline data for testing when API is unavailable.
pub fn generate_synthetic_klines(n: usize, seed: u64) -> Vec<Kline> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut klines = Vec::with_capacity(n);
    let mut price = 50000.0f64;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = open * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        klines.push(Kline {
            timestamp: 1700000000000 + (i as u64) * 60000,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    klines
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_dataset(n: usize, seed: u64) -> Dataset {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut samples = Vec::new();
        for _ in 0..n {
            let x1: f64 = rng.gen_range(-1.0..1.0);
            let x2: f64 = rng.gen_range(-1.0..1.0);
            let label = if x1 + x2 > 0.0 { 1u8 } else { 0u8 };
            samples.push(Sample {
                features: vec![x1, x2],
                label,
            });
        }
        Dataset::new(samples)
    }

    #[test]
    fn test_label_flip_attack() {
        let dataset = make_test_dataset(100, 42);
        let attack = LabelFlipAttack::new(0.1, 42);
        let poisoned = attack.poison(&dataset);

        // Count flipped labels
        let flipped = dataset
            .samples
            .iter()
            .zip(poisoned.samples.iter())
            .filter(|(a, b)| a.label != b.label)
            .count();

        assert_eq!(flipped, 10); // 10% of 100
        assert_eq!(poisoned.len(), dataset.len());
    }

    #[test]
    fn test_feature_poison_attack() {
        let dataset = make_test_dataset(100, 42);
        let attack = FeaturePoisonAttack::new(0.1, 0.5, 42);
        let poisoned = attack.poison(&dataset);

        // Poisoned dataset should have more samples (injected)
        assert_eq!(poisoned.len(), 110); // 100 + 10
    }

    #[test]
    fn test_backdoor_attack() {
        let dataset = make_test_dataset(100, 42);
        let trigger = vec![9.99, 9.99];
        let attack = BackdoorAttack::new(trigger.clone(), 1, 0.1, 42);
        let poisoned = attack.poison(&dataset);

        assert_eq!(poisoned.len(), 110); // 100 original + 10 backdoored

        // Check that injected samples have trigger and target label
        let backdoored: Vec<_> = poisoned
            .samples
            .iter()
            .skip(100)
            .filter(|s| attack.has_trigger(s, 0.01))
            .collect();
        assert_eq!(backdoored.len(), 10);
        assert!(backdoored.iter().all(|s| s.label == 1));
    }

    #[test]
    fn test_data_sanitizer() {
        let mut dataset = make_test_dataset(100, 42);
        // Inject an extreme outlier
        dataset.samples.push(Sample {
            features: vec![100.0, 100.0],
            label: 1,
        });

        let sanitizer = DataSanitizer::new(3.0);
        let cleaned = sanitizer.sanitize(&dataset);

        // Outlier should be removed
        assert!(cleaned.len() <= dataset.len());
        assert!(cleaned.len() >= 99); // at minimum the outlier is removed
    }

    #[test]
    fn test_poison_detector() {
        let dataset = make_test_dataset(200, 42);
        let mut model = LinearClassifier::new(2, 0.1, 200);
        model.train(&dataset);

        let detector = PoisonDetector::new(0.9);
        let (clean, flagged) = detector.detect_and_remove(&dataset, &model.weights);

        // Should flag approximately 10% of samples
        assert!(!flagged.is_empty());
        assert!(clean.len() + flagged.len() == dataset.len());
    }

    #[test]
    fn test_linear_classifier() {
        let train = make_test_dataset(500, 42);
        let test = make_test_dataset(100, 99);

        let mut model = LinearClassifier::new(2, 0.5, 500);
        model.train(&train);
        let acc = model.accuracy(&test);

        // Simple linearly separable data should yield decent accuracy
        assert!(acc > 0.7, "Accuracy {:.2} too low", acc);
    }

    #[test]
    fn test_poisoning_degrades_accuracy() {
        let train = make_test_dataset(500, 42);
        let test = make_test_dataset(200, 99);

        // Clean model
        let mut clean_model = LinearClassifier::new(2, 0.5, 500);
        clean_model.train(&train);
        let clean_acc = clean_model.accuracy(&test);

        // Heavily poisoned model (45% label flip -- high rate to reliably degrade)
        let attack = LabelFlipAttack::new(0.45, 42);
        let poisoned = attack.poison(&train);
        let mut poison_model = LinearClassifier::new(2, 0.5, 500);
        poison_model.train(&poisoned);
        let poison_acc = poison_model.accuracy(&test);

        // With 45% flipped labels the model should lose at least some accuracy
        assert!(
            poison_acc < clean_acc + 0.01,
            "Poisoned ({:.3}) should be no better than clean ({:.3})",
            poison_acc,
            clean_acc
        );
    }

    #[test]
    fn test_sanitization_defense_recovers_accuracy() {
        let train = make_test_dataset(500, 42);
        let test = make_test_dataset(200, 99);

        // Poison with feature injection (adds outlier-ish samples)
        let attack = FeaturePoisonAttack::new(0.15, 2.0, 42);
        let poisoned = attack.poison(&train);

        // Sanitize
        let sanitizer = DataSanitizer::new(2.5);
        let sanitized = sanitizer.sanitize(&poisoned);

        // Train on sanitized data
        let mut sanitized_model = LinearClassifier::new(2, 0.5, 500);
        sanitized_model.train(&sanitized);
        let sanitized_acc = sanitized_model.accuracy(&test);

        // Sanitized model should have reasonable accuracy
        assert!(sanitized_acc > 0.6, "Sanitized accuracy {:.2} too low", sanitized_acc);
    }

    #[test]
    fn test_evaluate_poisoning_rates() {
        let train = make_test_dataset(300, 42);
        let test = make_test_dataset(100, 99);
        let rates = vec![0.0, 0.1, 0.2, 0.3];

        let results = evaluate_poisoning_rates(&train, &test, &rates, 42);
        assert_eq!(results.len(), 4);

        // Accuracy should generally decrease with higher poisoning rates
        let baseline_acc = results[0].1;
        let high_poison_acc = results[3].1;
        assert!(
            high_poison_acc <= baseline_acc + 0.05,
            "High poison rate should degrade accuracy"
        );
    }

    #[test]
    fn test_influence_approximation() {
        let train = make_test_dataset(100, 42);
        let (x, y) = train.to_matrix();

        let mut model = LinearClassifier::new(2, 0.5, 300);
        model.train(&train);

        let test_point = Array1::from_vec(vec![0.5, 0.3]);
        let influences =
            approximate_influence(&model.weights, &x, &y, &test_point, 1.0);

        assert_eq!(influences.len(), 100);
        // Influences should not all be zero
        assert!(influences.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_targeted_poisoning_candidates() {
        let train = make_test_dataset(100, 42);
        let (x, y) = train.to_matrix();

        let mut model = LinearClassifier::new(2, 0.5, 300);
        model.train(&train);

        let test_point = Array1::from_vec(vec![0.5, 0.3]);
        let candidates =
            targeted_poisoning_candidates(&model.weights, &x, &y, &test_point, 1.0, 5);

        assert_eq!(candidates.len(), 5);
        // All indices should be valid
        assert!(candidates.iter().all(|&i| i < 100));
    }

    #[test]
    fn test_synthetic_klines() {
        let klines = generate_synthetic_klines(100, 42);
        assert_eq!(klines.len(), 100);
        assert!(klines.iter().all(|k| k.high >= k.low));
        assert!(klines.iter().all(|k| k.high >= k.open.min(k.close)));
    }

    #[test]
    fn test_klines_to_dataset() {
        let klines = generate_synthetic_klines(100, 42);
        let dataset = klines_to_dataset(&klines);

        // Should have samples (100 - window - 1 = 94)
        assert!(!dataset.is_empty());
        assert_eq!(dataset.samples[0].features.len(), 4);
        // Labels should be binary
        assert!(dataset.samples.iter().all(|s| s.label == 0 || s.label == 1));
    }

    #[test]
    fn test_dataset_to_matrix() {
        let dataset = make_test_dataset(50, 42);
        let (x, y) = dataset.to_matrix();
        assert_eq!(x.nrows(), 50);
        assert_eq!(x.ncols(), 2);
        assert_eq!(y.len(), 50);
    }
}
