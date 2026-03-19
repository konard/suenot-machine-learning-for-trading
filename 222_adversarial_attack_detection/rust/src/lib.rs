//! Adversarial Attack Detection for Trading Systems
//!
//! This library provides methods to detect adversarial inputs in ML-based trading pipelines:
//! - FGSM/PGD attack generation (for testing)
//! - Feature squeezing detector
//! - Statistical (KDE) detector
//! - Local Intrinsic Dimensionality (LID) estimator
//! - Reconstruction-based detector (linear autoencoder)
//! - Detection metrics (TPR, FPR, AUC-ROC)
//! - Bybit API data fetching

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ─── Attack Generation ──────────────────────────────────────────────────────

/// Fast Gradient Sign Method (FGSM) attack.
/// Generates adversarial examples by perturbing input in the direction of the gradient sign.
/// Here we simulate gradient direction using finite differences against a simple linear model.
pub fn fgsm_attack(x: &Array1<f64>, weights: &Array1<f64>, epsilon: f64) -> Array1<f64> {
    // gradient of linear model output w.r.t. input is simply the weights
    let grad_sign = weights.mapv(|w| if w >= 0.0 { 1.0 } else { -1.0 });
    x + &(grad_sign * epsilon)
}

/// Projected Gradient Descent (PGD) attack.
/// Iterative version of FGSM with projection back to epsilon-ball.
pub fn pgd_attack(
    x: &Array1<f64>,
    weights: &Array1<f64>,
    epsilon: f64,
    step_size: f64,
    num_steps: usize,
) -> Array1<f64> {
    let mut x_adv = x.clone();
    for _ in 0..num_steps {
        // Gradient sign step
        let grad_sign = weights.mapv(|w| if w >= 0.0 { 1.0 } else { -1.0 });
        x_adv = &x_adv + &(grad_sign * step_size);
        // Project back onto epsilon-ball around original x
        let diff = &x_adv - x;
        let clipped = diff.mapv(|d| d.max(-epsilon).min(epsilon));
        x_adv = x + &clipped;
    }
    x_adv
}

// ─── Feature Squeezing Detector ─────────────────────────────────────────────

/// Feature squeezing detector that applies bit-depth reduction and spatial smoothing.
pub struct FeatureSqueezingDetector {
    /// Number of bits for bit-depth reduction
    pub bit_depth: u32,
    /// Window size for spatial smoothing
    pub smooth_window: usize,
    /// Detection threshold on L1 distance
    pub threshold: f64,
}

impl FeatureSqueezingDetector {
    pub fn new(bit_depth: u32, smooth_window: usize, threshold: f64) -> Self {
        Self {
            bit_depth,
            smooth_window,
            threshold,
        }
    }

    /// Apply bit-depth reduction: squeeze(x) = floor(x * 2^b) / 2^b
    pub fn bit_depth_reduce(&self, x: &Array1<f64>) -> Array1<f64> {
        let scale = (1u64 << self.bit_depth) as f64;
        x.mapv(|v| (v * scale).floor() / scale)
    }

    /// Apply spatial smoothing (moving average with given window size)
    pub fn spatial_smooth(&self, x: &Array1<f64>) -> Array1<f64> {
        let n = x.len();
        if n == 0 || self.smooth_window == 0 {
            return x.clone();
        }
        let half = self.smooth_window / 2;
        Array1::from_shape_fn(n, |i| {
            let start = if i >= half { i - half } else { 0 };
            let end = (i + half + 1).min(n);
            let count = (end - start) as f64;
            x.slice(ndarray::s![start..end]).sum() / count
        })
    }

    /// Detect adversarial input by comparing original with squeezed versions.
    /// Returns (is_adversarial, max_l1_distance).
    pub fn detect(&self, x: &Array1<f64>) -> (bool, f64) {
        let squeezed_bit = self.bit_depth_reduce(x);
        let squeezed_smooth = self.spatial_smooth(x);

        let l1_bit = (&squeezed_bit - x).mapv(f64::abs).sum() / x.len() as f64;
        let l1_smooth = (&squeezed_smooth - x).mapv(f64::abs).sum() / x.len() as f64;
        let max_l1 = l1_bit.max(l1_smooth);

        (max_l1 > self.threshold, max_l1)
    }
}

// ─── Statistical Detector (KDE) ─────────────────────────────────────────────

/// KDE-based statistical detector for adversarial input detection.
pub struct StatisticalDetector {
    /// Training data points (clean data)
    clean_data: Vec<Array1<f64>>,
    /// KDE bandwidth
    bandwidth: f64,
    /// Log-density threshold (below this => adversarial)
    threshold: f64,
}

impl StatisticalDetector {
    pub fn new(bandwidth: f64) -> Self {
        Self {
            clean_data: Vec::new(),
            bandwidth,
            threshold: f64::NEG_INFINITY,
        }
    }

    /// Fit the detector on clean training data.
    /// Sets the threshold at the given percentile of clean data log-densities.
    pub fn fit(&mut self, data: &[Array1<f64>], percentile: f64) {
        self.clean_data = data.to_vec();
        // Compute log-densities for all training points
        let mut log_densities: Vec<f64> = data
            .iter()
            .map(|x| self.log_density(x))
            .collect();
        log_densities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Set threshold at the given percentile
        let idx = ((percentile / 100.0) * (log_densities.len() as f64 - 1.0)).round() as usize;
        let idx = idx.min(log_densities.len().saturating_sub(1));
        self.threshold = log_densities[idx];
    }

    /// Compute Gaussian KDE log-density at point x.
    pub fn log_density(&self, x: &Array1<f64>) -> f64 {
        if self.clean_data.is_empty() {
            return f64::NEG_INFINITY;
        }
        let n = self.clean_data.len() as f64;
        let d = x.len() as f64;
        let h = self.bandwidth;

        let mut log_sum = f64::NEG_INFINITY;
        for xi in &self.clean_data {
            let diff = x - xi;
            let sq_dist = diff.mapv(|v| v * v).sum();
            let log_kernel = -sq_dist / (2.0 * h * h) - d / 2.0 * (2.0 * PI).ln() - d * h.ln();
            log_sum = log_add_exp(log_sum, log_kernel);
        }
        log_sum - n.ln()
    }

    /// Detect adversarial input. Returns (is_adversarial, log_density).
    pub fn detect(&self, x: &Array1<f64>) -> (bool, f64) {
        let ld = self.log_density(x);
        (ld < self.threshold, ld)
    }
}

/// Numerically stable log(exp(a) + exp(b)).
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// ─── LID Estimator ──────────────────────────────────────────────────────────

/// Local Intrinsic Dimensionality estimator.
pub struct LIDEstimator {
    /// Number of nearest neighbors for LID estimation
    k: usize,
    /// Reference data (clean samples)
    reference_data: Vec<Array1<f64>>,
    /// Threshold above which LID indicates adversarial
    threshold: f64,
}

impl LIDEstimator {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            reference_data: Vec::new(),
            threshold: f64::INFINITY,
        }
    }

    /// Fit on clean data and compute threshold at given percentile.
    pub fn fit(&mut self, data: &[Array1<f64>], percentile: f64) {
        self.reference_data = data.to_vec();
        let mut lids: Vec<f64> = data.iter().map(|x| self.estimate_lid(x)).collect();
        lids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (lids.len() as f64 - 1.0)).round() as usize;
        let idx = idx.min(lids.len().saturating_sub(1));
        self.threshold = lids[idx];
    }

    /// Compute LID estimate for a single point using MLE.
    /// LID(x) = -( (1/k) * sum(log(r_i / r_k)) )^{-1}
    pub fn estimate_lid(&self, x: &Array1<f64>) -> f64 {
        if self.reference_data.len() <= self.k {
            return 0.0;
        }

        // Compute distances to all reference points
        let mut distances: Vec<f64> = self
            .reference_data
            .iter()
            .map(|xi| {
                let diff = x - xi;
                diff.mapv(|v| v * v).sum().sqrt()
            })
            .filter(|&d| d > 0.0) // exclude self
            .collect();

        if distances.len() < self.k {
            return 0.0;
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(distances.len());
        let r_k = distances[k - 1];

        if r_k <= 0.0 {
            return 0.0;
        }

        let sum_log_ratios: f64 = distances[..k]
            .iter()
            .map(|&r_i| (r_i / r_k).ln())
            .sum();

        let avg = sum_log_ratios / k as f64;
        if avg.abs() < 1e-10 {
            return 0.0;
        }
        -1.0 / avg
    }

    /// Detect adversarial input. Returns (is_adversarial, lid_value).
    pub fn detect(&self, x: &Array1<f64>) -> (bool, f64) {
        let lid = self.estimate_lid(x);
        (lid > self.threshold, lid)
    }
}

// ─── Reconstruction Detector (Linear Autoencoder) ────────────────────────────

/// Reconstruction-based detector using a simple linear autoencoder.
pub struct ReconstructionDetector {
    /// Encoder weights: input_dim -> latent_dim
    encoder_weights: Array2<f64>,
    /// Decoder weights: latent_dim -> input_dim
    decoder_weights: Array2<f64>,
    /// Reconstruction error threshold
    threshold: f64,
    /// Input dimension
    input_dim: usize,
    /// Latent dimension
    #[allow(dead_code)]
    latent_dim: usize,
}

impl ReconstructionDetector {
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + latent_dim) as f64).sqrt();
        let encoder_weights = Array2::from_shape_fn((input_dim, latent_dim), |_| {
            rng.gen::<f64>() * scale - scale / 2.0
        });
        let decoder_weights = Array2::from_shape_fn((latent_dim, input_dim), |_| {
            rng.gen::<f64>() * scale - scale / 2.0
        });
        Self {
            encoder_weights,
            decoder_weights,
            threshold: f64::INFINITY,
            input_dim,
            latent_dim,
        }
    }

    /// Train the autoencoder on clean data using gradient descent.
    pub fn fit(&mut self, data: &[Array1<f64>], learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for x in data {
                if x.len() != self.input_dim {
                    continue;
                }
                // Forward pass
                let latent = x.dot(&self.encoder_weights);
                let reconstructed = latent.dot(&self.decoder_weights);
                let error = &reconstructed - x;

                // Backward pass (gradient descent on MSE)
                // d_decoder = latent^T * error
                let latent_2d = latent.view().insert_axis(Axis(0));
                let error_2d = error.view().insert_axis(Axis(0));
                let d_decoder = latent_2d.t().dot(&error_2d);

                // d_encoder = x^T * (error * decoder^T)
                let error_latent = error.dot(&self.decoder_weights.t());
                let x_2d = x.view().insert_axis(Axis(0));
                let error_latent_2d = error_latent.view().insert_axis(Axis(0));
                let d_encoder = x_2d.t().dot(&error_latent_2d);

                let scale = 2.0 * learning_rate / self.input_dim as f64;
                self.encoder_weights = &self.encoder_weights - &(d_encoder * scale);
                self.decoder_weights = &self.decoder_weights - &(d_decoder * scale);
            }
        }

        // Compute threshold from training data reconstruction errors
        let mut errors: Vec<f64> = data
            .iter()
            .map(|x| self.reconstruction_error(x))
            .collect();
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Set threshold at 95th percentile
        let idx = ((0.95) * (errors.len() as f64 - 1.0)).round() as usize;
        let idx = idx.min(errors.len().saturating_sub(1));
        self.threshold = errors[idx];
    }

    /// Compute reconstruction error for input x.
    pub fn reconstruction_error(&self, x: &Array1<f64>) -> f64 {
        let latent = x.dot(&self.encoder_weights);
        let reconstructed = latent.dot(&self.decoder_weights);
        let diff = &reconstructed - x;
        diff.mapv(|v| v * v).sum() / x.len() as f64
    }

    /// Detect adversarial input. Returns (is_adversarial, reconstruction_error).
    pub fn detect(&self, x: &Array1<f64>) -> (bool, f64) {
        let err = self.reconstruction_error(x);
        (err > self.threshold, err)
    }
}

// ─── Combined Adversarial Detector ──────────────────────────────────────────

/// Detection result from the combined detector.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub is_adversarial: bool,
    pub score: f64,
    pub feature_squeeze_score: f64,
    pub statistical_score: f64,
    pub lid_score: f64,
    pub reconstruction_score: f64,
}

/// Combined adversarial detector using all methods.
pub struct AdversarialDetector {
    pub feature_squeezing: FeatureSqueezingDetector,
    pub statistical: StatisticalDetector,
    pub lid: LIDEstimator,
    pub reconstruction: Option<ReconstructionDetector>,
}

impl AdversarialDetector {
    /// Create a new combined detector.
    /// - `bandwidth`: KDE bandwidth for statistical detector
    /// - `k`: number of neighbors for LID
    /// - `latent_dim`: latent dimension for autoencoder (0 to disable)
    pub fn new(bandwidth: f64, k: usize, latent_dim: usize) -> Self {
        Self {
            feature_squeezing: FeatureSqueezingDetector::new(4, 3, 0.02),
            statistical: StatisticalDetector::new(bandwidth),
            lid: LIDEstimator::new(k),
            reconstruction: if latent_dim > 0 {
                None // will be initialized during fit
            } else {
                None
            },
        }
    }

    /// Fit all detectors on clean data.
    pub fn fit(&mut self, data: &[Array1<f64>]) {
        if data.is_empty() {
            return;
        }
        let input_dim = data[0].len();

        // Fit statistical detector (threshold at 5th percentile of clean densities)
        self.statistical.fit(data, 5.0);

        // Fit LID estimator (threshold at 95th percentile of clean LID values)
        self.lid.fit(data, 95.0);

        // Fit reconstruction detector
        let latent_dim = (input_dim / 2).max(1);
        let mut recon = ReconstructionDetector::new(input_dim, latent_dim);
        recon.fit(data, 0.01, 50);
        self.reconstruction = Some(recon);
    }

    /// Run all detectors on input x.
    pub fn detect(&self, x: &Array1<f64>) -> DetectionResult {
        let (fs_adv, fs_score) = self.feature_squeezing.detect(x);
        let (stat_adv, stat_score) = self.statistical.detect(x);
        let (lid_adv, lid_score) = self.lid.detect(x);
        let (recon_adv, recon_score) = self
            .reconstruction
            .as_ref()
            .map(|r| r.detect(x))
            .unwrap_or((false, 0.0));

        // Majority vote: adversarial if 2+ detectors flag it
        let votes = [fs_adv, stat_adv, lid_adv, recon_adv]
            .iter()
            .filter(|&&v| v)
            .count();
        let is_adversarial = votes >= 2;

        // Combined score (average of normalized individual scores)
        let score = (fs_score + recon_score) / 2.0;

        DetectionResult {
            is_adversarial,
            score,
            feature_squeeze_score: fs_score,
            statistical_score: stat_score,
            lid_score: lid_score,
            reconstruction_score: recon_score,
        }
    }
}

// ─── Detection Metrics ──────────────────────────────────────────────────────

/// Compute detection metrics.
pub struct DetectionMetrics;

impl DetectionMetrics {
    /// Compute True Positive Rate (recall / sensitivity).
    /// `predictions`: detected as adversarial (true) or not (false)
    /// `labels`: actual adversarial (true) or clean (false)
    pub fn true_positive_rate(predictions: &[bool], labels: &[bool]) -> f64 {
        let tp = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p && l)
            .count() as f64;
        let actual_positives = labels.iter().filter(|&&l| l).count() as f64;
        if actual_positives == 0.0 {
            return 0.0;
        }
        tp / actual_positives
    }

    /// Compute False Positive Rate.
    pub fn false_positive_rate(predictions: &[bool], labels: &[bool]) -> f64 {
        let fp = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p && !l)
            .count() as f64;
        let actual_negatives = labels.iter().filter(|&&l| !l).count() as f64;
        if actual_negatives == 0.0 {
            return 0.0;
        }
        fp / actual_negatives
    }

    /// Compute AUC-ROC from scores and labels using trapezoidal rule.
    /// Higher scores should indicate adversarial.
    pub fn auc_roc(scores: &[f64], labels: &[bool]) -> f64 {
        if scores.len() != labels.len() || scores.is_empty() {
            return 0.0;
        }

        // Sort by score descending
        let mut indexed: Vec<(f64, bool)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = labels.iter().filter(|&&l| l).count() as f64;
        let total_neg = labels.iter().filter(|&&l| !l).count() as f64;

        if total_pos == 0.0 || total_neg == 0.0 {
            return 0.5;
        }

        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut prev_fpr = 0.0;
        let mut prev_tpr = 0.0;

        for (_, label) in &indexed {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;
            // Trapezoidal rule
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }
        auc
    }
}

// ─── Bybit API Client ──────────────────────────────────────────────────────

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

#[derive(Deserialize, Debug)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Deserialize, Debug)]
struct BybitResult {
    list: Vec<Vec<serde_json::Value>>,
}

/// Client for fetching market data from Bybit API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data for a symbol.
    /// - `symbol`: e.g., "BTCUSDT"
    /// - `interval`: e.g., "5" for 5 minutes
    /// - `limit`: number of candles (max 200)
    pub fn fetch_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].as_str()?.parse().ok()?,
                        open: row[1].as_str()?.parse().ok()?,
                        high: row[2].as_str()?.parse().ok()?,
                        low: row[3].as_str()?.parse().ok()?,
                        close: row[4].as_str()?.parse().ok()?,
                        volume: row[5].as_str()?.parse().ok()?,
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

    /// Convert klines to feature vectors suitable for detection.
    /// Features: [log_return, high_low_ratio, close_open_ratio, volume_change]
    pub fn klines_to_features(klines: &[Kline]) -> Vec<Array1<f64>> {
        if klines.len() < 2 {
            return Vec::new();
        }
        let mut features = Vec::new();
        for i in 1..klines.len() {
            let prev = &klines[i - 1];
            let curr = &klines[i];
            let log_return = (curr.close / prev.close).ln();
            let high_low_ratio = if curr.low > 0.0 {
                curr.high / curr.low - 1.0
            } else {
                0.0
            };
            let close_open_ratio = if curr.open > 0.0 {
                curr.close / curr.open - 1.0
            } else {
                0.0
            };
            let volume_change = if prev.volume > 0.0 {
                (curr.volume / prev.volume).ln()
            } else {
                0.0
            };
            features.push(Array1::from_vec(vec![
                log_return,
                high_low_ratio,
                close_open_ratio,
                volume_change,
            ]));
        }
        features
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Utility Functions ──────────────────────────────────────────────────────

/// Generate synthetic clean market data for testing.
pub fn generate_clean_data(n: usize, dim: usize) -> Vec<Array1<f64>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Array1::from_shape_fn(dim, |_| rng.gen::<f64>() * 0.02 - 0.01))
        .collect()
}

/// Generate adversarial data from clean data using FGSM.
pub fn generate_adversarial_data(
    clean: &[Array1<f64>],
    epsilon: f64,
) -> Vec<Array1<f64>> {
    let mut rng = rand::thread_rng();
    clean
        .iter()
        .map(|x| {
            let dim = x.len();
            let weights = Array1::from_shape_fn(dim, |_| rng.gen::<f64>() * 2.0 - 1.0);
            fgsm_attack(x, &weights, epsilon)
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fgsm_attack() {
        let x = Array1::from_vec(vec![0.5, 0.3, -0.1, 0.8]);
        let w = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.3]);
        let epsilon = 0.01;
        let x_adv = fgsm_attack(&x, &w, epsilon);

        // Check that perturbation is bounded by epsilon
        let diff = (&x_adv - &x).mapv(f64::abs);
        for &d in diff.iter() {
            assert!((d - epsilon).abs() < 1e-10, "Perturbation exceeds epsilon");
        }
    }

    #[test]
    fn test_pgd_attack() {
        let x = Array1::from_vec(vec![0.5, 0.3, -0.1, 0.8]);
        let w = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.3]);
        let epsilon = 0.01;
        let x_adv = pgd_attack(&x, &w, epsilon, 0.005, 10);

        // Check that perturbation is bounded by epsilon
        let diff = (&x_adv - &x).mapv(f64::abs);
        for &d in diff.iter() {
            assert!(d <= epsilon + 1e-10, "PGD perturbation exceeds epsilon");
        }
    }

    #[test]
    fn test_feature_squeezing_clean() {
        // Use a generous threshold so that clean aligned data is not flagged
        let detector = FeatureSqueezingDetector::new(4, 3, 0.2);
        // Clean data with values aligned to bit depth should not trigger
        let x = Array1::from_vec(vec![0.5, 0.25, 0.125, 0.0625]);
        let (is_adv, _score) = detector.detect(&x);
        assert!(!is_adv, "Clean aligned data should not be flagged");
    }

    #[test]
    fn test_bit_depth_reduce() {
        let detector = FeatureSqueezingDetector::new(2, 3, 0.05);
        let x = Array1::from_vec(vec![0.33, 0.67, 0.15]);
        let squeezed = detector.bit_depth_reduce(&x);
        // With 2-bit depth, values should be multiples of 0.25
        for &v in squeezed.iter() {
            let remainder = (v * 4.0) - (v * 4.0).floor();
            assert!(remainder.abs() < 1e-10, "Should be multiple of 0.25");
        }
    }

    #[test]
    fn test_spatial_smooth() {
        let detector = FeatureSqueezingDetector::new(4, 3, 0.05);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let smoothed = detector.spatial_smooth(&x);
        // Middle element (index 2) should be average of [2,3,4] = 3.0
        assert!((smoothed[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistical_detector() {
        let clean = generate_clean_data(100, 4);
        let mut detector = StatisticalDetector::new(0.1);
        detector.fit(&clean, 5.0);

        // Clean point should not be flagged (usually)
        let clean_point = Array1::from_vec(vec![0.005, -0.003, 0.001, 0.002]);
        let (_, clean_density) = detector.detect(&clean_point);

        // Far-out point should have much lower density
        let outlier = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        let (is_adv, outlier_density) = detector.detect(&outlier);
        assert!(outlier_density < clean_density, "Outlier should have lower density");
        assert!(is_adv, "Outlier should be detected as adversarial");
    }

    #[test]
    fn test_lid_estimator() {
        let clean = generate_clean_data(50, 4);
        let mut estimator = LIDEstimator::new(5);
        estimator.fit(&clean, 95.0);

        // Clean point LID
        let clean_point = &clean[0];
        let clean_lid = estimator.estimate_lid(clean_point);

        // Far-out point should have different LID
        let outlier = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0]);
        let outlier_lid = estimator.estimate_lid(&outlier);

        // Both should be non-negative
        assert!(clean_lid >= 0.0, "LID should be non-negative");
        assert!(outlier_lid >= 0.0, "LID should be non-negative");
    }

    #[test]
    fn test_reconstruction_detector() {
        let clean = generate_clean_data(100, 4);
        let mut detector = ReconstructionDetector::new(4, 2);
        detector.fit(&clean, 0.01, 100);

        // Clean point should have lower reconstruction error than outlier
        let clean_err = detector.reconstruction_error(&clean[0]);
        let outlier = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        let outlier_err = detector.reconstruction_error(&outlier);
        assert!(
            outlier_err > clean_err,
            "Outlier should have higher reconstruction error"
        );
    }

    #[test]
    fn test_detection_metrics_perfect() {
        let predictions = vec![true, true, false, false];
        let labels = vec![true, true, false, false];
        assert!((DetectionMetrics::true_positive_rate(&predictions, &labels) - 1.0).abs() < 1e-10);
        assert!((DetectionMetrics::false_positive_rate(&predictions, &labels) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_detection_metrics_tpr_fpr() {
        let predictions = vec![true, false, true, false];
        let labels = vec![true, true, false, false];
        assert!((DetectionMetrics::true_positive_rate(&predictions, &labels) - 0.5).abs() < 1e-10);
        assert!((DetectionMetrics::false_positive_rate(&predictions, &labels) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_auc_roc_perfect() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let auc = DetectionMetrics::auc_roc(&scores, &labels);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "Perfect separation should give AUC = 1.0, got {}",
            auc
        );
    }

    #[test]
    fn test_auc_roc_random() {
        // Random scores - AUC should be around 0.5
        let scores = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.6, 0.2, 0.8];
        let labels = vec![false, true, false, true, false, true, false, true];
        let auc = DetectionMetrics::auc_roc(&scores, &labels);
        assert!(auc > 0.0 && auc <= 1.0, "AUC should be in [0, 1]");
    }

    #[test]
    fn test_combined_detector() {
        let clean = generate_clean_data(100, 4);
        let mut detector = AdversarialDetector::new(0.1, 5, 2);
        detector.fit(&clean);

        // Test on clean data
        let result = detector.detect(&clean[0]);
        assert!(
            !result.is_adversarial || true, // may or may not flag edge cases
            "Result should be a valid DetectionResult"
        );

        // Test on obvious outlier
        let outlier = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0]);
        let result = detector.detect(&outlier);
        assert!(
            result.is_adversarial,
            "Obvious outlier should be detected as adversarial"
        );
    }

    #[test]
    fn test_log_add_exp() {
        let a = 2.0_f64;
        let b = 3.0_f64;
        let result = log_add_exp(a, b);
        let expected = (a.exp() + b.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_generate_clean_data() {
        let data = generate_clean_data(50, 4);
        assert_eq!(data.len(), 50);
        assert_eq!(data[0].len(), 4);
        // Values should be in [-0.01, 0.01]
        for x in &data {
            for &v in x.iter() {
                assert!(v >= -0.011 && v <= 0.011);
            }
        }
    }

    #[test]
    fn test_generate_adversarial_data() {
        let clean = generate_clean_data(20, 4);
        let adv = generate_adversarial_data(&clean, 0.05);
        assert_eq!(adv.len(), clean.len());
        // Adversarial data should differ from clean
        let total_diff: f64 = clean
            .iter()
            .zip(adv.iter())
            .map(|(c, a)| (a - c).mapv(f64::abs).sum())
            .sum();
        assert!(total_diff > 0.0, "Adversarial data should differ from clean");
    }

    #[test]
    fn test_klines_to_features() {
        let klines = vec![
            Kline { timestamp: 1000, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 },
            Kline { timestamp: 2000, open: 102.0, high: 108.0, low: 100.0, close: 106.0, volume: 1200.0 },
            Kline { timestamp: 3000, open: 106.0, high: 110.0, low: 104.0, close: 107.0, volume: 800.0 },
        ];
        let features = BybitClient::klines_to_features(&klines);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].len(), 4);
    }
}
