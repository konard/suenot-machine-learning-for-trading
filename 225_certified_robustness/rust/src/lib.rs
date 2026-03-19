//! Certified Robustness for Trading
//!
//! Implements randomized smoothing, interval bound propagation (IBP),
//! and certified radius computation for trading classifiers.

use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::{PI, SQRT_2};

// ─── Bybit API types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
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

#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch klines from Bybit V5 API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
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
            if row.len() < 6 {
                return None;
            }
            Some(Kline {
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
    klines.reverse();
    Ok(klines)
}

/// Extract features from klines: log returns, volatility, momentum, volume ratio.
pub fn extract_features(klines: &[Kline], lookback: usize) -> Vec<Array1<f64>> {
    let n = klines.len();
    if n < lookback + 2 {
        return vec![];
    }

    let mut features = Vec::new();

    for i in lookback..n {
        let log_return = (klines[i].close / klines[i - 1].close).ln();

        // Rolling volatility
        let returns: Vec<f64> = (i - lookback..i)
            .map(|j| (klines[j + 1].close / klines[j].close).ln())
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = (returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64)
            .sqrt();

        // Momentum: cumulative return over lookback
        let momentum = (klines[i].close / klines[i - lookback].close).ln();

        // Volume ratio
        let avg_vol: f64 =
            klines[i - lookback..i].iter().map(|k| k.volume).sum::<f64>() / lookback as f64;
        let volume_ratio = if avg_vol > 0.0 {
            klines[i].volume / avg_vol
        } else {
            1.0
        };

        features.push(Array1::from_vec(vec![
            log_return,
            volatility,
            momentum,
            volume_ratio,
        ]));
    }

    features
}

/// Generate trading labels: 1 if next-period return > 0, else 0.
pub fn generate_labels(klines: &[Kline], lookback: usize) -> Vec<usize> {
    let n = klines.len();
    if n < lookback + 3 {
        return vec![];
    }

    (lookback..n - 1)
        .map(|i| {
            if klines[i + 1].close > klines[i].close {
                1
            } else {
                0
            }
        })
        .collect()
}

// ─── Normal distribution helpers ────────────────────────────────────────────

/// Standard normal CDF approximation (Abramowitz and Stegun).
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Inverse standard normal CDF (rational approximation).
pub fn normal_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Rational approximation (Peter Acklam)
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Error function approximation (Horner form).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ─── Confidence intervals ───────────────────────────────────────────────────

/// Lower bound of Clopper-Pearson confidence interval.
/// Uses a simple bisection on the regularized incomplete beta function.
pub fn clopper_pearson_lower(successes: usize, trials: usize, alpha: f64) -> f64 {
    if successes == 0 {
        return 0.0;
    }

    let k = successes as f64;
    let n = trials as f64;

    // Bisection: find p such that I_p(k, n-k+1) = alpha
    let mut lo = 0.0_f64;
    let mut hi = k / n;

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let cdf = regularized_incomplete_beta(mid, k, n - k + 1.0);
        if cdf < alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / 2.0
}

/// Regularized incomplete beta function approximation using continued fraction.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the continued fraction representation
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Lentz's algorithm for the continued fraction
    let mut cf = 1.0_f64;
    let mut c = 1.0_f64;
    let mut d;

    for m in 1..200 {
        let m_f = m as f64;

        // Even step
        let numerator_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + numerator_even / 1.0;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator_even / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        cf *= c * d;

        // Odd step
        let numerator_odd = -((a + m_f) * (a + b + m_f) * x)
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + numerator_odd;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator_odd / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = c * d;
        cf *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    let result = front * cf;

    // If x > (a+1)/(a+b+2), use the symmetry relation
    if x > (a + 1.0) / (a + b + 2.0) {
        1.0 - regularized_incomplete_beta(1.0 - x, b, a)
    } else {
        result.min(1.0).max(0.0)
    }
}

/// Lanczos approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];

    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x - 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (i, &c) in coefficients.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    -tmp + ((2.5066282746310005 * ser) / x).ln()
}

// ─── Simple Neural Network (base classifier) ───────────────────────────────

/// A simple feedforward neural network for classification.
#[derive(Debug, Clone)]
pub struct SimpleNeuralNetwork {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub num_classes: usize,
}

impl SimpleNeuralNetwork {
    /// Create a new network with given layer sizes.
    /// `layer_sizes` includes input, hidden layers, and output.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-0.5, 0.5);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];
            // Xavier-like initialization
            let scale = (2.0 / (rows + cols) as f64).sqrt();

            let w = Array2::from_shape_fn((rows, cols), |_| uniform.sample(&mut rng) * scale);
            let b = Array1::zeros(rows);

            weights.push(w);
            biases.push(b);
        }

        let num_classes = *layer_sizes.last().unwrap();

        SimpleNeuralNetwork {
            weights,
            biases,
            num_classes,
        }
    }

    /// Forward pass returning logits.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();

        for i in 0..self.weights.len() {
            x = self.weights[i].dot(&x) + &self.biases[i];

            // ReLU for all but last layer
            if i < self.weights.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }

        x
    }

    /// Predict class label.
    pub fn predict(&self, input: &Array1<f64>) -> usize {
        let logits = self.forward(input);
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Simple training with SGD (for demonstration).
    pub fn train(
        &mut self,
        features: &[Array1<f64>],
        labels: &[usize],
        epochs: usize,
        lr: f64,
    ) {
        let n = features.len().min(labels.len());
        if n == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        for _epoch in 0..epochs {
            let idx = rng.gen_range(0..n);
            let input = &features[idx];
            let target = labels[idx];

            // Forward pass with saved activations
            let mut activations: Vec<Array1<f64>> = vec![input.clone()];
            let mut x = input.clone();

            for i in 0..self.weights.len() {
                x = self.weights[i].dot(&x) + &self.biases[i];
                if i < self.weights.len() - 1 {
                    x.mapv_inplace(|v| v.max(0.0));
                }
                activations.push(x.clone());
            }

            // Softmax
            let logits = activations.last().unwrap();
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Array1<f64> = logits.mapv(|v| (v - max_logit).exp());
            let sum_exp = exp_logits.sum();
            let probs = &exp_logits / sum_exp;

            // Cross-entropy gradient at output
            let mut grad = probs.clone();
            grad[target] -= 1.0;

            // Backpropagation
            for i in (0..self.weights.len()).rev() {
                let act = &activations[i];

                // Weight gradient: grad * act^T
                let w_grad =
                    Array2::from_shape_fn(self.weights[i].dim(), |(r, c)| grad[r] * act[c]);

                // Update weights and biases
                self.weights[i] = &self.weights[i] - &(w_grad * lr);
                self.biases[i] = &self.biases[i] - &(&grad * lr);

                if i > 0 {
                    // Propagate gradient through weights
                    let new_grad = self.weights[i].t().dot(&grad);
                    // ReLU derivative
                    grad = Array1::from_shape_fn(new_grad.len(), |j| {
                        if activations[i][j] > 0.0 {
                            new_grad[j]
                        } else {
                            0.0
                        }
                    });
                }
            }
        }
    }
}

// ─── Randomized Smoothing ───────────────────────────────────────────────────

/// A smoothed classifier wrapping a base classifier with randomized smoothing.
pub struct SmoothedClassifier {
    pub base: SimpleNeuralNetwork,
    pub sigma: f64,
    pub num_classes: usize,
}

/// Result of certification for a single input.
#[derive(Debug, Clone)]
pub struct CertificationResult {
    pub predicted_class: usize,
    pub certified_radius: f64,
    pub p_lower: f64,
    pub is_certified: bool,
    pub counts: Vec<usize>,
}

impl SmoothedClassifier {
    pub fn new(base: SimpleNeuralNetwork, sigma: f64) -> Self {
        let num_classes = base.num_classes;
        SmoothedClassifier {
            base,
            sigma,
            num_classes,
        }
    }

    /// Add Gaussian noise to an input vector.
    fn add_noise(&self, input: &Array1<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let noise = Array1::from_shape_fn(input.len(), |_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            self.sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        });
        input + &noise
    }

    /// Predict using majority vote over noisy samples.
    pub fn predict(&self, input: &Array1<f64>, n_samples: usize) -> (usize, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let mut counts = vec![0usize; self.num_classes];

        for _ in 0..n_samples {
            let noisy = self.add_noise(input, &mut rng);
            let pred = self.base.predict(&noisy);
            if pred < self.num_classes {
                counts[pred] += 1;
            }
        }

        let predicted = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        (predicted, counts)
    }

    /// Certify a prediction: compute certified radius.
    ///
    /// Returns `CertificationResult` with the predicted class and certified radius.
    /// The prediction is made using `n_predict` samples, and certification uses `n_certify` samples.
    pub fn certify(
        &self,
        input: &Array1<f64>,
        n_predict: usize,
        n_certify: usize,
        alpha: f64,
    ) -> CertificationResult {
        // Phase 1: Predict
        let (predicted_class, _) = self.predict(input, n_predict);

        // Phase 2: Certify
        let mut rng = rand::thread_rng();
        let mut counts = vec![0usize; self.num_classes];

        for _ in 0..n_certify {
            let noisy = self.add_noise(input, &mut rng);
            let pred = self.base.predict(&noisy);
            if pred < self.num_classes {
                counts[pred] += 1;
            }
        }

        let top_count = counts[predicted_class];
        let p_lower = clopper_pearson_lower(top_count, n_certify, alpha);

        let certified_radius = if p_lower > 0.5 {
            self.sigma * normal_ppf(p_lower)
        } else {
            0.0
        };

        CertificationResult {
            predicted_class,
            certified_radius,
            p_lower,
            is_certified: certified_radius > 0.0,
            counts,
        }
    }
}

/// Compute certified accuracy at various radii.
/// Returns a vector of (radius, certified_accuracy) pairs.
pub fn certification_rate_at_radii(
    results: &[CertificationResult],
    radii: &[f64],
) -> Vec<(f64, f64)> {
    let n = results.len() as f64;
    if n == 0.0 {
        return radii.iter().map(|&r| (r, 0.0)).collect();
    }

    radii
        .iter()
        .map(|&r| {
            let certified_count = results
                .iter()
                .filter(|res| res.is_certified && res.certified_radius >= r)
                .count() as f64;
            (r, certified_count / n)
        })
        .collect()
}

// ─── Interval Bound Propagation (IBP) ──────────────────────────────────────

/// IBP certifier for simple feedforward networks.
pub struct IBPCertifier<'a> {
    pub network: &'a SimpleNeuralNetwork,
}

/// Result of IBP certification.
#[derive(Debug, Clone)]
pub struct IBPResult {
    pub lower_bounds: Array1<f64>,
    pub upper_bounds: Array1<f64>,
    pub is_verified: bool,
    pub predicted_class: usize,
    pub margin: f64,
}

impl<'a> IBPCertifier<'a> {
    pub fn new(network: &'a SimpleNeuralNetwork) -> Self {
        IBPCertifier { network }
    }

    /// Propagate interval bounds through the network.
    ///
    /// `epsilon` defines the l-infinity perturbation radius around `input`.
    pub fn certify(&self, input: &Array1<f64>, epsilon: f64) -> IBPResult {
        let mut lower = input.mapv(|v| v - epsilon);
        let mut upper = input.mapv(|v| v + epsilon);

        for i in 0..self.network.weights.len() {
            let w = &self.network.weights[i];
            let b = &self.network.biases[i];

            let w_pos = w.mapv(|v| v.max(0.0));
            let w_neg = w.mapv(|v| v.min(0.0));

            let new_lower = w_pos.dot(&lower) + w_neg.dot(&upper) + b;
            let new_upper = w_pos.dot(&upper) + w_neg.dot(&lower) + b;

            lower = new_lower;
            upper = new_upper;

            // ReLU for all but last layer
            if i < self.network.weights.len() - 1 {
                lower.mapv_inplace(|v| v.max(0.0));
                upper.mapv_inplace(|v| v.max(0.0));
            }
        }

        // Determine predicted class (using midpoint of bounds)
        let midpoint = (&lower + &upper) / 2.0;
        let predicted_class = midpoint
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Check if the predicted class lower bound exceeds all other upper bounds
        let pred_lower = lower[predicted_class];
        let mut is_verified = true;
        let mut min_margin = f64::INFINITY;

        for c in 0..self.network.num_classes {
            if c != predicted_class {
                let margin = pred_lower - upper[c];
                if margin < min_margin {
                    min_margin = margin;
                }
                if margin <= 0.0 {
                    is_verified = false;
                }
            }
        }

        if self.network.num_classes <= 1 {
            min_margin = 0.0;
        }

        IBPResult {
            lower_bounds: lower,
            upper_bounds: upper,
            is_verified,
            predicted_class,
            margin: min_margin,
        }
    }

    /// Find the maximum epsilon for which the prediction is certified.
    pub fn find_max_epsilon(&self, input: &Array1<f64>, max_eps: f64, steps: usize) -> f64 {
        let mut best_eps = 0.0;
        for step in 0..=steps {
            let eps = max_eps * step as f64 / steps as f64;
            let result = self.certify(input, eps);
            if result.is_verified {
                best_eps = eps;
            } else {
                break;
            }
        }
        best_eps
    }
}

// ─── Normalize features ─────────────────────────────────────────────────────

/// Normalize features to zero mean and unit variance.
pub fn normalize_features(features: &[Array1<f64>]) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    if features.is_empty() {
        return (vec![], Array1::zeros(0), Array1::ones(0));
    }

    let dim = features[0].len();
    let n = features.len() as f64;

    let mut mean = Array1::zeros(dim);
    for f in features {
        mean = mean + f;
    }
    mean /= n;

    let mut var = Array1::zeros(dim);
    for f in features {
        let diff = f - &mean;
        var = var + &diff.mapv(|v| v * v);
    }
    var /= n;
    let std = var.mapv(|v| v.sqrt().max(1e-8));

    let normalized: Vec<Array1<f64>> = features.iter().map(|f| (f - &mean) / &std).collect();

    (normalized, mean, std)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn test_normal_ppf() {
        assert!((normal_ppf(0.5) - 0.0).abs() < 1e-6);
        assert!((normal_ppf(0.975) - 1.96).abs() < 0.01);
        assert!((normal_ppf(0.025) - (-1.96)).abs() < 0.01);
    }

    #[test]
    fn test_ppf_cdf_inverse() {
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let x = normal_ppf(p);
            let p_back = normal_cdf(x);
            assert!(
                (p - p_back).abs() < 1e-4,
                "ppf/cdf roundtrip failed for p={}: got {}",
                p,
                p_back
            );
        }
    }

    #[test]
    fn test_clopper_pearson_lower() {
        // 90 successes out of 100 at alpha=0.05
        let lower = clopper_pearson_lower(90, 100, 0.05);
        assert!(lower > 0.8 && lower < 0.9, "Unexpected lower bound: {}", lower);

        // 0 successes should give 0
        assert_eq!(clopper_pearson_lower(0, 100, 0.05), 0.0);
    }

    #[test]
    fn test_neural_network_forward() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = net.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_neural_network_predict() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let pred = net.predict(&input);
        assert!(pred < 2);
    }

    #[test]
    fn test_smoothed_classifier_predict() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let sc = SmoothedClassifier::new(net, 0.25);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (pred, counts) = sc.predict(&input, 100);
        assert!(pred < 2);
        assert_eq!(counts.iter().sum::<usize>(), 100);
    }

    #[test]
    fn test_smoothed_classifier_certify() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let sc = SmoothedClassifier::new(net, 0.5);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let result = sc.certify(&input, 50, 200, 0.001);
        assert!(result.predicted_class < 2);
        assert!(result.certified_radius >= 0.0);
    }

    #[test]
    fn test_certification_rate_at_radii() {
        let results = vec![
            CertificationResult {
                predicted_class: 0,
                certified_radius: 0.5,
                p_lower: 0.8,
                is_certified: true,
                counts: vec![80, 20],
            },
            CertificationResult {
                predicted_class: 1,
                certified_radius: 0.2,
                p_lower: 0.6,
                is_certified: true,
                counts: vec![40, 60],
            },
            CertificationResult {
                predicted_class: 0,
                certified_radius: 0.0,
                p_lower: 0.45,
                is_certified: false,
                counts: vec![45, 55],
            },
        ];

        let radii = vec![0.0, 0.1, 0.3, 0.5, 1.0];
        let rates = certification_rate_at_radii(&results, &radii);

        // At radius 0.0: 2/3 certified
        assert!((rates[0].1 - 2.0 / 3.0).abs() < 1e-6);
        // At radius 0.1: 2/3 certified (both 0.5 and 0.2 qualify)
        assert!((rates[1].1 - 2.0 / 3.0).abs() < 1e-6);
        // At radius 0.3: 1/3 certified (only 0.5 qualifies)
        assert!((rates[2].1 - 1.0 / 3.0).abs() < 1e-6);
        // At radius 0.5: 1/3 certified
        assert!((rates[3].1 - 1.0 / 3.0).abs() < 1e-6);
        // At radius 1.0: 0 certified
        assert!((rates[4].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ibp_certifier() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let certifier = IBPCertifier::new(&net);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let result = certifier.certify(&input, 0.01);
        assert!(result.predicted_class < 2);
        // With tiny epsilon, bounds should be meaningful
        assert!(result.lower_bounds.len() == 2);
        assert!(result.upper_bounds.len() == 2);
    }

    #[test]
    fn test_ibp_find_max_epsilon() {
        let net = SimpleNeuralNetwork::new(&[4, 8, 2]);
        let certifier = IBPCertifier::new(&net);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let max_eps = certifier.find_max_epsilon(&input, 1.0, 100);
        assert!(max_eps >= 0.0);
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![5.0, 6.0]),
        ];

        let (normed, mean, _std) = normalize_features(&features);
        assert_eq!(normed.len(), 3);

        // Mean should be [3, 4]
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);

        // Normalized features should have roughly zero mean
        let norm_mean: f64 = normed.iter().map(|f| f[0]).sum::<f64>() / 3.0;
        assert!(norm_mean.abs() < 1e-6);
    }

    #[test]
    fn test_neural_network_training() {
        let mut net = SimpleNeuralNetwork::new(&[2, 4, 2]);
        let features = vec![
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![0.0, 0.0]),
        ];
        let labels = vec![0, 1, 0, 1];

        // Just ensure training doesn't panic
        net.train(&features, &labels, 100, 0.01);
        let pred = net.predict(&features[0]);
        assert!(pred < 2);
    }
}
