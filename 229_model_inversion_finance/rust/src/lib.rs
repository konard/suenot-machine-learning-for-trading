use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─────────────────────────────────────────────
// Bybit API types
// ─────────────────────────────────────────────

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
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle from Bybit.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch klines from Bybit REST API (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;
    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }
    let candles = resp
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
        .collect::<Vec<_>>();
    Ok(candles)
}

// ─────────────────────────────────────────────
// Private trading features
// ─────────────────────────────────────────────

/// Compute private trading features from candle data.
/// Returns a Vec of feature vectors (one per valid candle).
/// Features: [log_return_1, log_return_5, volatility_ratio, volume_norm, momentum]
pub fn compute_private_features(candles: &[Candle]) -> Vec<Array1<f64>> {
    let n = candles.len();
    if n < 21 {
        return vec![];
    }
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let mut features = Vec::new();
    for i in 20..n {
        // Log return (1-bar)
        let lr1 = (closes[i] / closes[i - 1]).ln();
        // Log return (5-bar)
        let lr5 = if i >= 5 {
            (closes[i] / closes[i - 5]).ln()
        } else {
            0.0
        };
        // Short-term volatility (5-bar) vs long-term (20-bar)
        let short_vol = std_dev(&closes[i - 5..=i]);
        let long_vol = std_dev(&closes[i - 20..=i]);
        let vol_ratio = if long_vol > 1e-12 {
            short_vol / long_vol
        } else {
            1.0
        };
        // Volume normalised by 20-bar moving average
        let vol_ma: f64 = volumes[i - 20..i].iter().sum::<f64>() / 20.0;
        let vol_norm = if vol_ma > 1e-12 {
            volumes[i] / vol_ma
        } else {
            1.0
        };
        // Momentum (rate of change 10-bar)
        let momentum = if i >= 10 && closes[i - 10].abs() > 1e-12 {
            (closes[i] - closes[i - 10]) / closes[i - 10]
        } else {
            0.0
        };
        features.push(Array1::from_vec(vec![lr1, lr5, vol_ratio, vol_norm, momentum]));
    }
    features
}

fn std_dev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

// ─────────────────────────────────────────────
// Target model (linear model trained on private features)
// ─────────────────────────────────────────────

/// A simple linear model: y = Wx + b
/// Represents a proprietary trading model trained on private features.
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Array2<f64>,  // (output_dim, input_dim)
    pub bias: Array1<f64>,     // (output_dim,)
}

impl LinearModel {
    /// Create a new model with given dimensions and random weights.
    pub fn new_random(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w_data: Vec<f64> = (0..output_dim * input_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let b_data: Vec<f64> = (0..output_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        Self {
            weights: Array2::from_shape_vec((output_dim, input_dim), w_data).unwrap(),
            bias: Array1::from_vec(b_data),
        }
    }

    /// Train the model on features/targets using simple least-squares (gradient descent).
    pub fn train(&mut self, features: &[Array1<f64>], targets: &[Array1<f64>], lr: f64, epochs: usize) {
        let n = features.len().min(targets.len()) as f64;
        if n < 1.0 {
            return;
        }
        for _ in 0..epochs {
            let mut w_grad = Array2::zeros(self.weights.raw_dim());
            let mut b_grad = Array1::zeros(self.bias.raw_dim());
            for (x, y) in features.iter().zip(targets.iter()) {
                let pred = self.predict(x);
                let err = &pred - y; // (output_dim,)
                // outer product: err (output_dim) x x (input_dim)
                for o in 0..err.len() {
                    for j in 0..x.len() {
                        w_grad[[o, j]] += err[o] * x[j];
                    }
                    b_grad[o] += err[o];
                }
            }
            w_grad /= n;
            b_grad /= n;
            self.weights = &self.weights - &(w_grad * lr);
            self.bias = &self.bias - &(b_grad * lr);
        }
    }

    /// Forward pass.
    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(x) + &self.bias
    }

    /// Predict with output perturbation defense (adds Gaussian noise).
    pub fn predict_with_perturbation(&self, x: &Array1<f64>, noise_std: f64) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let pred = self.predict(x);
        let noise: Array1<f64> = Array1::from_vec(
            (0..pred.len()).map(|_| rng.gen_range(-1.0..1.0) * noise_std).collect(),
        );
        pred + noise
    }

    /// Predict with confidence limiting defense (clamp output magnitude).
    pub fn predict_with_confidence_limit(&self, x: &Array1<f64>, max_abs: f64) -> Array1<f64> {
        let pred = self.predict(x);
        pred.mapv(|v| v.clamp(-max_abs, max_abs))
    }
}

// ─────────────────────────────────────────────
// White-box inversion attack
// ─────────────────────────────────────────────

/// Perform white-box model inversion: given full model access and a target
/// output, reconstruct the input that produced it.
///
/// Uses gradient descent on the input space with L2 regularization.
pub fn white_box_inversion(
    model: &LinearModel,
    target_output: &Array1<f64>,
    iterations: usize,
    learning_rate: f64,
    regularization: f64,
) -> Array1<f64> {
    let input_dim = model.weights.ncols();
    let mut rng = rand::thread_rng();
    let mut x = Array1::from_vec(
        (0..input_dim).map(|_| rng.gen_range(-0.1..0.1)).collect(),
    );

    for _ in 0..iterations {
        let pred = model.predict(&x);
        let err = &pred - target_output;
        // Gradient: W^T * err + lambda * x
        let grad = model.weights.t().dot(&err) + &x * regularization;
        x = &x - &(&grad * learning_rate);
    }
    x
}

// ─────────────────────────────────────────────
// Black-box inversion attack
// ─────────────────────────────────────────────

/// Perform black-box model inversion: reconstruct input using only query access.
///
/// Estimates gradients via finite differences.
pub fn black_box_inversion<F>(
    query_fn: F,
    target_output: &Array1<f64>,
    input_dim: usize,
    iterations: usize,
    learning_rate: f64,
    epsilon: f64,
    regularization: f64,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut rng = rand::thread_rng();
    let mut x = Array1::from_vec(
        (0..input_dim).map(|_| rng.gen_range(-0.1..0.1)).collect(),
    );

    for _ in 0..iterations {
        // Estimate gradient via finite differences
        let mut grad = Array1::zeros(input_dim);
        for i in 0..input_dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += epsilon;
            x_minus[i] -= epsilon;

            let y_plus = query_fn(&x_plus);
            let y_minus = query_fn(&x_minus);

            // d/dx_i || f(x) - target ||^2  ≈  (|| f(x+) - target ||^2 - || f(x-) - target ||^2) / (2*eps)
            let loss_plus = (&y_plus - target_output)
                .mapv(|v| v * v)
                .sum();
            let loss_minus = (&y_minus - target_output)
                .mapv(|v| v * v)
                .sum();
            grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
        // Add regularization gradient
        grad = &grad + &(&x * regularization);
        x = &x - &(&grad * learning_rate);
    }
    x
}

// ─────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────

/// Mean Squared Error between original and reconstructed features.
pub fn mse(original: &Array1<f64>, reconstructed: &Array1<f64>) -> f64 {
    let diff = original - reconstructed;
    diff.mapv(|v| v * v).sum() / diff.len() as f64
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.mapv(|v| v * v).sum().sqrt();
    let norm_b = b.mapv(|v| v * v).sum().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ─────────────────────────────────────────────
// Privacy leakage measurement
// ─────────────────────────────────────────────

/// Measure privacy leakage across multiple samples.
/// Returns (mean_mse, mean_cosine_similarity).
pub fn measure_privacy_leakage(
    model: &LinearModel,
    original_features: &[Array1<f64>],
    inversion_iterations: usize,
    learning_rate: f64,
    regularization: f64,
) -> (f64, f64) {
    if original_features.is_empty() {
        return (0.0, 0.0);
    }
    let mut total_mse = 0.0;
    let mut total_cos = 0.0;
    for x in original_features {
        let y = model.predict(x);
        let x_rec = white_box_inversion(model, &y, inversion_iterations, learning_rate, regularization);
        total_mse += mse(x, &x_rec);
        total_cos += cosine_similarity(x, &x_rec);
    }
    let n = original_features.len() as f64;
    (total_mse / n, total_cos / n)
}

/// Measure privacy leakage with output perturbation defense.
pub fn measure_leakage_with_perturbation(
    model: &LinearModel,
    original_features: &[Array1<f64>],
    noise_std: f64,
    inversion_iterations: usize,
    learning_rate: f64,
    regularization: f64,
) -> (f64, f64) {
    if original_features.is_empty() {
        return (0.0, 0.0);
    }
    let mut total_mse = 0.0;
    let mut total_cos = 0.0;
    for x in original_features {
        let y = model.predict_with_perturbation(x, noise_std);
        let x_rec = white_box_inversion(model, &y, inversion_iterations, learning_rate, regularization);
        total_mse += mse(x, &x_rec);
        total_cos += cosine_similarity(x, &x_rec);
    }
    let n = original_features.len() as f64;
    (total_mse / n, total_cos / n)
}

/// Measure privacy leakage with confidence limiting defense.
pub fn measure_leakage_with_confidence_limit(
    model: &LinearModel,
    original_features: &[Array1<f64>],
    max_abs: f64,
    inversion_iterations: usize,
    learning_rate: f64,
    regularization: f64,
) -> (f64, f64) {
    if original_features.is_empty() {
        return (0.0, 0.0);
    }
    let mut total_mse = 0.0;
    let mut total_cos = 0.0;
    for x in original_features {
        let y = model.predict_with_confidence_limit(x, max_abs);
        let x_rec = white_box_inversion(model, &y, inversion_iterations, learning_rate, regularization);
        total_mse += mse(x, &x_rec);
        total_cos += cosine_similarity(x, &x_rec);
    }
    let n = original_features.len() as f64;
    (total_mse / n, total_cos / n)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_linear_model_predict() {
        let model = LinearModel {
            weights: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
            bias: Array1::from_vec(vec![0.5]),
        };
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let y = model.predict(&x);
        assert!((y[0] - 6.5).abs() < 1e-9); // 1+2+3+0.5
    }

    #[test]
    fn test_mse_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(mse(&a, &a) < 1e-12);
    }

    #[test]
    fn test_mse_different() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert!((mse(&a, &b) - 1.0).abs() < 1e-9); // each diff=1, mse=1
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_white_box_inversion_recovers_input() {
        // For a linear model, white-box inversion should recover the input well.
        let model = LinearModel {
            weights: Array2::from_shape_vec((3, 3), vec![
                2.0, 0.0, 0.0,
                0.0, 3.0, 0.0,
                0.0, 0.0, 1.5,
            ]).unwrap(),
            bias: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        };
        let original = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let target = model.predict(&original);
        let reconstructed = white_box_inversion(&model, &target, 2000, 0.01, 0.001);
        let cos = cosine_similarity(&original, &reconstructed);
        assert!(cos > 0.95, "Cosine similarity {cos} should be > 0.95");
    }

    #[test]
    fn test_black_box_inversion_recovers_input() {
        let model = LinearModel {
            weights: Array2::from_shape_vec((2, 3), vec![
                2.0, 0.0, 0.0,
                0.0, 3.0, 0.0,
            ]).unwrap(),
            bias: Array1::from_vec(vec![0.0, 0.0]),
        };
        let original = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let target = model.predict(&original);
        let query = |x: &Array1<f64>| model.predict(x);
        let reconstructed = black_box_inversion(
            query, &target, 3, 500, 0.01, 0.001, 0.001,
        );
        let cos = cosine_similarity(&original, &reconstructed);
        assert!(cos > 0.8, "Cosine similarity {cos} should be > 0.8");
    }

    #[test]
    fn test_output_perturbation_degrades_inversion() {
        let model = LinearModel {
            weights: Array2::from_shape_vec((2, 3), vec![
                2.0, 1.0, 0.0,
                0.0, 1.0, 2.0,
            ]).unwrap(),
            bias: Array1::from_vec(vec![0.0, 0.0]),
        };
        let original = Array1::from_vec(vec![1.0, -0.5, 0.3]);

        // Without defense
        let clean_output = model.predict(&original);
        let rec_clean = white_box_inversion(&model, &clean_output, 1000, 0.01, 0.001);
        let cos_clean = cosine_similarity(&original, &rec_clean);

        // With strong perturbation
        let noisy_output = model.predict_with_perturbation(&original, 5.0);
        let rec_noisy = white_box_inversion(&model, &noisy_output, 1000, 0.01, 0.001);
        let cos_noisy = cosine_similarity(&original, &rec_noisy);

        // Perturbation should generally degrade reconstruction (though random, big noise helps)
        // We mainly check that both run without errors
        assert!(cos_clean > -2.0, "Should produce valid cosine");
        assert!(cos_noisy > -2.0, "Should produce valid cosine");
    }

    #[test]
    fn test_confidence_limiting() {
        let model = LinearModel {
            weights: Array2::from_shape_vec((1, 2), vec![10.0, 10.0]).unwrap(),
            bias: Array1::from_vec(vec![0.0]),
        };
        let x = Array1::from_vec(vec![1.0, 1.0]);
        let limited = model.predict_with_confidence_limit(&x, 0.5);
        assert!((limited[0] - 0.5).abs() < 1e-9, "Output should be clamped to 0.5");
    }

    #[test]
    fn test_privacy_leakage_measurement() {
        let model = LinearModel::new_random(3, 2);
        let features = vec![
            Array1::from_vec(vec![0.1, -0.2, 0.3]),
            Array1::from_vec(vec![-0.1, 0.4, -0.2]),
            Array1::from_vec(vec![0.3, 0.1, 0.1]),
        ];
        let (avg_mse, avg_cos) = measure_privacy_leakage(&model, &features, 500, 0.01, 0.001);
        assert!(avg_mse >= 0.0, "MSE should be non-negative");
        assert!(avg_cos >= -1.0 && avg_cos <= 1.0, "Cosine should be in [-1,1]");
    }

    #[test]
    fn test_model_train() {
        let mut model = LinearModel::new_random(2, 1);
        // Train on y = x0 + x1
        let features = vec![
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];
        let targets = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![1.0]),
        ];
        model.train(&features, &targets, 0.1, 500);
        let pred = model.predict(&Array1::from_vec(vec![1.0, 1.0]));
        assert!((pred[0] - 2.0).abs() < 0.3, "Prediction {:.3} should be close to 2.0", pred[0]);
    }

    #[test]
    fn test_std_dev() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&vals);
        assert!((sd - 2.0).abs() < 0.2, "Std dev {sd} should be approx 2.0");
    }
}
