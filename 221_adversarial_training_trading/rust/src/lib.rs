//! # Adversarial Training for Trading
//!
//! This crate implements adversarial training methods for robust trading models.
//! It includes FGSM and PGD attacks, adversarial training loops, TRADES-style
//! regularization, and Bybit API integration for fetching market data.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Neural Network
// ---------------------------------------------------------------------------

/// A simple two-layer feedforward neural network with ReLU activation.
/// Supports forward/backward passes for both parameter and input gradients.
pub struct NeuralNetwork {
    pub weights1: Array2<f64>, // (input_dim, hidden_dim)
    pub bias1: Array1<f64>,    // (hidden_dim,)
    pub weights2: Array2<f64>, // (hidden_dim, 1)
    pub bias2: Array1<f64>,    // (1,)
}

/// Cached intermediate values from forward pass, used for backward pass.
struct ForwardCache {
    z1: Array2<f64>,    // pre-activation of hidden layer
    a1: Array2<f64>,    // post-activation (ReLU) of hidden layer
    output: Array2<f64>, // final output (sigmoid)
}

/// Gradients with respect to model parameters.
struct ParamGrads {
    dw1: Array2<f64>,
    db1: Array1<f64>,
    dw2: Array2<f64>,
    db2: Array1<f64>,
}

impl NeuralNetwork {
    /// Create a new neural network with random weights (Xavier initialization).
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + 1) as f64).sqrt();

        let weights1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let bias1 = Array1::zeros(hidden_dim);
        let weights2 =
            Array2::from_shape_fn((hidden_dim, 1), |_| rng.gen_range(-scale2..scale2));
        let bias2 = Array1::zeros(1);

        NeuralNetwork { weights1, bias1, weights2, bias2 }
    }

    /// Clone the network (deep copy of all parameters).
    pub fn clone_network(&self) -> Self {
        NeuralNetwork {
            weights1: self.weights1.clone(),
            bias1: self.bias1.clone(),
            weights2: self.weights2.clone(),
            bias2: self.bias2.clone(),
        }
    }

    /// Forward pass returning cached intermediates.
    fn forward_cached(&self, x: &Array2<f64>) -> ForwardCache {
        // Hidden layer: z1 = x @ W1 + b1
        let z1 = x.dot(&self.weights1) + &self.bias1;
        // ReLU activation
        let a1 = z1.mapv(|v| v.max(0.0));
        // Output layer: z2 = a1 @ W2 + b2, then sigmoid
        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let output = z2.mapv(sigmoid);
        ForwardCache { z1, a1, output }
    }

    /// Forward pass returning predictions (n_samples, 1).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.forward_cached(x).output
    }

    /// Predict binary labels (threshold 0.5).
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let out = self.forward(x);
        out.column(0).mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 })
    }

    /// Compute binary cross-entropy loss.
    pub fn loss(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let preds = self.forward(x);
        binary_cross_entropy(&preds.column(0).to_owned(), y)
    }

    /// Compute accuracy (fraction of correct predictions).
    pub fn accuracy(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let preds = self.predict(x);
        let n = y.len() as f64;
        preds.iter().zip(y.iter()).filter(|(p, t)| (*p - *t).abs() < 0.5).count() as f64
            / n
    }

    /// Compute gradient of loss with respect to input x.
    /// Used for generating adversarial examples.
    pub fn input_gradient(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array2<f64> {
        let cache = self.forward_cached(x);
        let n = x.nrows() as f64;

        // d_loss/d_output for BCE with sigmoid
        let y_col = y.clone().insert_axis(Axis(1));
        let d_output = &cache.output - &y_col; // (n, 1)

        // Gradient through output layer: d_z2/d_a1 = W2^T
        let d_a1 = d_output.dot(&self.weights2.t()); // (n, hidden)

        // Gradient through ReLU
        let d_z1 = &d_a1 * &cache.z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        // Gradient w.r.t. input: d_z1/d_x = W1^T
        let d_x = d_z1.dot(&self.weights1.t()); // (n, input_dim)

        d_x / n
    }

    /// Backward pass: compute parameter gradients and update weights.
    fn backward_and_update(&mut self, x: &Array2<f64>, y: &Array1<f64>, lr: f64) {
        let grads = self.param_gradients(x, y);
        self.weights1 = &self.weights1 - &(&grads.dw1 * lr);
        self.bias1 = &self.bias1 - &(&grads.db1 * lr);
        self.weights2 = &self.weights2 - &(&grads.dw2 * lr);
        self.bias2 = &self.bias2 - &(&grads.db2 * lr);
    }

    /// Compute parameter gradients for BCE loss.
    fn param_gradients(&self, x: &Array2<f64>, y: &Array1<f64>) -> ParamGrads {
        let cache = self.forward_cached(x);
        let n = x.nrows() as f64;

        let y_col = y.clone().insert_axis(Axis(1));
        let d_output = &cache.output - &y_col; // (n, 1)

        // Output layer gradients
        let dw2 = cache.a1.t().dot(&d_output) / n;
        let db2 = d_output.sum_axis(Axis(0)) / n;

        // Hidden layer gradients
        let d_a1 = d_output.dot(&self.weights2.t());
        let d_z1 = &d_a1 * &cache.z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        let dw1 = x.t().dot(&d_z1) / n;
        let db1 = d_z1.sum_axis(Axis(0)) / n;

        ParamGrads { dw1, db1, dw2, db2 }
    }

    /// Standard training loop.
    pub fn train(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        for epoch in 0..epochs {
            self.backward_and_update(x_train, y_train, learning_rate);
            if (epoch + 1) % 50 == 0 {
                let loss = self.loss(x_train, y_train);
                let acc = self.accuracy(x_train, y_train);
                println!(
                    "  [Standard] Epoch {}/{}: loss={:.4}, acc={:.4}",
                    epoch + 1,
                    epochs,
                    loss,
                    acc
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Activation / loss helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn binary_cross_entropy(preds: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let eps = 1e-7;
    let n = preds.len() as f64;
    preds
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let p_clamped = p.clamp(eps, 1.0 - eps);
            -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
        })
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// Adversarial Attacks
// ---------------------------------------------------------------------------

/// Fast Gradient Sign Method (FGSM) attack.
///
/// Generates adversarial examples by perturbing inputs in the direction of
/// the gradient sign, scaled by epsilon:
///   x_adv = x + epsilon * sign(grad_x L(f(x), y))
pub fn fgsm_attack(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilon: f64,
) -> Array2<f64> {
    let grad = model.input_gradient(x, y);
    let perturbation = grad.mapv(|g| epsilon * g.signum());
    x + &perturbation
}

/// Projected Gradient Descent (PGD) attack.
///
/// Iteratively applies FGSM with step size alpha, projecting back onto the
/// epsilon-ball after each step. This is the strongest first-order attack.
///   x^{t+1} = clip(x^t + alpha * sign(grad), x - eps, x + eps)
pub fn pgd_attack(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilon: f64,
    alpha: f64,
    num_steps: usize,
) -> Array2<f64> {
    let mut rng = rand::thread_rng();

    // Random start within epsilon-ball
    let mut x_adv = if epsilon > 0.0 {
        x.mapv(|v| v + rng.gen_range(-epsilon..epsilon))
    } else {
        x.clone()
    };

    for _ in 0..num_steps {
        let grad = model.input_gradient(&x_adv, y);
        let perturbation = grad.mapv(|g| alpha * g.signum());
        x_adv = &x_adv + &perturbation;

        // Project back onto epsilon-ball centered at x
        x_adv = ndarray::Zip::from(&x_adv)
            .and(x)
            .map_collect(|&xa, &xo| xa.clamp(xo - epsilon, xo + epsilon));
    }

    x_adv
}

// ---------------------------------------------------------------------------
// Adversarial Training
// ---------------------------------------------------------------------------

/// PGD Adversarial Training (PGD-AT).
///
/// For each epoch, generates adversarial examples via PGD and trains
/// the model on those adversarial examples.
pub fn adversarial_train(
    model: &mut NeuralNetwork,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    epsilon: f64,
    learning_rate: f64,
) {
    let pgd_steps = 7;
    let alpha = epsilon / 4.0;

    for epoch in 0..epochs {
        // Generate adversarial examples
        let x_adv = pgd_attack(model, x_train, y_train, epsilon, alpha, pgd_steps);

        // Train on adversarial examples
        model.backward_and_update(&x_adv, y_train, learning_rate);

        if (epoch + 1) % 50 == 0 {
            let clean_loss = model.loss(x_train, y_train);
            let adv_loss = model.loss(&x_adv, y_train);
            let clean_acc = model.accuracy(x_train, y_train);
            println!(
                "  [PGD-AT] Epoch {}/{}: clean_loss={:.4}, adv_loss={:.4}, clean_acc={:.4}",
                epoch + 1,
                epochs,
                clean_loss,
                adv_loss,
                clean_acc
            );
        }
    }
}

/// TRADES-style adversarial training.
///
/// Optimizes: L(f(x), y) + beta * KL(f(x) || f(x+delta))
/// where delta is found by PGD to maximize the KL divergence.
pub fn trades_train(
    model: &mut NeuralNetwork,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    epsilon: f64,
    beta: f64,
    learning_rate: f64,
) {
    let pgd_steps = 7;
    let alpha = epsilon / 4.0;

    for epoch in 0..epochs {
        // Step 1: Generate adversarial examples to maximize KL divergence
        let x_adv = pgd_attack(model, x_train, y_train, epsilon, alpha, pgd_steps);

        // Step 2: Compute clean predictions (used for KL term)
        let clean_preds = model.forward(x_train);
        let clean_probs = clean_preds.column(0).to_owned();

        // Step 3: Compute combined gradient
        // Clean loss gradient (standard BCE)
        let clean_grads = model.param_gradients(x_train, y_train);

        // KL divergence term: use adversarial predictions as pseudo-targets
        let adv_preds = model.forward(&x_adv);
        let adv_probs = adv_preds.column(0).to_owned();

        // KL(clean || adv) gradients approximated by training on adversarial
        // examples with clean predictions as targets
        let adv_grads = model.param_gradients(&x_adv, &clean_probs);

        // Combined update: clean_grad + beta * kl_grad
        model.weights1 = &model.weights1
            - &((&clean_grads.dw1 + &(&adv_grads.dw1 * beta)) * learning_rate);
        model.bias1 = &model.bias1
            - &((&clean_grads.db1 + &(&adv_grads.db1 * beta)) * learning_rate);
        model.weights2 = &model.weights2
            - &((&clean_grads.dw2 + &(&adv_grads.dw2 * beta)) * learning_rate);
        model.bias2 = &model.bias2
            - &((&clean_grads.db2 + &(&adv_grads.db2 * beta)) * learning_rate);

        if (epoch + 1) % 50 == 0 {
            let clean_loss = model.loss(x_train, y_train);
            let kl_div = kl_divergence(&clean_probs, &adv_probs);
            let clean_acc = model.accuracy(x_train, y_train);
            println!(
                "  [TRADES] Epoch {}/{}: clean_loss={:.4}, kl_div={:.4}, clean_acc={:.4}",
                epoch + 1,
                epochs,
                clean_loss,
                kl_div,
                clean_acc
            );
        }
    }
}

/// KL divergence between two Bernoulli distributions.
fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    let eps = 1e-7;
    let n = p.len() as f64;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi.clamp(eps, 1.0 - eps);
            let qi = qi.clamp(eps, 1.0 - eps);
            pi * (pi / qi).ln() + (1.0 - pi) * ((1.0 - pi) / (1.0 - qi)).ln()
        })
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// Robustness Evaluation
// ---------------------------------------------------------------------------

/// Evaluate model accuracy under FGSM attack at various epsilon values.
pub fn evaluate_robustness_fgsm(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilons: &[f64],
) -> Vec<(f64, f64)> {
    epsilons
        .iter()
        .map(|&eps| {
            let x_adv = fgsm_attack(model, x, y, eps);
            let acc = model.accuracy(&x_adv, y);
            (eps, acc)
        })
        .collect()
}

/// Evaluate model accuracy under PGD attack at various epsilon values.
pub fn evaluate_robustness_pgd(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilons: &[f64],
    pgd_steps: usize,
) -> Vec<(f64, f64)> {
    epsilons
        .iter()
        .map(|&eps| {
            let alpha = eps / 4.0;
            let x_adv = pgd_attack(model, x, y, eps, alpha, pgd_steps);
            let acc = model.accuracy(&x_adv, y);
            (eps, acc)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bybit API Integration
// ---------------------------------------------------------------------------

/// Represents a single candlestick (OHLCV) data point.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit API v5.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Synchronous wrapper for fetching Bybit klines.
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    candles.reverse();
    Ok(candles)
}

/// Convert candles to feature matrix and binary labels.
///
/// Features per candle (using lookback of previous candle):
///   0: normalized return = (close - prev_close) / prev_close
///   1: volatility ratio = (high - low) / close
///   2: volume change = (volume - prev_volume) / prev_volume
///   3: price range position = (close - low) / (high - low)
///
/// Label: 1.0 if next candle's close > current close, else 0.0
pub fn candles_to_features(candles: &[Candle]) -> (Array2<f64>, Array1<f64>) {
    let n = candles.len();
    assert!(n >= 3, "Need at least 3 candles for feature engineering");

    let num_samples = n - 2; // lose 1 for lookback, 1 for label
    let num_features = 4;

    let mut features = Array2::zeros((num_samples, num_features));
    let mut labels = Array1::zeros(num_samples);

    for i in 0..num_samples {
        let prev = &candles[i];
        let curr = &candles[i + 1];
        let next = &candles[i + 2];

        // Normalized return
        let ret = if prev.close > 0.0 {
            (curr.close - prev.close) / prev.close
        } else {
            0.0
        };

        // Volatility ratio
        let vol_ratio = if curr.close > 0.0 {
            (curr.high - curr.low) / curr.close
        } else {
            0.0
        };

        // Volume change
        let vol_change = if prev.volume > 0.0 {
            (curr.volume - prev.volume) / prev.volume
        } else {
            0.0
        };

        // Price range position
        let range = curr.high - curr.low;
        let range_pos = if range > 0.0 {
            (curr.close - curr.low) / range
        } else {
            0.5
        };

        features[[i, 0]] = ret;
        features[[i, 1]] = vol_ratio;
        features[[i, 2]] = vol_change.clamp(-5.0, 5.0); // clip outliers
        features[[i, 3]] = range_pos;

        // Label: did price go up?
        labels[i] = if next.close > curr.close { 1.0 } else { 0.0 };
    }

    (features, labels)
}

/// Generate synthetic trading data for testing.
pub fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();

    let x = Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.gen_range(-1.0..1.0)
    });

    // Simple non-linear classification boundary
    let y = Array1::from_shape_fn(n_samples, |i| {
        let sum: f64 = (0..n_features.min(3))
            .map(|j| x[[i, j]])
            .sum();
        if sum + rng.gen_range(-0.3..0.3) > 0.0 {
            1.0
        } else {
            0.0
        }
    });

    (x, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Array2<f64>, Array1<f64>) {
        generate_synthetic_data(100, 4)
    }

    #[test]
    fn test_neural_network_forward() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, _) = make_simple_data();
        let out = nn.forward(&x);
        assert_eq!(out.nrows(), 100);
        assert_eq!(out.ncols(), 1);
        // All outputs should be between 0 and 1 (sigmoid)
        for &v in out.iter() {
            assert!(v >= 0.0 && v <= 1.0, "Output {} not in [0, 1]", v);
        }
    }

    #[test]
    fn test_neural_network_predict() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, _) = make_simple_data();
        let preds = nn.predict(&x);
        assert_eq!(preds.len(), 100);
        for &v in preds.iter() {
            assert!(v == 0.0 || v == 1.0, "Prediction {} not binary", v);
        }
    }

    #[test]
    fn test_standard_training_reduces_loss() {
        let (x, y) = make_simple_data();
        let mut nn = NeuralNetwork::new(4, 16);
        let loss_before = nn.loss(&x, &y);
        nn.train(&x, &y, 100, 0.1);
        let loss_after = nn.loss(&x, &y);
        assert!(
            loss_after < loss_before,
            "Training should reduce loss: {} -> {}",
            loss_before,
            loss_after
        );
    }

    #[test]
    fn test_fgsm_attack_changes_input() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, y) = make_simple_data();
        let x_adv = fgsm_attack(&nn, &x, &y, 0.1);

        // Adversarial examples should differ from originals
        let diff: f64 = (&x_adv - &x).mapv(|v| v.abs()).sum();
        assert!(diff > 0.0, "FGSM should produce different inputs");
    }

    #[test]
    fn test_fgsm_perturbation_bounded() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, y) = make_simple_data();
        let epsilon = 0.05;
        let x_adv = fgsm_attack(&nn, &x, &y, epsilon);

        // Every perturbation should be bounded by epsilon
        let diff = &x_adv - &x;
        for &v in diff.iter() {
            assert!(
                v.abs() <= epsilon + 1e-10,
                "FGSM perturbation {} exceeds epsilon {}",
                v.abs(),
                epsilon
            );
        }
    }

    #[test]
    fn test_pgd_attack_bounded() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, y) = make_simple_data();
        let epsilon = 0.1;
        let x_adv = pgd_attack(&nn, &x, &y, epsilon, 0.025, 10);

        let diff = &x_adv - &x;
        for &v in diff.iter() {
            assert!(
                v.abs() <= epsilon + 1e-10,
                "PGD perturbation {} exceeds epsilon {}",
                v.abs(),
                epsilon
            );
        }
    }

    #[test]
    fn test_pgd_stronger_than_fgsm() {
        let (x, y) = make_simple_data();
        let mut nn = NeuralNetwork::new(4, 16);
        nn.train(&x, &y, 200, 0.1);

        let epsilon = 0.2;
        let fgsm_adv = fgsm_attack(&nn, &x, &y, epsilon);
        let pgd_adv = pgd_attack(&nn, &x, &y, epsilon, 0.05, 20);

        let fgsm_loss = nn.loss(&fgsm_adv, &y);
        let pgd_loss = nn.loss(&pgd_adv, &y);

        // PGD should produce equal or higher loss (stronger attack)
        assert!(
            pgd_loss >= fgsm_loss - 0.1,
            "PGD loss {} should be >= FGSM loss {} (approximately)",
            pgd_loss,
            fgsm_loss
        );
    }

    #[test]
    fn test_adversarial_training_runs() {
        let (x, y) = make_simple_data();
        let mut nn = NeuralNetwork::new(4, 8);
        // Should not panic
        adversarial_train(&mut nn, &x, &y, 20, 0.1, 0.05);
        let acc = nn.accuracy(&x, &y);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_trades_training_runs() {
        let (x, y) = make_simple_data();
        let mut nn = NeuralNetwork::new(4, 8);
        // Should not panic
        trades_train(&mut nn, &x, &y, 20, 0.1, 1.0, 0.05);
        let acc = nn.accuracy(&x, &y);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_robustness_evaluation() {
        let (x, y) = make_simple_data();
        let nn = NeuralNetwork::new(4, 8);
        let epsilons = vec![0.0, 0.05, 0.1, 0.2];

        let results = evaluate_robustness_fgsm(&nn, &x, &y, &epsilons);
        assert_eq!(results.len(), 4);

        // Accuracy at eps=0 should equal clean accuracy
        let clean_acc = nn.accuracy(&x, &y);
        assert!((results[0].1 - clean_acc).abs() < 1e-10);
    }

    #[test]
    fn test_robustness_pgd_evaluation() {
        let (x, y) = make_simple_data();
        let nn = NeuralNetwork::new(4, 8);
        let epsilons = vec![0.0, 0.1, 0.2];

        let results = evaluate_robustness_pgd(&nn, &x, &y, &epsilons, 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_input_gradient_shape() {
        let nn = NeuralNetwork::new(4, 8);
        let (x, y) = make_simple_data();
        let grad = nn.input_gradient(&x, &y);
        assert_eq!(grad.shape(), x.shape());
    }

    #[test]
    fn test_kl_divergence_zero_for_same() {
        let p = Array1::from_vec(vec![0.3, 0.7, 0.5, 0.9]);
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "KL divergence of identical distributions should be ~0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = Array1::from_vec(vec![0.3, 0.7, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5, 0.8]);
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL divergence should be positive for different distributions");
    }

    #[test]
    fn test_generate_synthetic_data() {
        let (x, y) = generate_synthetic_data(50, 4);
        assert_eq!(x.nrows(), 50);
        assert_eq!(x.ncols(), 4);
        assert_eq!(y.len(), 50);
        for &v in y.iter() {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_candles_to_features() {
        let candles = vec![
            Candle { timestamp: 1, open: 100.0, high: 105.0, low: 99.0, close: 102.0, volume: 1000.0 },
            Candle { timestamp: 2, open: 102.0, high: 106.0, low: 101.0, close: 104.0, volume: 1200.0 },
            Candle { timestamp: 3, open: 104.0, high: 108.0, low: 103.0, close: 107.0, volume: 1100.0 },
            Candle { timestamp: 4, open: 107.0, high: 110.0, low: 105.0, close: 106.0, volume: 900.0 },
        ];

        let (features, labels) = candles_to_features(&candles);
        assert_eq!(features.nrows(), 2); // n-2 samples
        assert_eq!(features.ncols(), 4);
        assert_eq!(labels.len(), 2);

        // First sample: prev=candles[0], curr=candles[1], next=candles[2]
        // next.close (107) > curr.close (104) => label = 1.0
        assert_eq!(labels[0], 1.0);

        // Second sample: prev=candles[1], curr=candles[2], next=candles[3]
        // next.close (106) < curr.close (107) => label = 0.0
        assert_eq!(labels[1], 0.0);
    }

    #[test]
    fn test_clone_network() {
        let nn = NeuralNetwork::new(4, 8);
        let nn2 = nn.clone_network();
        assert_eq!(nn.weights1, nn2.weights1);
        assert_eq!(nn.bias1, nn2.bias1);
        assert_eq!(nn.weights2, nn2.weights2);
        assert_eq!(nn.bias2, nn2.bias2);
    }
}
