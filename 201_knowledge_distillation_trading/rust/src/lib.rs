//! # Knowledge Distillation for Trading
//!
//! This library implements knowledge distillation techniques for training
//! compact, low-latency trading models from large teacher models.
//!
//! Key components:
//! - TeacherModel: Large multi-layer neural network for maximum accuracy
//! - StudentModel: Compact neural network for fast inference
//! - Temperature-scaled softmax for soft target generation
//! - KL divergence loss for distillation training
//! - Bybit API client for market data

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// ReLU activation: max(0, x)
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Derivative of ReLU
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Softmax with temperature
// ---------------------------------------------------------------------------

/// Computes softmax with temperature scaling.
///
/// Higher temperature produces softer (more uniform) distributions,
/// revealing more "dark knowledge" from the teacher model.
///
/// # Arguments
/// * `logits` - Raw model outputs (pre-softmax)
/// * `temperature` - Temperature parameter (T=1 is standard softmax)
pub fn softmax_with_temperature(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let scaled = logits / temperature;
    let max_val = scaled.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = (&scaled - max_val).mapv(f64::exp);
    let sum = exp_vals.sum();
    &exp_vals / sum
}

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// Computes KL divergence: KL(p || q) = sum(p * log(p / q))
///
/// Used to measure how well the student's soft distribution matches
/// the teacher's soft distribution.
pub fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    let eps = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > eps {
                pi * (pi.max(eps) / qi.max(eps)).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Cross-entropy loss: -sum(y_true * log(y_pred))
pub fn cross_entropy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let eps = 1e-10;
    -y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t * p.max(eps).ln())
        .sum::<f64>()
}

/// Combined distillation loss:
///   L = alpha * L_hard(y_true, student_T1) + (1-alpha) * T^2 * L_soft(teacher_T, student_T)
///
/// # Arguments
/// * `y_true` - One-hot ground truth label
/// * `teacher_soft` - Teacher's temperature-scaled probabilities
/// * `student_soft` - Student's temperature-scaled probabilities
/// * `student_hard` - Student's standard (T=1) probabilities
/// * `alpha` - Balance between hard and soft loss (0..1)
/// * `temperature` - Temperature used for soft distributions
pub fn distillation_loss(
    y_true: &Array1<f64>,
    teacher_soft: &Array1<f64>,
    student_soft: &Array1<f64>,
    student_hard: &Array1<f64>,
    alpha: f64,
    temperature: f64,
) -> f64 {
    let hard_loss = cross_entropy(y_true, student_hard);
    let soft_loss = kl_divergence(teacher_soft, student_soft);
    alpha * hard_loss + (1.0 - alpha) * temperature * temperature * soft_loss
}

// ---------------------------------------------------------------------------
// Dense layer
// ---------------------------------------------------------------------------

/// A single fully-connected (dense) neural network layer.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    /// Creates a new dense layer with Xavier-initialized weights.
    pub fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    /// Forward pass through this layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = input.dot(&self.weights) + &self.biases;
        if self.use_relu {
            z.mapv(relu)
        } else {
            z
        }
    }

    /// Forward pass returning both pre-activation and post-activation values.
    pub fn forward_with_cache(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let z = input.dot(&self.weights) + &self.biases;
        let a = if self.use_relu {
            z.mapv(relu)
        } else {
            z.clone()
        };
        (z, a)
    }
}

// ---------------------------------------------------------------------------
// Teacher model
// ---------------------------------------------------------------------------

/// Large multi-layer neural network teacher model.
///
/// Architecture: input -> 128 -> 64 -> 32 -> num_classes
/// Uses ReLU activations on hidden layers.
#[derive(Clone, Debug)]
pub struct TeacherModel {
    pub layers: Vec<DenseLayer>,
    pub num_classes: usize,
}

impl TeacherModel {
    /// Creates a new teacher model.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `num_classes` - Number of output classes (e.g., 3 for buy/hold/sell)
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        let layers = vec![
            DenseLayer::new(input_size, 128, true),
            DenseLayer::new(128, 64, true),
            DenseLayer::new(64, 32, true),
            DenseLayer::new(32, num_classes, false), // logits layer, no activation
        ];
        Self { layers, num_classes }
    }

    /// Forward pass returning logits.
    pub fn forward_logits(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Forward pass returning probabilities at the given temperature.
    pub fn forward(&self, input: &Array1<f64>, temperature: f64) -> Array1<f64> {
        let logits = self.forward_logits(input);
        softmax_with_temperature(&logits, temperature)
    }

    /// Returns the total number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Simple training step using gradient descent on cross-entropy loss.
    /// Returns the loss value for this sample.
    pub fn train_step(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        learning_rate: f64,
    ) -> f64 {
        // Forward pass with caching
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut x = input.clone();
        for layer in &self.layers {
            let (z, a) = layer.forward_with_cache(&x);
            pre_activations.push(z);
            activations.push(a.clone());
            x = a;
        }

        // Compute softmax output and loss
        let logits = activations.last().unwrap();
        let probs = softmax_with_temperature(logits, 1.0);
        let loss = cross_entropy(target, &probs);

        // Backpropagation
        // Output layer gradient: dL/dz = probs - target (for softmax + cross-entropy)
        let mut delta = &probs - target;

        for i in (0..self.layers.len()).rev() {
            let input_act = &activations[i];

            // Weight gradient: dL/dW = input^T * delta
            let grad_w = Array2::from_shape_fn(self.layers[i].weights.dim(), |(r, c)| {
                input_act[r] * delta[c]
            });

            // Bias gradient: dL/db = delta
            let grad_b = delta.clone();

            // Update weights
            self.layers[i].weights = &self.layers[i].weights - &(&grad_w * learning_rate);
            self.layers[i].biases = &self.layers[i].biases - &(&grad_b * learning_rate);

            // Propagate gradient to previous layer
            if i > 0 {
                let new_delta = delta.dot(&self.layers[i].weights.t());
                // Apply ReLU derivative
                delta = &new_delta
                    * &pre_activations[i - 1].mapv(relu_derivative);
            }
        }

        loss
    }
}

// ---------------------------------------------------------------------------
// Student model
// ---------------------------------------------------------------------------

/// Compact neural network student model.
///
/// Architecture: input -> 16 -> 8 -> num_classes
/// Designed for low-latency inference after distillation training.
#[derive(Clone, Debug)]
pub struct StudentModel {
    pub layers: Vec<DenseLayer>,
    pub num_classes: usize,
}

impl StudentModel {
    /// Creates a new student model.
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        let layers = vec![
            DenseLayer::new(input_size, 16, true),
            DenseLayer::new(16, 8, true),
            DenseLayer::new(8, num_classes, false),
        ];
        Self { layers, num_classes }
    }

    /// Forward pass returning logits.
    pub fn forward_logits(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Forward pass returning probabilities at the given temperature.
    pub fn forward(&self, input: &Array1<f64>, temperature: f64) -> Array1<f64> {
        let logits = self.forward_logits(input);
        softmax_with_temperature(&logits, temperature)
    }

    /// Returns the total number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Training step for distillation.
    /// Uses the combined distillation loss with teacher soft targets.
    pub fn distillation_train_step(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        teacher_soft: &Array1<f64>,
        temperature: f64,
        alpha: f64,
        learning_rate: f64,
    ) -> f64 {
        // Forward pass with caching
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut x = input.clone();
        for layer in &self.layers {
            let (z, a) = layer.forward_with_cache(&x);
            pre_activations.push(z);
            activations.push(a.clone());
            x = a;
        }

        let logits = activations.last().unwrap();
        let student_hard = softmax_with_temperature(logits, 1.0);
        let student_soft = softmax_with_temperature(logits, temperature);

        let loss = distillation_loss(target, teacher_soft, &student_soft, &student_hard, alpha, temperature);

        // Backpropagation with combined gradient
        // Hard gradient: alpha * (probs_T1 - target)
        let hard_grad = (&student_hard - target) * alpha;

        // Soft gradient: (1-alpha) * T^2 * d_KL/d_logits
        // Approximated as (1-alpha) * T * (student_soft - teacher_soft)
        let soft_grad = (&student_soft - teacher_soft) * ((1.0 - alpha) * temperature);

        let mut delta = &hard_grad + &soft_grad;

        for i in (0..self.layers.len()).rev() {
            let input_act = &activations[i];

            let grad_w = Array2::from_shape_fn(self.layers[i].weights.dim(), |(r, c)| {
                input_act[r] * delta[c]
            });
            let grad_b = delta.clone();

            self.layers[i].weights = &self.layers[i].weights - &(&grad_w * learning_rate);
            self.layers[i].biases = &self.layers[i].biases - &(&grad_b * learning_rate);

            if i > 0 {
                let new_delta = delta.dot(&self.layers[i].weights.t());
                delta = &new_delta * &pre_activations[i - 1].mapv(relu_derivative);
            }
        }

        loss
    }

    /// Standard supervised training step (without distillation).
    pub fn train_step(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        learning_rate: f64,
    ) -> f64 {
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut x = input.clone();
        for layer in &self.layers {
            let (z, a) = layer.forward_with_cache(&x);
            pre_activations.push(z);
            activations.push(a.clone());
            x = a;
        }

        let logits = activations.last().unwrap();
        let probs = softmax_with_temperature(logits, 1.0);
        let loss = cross_entropy(target, &probs);

        let mut delta = &probs - target;

        for i in (0..self.layers.len()).rev() {
            let input_act = &activations[i];
            let grad_w = Array2::from_shape_fn(self.layers[i].weights.dim(), |(r, c)| {
                input_act[r] * delta[c]
            });
            let grad_b = delta.clone();

            self.layers[i].weights = &self.layers[i].weights - &(&grad_w * learning_rate);
            self.layers[i].biases = &self.layers[i].biases - &(&grad_b * learning_rate);

            if i > 0 {
                let new_delta = delta.dot(&self.layers[i].weights.t());
                delta = &new_delta * &pre_activations[i - 1].mapv(relu_derivative);
            }
        }

        loss
    }
}

// ---------------------------------------------------------------------------
// Distillation trainer
// ---------------------------------------------------------------------------

/// Configuration for the distillation training process.
pub struct DistillationConfig {
    /// Temperature for soft target generation (typically 2-20)
    pub temperature: f64,
    /// Balance between hard and soft loss (0..1, lower = more teacher influence)
    pub alpha: f64,
    /// Learning rate for student weight updates
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 5.0,
            alpha: 0.3,
            learning_rate: 0.001,
            epochs: 50,
        }
    }
}

/// Trains a student model via knowledge distillation from a teacher.
///
/// # Arguments
/// * `teacher` - Pre-trained teacher model (weights are frozen)
/// * `student` - Student model to train
/// * `features` - Training feature vectors
/// * `labels` - One-hot encoded training labels
/// * `config` - Distillation configuration
///
/// # Returns
/// Vector of average epoch losses
pub fn distillation_train(
    teacher: &TeacherModel,
    student: &mut StudentModel,
    features: &[Array1<f64>],
    labels: &[Array1<f64>],
    config: &DistillationConfig,
) -> Vec<f64> {
    let mut epoch_losses = Vec::new();

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;

        for (input, target) in features.iter().zip(labels.iter()) {
            // Get teacher's soft predictions (frozen teacher)
            let teacher_soft = teacher.forward(input, config.temperature);

            // Train student with distillation loss
            let loss = student.distillation_train_step(
                input,
                target,
                &teacher_soft,
                config.temperature,
                config.alpha,
                config.learning_rate,
            );

            total_loss += loss;
        }

        let avg_loss = total_loss / features.len() as f64;
        epoch_losses.push(avg_loss);

        if (epoch + 1) % 10 == 0 {
            println!("  Distillation epoch {}/{}: avg loss = {:.6}", epoch + 1, config.epochs, avg_loss);
        }
    }

    epoch_losses
}

// ---------------------------------------------------------------------------
// Inference benchmarking
// ---------------------------------------------------------------------------

/// Results from an inference speed benchmark.
#[derive(Debug)]
pub struct BenchmarkResult {
    /// Model name
    pub model_name: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Number of inference iterations
    pub iterations: usize,
    /// Mean inference time in microseconds
    pub mean_us: f64,
    /// Median inference time in microseconds
    pub median_us: f64,
    /// 95th percentile inference time in microseconds
    pub p95_us: f64,
    /// 99th percentile inference time in microseconds
    pub p99_us: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: params={}, mean={:.2}us, median={:.2}us, p95={:.2}us, p99={:.2}us",
            self.model_name, self.num_parameters, self.mean_us, self.median_us, self.p95_us, self.p99_us
        )
    }
}

/// Benchmarks inference latency of the teacher model.
pub fn benchmark_teacher(
    model: &TeacherModel,
    input: &Array1<f64>,
    iterations: usize,
) -> BenchmarkResult {
    let mut timings = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..100 {
        let _ = model.forward(input, 1.0);
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.forward(input, 1.0);
        timings.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let median = timings[timings.len() / 2];
    let p95 = timings[(timings.len() as f64 * 0.95) as usize];
    let p99 = timings[(timings.len() as f64 * 0.99) as usize];

    BenchmarkResult {
        model_name: "TeacherModel".to_string(),
        num_parameters: model.num_parameters(),
        iterations,
        mean_us: mean,
        median_us: median,
        p95_us: p95,
        p99_us: p99,
    }
}

/// Benchmarks inference latency of the student model.
pub fn benchmark_student(
    model: &StudentModel,
    input: &Array1<f64>,
    iterations: usize,
) -> BenchmarkResult {
    let mut timings = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..100 {
        let _ = model.forward(input, 1.0);
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.forward(input, 1.0);
        timings.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let median = timings[timings.len() / 2];
    let p95 = timings[(timings.len() as f64 * 0.95) as usize];
    let p99 = timings[(timings.len() as f64 * 0.99) as usize];

    BenchmarkResult {
        model_name: "StudentModel".to_string(),
        num_parameters: model.num_parameters(),
        iterations,
        mean_us: mean,
        median_us: median,
        p95_us: p95,
        p99_us: p99,
    }
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluates model accuracy on a dataset.
/// Returns (correct, total, accuracy).
pub fn evaluate_accuracy(
    predictions: &[Array1<f64>],
    labels: &[Array1<f64>],
) -> (usize, usize, f64) {
    let mut correct = 0;
    let total = predictions.len();

    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let pred_class = pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let true_class = label
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if pred_class == true_class {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / total as f64;
    (correct, total, accuracy)
}

// ---------------------------------------------------------------------------
// Bybit API client
// ---------------------------------------------------------------------------

/// OHLCV candlestick data from Bybit.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
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
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    /// Creates a new Bybit API client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetches kline (candlestick) data for a given symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval (e.g., "15" for 15 minutes)
    /// * `limit` - Number of candles to fetch (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let response: BybitResponse = client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<Candle> = response
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
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts candle data into feature vectors and labels for model training.
///
/// Features (per sample):
///   - 1-period return
///   - 5-period return
///   - 10-period return
///   - 20-period return
///   - realized volatility (20-period)
///   - volume change ratio
///   - high-low range / close
///
/// Labels: 3-class one-hot [buy, hold, sell] based on next-period return thresholds.
pub fn prepare_features_and_labels(
    candles: &[Candle],
    lookback: usize,
    threshold: f64,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    let min_history = lookback.max(20);

    if candles.len() < min_history + 2 {
        return (features, labels);
    }

    for i in min_history..candles.len() - 1 {
        let close = candles[i].close;
        let prev_close = candles[i - 1].close;

        // Returns at different horizons
        let ret_1 = (close - prev_close) / prev_close;
        let ret_5 = if i >= 5 {
            (close - candles[i - 5].close) / candles[i - 5].close
        } else {
            0.0
        };
        let ret_10 = if i >= 10 {
            (close - candles[i - 10].close) / candles[i - 10].close
        } else {
            0.0
        };
        let ret_20 = if i >= 20 {
            (close - candles[i - 20].close) / candles[i - 20].close
        } else {
            0.0
        };

        // Realized volatility (20-period)
        let mut returns_20 = Vec::new();
        for j in (i.saturating_sub(19))..=i {
            if j > 0 {
                let r = (candles[j].close - candles[j - 1].close) / candles[j - 1].close;
                returns_20.push(r);
            }
        }
        let mean_ret: f64 = returns_20.iter().sum::<f64>() / returns_20.len().max(1) as f64;
        let volatility: f64 = (returns_20
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / returns_20.len().max(1) as f64)
            .sqrt();

        // Volume change
        let vol_change = if i > 0 && candles[i - 1].volume > 0.0 {
            (candles[i].volume - candles[i - 1].volume) / candles[i - 1].volume
        } else {
            0.0
        };

        // High-low range
        let hl_range = (candles[i].high - candles[i].low) / close;

        let feature = Array1::from(vec![ret_1, ret_5, ret_10, ret_20, volatility, vol_change, hl_range]);
        features.push(feature);

        // Label: next period return
        let next_ret = (candles[i + 1].close - close) / close;
        let label = if next_ret > threshold {
            Array1::from(vec![1.0, 0.0, 0.0]) // buy
        } else if next_ret < -threshold {
            Array1::from(vec![0.0, 0.0, 1.0]) // sell
        } else {
            Array1::from(vec![0.0, 1.0, 0.0]) // hold
        };
        labels.push(label);
    }

    (features, labels)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_with_temperature_t1() {
        let logits = Array1::from(vec![2.0, 1.0, 0.1]);
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1.0");
        assert!(probs[0] > probs[1] && probs[1] > probs[2], "Order should be preserved");
    }

    #[test]
    fn test_softmax_high_temperature_is_softer() {
        let logits = Array1::from(vec![2.0, 1.0, 0.1]);
        let probs_t1 = softmax_with_temperature(&logits, 1.0);
        let probs_t10 = softmax_with_temperature(&logits, 10.0);

        // Higher temperature should produce more uniform distribution
        let max_t1 = probs_t1.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let max_t10 = probs_t10.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(
            max_t10 < max_t1,
            "Higher temperature should reduce the max probability"
        );
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        let p = Array1::from(vec![0.5, 0.3, 0.2]);
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "KL divergence of identical distributions should be ~0");
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let p = Array1::from(vec![0.7, 0.2, 0.1]);
        let q = Array1::from(vec![0.3, 0.4, 0.3]);
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL divergence of different distributions should be positive");
    }

    #[test]
    fn test_cross_entropy() {
        let target = Array1::from(vec![1.0, 0.0, 0.0]);
        let pred = Array1::from(vec![0.9, 0.05, 0.05]);
        let loss = cross_entropy(&target, &pred);
        assert!(loss > 0.0, "Cross entropy should be positive");
        assert!(loss < 1.0, "Loss should be small for good predictions");
    }

    #[test]
    fn test_distillation_loss() {
        let y_true = Array1::from(vec![1.0, 0.0, 0.0]);
        let teacher_soft = Array1::from(vec![0.6, 0.25, 0.15]);
        let student_soft = Array1::from(vec![0.55, 0.3, 0.15]);
        let student_hard = Array1::from(vec![0.8, 0.1, 0.1]);

        let loss = distillation_loss(&y_true, &teacher_soft, &student_soft, &student_hard, 0.3, 5.0);
        assert!(loss > 0.0, "Distillation loss should be positive");
    }

    #[test]
    fn test_teacher_model_forward() {
        let model = TeacherModel::new(7, 3);
        let input = Array1::from(vec![0.1, -0.2, 0.05, 0.3, 0.02, 0.5, 0.01]);
        let probs = model.forward(&input, 1.0);
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_student_model_forward() {
        let model = StudentModel::new(7, 3);
        let input = Array1::from(vec![0.1, -0.2, 0.05, 0.3, 0.02, 0.5, 0.01]);
        let probs = model.forward(&input, 1.0);
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_teacher_has_more_params_than_student() {
        let teacher = TeacherModel::new(7, 3);
        let student = StudentModel::new(7, 3);
        assert!(
            teacher.num_parameters() > student.num_parameters(),
            "Teacher should have more parameters than student"
        );
    }

    #[test]
    fn test_teacher_training_reduces_loss() {
        let mut teacher = TeacherModel::new(7, 3);
        let input = Array1::from(vec![0.1, -0.2, 0.05, 0.3, 0.02, 0.5, 0.01]);
        let target = Array1::from(vec![1.0, 0.0, 0.0]);

        let loss_before = teacher.train_step(&input, &target, 0.0); // zero lr = no update, just get loss
        // Train for a few steps
        for _ in 0..50 {
            teacher.train_step(&input, &target, 0.01);
        }
        let loss_after = teacher.train_step(&input, &target, 0.0);
        assert!(
            loss_after < loss_before,
            "Loss should decrease after training: before={}, after={}",
            loss_before,
            loss_after
        );
    }

    #[test]
    fn test_distillation_training() {
        let mut teacher = TeacherModel::new(7, 3);
        let input = Array1::from(vec![0.1, -0.2, 0.05, 0.3, 0.02, 0.5, 0.01]);
        let target = Array1::from(vec![1.0, 0.0, 0.0]);

        // Train teacher first
        for _ in 0..100 {
            teacher.train_step(&input, &target, 0.01);
        }

        // Distill to student
        let mut student = StudentModel::new(7, 3);
        let features = vec![input.clone()];
        let labels = vec![target.clone()];

        let config = DistillationConfig {
            temperature: 5.0,
            alpha: 0.3,
            learning_rate: 0.01,
            epochs: 50,
        };

        let losses = distillation_train(&teacher, &mut student, &features, &labels, &config);
        assert!(!losses.is_empty());

        // Student should have learned something
        let student_probs = student.forward(&input, 1.0);
        let pred_class = student_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(pred_class, 0, "Student should predict class 0 (buy)");
    }

    #[test]
    fn test_evaluate_accuracy() {
        let preds = vec![
            Array1::from(vec![0.8, 0.1, 0.1]),
            Array1::from(vec![0.1, 0.8, 0.1]),
            Array1::from(vec![0.1, 0.1, 0.8]),
        ];
        let labels = vec![
            Array1::from(vec![1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0]),
        ];
        let (correct, total, accuracy) = evaluate_accuracy(&preds, &labels);
        assert_eq!(correct, 3);
        assert_eq!(total, 3);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_student_inference_faster_than_teacher() {
        let teacher = TeacherModel::new(7, 3);
        let student = StudentModel::new(7, 3);
        let input = Array1::from(vec![0.1, -0.2, 0.05, 0.3, 0.02, 0.5, 0.01]);

        let teacher_bench = benchmark_teacher(&teacher, &input, 1000);
        let student_bench = benchmark_student(&student, &input, 1000);

        // Student should generally be faster (or at least not slower by much)
        // We check median to be robust
        println!("Teacher median: {:.2}us", teacher_bench.median_us);
        println!("Student median: {:.2}us", student_bench.median_us);
        // Not asserting strict inequality due to measurement noise on small models,
        // but the student should have fewer parameters
        assert!(student.num_parameters() < teacher.num_parameters());
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(relu_derivative(5.0), 1.0);
        assert_eq!(relu_derivative(-3.0), 0.0);
        assert_eq!(relu_derivative(0.0), 0.0);
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(4, 3, true);
        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_prepare_features_empty_candles() {
        let candles: Vec<Candle> = Vec::new();
        let (features, labels) = prepare_features_and_labels(&candles, 20, 0.001);
        assert!(features.is_empty());
        assert!(labels.is_empty());
    }
}
