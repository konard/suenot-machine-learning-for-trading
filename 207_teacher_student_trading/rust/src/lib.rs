use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ============================================================================
// Bybit API Data Fetching
// ============================================================================

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Fetch OHLCV klines from Bybit v5 API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 7 {
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    turnover: row[6].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Prepare features from candles: returns (features, labels).
/// Features: normalized [open, high, low, close, volume] for each candle in a window.
/// Labels: direction of next candle (1.0 for up, 0.0 for down).
pub fn prepare_data(candles: &[Candle], window_size: usize) -> (Vec<Array1<f64>>, Vec<f64>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    if candles.len() < window_size + 1 {
        return (features, labels);
    }

    // Compute mean and std for normalization
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let mean = closes.iter().sum::<f64>() / closes.len() as f64;
    let std = (closes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / closes.len() as f64)
        .sqrt()
        .max(1e-8);

    for i in window_size..candles.len() - 1 {
        let window = &candles[i - window_size..i];
        let mut feat = Vec::with_capacity(window_size * 5);
        for c in window {
            feat.push((c.open - mean) / std);
            feat.push((c.high - mean) / std);
            feat.push((c.low - mean) / std);
            feat.push((c.close - mean) / std);
            feat.push(c.volume / 1e6); // scale volume
        }
        features.push(Array1::from_vec(feat));

        // Label: next candle direction
        let label = if candles[i].close > candles[i - 1].close {
            1.0
        } else {
            0.0
        };
        labels.push(label);
    }

    (features, labels)
}

// ============================================================================
// Market Regime Detection
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

/// Simple regime detection using rolling returns.
pub fn detect_regime(candles: &[Candle], lookback: usize) -> MarketRegime {
    if candles.len() < lookback + 1 {
        return MarketRegime::Sideways;
    }

    let recent = &candles[candles.len() - lookback..];
    let returns: Vec<f64> = recent
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = (returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    let threshold = volatility * 0.5;

    if avg_return > threshold {
        MarketRegime::Bull
    } else if avg_return < -threshold {
        MarketRegime::Bear
    } else {
        MarketRegime::Sideways
    }
}

/// Filter candles by regime, returning indices of candles matching the regime.
pub fn filter_by_regime(
    candles: &[Candle],
    regime: MarketRegime,
    window: usize,
) -> Vec<usize> {
    let mut indices = Vec::new();
    if candles.len() < window + 1 {
        return indices;
    }
    for i in window..candles.len() {
        let local_regime = detect_regime(&candles[..=i], window);
        if local_regime == regime {
            indices.push(i);
        }
    }
    indices
}

// ============================================================================
// Activation Functions
// ============================================================================

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Dense Layer
// ============================================================================

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt(); // He initialization

        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);

        Self { weights, biases }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.biases
    }

    /// Update weights with gradient descent.
    pub fn update(&mut self, grad_w: &Array2<f64>, grad_b: &Array1<f64>, lr: f64) {
        self.weights = &self.weights - &(grad_w * lr);
        self.biases = &self.biases - &(grad_b * lr);
    }
}

// ============================================================================
// Teacher Model with Intermediate Feature Extraction (Hint Layers)
// ============================================================================

#[derive(Debug, Clone)]
pub struct TeacherModel {
    pub layers: Vec<DenseLayer>,
    pub hint_layer_indices: Vec<usize>,
}

impl TeacherModel {
    /// Create a teacher model.
    /// `layer_sizes`: sizes of all layers including input and output.
    /// `hint_layer_indices`: indices of hidden layers to use as hint layers (0-based, into hidden layers).
    pub fn new(layer_sizes: &[usize], hint_layer_indices: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self {
            layers,
            hint_layer_indices: hint_layer_indices.to_vec(),
        }
    }

    /// Forward pass returning (output, hint_features).
    /// hint_features: intermediate activations at hint layers.
    pub fn forward_with_hints(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<Array1<f64>>) {
        let mut x = input.clone();
        let mut hints = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);

            // Apply ReLU to hidden layers, sigmoid to output
            if i < self.layers.len() - 1 {
                x.mapv_inplace(relu);
            } else {
                x.mapv_inplace(sigmoid);
            }

            if self.hint_layer_indices.contains(&i) {
                hints.push(x.clone());
            }
        }

        (x, hints)
    }

    /// Forward pass returning only the output.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward_with_hints(input).0
    }

    /// Simple training step with MSE loss (for pre-training teachers).
    pub fn train_step(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        lr: f64,
    ) -> f64 {
        // Forward pass storing all activations
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(&x);
            pre_activations.push(z.clone());
            x = if i < self.layers.len() - 1 {
                z.mapv(relu)
            } else {
                z.mapv(sigmoid)
            };
            activations.push(x.clone());
        }

        let output = activations.last().unwrap();
        let loss = (&(output - target)).mapv(|v| v.powi(2)).sum() / output.len() as f64;

        // Backprop
        let n = self.layers.len();
        // dL/doutput for MSE
        let mut delta = (output - target) * 2.0 / output.len() as f64;

        // Output layer: sigmoid derivative
        let sig_out = &activations[n];
        delta = &delta * &sig_out.mapv(|s| s * (1.0 - s));

        for i in (0..n).rev() {
            let input_act = &activations[i];
            let grad_w = outer_product(input_act, &delta);
            let grad_b = delta.clone();
            self.layers[i].update(&grad_w, &grad_b, lr);

            if i > 0 {
                delta = self.layers[i].weights.dot(&delta);
                // ReLU derivative
                let pre_act = &pre_activations[i - 1];
                delta = &delta * &pre_act.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            }
        }

        loss
    }
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

// ============================================================================
// Student Model with Hint Layer Matching
// ============================================================================

#[derive(Debug, Clone)]
pub struct StudentModel {
    pub layers: Vec<DenseLayer>,
    pub hint_layer_indices: Vec<usize>,
    /// Regressor layers to project student hint features to teacher hint dimensions.
    pub regressors: Vec<DenseLayer>,
}

impl StudentModel {
    /// Create a student model.
    /// `layer_sizes`: sizes of all layers including input and output.
    /// `hint_layer_indices`: indices of hidden layers to use as hint layers.
    /// `regressor_dims`: pairs of (student_dim, teacher_dim) for each hint layer.
    pub fn new(
        layer_sizes: &[usize],
        hint_layer_indices: &[usize],
        regressor_dims: &[(usize, usize)],
    ) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        let regressors = regressor_dims
            .iter()
            .map(|&(s_dim, t_dim)| DenseLayer::new(s_dim, t_dim))
            .collect();

        Self {
            layers,
            hint_layer_indices: hint_layer_indices.to_vec(),
            regressors,
        }
    }

    /// Forward pass returning (output, hint_features).
    pub fn forward_with_hints(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<Array1<f64>>) {
        let mut x = input.clone();
        let mut hints = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);

            if i < self.layers.len() - 1 {
                x.mapv_inplace(relu);
            } else {
                x.mapv_inplace(sigmoid);
            }

            if self.hint_layer_indices.contains(&i) {
                hints.push(x.clone());
            }
        }

        (x, hints)
    }

    /// Forward pass returning only the output.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward_with_hints(input).0
    }

    /// Project student hint features through regressors to match teacher dimensions.
    pub fn project_hints(&self, student_hints: &[Array1<f64>]) -> Vec<Array1<f64>> {
        student_hints
            .iter()
            .zip(self.regressors.iter())
            .map(|(hint, regressor)| regressor.forward(hint))
            .collect()
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// MSE loss between two arrays.
pub fn mse_loss(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    (pred - target).mapv(|v| v.powi(2)).sum() / pred.len() as f64
}

/// Feature matching loss: MSE between projected student hints and teacher hints.
pub fn feature_matching_loss(
    student_hints: &[Array1<f64>],
    teacher_hints: &[Array1<f64>],
    regressors: &[DenseLayer],
) -> f64 {
    let mut total_loss = 0.0;
    let count = student_hints.len().min(teacher_hints.len()).min(regressors.len());

    for i in 0..count {
        let projected = regressors[i].forward(&student_hints[i]);
        total_loss += mse_loss(&projected, &teacher_hints[i]);
    }

    if count > 0 {
        total_loss / count as f64
    } else {
        0.0
    }
}

/// Compute attention map from feature vector: sum of squared values (scalar for 1D).
fn attention_map(features: &Array1<f64>) -> Array1<f64> {
    let attn = features.mapv(|v| v.powi(2));
    let norm = attn.mapv(|v| v.powi(2)).sum().sqrt().max(1e-8);
    attn / norm
}

/// Attention transfer loss between student and teacher hint features.
pub fn attention_transfer_loss(
    student_hints: &[Array1<f64>],
    teacher_hints: &[Array1<f64>],
) -> f64 {
    let mut total_loss = 0.0;
    let count = student_hints.len().min(teacher_hints.len());

    for i in 0..count {
        // For attention transfer with different dimensions, we compare
        // the L2-normalized attention statistics (mean and variance)
        let s_attn = attention_map(&student_hints[i]);
        let t_attn = attention_map(&teacher_hints[i]);

        // Compare summary statistics since dimensions may differ
        let s_mean = s_attn.mean().unwrap_or(0.0);
        let t_mean = t_attn.mean().unwrap_or(0.0);
        let s_var = s_attn.mapv(|v| (v - s_mean).powi(2)).mean().unwrap_or(0.0);
        let t_var = t_attn.mapv(|v| (v - t_mean).powi(2)).mean().unwrap_or(0.0);

        total_loss += (s_mean - t_mean).powi(2) + (s_var - t_var).powi(2);
    }

    if count > 0 {
        total_loss / count as f64
    } else {
        0.0
    }
}

/// Combined distillation loss.
pub fn combined_loss(
    student_output: &Array1<f64>,
    target: &Array1<f64>,
    teacher_output: &Array1<f64>,
    student_hints: &[Array1<f64>],
    teacher_hints: &[Array1<f64>],
    regressors: &[DenseLayer],
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> f64 {
    let task_loss = mse_loss(student_output, target);
    let kd_loss = mse_loss(student_output, teacher_output);
    let feat_loss = feature_matching_loss(student_hints, teacher_hints, regressors);
    let attn_loss = attention_transfer_loss(student_hints, teacher_hints);

    alpha * task_loss + beta * kd_loss + gamma * feat_loss + delta * attn_loss
}

// ============================================================================
// Multi-Teacher Aggregation
// ============================================================================

/// Aggregates predictions from multiple teacher models using weighted ensemble.
pub struct MultiTeacherAggregator {
    pub teachers: Vec<TeacherModel>,
    pub weights: Vec<f64>,
}

impl MultiTeacherAggregator {
    pub fn new(teachers: Vec<TeacherModel>, weights: Vec<f64>) -> Self {
        assert_eq!(teachers.len(), weights.len());
        // Normalize weights
        let sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        Self { teachers, weights }
    }

    /// Aggregate teacher predictions with weighted average.
    pub fn aggregate(&self, input: &Array1<f64>) -> Array1<f64> {
        let outputs: Vec<Array1<f64>> = self
            .teachers
            .iter()
            .map(|t| t.forward(input))
            .collect();

        let output_size = outputs[0].len();
        let mut result = Array1::zeros(output_size);

        for (output, &weight) in outputs.iter().zip(self.weights.iter()) {
            result = result + output * weight;
        }

        result
    }

    /// Aggregate with hints from all teachers.
    pub fn aggregate_with_hints(
        &self,
        input: &Array1<f64>,
    ) -> (Array1<f64>, Vec<Vec<Array1<f64>>>) {
        let mut all_outputs = Vec::new();
        let mut all_hints = Vec::new();

        for teacher in &self.teachers {
            let (output, hints) = teacher.forward_with_hints(input);
            all_outputs.push(output);
            all_hints.push(hints);
        }

        let output_size = all_outputs[0].len();
        let mut result = Array1::zeros(output_size);
        for (output, &weight) in all_outputs.iter().zip(self.weights.iter()) {
            result = result + output * weight;
        }

        (result, all_hints)
    }
}

// ============================================================================
// Teacher-Student Trainer
// ============================================================================

pub struct TeacherStudentTrainer {
    pub student: StudentModel,
    pub alpha: f64, // task loss weight
    pub beta: f64,  // KD loss weight
    pub gamma: f64, // feature matching weight
    pub delta: f64, // attention transfer weight
    pub learning_rate: f64,
}

impl TeacherStudentTrainer {
    pub fn new(student: StudentModel, learning_rate: f64) -> Self {
        Self {
            student,
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.3,
            delta: 0.1,
            learning_rate,
        }
    }

    pub fn with_loss_weights(mut self, alpha: f64, beta: f64, gamma: f64, delta: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.gamma = gamma;
        self.delta = delta;
        self
    }

    /// Train student for one epoch using a single teacher.
    pub fn train_epoch(
        &mut self,
        teacher: &TeacherModel,
        inputs: &[Array1<f64>],
        targets: &[f64],
    ) -> f64 {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let target_arr = Array1::from_vec(vec![target]);
            let (teacher_output, teacher_hints) = teacher.forward_with_hints(input);
            let (student_output, student_hints) = self.student.forward_with_hints(input);

            let loss = combined_loss(
                &student_output,
                &target_arr,
                &teacher_output,
                &student_hints,
                &teacher_hints,
                &self.student.regressors,
                self.alpha,
                self.beta,
                self.gamma,
                self.delta,
            );
            total_loss += loss;

            // Numerical gradient update for student layers
            self.numerical_update(input, &target_arr, &teacher_output, &teacher_hints);
        }

        total_loss / inputs.len() as f64
    }

    /// Train student for one epoch using multi-teacher aggregator.
    pub fn train_epoch_multi_teacher(
        &mut self,
        aggregator: &MultiTeacherAggregator,
        inputs: &[Array1<f64>],
        targets: &[f64],
    ) -> f64 {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let target_arr = Array1::from_vec(vec![target]);
            let (teacher_output, all_hints) = aggregator.aggregate_with_hints(input);

            // Use first teacher's hints as reference (could also aggregate hints)
            let teacher_hints = if !all_hints.is_empty() {
                all_hints[0].clone()
            } else {
                vec![]
            };

            let (student_output, student_hints) = self.student.forward_with_hints(input);

            let loss = combined_loss(
                &student_output,
                &target_arr,
                &teacher_output,
                &student_hints,
                &teacher_hints,
                &self.student.regressors,
                self.alpha,
                self.beta,
                self.gamma,
                self.delta,
            );
            total_loss += loss;

            self.numerical_update(input, &target_arr, &teacher_output, &teacher_hints);
        }

        total_loss / inputs.len() as f64
    }

    /// Numerical gradient update (perturbation-based).
    fn numerical_update(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        teacher_output: &Array1<f64>,
        teacher_hints: &[Array1<f64>],
    ) {
        let epsilon = 1e-4;
        let lr = self.learning_rate;

        // Update each layer's weights
        for layer_idx in 0..self.student.layers.len() {
            let (rows, cols) = self.student.layers[layer_idx].weights.dim();
            // Sample a subset of weights to update for efficiency
            let mut rng = rand::thread_rng();
            let num_updates = (rows * cols).min(50);

            for _ in 0..num_updates {
                let r = rng.gen_range(0..rows);
                let c = rng.gen_range(0..cols);

                // Positive perturbation
                self.student.layers[layer_idx].weights[[r, c]] += epsilon;
                let (out_plus, hints_plus) = self.student.forward_with_hints(input);
                let loss_plus = combined_loss(
                    &out_plus, target, teacher_output, &hints_plus, teacher_hints,
                    &self.student.regressors, self.alpha, self.beta, self.gamma, self.delta,
                );

                // Negative perturbation
                self.student.layers[layer_idx].weights[[r, c]] -= 2.0 * epsilon;
                let (out_minus, hints_minus) = self.student.forward_with_hints(input);
                let loss_minus = combined_loss(
                    &out_minus, target, teacher_output, &hints_minus, teacher_hints,
                    &self.student.regressors, self.alpha, self.beta, self.gamma, self.delta,
                );

                // Restore and update
                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                self.student.layers[layer_idx].weights[[r, c]] += epsilon; // restore
                self.student.layers[layer_idx].weights[[r, c]] -= lr * grad;
            }

            // Update biases similarly
            let bias_len = self.student.layers[layer_idx].biases.len();
            for b in 0..bias_len {
                self.student.layers[layer_idx].biases[b] += epsilon;
                let (out_plus, hints_plus) = self.student.forward_with_hints(input);
                let loss_plus = combined_loss(
                    &out_plus, target, teacher_output, &hints_plus, teacher_hints,
                    &self.student.regressors, self.alpha, self.beta, self.gamma, self.delta,
                );

                self.student.layers[layer_idx].biases[b] -= 2.0 * epsilon;
                let (out_minus, hints_minus) = self.student.forward_with_hints(input);
                let loss_minus = combined_loss(
                    &out_minus, target, teacher_output, &hints_minus, teacher_hints,
                    &self.student.regressors, self.alpha, self.beta, self.gamma, self.delta,
                );

                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                self.student.layers[layer_idx].biases[b] += epsilon;
                self.student.layers[layer_idx].biases[b] -= lr * grad;
            }
        }
    }

    /// Evaluate accuracy on test data.
    pub fn evaluate(&self, inputs: &[Array1<f64>], targets: &[f64]) -> f64 {
        let mut correct = 0;
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let output = self.student.forward(input);
            let pred = if output[0] > 0.5 { 1.0 } else { 0.0 };
            if (pred - target).abs() < 0.5 {
                correct += 1;
            }
        }
        correct as f64 / inputs.len() as f64
    }
}

/// Evaluate a teacher model's accuracy.
pub fn evaluate_teacher(teacher: &TeacherModel, inputs: &[Array1<f64>], targets: &[f64]) -> f64 {
    let mut correct = 0;
    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let output = teacher.forward(input);
        let pred = if output[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred - target).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f64 / inputs.len() as f64
}

/// Evaluate multi-teacher ensemble accuracy.
pub fn evaluate_ensemble(
    aggregator: &MultiTeacherAggregator,
    inputs: &[Array1<f64>],
    targets: &[f64],
) -> f64 {
    let mut correct = 0;
    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let output = aggregator.aggregate(input);
        let pred = if output[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred - target).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f64 / inputs.len() as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(4, 3);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_teacher_model_forward() {
        let teacher = TeacherModel::new(&[10, 64, 32, 1], &[0, 1]);
        let input = Array1::from_vec(vec![0.1; 10]);
        let (output, hints) = teacher.forward_with_hints(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].len(), 64);
        assert_eq!(hints[1].len(), 32);
    }

    #[test]
    fn test_student_model_forward() {
        let student = StudentModel::new(&[10, 16, 8, 1], &[0], &[(16, 64)]);
        let input = Array1::from_vec(vec![0.1; 10]);
        let (output, hints) = student.forward_with_hints(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].len(), 16);
    }

    #[test]
    fn test_hint_projection() {
        let student = StudentModel::new(&[10, 16, 8, 1], &[0], &[(16, 64)]);
        let input = Array1::from_vec(vec![0.1; 10]);
        let (_, hints) = student.forward_with_hints(&input);
        let projected = student.project_hints(&hints);
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].len(), 64);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((mse_loss(&pred, &target)).abs() < 1e-10);

        let pred2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target2 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert!((mse_loss(&pred2, &target2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_matching_loss() {
        let s_hints = vec![Array1::from_vec(vec![1.0, 2.0])];
        let t_hints = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let regressors = vec![DenseLayer::new(2, 3)];
        let loss = feature_matching_loss(&s_hints, &t_hints, &regressors);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_attention_transfer_loss() {
        let s_hints = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let t_hints = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let loss = attention_transfer_loss(&s_hints, &t_hints);
        assert!(loss.abs() < 1e-10); // Same features should give zero loss
    }

    #[test]
    fn test_attention_transfer_loss_different() {
        let s_hints = vec![Array1::from_vec(vec![1.0, 0.0, 0.0])];
        let t_hints = vec![Array1::from_vec(vec![0.0, 0.0, 1.0])];
        let loss = attention_transfer_loss(&s_hints, &t_hints);
        assert!(loss.abs() < 1e-10); // Same magnitude distribution
    }

    #[test]
    fn test_combined_loss() {
        let student_out = Array1::from_vec(vec![0.5]);
        let target = Array1::from_vec(vec![1.0]);
        let teacher_out = Array1::from_vec(vec![0.8]);
        let s_hints = vec![Array1::from_vec(vec![1.0, 2.0])];
        let t_hints = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let regressors = vec![DenseLayer::new(2, 3)];

        let loss = combined_loss(
            &student_out, &target, &teacher_out,
            &s_hints, &t_hints, &regressors,
            1.0, 0.5, 0.3, 0.1,
        );
        assert!(loss > 0.0);
    }

    #[test]
    fn test_multi_teacher_aggregator() {
        let t1 = TeacherModel::new(&[10, 32, 1], &[]);
        let t2 = TeacherModel::new(&[10, 32, 1], &[]);
        let t3 = TeacherModel::new(&[10, 32, 1], &[]);

        let aggregator = MultiTeacherAggregator::new(
            vec![t1, t2, t3],
            vec![1.0, 1.0, 1.0],
        );

        let input = Array1::from_vec(vec![0.1; 10]);
        let output = aggregator.aggregate(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_regime_detection() {
        // Create synthetic bull market candles
        let mut candles = Vec::new();
        for i in 0..30 {
            let price = 100.0 + i as f64 * 2.0; // steadily rising
            candles.push(Candle {
                timestamp: i as u64,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
                turnover: price * 1000.0,
            });
        }
        let regime = detect_regime(&candles, 20);
        assert_eq!(regime, MarketRegime::Bull);
    }

    #[test]
    fn test_regime_detection_bear() {
        let mut candles = Vec::new();
        for i in 0..30 {
            let price = 200.0 - i as f64 * 2.0; // steadily falling
            candles.push(Candle {
                timestamp: i as u64,
                open: price + 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
                turnover: price * 1000.0,
            });
        }
        let regime = detect_regime(&candles, 20);
        assert_eq!(regime, MarketRegime::Bear);
    }

    #[test]
    fn test_prepare_data() {
        let mut candles = Vec::new();
        for i in 0..20 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            candles.push(Candle {
                timestamp: i as u64,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
                turnover: price * 1000.0,
            });
        }
        let (features, labels) = prepare_data(&candles, 5);
        assert!(!features.is_empty());
        assert_eq!(features.len(), labels.len());
        assert_eq!(features[0].len(), 25); // 5 candles * 5 features
    }

    #[test]
    fn test_teacher_training() {
        let mut teacher = TeacherModel::new(&[10, 32, 16, 1], &[0, 1]);
        let input = Array1::from_vec(vec![0.5; 10]);
        let target = Array1::from_vec(vec![1.0]);

        let loss1 = teacher.train_step(&input, &target, 0.01);
        // Run a few more steps
        let mut loss_final = loss1;
        for _ in 0..50 {
            loss_final = teacher.train_step(&input, &target, 0.01);
        }
        // Loss should decrease
        assert!(loss_final < loss1 || loss1 < 0.01);
    }

    #[test]
    fn test_trainer_creation() {
        let student = StudentModel::new(&[10, 16, 1], &[0], &[(16, 32)]);
        let trainer = TeacherStudentTrainer::new(student, 0.001)
            .with_loss_weights(1.0, 0.5, 0.3, 0.1);
        assert_eq!(trainer.alpha, 1.0);
        assert_eq!(trainer.beta, 0.5);
        assert_eq!(trainer.gamma, 0.3);
        assert_eq!(trainer.delta, 0.1);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_outer_product() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0, 5.0]);
        let result = outer_product(&a, &b);
        assert_eq!(result.dim(), (2, 3));
        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[1, 2]], 10.0);
    }
}
