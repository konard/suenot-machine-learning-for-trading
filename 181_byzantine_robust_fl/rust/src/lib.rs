//! # Byzantine-Robust Federated Learning for Trading
//!
//! This library implements Byzantine-robust aggregation methods for federated
//! learning in trading systems. It provides three robust aggregation rules:
//!
//! - **Krum**: selects the gradient closest to its neighbors
//! - **Trimmed Mean**: coordinate-wise trimming of extreme values
//! - **Median**: coordinate-wise median for maximum breakdown point
//!
//! These methods protect against Byzantine (malicious or faulty) participants
//! that may send arbitrary gradient updates.

use ndarray::Array1;
use rand::Rng;

// --------------- Aggregation Methods ---------------

/// Krum aggregation: selects the gradient closest to its (n - f - 2) nearest
/// neighbors. Byzantine gradients tend to be outliers, so Krum picks the
/// gradient at the center of the honest cluster.
///
/// # Arguments
/// * `gradients` - Slice of gradient vectors from all clients
/// * `num_byzantine` - Upper bound on the number of Byzantine clients (f)
///
/// # Returns
/// The selected gradient vector (clone of the winning gradient)
///
/// # Panics
/// Panics if `gradients.len() < num_byzantine + 3` (need at least n-f-2 >= 1 neighbors)
pub fn krum_aggregate(gradients: &[Array1<f64>], num_byzantine: usize) -> Array1<f64> {
    let n = gradients.len();
    assert!(
        n >= num_byzantine + 3,
        "Need at least {} gradients for Krum with f={}, got {}",
        num_byzantine + 3,
        num_byzantine,
        n
    );

    let num_closest = n - num_byzantine - 2;
    let mut best_idx = 0;
    let mut best_score = f64::MAX;

    for i in 0..n {
        // Compute squared distances to all other gradients
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (&gradients[i] - &gradients[j]).mapv(|x| x * x).sum())
            .collect();

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Score = sum of distances to num_closest nearest neighbors
        let score: f64 = dists[..num_closest].iter().sum();

        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }

    gradients[best_idx].clone()
}

/// Multi-Krum aggregation: selects the top `m` gradients by Krum score
/// and averages them. This reduces variance compared to single Krum.
///
/// # Arguments
/// * `gradients` - Slice of gradient vectors from all clients
/// * `num_byzantine` - Upper bound on the number of Byzantine clients (f)
/// * `m` - Number of top gradients to average (must satisfy m <= n - f)
pub fn multi_krum_aggregate(
    gradients: &[Array1<f64>],
    num_byzantine: usize,
    m: usize,
) -> Array1<f64> {
    let n = gradients.len();
    assert!(n >= num_byzantine + 3);
    assert!(m <= n - num_byzantine && m > 0);

    let num_closest = n - num_byzantine - 2;

    // Compute Krum scores for all gradients
    let mut scores: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (&gradients[i] - &gradients[j]).mapv(|x| x * x).sum())
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f64 = dists[..num_closest].iter().sum();
            (i, score)
        })
        .collect();

    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Average the top m gradients
    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);
    for &(idx, _) in scores[..m].iter() {
        result = result + &gradients[idx];
    }
    result / m as f64
}

/// Coordinate-wise trimmed mean aggregation. For each coordinate, sorts
/// the values, removes the `trim_count` smallest and largest, and averages
/// the remaining values.
///
/// # Arguments
/// * `gradients` - Slice of gradient vectors from all clients
/// * `trim_count` - Number of values to trim from each end (should be >= f)
///
/// # Panics
/// Panics if `gradients.len() <= 2 * trim_count`
pub fn trimmed_mean_aggregate(
    gradients: &[Array1<f64>],
    trim_count: usize,
) -> Array1<f64> {
    let n = gradients.len();
    assert!(
        n > 2 * trim_count,
        "Need more than {} gradients for trim_count={}, got {}",
        2 * trim_count,
        trim_count,
        n
    );

    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);
    let remaining = n - 2 * trim_count;

    for j in 0..dim {
        let mut vals: Vec<f64> = gradients.iter().map(|g| g[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let trimmed_sum: f64 = vals[trim_count..n - trim_count].iter().sum();
        result[j] = trimmed_sum / remaining as f64;
    }

    result
}

/// Coordinate-wise median aggregation. Takes the median of each coordinate
/// independently. Achieves a breakdown point of 50% -- the maximum possible.
///
/// # Arguments
/// * `gradients` - Slice of gradient vectors from all clients
pub fn median_aggregate(gradients: &[Array1<f64>]) -> Array1<f64> {
    let n = gradients.len();
    assert!(!gradients.is_empty(), "Need at least one gradient");

    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);

    for j in 0..dim {
        let mut vals: Vec<f64> = gradients.iter().map(|g| g[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        result[j] = if n % 2 == 0 {
            (vals[n / 2 - 1] + vals[n / 2]) / 2.0
        } else {
            vals[n / 2]
        };
    }

    result
}

/// Simple averaging (FedAvg) -- no Byzantine robustness.
/// Included for comparison.
pub fn fedavg_aggregate(gradients: &[Array1<f64>]) -> Array1<f64> {
    let n = gradients.len();
    assert!(!gradients.is_empty());

    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);
    for g in gradients {
        result = result + g;
    }
    result / n as f64
}

// --------------- Activation & Loss Functions ---------------

/// Numerically stable softmax function.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits.iter().map(|&z| (z - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Cross-entropy loss between prediction and target.
pub fn cross_entropy(pred: &[f64], target: &[f64]) -> f64 {
    -pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            if t > 0.0 {
                t * p.max(1e-10).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// ReLU activation function.
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// One-hot encode a class label.
pub fn one_hot(label: usize, num_classes: usize) -> Vec<f64> {
    let mut v = vec![0.0; num_classes];
    if label < num_classes {
        v[label] = 1.0;
    }
    v
}

/// Return the index of the maximum value.
pub fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// --------------- Simple Neural Network Model ---------------

/// A simple 2-layer feedforward neural network for market direction prediction.
///
/// Architecture: input -> hidden (ReLU) -> output (softmax)
///
/// Weights are stored as flattened vectors for easy gradient extraction.
pub struct SimpleModel {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    learning_rate: f64,

    // Layer 1: input -> hidden
    w1: Vec<f64>, // input_dim * hidden_dim
    b1: Vec<f64>, // hidden_dim

    // Layer 2: hidden -> output
    w2: Vec<f64>, // hidden_dim * output_dim
    b2: Vec<f64>, // output_dim
}

impl SimpleModel {
    /// Create a new model with Xavier-initialized weights.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let w1_scale = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let w2_scale = (2.0 / (hidden_dim + output_dim) as f64).sqrt();

        let w1: Vec<f64> = (0..input_dim * hidden_dim)
            .map(|_| rng.gen_range(-w1_scale..w1_scale))
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2: Vec<f64> = (0..hidden_dim * output_dim)
            .map(|_| rng.gen_range(-w2_scale..w2_scale))
            .collect();
        let b2 = vec![0.0; output_dim];

        SimpleModel {
            input_dim,
            hidden_dim,
            output_dim,
            learning_rate,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Total number of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Forward pass: returns softmax probabilities.
    pub fn forward(&self, input: &Array1<f64>) -> Vec<f64> {
        // Hidden layer: ReLU(W1 * x + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut val = self.b1[h];
            for i in 0..self.input_dim {
                val += self.w1[i * self.hidden_dim + h] * input[i];
            }
            hidden[h] = relu(val);
        }

        // Output layer: softmax(W2 * hidden + b2)
        let mut logits = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            let mut val = self.b2[o];
            for h in 0..self.hidden_dim {
                val += self.w2[h * self.output_dim + o] * hidden[h];
            }
            logits[o] = val;
        }

        softmax(&logits)
    }

    /// Get all parameters as a flat vector.
    pub fn get_params(&self) -> Array1<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        params.extend_from_slice(&self.w1);
        params.extend_from_slice(&self.b1);
        params.extend_from_slice(&self.w2);
        params.extend_from_slice(&self.b2);
        Array1::from_vec(params)
    }

    /// Set all parameters from a flat vector.
    pub fn set_params(&mut self, params: &Array1<f64>) {
        let mut offset = 0;

        let w1_len = self.w1.len();
        self.w1.copy_from_slice(&params.as_slice().unwrap()[offset..offset + w1_len]);
        offset += w1_len;

        let b1_len = self.b1.len();
        self.b1.copy_from_slice(&params.as_slice().unwrap()[offset..offset + b1_len]);
        offset += b1_len;

        let w2_len = self.w2.len();
        self.w2.copy_from_slice(&params.as_slice().unwrap()[offset..offset + w2_len]);
        offset += w2_len;

        let b2_len = self.b2.len();
        self.b2.copy_from_slice(&params.as_slice().unwrap()[offset..offset + b2_len]);
        offset += b2_len;

        assert_eq!(offset, params.len());
    }

    /// Compute gradient of cross-entropy loss via backpropagation.
    /// Returns the gradient as a flat vector matching the parameter layout.
    pub fn compute_gradient(&self, input: &Array1<f64>, target: &[f64]) -> Array1<f64> {
        // Forward pass (store intermediates)
        let mut hidden_pre = vec![0.0; self.hidden_dim];
        let mut hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut val = self.b1[h];
            for i in 0..self.input_dim {
                val += self.w1[i * self.hidden_dim + h] * input[i];
            }
            hidden_pre[h] = val;
            hidden[h] = relu(val);
        }

        let mut logits = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            let mut val = self.b2[o];
            for h in 0..self.hidden_dim {
                val += self.w2[h * self.output_dim + o] * hidden[h];
            }
            logits[o] = val;
        }

        let probs = softmax(&logits);

        // Backward pass
        // dL/d_logits = probs - target (for cross-entropy + softmax)
        let mut d_logits = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            d_logits[o] = probs[o] - target[o];
        }

        // Gradients for W2 and b2
        let mut dw2 = vec![0.0; self.hidden_dim * self.output_dim];
        let db2 = d_logits.clone();
        for h in 0..self.hidden_dim {
            for o in 0..self.output_dim {
                dw2[h * self.output_dim + o] = hidden[h] * d_logits[o];
            }
        }

        // Backprop through hidden layer
        let mut d_hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            for o in 0..self.output_dim {
                d_hidden[h] += self.w2[h * self.output_dim + o] * d_logits[o];
            }
            // ReLU derivative
            if hidden_pre[h] <= 0.0 {
                d_hidden[h] = 0.0;
            }
        }

        // Gradients for W1 and b1
        let mut dw1 = vec![0.0; self.input_dim * self.hidden_dim];
        let db1 = d_hidden.clone();
        for i in 0..self.input_dim {
            for h in 0..self.hidden_dim {
                dw1[i * self.hidden_dim + h] = input[i] * d_hidden[h];
            }
        }

        // Pack into flat vector
        let mut grad = Vec::with_capacity(self.num_parameters());
        grad.extend_from_slice(&dw1);
        grad.extend_from_slice(&db1);
        grad.extend_from_slice(&dw2);
        grad.extend_from_slice(&db2);
        Array1::from_vec(grad)
    }

    /// Train on a single sample using SGD.
    pub fn train_step(&mut self, input: &Array1<f64>, target: &[f64]) {
        let grad = self.compute_gradient(input, target);
        let params = self.get_params();
        let new_params = params - self.learning_rate * &grad;
        self.set_params(&new_params);
    }

    /// Train on a batch of samples for the given number of epochs.
    pub fn train_batch(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.train_step(input, target);
            }
        }
    }

    /// Compute average gradient over a batch of samples.
    pub fn batch_gradient(
        &self,
        inputs: &[Array1<f64>],
        targets: &[Vec<f64>],
    ) -> Array1<f64> {
        let n = inputs.len();
        assert!(n > 0, "Need at least one sample");

        let mut avg_grad = Array1::zeros(self.num_parameters());
        for (input, target) in inputs.iter().zip(targets.iter()) {
            avg_grad = avg_grad + self.compute_gradient(input, target);
        }
        avg_grad / n as f64
    }
}

impl Clone for SimpleModel {
    fn clone(&self) -> Self {
        SimpleModel {
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            output_dim: self.output_dim,
            learning_rate: self.learning_rate,
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }
}

// --------------- Byzantine Attack Functions ---------------

/// Sign-flip attack: negates the gradient and scales by factor.
/// This is one of the most effective Byzantine attacks.
pub fn sign_flip_attack(gradient: &Array1<f64>, scale: f64) -> Array1<f64> {
    gradient.mapv(|x| -x * scale)
}

/// Random noise attack: replaces gradient with random noise.
pub fn random_noise_attack(dim: usize, noise_scale: f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    Array1::from_vec(
        (0..dim)
            .map(|_| rng.gen_range(-noise_scale..noise_scale))
            .collect(),
    )
}

// --------------- Aggregation Rule Enum ---------------

/// Aggregation rule selector for Byzantine-robust FL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationRule {
    FedAvg,
    Krum,
    MultiKrum(usize), // m
    TrimmedMean(usize), // trim_count
    Median,
}

// --------------- Federated Learning Client ---------------

/// A client in the Byzantine-robust FL system.
pub struct FLClient {
    pub model: SimpleModel,
    pub local_inputs: Vec<Array1<f64>>,
    pub local_targets: Vec<Vec<f64>>,
    pub is_byzantine: bool,
}

// --------------- Byzantine-Robust FL Coordinator ---------------

/// Coordinator for Byzantine-robust federated learning.
///
/// Manages the global model, collects gradients from clients,
/// applies a robust aggregation rule, and updates the global model.
pub struct ByzantineFL {
    pub global_model: SimpleModel,
    pub clients: Vec<FLClient>,
    pub aggregation_rule: AggregationRule,
    pub num_byzantine: usize,
}

impl ByzantineFL {
    /// Create a new Byzantine-robust FL coordinator.
    pub fn new(
        global_model: SimpleModel,
        clients: Vec<FLClient>,
        aggregation_rule: AggregationRule,
        num_byzantine: usize,
    ) -> Self {
        ByzantineFL {
            global_model,
            clients,
            aggregation_rule,
            num_byzantine,
        }
    }

    /// Run one round of federated learning.
    ///
    /// 1. Broadcast global model to all clients
    /// 2. Each client computes gradient on local data
    /// 3. Byzantine clients corrupt their gradient
    /// 4. Server aggregates using robust rule
    /// 5. Update global model
    pub fn run_round(&mut self, local_epochs: usize) {
        let global_params = self.global_model.get_params();

        // Collect gradients from all clients
        let mut gradients: Vec<Array1<f64>> = Vec::with_capacity(self.clients.len());

        for client in &mut self.clients {
            // Sync client model with global model
            client.model.set_params(&global_params);

            // Train locally for several epochs
            let mut local_model = client.model.clone();
            local_model.train_batch(
                &client.local_inputs,
                &client.local_targets,
                local_epochs,
            );

            // Gradient = old_params - new_params (pseudo-gradient from local training)
            let local_params = local_model.get_params();
            let mut gradient = &global_params - &local_params;

            // Byzantine clients apply attack
            if client.is_byzantine {
                gradient = sign_flip_attack(&gradient, 3.0);
            }

            gradients.push(gradient);
        }

        // Aggregate using robust rule
        let aggregated = match self.aggregation_rule {
            AggregationRule::FedAvg => fedavg_aggregate(&gradients),
            AggregationRule::Krum => krum_aggregate(&gradients, self.num_byzantine),
            AggregationRule::MultiKrum(m) => {
                multi_krum_aggregate(&gradients, self.num_byzantine, m)
            }
            AggregationRule::TrimmedMean(trim) => trimmed_mean_aggregate(&gradients, trim),
            AggregationRule::Median => median_aggregate(&gradients),
        };

        // Update global model: θ_new = θ_old - aggregated_gradient
        let new_params = &global_params - &aggregated;
        self.global_model.set_params(&new_params);
    }

    /// Run multiple rounds of federated learning.
    pub fn run(&mut self, num_rounds: usize, local_epochs: usize) {
        for round in 0..num_rounds {
            self.run_round(local_epochs);
            if (round + 1) % 5 == 0 || round == 0 {
                println!("    Round {}/{} completed", round + 1, num_rounds);
            }
        }
    }

    /// Evaluate the global model on a test set.
    pub fn evaluate(
        &self,
        test_inputs: &[Array1<f64>],
        test_targets: &[Vec<f64>],
    ) -> f64 {
        if test_inputs.is_empty() {
            return 0.0;
        }

        let mut correct = 0;
        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let pred = self.global_model.forward(input);
            if argmax(&pred) == argmax(target) {
                correct += 1;
            }
        }
        correct as f64 / test_inputs.len() as f64
    }
}

// --------------- Tests ---------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1.0");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax must be stable for large values");
    }

    #[test]
    fn test_one_hot() {
        let oh = one_hot(1, 3);
        assert_eq!(oh, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.9, 0.05, 0.05]), 0);
    }

    #[test]
    fn test_krum_no_byzantine() {
        // All honest gradients clustered together, no Byzantine
        let g1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let g2 = Array1::from_vec(vec![1.1, 2.1, 3.1]);
        let g3 = Array1::from_vec(vec![0.9, 1.9, 2.9]);
        let g4 = Array1::from_vec(vec![1.05, 2.05, 3.05]);

        let result = krum_aggregate(&[g1, g2, g3, g4], 0);
        // Result should be one of the honest gradients
        assert!(result[0] > 0.5 && result[0] < 1.5);
    }

    #[test]
    fn test_krum_with_byzantine() {
        // 4 honest gradients near [1,1,1], 1 Byzantine at [100,100,100]
        let honest: Vec<Array1<f64>> = (0..4)
            .map(|i| Array1::from_vec(vec![1.0 + i as f64 * 0.1, 1.0, 1.0]))
            .collect();

        let byzantine = Array1::from_vec(vec![100.0, 100.0, 100.0]);

        let mut all = honest.clone();
        all.push(byzantine);

        let result = krum_aggregate(&all, 1);
        // Krum should select an honest gradient, not the Byzantine one
        assert!(
            result[0] < 2.0,
            "Krum selected Byzantine gradient: {:?}",
            result
        );
    }

    #[test]
    fn test_trimmed_mean_basic() {
        let g1 = Array1::from_vec(vec![1.0, 2.0]);
        let g2 = Array1::from_vec(vec![2.0, 3.0]);
        let g3 = Array1::from_vec(vec![3.0, 4.0]);
        let g4 = Array1::from_vec(vec![100.0, -100.0]); // Byzantine
        let g5 = Array1::from_vec(vec![-100.0, 200.0]); // Byzantine

        let result = trimmed_mean_aggregate(&[g1, g2, g3, g4, g5], 1);
        // After trimming 1 from each end, should average [2,3] for coordinate 0
        // and [3,4] (approximately) for coordinate 1
        assert!(
            (result[0] - 2.0).abs() < 0.1,
            "Expected ~2.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_median_basic() {
        let g1 = Array1::from_vec(vec![1.0]);
        let g2 = Array1::from_vec(vec![2.0]);
        let g3 = Array1::from_vec(vec![3.0]);
        let g4 = Array1::from_vec(vec![1000.0]); // Byzantine
        let g5 = Array1::from_vec(vec![-1000.0]); // Byzantine

        let result = median_aggregate(&[g1, g2, g3, g4, g5]);
        assert!(
            (result[0] - 2.0).abs() < 1e-10,
            "Median should be 2.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_fedavg_aggregate() {
        let g1 = Array1::from_vec(vec![1.0, 2.0]);
        let g2 = Array1::from_vec(vec![3.0, 4.0]);

        let result = fedavg_aggregate(&[g1, g2]);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sign_flip_attack() {
        let grad = Array1::from_vec(vec![1.0, -2.0, 3.0]);
        let attacked = sign_flip_attack(&grad, 2.0);
        assert!((attacked[0] - (-2.0)).abs() < 1e-10);
        assert!((attacked[1] - 4.0).abs() < 1e-10);
        assert!((attacked[2] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_model_forward_output_valid() {
        let model = SimpleModel::new(4, 8, 3, 0.01);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = model.forward(&input);

        assert_eq!(output.len(), 3);
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Output should sum to 1.0");
        assert!(output.iter().all(|&x| x >= 0.0), "All probs should be >= 0");
    }

    #[test]
    fn test_model_params_roundtrip() {
        let model = SimpleModel::new(4, 8, 3, 0.01);
        let params = model.get_params();
        assert_eq!(params.len(), model.num_parameters());

        let mut model2 = SimpleModel::new(4, 8, 3, 0.01);
        model2.set_params(&params);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let out1 = model.forward(&input);
        let out2 = model2.forward(&input);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_model_gradient_shape() {
        let model = SimpleModel::new(4, 8, 3, 0.01);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let target = vec![0.0, 1.0, 0.0];
        let grad = model.compute_gradient(&input, &target);
        assert_eq!(grad.len(), model.num_parameters());
    }

    #[test]
    fn test_multi_krum() {
        let honest: Vec<Array1<f64>> = (0..5)
            .map(|i| Array1::from_vec(vec![1.0 + i as f64 * 0.05, 2.0]))
            .collect();
        let byzantine = Array1::from_vec(vec![50.0, 50.0]);

        let mut all = honest;
        all.push(byzantine);

        let result = multi_krum_aggregate(&all, 1, 3);
        assert!(result[0] < 2.0, "Multi-Krum should avoid Byzantine gradient");
    }

    #[test]
    fn test_byzantine_fl_runs() {
        let num_features = 4;
        let num_classes = 3;
        let mut rng = rand::thread_rng();

        // Generate synthetic data
        let num_samples = 30;
        let inputs: Vec<Array1<f64>> = (0..num_samples)
            .map(|_| {
                Array1::from_vec(
                    (0..num_features)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect(),
                )
            })
            .collect();

        let targets: Vec<Vec<f64>> = (0..num_samples)
            .map(|_| one_hot(rng.gen_range(0..num_classes), num_classes))
            .collect();

        // Split data among 4 clients (1 Byzantine)
        let chunk_size = num_samples / 4;
        let mut clients = Vec::new();
        for i in 0..4 {
            let start = i * chunk_size;
            let end = if i == 3 { num_samples } else { (i + 1) * chunk_size };
            clients.push(FLClient {
                model: SimpleModel::new(num_features, 8, num_classes, 0.01),
                local_inputs: inputs[start..end].to_vec(),
                local_targets: targets[start..end].to_vec(),
                is_byzantine: i == 3, // Last client is Byzantine
            });
        }

        let global_model = SimpleModel::new(num_features, 8, num_classes, 0.01);
        let mut fl = ByzantineFL::new(global_model, clients, AggregationRule::Krum, 1);

        // Should run without panicking
        fl.run(3, 2);

        let acc = fl.evaluate(&inputs[..5], &targets[..5]);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
