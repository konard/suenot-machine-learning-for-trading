//! # Knowledge Distillation in Federated Learning for Trading
//!
//! This library implements the FedMD (Federated Model Distillation) framework,
//! combining knowledge distillation with federated learning for efficient
//! deployment of trading models on edge devices.
//!
//! ## Key Components
//! - `TeacherModel`: Large multi-layer network that generates soft labels
//! - `StudentModel`: Compact network that learns from teacher's soft labels
//! - `FederatedDistillation`: FedMD coordinator managing teacher-student communication
//! - Utility functions: softmax with temperature, KL divergence, cross-entropy

use ndarray::{Array1, Array2};
use rand::Rng;

// ---------------------------------------------------------------------------
// Activation & Loss Functions
// ---------------------------------------------------------------------------

/// Compute softmax with temperature scaling.
///
/// Temperature T > 1 produces softer distributions, revealing the teacher's
/// "dark knowledge" about inter-class relationships.
///
/// Uses the log-sum-exp trick (subtracting max) for numerical stability.
pub fn softmax_with_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits
        .iter()
        .map(|&z| ((z - max_logit) / temperature).exp())
        .collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Compute KL divergence D_KL(P || Q).
///
/// Measures how much information is lost when using Q to approximate P.
/// Returns 0 when P == Q. Always non-negative.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "Distributions must have equal length");
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > 1e-10 {
                pi * (pi / qi.max(1e-10)).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Compute cross-entropy loss H(P, Q) = -sum(P * log(Q)).
pub fn cross_entropy(target: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(target.len(), predicted.len());
    -target
        .iter()
        .zip(predicted.iter())
        .map(|(&t, &p)| {
            if t > 1e-10 {
                t * p.max(1e-10).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// ReLU activation: max(0, x).
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Convert a class index to a one-hot vector.
pub fn one_hot(class: usize, num_classes: usize) -> Vec<f64> {
    let mut v = vec![0.0; num_classes];
    if class < num_classes {
        v[class] = 1.0;
    }
    v
}

/// Return the index of the maximum element (argmax).
pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Teacher Model
// ---------------------------------------------------------------------------

/// A multi-layer neural network serving as the teacher model.
///
/// Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> output
///
/// The teacher is a larger model that captures complex market patterns
/// and generates soft labels for student training.
pub struct TeacherModel {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
    pub learning_rate: f64,
}

impl TeacherModel {
    /// Create a new teacher model with random Xavier initialization.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden1_size` - First hidden layer size
    /// * `hidden2_size` - Second hidden layer size
    /// * `output_size` - Number of output classes
    /// * `learning_rate` - SGD learning rate
    pub fn new(
        input_size: usize,
        hidden1_size: usize,
        hidden2_size: usize,
        output_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut xavier = |fan_in: usize, fan_out: usize| -> f64 {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            rng.gen_range(-limit..limit)
        };

        let weights1 =
            Array2::from_shape_fn((input_size, hidden1_size), |_| xavier(input_size, hidden1_size));
        let bias1 = Array1::zeros(hidden1_size);
        let weights2 = Array2::from_shape_fn((hidden1_size, hidden2_size), |_| {
            xavier(hidden1_size, hidden2_size)
        });
        let bias2 = Array1::zeros(hidden2_size);
        let weights3 = Array2::from_shape_fn((hidden2_size, output_size), |_| {
            xavier(hidden2_size, output_size)
        });
        let bias3 = Array1::zeros(output_size);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
            learning_rate,
        }
    }

    /// Forward pass: compute output logits from input features.
    pub fn forward(&self, input: &Array1<f64>) -> Vec<f64> {
        // Layer 1: input -> hidden1
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);

        // Layer 2: hidden1 -> hidden2
        let h2_raw = h1.dot(&self.weights2) + &self.bias2;
        let h2: Array1<f64> = h2_raw.mapv(relu);

        // Layer 3: hidden2 -> output (logits, no activation)
        let output = h2.dot(&self.weights3) + &self.bias3;
        output.to_vec()
    }

    /// Generate soft labels using temperature scaling.
    pub fn soft_predict(&self, input: &Array1<f64>, temperature: f64) -> Vec<f64> {
        let logits = self.forward(input);
        softmax_with_temperature(&logits, temperature)
    }

    /// Train on a single sample using backpropagation with hard labels.
    ///
    /// Uses simplified gradient descent with numerical stability guards.
    pub fn train_step(&mut self, input: &Array1<f64>, target: &[f64]) {
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);
        let h2_raw = h1.dot(&self.weights2) + &self.bias2;
        let h2: Array1<f64> = h2_raw.mapv(relu);
        let logits_arr = h2.dot(&self.weights3) + &self.bias3;
        let logits = logits_arr.to_vec();
        let probs = softmax_with_temperature(&logits, 1.0);

        // Output gradient: dL/d_logits = probs - target
        let num_out = probs.len();
        let mut d_logits = Array1::zeros(num_out);
        for i in 0..num_out {
            d_logits[i] = probs[i] - target[i];
        }

        // Gradient for weights3: h2^T * d_logits
        let h2_col = h2.clone().into_shape((h2.len(), 1)).unwrap();
        let dl_row = d_logits.clone().into_shape((1, num_out)).unwrap();
        let dw3 = h2_col.dot(&dl_row);

        // Backprop through layer 2
        let d_h2 = d_logits.dot(&self.weights3.t());
        let d_h2_relu: Array1<f64> = d_h2
            .iter()
            .zip(h2_raw.iter())
            .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
            .collect();

        let h1_col = h1.clone().into_shape((h1.len(), 1)).unwrap();
        let dh2_row = d_h2_relu
            .clone()
            .into_shape((1, d_h2_relu.len()))
            .unwrap();
        let dw2 = h1_col.dot(&dh2_row);

        // Backprop through layer 1
        let d_h1 = d_h2_relu.dot(&self.weights2.t());
        let d_h1_relu: Array1<f64> = d_h1
            .iter()
            .zip(h1_raw.iter())
            .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
            .collect();

        let inp_col = input.clone().into_shape((input.len(), 1)).unwrap();
        let dh1_row = d_h1_relu
            .clone()
            .into_shape((1, d_h1_relu.len()))
            .unwrap();
        let dw1 = inp_col.dot(&dh1_row);

        // Update weights
        let lr = self.learning_rate;
        self.weights3 = &self.weights3 - &(dw3 * lr);
        self.bias3 = &self.bias3 - &(d_logits * lr);
        self.weights2 = &self.weights2 - &(dw2 * lr);
        self.bias2 = &self.bias2 - &(d_h2_relu * lr);
        self.weights1 = &self.weights1 - &(dw1 * lr);
        self.bias1 = &self.bias1 - &(d_h1_relu * lr);
    }

    /// Train on a batch of samples for multiple epochs.
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
}

// ---------------------------------------------------------------------------
// Student Model
// ---------------------------------------------------------------------------

/// A compact neural network serving as the student model.
///
/// Architecture: input -> hidden (ReLU) -> output
///
/// Designed for edge deployment with minimal parameters and fast inference.
pub struct StudentModel {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub learning_rate: f64,
    pub num_classes: usize,
}

impl StudentModel {
    /// Create a new student model with random Xavier initialization.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut xavier = |fan_in: usize, fan_out: usize| -> f64 {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            rng.gen_range(-limit..limit)
        };

        let weights1 =
            Array2::from_shape_fn((input_size, hidden_size), |_| xavier(input_size, hidden_size));
        let bias1 = Array1::zeros(hidden_size);
        let weights2 =
            Array2::from_shape_fn((hidden_size, output_size), |_| xavier(hidden_size, output_size));
        let bias2 = Array1::zeros(output_size);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            learning_rate,
            num_classes: output_size,
        }
    }

    /// Forward pass: compute output logits.
    pub fn forward(&self, input: &Array1<f64>) -> Vec<f64> {
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);
        let output = h1.dot(&self.weights2) + &self.bias2;
        output.to_vec()
    }

    /// Predict class probabilities with standard softmax.
    pub fn predict(&self, input: &Array1<f64>) -> Vec<f64> {
        let logits = self.forward(input);
        softmax_with_temperature(&logits, 1.0)
    }

    /// Predict using temperature-scaled softmax (for distillation).
    pub fn soft_predict(&self, input: &Array1<f64>, temperature: f64) -> Vec<f64> {
        let logits = self.forward(input);
        softmax_with_temperature(&logits, temperature)
    }

    /// Train with combined distillation + hard-label loss.
    ///
    /// Loss = alpha * T^2 * KL(teacher_soft || student_soft)
    ///      + (1 - alpha) * CrossEntropy(hard_label, student)
    ///
    /// # Arguments
    /// * `input` - Input feature vector
    /// * `hard_target` - One-hot encoded ground truth label
    /// * `teacher_soft` - Teacher's temperature-scaled soft predictions
    /// * `temperature` - Temperature for soft label generation
    /// * `alpha` - Weight for distillation loss (0 = hard only, 1 = soft only)
    pub fn distillation_step(
        &mut self,
        input: &Array1<f64>,
        hard_target: &[f64],
        teacher_soft: &[f64],
        temperature: f64,
        alpha: f64,
    ) {
        // Forward pass
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);
        let logits_arr = h1.dot(&self.weights2) + &self.bias2;
        let logits = logits_arr.to_vec();

        let student_hard = softmax_with_temperature(&logits, 1.0);
        let student_soft = softmax_with_temperature(&logits, temperature);

        // Combined gradient at output logits
        let num_out = student_hard.len();
        let mut d_logits = Array1::zeros(num_out);
        for i in 0..num_out {
            // Hard label gradient: (student_prob - target)
            let d_hard = student_hard[i] - hard_target[i];
            // Soft label gradient: T * (student_soft - teacher_soft)
            // The T^2 in the loss and 1/T from gradient cancel to give T
            let d_soft = temperature * (student_soft[i] - teacher_soft[i]);
            d_logits[i] = (1.0 - alpha) * d_hard + alpha * d_soft;
        }

        // Backprop through output layer
        let h1_col = h1.clone().into_shape((h1.len(), 1)).unwrap();
        let dl_row = d_logits.clone().into_shape((1, num_out)).unwrap();
        let dw2 = h1_col.dot(&dl_row);

        // Backprop through hidden layer
        let d_h1 = d_logits.dot(&self.weights2.t());
        let d_h1_relu: Array1<f64> = d_h1
            .iter()
            .zip(h1_raw.iter())
            .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
            .collect();

        let inp_col = input.clone().into_shape((input.len(), 1)).unwrap();
        let dh1_row = d_h1_relu
            .clone()
            .into_shape((1, d_h1_relu.len()))
            .unwrap();
        let dw1 = inp_col.dot(&dh1_row);

        // Update weights
        let lr = self.learning_rate;
        self.weights2 = &self.weights2 - &(dw2 * lr);
        self.bias2 = &self.bias2 - &(d_logits * lr);
        self.weights1 = &self.weights1 - &(dw1 * lr);
        self.bias1 = &self.bias1 - &(d_h1_relu * lr);
    }

    /// Batch distillation training.
    pub fn distill_batch(
        &mut self,
        inputs: &[Array1<f64>],
        hard_targets: &[Vec<f64>],
        teacher_softs: &[Vec<f64>],
        temperature: f64,
        alpha: f64,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for i in 0..inputs.len() {
                self.distillation_step(
                    &inputs[i],
                    &hard_targets[i],
                    &teacher_softs[i],
                    temperature,
                    alpha,
                );
            }
        }
    }

    /// Count total parameters in the student model.
    pub fn num_parameters(&self) -> usize {
        let (r1, c1) = self.weights1.dim();
        let (r2, c2) = self.weights2.dim();
        r1 * c1 + self.bias1.len() + r2 * c2 + self.bias2.len()
    }
}

// ---------------------------------------------------------------------------
// Federated Distillation (FedMD)
// ---------------------------------------------------------------------------

/// A single client in the FedMD federation.
pub struct FedClient {
    /// The client's local student model.
    pub model: StudentModel,
    /// Local training data (features).
    pub local_inputs: Vec<Array1<f64>>,
    /// Local training labels (one-hot).
    pub local_targets: Vec<Vec<f64>>,
}

/// FedMD coordinator managing the federated distillation process.
///
/// Orchestrates teacher-student knowledge transfer across multiple clients,
/// each potentially running a different model architecture.
pub struct FederatedDistillation {
    /// Central teacher model.
    pub teacher: TeacherModel,
    /// Participating clients with their student models and local data.
    pub clients: Vec<FedClient>,
    /// Shared public dataset used for knowledge transfer.
    pub public_inputs: Vec<Array1<f64>>,
    /// Hard labels for the public dataset.
    pub public_targets: Vec<Vec<f64>>,
    /// Temperature for soft label generation.
    pub temperature: f64,
    /// Balance between distillation and hard-label loss.
    pub alpha: f64,
}

impl FederatedDistillation {
    /// Create a new FedMD coordinator.
    pub fn new(
        teacher: TeacherModel,
        clients: Vec<FedClient>,
        public_inputs: Vec<Array1<f64>>,
        public_targets: Vec<Vec<f64>>,
        temperature: f64,
        alpha: f64,
    ) -> Self {
        Self {
            teacher,
            clients,
            public_inputs,
            public_targets,
            temperature,
            alpha,
        }
    }

    /// Execute one round of FedMD.
    ///
    /// 1. Teacher generates soft labels on public data
    /// 2. Each client trains on local data + distills from teacher soft labels
    /// 3. Clients produce updated predictions on public data
    /// 4. Server aggregates predictions and updates teacher
    pub fn run_round(&mut self, local_epochs: usize, distill_epochs: usize) {
        let temp = self.temperature;
        let alpha = self.alpha;

        // Step 1: Teacher generates soft labels on public data
        let teacher_soft_labels: Vec<Vec<f64>> = self
            .public_inputs
            .iter()
            .map(|x| self.teacher.soft_predict(x, temp))
            .collect();

        // Step 2 & 3: Each client trains locally and distills
        let num_clients = self.clients.len();
        let mut all_client_preds: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_clients);

        for client in &mut self.clients {
            // Local training with hard labels
            for _ in 0..local_epochs {
                for (input, target) in client
                    .local_inputs
                    .iter()
                    .zip(client.local_targets.iter())
                {
                    // Simple hard-label training step using distillation with alpha=0
                    client.model.distillation_step(
                        input,
                        target,
                        &vec![1.0 / target.len() as f64; target.len()], // uniform soft (ignored with alpha=0)
                        temp,
                        0.0, // pure hard label
                    );
                }
            }

            // Distill from teacher soft labels on public data
            client.model.distill_batch(
                &self.public_inputs.clone(),
                &self.public_targets.clone(),
                &teacher_soft_labels,
                temp,
                alpha,
                distill_epochs,
            );

            // Produce predictions on public data
            let preds: Vec<Vec<f64>> = self
                .public_inputs
                .iter()
                .map(|x| client.model.soft_predict(x, temp))
                .collect();
            all_client_preds.push(preds);
        }

        // Step 4: Aggregate client predictions (average)
        if !all_client_preds.is_empty() && !self.public_inputs.is_empty() {
            let num_samples = self.public_inputs.len();
            let num_classes = self.public_targets[0].len();

            let mut aggregated = vec![vec![0.0; num_classes]; num_samples];
            for client_preds in &all_client_preds {
                for (s, preds) in client_preds.iter().enumerate() {
                    for (c, &p) in preds.iter().enumerate() {
                        aggregated[s][c] += p / num_clients as f64;
                    }
                }
            }

            // Update teacher using aggregated predictions as soft targets
            for (input, agg_target) in self.public_inputs.iter().zip(aggregated.iter()) {
                self.teacher.train_step(input, agg_target);
            }
        }
    }

    /// Run multiple rounds of FedMD.
    pub fn run(&mut self, rounds: usize, local_epochs: usize, distill_epochs: usize) {
        for round in 0..rounds {
            self.run_round(local_epochs, distill_epochs);
            if (round + 1) % 5 == 0 || round == 0 {
                let acc = self.evaluate_teacher();
                println!(
                    "  FedMD Round {}/{}: Teacher accuracy = {:.1}%",
                    round + 1,
                    rounds,
                    acc * 100.0
                );
            }
        }
    }

    /// Evaluate teacher accuracy on the public dataset.
    pub fn evaluate_teacher(&self) -> f64 {
        let mut correct = 0;
        let total = self.public_inputs.len();
        for (input, target) in self.public_inputs.iter().zip(self.public_targets.iter()) {
            let pred = self.teacher.soft_predict(input, 1.0);
            if argmax(&pred) == argmax(target) {
                correct += 1;
            }
        }
        correct as f64 / total.max(1) as f64
    }

    /// Evaluate a specific client's student model on the public dataset.
    pub fn evaluate_client(&self, client_idx: usize) -> f64 {
        if client_idx >= self.clients.len() {
            return 0.0;
        }
        let client = &self.clients[client_idx];
        let mut correct = 0;
        let total = self.public_inputs.len();
        for (input, target) in self.public_inputs.iter().zip(self.public_targets.iter()) {
            let pred = client.model.predict(input);
            if argmax(&pred) == argmax(target) {
                correct += 1;
            }
        }
        correct as f64 / total.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_high_temperature() {
        let logits = vec![1.0, 2.0, 3.0];
        let hard = softmax_with_temperature(&logits, 1.0);
        let soft = softmax_with_temperature(&logits, 10.0);
        // Higher temperature -> more uniform distribution
        let hard_range = hard[2] - hard[0];
        let soft_range = soft[2] - soft[0];
        assert!(soft_range < hard_range);
    }

    #[test]
    fn test_kl_divergence_same_dist() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different_dist() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_cross_entropy() {
        let target = vec![1.0, 0.0, 0.0];
        let pred = vec![0.9, 0.05, 0.05];
        let ce = cross_entropy(&target, &pred);
        assert!(ce > 0.0);
        assert!(ce < 1.0);
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
    fn test_teacher_forward() {
        let teacher = TeacherModel::new(4, 16, 8, 3, 0.01);
        let input = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let logits = teacher.forward(&input);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_student_forward() {
        let student = StudentModel::new(4, 8, 3, 0.01);
        let input = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let probs = student.predict(&input);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_student_distillation() {
        let mut student = StudentModel::new(4, 8, 3, 0.01);
        let input = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let hard_target = vec![1.0, 0.0, 0.0];
        let teacher_soft = vec![0.7, 0.2, 0.1];

        // Should not panic
        student.distillation_step(&input, &hard_target, &teacher_soft, 3.0, 0.7);
        let probs = student.predict(&input);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_student_num_parameters() {
        let student = StudentModel::new(4, 8, 3, 0.01);
        // weights1: 4*8 = 32, bias1: 8, weights2: 8*3 = 24, bias2: 3 = 67
        assert_eq!(student.num_parameters(), 67);
    }

    #[test]
    fn test_federated_distillation() {
        let num_features = 4;
        let num_classes = 3;
        let mut rng = rand::thread_rng();

        let teacher = TeacherModel::new(num_features, 16, 8, num_classes, 0.01);

        // Create 2 clients with different hidden sizes
        let clients: Vec<FedClient> = (0..2)
            .map(|_| {
                let hidden = 6 + rng.gen_range(0..5);
                let model = StudentModel::new(num_features, hidden, num_classes, 0.01);
                let local_inputs: Vec<Array1<f64>> = (0..10)
                    .map(|_| Array1::from_vec(vec![rng.gen_range(-1.0..1.0); num_features]))
                    .collect();
                let local_targets: Vec<Vec<f64>> = (0..10)
                    .map(|_| one_hot(rng.gen_range(0..num_classes), num_classes))
                    .collect();
                FedClient {
                    model,
                    local_inputs,
                    local_targets,
                }
            })
            .collect();

        let public_inputs: Vec<Array1<f64>> = (0..20)
            .map(|_| Array1::from_vec(vec![rng.gen_range(-1.0..1.0); num_features]))
            .collect();
        let public_targets: Vec<Vec<f64>> = (0..20)
            .map(|_| one_hot(rng.gen_range(0..num_classes), num_classes))
            .collect();

        let mut fedmd = FederatedDistillation::new(
            teacher,
            clients,
            public_inputs,
            public_targets,
            3.0,
            0.7,
        );

        // Should run without panicking
        fedmd.run_round(2, 2);

        let teacher_acc = fedmd.evaluate_teacher();
        assert!(teacher_acc >= 0.0 && teacher_acc <= 1.0);

        let client_acc = fedmd.evaluate_client(0);
        assert!(client_acc >= 0.0 && client_acc <= 1.0);
    }
}
