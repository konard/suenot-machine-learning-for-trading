//! Simple neural network classifier for trading signals.

use rand::Rng;
use rand_distr::StandardNormal;

use super::ModelConfig;

/// Trading classifier using a simple feedforward neural network.
///
/// This is a simplified implementation for demonstration purposes.
/// For production use, consider using a proper ML framework.
pub struct TradingClassifier {
    config: ModelConfig,
    // Weights: input -> hidden
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    // Weights: hidden -> hidden
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    // Weights: hidden -> output
    w3: Vec<Vec<f64>>,
    b3: Vec<f64>,
}

impl TradingClassifier {
    /// Create a new classifier with random weights.
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Self {
        let config = ModelConfig::new(input_dim, hidden_dim, num_classes);

        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + hidden_dim) as f64).sqrt();
        let scale3 = (2.0 / (hidden_dim + num_classes) as f64).sqrt();

        let w1 = Self::init_weights(&mut rng, hidden_dim, input_dim, scale1);
        let b1 = vec![0.0; hidden_dim];

        let w2 = Self::init_weights(&mut rng, hidden_dim, hidden_dim, scale2);
        let b2 = vec![0.0; hidden_dim];

        let w3 = Self::init_weights(&mut rng, num_classes, hidden_dim, scale3);
        let b3 = vec![0.0; num_classes];

        Self {
            config,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    fn init_weights<R: Rng>(rng: &mut R, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * scale)
                    .collect()
            })
            .collect()
    }

    /// Forward pass through the network.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // Layer 1: Linear + ReLU
        let h1 = self.linear(&self.w1, &self.b1, x);
        let h1 = self.relu(&h1);

        // Layer 2: Linear + ReLU
        let h2 = self.linear(&self.w2, &self.b2, &h1);
        let h2 = self.relu(&h2);

        // Layer 3: Linear (logits)
        self.linear(&self.w3, &self.b3, &h2)
    }

    /// Predict class probabilities.
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        let logits = self.forward(x);
        self.softmax(&logits)
    }

    /// Predict class index.
    pub fn predict(&self, x: &[f64]) -> usize {
        let probs = self.predict_proba(x);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1)
    }

    /// Get class name.
    pub fn get_class_name(&self, class_idx: usize) -> &'static str {
        match class_idx {
            0 => "SELL",
            1 => "HOLD",
            2 => "BUY",
            _ => "UNKNOWN",
        }
    }

    /// Train the model on data.
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        labels: &[usize],
        epochs: usize,
        learning_rate: f64,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            for (x, &y) in features.iter().zip(labels.iter()) {
                // Forward pass
                let h1 = self.linear(&self.w1, &self.b1, x);
                let h1_relu = self.relu(&h1);

                let h2 = self.linear(&self.w2, &self.b2, &h1_relu);
                let h2_relu = self.relu(&h2);

                let logits = self.linear(&self.w3, &self.b3, &h2_relu);
                let probs = self.softmax(&logits);

                // Compute loss
                let loss = -probs[y].ln();
                total_loss += loss;

                if self.predict(x) == y {
                    correct += 1;
                }

                // Backward pass (simplified gradient descent)
                // Gradient of softmax cross-entropy
                let mut grad_logits = probs.clone();
                grad_logits[y] -= 1.0;

                // Backprop through layer 3
                let grad_h2 = self.matmul_transpose(&self.w3, &grad_logits);
                self.update_weights(&mut self.w3, &mut self.b3, &h2_relu, &grad_logits, learning_rate);

                // Backprop through layer 2
                let grad_h2_relu = self.relu_backward(&h2, &grad_h2);
                let grad_h1 = self.matmul_transpose(&self.w2, &grad_h2_relu);
                self.update_weights(&mut self.w2, &mut self.b2, &h1_relu, &grad_h2_relu, learning_rate);

                // Backprop through layer 1
                let grad_h1_relu = self.relu_backward(&h1, &grad_h1);
                self.update_weights(&mut self.w1, &mut self.b1, x, &grad_h1_relu, learning_rate);
            }

            if (epoch + 1) % 10 == 0 {
                let avg_loss = total_loss / features.len() as f64;
                let accuracy = correct as f64 / features.len() as f64;
                println!(
                    "Epoch {}: loss={:.4}, accuracy={:.2}%",
                    epoch + 1,
                    avg_loss,
                    accuracy * 100.0
                );
            }
        }
    }

    fn linear(&self, w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
        w.iter()
            .zip(b.iter())
            .map(|(row, &bias)| {
                row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum::<f64>() + bias
            })
            .collect()
    }

    fn relu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    fn relu_backward(&self, x: &[f64], grad: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(grad.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect()
    }

    fn softmax(&self, x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
        x.iter().map(|&v| (v - max_val).exp() / exp_sum).collect()
    }

    fn matmul_transpose(&self, w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let cols = w[0].len();
        (0..cols)
            .map(|j| {
                w.iter()
                    .zip(x.iter())
                    .map(|(row, &xi)| row[j] * xi)
                    .sum()
            })
            .collect()
    }

    fn update_weights(
        &mut self,
        w: &mut [Vec<f64>],
        b: &mut [f64],
        input: &[f64],
        grad: &[f64],
        lr: f64,
    ) {
        for (i, (row, &g)) in w.iter_mut().zip(grad.iter()).enumerate() {
            for (j, w_ij) in row.iter_mut().enumerate() {
                *w_ij -= lr * g * input[j];
            }
            b[i] -= lr * g;
        }
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Get number of classes.
    pub fn num_classes(&self) -> usize {
        self.config.num_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let model = TradingClassifier::new(6, 64, 3);
        assert_eq!(model.input_dim(), 6);
        assert_eq!(model.num_classes(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let model = TradingClassifier::new(6, 64, 3);
        let x = vec![0.5, 0.1, -0.2, 0.3, 0.0, -0.1];
        let logits = model.forward(&x);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_predict() {
        let model = TradingClassifier::new(6, 64, 3);
        let x = vec![0.5, 0.1, -0.2, 0.3, 0.0, -0.1];
        let pred = model.predict(&x);
        assert!(pred < 3);
    }
}
