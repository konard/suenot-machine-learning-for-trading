//! Multi-task neural network with shared encoder and task-specific heads.
//!
//! Architecture:
//!   Input -> [SharedEncoder] -> TaskHead(return)
//!                            -> TaskHead(volatility)
//!                            -> TaskHead(direction)
//!                            -> TaskHead(volume)

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Predictions from the multi-task model.
#[derive(Debug, Clone)]
pub struct TaskPredictions {
    pub return_pred: Vec<f64>,
    pub volatility_pred: Vec<f64>,
    pub direction_pred: Vec<f64>,
    pub volume_pred: Vec<f64>,
}

/// A single linear layer: y = Wx + b
#[derive(Debug, Clone)]
struct LinearLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_size: usize,
    output_size: usize,
}

impl LinearLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let biases = vec![0.0; output_size];

        Self { weights, biases, input_size, output_size }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        (0..self.output_size)
            .map(|o| {
                let dot: f64 = self.weights[o].iter().zip(input).map(|(w, x)| w * x).sum();
                dot + self.biases[o]
            })
            .collect()
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for row in &self.weights {
            params.extend(row);
        }
        params.extend(&self.biases);
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let mut idx = 0;
        for row in &mut self.weights {
            for w in row.iter_mut() {
                *w = params[idx];
                idx += 1;
            }
        }
        for b in &mut self.biases {
            *b = params[idx];
            idx += 1;
        }
        idx
    }

    fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

/// Task-specific output head.
#[derive(Debug, Clone)]
struct TaskHead {
    hidden: LinearLayer,
    output: LinearLayer,
    is_classification: bool,
}

impl TaskHead {
    fn new(input_size: usize, hidden_size: usize, is_classification: bool) -> Self {
        Self {
            hidden: LinearLayer::new(input_size, hidden_size),
            output: LinearLayer::new(hidden_size, 1),
            is_classification,
        }
    }

    fn forward(&self, input: &[f64]) -> f64 {
        let h: Vec<f64> = self.hidden.forward(input).into_iter().map(|x| x.max(0.0)).collect();
        let out = self.output.forward(&h)[0];
        if self.is_classification {
            1.0 / (1.0 + (-out).exp()) // sigmoid
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = self.hidden.parameters();
        params.extend(self.output.parameters());
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let used1 = self.hidden.set_parameters(params);
        let used2 = self.output.set_parameters(&params[used1..]);
        used1 + used2
    }

    fn num_parameters(&self) -> usize {
        self.hidden.num_parameters() + self.output.num_parameters()
    }
}

/// Multi-Task Learning model with shared encoder and task-specific heads.
#[derive(Debug, Clone)]
pub struct MultiTaskModel {
    encoder_layers: Vec<LinearLayer>,
    return_head: TaskHead,
    volatility_head: TaskHead,
    direction_head: TaskHead,
    volume_head: TaskHead,
    input_size: usize,
}

impl MultiTaskModel {
    /// Create a new multi-task model.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_sizes` - Sizes of shared encoder layers
    /// * `head_hidden` - Hidden size for task-specific heads
    pub fn new(input_size: usize, hidden_sizes: &[usize], head_hidden: usize) -> Self {
        let mut encoder_layers = Vec::new();
        let mut prev = input_size;
        for &h in hidden_sizes {
            encoder_layers.push(LinearLayer::new(prev, h));
            prev = h;
        }

        let enc_out = *hidden_sizes.last().unwrap_or(&input_size);

        Self {
            encoder_layers,
            return_head: TaskHead::new(enc_out, head_hidden, false),
            volatility_head: TaskHead::new(enc_out, head_hidden, false),
            direction_head: TaskHead::new(enc_out, head_hidden, true),
            volume_head: TaskHead::new(enc_out, head_hidden, false),
            input_size,
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, input: &[f64]) -> TaskPredictions {
        let encoded = self.encode(input);
        TaskPredictions {
            return_pred: vec![self.return_head.forward(&encoded)],
            volatility_pred: vec![self.volatility_head.forward(&encoded)],
            direction_pred: vec![self.direction_head.forward(&encoded)],
            volume_pred: vec![self.volume_head.forward(&encoded)],
        }
    }

    /// Forward pass for a batch of samples.
    pub fn forward_batch(&self, inputs: &[Vec<f64>]) -> TaskPredictions {
        let mut ret = Vec::with_capacity(inputs.len());
        let mut vol = Vec::with_capacity(inputs.len());
        let mut dir = Vec::with_capacity(inputs.len());
        let mut volume = Vec::with_capacity(inputs.len());

        for input in inputs {
            let encoded = self.encode(input);
            ret.push(self.return_head.forward(&encoded));
            vol.push(self.volatility_head.forward(&encoded));
            dir.push(self.direction_head.forward(&encoded));
            volume.push(self.volume_head.forward(&encoded));
        }

        TaskPredictions {
            return_pred: ret,
            volatility_pred: vol,
            direction_pred: dir,
            volume_pred: volume,
        }
    }

    /// Encode input through shared layers.
    fn encode(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &self.encoder_layers {
            x = layer.forward(&x);
            // ReLU activation
            x.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        x
    }

    /// Get all model parameters as a flat vector.
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &self.encoder_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.return_head.parameters());
        params.extend(self.volatility_head.parameters());
        params.extend(self.direction_head.parameters());
        params.extend(self.volume_head.parameters());
        params
    }

    /// Set model parameters from a flat vector.
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut offset = 0;
        for layer in &mut self.encoder_layers {
            offset += layer.set_parameters(&params[offset..]);
        }
        offset += self.return_head.set_parameters(&params[offset..]);
        offset += self.volatility_head.set_parameters(&params[offset..]);
        offset += self.direction_head.set_parameters(&params[offset..]);
        self.volume_head.set_parameters(&params[offset..]);
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let encoder: usize = self.encoder_layers.iter().map(|l| l.num_parameters()).sum();
        encoder
            + self.return_head.num_parameters()
            + self.volatility_head.num_parameters()
            + self.direction_head.num_parameters()
            + self.volume_head.num_parameters()
    }

    /// Get the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = MultiTaskModel::new(10, &[64, 32], 16);
        assert!(model.num_parameters() > 0);
        assert_eq!(model.input_size(), 10);
    }

    #[test]
    fn test_forward_pass() {
        let model = MultiTaskModel::new(5, &[16, 8], 4);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        let preds = model.forward(&input);

        assert_eq!(preds.return_pred.len(), 1);
        assert_eq!(preds.volatility_pred.len(), 1);
        assert_eq!(preds.direction_pred.len(), 1);
        assert_eq!(preds.volume_pred.len(), 1);

        // Direction should be in [0, 1] (sigmoid output)
        assert!(preds.direction_pred[0] >= 0.0 && preds.direction_pred[0] <= 1.0);
    }

    #[test]
    fn test_batch_forward() {
        let model = MultiTaskModel::new(3, &[8], 4);
        let inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![-0.1, 0.0, 0.5],
            vec![0.5, -0.3, 0.1],
        ];
        let preds = model.forward_batch(&inputs);

        assert_eq!(preds.return_pred.len(), 3);
        assert_eq!(preds.volatility_pred.len(), 3);
    }

    #[test]
    fn test_parameter_roundtrip() {
        let model = MultiTaskModel::new(5, &[16], 8);
        let params = model.parameters();
        let mut model2 = MultiTaskModel::new(5, &[16], 8);
        model2.set_parameters(&params);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p1 = model.forward(&input);
        let p2 = model2.forward(&input);

        assert!((p1.return_pred[0] - p2.return_pred[0]).abs() < 1e-10);
    }
}
