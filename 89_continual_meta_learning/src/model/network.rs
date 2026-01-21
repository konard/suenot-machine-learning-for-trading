//! Neural network implementation for trading predictions.
//!
//! This module provides a simple feedforward neural network
//! suitable for meta-learning with the CML algorithm.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// A simple feedforward neural network for trading signal prediction.
///
/// Architecture:
/// - Input layer -> Hidden layer 1 (with ReLU) -> Hidden layer 2 (with ReLU) -> Output
#[derive(Debug, Clone)]
pub struct TradingModel {
    /// Weights for input to hidden layer 1
    w1: Vec<Vec<f64>>,
    /// Biases for hidden layer 1
    b1: Vec<f64>,
    /// Weights for hidden layer 1 to hidden layer 2
    w2: Vec<Vec<f64>>,
    /// Biases for hidden layer 2
    b2: Vec<f64>,
    /// Weights for hidden layer 2 to output
    w3: Vec<Vec<f64>>,
    /// Biases for output layer
    b3: Vec<f64>,
    /// Input size
    input_size: usize,
    /// Hidden layer size
    hidden_size: usize,
    /// Output size
    output_size: usize,
}

impl TradingModel {
    /// Create a new trading model with random initialization.
    ///
    /// Uses Xavier/Glorot initialization for weights.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization scale factors
        let scale1 = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let scale2 = (2.0 / (hidden_size + hidden_size) as f64).sqrt();
        let scale3 = (2.0 / (hidden_size + output_size) as f64).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initialize weights
        let w1 = (0..hidden_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| normal.sample(&mut rng) * scale1)
                    .collect()
            })
            .collect();

        let w2 = (0..hidden_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| normal.sample(&mut rng) * scale2)
                    .collect()
            })
            .collect();

        let w3 = (0..output_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| normal.sample(&mut rng) * scale3)
                    .collect()
            })
            .collect();

        // Initialize biases to zero
        let b1 = vec![0.0; hidden_size];
        let b2 = vec![0.0; hidden_size];
        let b3 = vec![0.0; output_size];

        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            input_size,
            hidden_size,
            output_size,
        }
    }

    /// Forward pass through the network.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);

        // Layer 1: Linear + ReLU
        let h1: Vec<f64> = self
            .w1
            .iter()
            .zip(&self.b1)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
                (sum + bias).max(0.0) // ReLU
            })
            .collect();

        // Layer 2: Linear + ReLU
        let h2: Vec<f64> = self
            .w2
            .iter()
            .zip(&self.b2)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(&h1).map(|(w, x)| w * x).sum();
                (sum + bias).max(0.0) // ReLU
            })
            .collect();

        // Output layer: Linear (no activation)
        self.w3
            .iter()
            .zip(&self.b3)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(&h2).map(|(w, x)| w * x).sum();
                sum + bias
            })
            .collect()
    }

    /// Compute gradients using backpropagation.
    ///
    /// Returns gradients for all parameters.
    pub fn backward(
        &self,
        input: &[f64],
        target: &[f64],
    ) -> (
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<f64>,
    ) {
        // Forward pass with cached activations
        let z1: Vec<f64> = self
            .w1
            .iter()
            .zip(&self.b1)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
                sum + bias
            })
            .collect();
        let h1: Vec<f64> = z1.iter().map(|&x| x.max(0.0)).collect();

        let z2: Vec<f64> = self
            .w2
            .iter()
            .zip(&self.b2)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(&h1).map(|(w, x)| w * x).sum();
                sum + bias
            })
            .collect();
        let h2: Vec<f64> = z2.iter().map(|&x| x.max(0.0)).collect();

        let output: Vec<f64> = self
            .w3
            .iter()
            .zip(&self.b3)
            .map(|(weights, bias)| {
                let sum: f64 = weights.iter().zip(&h2).map(|(w, x)| w * x).sum();
                sum + bias
            })
            .collect();

        // Backward pass
        // Output layer gradient (MSE loss derivative)
        let d_output: Vec<f64> = output
            .iter()
            .zip(target)
            .map(|(o, t)| 2.0 * (o - t) / target.len() as f64)
            .collect();

        // Gradients for W3 and b3
        let mut dw3 = vec![vec![0.0; self.hidden_size]; self.output_size];
        let mut db3 = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            db3[i] = d_output[i];
            for j in 0..self.hidden_size {
                dw3[i][j] = d_output[i] * h2[j];
            }
        }

        // Hidden layer 2 gradient
        let mut d_h2 = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                d_h2[i] += self.w3[j][i] * d_output[j];
            }
            // ReLU derivative
            if z2[i] <= 0.0 {
                d_h2[i] = 0.0;
            }
        }

        // Gradients for W2 and b2
        let mut dw2 = vec![vec![0.0; self.hidden_size]; self.hidden_size];
        let mut db2 = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            db2[i] = d_h2[i];
            for j in 0..self.hidden_size {
                dw2[i][j] = d_h2[i] * h1[j];
            }
        }

        // Hidden layer 1 gradient
        let mut d_h1 = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            for j in 0..self.hidden_size {
                d_h1[i] += self.w2[j][i] * d_h2[j];
            }
            // ReLU derivative
            if z1[i] <= 0.0 {
                d_h1[i] = 0.0;
            }
        }

        // Gradients for W1 and b1
        let mut dw1 = vec![vec![0.0; self.input_size]; self.hidden_size];
        let mut db1 = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            db1[i] = d_h1[i];
            for j in 0..self.input_size {
                dw1[i][j] = d_h1[i] * input[j];
            }
        }

        (dw1, db1, dw2, db2, dw3, db3)
    }

    /// Update parameters using SGD.
    pub fn update(
        &mut self,
        dw1: &[Vec<f64>],
        db1: &[f64],
        dw2: &[Vec<f64>],
        db2: &[f64],
        dw3: &[Vec<f64>],
        db3: &[f64],
        learning_rate: f64,
    ) {
        for i in 0..self.hidden_size {
            self.b1[i] -= learning_rate * db1[i];
            for j in 0..self.input_size {
                self.w1[i][j] -= learning_rate * dw1[i][j];
            }
        }

        for i in 0..self.hidden_size {
            self.b2[i] -= learning_rate * db2[i];
            for j in 0..self.hidden_size {
                self.w2[i][j] -= learning_rate * dw2[i][j];
            }
        }

        for i in 0..self.output_size {
            self.b3[i] -= learning_rate * db3[i];
            for j in 0..self.hidden_size {
                self.w3[i][j] -= learning_rate * dw3[i][j];
            }
        }
    }

    /// Compute MSE loss.
    pub fn compute_loss(&self, input: &[f64], target: &[f64]) -> f64 {
        let output = self.forward(input);
        output
            .iter()
            .zip(target)
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / target.len() as f64
    }

    /// Get all parameters as a flat vector.
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();

        for row in &self.w1 {
            params.extend(row);
        }
        params.extend(&self.b1);

        for row in &self.w2 {
            params.extend(row);
        }
        params.extend(&self.b2);

        for row in &self.w3 {
            params.extend(row);
        }
        params.extend(&self.b3);

        params
    }

    /// Set all parameters from a flat vector.
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;

        for row in &mut self.w1 {
            for val in row {
                *val = params[idx];
                idx += 1;
            }
        }
        for val in &mut self.b1 {
            *val = params[idx];
            idx += 1;
        }

        for row in &mut self.w2 {
            for val in row {
                *val = params[idx];
                idx += 1;
            }
        }
        for val in &mut self.b2 {
            *val = params[idx];
            idx += 1;
        }

        for row in &mut self.w3 {
            for val in row {
                *val = params[idx];
                idx += 1;
            }
        }
        for val in &mut self.b3 {
            *val = params[idx];
            idx += 1;
        }
    }

    /// Get the total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
    }

    /// Alias for get_parameters().
    pub fn get_params(&self) -> Vec<f64> {
        self.get_parameters()
    }

    /// Alias for set_parameters().
    pub fn set_params(&mut self, params: &[f64]) {
        self.set_parameters(params);
    }

    /// Simplified update using a flat gradient vector.
    ///
    /// This assumes the gradients are in the same order as get_params().
    pub fn update_from_flat(&mut self, gradients: &[f64], learning_rate: f64) {
        let mut params = self.get_params();
        let n = params.len().min(gradients.len());
        for i in 0..n {
            params[i] -= learning_rate * gradients[i];
        }
        self.set_params(&params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let model = TradingModel::new(4, 8, 1);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_backward() {
        let model = TradingModel::new(4, 8, 1);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let target = vec![0.5];
        let (dw1, db1, dw2, db2, dw3, db3) = model.backward(&input, &target);

        assert_eq!(dw1.len(), 8);
        assert_eq!(db1.len(), 8);
        assert_eq!(dw2.len(), 8);
        assert_eq!(db2.len(), 8);
        assert_eq!(dw3.len(), 1);
        assert_eq!(db3.len(), 1);
    }

    #[test]
    fn test_parameter_count() {
        let model = TradingModel::new(8, 64, 1);
        let params = model.get_parameters();
        assert_eq!(params.len(), model.num_parameters());
    }
}
