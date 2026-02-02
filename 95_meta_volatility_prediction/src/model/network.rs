//! Neural network for volatility prediction.
//!
//! Implements a feedforward network with Softplus output activation
//! to ensure predicted volatility is always positive.

use rand::Rng;
use rand_distr::Normal;

/// A single linear layer: y = Wx + b
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl LinearLayer {
    /// Create a new layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();

        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.sample(dist)).collect())
            .collect();
        let biases = vec![0.0; output_dim];

        Self {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: y = Wx + b
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = self.biases.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                output[i] += w * input[j];
            }
        }
        output
    }

    /// Forward pass with explicit weights (for MAML inner loop).
    pub fn forward_with_weights(
        &self,
        input: &[f64],
        weights: &[Vec<f64>],
        biases: &[f64],
    ) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = biases.to_vec();
        for (i, row) in weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                output[i] += w * input[j];
            }
        }
        output
    }

    /// Get total number of parameters.
    pub fn num_params(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    /// Flatten weights and biases into a single vector.
    pub fn flatten_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_params());
        for row in &self.weights {
            params.extend(row);
        }
        params.extend(&self.biases);
        params
    }

    /// Restore weights and biases from a flat vector.
    pub fn unflatten_params(&mut self, params: &[f64]) {
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
    }
}

/// Meta-Volatility neural network.
///
/// Architecture: Input -> Dense -> ReLU -> Dense -> ReLU -> Dense -> ReLU -> Dense -> Softplus
#[derive(Debug, Clone)]
pub struct MetaVolatilityNet {
    pub layers: Vec<LinearLayer>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

impl MetaVolatilityNet {
    /// Create a new network with given dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let layers = vec![
            LinearLayer::new(input_dim, hidden_dim),
            LinearLayer::new(hidden_dim, hidden_dim),
            LinearLayer::new(hidden_dim, hidden_dim / 2),
            LinearLayer::new(hidden_dim / 2, 1),
        ];

        Self {
            layers,
            input_dim,
            hidden_dim,
        }
    }

    /// Forward pass through the network.
    pub fn forward(&self, input: &[f64]) -> f64 {
        let mut h = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.layers.len() - 1 {
                // ReLU activation
                for v in h.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
        // Softplus activation on final output
        softplus(h[0])
    }

    /// Forward pass with explicit parameters (for MAML).
    pub fn forward_with_params(&self, input: &[f64], params: &[f64]) -> f64 {
        let mut h = input.to_vec();
        let mut offset = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let w_size = layer.input_dim * layer.output_dim;
            let b_size = layer.output_dim;

            // Extract weights and biases from flat params
            let flat_w = &params[offset..offset + w_size];
            let biases = &params[offset + w_size..offset + w_size + b_size];

            let weights: Vec<Vec<f64>> = (0..layer.output_dim)
                .map(|r| {
                    flat_w[r * layer.input_dim..(r + 1) * layer.input_dim].to_vec()
                })
                .collect();

            h = layer.forward_with_weights(&h, &weights, biases);
            offset += w_size + b_size;

            if i < self.layers.len() - 1 {
                for v in h.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }

        softplus(h[0])
    }

    /// Get total number of parameters.
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }

    /// Flatten all parameters into a single vector.
    pub fn flatten_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_params());
        for layer in &self.layers {
            params.extend(layer.flatten_params());
        }
        params
    }

    /// Restore parameters from a flat vector.
    pub fn unflatten_params(&mut self, params: &[f64]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let n = layer.num_params();
            layer.unflatten_params(&params[offset..offset + n]);
            offset += n;
        }
    }
}

/// Softplus activation: ln(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // Avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softplus_positive() {
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(-5.0) > 0.0);
        assert!(softplus(5.0) > 0.0);
    }

    #[test]
    fn test_network_output_positive() {
        let net = MetaVolatilityNet::new(10, 64);
        let input = vec![0.01, 0.02, 0.001, 0.05, 0.04, 0.03, 1.2, 0.0, 50.0, 0.1];
        let output = net.forward(&input);
        assert!(output > 0.0, "Volatility prediction must be positive");
    }

    #[test]
    fn test_param_flatten_roundtrip() {
        let net = MetaVolatilityNet::new(10, 64);
        let params = net.flatten_params();
        let mut net2 = MetaVolatilityNet::new(10, 64);
        net2.unflatten_params(&params);

        let input = vec![0.01; 10];
        let out1 = net.forward(&input);
        let out2 = net2.forward(&input);
        assert!((out1 - out2).abs() < 1e-10);
    }

    #[test]
    fn test_forward_with_params() {
        let net = MetaVolatilityNet::new(10, 64);
        let params = net.flatten_params();
        let input = vec![0.01; 10];

        let out1 = net.forward(&input);
        let out2 = net.forward_with_params(&input, &params);
        assert!((out1 - out2).abs() < 1e-10);
    }
}
