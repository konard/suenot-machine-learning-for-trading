//! Neural network layers for MC Dropout

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Linear layer with optional dropout
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix [out_features, in_features]
    pub weights: Array2<f64>,
    /// Bias vector [out_features]
    pub bias: Array1<f64>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((out_features, in_features), |_| {
            normal.sample(&mut rng)
        });

        let bias = Array1::zeros(out_features);

        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = Wx + b
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(x) + &self.bias
    }

    /// Forward pass for batch: Y = XW^T + b
    pub fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.dot(&self.weights.t());
        for mut row in result.rows_mut() {
            row += &self.bias;
        }
        result
    }
}

/// Dropout layer
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// Dropout probability
    pub p: f64,
    /// Whether dropout is active
    pub training: bool,
}

impl DropoutLayer {
    /// Create new dropout layer
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 0.99),
            training: true,
        }
    }

    /// Apply dropout to input
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);
        let scale = 1.0 / (1.0 - self.p);

        Array1::from_iter(x.iter().map(|&val| {
            if uniform.sample(&mut rng) > self.p {
                val * scale
            } else {
                0.0
            }
        }))
    }

    /// Apply dropout to batch
    pub fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);
        let scale = 1.0 / (1.0 - self.p);

        Array2::from_shape_fn(x.dim(), |idx| {
            let val = x[idx];
            if uniform.sample(&mut rng) > self.p {
                val * scale
            } else {
                0.0
            }
        })
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// ReLU activation function
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// ReLU for batch
pub fn relu_batch(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Softmax activation function
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = x.mapv(|v| (v - max).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(10, 5);
        let input = Array1::ones(10);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_dropout_training() {
        let layer = DropoutLayer::new(0.5);
        let input = Array1::ones(1000);
        let output = layer.forward(&input);

        // With 50% dropout, roughly half should be 0
        let zeros = output.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 300 && zeros < 700); // Roughly 50%
    }

    #[test]
    fn test_dropout_not_training() {
        let mut layer = DropoutLayer::new(0.5);
        layer.set_training(false);

        let input = Array1::ones(100);
        let output = layer.forward(&input);

        // In non-training mode, no dropout
        assert_relative_eq!(output.sum(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_relu() {
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let output = relu(&input);
        assert_eq!(output.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
