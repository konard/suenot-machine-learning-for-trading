//! # Activation Normalization Layer
//!
//! ActNorm performs data-dependent initialization for stable training.
//! It learns a scale and bias that normalizes activations.

use ndarray::{Array1, Array2, Axis};

/// Activation Normalization Layer
///
/// Applies: y = (x + bias) * exp(log_scale)
#[derive(Debug, Clone)]
pub struct ActNorm {
    /// Dimension of input
    dim: usize,
    /// Log scale parameters
    log_scale: Array1<f64>,
    /// Bias parameters
    bias: Array1<f64>,
    /// Whether the layer has been initialized
    initialized: bool,
}

impl ActNorm {
    /// Create a new ActNorm layer
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            log_scale: Array1::zeros(dim),
            bias: Array1::zeros(dim),
            initialized: false,
        }
    }

    /// Initialize with first batch statistics
    pub fn initialize(&mut self, x: &Array2<f64>) {
        if self.initialized {
            return;
        }

        let mean = x.mean_axis(Axis(0)).unwrap();
        let std = x.std_axis(Axis(0), 0.0);

        // bias = -mean
        self.bias = -mean;

        // log_scale = -log(std + eps)
        self.log_scale = std.mapv(|s| -(s + 1e-6).ln());

        self.initialized = true;
    }

    /// Forward pass: x -> y
    ///
    /// Returns (y, log_det)
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let y = (x + &self.bias) * &self.log_scale.mapv(f64::exp);
        let log_det = self.log_scale.sum();
        (y, log_det)
    }

    /// Inverse pass: y -> x
    pub fn inverse(&self, y: &Array1<f64>) -> Array1<f64> {
        y * &self.log_scale.mapv(|s| (-s).exp()) - &self.bias
    }

    /// Forward pass for batch
    pub fn forward_batch(&mut self, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        // Initialize on first batch if needed
        if !self.initialized {
            self.initialize(x);
        }

        let batch_size = x.nrows();
        let mut y = Array2::zeros((batch_size, self.dim));
        let log_det = self.log_scale.sum();

        for i in 0..batch_size {
            let x_i = x.row(i).to_owned();
            let (y_i, _) = self.forward(&x_i);
            y.row_mut(i).assign(&y_i);
        }

        let log_dets = Array1::from_elem(batch_size, log_det);
        (y, log_dets)
    }

    /// Inverse pass for batch
    pub fn inverse_batch(&self, y: &Array2<f64>) -> Array2<f64> {
        let batch_size = y.nrows();
        let mut x = Array2::zeros((batch_size, self.dim));

        for i in 0..batch_size {
            let y_i = y.row(i).to_owned();
            let x_i = self.inverse(&y_i);
            x.row_mut(i).assign(&x_i);
        }

        x
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = self.log_scale.to_vec();
        params.extend(self.bias.iter());
        params
    }

    /// Set parameters
    pub fn set_parameters(&mut self, params: &[f64]) {
        for (i, val) in self.log_scale.iter_mut().enumerate() {
            *val = params[i];
        }
        for (i, val) in self.bias.iter_mut().enumerate() {
            *val = params[self.dim + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_actnorm_forward_inverse() {
        let mut actnorm = ActNorm::new(4);

        // Initialize with some data
        let init_data = Array2::from_shape_vec(
            (10, 4),
            (0..40).map(|x| x as f64 / 10.0).collect(),
        ).unwrap();
        actnorm.initialize(&init_data);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let (y, _) = actnorm.forward(&x);
        let x_reconstructed = actnorm.inverse(&y);

        for (a, b) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_actnorm_initialization() {
        let mut actnorm = ActNorm::new(2);
        assert!(!actnorm.is_initialized());

        let data = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|x| x as f64).collect(),
        ).unwrap();
        actnorm.initialize(&data);

        assert!(actnorm.is_initialized());
    }
}
