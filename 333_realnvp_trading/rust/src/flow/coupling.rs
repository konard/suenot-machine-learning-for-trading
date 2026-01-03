//! # Affine Coupling Layer
//!
//! Implementation of the affine coupling layer used in RealNVP.
//!
//! The coupling layer splits the input into two halves and transforms
//! one half conditioned on the other, enabling tractable Jacobian computation.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use crate::config::DEFAULT_SCALE_CLAMP;

/// Mask type for coupling layer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskType {
    /// Even indices are masked (passed through)
    Even,
    /// Odd indices are masked (passed through)
    Odd,
}

/// Simple MLP (Multi-Layer Perceptron) for scale and translation networks
#[derive(Debug, Clone)]
pub struct MLP {
    /// Weight matrices for each layer
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
}

impl MLP {
    /// Create a new MLP with given architecture
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Input layer
        let w_in = Array2::from_shape_fn((hidden_dim, input_dim), |_| normal.sample(&mut rng));
        let b_in = Array1::zeros(hidden_dim);
        weights.push(w_in);
        biases.push(b_in);

        // Hidden layers
        for _ in 0..num_hidden.saturating_sub(1) {
            let w = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| normal.sample(&mut rng));
            let b = Array1::zeros(hidden_dim);
            weights.push(w);
            biases.push(b);
        }

        // Output layer
        let w_out = Array2::from_shape_fn((output_dim, hidden_dim), |_| normal.sample(&mut rng));
        let b_out = Array1::zeros(output_dim);
        weights.push(w_out);
        biases.push(b_out);

        Self { weights, biases }
    }

    /// Forward pass through MLP with ReLU activations
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut h = x.clone();

        for i in 0..self.weights.len() - 1 {
            h = self.weights[i].dot(&h) + &self.biases[i];
            // ReLU activation
            h.mapv_inplace(|v| v.max(0.0));
        }

        // Final layer (no activation)
        let last_idx = self.weights.len() - 1;
        self.weights[last_idx].dot(&h) + &self.biases[last_idx]
    }

    /// Get all parameters as a flat vector (for optimization)
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            params.extend(w.iter().cloned());
            params.extend(b.iter().cloned());
        }
        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            let w_size = w.len();
            let b_size = b.len();

            for (i, val) in w.iter_mut().enumerate() {
                *val = params[idx + i];
            }
            idx += w_size;

            for (i, val) in b.iter_mut().enumerate() {
                *val = params[idx + i];
            }
            idx += b_size;
        }
    }

    /// Update parameters using gradients and learning rate
    pub fn update(&mut self, gradients: &[f64], lr: f64) {
        let mut idx = 0;
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            for val in w.iter_mut() {
                *val -= lr * gradients[idx];
                idx += 1;
            }
            for val in b.iter_mut() {
                *val -= lr * gradients[idx];
                idx += 1;
            }
        }
    }
}

/// Affine Coupling Layer for RealNVP
///
/// Transforms input x by:
/// - Splitting into (x_masked, x_unmasked)
/// - Computing scale s and translation t from x_masked
/// - Transforming: y_unmasked = x_unmasked * exp(s) + t, y_masked = x_masked
#[derive(Debug, Clone)]
pub struct CouplingLayer {
    /// Input dimension
    dim: usize,
    /// Mask type (even or odd)
    mask_type: MaskType,
    /// Mask indices (true = passed through)
    mask: Vec<bool>,
    /// Number of masked elements
    n_masked: usize,
    /// Scale network
    scale_net: MLP,
    /// Translation network
    translation_net: MLP,
    /// Scale clamping value
    scale_clamp: f64,
}

impl CouplingLayer {
    /// Create a new coupling layer
    pub fn new(dim: usize, hidden_dim: usize, num_hidden: usize, mask_type: MaskType) -> Self {
        // Create mask
        let mask: Vec<bool> = (0..dim)
            .map(|i| match mask_type {
                MaskType::Even => i % 2 == 0,
                MaskType::Odd => i % 2 == 1,
            })
            .collect();

        let n_masked = mask.iter().filter(|&&m| m).count();
        let n_unmasked = dim - n_masked;

        // Create scale and translation networks
        let scale_net = MLP::new(n_masked, hidden_dim, n_unmasked, num_hidden);
        let translation_net = MLP::new(n_masked, hidden_dim, n_unmasked, num_hidden);

        Self {
            dim,
            mask_type,
            mask,
            n_masked,
            scale_net,
            translation_net,
            scale_clamp: DEFAULT_SCALE_CLAMP,
        }
    }

    /// Extract masked elements from input
    fn extract_masked(&self, x: &Array1<f64>) -> Array1<f64> {
        let masked: Vec<f64> = x.iter()
            .zip(self.mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();
        Array1::from_vec(masked)
    }

    /// Extract unmasked elements from input
    fn extract_unmasked(&self, x: &Array1<f64>) -> Array1<f64> {
        let unmasked: Vec<f64> = x.iter()
            .zip(self.mask.iter())
            .filter(|(_, &m)| !m)
            .map(|(&v, _)| v)
            .collect();
        Array1::from_vec(unmasked)
    }

    /// Combine masked and unmasked elements back
    fn combine(&self, masked: &Array1<f64>, unmasked: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(self.dim);
        let mut m_idx = 0;
        let mut u_idx = 0;

        for (i, &is_masked) in self.mask.iter().enumerate() {
            if is_masked {
                result[i] = masked[m_idx];
                m_idx += 1;
            } else {
                result[i] = unmasked[u_idx];
                u_idx += 1;
            }
        }

        result
    }

    /// Forward pass: x -> y
    ///
    /// Returns (y, log_det) where log_det is the log determinant of the Jacobian
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let x_masked = self.extract_masked(x);
        let x_unmasked = self.extract_unmasked(x);

        // Compute scale and translation
        let s = self.scale_net.forward(&x_masked);
        let t = self.translation_net.forward(&x_masked);

        // Clamp scale for numerical stability
        let s_clamped: Array1<f64> = s.mapv(|v| v.clamp(-self.scale_clamp, self.scale_clamp));

        // Transform unmasked part: y = x * exp(s) + t
        let y_unmasked: Array1<f64> = &x_unmasked * &s_clamped.mapv(f64::exp) + &t;

        // Combine
        let y = self.combine(&x_masked, &y_unmasked);

        // Log determinant is sum of scales
        let log_det = s_clamped.sum();

        (y, log_det)
    }

    /// Inverse pass: y -> x
    pub fn inverse(&self, y: &Array1<f64>) -> Array1<f64> {
        let y_masked = self.extract_masked(y);
        let y_unmasked = self.extract_unmasked(y);

        // Compute scale and translation (using masked part which is same as input)
        let s = self.scale_net.forward(&y_masked);
        let t = self.translation_net.forward(&y_masked);

        // Clamp scale
        let s_clamped: Array1<f64> = s.mapv(|v| v.clamp(-self.scale_clamp, self.scale_clamp));

        // Inverse transform: x = (y - t) * exp(-s)
        let x_unmasked: Array1<f64> = (&y_unmasked - &t) * &s_clamped.mapv(|v| (-v).exp());

        // Combine
        self.combine(&y_masked, &x_unmasked)
    }

    /// Forward pass for batch of inputs
    pub fn forward_batch(&self, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        let batch_size = x.nrows();
        let mut y = Array2::zeros((batch_size, self.dim));
        let mut log_dets = Array1::zeros(batch_size);

        for i in 0..batch_size {
            let x_i = x.row(i).to_owned();
            let (y_i, log_det_i) = self.forward(&x_i);
            y.row_mut(i).assign(&y_i);
            log_dets[i] = log_det_i;
        }

        (y, log_dets)
    }

    /// Inverse pass for batch of inputs
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

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get mask type
    pub fn mask_type(&self) -> MaskType {
        self.mask_type
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = self.scale_net.parameters();
        params.extend(self.translation_net.parameters());
        params
    }

    /// Set parameters
    pub fn set_parameters(&mut self, params: &[f64]) {
        let scale_params = self.scale_net.parameters().len();
        self.scale_net.set_parameters(&params[..scale_params]);
        self.translation_net.set_parameters(&params[scale_params..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_coupling_layer_forward_inverse() {
        let layer = CouplingLayer::new(4, 16, 2, MaskType::Even);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let (y, _log_det) = layer.forward(&x);
        let x_reconstructed = layer.inverse(&y);

        for (a, b) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_coupling_layer_log_det() {
        let layer = CouplingLayer::new(4, 16, 2, MaskType::Odd);

        let x = Array1::from_vec(vec![0.5, -0.5, 1.0, -1.0]);
        let (_, log_det) = layer.forward(&x);

        // Log det should be a finite number
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_extract_mask() {
        let layer = CouplingLayer::new(4, 16, 2, MaskType::Even);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let masked = layer.extract_masked(&x);
        let unmasked = layer.extract_unmasked(&x);

        assert_eq!(masked.len(), 2);
        assert_eq!(unmasked.len(), 2);
        assert_eq!(masked[0], 1.0);
        assert_eq!(masked[1], 3.0);
        assert_eq!(unmasked[0], 2.0);
        assert_eq!(unmasked[1], 4.0);
    }
}
